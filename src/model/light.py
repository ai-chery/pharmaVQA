import os

import torch
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import numpy as np
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence

from src.data.featurizer import VIRTUAL_ATOM_FEATURE_PLACEHOLDER, VIRTUAL_BOND_FEATURE_PLACEHOLDER
from src.model.ban import BANLayer


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation, d_hidden_feats=d_out_feats * 4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x


class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats


class TripletTransformer(nn.Module):
    def __init__(self,
                 d_feats,
                 d_hpath_ratio,
                 path_length,
                 n_heads,
                 n_ffn_dense_layers,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU()):
        super(TripletTransformer, self).__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, activation)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, triplet_h, dist_attn, path_attn):
        g = g.local_var()
        new_triplet_h = self.attention_norm(triplet_h)
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + path_attn.reshape(
            len(g.edata['node_attn']), -1, 1)
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))

        g.ndata['hv'] = v.view(-1, self.d_feats)
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats // self.n_heads)) * g.edata['sa']).view(-1,
                                                                                                                    self.d_feats)

        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        return self.node_out_layer(triplet_h, g.ndata['agg_h'])

    def _device(self):
        return next(self.parameters()).device


class LiGhT(nn.Module):
    def __init__(self,
                 d_g_feats,
                 d_hpath_ratio,
                 path_length,
                 n_mol_layers=2,
                 n_heads=4,
                 n_ffn_dense_layers=4,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU()):
        super(LiGhT, self).__init__()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats // d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Distance Attention
        self.path_len_emb = nn.Embedding(path_length + 1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(self.d_g_feats, self.d_g_feats),
            activation,
            nn.Linear(self.d_g_feats, n_heads)
        )
        # Path Attention  
        self.trip_fortrans = nn.ModuleList([
            MLP(d_g_feats, self.d_trip_path, 2, activation) for _ in range(self.path_length)
        ])
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            activation,
            nn.Linear(self.d_trip_path, n_heads)
        )
        # Molecule Transformer Layers
        self.mol_T_layers = nn.ModuleList([
            TripletTransformer(d_g_feats, d_hpath_ratio, path_length, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
                               activation) for _ in range(n_mol_layers)
        ])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def _featurize_path(self, g, path_indices):
        mask = (path_indices[:, :] >= 0).to(torch.int32)
        path_feats = torch.sum(mask, dim=-1)
        path_feats = self.path_len_emb(path_feats)
        path_feats[g.edata['vp'] == 1] = self.virtual_path_emb.weight  # virtual path
        path_feats[g.edata['sl'] == 1] = self.self_loop_emb.weight  # self loop
        return path_feats

    def _init_path(self, g, triplet_h, path_indices):
        g = g.local_var()
        path_indices[path_indices < -99] = -1
        path_h = []
        for i in range(self.path_length):
            path_h.append(torch.cat(
                [self.trip_fortrans[i](triplet_h), torch.zeros(size=(1, self.d_trip_path)).to(self._device())], dim=0)[
                              path_indices[:, i]])
        path_h = torch.stack(path_h, dim=-1)
        mask = (path_indices >= 0).to(torch.int32)
        path_size = torch.sum(mask, dim=-1, keepdim=True)
        path_h = torch.sum(path_h, dim=-1) / path_size
        return path_h

    def forward(self, g, triplet_h):
        path_indices = g.edata['path']
        dist_h = self._featurize_path(g, path_indices)
        path_h = self._init_path(g, triplet_h, path_indices)
        dist_attn, path_attn = self.dist_attn_layer(dist_h), self.path_attn_layer(path_h)
        for i in range(self.n_mol_layers):
            triplet_h = self.mol_T_layers[i](g, triplet_h, dist_attn, path_attn)
        return triplet_h

    def _device(self):
        return next(self.parameters()).device


class AtomEmbedding(nn.Module):
    def __init__(
            self,
            d_atom_feats,
            d_g_feats,
            input_drop):
        super(AtomEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, pair_node_feats, indicators):
        pair_node_h = self.in_proj(pair_node_feats)
        pair_node_h[indicators == VIRTUAL_ATOM_FEATURE_PLACEHOLDER, 1, :] = self.virtual_atom_emb.weight  # .half()
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)


class BondEmbedding(nn.Module):
    def __init__(
            self,
            d_bond_feats,
            d_g_feats,
            input_drop):
        super(BondEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, edge_feats, indicators):
        edge_h = self.in_proj(edge_feats)
        edge_h[indicators == VIRTUAL_BOND_FEATURE_PLACEHOLDER] = self.virutal_bond_emb.weight  # .half()
        return self.input_dropout(edge_h)


class TripletEmbedding(nn.Module):
    def __init__(
            self,
            d_g_feats,
            d_fp_feats,
            d_md_feats,
            activation=nn.GELU()):
        super(TripletEmbedding, self).__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(self, node_h, edge_h, fp, md, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        triplet_h[indicators == 1] = self.fp_proj(fp)
        triplet_h[indicators == 2] = self.md_proj(md)
        return triplet_h


class LiGhTPredictor(nn.Module):
    def __init__(self,
                 d_node_feats=40,
                 d_edge_feats=12,
                 d_g_feats=128,
                 d_fp_feats=512,
                 d_md_feats=200,
                 d_hpath_ratio=1,
                 n_mol_layers=2,
                 path_length=5,
                 n_heads=4,
                 n_ffn_dense_layers=2,
                 input_drop=0.,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU(),
                 n_node_types=1,
                 readout_mode='mean'
                 ):
        super(LiGhTPredictor, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Model
        self.model = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        # Predict
        # self.node_predictor = nn.Linear(d_g_feats, n_node_types)
        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_md_feats)
        )

        self.apply(lambda module: init_params(module))

    def forward(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h[g.ndata['mask'] == 1] = self.mask_emb.weight
        # Model
        triplet_h = self.model(g, triplet_h)
        # Predict
        return self.node_predictor(triplet_h[g.ndata['mask'] >= 1]), self.fp_predictor(
            triplet_h[indicators == 1]), self.md_predictor(triplet_h[indicators == 2])

    def forward_tune(self, g, fp, md, text=None, text_mask=None, mode='downstream'):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return self.predictor(g_feats)

    def generate_fps(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats


class LiGhT_graph_model(nn.Module):
    def __init__(self,
                 d_node_feats=40,
                 d_edge_feats=12,
                 d_g_feats=128,
                 d_fp_feats=512,
                 d_md_feats=200,
                 d_hpath_ratio=1,
                 n_mol_layers=2,
                 path_length=5,
                 n_heads=4,
                 n_ffn_dense_layers=2,
                 input_drop=0.,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU(),
                 n_node_types=1,
                 readout_mode='mean'
                 ):
        super(LiGhT_graph_model, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Model
        self.model = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        self.apply(lambda module: init_params(module))

    def forward(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        return readout

    def generate_fps(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats


class PharVQA(nn.Module):
    def __init__(self,
                 d_node_feats=40,
                 d_edge_feats=12,
                 d_g_feats=128,
                 d_fp_feats=512,
                 d_md_feats=200,
                 d_hpath_ratio=1,
                 n_mol_layers=2,
                 path_length=5,
                 n_heads=4,
                 n_ffn_dense_layers=2,
                 input_drop=0.,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU(),
                 n_node_types=1,
                 readout_mode='mean',
                 load=True
                 ):
        super(PharVQA, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Model
        self.model = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        self.graph_model_feat = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        self.text_model = TextEncoder(load=load)

        self.bcn = weight_norm(
            BANLayer(v_dim=d_g_feats, q_dim=768, h_dim=d_g_feats, h_out=4),
            name='h_mat', dim=None)

        # Predict
        self.graph_prompt_proj = nn.Linear(d_g_feats, d_g_feats)
        self.text_prompt_proj = nn.Linear(768, 768)

        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_md_feats)
        )

        self.apply(lambda module: init_params(module))

    def forward(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h[g.ndata['mask'] == 1] = self.mask_emb.weight
        # Model
        triplet_h = self.model(g, triplet_h)
        # Predict
        return self.node_predictor(triplet_h[g.ndata['mask'] >= 1]), self.fp_predictor(
            triplet_h[indicators == 1]), self.md_predictor(triplet_h[indicators == 2])

    def forward_tune_kpgt(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return self.predictor(g_feats)

    def forward_tune(self, g, fp, md, batch_text=None, batch_text_mask=None, softmax=False):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        molecule_node_repr = pad_sequence(torch.split(triplet_h[indicators <= 0], g.batch_num_nodes().tolist()),
                                          batch_first=False, padding_value=-999)
        vv2 = molecule_node_repr.new_ones(
            (max(g.batch_num_nodes().tolist()), g.batch_size, molecule_node_repr.shape[2])) * -999
        vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
        molecule_node_repr = vv2.transpose(0, 1)
        molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

        molecule_node_repr = self.graph_prompt_proj(molecule_node_repr)

        # text
        question_nums = batch_text.size()[1]
        text_reprs = [self.text_prompt_proj(self.get_text_repr(batch_text[:, idx, :], batch_text_mask[:, idx, :])) for
                      idx in range(question_nums)]
        # BAN network
        logits_list, att_list = [], []
        for question_idx in range(question_nums):
            logits_vad, att = self.bcn(molecule_node_repr, text_reprs[question_idx], molecule_node_mask,
                                       batch_text_mask[:, question_idx, :], softmax=softmax)
            logits_list.append(logits_vad)
            att_list.append(att[:, -1, :, :])

        molecule_repr_prompt = torch.stack(logits_list).transpose(0, 1)

        return molecule_repr_prompt, att_list

    def get_graph_feat(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.graph_model_feat(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return g_feats

    def get_graph_feat_only_graph(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.graph_model_feat(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        return readout

    def get_node_feat(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.graph_model_feat(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        molecule_node_repr = pad_sequence(torch.split(triplet_h[indicators <= 0], g.batch_num_nodes().tolist()),
                                          batch_first=False, padding_value=-999)
        vv2 = molecule_node_repr.new_ones(
            (max(g.batch_num_nodes().tolist()), g.batch_size, molecule_node_repr.shape[2])) * -999
        vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
        molecule_node_repr = vv2.transpose(0, 1)
        molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

        molecule_node_repr = self.graph_prompt_proj(molecule_node_repr)

        # fp_vn = triplet_h[indicators == 1]
        # md_vn = triplet_h[indicators == 2]
        #
        # g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        # readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        # g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return molecule_node_repr, molecule_node_mask

    def get_hnode_feat(self, g, fp, md, readout_mode='mean'):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.graph_model_feat(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])

        molecule_node_repr = pad_sequence(torch.split(triplet_h[indicators <= 0], g.batch_num_nodes().tolist()),
                                          batch_first=False, padding_value=-999)
        vv2 = molecule_node_repr.new_ones(
            (max(g.batch_num_nodes().tolist()), g.batch_size, molecule_node_repr.shape[2])) * -999
        vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
        molecule_node_repr = vv2.transpose(0, 1)
        molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

        molecule_node_repr = self.graph_prompt_proj(molecule_node_repr)

        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        # readout = dgl.readout_nodes(g, 'ht', op=readout_mode)
        # g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return molecule_node_repr, molecule_node_mask, g.ndata['ht']

    def generate_fps(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats

    def get_text_repr(self, text_tokens_ids, text_masks):
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output
        return text_repr


class PharVQA_ddi(nn.Module):
    def __init__(self,
                 d_node_feats=40,
                 d_edge_feats=12,
                 d_g_feats=128,
                 d_fp_feats=512,
                 d_md_feats=200,
                 d_hpath_ratio=1,
                 n_mol_layers=2,
                 path_length=5,
                 n_heads=4,
                 n_ffn_dense_layers=2,
                 input_drop=0.,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU(),
                 n_node_types=1,
                 readout_mode='mean',
                 load=True
                 ):
        super(PharVQA_ddi, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Model
        self.model = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        self.graph_model_feat_d1 = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        self.graph_model_feat_d2 = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        self.text_model = TextEncoder(load=load)

        self.bcn = weight_norm(
            BANLayer(v_dim=d_g_feats, q_dim=768, h_dim=d_g_feats, h_out=4),
            name='h_mat', dim=None)

        # Predict
        self.graph_prompt_proj = nn.Linear(d_g_feats, d_g_feats)
        self.text_prompt_proj = nn.Linear(768, 768)

        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_md_feats)
        )

        self.apply(lambda module: init_params(module))

    def forward(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h[g.ndata['mask'] == 1] = self.mask_emb.weight
        # Model
        triplet_h = self.model(g, triplet_h)
        # Predict
        return self.node_predictor(triplet_h[g.ndata['mask'] >= 1]), self.fp_predictor(
            triplet_h[indicators == 1]), self.md_predictor(triplet_h[indicators == 2])

    def forward_tune_kpgt(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return self.predictor(g_feats)

    def forward_tune(self, g, fp, md, batch_text=None, batch_text_mask=None, mode='downstream'):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        molecule_node_repr = pad_sequence(torch.split(triplet_h[indicators <= 0], g.batch_num_nodes().tolist()),
                                          batch_first=False, padding_value=-999)
        vv2 = molecule_node_repr.new_ones(
            (max(g.batch_num_nodes().tolist()), g.batch_size, molecule_node_repr.shape[2])) * -999
        vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
        molecule_node_repr = vv2.transpose(0, 1)
        molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

        molecule_node_repr = self.graph_prompt_proj(molecule_node_repr)

        # text
        question_nums = batch_text.size()[1]
        text_reprs = [self.text_prompt_proj(self.get_text_repr(batch_text[:, idx, :], batch_text_mask[:, idx, :])) for
                      idx in
                      range(question_nums)]
        # BAN network
        logits_list, att_list = [], []
        for question_idx in range(question_nums):
            logits_vad, att = self.bcn(molecule_node_repr, text_reprs[question_idx], molecule_node_mask,
                                       batch_text_mask[:, question_idx, :], softmax=True)
            logits_list.append(logits_vad)
            att_list.append(att[:, -1, :, :])

        molecule_repr_prompt = torch.stack(logits_list).transpose(0, 1)

        return molecule_repr_prompt, att_list

    def get_graph_feat(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.graph_model_feat(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return g_feats

    def get_graph_feat_only_graph(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.graph_model_feat(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        return readout

    def get_node_feat(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.graph_model_feat(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        molecule_node_repr = pad_sequence(torch.split(triplet_h[indicators <= 0], g.batch_num_nodes().tolist()),
                                          batch_first=False, padding_value=-999)
        vv2 = molecule_node_repr.new_ones(
            (max(g.batch_num_nodes().tolist()), g.batch_size, molecule_node_repr.shape[2])) * -999
        vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
        molecule_node_repr = vv2.transpose(0, 1)
        molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

        molecule_node_repr = self.graph_prompt_proj(molecule_node_repr)

        # fp_vn = triplet_h[indicators == 1]
        # md_vn = triplet_h[indicators == 2]
        #
        # g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        # readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        # g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return molecule_node_repr, molecule_node_mask

    def get_ddi_feat(self, g_1, fp_1, md_1, g_2, fp_2, md_2):
        indicators_d1 = g_1.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        indicators_d2 = g_2.ndata['vavn']

        # Input
        node_h = self.node_emb(g_1.ndata['begin_end'], indicators_d1)
        edge_h = self.edge_emb(g_1.ndata['edge'], indicators_d1)
        triplet_h_d1 = self.triplet_emb(node_h, edge_h, fp_1, md_1, indicators_d1)

        node_h = self.node_emb(g_2.ndata['begin_end'], indicators_d2)
        edge_h = self.edge_emb(g_2.ndata['edge'], indicators_d2)
        triplet_h_d2 = self.triplet_emb(node_h, edge_h, fp_2, md_2, indicators_d2)

        indicators_d1 = g_1.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        indicators_d2 = g_2.ndata['vavn']

        # Model
        triplet_h_d1 = self.graph_model_feat_d1(g_1, triplet_h_d1)
        triplet_h_d2 = self.graph_model_feat_d2(g_2, triplet_h_d2)

        # Readout
        g_1.ndata['ht'] = triplet_h_d1
        g_2.ndata['ht'] = triplet_h_d2

        fp_vn_1 = triplet_h_d1[indicators_d1 == 1]
        md_vn_1 = triplet_h_d1[indicators_d1 == 2]

        fp_vn_2 = triplet_h_d2[indicators_d2 == 1]
        md_vn_2 = triplet_h_d2[indicators_d2 == 2]

        g_1.remove_nodes(np.where(indicators_d1.detach().cpu().numpy() >= 1)[0])
        g_2.remove_nodes(np.where(indicators_d2.detach().cpu().numpy() >= 1)[0])

        readout_d1 = dgl.readout_nodes(g_1, 'ht', op=self.readout_mode)
        readout_d2 = dgl.readout_nodes(g_2, 'ht', op=self.readout_mode)

        g1_feats = torch.cat([fp_vn_1, md_vn_1, readout_d1], dim=-1)
        g2_feats = torch.cat([fp_vn_2, md_vn_2, readout_d2], dim=-1)
        return g1_feats, g2_feats

    def generate_fps(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats

    def get_text_repr(self, text_tokens_ids, text_masks):
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output
        return text_repr


from transformers import AutoModel
from transformers import BertModel, BertConfig


class TextEncoder(nn.Module):
    def __init__(self, pretrained=True, load=True):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            self.main_model = AutoModel.from_pretrained('../pretrained/scibert_scivocab_uncased')

            # self.main_model = BertModel.from_pretrained('bert_pretrained/')
        else:
            config = BertConfig(vocab_size=31090, )
            self.main_model = BertModel(config)
        if load:
            model_path = '../checkpoints/MoleculeSTM_checkpoints/pretrained_model/'
            text_model_path = os.path.join(model_path, "text_model.pth")
            print("Loading from pretrained text model (MoleculeSTM) {}...".format(
                os.path.join(text_model_path, 'text_model.pth')))
            state_dict = torch.load(text_model_path, map_location='cpu')
            self.main_model.load_state_dict(state_dict)

        self.dropout = nn.Dropout(0.1)
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, input_ids, attention_mask=None, embedding=None):
        device = input_ids.device
        typ = torch.zeros(input_ids.shape).long().to(device)
        output = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask)[0]  # b,d
        logits = self.dropout(output)
        return logits
