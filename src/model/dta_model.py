import os
from copy import deepcopy

from rdkit import Chem
from torch_geometric.nn import global_mean_pool
import torch
from torch import nn

from DTA.encoder import PosEncoder
from DTA.model import GNN_v2, GNNComplete_v2
from src.model.light import PharVQA

import torch.nn.functional as F


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise NotImplementedError()


class TokenMAEClf(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("GNNTransformer - Training Config")
        ## gnn parameters
        group.add_argument('--gnn_emb_dim', type=int, default=300,
                           help='dimensionality of hidden units in GNNs (default: 300)')
        group.add_argument('--gnn_dropout', type=float, default=0.5)
        group.add_argument('--gnn_JK', type=str, default='last')
        group.add_argument('--gnn_type', type=str, default='gin_v3')
        group.add_argument("--gnn_activation", type=str, default="relu")

        ## transformer parameters
        group.add_argument('--d_model', type=int, default=128)
        group.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument("--transformer_dropout", type=float, default=0.3)
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--transformer_norm_input", action="store_true", default=True)
        group.add_argument('--custom_trans', action='store_true', default=True)
        # group.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")

        ## encoder parameters
        group.add_argument('--gnn_token_layer', type=int, default=1)
        group.add_argument('--gnn_encoder_layer', type=int, default=5)
        group.add_argument('--trans_encoder_layer', type=int, default=4)
        group.add_argument('--freeze_token', action='store_true', default=False)
        group.add_argument('--trans_pooling', type=str, default='none')
        group.add_argument('--complete_feature', action='store_true', default=False)

        group_pe = parser.add_argument_group("PE Config")
        group_pe.add_argument('--pe_type', type=str, default='none',
                              choices=['none', 'signnet', 'lap', 'lap_v2', 'signnet_v2', 'rwse', 'signnet_v3'])
        group_pe.add_argument('--laplacian_norm', type=str, default='none')
        group_pe.add_argument('--max_freqs', type=int, default=20)
        group_pe.add_argument('--eigvec_norm', type=str, default='L2')
        group_pe.add_argument('--raw_norm_type', type=str, default='none', choices=['none', 'batchnorm'])
        group_pe.add_argument('--kernel_times', type=list, default=[])  # cmd line param not supported yet
        group_pe.add_argument('--kernel_times_func', type=str, default='none')
        group_pe.add_argument('--layers', type=int, default=3)
        group_pe.add_argument('--post_layers', type=int, default=2)
        group_pe.add_argument('--dim_pe', type=int, default=28, help='dim of node positional encoding')
        group_pe.add_argument('--phi_hidden_dim', type=int, default=32)
        group_pe.add_argument('--phi_out_dim', type=int, default=32)

    def __init__(self, freeze_token, token_layer, encoder_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin",
                 d_model=128, trans_encoder_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0,
                 transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, args=None):
        super().__init__()
        assert JK == 'last'
        self.pos_encoder = PosEncoder(args)
        self.freeze_token = freeze_token
        self.gnn_act = get_activation(args.gnn_activation)

        if args.complete_feature:
            self.tokenizer = GNNComplete_v2(1, emb_dim, True, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type,
                                            gnn_activation=args.gnn_activation)
            self.encoder = GNNComplete_v2(encoder_layer - 1, emb_dim, False, JK=JK, drop_ratio=drop_ratio,
                                          gnn_type=gnn_type, gnn_activation=args.gnn_activation,
                                          d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead,
                                          dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout,
                                          transformer_activation=transformer_activation,
                                          transformer_norm_input=transformer_norm_input, custom_trans=custom_trans,
                                          pe_dim=self.pos_encoder.pe_dim, trans_pooling=args.trans_pooling)
        else:
            self.tokenizer = GNN_v2(1, emb_dim, True, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type,
                                    gnn_activation=args.gnn_activation)
            self.encoder = GNN_v2(encoder_layer - 1, emb_dim, False, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type,
                                  gnn_activation=args.gnn_activation,
                                  d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead,
                                  dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout,
                                  transformer_activation=transformer_activation,
                                  transformer_norm_input=transformer_norm_input, custom_trans=custom_trans,
                                  pe_dim=self.pos_encoder.pe_dim, trans_pooling=args.trans_pooling)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.freeze_token:
            with torch.no_grad():
                h = self.tokenizer(x, edge_index, edge_attr).detach()
        else:
            h = self.tokenizer(x, edge_index, edge_attr)

        pe_tokens = self.pos_encoder(data)
        h = self.encoder(self.gnn_act(h), edge_index, edge_attr, data.batch, pe_tokens=pe_tokens)
        return h


class ConvMLP(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("MLP Conv - Training Config")
        # protein embbeding dimensions
        group.add_argument('--num_features', type=int, default=25)
        group.add_argument('--prot_emb_dim', type=int, default=128)
        group.add_argument('--prot_output_dim', type=int, default=128)
        # conv settings
        group.add_argument('--n_filters', type=int, default=32)
        group.add_argument('--kernel_size', type=int, default=8)
        group.add_argument('--in_channels', type=int, default=1000)
        # mlp settings
        group.add_argument('--mlp_dim1', type=int, default=1024)
        group.add_argument('--mlp_dim2', type=int, default=256)
        group.add_argument('--mlp_dropout', type=float, default=0.0)
        group.add_argument('--conv_norm', type=str, default='layer')
        group.add_argument("--clf_norm", type=str, default='layer')

    def __init__(self, args=None):
        super(ConvMLP, self).__init__()
        self.dropout = nn.Dropout(args.mlp_dropout)
        self.relu = nn.ReLU()

        # 1D conv on protein
        self.embedding_xt = nn.Embedding(args.num_features + 1, args.prot_emb_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=args.in_channels, out_channels=args.n_filters,
                                   kernel_size=args.kernel_size)
        intermediate_dim = args.prot_emb_dim - args.kernel_size + 1
        self.fc1_xt_dim = args.n_filters * intermediate_dim
        self.fc1_xt = nn.Linear(self.fc1_xt_dim, args.prot_output_dim)

        # mlp for conbined features
        if args.trans_encoder_layer > 0:
            clf_emb_dim = args.d_model
        else:
            clf_emb_dim = args.gnn_emb_dim
        input_dim = clf_emb_dim + args.prot_output_dim

        self.fc1 = nn.Linear(input_dim, args.mlp_dim1)
        self.fc2 = nn.Linear(args.mlp_dim1, args.mlp_dim2)
        self.out = nn.Linear(args.mlp_dim2, args.num_task)

        torch.nn.init.xavier_uniform_(self.embedding_xt.weight.data)
        # norm for conv model
        if args.conv_norm == 'layer':
            self.conv_norm = nn.LayerNorm(args.prot_output_dim)
        elif args.conv_norm == 'l2':
            self.conv_norm = lambda x: nn.functional.normalize(x, p=2, dim=-1)
        else:
            self.conv_norm = None

        # norm for gnn/graphTrans model
        if args.clf_norm == 'layer':
            self.clf_norm = nn.LayerNorm(clf_emb_dim)
        elif args.clf_norm == 'l2':
            self.clf_norm = lambda x: nn.functional.normalize(x, p=2, dim=-1)
        else:
            self.clf_norm = None

    def forward(self, h, target):
        # 1D conv on protein
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, self.fc1_xt_dim)
        xt = self.fc1_xt(xt)

        # mlp for conbined features
        if self.conv_norm:
            xt = self.conv_norm(xt)
        if self.clf_norm:
            h = self.clf_norm(h)
        h = torch.cat((h, xt), 1)
        h = self.dropout(self.relu(self.fc1(h)))
        h = self.dropout(self.relu(self.fc2(h)))
        out = self.out(h)
        return out


class phar_dta_model(nn.Module):
    def __init__(self, args, config, vocab, d_fps, d_mds, gnn):
        super(phar_dta_model, self).__init__()
        self.args = args
        model = PharVQA(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=d_fps,
            d_md_feats=d_mds,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=config['dropout'],
            feat_drop=config['dropout'],
            n_node_types=vocab.vocab_size,
            load=True,
        )
        model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                      n_tasks=config['d_g_feats'],
                                                      n_layers=2, predictor_drop=0.0,
                                                      d_hidden_feats=config['d_g_feats'])
        model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                                  n_layers=2, predictor_drop=0.0,
                                                  d_hidden_feats=config['d_g_feats'])
        self.embedding_model = model

        self.gnn = gnn
        self.pool = global_mean_pool
        args.gnn_emb_dim = config['d_g_feats'] + config['d_g_feats'] + config['d_g_feats'] + config['d_g_feats']
        # protein side
        self.convmlp = ConvMLP(args)

        self.in_channels = args.in_channels

    def forward(self, ecfp, md, g, text, text_mask, protein_batch, data):
        """Compound vector with attention-CNN."""
        batch_size = ecfp.size()[0]
        # graph input
        g_1 = deepcopy(g)
        molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
        molecules_prompt = self.embedding_model.prompt_projection_model(
            molecules_phar_prompt.reshape(molecules_phar_prompt.shape[0], -1))
        molecule_repr = self.embedding_model.get_graph_feat(g_1, ecfp, md)
        molecules_repr = torch.cat((molecules_prompt, molecule_repr), dim=-1)
        # GNN input
        # node_representation = self.gnn(data)
        # graph_rep = self.pool(node_representation, data.batch)
        # molecules_repr = molecules_prompt + graph_rep
        # molecules_repr = torch.cat((molecules_prompt, graph_rep), dim=-1)

        # molecule_repr = self.embedding_model.get_graph_feat_only_graph(g_1, ecfp, md)
        # # node_repr, input_attention_mask = self.embedding_model.get_node_feat(g_1, ecfp, md)
        # molecules_repr = torch.cat((molecules_prompt, molecule_repr), dim=-1)

        interaction = self.convmlp(molecules_repr, protein_batch.reshape(-1, self.in_channels))

        # pharVQA count loss
        pred_phar_num = self.embedding_model.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(-1)

        return interaction, pred_phar_num, atten

    @staticmethod
    def get_loss_func(args):
        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=None):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression':
                pred_loss = nn.MSELoss(reduction='none')
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # TODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # in eval mode.
                return pred_loss(preds, targets.float())

            # in train mode.
            dist_loss = nn.MSELoss(reduction='none')
            # dist_loss = nn.CosineSimilarity(dim=0)
            # print(pred_loss)

            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func
