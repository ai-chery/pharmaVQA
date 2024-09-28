from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from copy import deepcopy

import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

sys.path.append('../')
sys.path.append('../retroCap/')
from models.decoder import TransformerDecoder
from models.embedding import Embedding
from models.encoder import TransformerEncoder
from models.module import MultiHeadedAttention
from src.model.light import PharVQA
from src.model.xbert import BertForMaskedLM, BertConfig


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


class retro_model(nn.Module):
    def __init__(self, args, config, encoder_num_layers, decoder_num_layers, d_model, heads, d_ff, dropout,
                 vocab_size_src, vocab_size_tgt, src_pad_idx, tgt_pad_idx, d_fps=512, d_mds=200, device='cpu'):
        super(retro_model, self).__init__()
        self.args = args
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.device = device

        # embedding module
        self.embedding_src = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx)
        self.embedding_tgt = Embedding(vocab_size=vocab_size_tgt + 1, embed_size=d_model, padding_idx=tgt_pad_idx)

        text_en_num_layers = encoder_num_layers // 2
        encoder_num_layers = encoder_num_layers - text_en_num_layers

        multihead_attn_modules_en_text = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, d_model, dropout=dropout)
             for _ in range(text_en_num_layers)])

        self.text_encoder = TransformerEncoder(num_layers=text_en_num_layers,
                                               d_model=d_model, heads=heads,
                                               d_ff=d_ff, dropout=dropout,
                                               embeddings=self.embedding_src,
                                               context_attn=multihead_attn_modules_en_text)

        multihead_attn_modules_en = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, 2 * config['d_g_feats'], dropout=dropout)
             for _ in range(encoder_num_layers)])

        multihead_attn_modules_de = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, d_model, dropout=dropout)
             for _ in range(decoder_num_layers)])

        self.encoder = TransformerEncoder(num_layers=encoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_src,
                                          context_attn=multihead_attn_modules_en)

        self.decoder = TransformerDecoder(num_layers=decoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_tgt,
                                          context_attn=multihead_attn_modules_de)

        # pharVQA side drug encoder
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
            n_node_types=vocab_size_src,
            load=True,
        )
        model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                      n_tasks=config['d_g_feats'],
                                                      n_layers=2, predictor_drop=0.0,
                                                      d_hidden_feats=config['d_g_feats'])
        model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1,
                                                  n_layers=2, predictor_drop=0.0,
                                                  d_hidden_feats=config['d_g_feats'])
        self.embedding_model = model

        self.gnn_z = nn.Sequential(
            nn.Linear(config['d_g_feats'] + config['d_g_feats'], self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model)
        )

        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt),
                                       nn.LogSoftmax(dim=-1))

        # define the embedding layer of retro model
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pooling_layer = AvgPooling()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, ecfp, md, text, text_mask, product_graph, product_graph_num=None):
        encoder_out, batch_graph_embed_z, pred_phar_num, pharVQA_atten = self.cross_encoder(src, product_graph,
                                                                                            product_graph_num,
                                                                                            ecfp, md,
                                                                                            text, text_mask)
        encoder_out = encoder_out.transpose(0, 1).contiguous()
        decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out)

        generative_scores = self.generator(decoder_out)

        return generative_scores, top_aligns, pred_phar_num, pharVQA_atten

    def cross_encoder(self, seq_index, g, graph_num, ecfp, md, text, text_mask):

        # graph feature
        batch_graph_embed, graph_mask, pred_phar_num, pharVQA_atten = self.get_graph_feature(ecfp, md, g, text,
                                                                                             text_mask)  # [B, N, d]
        batch_graph_embed_z = self.gnn_z(batch_graph_embed)
        # smiles input
        text_out, _ = self.text_encoder(seq_index, bond=None)
        text_graph_sim = torch.matmul(text_out.transpose(0, 1), batch_graph_embed_z.transpose(1, 2))
        for i in range(len(graph_num[0])):
            text_graph_sim[i, :, graph_num[0][i]:] = -1e18

        text_graph_sim = self.softmax(text_graph_sim).sum(dim=1)
        for i in range(len(graph_num[0])):
            text_graph_sim[i, graph_num[0][i]:] = -1e18
        text_graph_sim = self.softmax(text_graph_sim).unsqueeze(dim=2)
        weighted_batch_graph_embed = text_graph_sim * batch_graph_embed

        encoder_out, _ = self.encoder(seq_index, bond=None, emb=text_out,
                                      graph_embed=weighted_batch_graph_embed, graph_mask=graph_mask)  # [L, B, d]
        encoder_out = encoder_out.transpose(0, 1)
        return encoder_out, batch_graph_embed_z, pred_phar_num, pharVQA_atten

    def get_graph_feature(self, ecfp, md, g, text, text_mask):
        g_node = deepcopy(g)
        molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
        molecules_repr, input_attention_mask = self.embedding_model.get_node_feat(g_node, ecfp, md)
        molecules_prompt = self.embedding_model.prompt_projection_model(
            molecules_phar_prompt.reshape(molecules_repr.shape[0], -1)).unsqueeze(1).repeat(1, molecules_repr.shape[1],
                                                                                            1)
        graph_feature = torch.cat((molecules_prompt, molecules_repr), dim=-1)
        pred_phar_num = self.embedding_model.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(
            -1)
        return graph_feature, (1 - input_attention_mask).unsqueeze(1), pred_phar_num, atten
