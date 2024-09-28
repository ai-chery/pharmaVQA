import copy
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical

from src.model.atten_module import MultiHeadedAttention
from src.model.encoder import TransformerEncoder
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


def get_attention_layer(num_layers, heads, d_model_kv, d_model_q, dropout=0.0, device='cpu'):
    multihead_attn_modules = nn.ModuleList(
        [MultiHeadedAttention(heads, d_model_kv, d_model_q, dropout=dropout)
         for _ in range(num_layers)])

    encoder = TransformerEncoder(num_layers=num_layers,
                                 d_model=d_model_q, heads=heads,
                                 d_ff=d_model_q, dropout=dropout,
                                 attn_modules=multihead_attn_modules)
    return encoder


class phar_yeild_model(nn.Module):
    def __init__(self, args, config, vocab, d_fps, d_mds):
        super().__init__()

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
        model.prompt_aggregation_model = get_attention_layer(num_layers=2, heads=4, d_model_kv=config['d_g_feats'],
                                                             d_model_q=config['d_g_feats'],
                                                             dropout=0.0, )
        self.embedding_model_r = model

        model2 = PharVQA(
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
        model2.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                       n_tasks=config['d_g_feats'],
                                                       n_layers=2, predictor_drop=0.0,
                                                       d_hidden_feats=config['d_g_feats'])
        model2.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                                   n_layers=2, predictor_drop=0.0,
                                                   d_hidden_feats=config['d_g_feats'])
        model2.prompt_aggregation_model = get_attention_layer(num_layers=2, heads=4, d_model_kv=config['d_g_feats'],
                                                              d_model_q=config['d_g_feats'],
                                                              dropout=0.0, )
        self.embedding_model_p = model2

        self.project_r = get_predictor(d_input_feats=(config['d_g_feats'] * 4), n_tasks=config['d_g_feats'], \
                                       n_layers=2, predictor_drop=0.0,
                                       d_hidden_feats=config['d_g_feats'])
        self.project_p = get_predictor(d_input_feats=(config['d_g_feats'] * 4), n_tasks=config['d_g_feats'], \
                                       n_layers=2, predictor_drop=0.0,
                                       d_hidden_feats=config['d_g_feats'])
        self.out = get_predictor(d_input_feats=2 * (config['d_g_feats']), n_tasks=1, \
                                 n_layers=2, predictor_drop=0.0,
                                 d_hidden_feats=config['d_g_feats'])
    def forward(self, ecfp_r, md_r, g_r, ecfp_p, md_p, g_p, text, text_mask):

        # BAN question prompt

        """prompt + graph input"""
        g_r_1 = deepcopy(g_r)
        molecules_phar_prompt_r, atten_r = self.embedding_model_r.forward_tune(g_r, ecfp_r, md_r, text, text_mask)
        molecules_repr_r = self.embedding_model_r.get_graph_feat(g_r_1, ecfp_r, md_r)
        molecules_prompt_r = self.embedding_model_r.prompt_projection_model(
            molecules_phar_prompt_r.reshape(molecules_repr_r.shape[0], -1))
        molecules_prompt_rep_r = torch.cat((molecules_prompt_r, molecules_repr_r), dim=-1)
        pred_phar_num_r = self.embedding_model_r.prompt_linear_model(molecules_phar_prompt_r).to(torch.float64).squeeze(
            -1)

        g_p_1 = deepcopy(g_p)
        molecules_phar_prompt_p, atten_p = self.embedding_model_p.forward_tune(g_p, ecfp_p, md_p, text, text_mask)
        molecules_repr_p = self.embedding_model_p.get_graph_feat(g_p_1, ecfp_p, md_p)
        molecules_prompt_p = self.embedding_model_p.prompt_projection_model(
            molecules_phar_prompt_p.reshape(molecules_repr_p.shape[0], -1))
        molecules_prompt_rep_p = torch.cat((molecules_prompt_p, molecules_repr_p), dim=-1)
        pred_phar_num_p = self.embedding_model_p.prompt_linear_model(molecules_phar_prompt_p).to(torch.float64).squeeze(
            -1)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((self.project_r(molecules_prompt_rep_r), self.project_p(molecules_prompt_rep_p)),dim=1)
        prediction = self.out(cat_vector)

        return prediction, pred_phar_num_r, pred_phar_num_p, atten_r, atten_p

    def load_model(self, model_path, text_model_path):
        self.embedding_model_r.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(f'{model_path}', map_location='cpu').items()},
            strict=False)
        for name, param in self.embedding_model_r.model.state_dict().items():
            self.embedding_model_r.graph_model_feat.state_dict()[name].copy_(param)

        self.embedding_model_p.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(f'{model_path}', map_location='cpu').items()},
            strict=False)
        for name, param in self.embedding_model_p.model.state_dict().items():
            self.embedding_model_p.graph_model_feat.state_dict()[name].copy_(param)
