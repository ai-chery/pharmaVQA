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


class phar_rxn_model(nn.Module):
    def __init__(self, args, config, cls_config, vocab, d_fps, d_mds, cp=None):
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
        self.embedding_model = model

        self.geration_model = BertForMaskedLM(config=BertConfig.from_json_file(cls_config['bert_config_text']))
        self.text_encoder2 = BertForMaskedLM(config=BertConfig.from_json_file(cls_config['bert_config_smiles']))
        # copy weights of checkpoint's SMILES encoder to text_encoder2
        if cp:
            checkpoint = torch.load(cp, map_location='cpu')
            try:
                state_dict = copy.deepcopy(checkpoint['model'])
            except:
                state_dict = copy.deepcopy(checkpoint['state_dict'])
            for key in list(state_dict.keys()):
                if 'text_encoder.' in key:
                    new_key = key.replace('text_encoder.', '')
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
            msg = self.text_encoder2.load_state_dict(state_dict, strict=False)
            print('cp text_encoder successfully')
            # print(msg)
            del state_dict

        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, ecfp, md, g, text, text_mask, input_ids, input_attention_mask, output_ids,
                product_attention_mask):

        # BAN question prompt

        """graph input"""
        # molecule_repr = self.embedding_model.get_graph_feat(g, ecfp, md)
        # molecules_repr, input_attention_mask = self.embedding_model.get_node_feat(g, ecfp, md)

        """prompt input"""
        # molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
        # molecules_prompt = self.embedding_model.prompt_projection_model(
        #     molecules_phar_prompt.reshape(ecfp.shape[0], -1)).unsqueeze(1)
        # input_attention_mask = torch.ones(ecfp.shape[0], 1).to(ecfp.device)

        """prompt + graph input"""
        g_1 = deepcopy(g)
        molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
        molecules_repr, input_attention_mask = self.embedding_model.get_node_feat(g_1, ecfp, md)
        molecules_prompt = self.embedding_model.prompt_projection_model(
            molecules_phar_prompt.reshape(molecules_repr.shape[0], -1)).unsqueeze(1).repeat(1, molecules_repr.shape[1], 1)
        molecules_prompt = torch.cat((molecules_prompt, molecules_repr), dim=-1)
        # molecules_repr = molecules_prompt + molecules_repr

        """prompt + smiles input"""
        # molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
        # text_embeds = self.text_encoder2.bert(input_ids, attention_mask=input_attention_mask, return_dict=True,
        #                                       mode='text').last_hidden_state
        # text_embeds, atten = self.embedding_model.prompt_aggregation_model(text_embeds, molecules_phar_prompt)
        # molecules_prompt = self.embedding_model.prompt_projection_model(
        #     molecules_phar_prompt.reshape(molecules_phar_prompt.shape[0], -1)).unsqueeze(1)
        # molecules_repr = torch.cat((molecules_prompt, text_embeds), dim=1)
        #
        # input_attention_mask = torch.cat(
        #     (torch.ones(input_attention_mask.shape[0], 1).to(self.args.device), input_attention_mask), dim=1)

        # decode
        input_ids_c = output_ids.clone()
        labels = input_ids_c.clone()[:, 1:]
        mlm_output = self.geration_model(input_ids_c,
                                         attention_mask=product_attention_mask,
                                         encoder_hidden_states=molecules_repr,
                                         encoder_attention_mask=input_attention_mask,
                                         return_dict=True,
                                         is_decoder=True,
                                         return_logits=True,
                                         )[:, :-1, :]

        loss_mlm = self.loss(mlm_output.permute((0, 2, 1)), labels)

        return loss_mlm

    def generate(self, text_embeds, text_mask, product_input, stochastic=False, k=None):
        product_atts = torch.where(product_input == 0, 0, 1)

        token_output = self.geration_model(product_input,
                                           attention_mask=product_atts,
                                           encoder_hidden_states=text_embeds,
                                           encoder_attention_mask=text_mask,
                                           return_dict=True,
                                           is_decoder=True,
                                           return_logits=True,
                                           )[:, -1, :]  # batch*300
        if k:
            p = torch.softmax(token_output, dim=-1)
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
        if stochastic:
            p = torch.softmax(token_output, dim=-1)
            m = Categorical(p)
            token_output = m.sample()
        else:
            token_output = torch.argmax(token_output, dim=-1)
        return token_output.unsqueeze(1)  # batch*1
