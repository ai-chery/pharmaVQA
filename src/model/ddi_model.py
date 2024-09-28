from copy import deepcopy

import torch
from torch import nn

from src.model.light import PharVQA, PharVQA_ddi


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


class phar_ddi_model(nn.Module):
    def __init__(self, args, config, vocab, d_fps, d_mds, \
                 dim=768, window=11, layer_cnn=3, layer_output=3, n_word=0):
        super(phar_ddi_model, self).__init__()
        self.args = args
        model = PharVQA_ddi(
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
        model.out = get_predictor(d_input_feats=2 * (config['d_g_feats'] * 3), n_tasks=1, \
                                  n_layers=2, predictor_drop=0.0,
                                  d_hidden_feats=config['d_g_feats'])

        self.embedding_model = model

        self.W_out = nn.ModuleList([nn.Linear(2 * (config['d_g_feats']), 2 * (config['d_g_feats']))
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * (config['d_g_feats']), 1)
        self.layer_output = layer_output

    def forward(self, ecfp_1, md_1, g_1, ecfp_2, md_2, g_2, text, text_mask, ):
        """Compound vector with attention-CNN."""
        # graph input
        g_1_c = deepcopy(g_1)
        g_2_c = deepcopy(g_2)
        # molecules_phar_prompt, atten = self.embedding_model_1.forward_tune(g_1, ecfp_1, md_1, text, text_mask)
        molecules_repr_1, molecules_repr_2 = self.embedding_model.get_ddi_feat(g_1_c, ecfp_1, md_1, g_2_c, ecfp_2, md_2)

        # molecules_prompt = self.embedding_model_1.prompt_projection_model(
        #     molecules_phar_prompt.reshape(molecule_repr.shape[0], -1))
        # molecules_repr_1 = torch.cat((molecules_prompt, molecule_repr), dim=-1)
        # molecules_repr_1 = molecules_prompt + molecule_repr

        # graph input
        # molecules_phar_prompt, atten = self.embedding_model_2.forward_tune(g_2, ecfp_2, md_2, text, text_mask)

        # molecules_prompt = self.embedding_model_2.prompt_projection_model(
        #     molecules_phar_prompt.reshape(molecule_repr.shape[0], -1))
        # molecules_repr_2 = torch.cat((molecules_prompt, molecule_repr), dim=-1)
        # molecules_repr_2 = molecules_prompt + molecule_repr

        """Concatenate the above two vectors and output the interaction."""
        cat_vector1 = torch.cat((molecules_repr_1, molecules_repr_2), 1)
        # for j in range(self.layer_output):
        #     cat_vector1 = torch.relu(self.W_out[j](cat_vector1))
        # interaction = self.W_interaction(cat_vector1)
        interaction = self.embedding_model.out(cat_vector1)
        return interaction

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
