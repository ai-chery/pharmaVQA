from copy import deepcopy

import torch
from torch import nn

from src.model.light import PharVQA


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


class phar_dti_model(nn.Module):
    def __init__(self, args, config, vocab, d_fps, d_mds, \
                 dim=768, window=11, layer_cnn=3, layer_output=3, n_word=0):
        super(phar_dti_model, self).__init__()
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

        # protein side
        self.embed_word = nn.Embedding(n_word, dim)

        self.W_cnn = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=2 * window + 1,
            stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 1)
        self.layer_cnn = layer_cnn
        self.layer_output = layer_output
        self.dummy = False
        self.sigmoid = nn.Sigmoid()

        self.mol_norm = nn.LayerNorm(dim)
        self.protein_norm = nn.LayerNorm(dim)



    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""
        # x: compound, xs: protein (n,len,hid)

        xs = torch.unsqueeze(xs, 1)  # (n,1,len,hid)
        # print('xs',xs.shape)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        # print('xs1',xs.shape) #(n,1,len,hid)
        xs = torch.squeeze(xs, 1)
        # print('xs2',xs.shape)# (n,len,hid)

        h = torch.relu(self.W_attention(x))  # n,hid
        hs = torch.relu(self.W_attention(xs))  # n,len,hid
        weights = torch.tanh(torch.bmm(h.unsqueeze(1), hs.permute(0, 2, 1)))  # torch.tanh(F.linear(h, hs))#n,len
        ys = weights.permute(0, 2, 1) * hs  # n,l,h
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.mean(ys, 1)

    def forward(self, ecfp, md, g, text, text_mask, protein_batch):
        """Compound vector with attention-CNN."""
        # graph input
        g_1 = deepcopy(g)
        molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
        molecule_repr = self.embedding_model.get_graph_feat_only_graph(g_1, ecfp, md)
        # node_repr, input_attention_mask = self.embedding_model.get_node_feat(g_1, ecfp, md)

        molecules_prompt = self.embedding_model.prompt_projection_model(
            molecules_phar_prompt.reshape(molecule_repr.shape[0], -1))
        # molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
        molecules_repr = molecules_prompt + molecule_repr


        """Protein vector with attention-CNN."""
        protein_batch = torch.tensor(protein_batch, dtype=torch.int64)
        protein_batch = protein_batch.cuda()
        word_vectors = self.embed_word(protein_batch)
        # print('word',word_vectors.shape) #(len,hid)

        if self.mol_norm:
            molecules_repr = self.mol_norm(molecules_repr)
        if self.protein_norm:
            word_vectors = self.protein_norm(word_vectors)

        protein_vector1 = self.attention_cnn(molecules_repr, word_vectors, self.layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector1 = torch.cat((molecules_repr, protein_vector1), 1)
        for j in range(self.layer_output):
            cat_vector1 = torch.relu(self.W_out[j](cat_vector1))
        interaction = self.W_interaction(cat_vector1)

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
