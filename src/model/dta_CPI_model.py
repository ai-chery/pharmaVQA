import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_max_pool as gmp, global_mean_pool as gep

from src.model.ban import BANLayer
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


class pharVQA_dta_CPI(torch.nn.Module):
    def __init__(self, args, device, config, d_fps, d_mds, vocab, emb_size, max_length, num_questions=7, n_output=1,
                 hidden_size=128, num_features_mol=78, dropout=None):
        super(pharVQA_dta_CPI, self).__init__()

        print('CPI_regression model Loaded..')
        self.device = device
        self.skip = 1
        self.n_output = n_output
        self.max_length = max_length
        self.args = args

        # proteins network
        self.prot_rnn = nn.LSTM(emb_size, hidden_size, 1)
        self.relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)

        # compounds network
        if args.use_encoder == 'KPGT':
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
            model.prompt_projection_model = get_predictor(d_input_feats=(num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'],
                                                          n_layers=2, predictor_drop=0.0,
                                                          d_hidden_feats=config['d_g_feats'])
            model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                                      n_layers=2, predictor_drop=0.0,
                                                      d_hidden_feats=config['d_g_feats'])
            self.embedding_model = model
            self.prot_comp_mix = nn.Sequential(nn.Linear((config['d_g_feats'] * 4 + 1) * (hidden_size + 1), 1024),
                                               nn.LeakyReLU(),
                                               nn.Dropout(dropout))
            self.fc = nn.Sequential(nn.Linear(1024 + (config['d_g_feats'] * 4 + 1) + (hidden_size + 1), 512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(512, self.n_output))

        elif args.use_encoder == 'SAGE':

            # graph encoder
            self.g1 = MolecularGraphNet(num_features_mol, hidden_size)
            self.g2 = MolecularGraphNet(num_features_mol, hidden_size)

            # prompt side
            self.text_model = TextEncoder(load=True)
            self.text_prompt_proj = nn.Linear(768, 768)

            self.bcn = weight_norm(
                BANLayer(v_dim=hidden_size, q_dim=768, h_dim=hidden_size, h_out=4),
                name='h_mat', dim=None)

            self.prompt_projection_model = get_predictor(d_input_feats=(num_questions) * hidden_size,
                                                         n_tasks=hidden_size,
                                                         n_layers=2, predictor_drop=0.0,
                                                         d_hidden_feats=hidden_size)
            self.prompt_linear_model = get_predictor(d_input_feats=hidden_size, n_tasks=1, \
                                                     n_layers=2, predictor_drop=0.0,
                                                     d_hidden_feats=hidden_size)

            self.prot_comp_mix = nn.Sequential(nn.Linear((hidden_size * 2 + 1) * (hidden_size + 1), 1024),
                                               nn.LeakyReLU(),
                                               nn.Dropout(dropout))
            self.fc = nn.Sequential(nn.Linear(1024 + (hidden_size * 2 + 1) + (hidden_size + 1), 512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(512, self.n_output))

    def forward(self, ecfp, md, g, data_mol, text, text_mask, data_pro, data_pro_len):

        if self.args.use_encoder == 'KPGT':
            '''use KPGT encoder'''
            g_1 = deepcopy(g)
            molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
            molecules_prompt = self.embedding_model.prompt_projection_model(
                molecules_phar_prompt.reshape(molecules_phar_prompt.shape[0], -1))

            # graph input
            molecule_repr = self.embedding_model.get_graph_feat(g_1, ecfp, md)
            molecules_repr = torch.cat((molecules_prompt, molecule_repr), dim=-1)
            # pharVQA count loss
            pred_phar_num = self.embedding_model.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(
                -1)

        elif self.args.use_encoder == 'SAGE':
            '''use CPI encoder'''
            # extract molecules representation
            mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch

            graph_rep = self.g1(mol_x, mol_edge_index)
            graph_rep = self.dropout(graph_rep)

            unique_labels, counts = torch.unique(mol_batch, return_counts=True)
            molecule_node_repr = pad_sequence(torch.split(graph_rep, counts.tolist()),
                                              batch_first=False, padding_value=-999)

            vv2 = molecule_node_repr.new_ones(
                (max(g.batch_num_nodes().tolist()), len(counts), molecule_node_repr.shape[2])) * -999
            vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
            molecule_node_repr = vv2.transpose(0, 1)
            molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

            # text
            question_nums = text.size()[1]
            text_reprs = [self.text_prompt_proj(self.get_text_repr(text[:, idx, :], text_mask[:, idx, :]))
                          for idx in range(question_nums)]
            # BAN network
            logits_list, atten = [], []
            for question_idx in range(question_nums):
                logits_vad, att = self.bcn(molecule_node_repr, text_reprs[question_idx], molecule_node_mask,
                                           text_mask[:, question_idx, :], softmax=True)
                logits_list.append(logits_vad)
                atten.append(att[:, -1, :, :])

            # fusion
            molecules_phar_prompt = torch.stack(logits_list).transpose(0, 1)
            molecules_prompt = self.prompt_projection_model(
                molecules_phar_prompt.reshape(molecules_phar_prompt.shape[0], -1))

            graph_rep = self.g2(mol_x, mol_edge_index)
            molecules_rep = gmp(graph_rep, mol_batch)  # global max pooling
            molecules_rep = self.dropout(molecules_rep)

            # cat
            molecules_repr = torch.cat((molecules_prompt, molecules_rep), dim=-1)
            pred_phar_num = self.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(-1)

        # protein network
        pro_seq_lengths, pro_idx_sort = torch.sort(data_pro_len.view(-1), descending=True)[::-1][1], torch.argsort(
            -data_pro_len.view(-1))
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True)
        xt, _ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=max(1500, self.max_length))[0]
        xt = xt.index_select(0, pro_idx_unsort)
        xt = xt.mean(1)

        # kronecker product
        prot_out = torch.cat((xt, torch.FloatTensor(xt.shape[0], 1).fill_(1).to(self.device)), 1)
        comp_out = torch.cat((molecules_repr, torch.FloatTensor(molecules_repr.shape[0], 1).fill_(1).to(self.device)),
                             1)
        output = torch.bmm(prot_out.unsqueeze(2), comp_out.unsqueeze(1)).flatten(start_dim=1)
        output = self.dropout(output)
        output = self.prot_comp_mix(output)
        if self.skip:
            output = torch.cat((output, prot_out, comp_out), 1)
        output = self.fc(output)

        return output, pred_phar_num, atten

    def get_text_repr(self, text_tokens_ids, text_masks):
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output
        return text_repr


class MolecularGraphNet(nn.Module):
    def __init__(self, num_features_mol, hidden_size):
        super(MolecularGraphNet, self).__init__()
        self.conv1 = SAGEConv(num_features_mol, num_features_mol * 2, 'mean')
        self.conv2 = SAGEConv(num_features_mol * 2, num_features_mol * 2, 'mean')
        self.conv3 = SAGEConv(num_features_mol * 2, num_features_mol * 4, 'mean')

        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, hidden_size)

        self.fc1 = nn.Linear(num_features_mol * 4, hidden_size)
        self.fc2 = nn.Linear(num_features_mol * 4, hidden_size)

    def forward(self, x, edge_index):
        # 第一个图卷积层
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # 第二个图卷积层
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # 第三个图卷积层
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        out = self.mol_fc_g1(x)

        return out


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
