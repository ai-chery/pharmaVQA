import json
import sys
from copy import deepcopy

import pandas as pd
from dgl import load_graphs
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from tdc import Evaluator
from tqdm import tqdm

sys.path.append('..')
from src.model.atten_module import MultiHeadedAttention
from src.model.encoder import TransformerEncoder
import seaborn as sns
from src.utils import set_random_seed
import argparse
import torch
import os
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import pharVQADataset, pharVQADataset_eval, phar_Dict
from src.data.collator import Collator_pharVQA, Collator_pharVQA_eval
from src.model.light import PharVQA

from src.model_config import config_dict

import warnings

# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
warnings.filterwarnings("ignore")
SPLIT_TO_ID = {'train': 0, 'val': 1, 'test': 2}


# torch.cuda.device_count()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def _eval_rmse(y_true, y_pred, mean=None, std=None):
    '''
        compute RMSE score averaged across tasks
    '''
    rmse_list = []
    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        if (mean is not None) and (std is not None):
            rmse_list.append(np.sqrt(
                ((y_true[is_labeled, i] - (y_pred[is_labeled, i] * std[i] + mean[i])) ** 2).mean()))
        else:
            rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))
    return sum(rmse_list) / len(rmse_list)


def _eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


# device = 'cpu'

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training pharVQA")
    parser.add_argument("--device", type=str, default='cuda:1')

    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--num_runs", type=int, default=1)

    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument('--noise_question', type=str, default='False')

    parser.add_argument("--save_path", type=str, default='../save/')
    parser.add_argument("--split", type=str, default='scaffold-0')
    parser.add_argument('--prompt_mode', type=str, default='cat')
    parser.add_argument('--normalize', type=str, default='True')
    parser.add_argument('--softmax', type=bool, default=True)

    parser.add_argument("--train_kpgt", type=str, default="False")
    parser.add_argument("--plot_attention", type=str, default="False")

    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_name", type=str, default='lipo')
    parser.add_argument("--dataset", type=str, default='lipo')
    parser.add_argument("--data_path", type=str, default='../datasets/benchmark')
    parser.add_argument("--dataset_type", type=str, default='regression')
    parser.add_argument("--metric", type=str, default='rmse')

    parser.add_argument('--path_length', type=int, default=5)

    parser.add_argument('--num_questions', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00003, help='model learning rate for training (default: 0.00003)')

    parser.add_argument("--n_threads", type=int, default=1)
    args = parser.parse_args()
    return args


def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
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
    return predictor.to(device)


def get_attention_layer(num_layers, heads, d_model_kv, d_model_q, dropout=0.0, device='cpu'):
    multihead_attn_modules = nn.ModuleList(
        [MultiHeadedAttention(heads, d_model_kv, d_model_q, dropout=dropout)
         for _ in range(num_layers)])

    encoder = TransformerEncoder(num_layers=num_layers,
                                 d_model=d_model_q, heads=heads,
                                 d_ff=d_model_q, dropout=dropout,
                                 attn_modules=multihead_attn_modules)
    return encoder.to(device)


def finetune_reg(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    args.save_map_path = f'{args.save_path}/attention_map/softmax_{args.softmax}/'
    print(f'dataset moleculesNet on {args.dataset} split {args.split}')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.save_map_path):
        os.makedirs(args.save_map_path)

    collator = Collator_pharVQA(config['path_length'])
    test_dataset = pharVQADataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type,
                                  question_num=args.num_questions, split='test',
                                  split_name=f'{args.split}', device=device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads,
                             worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    # load mean and std
    if args.normalize == 'True':
        cache_path = os.path.join(args.data_path, f"{args.model_name}/{args.model_name}_{args.path_length}.pkl")
        if args.model_name in ['muv', 'hiv', 'FGFR1', 'HPK1', 'PTP1B', 'PTPN2', 'VIM1', 'HPK1_KI']:
            split_path = os.path.join(args.data_path, f"{args.model_name}/splits/{args.split}.json")
            with open(split_path, 'r', encoding='utf-8') as fp:
                train_idxs = json.load(fp)[SPLIT_TO_ID['train']]
        else:
            split_path = os.path.join(args.data_path, f"{args.model_name}/splits/{args.split}.npy")
            train_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID['train']]
        _, label_dict = load_graphs(cache_path)
        mean, std = set_mean_and_std(label_dict['labels'][train_idxs])
        # mean, std = mean.to(device), std.to(device)

    # Model Initialization
    model = PharVQA(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=test_dataset.d_fps,
        d_md_feats=test_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
    if args.train_kpgt == "True":
        model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                        n_tasks=1,
                                        n_layers=2,
                                        predictor_drop=0.0,
                                        device=device, d_hidden_feats=config['d_g_feats'])
    else:
        model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                                  n_layers=2, predictor_drop=0.0, device=device,
                                                  d_hidden_feats=config['d_g_feats'])
        if args.prompt_mode == 'add':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'] * 3,
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                            n_tasks=1,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
        elif args.prompt_mode == 'cat':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'],
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 4,
                                            n_tasks=1,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    # Finetuning Setting
    # freeze pretrained text model
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}', map_location='cpu').items()})

    # eval
    spearman_e = Evaluator(name='Spearman')
    model.eval()
    predictions_all = []
    labels_all = []
    with torch.no_grad():
        for batch_index, batched_data in enumerate(tqdm(test_loader)):
            (smiles, g, ecfp, md, text, text_mask, phar_targets, labels, atom_phar_target_map, v_phar_atom_id,
             v_phar_name, random_questions) = batched_data
            ecfp, md, g, labels, text, text_mask, phar_targets = ecfp.to(args.device), md.to(args.device), g.to(
                args.device), labels.to(args.device), text.to(args.device), text_mask.to(args.device), phar_targets.to(
                args.device)
            if args.train_kpgt == 'False':
                g_1 = deepcopy(g)
                batch_size = batched_data[1].batch_size

                molecules_phar_prompt, atten = model.forward_tune(g, ecfp, md, text, text_mask, softmax=args.softmax)
                molecule_repr = model.get_graph_feat(g_1, ecfp, md)

                molecules_prompt = model.prompt_projection_model(molecules_phar_prompt.reshape(batch_size, -1))
                molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
                pred = model.predictor(molecules_prompt)

                if args.plot_attention == "True":
                    phar_target_mx = torch.zeros((batch_size, atten[0].size()[-2], len(atten)))
                    mols = [Chem.MolFromSmiles(s) for s in smiles]
                    batch_bond_atom_map = []
                    for mol_index, mol in enumerate(mols):
                        random_question = random_questions[mol_index]
                        mol_phar_mx = atom_phar_target_map[mol_index]
                        bond_atom_map = torch.zeros(len(mol.GetBonds())).tolist()
                        for bond_index, bond in enumerate(mol.GetBonds()):
                            begin_atom_id, end_atom_id = np.sort(
                                [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                            # phar_target_mx[mol_index, bond_index] = mol_phar_mx[
                            #     [begin_atom_id, end_atom_id], phar_Dict[random_question[0]]].sum(dim=0).to(
                            #     torch.bool).to(torch.float32)
                            bond_atom_map[bond_index] = [begin_atom_id, end_atom_id]
                            for question in random_question:
                                phar_target_mx[mol_index, bond_index, phar_Dict[question]] += mol_phar_mx[
                                    [begin_atom_id, end_atom_id], phar_Dict[question]].sum(dim=0).to(
                                    torch.bool).to(torch.float32)
                        batch_bond_atom_map.append(bond_atom_map)
                    atten = torch.stack(atten).transpose(0, 1)
                    for text_idx, text_token in enumerate(text):
                        random_question = random_questions[text_idx]
                        mol_phar_mx = atom_phar_target_map[text_idx]
                        bond_atom_map = batch_bond_atom_map[text_idx]
                        for question_index in range(atten.size()[1]):
                            question_name = random_question[question_index]
                            text_token_mask = text_mask[text_idx, question_index]
                            q_text_VQAindex = text_token[question_index, :text_token_mask.count_nonzero()]
                            q_text_VQA = test_dataset.tokenizer.convert_ids_to_tokens(q_text_VQAindex)

                            value_mt = atten[text_idx, question_index, :mols[text_idx].GetNumBonds(),
                                       :text_token_mask.count_nonzero()]
                            target_mt = phar_target_mx[text_idx, :mols[text_idx].GetNumBonds(),
                                        question_index].reshape(-1, 1).to(torch.bool)
                            min_value = value_mt.min()
                            max_value = value_mt.max()
                            norm_value_mt = (value_mt - min_value) / (max_value - min_value)

                            norm_min_value = 0
                            norm_max_value = 1
                            target_tensor = norm_min_value * torch.ones_like(target_mt, dtype=torch.float)
                            target_tensor[target_mt] = norm_max_value
                            # tensor_np = torch.cat((norm_value_mt, target_tensor), dim=-1).numpy()
                            # plt.figure(figsize=(20, 20))
                            # ax = sns.heatmap(tensor_np, cmap='coolwarm', linewidths=0.1, linecolor='white')
                            # ax.set_xticklabels(q_text_VQA + ['label'], rotation=30, fontsize=16)
                            # ax.set_yticklabels(range(mols[text_idx].GetNumBonds()), rotation=0, fontsize=16)
                            if not os.path.exists(
                                    f"{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}"):
                                os.makedirs(f"{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}")
                            # plt.savefig(
                            #     f'{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_hotmap.png')
                            # plt.close()

                            mol = mols[text_idx]
                            for atom in mol.GetAtoms():
                                atom.SetAtomMapNum(atom.GetIdx())
                            target_idx = mol_phar_mx[:, question_index]

                            target_atom_num = target_idx.nonzero(as_tuple=True)[0].tolist()
                            bond_atom_list_all = []
                            for target_atom_index in target_atom_num:
                                bond_atom_list = []
                                for bond_index, bond in enumerate(target_tensor.nonzero()[:, 0]):
                                    bond_atom_index = bond_atom_map[bond]
                                    if target_atom_index in bond_atom_index:
                                        bond_atom_list.append(norm_value_mt[bond])
                                #
                                if len(bond_atom_list) == 1:
                                    bond_atom_tensor = torch.stack(bond_atom_list).mean(dim=0)
                                else:
                                    bond_atom_tensor = torch.stack(bond_atom_list).mean(dim=0)
                                bond_atom_list_all.append(bond_atom_tensor)

                            if len(target_idx.nonzero().tolist()) == 0:
                                image = Chem.Draw.MolToImage(mol, size=(500, 500), kekulize=True)
                            else:
                                image = Chem.Draw.MolToImage(mol, size=(500, 500), kekulize=True,
                                                             highlightAtoms=target_idx.nonzero(as_tuple=True)[
                                                                 0].tolist())
                            image.save(
                                f'{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_mol.png')

                            if len(bond_atom_list_all) == 0:
                                continue
                            else:
                                sort_texts = torch.stack(bond_atom_list_all).argsort(dim=1)
                                text_rank = []
                                for index, sort_text in enumerate(sort_texts):
                                    text_rank.append([q_text_VQA[i] for i in sort_text])
                                with open(
                                        f'{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_mol.txt',
                                        'w', encoding='utf-8') as file:
                                    for data_index, data_slice in enumerate(text_rank):
                                        bond_num_2_atom_num = target_atom_num[data_index]
                                        list_str = ', '.join(data_slice)
                                        file.write('atom number' + str(bond_num_2_atom_num) + ': \t' + list_str + '\n')
            else:
                pred = model.forward_tune_kpgt(g, ecfp, md)

            predictions_all.append(pred.detach().cpu())
            labels_all.append(labels.detach().cpu())

    y_pred = torch.cat(predictions_all)
    y_true = torch.cat(labels_all)

    for i in range(y_true.numpy().shape[1]):
        # ignore nan values
        is_labeled = y_true.numpy()[:, i] == y_true.numpy()[:, i]
        if (mean is not None) and (std is not None):
            y_pred[is_labeled, i] = y_pred[is_labeled, i] * std[i] + mean[i]

    pear_results = pearsonr(y_true.squeeze(), y_pred.squeeze())[0]
    spearman_results = spearman_e(y_true.numpy(), y_pred.numpy())
    rmse_results = _eval_rmse(y_true.numpy(), y_pred.numpy(), None, None)

    print('rmse: {:.4f}'.format(rmse_results))
    print('pearson correlation: {:.4f}'.format(pear_results))
    print('spearman correlation: {:.4f}'.format(spearman_results))
    return pear_results, spearman_results


def finetune_cls(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    print(f'dataset moleculesNet on {args.dataset} split {args.split}')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    collator = Collator_pharVQA(config['path_length'])
    test_dataset = pharVQADataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type,
                                  question_num=args.num_questions, split='test',
                                  split_name=f'{args.split}', device=device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads,
                             worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

    # Model Initialization
    model = PharVQA(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=test_dataset.d_fps,
        d_md_feats=test_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
    if args.train_kpgt == "True":
        model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                        n_tasks=1,
                                        n_layers=2,
                                        predictor_drop=0.0,
                                        device=device, d_hidden_feats=config['d_g_feats'])
    else:
        model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                                  n_layers=2, predictor_drop=0.0, device=device,
                                                  d_hidden_feats=config['d_g_feats'])
        if args.prompt_mode == 'add':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'] * 3,
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                            n_tasks=test_dataset.n_tasks,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
        elif args.prompt_mode == 'cat':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'],
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 4,
                                            n_tasks=test_dataset.n_tasks,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
    # Finetuning Setting
    # freeze pretrained text model
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}', map_location='cpu').items()},
        strict=False)

    # eval
    spearman_e = Evaluator(name='Spearman')
    model.eval()
    predictions_all = []
    labels_all = []
    with torch.no_grad():
        for batch_index, batched_data in enumerate(tqdm(test_loader)):
            (smiles, g, ecfp, md, text, text_mask, phar_targets, labels, atom_phar_target_map, v_phar_atom_id,
             v_phar_name, random_questions) = batched_data
            ecfp, md, g, labels, text, text_mask, phar_targets = ecfp.to(args.device), md.to(args.device), g.to(
                args.device), labels.to(args.device), text.to(args.device), text_mask.to(args.device), phar_targets.to(
                args.device)
            if args.train_kpgt == 'False':
                g_1 = deepcopy(g)
                batch_size = batched_data[1].batch_size

                molecules_phar_prompt, atten = model.forward_tune(g, ecfp, md, text, text_mask, softmax=args.softmax)
                molecule_repr = model.get_graph_feat(g_1, ecfp, md)

                molecules_prompt = model.prompt_projection_model(molecules_phar_prompt.reshape(batch_size, -1))
                molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
                pred = model.predictor(molecules_prompt)

                if args.plot_attention == "True":
                    phar_target_mx = torch.zeros((batch_size, atten[0].size()[-2], len(atten)))
                    mols = [Chem.MolFromSmiles(s) for s in smiles]
                    batch_bond_atom_map = []
                    for mol_index, mol in enumerate(mols):
                        random_question = random_questions[mol_index]
                        mol_phar_mx = atom_phar_target_map[mol_index]
                        bond_atom_map = torch.zeros(len(mol.GetBonds())).tolist()
                        for bond_index, bond in enumerate(mol.GetBonds()):
                            begin_atom_id, end_atom_id = np.sort(
                                [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                            # phar_target_mx[mol_index, bond_index] = mol_phar_mx[
                            #     [begin_atom_id, end_atom_id], phar_Dict[random_question[0]]].sum(dim=0).to(
                            #     torch.bool).to(torch.float32)
                            bond_atom_map[bond_index] = [begin_atom_id, end_atom_id]
                            for question in random_question:
                                phar_target_mx[mol_index, bond_index, phar_Dict[question]] += mol_phar_mx[
                                    [begin_atom_id, end_atom_id], phar_Dict[question]].sum(dim=0).to(
                                    torch.bool).to(torch.float32)
                        batch_bond_atom_map.append(bond_atom_map)
                    atten = torch.stack(atten).transpose(0, 1)
                    for text_idx, text_token in enumerate(text):
                        random_question = random_questions[text_idx]
                        mol_phar_mx = atom_phar_target_map[text_idx]
                        bond_atom_map = batch_bond_atom_map[text_idx]
                        for question_index in range(atten.size()[1]):
                            question_name = random_question[question_index]
                            text_token_mask = text_mask[text_idx, question_index]
                            q_text_VQAindex = text_token[question_index, :text_token_mask.count_nonzero()]
                            q_text_VQA = test_dataset.tokenizer.convert_ids_to_tokens(q_text_VQAindex)

                            value_mt = atten[text_idx, question_index, :mols[text_idx].GetNumBonds(),
                                       :text_token_mask.count_nonzero()]
                            target_mt = phar_target_mx[text_idx, :mols[text_idx].GetNumBonds(),
                                        question_index].reshape(-1, 1).to(torch.bool)
                            min_value = value_mt.min()
                            max_value = value_mt.max()
                            norm_value_mt = (value_mt - min_value) / (max_value - min_value)

                            norm_min_value = 0
                            norm_max_value = 1
                            target_tensor = norm_min_value * torch.ones_like(target_mt, dtype=torch.float)
                            target_tensor[target_mt] = norm_max_value
                            # tensor_np = torch.cat((norm_value_mt, target_tensor), dim=-1).numpy()
                            # plt.figure(figsize=(20, 20))
                            # ax = sns.heatmap(tensor_np, cmap='coolwarm', linewidths=0.1, linecolor='white')
                            # ax.set_xticklabels(q_text_VQA + ['label'], rotation=30, fontsize=16)
                            # ax.set_yticklabels(range(mols[text_idx].GetNumBonds()), rotation=0, fontsize=16)
                            if not os.path.exists(
                                    f"{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}"):
                                os.makedirs(f"{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}")
                            # plt.savefig(
                            #     f'{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_hotmap.png')
                            # plt.close()

                            mol = mols[text_idx]
                            for atom in mol.GetAtoms():
                                atom.SetAtomMapNum(atom.GetIdx())
                            target_idx = mol_phar_mx[:, question_index]

                            target_atom_num = target_idx.nonzero(as_tuple=True)[0].tolist()
                            bond_atom_list_all = []
                            for target_atom_index in target_atom_num:
                                bond_atom_list = []
                                for bond_index, bond in enumerate(target_tensor.nonzero()[:, 0]):
                                    bond_atom_index = bond_atom_map[bond]
                                    if target_atom_index in bond_atom_index:
                                        bond_atom_list.append(norm_value_mt[bond])
                                #
                                if len(bond_atom_list) == 1:
                                    bond_atom_tensor = torch.stack(bond_atom_list).mean(dim=0)
                                else:
                                    bond_atom_tensor = torch.stack(bond_atom_list).mean(dim=0)
                                bond_atom_list_all.append(bond_atom_tensor)

                            if len(target_idx.nonzero().tolist()) == 0:
                                image = Chem.Draw.MolToImage(mol, size=(500, 500), kekulize=True)
                            else:
                                image = Chem.Draw.MolToImage(mol, size=(500, 500), kekulize=True,
                                                             highlightAtoms=target_idx.nonzero(as_tuple=True)[
                                                                 0].tolist())
                            image.save(
                                f'{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_mol.png')

                            if len(bond_atom_list_all) == 0:
                                continue
                            else:
                                sort_texts = torch.stack(bond_atom_list_all).argsort(dim=1)
                                text_rank = []
                                for index, sort_text in enumerate(sort_texts):
                                    text_rank.append([q_text_VQA[i] for i in sort_text])
                                with open(
                                        f'{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_mol.txt',
                                        'w', encoding='utf-8') as file:
                                    for data_index, data_slice in enumerate(text_rank):
                                        bond_num_2_atom_num = target_atom_num[data_index]
                                        list_str = ', '.join(data_slice)
                                        file.write('atom number' + str(bond_num_2_atom_num) + ': \t' + list_str + '\n')
            else:
                pred = model.forward_tune_kpgt(g, ecfp, md)

            predictions_all.append(pred)
            labels_all.append(labels)

    y_pred = torch.cat(predictions_all).detach().cpu().numpy()
    y_true = torch.cat(labels_all).detach().cpu().numpy()

    roc_results = _eval_rocauc(y_true, y_pred)
    return roc_results


def set_mean_and_std(labels, mean=None, std=None):
    if mean is None:
        mean = torch.from_numpy(np.nanmean(labels.numpy(), axis=0))
    if std is None:
        std = torch.from_numpy(np.nanstd(labels.numpy(), axis=0))
    return mean, std


def eval_drugbank(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    args.save_path = f'../save/{args.dataset}/question_num{args.num_questions}/{args.seed}/train_kpgt_{args.train_kpgt}/prompt_{args.prompt_mode}_normalize_{args.normalize}/'
    args.save_map_path = f'{args.save_path}/attention_map'
    print(f'dataset moleculesNet on {args.dataset} split {args.split}')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.save_map_path):
        os.makedirs(args.save_map_path)
    collator = Collator_pharVQA_eval(config['path_length'])

    test_dataset = pharVQADataset_eval(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type,
                                       question_num=args.num_questions,
                                       split_name=f'{args.split}', device=device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads,
                             worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

    # load mean and std
    split_path = os.path.join(args.data_path, f"{args.model_name}/splits/{args.split}.json")
    cache_path = os.path.join(args.data_path, f"{args.model_name}/{args.model_name}_{args.path_length}.pkl")
    with open(split_path, 'r', encoding='utf-8') as fp:
        train_idxs = json.load(fp)[SPLIT_TO_ID['train']]

    _, label_dict = load_graphs(cache_path)
    mean, std = set_mean_and_std(label_dict['labels'][train_idxs])
    mean, std = mean.to(device), std.to(device)
    # Model Initialization
    model = PharVQA(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=test_dataset.d_fps,
        d_md_feats=test_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
    if args.train_kpgt == "True":
        model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                        n_tasks=1,
                                        n_layers=2,
                                        predictor_drop=0.0,
                                        device=device, d_hidden_feats=config['d_g_feats'])
    else:
        model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                                  n_layers=2, predictor_drop=0.0, device=device,
                                                  d_hidden_feats=config['d_g_feats'])
        if args.prompt_mode == 'add':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'] * 3,
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                            n_tasks=1,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
        elif args.prompt_mode == 'cat':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'],
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 4,
                                            n_tasks=1,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
    # Finetuning Setting
    # freeze pretrained text model
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}', map_location='cpu').items()},
        strict=False)

    # eval
    model.eval()
    predictions_all = []
    smiles_all = []
    with torch.no_grad():
        for batch_index, batched_data in tqdm(enumerate(test_loader)):
            (smiles, g, ecfp, md, text, text_mask, phar_targets, atom_phar_target_map, v_phar_atom_id,
             v_phar_name, random_questions) = batched_data
            ecfp, md, g, text, text_mask, phar_targets = ecfp.to(args.device), md.to(args.device), g.to(
                args.device), text.to(args.device), text_mask.to(args.device), phar_targets.to(
                args.device)
            if args.train_kpgt == 'False':
                g_1 = deepcopy(g)
                batch_size = batched_data[1].batch_size

                molecules_phar_prompt, atten = model.forward_tune(g, ecfp, md, text, text_mask)
                molecule_repr = model.get_graph_feat(g_1, ecfp, md)

                molecules_prompt = model.prompt_projection_model(molecules_phar_prompt.reshape(batch_size, -1))
                if args.prompt_mode == 'add':
                    molecules_prompt = molecules_prompt + molecule_repr
                elif args.prompt_mode == 'cat':
                    molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
                # molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
                pred = model.predictor(molecules_prompt)
            else:
                pred = model.forward_tune_kpgt(g, ecfp, md)
            if args.normalize == 'True':
                pred = ((pred * std) + mean)
            predictions_all.append(pred.detach().cpu())
            smiles_all.extend(smiles)

    y_pred = torch.cat(predictions_all).detach().cpu().reshape(-1).numpy()
    # smiles_list = [item for sublist in smiles_all for item in sublist]
    return smiles_all, y_pred


if __name__ == '__main__':
    args = parse_args()
    results_list = []
    if args.dataset == 'DrugBank':
        if args.normalize == 'True':
            args.model_path = f'../save/{args.model_name}/question_num7/{args.split}/22/train_kpgt_{args.train_kpgt}/prompt_{args.prompt_mode}/best_model.pth'
        else:
            args.model_path = f'../save/{args.model_name}/question_num7/{args.split}/22/train_kpgt_{args.train_kpgt}/prompt_{args.prompt_mode}_normalize_{args.normalize}/best_model.pth'
        smiles_list, y_pred = eval_drugbank(args)
        # results_list.append([smiles, prediction])
        df = pd.DataFrame([smiles_list, y_pred]).T

        df_sorted = df.sort_values(by=1, ascending=False)
        df_sorted = df_sorted.rename(columns={0: 'SMILES', 1: 'pIC50'})

        ref_df = pd.read_csv(f"../datasets/vs/{args.dataset}/structure_links.csv")
        ref_df = ref_df.loc[:, ['DrugBank ID', 'SMILES']]
        ref_df.dropna(inplace=True)

        # 创建一个辅助DataFrame，用于排序
        df_sorted['Order'] = range(len(smiles_list))

        # 合并DataFrames，根据SMILES匹配
        df_merged = ref_df.merge(df_sorted, on='SMILES', how='right')

        # 根据Order列排序
        df_sorted = df_merged.sort_values(by='Order').drop(columns=['Order'])  # 移除辅助列

        df_sorted.to_csv(
            f'{args.save_path}/{args.dataset}_questionNum{args.num_questions}_{args.split}_{args.model_name}_eval_filter_top20.csv',
            index=False, header=False
        )

    else:
        args.model_path = f'../save/{args.dataset}/question_num{args.num_questions}/{args.split}/{args.seed}/train_kpgt_{args.train_kpgt}/prompt_{args.prompt_mode}/Noise_{args.noise_question}/best_model.pth'
        args.save_path = f'../save/{args.dataset}/question_num{args.num_questions}/{args.split}/{args.seed}/train_kpgt_{args.train_kpgt}/prompt_{args.prompt_mode}/Noise_{args.noise_question}'
        for runs in range(args.num_runs):
            args.seed = args.seed + runs
            set_random_seed(args.seed)
            if args.dataset_type == 'classification':
                results = finetune_cls(args)
            elif args.dataset_type == 'regression':
                results = finetune_reg(args)

            results_list.append([runs, results])
            print(f'Dataset {args.dataset} on {args.split} results is: {results}')
        print(results_list)
        df = pd.DataFrame(results_list)
        df.to_csv(
            f'{args.save_path}/{args.dataset}_questionNum{args.num_questions}_{args.split}_finetune.csv',
            mode='a', index=False, header=False
        )
