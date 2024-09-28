import argparse
import json
import pickle
import sys
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, r2_score
from torch.nn import BCEWithLogitsLoss

sys.path.append('../')
import pandas as pd
from dgl import load_graphs
from rdkit import Chem
import torch.nn as nn
from tqdm import tqdm
import scipy.sparse as sps
from DTA.CPI.utils import create_dataset_for_train, collate, target_matrics, get_mse, get_pearson, smile_to_graph
from src.data.collator import Collator_pharVQA_dta_CPI
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import pharVQADataset_dta_CPI
from src.model.dta_CPI_model import pharVQA_dta_CPI
from src.model_config import config_dict
from src.utils import LabelSmoothingLoss

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
import warnings

warnings.filterwarnings("ignore")

LR = 0.0001
NUM_EPOCHS = 5
seed = 0
dropout_global = 0.2
train_val_ratio = 0.9
early_stop = 20
stop_epoch = 0
best_epoch = -1
best_mse = 100
best_auc = 0.0
best_test_mse = 100
last_epoch = 1
phar_Dict = {'Donor': 0, 'Acceptor': 1, 'NegIonizable': 2, 'PosIonizable': 3, 'Aromatic': 4, 'Hydrophobe': 5,
             'LumpedHydrophobe': 6}


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--prompt_mode', type=str, default='cat')
    parser.add_argument('--use_encoder', type=str, default='KPGT')
    parser.add_argument('--load_model_path', type=str, default='../BindingDB_reg/ratio_0.8_bach_32_LR_0.0001_seed_0_tape_use_encoder_KPGT_use_atten_loss_True.pt')
    parser.add_argument('--use_atten_loss', type=str, default='True')

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, default='../pretrained/base/base.pth')
    parser.add_argument("--data_path", type=str, default='../datasets/DTA/')

    parser.add_argument('--num_questions', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00003, help='model learning rate for training (default: 0.00003)')

    parser.add_argument("--dataset", type=str,
                        choices=['toy_reg', 'BindingDB_cls', 'BindingDB_reg', 'JAK', 'AR', 'CYP_reg'],
                        default='BindingDB_reg')
    parser.add_argument("--protein_embedding", type=str, choices=['onehot', 'bio2vec', 'tape', 'esm2'], default='tape')
    parser.add_argument("--emb_size", type=int, choices=[20, 100, 768, 1280], default=768)
    args = parser.parse_args()
    return args


def load_dataset_dta_CPI(args, evaluate='False'):
    if evaluate == 'False':
        train_dataset_path = "../datasets/DTA/" + args.dataset + '/train/'
        smile_file_name = train_dataset_path + '/smile_graph'
        train_cache_path = os.path.join(train_dataset_path, f"{args.dataset}_5.pkl")

        # load all entries to generate train_val set
        raw_fold = eval(open(train_dataset_path + 'valid_entries.txt', 'r').read())
        np.random.seed(seed)
        random_entries = np.random.permutation(raw_fold)
        ptr = int(train_val_ratio * len(random_entries))
        train_val = [random_entries[:ptr], random_entries[ptr:]]

        ligands = json.load(open(train_dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
        proteins = json.load(open(train_dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(train_dataset_path + 'Y', 'rb'), encoding='latin1')

        ecfp_path = os.path.join(train_dataset_path, f"rdkfp1-7_512.npz")
        md_path = os.path.join(train_dataset_path, f"molecular_descriptors.npz")
        phar_path = os.path.join(train_dataset_path, f"phar_features.pkl")

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        compound_iso_smiles = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            compound_iso_smiles.append(lg)

        if os.path.exists(smile_file_name):
            print('load smile graph ...')
            smile_graph = pickle.load(open(smile_file_name, 'rb'))
        else:
            # create smile graph
            print('create smile_graph ...')
            smile_graph = {}
            for smile in tqdm(compound_iso_smiles):
                g = smile_to_graph(smile)
                smile_graph[smile] = g
            with open(smile_file_name, 'wb+') as f:
                pickle.dump(smile_graph, f)

        graphs, labels = load_graphs(train_cache_path)
        with open(phar_path, 'rb') as f:
            phars = pickle.load(f)

        # load seqs
        target_key = []
        target_graph = {}
        for key in proteins.keys():
            target_key.append(key)
            target_graph[key] = target_matrics(key, args.embedding_path)

        train_fold = train_val[0]
        valid_fold = train_val[1]

        rows, cols = np.where(np.isnan(affinity) == False)
        trows, tcols = rows[train_fold], cols[train_fold]
        train_fold_entries = {'compound_iso_smiles': [], 'target_key': [], 'affinity': [], 'graphs': [], 'fps': [],
                              'mds': [], 'phars': []}

        for pair_ind in range(len(trows)):
            train_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[trows[pair_ind]]]
            train_fold_entries['target_key'] += [target_key[tcols[pair_ind]]]
            train_fold_entries['affinity'] += [affinity[trows[pair_ind], tcols[pair_ind]]]
            train_fold_entries['graphs'] += [graphs[trows[pair_ind]]]
            train_fold_entries['fps'] += [fps[trows[pair_ind]]]
            train_fold_entries['mds'] += [mds[trows[pair_ind]]]
            train_fold_entries['phars'] += [phars[trows[pair_ind]]]

        trows, tcols = rows[valid_fold], cols[valid_fold]
        valid_fold_entries = {'compound_iso_smiles': [], 'target_key': [], 'affinity': [], 'graphs': [], 'fps': [],
                              'mds': [], 'phars': []}
        for pair_ind in range(len(trows)):
            valid_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[trows[pair_ind]]]
            valid_fold_entries['target_key'] += [target_key[tcols[pair_ind]]]
            valid_fold_entries['affinity'] += [affinity[trows[pair_ind], tcols[pair_ind]]]
            valid_fold_entries['graphs'] += [graphs[trows[pair_ind]]]

            valid_fold_entries['fps'] += [fps[trows[pair_ind]]]
            valid_fold_entries['mds'] += [mds[trows[pair_ind]]]
            valid_fold_entries['phars'] += [phars[trows[pair_ind]]]

        df_train_fold = pd.DataFrame(train_fold_entries)

        train_drugs, train_prot_keys, train_Y, train_graph, train_fps, train_mds, train_phars = \
            np.asarray(list(df_train_fold['compound_iso_smiles'])), np.asarray(list(df_train_fold['target_key'])), \
                np.asarray(list(df_train_fold['affinity'])), list(df_train_fold['graphs']), \
                list(df_train_fold['fps']), list(df_train_fold['mds']), list(df_train_fold['phars'])
        train_data = pharVQADataset_dta_CPI(root_path=args.data_path, dataset=args.dataset,
                                            smiles=train_drugs, proteins_keys=train_prot_keys, targets=train_Y,
                                            smile_graph=smile_graph, smile_line_graph=train_graph,
                                            protein_graph=target_graph,
                                            fps=train_fps, mds=train_mds, phars=train_phars,
                                            question_num=args.num_questions,
                                            device=args.device)

        df_valid_fold = pd.DataFrame(valid_fold_entries)

        valid_drugs, valid_prot_keys, valid_Y, valid_graph, valid_fps, valid_mds, valid_phars = \
            np.asarray(list(df_valid_fold['compound_iso_smiles'])), np.asarray(list(df_valid_fold['target_key'])), \
                np.asarray(list(df_valid_fold['affinity'])), list(df_valid_fold['graphs']), \
                list(df_valid_fold['fps']), list(df_valid_fold['mds']), list(df_valid_fold['phars'])

        valid_data = pharVQADataset_dta_CPI(root_path=args.data_path, dataset=args.dataset,
                                            smiles=valid_drugs, proteins_keys=valid_prot_keys, targets=valid_Y,
                                            smile_graph=smile_graph, smile_line_graph=valid_graph,
                                            protein_graph=target_graph,
                                            fps=valid_fps, mds=valid_mds, phars=valid_phars,
                                            question_num=args.num_questions,
                                            device=args.device)
        return train_data, valid_data, None
    else:
        # test
        test_dataset_path = "../datasets/DTA/" + args.dataset + '/ind_test/'
        smile_file_name = test_dataset_path + '/smile_graph'

        test_cache_path = os.path.join(test_dataset_path, f"{args.dataset}_5.pkl")

        test_fold = eval(open(test_dataset_path + 'valid_entries.txt', 'r').read())
        ligands = json.load(open(test_dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
        proteins = json.load(open(test_dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(test_dataset_path + 'Y', 'rb'), encoding='latin1')

        ecfp_path = os.path.join(test_dataset_path, f"rdkfp1-7_512.npz")
        md_path = os.path.join(test_dataset_path, f"molecular_descriptors.npz")
        phar_path = os.path.join(test_dataset_path, f"phar_features.pkl")

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        # load compounds graph
        compound_iso_smiles = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            compound_iso_smiles.append(lg)

        if os.path.exists(smile_file_name):
            print('load smile graph ...')
            smile_graph = pickle.load(open(smile_file_name, 'rb'))
        else:
            # create smile graph
            print('create smile_graph ...')
            smile_graph = {}
            for smile in tqdm(compound_iso_smiles):
                g = smile_to_graph(smile)
                smile_graph[smile] = g
            with open(smile_file_name, 'wb+') as f:
                pickle.dump(smile_graph, f)

        graphs, labels = load_graphs(test_cache_path)

        with open(phar_path, 'rb') as f:
            phars = pickle.load(f)

        # load seqs
        target_key = []
        target_graph = {}
        for key in proteins.keys():
            target_key.append(key)
            target_graph[key] = target_matrics(key, args.embedding_path)

        # load affinity matrix...
        print('load affinity matrix...')
        # pick out test entries
        rows, cols = np.where(np.isnan(affinity) == False)
        rows, cols = rows[test_fold], cols[test_fold]
        test_fold_entries = {'compound_iso_smiles': [], 'target_key': [], 'affinity': [], 'graphs': [], 'fps': [],
                             'mds': [], 'phars': []}
        for pair_ind in range(len(rows)):
            test_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[rows[pair_ind]]]
            test_fold_entries['target_key'] += [target_key[cols[pair_ind]]]
            test_fold_entries['affinity'] += [affinity[rows[pair_ind], cols[pair_ind]]]
            test_fold_entries['graphs'] += [graphs[rows[pair_ind]]]
            test_fold_entries['fps'] += [fps[rows[pair_ind]]]
            test_fold_entries['mds'] += [mds[rows[pair_ind]]]
            test_fold_entries['phars'] += [phars[rows[pair_ind]]]

        test_fold = pd.DataFrame(test_fold_entries)
        test_drugs, test_prot_keys, test_Y, test_graph, test_fps, test_mds, test_phars = \
            np.asarray(list(test_fold['compound_iso_smiles'])), np.asarray(list(test_fold['target_key'])), \
                np.asarray(list(test_fold['affinity'])), list(test_fold['graphs']), \
                list(test_fold['fps']), list(test_fold['mds']), list(test_fold['phars'])

        test_data = pharVQADataset_dta_CPI(root_path=args.data_path, dataset=args.dataset,
                                           smiles=test_drugs, proteins_keys=test_prot_keys, targets=test_Y,
                                           smile_graph=smile_graph, smile_line_graph=test_graph,
                                           protein_graph=target_graph,
                                           fps=test_fps, mds=test_mds, phars=test_phars,
                                           question_num=args.num_questions,
                                           device=args.device)

        return None, None, test_data


def test(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            (input, data_pro, data_pro_len, labels, smiles_graph, g, ecfp, md, text, text_mask, phar_targets,
             atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions) = data
            data_pro, data_pro_len, y_label, ecfp, md, g, smiles_graph, text, text_mask, phar_targets = data_pro.to(
                device), data_pro_len.to(device), labels.to(device), ecfp.to(device), md.to(device), g.to(
                device), smiles_graph.to(device), text.to(device), text_mask.to(device), phar_targets.to(device)
            output, pred_phar_num, atten = model(ecfp, md, g, smiles_graph, text, text_mask, data_pro, data_pro_len)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, y_label.view(output.shape).to(torch.float64).cpu()), 0)
            # sys.stdout.write(str(format(100. *batch_idx / len(loader),'.2f'))+"%\r")

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def check_update(data_type, predict_score, best_score):
    if data_type == 'regression':
        if predict_score <= best_score:
            return True
        else:
            return False
    elif data_type == 'classification':
        if predict_score >= best_score:
            return True
        else:
            return False


def get_score(dataset_type, T, P):
    if dataset_type == 'classification':
        AUC = roc_auc_score(T, P)
        tpr, fpr, _ = precision_recall_curve(T, P)
        AUPR = auc(fpr, tpr)
        return AUC, AUPR
    else:
        mse = get_mse(T, P)
        pearson = get_pearson(T, P)
        r2 = r2_score(T, P)
        return mse, pearson, r2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(args):
    dataset = args.dataset
    TRAIN_BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.batch_size

    if dataset == 'BindingDB_reg':
        args.dataset_type = 'regression'
    elif dataset == 'BindingDB_cls':
        args.dataset_type = 'classification'

    # choose embedding
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    protein_embedding = args.protein_embedding
    emb_size = args.emb_size

    # parmeter = 'ratio_' + str(train_val_ratio) + 'bach' + str(
    #     TRAIN_BATCH_SIZE) + 'LR_1e-4' + 'random_0_' + protein_embedding
    # Path=os.path.abspath(os.path.join(os.getcwd(),"../.."))
    # parmeter = 'ratio_' + str(train_val_ratio) + 'bach' + str(
    #     TRAIN_BATCH_SIZE) + 'LR_1e-4' + 'random_0_' + protein_embedding + 'use_encoder' + args.use_encoder + 'use_atten_loss' + args.use_atten_loss
    # parmeter = f'ratio_{args.train_val_ratio}_' + f'bach_{TRAIN_BATCH_SIZE}_' + f'LR_{LR}_' \
    #            + f'seed_{args.seed}_' + protein_embedding + f'_use_encoder_{args.use_encoder}_' + \
    #            f'use_atten_loss_{args.use_atten_loss}'

    model_file_dir = dataset + '/'
    args.embedding_path = "../datasets/DTA/%s/%s/" % (dataset, protein_embedding)
    # set protein length capacity, which default=1500(according to BindingDB regression dataset)
    max_length = max(1500, int(open("../datasets/DTA/" + dataset + '/max_length.txt', 'r').read()))
    # model_name = model_file_dir + parmeter + '.pt'
    log_dir = model_file_dir + 'logs/'
    if not os.path.exists(model_file_dir):
        os.makedirs(model_file_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # load data
    config = config_dict[args.config]
    collator = Collator_pharVQA_dta_CPI(config['path_length'])
    _, _, test_data = load_dataset_dta_CPI(args, evaluate="True")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collator)
    # instantiate a model
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = pharVQA_dta_CPI(args, device, config, test_data.d_fps, test_data.d_mds, vocab,
                            emb_size=emb_size, max_length=max_length, dropout=dropout_global)

    model.load_state_dict(torch.load(args.load_model_path, map_location='cpu'),strict=False)
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))
    model.to(device)

    T, P = test(model, device, test_loader)
    score = get_score(args.dataset_type, T, P)
    if args.dataset_type == 'classification':
        print(dataset, f' \t auc score :', score[0], 'aupr score', score[1])
    elif args.dataset_type == 'regression':
        print(dataset, f' \t mse score :', score[0], 'pearson score', score[1], 'r2 score', score[2])


if __name__ == '__main__':
    args = parse_args()
    main(args)
