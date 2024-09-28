import argparse
import json
import pickle
import sys
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, r2_score
from torch.nn import BCEWithLogitsLoss

sys.path.append('../')
import pandas as pd
from dgl import load_graphs
from rdkit import Chem
import torch.nn as nn
from tqdm import tqdm
import scipy.sparse as sps
from DTA.CPI.utils import create_dataset_for_train, collate, target_matrics, get_mse, smile_to_graph, get_pearson
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

dropout_global = 0.2
early_stop = 20
stop_epoch = 0
best_epoch = -1
best_loss = 1000
best_mse = 100
best_auc = 0.0
best_test_mse = 100
last_epoch = 1
phar_Dict = {'Donor': 0, 'Acceptor': 1, 'NegIonizable': 2, 'PosIonizable': 3, 'Aromatic': 4, 'Hydrophobe': 5,
             'LumpedHydrophobe': 6}


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--device", type=str, default='cuda:3')
    parser.add_argument('--use_encoder', type=str, default='KPGT')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--use_atten_loss', type=str, default='True')
    parser.add_argument('--train_val_ratio', type=float, default=0.9)

    parser.add_argument('--prompt_mode', type=str, default='cat')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--test_per_step", type=int, default=5)

    parser.add_argument("--save_path", type=str, default='../save/')

    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, default='../pretrained/base/base.pth')
    parser.add_argument("--data_path", type=str, default='../datasets/DTA/')

    parser.add_argument('--num_questions', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.0001, help='model learning rate for training (default: 0.00003)')

    parser.add_argument("--dataset", type=str,
                        choices=['toy_reg', 'BindingDB_cls', 'BindingDB_reg', 'JAK', 'AR', 'CYP_reg'],
                        default='BindingDB_cls')
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
        np.random.seed(args.seed)
        random_entries = np.random.permutation(raw_fold)
        ptr = int(args.train_val_ratio * len(random_entries))
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


def train(model, device, train_loader, optimizer, loss_fn, epoch, writer):
    model.train()
    align_loss = 0
    train_loss = []
    phar_loss_fn = nn.MSELoss(reduction="none")
    align_loss_fn = LabelSmoothingLoss(reduction='mean', smoothing=0.5)
    for batch_idx, data in enumerate(tqdm(train_loader)):
        (input, data_pro, data_pro_len, labels, smiles_graph, g, ecfp, md, text, text_mask, phar_targets,
         atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions) = data
        data_pro, data_pro_len, y_label, ecfp, md, g, smiles_graph, text, text_mask, phar_targets = data_pro.to(
            device), data_pro_len.to(device), labels.to(device), ecfp.to(device), md.to(device), g.to(
            device), smiles_graph.to(device), text.to(device), text_mask.to(device), phar_targets.to(device)
        batch_size = len(input)
        optimizer.zero_grad()

        output, pred_phar_num, atten = model(ecfp, md, g, smiles_graph, text, text_mask, data_pro, data_pro_len)

        phar_y = phar_targets.to(torch.float64)

        # Loss matrix
        loss_mat = loss_fn(output, y_label.float())
        phar_loss = phar_loss_fn(pred_phar_num, phar_y).mean()

        # atten loss Optinal
        if args.use_atten_loss == 'True':
            phar_target_mx = torch.zeros((batch_size, atten[0].size()[-2], len(atten))).to(device)
            mols = [Chem.MolFromSmiles(s) for s in input]
            for mol_index, mol in enumerate(mols):
                random_question = random_questions[mol_index]
                mol_phar_mx = atom_phar_target_map[mol_index]
                for bond_index, bond in enumerate(mol.GetBonds()):
                    begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    phar_target_mx[mol_index, bond_index] = mol_phar_mx[
                        [begin_atom_id, end_atom_id], phar_Dict[random_question[0]]].sum(dim=0).to(
                        torch.bool).to(torch.float32)
            atten = torch.stack(atten, dim=2).sum(dim=-1)

            align_loss = align_loss_fn(atten.view(atten.size()[0], -1),
                                       phar_target_mx.view(phar_target_mx.size()[0], -1)).to(torch.float64)

        loss = loss_mat + align_loss + phar_loss

        loss.backward()
        optimizer.step()

        writer.add_scalar('Train/loss_mat', loss_mat, (epoch) * len(train_loader) + batch_idx + 1)
        writer.add_scalar('Train/align_loss', align_loss, (epoch) * len(train_loader) + batch_idx + 1)
        writer.add_scalar('Train/phar_loss', phar_loss, (epoch) * len(train_loader) + batch_idx + 1)
        writer.add_scalar('Train/loss_total', loss, (epoch) * len(train_loader) + batch_idx + 1)

    train_loss.append(loss.item())
    train_loss = np.average(train_loss)
    writer.add_scalar('Train/Loss', train_loss, epoch)
    return train_loss


def evaluate(model, device, loader, loss_fn):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    eval_loss = []
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

            # Loss matrix
            loss_mat = loss_fn(output, y_label)
        eval_loss.append(loss_mat.item())
    eval_loss = np.average(eval_loss)
    return eval_loss, total_labels.numpy().flatten(), total_preds.numpy().flatten()


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
    if data_type == 'regression' or data_type == 'loss':
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
    LR = args.lr
    NUM_EPOCHS = args.n_epochs

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

    parmeter = f'ratio_{args.train_val_ratio}_' + f'bach_{TRAIN_BATCH_SIZE}_' + f'LR_{LR}_' \
               + f'seed_{args.seed}_' + protein_embedding + f'_use_encoder_{args.use_encoder}_' + \
               f'use_atten_loss_{args.use_atten_loss}'

    model_file_dir = dataset + '/'
    args.embedding_path = "../datasets/DTA/%s/%s/" % (dataset, protein_embedding)

    # set protein length capacity, which default=1500(according to BindingDB regression dataset)
    max_length = max(1500, int(open("../datasets/DTA/" + dataset + '/max_length.txt', 'r').read()))
    model_name = model_file_dir + parmeter + '.pt'
    log_dir = model_file_dir + f'logs/{parmeter}/'
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=parmeter)
    if not os.path.exists(model_file_dir):
        os.makedirs(model_file_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # load data
    config = config_dict[args.config]
    collator = Collator_pharVQA_dta_CPI(config['path_length'])
    train_data, valid_data, _ = load_dataset_dta_CPI(args)
    _, _, test_data = load_dataset_dta_CPI(args, evaluate="True")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collator, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False,
                                               collate_fn=collator)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collator)
    # instantiate a model
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = pharVQA_dta_CPI(args, device, config, train_data.d_fps, train_data.d_mds, vocab,
                            emb_size=emb_size, max_length=max_length, dropout=dropout_global)
    if args.use_encoder == 'KPGT':
        model.embedding_model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}', map_location='cpu').items()},
            strict=False)
        for name, param in model.embedding_model.model.state_dict().items():
            model.embedding_model.graph_model_feat.state_dict()[name].copy_(param)

        del model.embedding_model.md_predictor
        del model.embedding_model.fp_predictor
        del model.embedding_model.node_predictor
        model.embedding_model.text_model.requires_grad_(False)
        model.embedding_model.model.requires_grad_(False)

    elif args.use_encoder == 'SAGE':
        model.text_model.requires_grad_(False)

    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print('load model success')

    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    # train
    if args.dataset_type == 'classification':
        best_score = best_auc
        loss_fn = BCEWithLogitsLoss()
    else:
        best_score = best_mse
        loss_fn = torch.nn.MSELoss()

    stop_epoch = 0
    best_epoch = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, device, train_loader, optimizer, loss_fn, epoch + 1, writer)
        val_loss, T, P = evaluate(model, device, valid_loader, loss_fn)
        valid_score = get_score(args.dataset_type, T, P)

        if args.dataset_type == 'classification':
            print(dataset, f'Valid at {epoch + 1} step \t auc score :', valid_score[0], 'aupr score', valid_score[1])
            writer.add_scalar('Valid/auc', valid_score[0], epoch)
            writer.add_scalar('Valid/aupr', valid_score[1], epoch)
        elif args.dataset_type == 'regression':
            print(dataset, f'Valid at {epoch + 1} step \t mse score :', valid_score[0], 'pearson score', valid_score[1],
                  'r2 score', valid_score[2])
            writer.add_scalar('Valid/mse', valid_score[0], epoch)
            writer.add_scalar('Valid/pearson', valid_score[1], epoch)
            writer.add_scalar('Valid/r2', valid_score[2], epoch)

        writer.add_scalar('Valid/Loss', val_loss, epoch)
        print('epoch\t', epoch + 1, 'train_loss\t', train_loss, 'val_loss\t', val_loss)

        stop_epoch += 1
        if check_update(args.dataset_type, valid_score[0], best_score):
            best_score = valid_score[0]
            stop_epoch = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_name)

            T, P = test(model, device, test_loader)
            test_score = get_score(args.dataset_type, T, P)
            if args.dataset_type == 'classification':
                predict = sigmoid(P)
                print(dataset, f'at {epoch + 1} step \t auc score :', test_score[0], 'aupr score', test_score[1])
                writer.add_scalar('Test/auc', test_score[0], epoch)
                writer.add_scalar('Test/aupr', test_score[1], epoch)
                np.savetxt(f'{log_dir}/Results_epoch_{epoch}.txt',
                           np.concatenate(
                               (T.reshape(-1, 1), predict.reshape(-1, 1), (predict >= 0.5).astype(int).reshape(-1, 1)),
                               axis=1), fmt='%.4f')
            elif args.dataset_type == 'regression':
                print(dataset, f'at {epoch + 1} step \t mse score :', test_score[0], 'pearson score', test_score[1],
                      'r2 score', test_score[2])
                writer.add_scalar('Test/mse', test_score[0], epoch)
                writer.add_scalar('Test/pearson', test_score[1], epoch)
                writer.add_scalar('Test/r2', test_score[2], epoch)
                np.savetxt(f'{log_dir}/Results_epoch_{epoch}.txt',
                           np.concatenate((T.reshape(-1, 1), P.reshape(-1, 1)),
                                          axis=1), fmt='%.4f')
        if stop_epoch == early_stop:
            print('(EARLY STOP) No improvement since epoch ', best_epoch)
            break
        scheduler.step(val_loss)
    if args.dataset_type == 'classification':
        print(
            'Best epoch %s; best_test_auc_score%s; best_test_aupr_score%s; dataset:%s; train ratio:%s' % (
                best_epoch, test_score[0], test_score[1], dataset, args.train_val_ratio))
    elif args.dataset_type == 'regression':
        print(
            'Best epoch %s; best_test_mse_score%s; best_test_pearson_score%s; dataset:%s; train ratio:%s' % (
                best_epoch, test_score[0], test_score[1], dataset, args.train_val_ratio))


if __name__ == '__main__':
    args = parse_args()
    main(args)
