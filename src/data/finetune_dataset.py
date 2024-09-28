import itertools
import json
import pickle
import random
import sqlite3
import sys
from typing import List, OrderedDict

import dgl
import lmdb
from ogb.utils import smiles2graph
from pysmilesutils.augment import MolAugmenter
from rdkit import Chem
# from rdkit.Chem.Crippen import MolLogP
# from rdkit.Chem.QED import qed
# from rdkit.Contrib.SA_Score import sascorer
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from dgl.data.utils import load_graphs
import torch
import dgl.backend as F
import scipy.sparse as sps
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric import data as DATA
from retroCap.main_align_rerank_smi import get_retro_rsmiles, clear_map_canonical_smiles
from retroCap.utils.graph_utils import init_positional_encoding
from retroCap.utils.smiles_utils import smi_tokenizer, clear_map_number, canonical_smiles_with_am, \
    remove_am_without_canonical, randomize_smiles_with_am, get_nonreactive_mask, extract_relative_mapping, get_3d_adj

SPLIT_TO_ID = {'train': 0, 'val': 1, 'test': 2}
phar_Dict = {'Donor': 0, 'Acceptor': 1, 'NegIonizable': 2, 'PosIonizable': 3, 'Aromatic': 4, 'Hydrophobe': 5,
             'LumpedHydrophobe': 6}


def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=value)


def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')

    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


def prepare_text_tokens(description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(np.array(tokens_ids)).long()
    masks = torch.Tensor(np.array(masks)).bool()
    return tokens_ids, masks


class MoleculeDataset(Dataset):
    def __init__(self, root_path, dataset, dataset_type, path_length=5, n_virtual_nodes=2, split_name=None, split=None,
                 question_num=None):

        if dataset in ['FGFR1', 'HPK1', 'PTP1B', 'PTPN2']:
            dataset_path = os.path.join(root_path, f"{dataset}/{dataset}_IC50.csv")
        else:
            dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")

        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")

        ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        # Load Data
        df = pd.read_csv(dataset_path)
        df.dropna(how='all', inplace=True)

        if split is not None:
            if dataset in ['muv', 'hiv', 'FGFR1', 'HPK1', 'PTP1B', 'PTPN2']:
                split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.json")
                with open(split_path, 'r', encoding='utf-8') as fp:
                    use_idxs = json.load(fp)[SPLIT_TO_ID[split]]
            else:
                split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.npy")
                use_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID[split]]
        else:
            use_idxs = np.arange(0, len(df))

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        self.df, self.fps, self.mds = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs]

        try:
            self.smiless = self.df['smiles'].tolist()
            self.task_names = self.df.columns.drop(['smiles']).tolist()
        except:
            self.smiless = self.df['Smiles'].tolist()
            self.task_names = self.df.columns.drop(['Smiles']).tolist()

        self.use_idxs = use_idxs
        # Dataset Setting
        self.n_tasks = len(self.task_names)
        self._pre_process()
        self.mean = None
        self.std = None
        if dataset_type == 'classification':
            self._task_pos_weights = self.task_pos_weights()
        elif dataset_type == 'regression':
            self.set_mean_and_std()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
        self.fps, self.mds = self.fps, self.mds

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx], self.labels[idx]

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std


class pharVQADataset(Dataset):
    def __init__(self, root_path, dataset, dataset_type, path_length=5, n_virtual_nodes=2, split_name=None, split=None,noise_question='False',
                 question_num=1, question_name=None, text_max_len=256, device=None):
        self.dataset = dataset
        if dataset in ['FGFR1', 'HPK1', 'PTP1B', 'PTPN2', 'VIM1']:
            dataset_path = os.path.join(root_path, f"{dataset}/{dataset}_IC50.csv")
        elif dataset in ['DrugBank']:
            dataset_path = os.path.join(root_path, f"{dataset}/structure_links.csv")
        else:
            dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")
        ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        phar_path = os.path.join(root_path, f"{dataset}/phar_features.pkl")
        # Load Data
        df = pd.read_csv(dataset_path)
        df.dropna(how='all', inplace=True)

        if split is not None:
            if dataset in ['muv', 'hiv', 'FGFR1', 'HPK1', 'PTP1B', 'PTPN2', 'VIM1', 'HPK1_KI']:
                split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.json")
                with open(split_path, 'r', encoding='utf-8') as fp:
                    use_idxs = json.load(fp)[SPLIT_TO_ID[split]]
            else:
                split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.npy")
                use_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID[split]]
        else:
            use_idxs = np.arange(0, len(df))

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        with open(phar_path, 'rb') as f:
            phars = pickle.load(f)

        # phar task Setting
        self.noise_question = noise_question
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')

        text_path = os.path.join(root_path, '../text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)

        self.df, self.fps, self.mds, self.phars = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs], [phars[i] for i in
                                                                                                    use_idxs]
        try:
            self.smiless = self.df['smiles'].tolist()
            self.task_names = self.df.columns.drop(['smiles']).tolist()
        except:
            self.smiless = self.df['Smiles'].tolist()
            self.task_names = self.df.columns.drop(['Smiles']).tolist()

        self.use_idxs = use_idxs
        # Dataset Setting

        self.n_tasks = len(self.task_names)
        self._pre_process()
        self.mean = None
        self.std = None
        if dataset_type == 'classification':
            self._task_pos_weights = self.task_pos_weights()
        elif dataset_type == 'regression':
            self.set_mean_and_std()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

        self.device = device

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
        self.fps, self.mds = self.fps, self.mds

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        if self.noise_question == 'True':
            random_questions = None
            select_question = ["To be, or not to be, that is the question."]
        else:
            if self.sample_num == len(list(self.text_list.keys())):
                random_questions = list(self.text_list.keys())
            elif self.sample_num == 1:
                random_questions = list([self.question_name])
            else:
                random_questions = self.question_name
            select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question
                           in random_questions]
        # prop_text = [self.text_prop_list['logP']['1'], self.text_prop_list['QED']['1']]
        # text = prop_text + select_question
        text = select_question
        texts, masks = self.tokenizer_text(text)
        # count phar group
        v_phar_name, v_phar_atom_id, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars[idx]:
            v_phar_name.append(feat[0].split(".")[0])
            v_phar_atom_id.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        if self.noise_question == 'True':
            attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
        else:
            attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
            for phar_index, phar_task in enumerate(random_questions):
                attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name]] = 1

        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol = Chem.MolFromSmiles(self.smiless[idx])
        num_nodes = mol.GetNumAtoms()
        atom_phar_target_map = torch.zeros(num_nodes, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name, v_phar_atom_id):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map[y, phar_index] = 1

        # logP = MolLogP(mol)
        # QED = qed(mol)
        # SA = sascorer.calculateScore(mol)

        # phar_targets = torch.tensor([logP, QED, phar_targets_num])
        phar_targets = phar_targets_num
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[
            idx], v_phar_name, v_phar_atom_id, texts, masks, phar_targets, self.labels[
            idx], atom_phar_target_map, random_questions

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

    # def tokenizer_text(self, text):
    #     tokenizer = self.tokenizer
    #     sentence_token = tokenizer(text=text,
    #                                truncation=True,
    #                                padding='max_length',
    #                                add_special_tokens=False,
    #                                max_length=self.text_max_len,
    #                                return_tensors='pt',
    #                                return_attention_mask=True)
    #     input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
    #     attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
    #     return input_ids, attention_mask
    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


class pharVQADataset_eval(Dataset):
    def __init__(self, root_path, dataset, dataset_type, path_length=5, n_virtual_nodes=2, split_name=None, split=None,
                 question_num=1, question_name=None, text_max_len=256, device=None):
        self.dataset = dataset
        if dataset in ['FGFR1', 'HPK1', 'PTP1B', 'PTPN2', 'VIM1']:
            dataset_path = os.path.join(root_path, f"{dataset}/{dataset}_IC50.csv")
        elif dataset in ['DrugBank']:
            dataset_path = os.path.join(root_path, f"{dataset}/structure_links.csv")
        else:
            dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")
        ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        phar_path = os.path.join(root_path, f"{dataset}/phar_features.pkl")
        # Load Data
        df = pd.read_csv(dataset_path)
        df.dropna(how='all', inplace=True)

        # self.smiless = df['SMILES'].dropna().tolist()
        try:
            self.smiless = df['SMILES'].dropna().tolist()
            self.task_names = df.columns.drop(['SMILES']).tolist()
        except:
            self.smiless = df['smiles'].dropna().tolist()
            self.task_names = df.columns.drop(['smiles']).tolist()
        self.smiless = check_valid_single(self.smiless)
        if split is not None:
            if dataset in ['muv', 'hiv', 'FGFR1', 'HPK1', 'PTP1B', 'PTPN2', 'VIM1']:
                split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.json")
                with open(split_path, 'r', encoding='utf-8') as fp:
                    use_idxs = json.load(fp)[SPLIT_TO_ID[split]]
            else:
                split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.npy")
                use_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID[split]]
        else:
            use_idxs = np.arange(0, len(self.smiless))

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        with open(phar_path, 'rb') as f:
            phars = pickle.load(f)

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')

        text_path = os.path.join(root_path, '../text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)

        self.df, self.fps, self.mds, self.phars = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs], [phars[i] for i in
                                                                                                    use_idxs]

        self.use_idxs = use_idxs
        # Dataset Setting
        self._pre_process()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

        self.device = device

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
        self.fps, self.mds = self.fps, self.mds

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        try:
            if self.sample_num == len(list(self.text_list.keys())):
                random_questions = list(self.text_list.keys())
            elif self.sample_num == 1:
                random_questions = list([self.question_name])
            else:
                random_questions = self.question_name
            select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question
                               in
                               random_questions]
            # prop_text = [self.text_prop_list['logP']['1'], self.text_prop_list['QED']['1']]
            # text = prop_text + select_question
            text = select_question
            texts, masks = self.tokenizer_text(text)
            # count phar group
            v_phar_name, v_phar_atom_id, v_phar_phar_pos, v_phar_feature = [], [], [], []
            for feat in self.phars[idx]:
                v_phar_name.append(feat[0].split(".")[0])
                v_phar_atom_id.append(feat[1])
                v_phar_phar_pos.append(feat[2])

            attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
            for phar_index, phar_task in enumerate(random_questions):
                attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name]] = 1
            phar_targets_num = attn_phar_atom.sum(dim=1).long()

            mol = Chem.MolFromSmiles(self.smiless[idx])
            num_nodes = mol.GetNumAtoms()
            atom_phar_target_map = torch.zeros(num_nodes, len(list(self.text_list.keys())))
            for x, y in zip(v_phar_name, v_phar_atom_id):
                if x not in list(self.text_list.keys()):
                    continue
                phar_index = list(self.text_list.keys()).index(x)
                atom_phar_target_map[y, phar_index] = 1

            # logP = MolLogP(mol)
            # QED = qed(mol)
            # SA = sascorer.calculateScore(mol)

            # phar_targets = torch.tensor([logP, QED, phar_targets_num])
            phar_targets = phar_targets_num
            return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[
                idx], v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions
        except:
            print('get item error')

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


def check_valid(reactant, product):
    valid_index = []
    for i, (r, p) in enumerate(zip(reactant, product)):
        if Chem.MolFromSmiles(r) is not None and Chem.MolFromSmiles(p) is not None:
            valid_index.append(i)
    reactant = [reactant[i] for i in valid_index]
    product = [product[i] for i in valid_index]
    return reactant, product


def check_valid_single(smiles):
    valid_index = []
    for i, p in enumerate(smiles):
        if Chem.MolFromSmiles(p) is not None:
            valid_index.append(i)
    product = [smiles[i] for i in valid_index]
    return product


class pharVQADataset_rxn(Dataset):
    def __init__(self, root_path, dataset_type, path_length=5, split_name=None, split=None,
                 question_num=1, question_name=None, text_max_len=256, mode='forward', device=None, aug=False):
        self.mode = mode
        self.is_aug = aug
        self.aug = MolAugmenter()
        if mode == 'forward':
            dataset_path = os.path.join(root_path, f"USPTO-480k/{dataset_type}_parsed.txt")
            self.cache_path = os.path.join(root_path,
                                           f"USPTO-480k/{dataset_type}_parsed/{dataset_type}_parsed_{path_length}_r.pkl")

            ecfp_path = os.path.join(root_path, f"USPTO-480k/{dataset_type}_parsed/rdkfp1-7_512_r.npz")
            md_path = os.path.join(root_path, f"USPTO-480k/{dataset_type}_parsed/molecular_descriptors_r.npz")
            phar_path = os.path.join(root_path, f"USPTO-480k/{dataset_type}_parsed/phar_features_r.pkl")

            df = pd.read_csv(dataset_path, header=None, sep='\t')
            df = df.dropna(axis=0, how='all')
            reactant = df[0].values.tolist()
            product = df[1].values.tolist()

            reactant, product = check_valid(reactant, product)
            self.smiless = reactant

        elif mode == 'retro':
            self.cache_path = os.path.join(root_path,
                                           f"USPTO-50k/{dataset_type}/{dataset_type}_{path_length}_p.pkl")

            with open(f'{root_path}/USPTO-50k/uspto_50.pickle', 'rb') as f:
                data = pickle.load(f)
            data = [data.iloc[i] for i in range(len(data))]
            data = [d for d in data if d['set'] == dataset_type]
            reactant = [r['reactants_mol'] for r in data]
            product = [p['products_mol'] for p in data]

            ecfp_path = os.path.join(root_path, f"USPTO-50k/{dataset_type}/rdkfp1-7_512_p.npz")
            md_path = os.path.join(root_path, f"USPTO-50k/{dataset_type}/molecular_descriptors_p.npz")
            phar_path = os.path.join(root_path, f"USPTO-50k/{dataset_type}/phar_features_p.pkl")

            reactant = [Chem.MolToSmiles(r, isomericSmiles=False) for r in reactant]
            product = [Chem.MolToSmiles(p, isomericSmiles=False) for p in product]

            self.smiless = product

        self.r_smiless = reactant
        self.p_smiless = product

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        with open(phar_path, 'rb') as f:
            phars = pickle.load(f)

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased', local_files_only=True)

        text_path = os.path.join(root_path, 'text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)

        self.fps, self.mds, self.phars = fps, mds, phars

        # Dataset Setting
        self._pre_process()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

        self.device = device

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = graphs
        self.fps, self.mds = self.fps, self.mds

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        elif self.sample_num == 1:
            random_questions = list([self.question_name])
        else:
            random_questions = self.question_name
        select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question in
                           random_questions]
        text = select_question
        texts, masks = self.tokenizer_text(text)
        # count phar group
        v_phar_name, v_phar_atom_id, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars[idx]:
            v_phar_name.append(feat[0].split(".")[0])
            v_phar_atom_id.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol = Chem.MolFromSmiles(self.smiless[idx])
        num_nodes = mol.GetNumAtoms()
        atom_phar_target_map = torch.zeros(num_nodes, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name, v_phar_atom_id):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map[y, phar_index] = 1

        # phar_targets = torch.tensor([logP, QED, phar_targets_num])
        # get_retro_rsmiles(reactant, product, augmentation)
        phar_targets = phar_targets_num
        if self.is_aug and random.random() > 0.5:
            r_mol = self.aug([Chem.MolFromSmiles(self.r_smiless[idx])])[0]
            r_smiles = Chem.MolToSmiles(r_mol, canonical=False, isomericSmiles=False)
            p_mol = self.aug([Chem.MolFromSmiles(self.p_smiless[idx])])[0]
            p_smiles = Chem.MolToSmiles(p_mol, canonical=False, isomericSmiles=False)
        else:
            r_smiles = self.r_smiless[idx]
            p_smiles = self.p_smiless[idx]

        if self.mode == 'forward':
            return '[CLS]' + r_smiles, '[CLS]' + p_smiles, self.graphs[idx], self.fps[idx], \
                self.mds[idx], v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, \
                random_questions
        elif self.mode == 'retro':
            return '[CLS]' + p_smiles, '[CLS]' + r_smiles, self.graphs[idx], self.fps[idx], \
                self.mds[idx], v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, \
                random_questions

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


class pharVQADataset_yeild(Dataset):
    def __init__(self, root_path, data_name, reactant_smiles, product_smiles, label, data_idx, path_length=5,
                 question_num=1, question_name=None, text_max_len=256, device=None):
        self.use_idxs = data_idx
        self.cache_path_r = os.path.join(f"{root_path}/{data_name}/", f"{data_name}_{path_length}_r.pkl")
        self.cache_path_p = os.path.join(f"{root_path}/{data_name}/", f"{data_name}_{path_length}_p.pkl")
        ecfp_path_r = os.path.join(f"{root_path}/{data_name}/", f"rdkfp1-7_512_r.npz")
        md_path_r = os.path.join(f"{root_path}/{data_name}/", f"molecular_descriptors_r.npz")
        phar_path_r = os.path.join(f"{root_path}/{data_name}/", f"phar_features_r.pkl")
        ecfp_path_p = os.path.join(f"{root_path}/{data_name}/", f"rdkfp1-7_512_p.npz")
        md_path_p = os.path.join(f"{root_path}/{data_name}/", f"molecular_descriptors_p.npz")
        phar_path_p = os.path.join(f"{root_path}/{data_name}/", f"phar_features_p.pkl")

        fps_r = torch.from_numpy(sps.load_npz(ecfp_path_r).todense().astype(np.float32))
        mds_r = np.load(md_path_r)['md'].astype(np.float32)
        mds_r = torch.from_numpy(np.where(np.isnan(mds_r), 0, mds_r))

        fps_p = torch.from_numpy(sps.load_npz(ecfp_path_p).todense().astype(np.float32))
        mds_p = np.load(md_path_p)['md'].astype(np.float32)
        mds_p = torch.from_numpy(np.where(np.isnan(mds_p), 0, mds_p))

        with open(phar_path_r, 'rb') as f:
            phars_r = pickle.load(f)
        with open(phar_path_p, 'rb') as f:
            phars_p = pickle.load(f)

        self.smiless_r = [reactant_smiles[i] for i in data_idx]
        self.smiless_p = [product_smiles[i] for i in data_idx]
        data_idx = list(data_idx)
        self.fps_r, self.mds_r, self.phars_r = fps_r[data_idx], mds_r[data_idx], [phars_r[i] for i in data_idx]
        self.fps_p, self.mds_p, self.phars_p = fps_p[data_idx], mds_p[data_idx], [phars_p[i] for i in data_idx]

        # Dataset Setting
        self._pre_process()
        self.labels = torch.tensor(label).to(torch.float32)
        self.d_fps = self.fps_r.shape[1]
        self.d_mds = self.mds_r.shape[1]

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased', local_files_only=True)

        text_path = os.path.join(root_path, '../text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)
        self.set_mean_and_std()
        self.device = device

    def _pre_process(self):
        if not os.path.exists(self.cache_path_r) or not os.path.exists(self.cache_path_p):
            print(f"{self.cache_path_r} or {self.cache_path_p} not exists, please run preprocess.py")
        else:
            graphs_r, label_dict = load_graphs(self.cache_path_r)
            graphs_p, label_dict = load_graphs(self.cache_path_p)
            self.graphs_r = []
            self.graphs_p = []
            for i in self.use_idxs:
                self.graphs_r.append(graphs_r[i])
                self.graphs_p.append(graphs_p[i])
            # self.labels = label_dict['labels']

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = self.labels.mean()
        if std is None:
            std = self.labels.std()
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.smiless_r)

    def __getitem__(self, idx):
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        elif self.sample_num == 1:
            random_questions = list([self.question_name])
        else:
            random_questions = self.question_name
        select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question in
                           random_questions]
        text = select_question
        texts, masks = self.tokenizer_text(text)
        # count phar 1 group
        v_phar_name_r, v_phar_atom_id_r, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars_r[idx]:
            v_phar_name_r.append(feat[0].split(".")[0])
            v_phar_atom_id_r.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name_r))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name_r]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol_r = Chem.MolFromSmiles(self.smiless_r[idx])
        num_nodes_r = mol_r.GetNumAtoms()
        atom_phar_target_map_r = torch.zeros(num_nodes_r, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name_r, v_phar_atom_id_r):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map_r[y, phar_index] = 1
        phar_targets_r = phar_targets_num

        # count phar 2 group
        v_phar_name_p, v_phar_atom_id_p, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars_p[idx]:
            v_phar_name_p.append(feat[0].split(".")[0])
            v_phar_atom_id_p.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name_p))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name_p]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol_p = Chem.MolFromSmiles(self.smiless_p[idx])
        num_nodes_p = mol_p.GetNumAtoms()
        atom_phar_target_map_p = torch.zeros(num_nodes_p, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name_p, v_phar_atom_id_p):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map_p[y, phar_index] = 1
        phar_targets_p = phar_targets_num

        return self.smiless_r[idx], self.graphs_r[idx], self.fps_r[idx], self.mds_r[
            idx], v_phar_name_r, v_phar_atom_id_r, phar_targets_r, atom_phar_target_map_r, \
            self.smiless_p[idx], self.graphs_p[idx], self.fps_p[idx], self.mds_p[
            idx], v_phar_name_p, v_phar_atom_id_p, phar_targets_p, atom_phar_target_map_p, \
            self.labels[idx], texts, masks, random_questions

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


class pharVQADataset_dta(Dataset):
    def __init__(self, root_path, dataset, path_length=5, use_idxs=[], mode='train',
                 question_num=1, question_name=None, text_max_len=256, device=None):

        # load compound and protein data
        data_index_path = f"{root_path}/{dataset}/{mode}.csv"
        input_idx = pd.read_csv(data_index_path)

        smiles_idx = input_idx['smiles_id'].values
        target_idx = input_idx['target_id'].values
        labels = input_idx['affinity'].values

        smiles_path = f"{root_path}/{dataset}/smiles.csv"
        # protein_path = f"{root_path}/{dataset}/protein.csv"

        self.smiless = pd.read_csv(smiles_path)['smiles'].values[smiles_idx]
        # self.proteins = pd.read_csv(protein_path)['smiles'].values[target_idx]

        valid_idx = []
        for idx, d in enumerate(self.smiless):
            if Chem.MolFromSmiles(d) == None:
                continue
            valid_idx.append(idx)

        self.valid_idx = valid_idx
        self.smiless = self.smiless[valid_idx]
        # self.proteins = self.proteins[valid_idx]
        # self.labels = labels[valid_idx]
        self.data = torch.load(os.path.join(root_path, f"{dataset}/{mode}/Data_{dataset}.pth"))

        self.cache_path = os.path.join(root_path, f"{dataset}/{mode}/{dataset}_{path_length}.pkl")

        self._pre_process()

        ecfp_path = os.path.join(root_path, f"{dataset}/{mode}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/{mode}/molecular_descriptors.npz")
        phar_path = os.path.join(root_path, f"{dataset}/{mode}/phar_features.pkl")

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        with open(phar_path, 'rb') as f:
            phars = pickle.load(f)

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')

        text_path = os.path.join(root_path, '../text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)

        self.fps, self.mds, self.phars = fps, mds, phars

        # Dataset Setting
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
        self.device = device

    def split_sequence(self, sequence, ngram):
        sequence = '_' + sequence + '='
        words = [self.word_dict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = graphs
            self.labels = label_dict['labels']
            self.proteins = label_dict['proteins']

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        elif self.sample_num == 1:
            random_questions = list([self.question_name])
        else:
            random_questions = self.question_name
        select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question in
                           random_questions]
        # prop_text = [self.text_prop_list['logP']['1'], self.text_prop_list['QED']['1']]
        # text = prop_text + select_question
        text = select_question
        texts, masks = self.tokenizer_text(text)
        # count phar group
        v_phar_name, v_phar_atom_id, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars[idx]:
            v_phar_name.append(feat[0].split(".")[0])
            v_phar_atom_id.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol = Chem.MolFromSmiles(self.smiless[idx])
        num_nodes = mol.GetNumAtoms()
        atom_phar_target_map = torch.zeros(num_nodes, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name, v_phar_atom_id):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map[y, phar_index] = 1

        # logP = MolLogP(mol)
        # QED = qed(mol)
        # SA = sascorer.calculateScore(mol)

        # phar_targets = torch.tensor([logP, QED, phar_targets_num])
        phar_targets = phar_targets_num
        return self.smiless[idx], self.proteins[idx], self.labels[idx], self.graphs[idx], \
            self.fps[idx], self.mds[idx], \
            v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions, self.data[
            idx]

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


class pharVQADataset_dta_CPI(Dataset):
    def __init__(self, root_path, dataset, smiles, proteins_keys, targets, smile_graph, smile_line_graph, protein_graph,
                 fps, mds, phars,
                 question_num=1, question_name=None, text_max_len=256, device=None):
        self.smiless = smiles
        self.proteins_keys = proteins_keys
        self.labels = targets
        self.smile_graph = smile_graph
        self.smiles_line_graph = smile_line_graph

        self.protein_graph = protein_graph
        self.preprocess()

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')

        text_path = os.path.join(root_path, '../text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)

        self.fps, self.mds, self.phars = fps, mds, phars

        # Dataset Setting
        self.d_fps = len(self.fps[0])
        self.d_mds = len(self.mds[0])
        self.device = device

    def preprocess(self):
        assert (len(self.smiless) == len(self.proteins_keys) and len(self.smiless) == len(
            self.labels)), 'These lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_list_pro_len = []
        data_len = len(self.smiless)
        for i in range(data_len):
            smiles = self.smiless[i]
            tar_key = self.proteins_keys[i]
            c_size, features, edge_index = self.smile_graph[smiles]
            target_features, target_size = self.protein_graph[tar_key]

            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([self.labels[i]]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list_pro_len.append(target_size)
            data_list_pro.append(target_features)
            data_list_mol.append(GCNData_mol)
        self.data_mol = data_list_mol
        self.proteins_features = data_list_pro
        self.proteins_len = data_list_pro_len

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        elif self.sample_num == 1:
            random_questions = list([self.question_name])
        else:
            random_questions = self.question_name
        select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question in
                           random_questions]
        # prop_text = [self.text_prop_list['logP']['1'], self.text_prop_list['QED']['1']]
        # text = prop_text + select_question
        text = select_question
        texts, masks = self.tokenizer_text(text)
        # count phar group
        v_phar_name, v_phar_atom_id, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars[idx]:
            v_phar_name.append(feat[0].split(".")[0])
            v_phar_atom_id.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol = Chem.MolFromSmiles(self.smiless[idx])
        num_nodes = mol.GetNumAtoms()
        atom_phar_target_map = torch.zeros(num_nodes, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name, v_phar_atom_id):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map[y, phar_index] = 1

        # phar_targets = torch.tensor([logP, QED, phar_targets_num])
        phar_targets = phar_targets_num
        return self.smiless[idx], self.proteins_features[idx], self.proteins_len[idx], self.labels[idx], \
            self.smiles_line_graph[idx], self.data_mol[idx], \
            self.fps[idx], self.mds[idx], \
            v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


class pharVQADataset_dti(Dataset):
    def __init__(self, root_path, dataset, path_length=5, use_idxs=[],
                 question_num=1, question_name=None, text_max_len=256, device=None):

        dataset_path = os.path.join(root_path, f"{dataset}/raw/data.txt")
        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")
        # load protein dict
        with open('../datasets/DTI/3ngram_vocab', 'r') as f:
            word_dic = f.read().split('\n')
            if word_dic[-1] == '':
                word_dic.pop()
        self.word_dict = {}
        for i, item in enumerate(word_dic):
            self.word_dict[item] = i
        self.n_word = len(self.word_dict)

        # load compound and protein data
        self.use_idxs = use_idxs
        self._pre_process()
        # load data

        # load compound and protein data
        with open(dataset_path, 'r') as f:
            data_list = f.read().strip().split('\n')
        data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

        molecules = []
        for no, data in enumerate((data_list)):
            smiles, sequence, interaction = data.strip().split()
            if Chem.MolFromSmiles(smiles) == None:
                continue
            molecules.append(smiles)

        self.smiless = [molecules[i] for i in use_idxs]

        ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        phar_path = os.path.join(root_path, f"{dataset}/phar_features.pkl")

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        with open(phar_path, 'rb') as f:
            phars = pickle.load(f)

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')

        text_path = os.path.join(root_path, 'text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)

        self.fps, self.mds, self.phars = fps[self.use_idxs], mds[self.use_idxs], [phars[i] for i in self.use_idxs]

        # Dataset Setting
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
        self.device = device

    def split_sequence(self, sequence, ngram):
        sequence = '_' + sequence + '='
        words = [self.word_dict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
            self.proteins = label_dict['proteins'][self.use_idxs]
            self.protein_lens = label_dict['protein_lens'][self.use_idxs]

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        elif self.sample_num == 1:
            random_questions = list([self.question_name])
        else:
            random_questions = self.question_name
        select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question in
                           random_questions]
        # prop_text = [self.text_prop_list['logP']['1'], self.text_prop_list['QED']['1']]
        # text = prop_text + select_question
        text = select_question
        texts, masks = self.tokenizer_text(text)
        # count phar group
        v_phar_name, v_phar_atom_id, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars[idx]:
            v_phar_name.append(feat[0].split(".")[0])
            v_phar_atom_id.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol = Chem.MolFromSmiles(self.smiless[idx])
        num_nodes = mol.GetNumAtoms()
        atom_phar_target_map = torch.zeros(num_nodes, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name, v_phar_atom_id):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map[y, phar_index] = 1

        # logP = MolLogP(mol)
        # QED = qed(mol)
        # SA = sascorer.calculateScore(mol)

        # phar_targets = torch.tensor([logP, QED, phar_targets_num])
        phar_targets = phar_targets_num
        return self.smiless[idx], self.proteins[idx], self.protein_lens[idx], self.labels[idx], self.graphs[idx], \
            self.fps[idx], self.mds[idx], \
            v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions, self.data[
            idx]

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


class pharVQADataset_ddi(Dataset):
    def __init__(self, root_path, dataset, path_length=5, use_idxs=None,
                 question_num=1, question_name=None, text_max_len=256, device=None):

        if dataset == 'biosnap':
            data_path = f"{root_path}/{dataset}/raw/all.csv"
            df = pd.read_csv(data_path)
            drugs1 = df['Drug1_SMILES'].values
            drugs2 = df['Drug2_SMILES'].values
            labels = df['label'].values
        elif dataset == 'twosides':
            data_path = f"{root_path}/{dataset}/raw/Drug_META_DDIE.db"
            conn = sqlite3.connect(data_path)
            drug = pd.read_sql("select * from Drug", conn)
            idToSmiles = {}
            for i in range(drug.shape[0]):
                idToSmiles[drug.loc[i][0]] = drug.loc[i][3]
            smile = drug['smile']
            positive_path = f"{root_path}/{dataset}/raw/twosides_interactions.csv"
            negative_path = f"{root_path}/{dataset}/raw/reliable_negatives.csv"
            positive = pd.read_csv(positive_path, header=None)
            negative = pd.read_csv(negative_path, header=None)
            df = pd.concat([positive, negative])

            drugs1 = []
            drugs2 = []
            labels = []
            for i in tqdm(range(df.shape[0])):
                try:
                    data1 = idToSmiles[df.iloc[i][0]]
                    data2 = idToSmiles[df.iloc[i][1]]
                    target = df.iloc[i][2]
                    drugs1.append(data1)
                    drugs2.append(data2)
                    labels.append(float(target))

                except:
                    continue
            drugs1 = np.array(drugs1)
            drugs2 = np.array(drugs2)
            labels = np.array(labels)

        valid_d1 = []
        valid_d2 = []
        valid_idx = []
        for idx, (d1, d2) in enumerate(zip(drugs1, drugs2)):
            if Chem.MolFromSmiles(d1) == None or Chem.MolFromSmiles(d2) == None:
                continue
            valid_d1.append(d1)
            valid_d2.append(d2)
            valid_idx.append(idx)
        valid_labels = labels[valid_idx]

        if use_idxs is None:
            self.use_idxs = range(len(valid_labels))
        else:
            self.use_idxs = use_idxs

        self.cache_path_d1 = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}_1.pkl")
        self.cache_path_d2 = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}_2.pkl")

        # load compound and protein data

        self._pre_process()
        self.smiless_d1 = [valid_d1[i] for i in use_idxs]
        self.smiless_d2 = [valid_d2[i] for i in use_idxs]

        # for drug1
        ecfp_path_1 = os.path.join(root_path, f"{dataset}/rdkfp1-7_512_1.npz")
        md_path_1 = os.path.join(root_path, f"{dataset}/molecular_descriptors_1.npz")
        phar_path_1 = os.path.join(root_path, f"{dataset}/phar_features_1.pkl")

        fps = torch.from_numpy(sps.load_npz(ecfp_path_1).todense().astype(np.float32))
        mds = np.load(md_path_1)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        with open(phar_path_1, 'rb') as f:
            phars_1 = pickle.load(f)

        self.fps_1, self.mds_1, self.phars_1 = fps[self.use_idxs], mds[self.use_idxs], [phars_1[i] for i in
                                                                                        self.use_idxs]

        # Dataset Setting
        self.d_fps_1 = self.fps_1.shape[1]
        self.d_mds_1 = self.mds_1.shape[1]

        # for drug2
        ecfp_path_2 = os.path.join(root_path, f"{dataset}/rdkfp1-7_512_2.npz")
        md_path_2 = os.path.join(root_path, f"{dataset}/molecular_descriptors_2.npz")
        phar_path_2 = os.path.join(root_path, f"{dataset}/phar_features_2.pkl")

        fps = torch.from_numpy(sps.load_npz(ecfp_path_2).todense().astype(np.float32))
        mds = np.load(md_path_2)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

        with open(phar_path_2, 'rb') as f:
            phars_2 = pickle.load(f)

        self.fps_2, self.mds_2, self.phars_2 = fps[self.use_idxs], mds[self.use_idxs], [phars_2[i] for i in
                                                                                        self.use_idxs]

        # Dataset Setting
        self.d_fps_2 = self.fps_2.shape[1]
        self.d_mds_2 = self.mds_2.shape[1]

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')

        text_path = os.path.join(root_path, '../text', 'phar_question_howmany.json')
        # text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        # with open(text_properity_path, 'r', encoding='utf-8') as fp:
        #     self.text_prop_list = json.load(fp)

        self.device = device

    def _pre_process(self):
        if not os.path.exists(self.cache_path_d1) or not os.path.exists(self.cache_path_d2):
            print(f"{self.cache_path_d1} not exists, please run preprocess.py")
        else:
            graphs_1, label_dict = load_graphs(self.cache_path_d1)
            graphs_2, label_dict = load_graphs(self.cache_path_d2)
            self.graphs_1 = []
            self.graphs_2 = []
            for i in self.use_idxs:
                self.graphs_1.append(graphs_1[i])
                self.graphs_2.append(graphs_2[i])
            self.labels = label_dict['labels'][self.use_idxs]

    def __len__(self):
        return len(self.smiless_d1)

    def __getitem__(self, idx):
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        elif self.sample_num == 1:
            random_questions = list([self.question_name])
        else:
            random_questions = self.question_name
        select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question in
                           random_questions]
        # prop_text = [self.text_prop_list['logP']['1'], self.text_prop_list['QED']['1']]
        # text = prop_text + select_question
        text = select_question
        texts, masks = self.tokenizer_text(text)

        # count phar 1 group
        v_phar_name_d1, v_phar_atom_id_d1, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars_1[idx]:
            v_phar_name_d1.append(feat[0].split(".")[0])
            v_phar_atom_id_d1.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name_d1))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name_d1]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol_d1 = Chem.MolFromSmiles(self.smiless_d1[idx])
        num_nodes_d1 = mol_d1.GetNumAtoms()
        atom_phar_target_map_d1 = torch.zeros(num_nodes_d1, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name_d1, v_phar_atom_id_d1):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map_d1[y, phar_index] = 1
        phar_targets_d1 = phar_targets_num

        # count phar 2 group
        v_phar_name_d2, v_phar_atom_id_d2, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in self.phars_1[idx]:
            v_phar_name_d2.append(feat[0].split(".")[0])
            v_phar_atom_id_d2.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name_d2))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name_d2]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol_d2 = Chem.MolFromSmiles(self.smiless_d1[idx])
        num_nodes_d2 = mol_d2.GetNumAtoms()
        atom_phar_target_map_d2 = torch.zeros(num_nodes_d2, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name_d2, v_phar_atom_id_d2):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map_d2[y, phar_index] = 1
        phar_targets_d2 = phar_targets_num

        # logP = MolLogP(mol)
        # QED = qed(mol)
        # SA = sascorer.calculateScore(mol)
        return self.smiless_d1[idx], self.graphs_1[idx], self.fps_1[idx], self.mds_1[
            idx], v_phar_name_d1, v_phar_atom_id_d1, phar_targets_d1, atom_phar_target_map_d1, \
            self.smiless_d2[idx], self.graphs_2[idx], self.fps_2[idx], self.mds_2[
            idx], v_phar_name_d2, v_phar_atom_id_d2, phar_targets_d2, atom_phar_target_map_d2, \
            self.labels[idx], texts, masks, random_questions

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks


class pharVQADataset_retro(Dataset):
    def __init__(self, mode, root_path='./data/data',
                 known_class=False, shared_vocab=False, augment=False, rerank_num=1,
                 question_num=1, question_name=None, text_max_len=256, path_length=5, device=None):
        self.root_path = root_path
        assert mode in ['train', 'test', 'valid']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.rerank_num = rerank_num
        self.is_aug = augment
        self.aug = MolAugmenter()
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        print('Building {} data from: {}'.format(mode, root_path))
        vocab_file = ''
        if 'full' in self.root_path:
            vocab_file = 'full_'
        if shared_vocab:
            vocab_file += 'vocab_share.pk'
        else:
            vocab_file += 'vocab.pk'

        if mode != 'train':
            assert vocab_file in os.listdir(root_path)
            with open(os.path.join(root_path, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)
            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
        else:
            with open(os.path.join(root_path, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)

            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

        # Build and load processed data into lmdb
        self.env = lmdb.open(
            os.path.join(root_path, f'{mode}/Class_{known_class}_aug_{rerank_num}/cooked_{self.mode}.lmdb'),
            max_readers=1, readonly=True,
            lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))

        # get pharVQA dataset

        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')

        text_path = os.path.join(root_path, '../text', 'phar_question_howmany.json')
        text_properity_path = os.path.join(root_path, '../text', 'properities.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)
        with open(text_properity_path, 'r', encoding='utf-8') as fp:
            self.text_prop_list = json.load(fp)

        # Dataset Setting
        self.d_fps = 512
        self.d_mds = 200
        self.device = device

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return len(self.product_keys)

    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        # new_index = int(p_key.decode().split(' ')[0])
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        p_smiles = processed['raw_product']
        r_smiles = processed['raw_reactants']
        context_align = processed['context_align']
        src, tgt, graph = processed['product_token'], processed['reactant_token'], processed['product_graph']

        fps = torch.from_numpy(processed['fps'].todense().astype(np.float32))
        mds = processed['mds'][1:].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        phar = processed['phars']

        if self.known_class == 'False':
            src = src[1:]
        if self.is_aug and random.random() > 0.5:

            p_mol = self.aug([Chem.MolFromSmiles(p_smiles)])[0]
            r_mol = self.aug([Chem.MolFromSmiles(r_smiles)])[0]

            p_smiles = Chem.MolToSmiles(p_mol, canonical=not self.is_aug, isomericSmiles=False)
            r_smiles = Chem.MolToSmiles(r_mol, canonical=not self.is_aug, isomericSmiles=False)

            cano_p_smiles = remove_am_without_canonical(p_smiles)
            cano_r_smiles = remove_am_without_canonical(r_smiles)

            src_token = smi_tokenizer(cano_p_smiles)
            tgt_token = ['<sos>'] + smi_tokenizer(cano_r_smiles) + ['<eos>']

            if self.known_class == 'False':
                src = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
            else:
                src = [src[0]] + [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]

            tgt = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]
            position_mapping_list = extract_relative_mapping(p_smiles, r_smiles)
            context_align = torch.zeros(
                (len(smi_tokenizer(r_smiles)) + 1, len(smi_tokenizer(p_smiles)) + 1)).long()
            for i, j in position_mapping_list:
                context_align[i][j + 1] = 1
        # pharVQA
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        elif self.sample_num == 1:
            random_questions = list([self.question_name])
        else:
            random_questions = self.question_name
        select_question = [random.sample(list(self.text_list[random_question].values()), 1)[0] for random_question
                           in random_questions]
        # prop_text = [self.text_prop_list['logP']['1'], self.text_prop_list['QED']['1']]
        # text = prop_text + select_question
        text = select_question
        texts, masks = self.tokenizer_text(text)
        # count phar group
        v_phar_name, v_phar_atom_id, v_phar_phar_pos, v_phar_feature = [], [], [], []
        for feat in phar:
            v_phar_name.append(feat[0].split(".")[0])
            v_phar_atom_id.append(feat[1])
            v_phar_phar_pos.append(feat[2])

        attn_phar_atom = torch.zeros(self.sample_num, len(v_phar_name))
        for phar_index, phar_task in enumerate(random_questions):
            attn_phar_atom[phar_index, [i == phar_task for i in v_phar_name]] = 1
        phar_targets_num = attn_phar_atom.sum(dim=1).long()

        mol = Chem.MolFromSmiles(p_smiles)
        num_nodes = mol.GetNumAtoms()
        atom_phar_target_map = torch.zeros(num_nodes, len(list(self.text_list.keys())))
        for x, y in zip(v_phar_name, v_phar_atom_id):
            if x not in list(self.text_list.keys()):
                continue
            phar_index = list(self.text_list.keys()).index(x)
            atom_phar_target_map[y, phar_index] = 1

        # logP = MolLogP(mol)
        # QED = qed(mol)
        # SA = sascorer.calculateScore(mol)

        # phar_targets = torch.tensor([logP, QED, phar_targets_num])
        phar_targets = phar_targets_num

        return p_smiles, r_smiles, src, tgt, context_align, graph, fps, mds, v_phar_name, v_phar_atom_id, \
            texts, masks, phar_targets, atom_phar_target_map, random_questions

    def tokenizer_text(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            description=text, tokenizer=self.tokenizer, max_seq_len=self.text_max_len)
        return text_tokens_ids, text_masks
