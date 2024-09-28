import json
import pickle
import random

from rdkit import Chem
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.sparse as sps
import torch
import dgl.backend as F
from transformers import AutoTokenizer
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
    tokens_ids = torch.Tensor(tokens_ids).long()
    masks = torch.Tensor(masks).bool()
    return tokens_ids, masks

class MoleculeDataset(Dataset):
    def __init__(self, root_path):
        smiles_path = os.path.join(root_path, "smiles.smi")
        fp_path = os.path.join(root_path, "rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, "molecular_descriptors.npz")
        with open(smiles_path, 'r') as f:
            lines = f.readlines()
            self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

        self._task_pos_weights = self.task_pos_weights()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx], self.fps[idx], self.mds[idx]

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.fps.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.fps, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.fps.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights


class pharVQADataset(Dataset):
    def __init__(self, root_path, dataset, question_num=1, text_max_len=256, device=None):
        smiles_path = os.path.join(root_path, f"{dataset}/smiles.smi")
        fp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        phar_path = os.path.join(root_path, f"{dataset}/phar_features.pkl")
        # Load Data
        with open(smiles_path, 'r') as f:
            lines = f.readlines()
            self.smiless = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

        with open(phar_path, 'rb') as f:
            self.phars = pickle.load(f)

        # phar task Setting
        self.sample_num = question_num
        self.text_max_len = text_max_len
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')
        text_path = os.path.join(root_path, 'text', 'phar_question_howmany.json')

        with open(text_path, 'r', encoding='utf-8') as fp:
            self.text_list = json.load(fp)

        # Dataset Setting
        self._task_pos_weights = self.task_pos_weights()
        self.device = device

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        if self.sample_num == len(list(self.text_list.keys())):
            random_questions = list(self.text_list.keys())
        else:
            random_questions = random.sample(list(self.text_list.keys()), self.sample_num)
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
        return self.smiless[idx], self.fps[idx], self.mds[
            idx], v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.fps.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.fps, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.fps.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights


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
