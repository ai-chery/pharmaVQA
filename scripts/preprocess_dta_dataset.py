import json
import pickle
import random
import sys
from collections import OrderedDict

import torch
from pysmilesutils.augment import MolAugmenter
from rdkit.Chem import AllChem
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.append("..")
from DTA.loader import mol_to_graph_data_obj_simple

import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
import argparse

from src.data.featurizer import smiles_to_graph_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--root_path", type=str, default='../datasets/DTA/')
    parser.add_argument("--dataset", type=str, default='BindingDB_reg')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--embedding_path", type=str, default='./')

    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args


def split_sequence(word_dict, sequence, ngram):
    sequence = '_' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def preprocess_dataset_dta(args):

    data_path = f"{args.root_path}/{args.dataset}/{args.mode}.csv"
    cache_file_path = f"{args.root_path}/{args.dataset}/{args.mode}/{args.dataset}_{args.path_length}"

    input_idx = pd.read_csv(data_path)

    smiles_idx = input_idx['smiles_id'].values
    target_idx = input_idx['target_id'].values
    labels = input_idx['affinity'].values

    smiles_path = f"{args.root_path}/{args.dataset}/smiles.csv"
    protein_path = f"{args.root_path}/{args.dataset}/protein.csv"

    smiles = pd.read_csv(smiles_path)['smiles'].values[smiles_idx]
    protein = pd.read_csv(protein_path)['protein'].values[target_idx]
    protein_list = [list(seq_cat(t)) for t in protein]
    # Graph preprocess

    data_list = []
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles]
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
    for i in tqdm(range(len(smiles))):
        rdkit_mol = preprocessed_rdkit_mol_objs_list[i]
        if rdkit_mol != None:
            data = mol_to_graph_data_obj_simple(rdkit_mol, extra_feature=True)
            data.id = torch.tensor([i])
            data.y = torch.FloatTensor([labels[i]])
            data.target = torch.LongTensor(protein_list[i])
            data_list.append(data)

    torch.save(data_list, f"{args.root_path}/{args.dataset}/{args.mode}/Data_{args.dataset}.pth")

    # Molecules pre-process
    # print('constructing graphs')
    # graphs = pmap(smiles_to_graph_tune,
    #               smiles,
    #               max_length=args.path_length,
    #               n_virtual_nodes=2,
    #               n_jobs=args.n_jobs)
    # valid_ids = []
    # valid_graphs = []
    # for i, g in enumerate(graphs):
    #     if g is not None:
    #         valid_ids.append(i)
    #         valid_graphs.append(g)
    # valid_smiles = smiles[valid_ids]
    # valid_proteins = protein[valid_ids]
    # valid_labels = labels[valid_ids]
    #
    # protein_list = [list(seq_cat(t)) for t in valid_proteins]
    #
    # print('saving graphs')
    # save_graphs(cache_file_path + '.pkl', valid_graphs,
    #             labels={'labels': torch.tensor(valid_labels),'proteins': torch.LongTensor(protein_list)})
    #
    # print('extracting reactant fingerprints')
    # FP_list = []
    # for r_smiles in valid_smiles:
    #     mol = Chem.MolFromSmiles(r_smiles)
    #     FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    # FP_arr = np.array(FP_list)
    # FP_sp_mat = sp.csc_matrix(FP_arr)
    # print('saving fingerprints')
    # sp.save_npz(f"{args.root_path}/{args.dataset}/{args.mode}/rdkfp1-7_512.npz", FP_sp_mat)
    #
    # print('extracting molecular descriptors')
    # fn = RDKit2DNormalized()
    # features_map = []
    # for mol in tqdm(valid_smiles):
    #     features_map.append(fn.process(mol))
    # # generator = RDKit2DNormalized()
    # # features_map = Pool(args.n_jobs).imap(generator.process, molecules)
    # arr = np.array(list(features_map))
    # np.savez_compressed(f"{args.root_path}/{args.dataset}/{args.mode}/molecular_descriptors.npz", md=arr[:, 1:])

def preprocess_dataset_dta_CPI(args):

    data_path = f"{args.root_path}/{args.dataset}/{args.mode}/"
    cache_file_path = f"{args.root_path}/{args.dataset}/{args.mode}/{args.dataset}_{args.path_length}"

    ligands = json.load(open(data_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    # proteins = json.load(open(data_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    # affinity = pickle.load(open(data_path + 'Y', 'rb'), encoding='latin1')
    # smile_file_name = data_path + '/smile_graph'

    # load compounds graph
    smiles = []
    smile_graph = {}
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        smiles.append(lg)
    # # load seqs
    # target_key = []
    # target_graph = {}
    # for key in proteins.keys():
    #     target_key.append(key)
    #     target_graph[key] = target_matrics(key, args.embedding_path)

    # Molecules pre-process
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  smiles,
                  max_length=args.path_length,
                  n_virtual_nodes=2,
                  n_jobs=args.n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    valid_smiles = [smiles[idx] for idx in valid_ids]

    print('saving graphs')
    save_graphs(cache_file_path + '.pkl', valid_graphs,
                labels={'valid_idx': torch.LongTensor(valid_ids)})

    print('extracting reactant fingerprints')
    FP_list = []
    for r_smiles in valid_smiles:
        mol = Chem.MolFromSmiles(r_smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{args.root_path}/{args.dataset}/{args.mode}/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    fn = RDKit2DNormalized()
    features_map = []
    for mol in tqdm(valid_smiles):
        features_map.append(fn.process(mol))
    # generator = RDKit2DNormalized()
    # features_map = Pool(args.n_jobs).imap(generator.process, molecules)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.root_path}/{args.dataset}/{args.mode}/molecular_descriptors.npz", md=arr[:, 1:])

if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset_dta_CPI(args)
