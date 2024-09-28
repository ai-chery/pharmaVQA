import os

import pandas as pd
import torch
import numpy as np

import json

from src.data.splitters import random_split, random_scaffold_split, scaffold_split

root_path = '../datasets'
dataset = 'hiv'
use_split_method = 'MoleculeSTM'
seeds = [1, 2, 3, 4]
MPG_seeds = [88, 89, 90]
dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")

if use_split_method == 'use_recommend_split':
    smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    print(f'{dataset} has {len(smiles_list)} smiles')
    for seed in seeds:
        if dataset in ['tox21', 'toxcast', 'sider', 'clintox', 'pcba', 'muv', 'esol', 'lipo', 'freesolv']:
            train_index, valid_index, test_index = random_split(
                torch.arange(len(smiles_list)), smiles_list=smiles_list, null_value=0, frac_train=0.8,
                frac_valid=0.1, frac_test=0.1, seed=seed)
        elif dataset in ['bace', 'bbbp', 'hiv']:
            train_index, valid_index, test_index = random_scaffold_split(
                torch.arange(len(smiles_list)), smiles_list=smiles_list, null_value=0, frac_train=0.8,
                frac_valid=0.1, frac_test=0.1, seed=seed)

        save_index = [train_index.numpy().tolist(), valid_index.numpy().tolist(), test_index.numpy().tolist()]

        filename = os.path.join(root_path, f"chem_datasets/{dataset}/splits/scaffold-random{seed}.json")
        with open(filename, 'w') as file_obj:
            json.dump(save_index, file_obj)

elif use_split_method == 'GEM':
    smiles_list = pd.read_csv(dataset_path)['smiles']
    print(f'{dataset} has {len(smiles_list)} smiles')
    for seed in seeds:
        scaffold_split = GEMScaffoldSplitter()
        train_index, valid_index, test_index = scaffold_split.split(
            dataset=smiles_list,all_index = torch.arange(len(smiles_list)), frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        save_index = [train_index.numpy().tolist(), valid_index.numpy().tolist(), test_index.numpy().tolist()]
        filename = os.path.join(root_path, f"chem_datasets/{dataset}/splits/scaffold-GEM{seed}.json")
        with open(filename, 'w') as file_obj:
            json.dump(save_index, file_obj)
elif use_split_method == 'MoleculeSTM':
    smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    print(f'{dataset} has {len(smiles_list)} smiles')
    train_index, valid_index, test_index = scaffold_split(
        torch.arange(len(smiles_list)), smiles_list=smiles_list, null_value=0, frac_train=0.8,
        frac_valid=0.1, frac_test=0.1)
    save_index = [train_index.numpy().tolist(), valid_index.numpy().tolist(), test_index.numpy().tolist()]
    filename = os.path.join(root_path, f"{dataset}/splits/scaffold-{use_split_method}.json")
    with open(filename, 'w') as file_obj:
        json.dump(save_index, file_obj)
# with open(filename, 'r', encoding='utf-8') as fp:
#     use_idxs = json.load(fp)

print(f"finishing in spliting {dataset}...")
