import multiprocessing
import os
import pickle
import time
from multiprocessing import Pool
from threading import Timer
import multiprocessing as mp

import pandas as pd
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from tqdm import tqdm

from extract_phar_feature.phar_utils import extract_phar_feature_multi
from model.datasets.MoleculeNet_Graph import _load_bace_dataset, _load_tox21_dataset, _load_hiv_dataset, \
    _load_bbbp_dataset, _load_clintox_dataset, _load_esol_dataset, _load_freesolv_dataset, _load_lipophilicity_dataset, \
    _load_malaria_dataset, _load_cep_dataset, _load_muv_dataset, _load_sider_dataset, _load_toxcast_dataset, \
    _load_estrogen_dataset

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

data_names = ['muv']


# data_names = ['tox21', 'freesolv', 'bace', 'bbbp', 'clintox', 'sider', 'esol', 'estrogen']
# data_names = ['lipo','metstab','toxcast']
# data_names = ['chembl29']
# data_names = ['hiv','muv','pcba']


def data_process(dataset,data_path):
    if dataset == 'tox21':
        smiles_list, rdkit_mol_objs, labels = \
            _load_tox21_dataset(data_path)
    elif dataset == 'hiv':
        smiles_list, rdkit_mol_objs, labels = \
            _load_hiv_dataset(data_path)
    elif dataset == 'bace':
        smiles_list, rdkit_mol_objs, folds, labels = \
            _load_bace_dataset(data_path)
    elif dataset == 'bbbp':
        smiles_list, rdkit_mol_objs, labels = \
            _load_bbbp_dataset(data_path)
    elif dataset == 'clintox':
        smiles_list, rdkit_mol_objs, labels = \
            _load_clintox_dataset(data_path)
    elif dataset == 'esol':
        smiles_list, rdkit_mol_objs, labels = \
            _load_esol_dataset(data_path)
    elif dataset == 'freesolv':
        smiles_list, rdkit_mol_objs, labels = \
            _load_freesolv_dataset(data_path)
    elif dataset == 'lipophilicity':
        smiles_list, rdkit_mol_objs, labels = \
            _load_lipophilicity_dataset(data_path)
    elif dataset == 'malaria':
        smiles_list, rdkit_mol_objs, labels = \
            _load_malaria_dataset(data_path)
    elif dataset == 'cep':
        smiles_list, rdkit_mol_objs, labels = \
            _load_cep_dataset(data_path)
    elif dataset == 'muv':
        smiles_list, rdkit_mol_objs, labels = \
            _load_muv_dataset(data_path)
    elif dataset == 'pcba':
        smiles_list, rdkit_mol_objs, labels = \
            _load_pcba_dataset(data_path)
    elif dataset == 'sider':
        smiles_list, rdkit_mol_objs, labels = \
            _load_sider_dataset(data_path)
    elif dataset == 'toxcast':
        smiles_list, rdkit_mol_objs, labels = \
            _load_toxcast_dataset(data_path)
    elif dataset == 'estrogen':
        smiles_list, rdkit_mol_objs, labels = \
            _load_estrogen_dataset(data_path)
    elif dataset == 'metstab':
        smiles_list, rdkit_mol_objs, labels = \
            _load_toxcast_dataset(data_path)

    return smiles_list


for index, data_name in enumerate(data_names):
    print(f'processing {data_name} dataset...')
    if data_name != 'chembl29':
        data_path = os.path.join('../data/chem_dataset_MoMu/dataset', f'{data_name}/raw/{data_name}.csv')
        datas = data_process(data_name, data_path)
    else:
        data_path = os.path.join('../data/chem_dataset_MoMu/dataset', f'{data_name}/smiles.smi')
        with open(data_path, 'r') as f:
            lines = f.readlines()
            datas = [line.strip('\n') for line in lines if len(line.strip('\n')) < 300]
    datasets = []
    targets = []
    num_workers = multiprocessing.cpu_count() - 4
    # multi-processing
    # 创建进程池
    # result = []
    # for i in tqdm(datas):
    #     if i is None:
    #         continue
    #     result.append(extract_phar_feature_multi(i))
    with Pool(processes=num_workers) as pool:
        result = list(tqdm(pool.imap(extract_phar_feature_multi, datas), total=len(datas)))
    count_empty_lists = sum(1 for sublist in result if not sublist)
    print(f'the dataset {data_name} has {count_empty_lists} molecules without phars')
    with open(os.path.join('../datasets/RXNprediction/dataset', f'{data_name}/raw/phar_features.pkl'),
              'wb') as file_handle:
        pickle.dump(result, file_handle)

    # with open(os.path.join('../data/chem_datasets/', f'{data_name}/smiles_dumplen300.smi'), 'w') as f:
    #     for data in datas:
    #         f.write(data+'\n')
    # for task in factory.GetFeatureFamilies():
    #     dataset, target = extract_phar_feature(datas, task=task)
    #     datasets.append(dataset)
    #     targets.append(target)
