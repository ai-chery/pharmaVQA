import multiprocessing
import os
import pickle
import time
from multiprocessing import Pool
from threading import Timer
import multiprocessing as mp

import pandas as pd
from rdkit import RDConfig, Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from tqdm import tqdm

from MoleculeNet_Graph import _load_bace_dataset, _load_tox21_dataset, _load_hiv_dataset, \
    _load_bbbp_dataset, _load_clintox_dataset, _load_esol_dataset, _load_freesolv_dataset, _load_lipophilicity_dataset, \
    _load_malaria_dataset, _load_cep_dataset, _load_muv_dataset, _load_sider_dataset, _load_toxcast_dataset, \
    _load_estrogen_dataset

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def extract_phar_feature_multi(data):
    datas_feat, datas_index = [], []
    all_fea_num = 0
    phar_target = []

    m = Chem.MolFromSmiles(data)
    # get 2D corr
    AllChem.Compute2DCoords(m)
    feats = factory.GetFeaturesForMol(m)
    # for every molecules, find corresponding features
    fp_name = []
    fp = []
    phar_targets = []
    for feat in feats:
        fp_name.append(['.'.join((feat.GetFamily(), feat.GetType())), feat.GetAtomIds(), list(feat.GetPos())])
        fp.append('.'.join((feat.GetFamily(), feat.GetType())))

    all_fea_num += len(feats)
    return fp_name

data_names = ['bace']


# data_names = ['tox21', 'freesolv', 'bace', 'bbbp', 'clintox', 'sider', 'esol', 'estrogen']
# data_names = ['lipo','metstab','toxcast']
# data_names = ['chembl29']
# data_names = ['hiv','muv','pcba']

for index, data_name in enumerate(data_names):
    print(f'processing {data_name} dataset...')
    if data_name != 'chembl29':
        data_path = os.path.join('../../datasets/', f'{data_name}/{data_name}.csv')
        datas = pd.read_csv(data_path)['smiles']
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
    result = []
    for i in tqdm(datas):
        if i is None:
            continue
        result.append(extract_phar_feature_multi(i))
    # with Pool(processes=num_workers) as pool:
    #     result = list(tqdm(pool.imap(extract_phar_feature_multi, datas), total=len(datas)))
    count_empty_lists = sum(1 for sublist in result if not sublist)
    # with open(os.path.join('../data/chem_datasets/', f'{data_name}/smiles_dumplen300.smi'), 'w') as f:
    #     for data in datas:
    #         f.write(data+'\n')
    # for task in factory.GetFeatureFamilies():
    #     dataset, target = extract_phar_feature(datas, task=task)
    #     datasets.append(dataset)
    #     targets.append(target)
