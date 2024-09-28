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

# from MoleculeNet_Graph import _load_bace_dataset, _load_tox21_dataset, _load_hiv_dataset, \
#     _load_bbbp_dataset, _load_clintox_dataset, _load_esol_dataset, _load_freesolv_dataset, _load_lipophilicity_dataset, \
#     _load_malaria_dataset, _load_cep_dataset, _load_muv_dataset, _load_sider_dataset, _load_toxcast_dataset, \
#     _load_estrogen_dataset, _load_moleculeace_dataset, _load_tdc_dataset

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

# data_names = ['CHEMBL204_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', \
#               'CHEMBL233_Ki', 'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 'CHEMBL237_Ki', \
#               'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL287_Ki',
#               'CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 'CHEMBL2147_Ki', 'CHEMBL2835_Ki',
#               'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 'CHEMBL4616_EC50', 'CHEMBL4792_Ki']


# data_names = ['ames', 'bbb_martins', 'bioavailability_ma', 'caco2_wang', 'clearance_hepatocyte_az',
#               'clearance_microsome_az', \
#               'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith',
#               'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith', \
#               'dili', 'half_life_obach', 'herg', 'hia_hou', 'ld50_zhu', 'lipophilicity_astrazeneca',
#               'pgp_broccatelli', 'ppbr_az', 'solubility_aqsoldb', 'vdss_lombardo']

# data_names = ['CHEMBL1862_Ki']


# data_names = ['tox21', 'freesolv', 'bace', 'bbbp', 'clintox', 'sider', 'esol', 'estrogen']
# data_names = ['lipo','metstab','toxcast']
data_names = ['chembl29']
# data_names = ['hiv','muv','pcba']


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


def data_process(dataset, data_path):
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
    elif dataset == 'moleculeace':
        smiles_list, rdkit_mol_objs, labels = \
            _load_moleculeace_dataset(data_path)
    elif dataset == 'tdc':
        smiles_list, rdkit_mol_objs, labels = \
        _load_tdc_dataset(data_path)

    return smiles_list


for index, data_name in enumerate(data_names):
    print(f'processing {data_name} dataset...')
    if data_name != 'chembl29':
        data_path = os.path.join('../datasets/tdc', f'{data_name}/{data_name}.csv')
        datas = data_process('tdc', data_path)
    else:
        data_path = os.path.join('../datasets/pretrain', f'{data_name}/smiles.smi')
        with open(data_path, 'r') as f:
            lines = f.readlines()
            datas = [line.strip('\n') for line in lines if len(line.strip('\n'))]
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
    print(f'the dataset {data_name} has {count_empty_lists} molecules without phars')
    with open(os.path.join('../datasets/pretrain', f'{data_name}/phar_features.pkl'),
              'wb') as file_handle:
        pickle.dump(result, file_handle)

    # with open(os.path.join('../data/chem_datasets/', f'{data_name}/smiles_dumplen300.smi'), 'w') as f:
    #     for data in datas:
    #         f.write(data+'\n')
    # for task in factory.GetFeatureFamilies():
    #     dataset, target = extract_phar_feature(datas, task=task)
    #     datasets.append(dataset)
    #     targets.append(target)
