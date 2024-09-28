import argparse
import multiprocessing
import os
import pickle
import time
from multiprocessing import Pool
from threading import Timer
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from rdkit import RDConfig, Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

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


def check_valid(reactant, product):
    valid_index = []
    for i, (r, p) in enumerate(zip(reactant, product)):
        if Chem.MolFromSmiles(r) is not None and Chem.MolFromSmiles(p) is not None:
            valid_index.append(i)
    reactant = [reactant[i] for i in valid_index]
    product = [product[i] for i in valid_index]
    return reactant, product


def split_sequence(word_dict, sequence, ngram):
    sequence = '_' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def main(args):
    args.data_path = f"{args.root_path}/{args.dataset}/raw/data.txt"
    out_path = os.path.join(args.root_path, f"{args.dataset}/phar_features.pkl")

    # load protein dict
    with open('../../datasets/DTI/3ngram_vocab', 'r') as f:
        word_dic = f.read().split('\n')
        if word_dic[-1] == '':
            word_dic.pop()
    word_dict = {}
    for i, item in enumerate(word_dic):
        word_dict[item] = i

    # load compound and protein data
    with open(args.data_path, 'r') as f:
        data_list = f.read().strip().split('\n')
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    molecules = []
    for no, data in enumerate(tqdm(data_list)):
        smiles, sequence, interaction = data.strip().split()
        if Chem.MolFromSmiles(smiles) == None:
            continue
        molecules.append(smiles)

    num_workers = multiprocessing.cpu_count() - 4
    print(f'processing {args.dataset} dataset on {args.root_path}...')
    # multi-processing
    with Pool(processes=num_workers) as pool:
        result = list(tqdm(pool.imap(extract_phar_feature_multi, molecules), total=len(molecules)))

    with open(out_path, 'wb') as file_handle:
        pickle.dump(result, file_handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='human')
    parser.add_argument("--root_path", type=str, default='../../datasets/DTI/')
    args = parser.parse_args()

    main(args)
