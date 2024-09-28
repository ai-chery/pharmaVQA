import argparse
import multiprocessing
import os
import pickle
import sqlite3
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
    if args.dataset == 'biosnap':
        data_path = f"{args.root_path}/{args.dataset}/raw/all.csv"
        df = pd.read_csv(data_path)
        drugs1 = df['Drug1_SMILES'].values
        drugs2 = df['Drug2_SMILES'].values
        labels = df['label'].values
    elif args.dataset == 'twosides':
        data_path = f"{args.root_path}/{args.dataset}/raw/Drug_META_DDIE.db"
        conn = sqlite3.connect(data_path)
        drug = pd.read_sql("select * from Drug", conn)
        idToSmiles = {}
        for i in range(drug.shape[0]):
            idToSmiles[drug.loc[i][0]] = drug.loc[i][3]
        smile = drug['smile']
        # positive_path = os.path.join('../data/twosides/raw/', 'twosides_interactions.csv')
        # negative_path = os.path.join(self.root, 'raw', 'reliable_negatives.csv')
        positive_path = f"{args.root_path}/{args.dataset}/raw/twosides_interactions.csv"
        negative_path = f"{args.root_path}/{args.dataset}/raw/reliable_negatives.csv"
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

    args.data_path = f"{args.root_path}/{args.dataset}/raw/data.txt"
    out_path1 = os.path.join(args.root_path, f"{args.dataset}/phar_features_1.pkl")
    out_path2 = os.path.join(args.root_path, f"{args.dataset}/phar_features_2.pkl")

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

    num_workers = multiprocessing.cpu_count() - 4
    print(f'processing {args.dataset} dataset on {args.root_path}...')
    # multi-processing
    with Pool(processes=num_workers) as pool:
        result = list(tqdm(pool.imap(extract_phar_feature_multi, valid_d1), total=len(valid_d1)))

    with open(out_path1, 'wb') as file_handle:
        pickle.dump(result, file_handle)

    # multi-processing
    with Pool(processes=num_workers) as pool:
        result = list(tqdm(pool.imap(extract_phar_feature_multi, valid_d2), total=len(valid_d2)))

    with open(out_path2, 'wb') as file_handle:
        pickle.dump(result, file_handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='biosnap')
    parser.add_argument("--root_path", type=str, default='../../datasets/DDI/')
    args = parser.parse_args()

    main(args)
