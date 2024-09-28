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


def check_valid(smiles):
    valid_index = []
    for i, p in enumerate(smiles):
        if Chem.MolFromSmiles(p) is not None:
            valid_index.append(i)
    product = [smiles[i] for i in valid_index]
    return product


def split_sequence(word_dict, sequence, ngram):
    sequence = '_' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def main(args):
    out_path = os.path.join(args.root_path, f"{args.dataset}/phar_features.pkl")
    if args.dataset == 'DrugBank':
        df = pd.read_csv(f"{args.root_path}/{args.dataset}/structure_links.csv")
        smiless = df['SMILES'].dropna().tolist()
    else:
        df = pd.read_csv(f"{args.root_path}/{args.dataset}/{args.dataset}.csv")
        df = df.dropna(axis=0, how='all')
        try:
            smiless = df.smiles.values.tolist()
        except:
            smiless = df.Smiles.values.tolist()

    smiless = check_valid(smiless)
    num_workers = multiprocessing.cpu_count() - 4
    print(f'processing {args.dataset} dataset on {args.root_path}...')
    # multi-processing
    with Pool(processes=num_workers) as pool:
        result = list(tqdm(pool.imap(extract_phar_feature_multi, smiless), total=len(smiless)))

    with open(out_path, 'wb') as file_handle:
        pickle.dump(result, file_handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HPK1_KI')
    parser.add_argument("--root_path", type=str, default='../../datasets/vs/')
    args = parser.parse_args()

    main(args)
