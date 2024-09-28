import argparse
import json
import multiprocessing
import os
import pickle
import sqlite3
import time
from collections import OrderedDict
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


def check_valid(reactant):
    valid_index = []
    for i, r in enumerate(reactant):
        if Chem.MolFromSmiles(r) is not None:
            valid_index.append(i)
    reactant = [reactant[i] for i in valid_index]
    return reactant


def split_sequence(word_dict, sequence, ngram):
    sequence = '_' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def main(args):
    data_path = f"{args.root_path}/{args.dataset}/{args.mode}/"

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


    out_path = os.path.join(args.root_path, f"{args.dataset}/{args.mode}/phar_features.pkl")
    num_workers = multiprocessing.cpu_count() - 4
    print(f'processing {args.dataset} dataset on {args.root_path}...')
    # multi-processing
    with Pool(processes=num_workers) as pool:
        result = list(tqdm(pool.imap(extract_phar_feature_multi, smiles), total=len(smiles)))

    with open(out_path, 'wb') as file_handle:
        pickle.dump(result, file_handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BindingDB_reg')
    parser.add_argument('--mode', type=str, default='ind_test')
    parser.add_argument("--root_path", type=str, default='../../datasets/DTA/')
    args = parser.parse_args()

    main(args)
