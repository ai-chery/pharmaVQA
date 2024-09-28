import argparse
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
from rdkit import RDLogger

from RXNprediction.utils import get_mols
from scripts.preprocess_rxn_dataset import reaction_loader

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


def main(args):
    if args.data_type == 'forward':
        dataset_path = os.path.join(args.root_path, f"USPTO-480k/{args.data_name}_parsed.txt")
        out_path_r = os.path.join(args.root_path, f"USPTO-480k/{args.data_name}_parsed/phar_features_r.pkl")
        out_path_p = os.path.join(args.root_path, f"USPTO-480k/{args.data_name}_parsed/phar_features_p.pkl")

        df = pd.read_csv(dataset_path, header=None, sep='\t')
        df = df.dropna(axis=0, how='all')

        reactant = df[0].values.tolist()
        product = df[1].values.tolist()

        reactant, product = check_valid(reactant, product)

    elif args.data_type == 'retro':
        out_path_r = os.path.join(args.root_path, f"USPTO-50k/{args.data_name}/phar_features_r.pkl")
        out_path_p = os.path.join(args.root_path, f"USPTO-50k/{args.data_name}/phar_features_p.pkl")

        with open(f'{args.root_path}/USPTO-50k/uspto_50.pickle', 'rb') as f:
            data = pickle.load(f)
        data = [data.iloc[i] for i in range(len(data))]
        data = [d for d in data if d['set'] == args.data_name]
        reactant = [r['reactants_mol'] for r in data]
        product = [p['products_mol'] for p in data]

        reactant = [Chem.MolToSmiles(r, isomericSmiles=False) for r in reactant]
        product = [Chem.MolToSmiles(p, isomericSmiles=False) for p in product]
    elif args.data_type == 'yeild':
        data_x, data_y = reaction_loader(f"{args.root_path}/{args.data_type}/{args.dataset}")
        for (smiles, y) in [[data_x, data_y]]:
            out_path_r = os.path.join(args.root_path, f"{args.data_type}/{args.dataset}/phar_features_r.pkl")
            out_path_p = os.path.join(args.root_path, f"{args.data_type}/{args.dataset}/phar_features_p.pkl")

            reactant = []
            product = []
            for rxn in smiles:
                reactants, products = get_mols(rxn)
                if reactants is not None and products is not None:
                    reactant.append(Chem.MolToSmiles(reactants))
                    product.append(Chem.MolToSmiles(products))

    num_workers = multiprocessing.cpu_count() - 4
    print(f'processing {args.data_name} dataset on {args.root_path}...')
    # multi-processing
    print(f'processing reactant...')
    with Pool(processes=num_workers) as pool:
        r_result = list(tqdm(pool.imap(extract_phar_feature_multi, reactant), total=len(reactant)))
    print(f'processing product...')
    with Pool(processes=num_workers) as pool:
        p_result = list(tqdm(pool.imap(extract_phar_feature_multi, product), total=len(product)))

    with open(out_path_r, 'wb') as file_handle:
        pickle.dump(r_result, file_handle)
    with open(out_path_p, 'wb') as file_handle:
        pickle.dump(p_result, file_handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='yeild')
    parser.add_argument('--dataset', default='ELN')
    parser.add_argument('--data_name', default='train')
    parser.add_argument("--root_path", type=str, default='../../datasets/RXNprediction/')
    args = parser.parse_args()

    main(args)
