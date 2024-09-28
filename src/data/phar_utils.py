import os
from threading import Timer

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
import tqdm

# 以下代码部分熟悉BaseFeatures.fdef
# 读取及熟悉rdkit内置药效团文件
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
Feature_family_name = list(factory.GetFeatureDefs().keys())


# Feature_family_name = list(factory.GetFeatureFamilies()  )

# # load chem dataset
# chem_name = 'clintox'
# data_path = os.path.join('../data/chem_datasets/', f'{chem_name}/{chem_name}.csv')
# df_datas = pd.read_csv(data_path)
#
# datas = df_datas['smiles']
from multiprocessing import Pool

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



def extract_phar_feature(datas,task):
    datas_feat, datas_index = [], []
    all_fea_num = 0
    phar_target = []
    for data in (datas):
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

            if task == feat.GetFamily():
                phar_targets.append(1)
            else:
                phar_targets.append(0)

        phar_target.append(sum(phar_targets))
        all_fea_num += len(feats)
        datas_feat.append(fp_name)

    return datas_feat, phar_target

