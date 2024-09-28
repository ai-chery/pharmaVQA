import numpy as np
import torch
from rdkit import Chem
import dgl
from dgllife.utils.featurizers import ConcatFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, \
    bond_stereo_one_hot, atomic_number_one_hot, atom_degree_one_hot, atom_formal_charge, \
    atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_is_aromatic, atom_total_num_H_one_hot, \
    atom_is_chiral_center, atom_chirality_type_one_hot, atom_mass
from functools import partial
from itertools import permutations
import networkx as nx

INF = 1e6
VIRTUAL_ATOM_INDICATOR = -1
VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1
VIRTUAL_PATH_INDICATOR = -INF

N_ATOM_TYPES = 101
N_BOND_TYPES = 5
bond_featurizer_all = ConcatFeaturizer([  # 14
    partial(bond_type_one_hot, encode_unknown=True),  # 5
    bond_is_conjugated,  # 1
    bond_is_in_ring,  # 1
    partial(bond_stereo_one_hot, encode_unknown=True)  # 7
])
atom_featurizer_all = ConcatFeaturizer([  # 137
    partial(atomic_number_one_hot, encode_unknown=True),  # 101
    partial(atom_degree_one_hot, encode_unknown=True),  # 12
    atom_formal_charge,  # 1
    partial(atom_num_radical_electrons_one_hot, encode_unknown=True),  # 6
    partial(atom_hybridization_one_hot, encode_unknown=True),  # 6
    atom_is_aromatic,  # 1
    partial(atom_total_num_H_one_hot, encode_unknown=True),  # 6
    atom_is_chiral_center,  # 1
    atom_chirality_type_one_hot,  # 2
    atom_mass,  # 1
])


class Vocab(object):
    def __init__(self, n_atom_types, n_bond_types):
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self.vocab = self.construct()

    def construct(self):
        vocab = {}
        # bonded Triplets
        atom_ids = list(range(self.n_atom_types))
        bond_ids = list(range(self.n_bond_types))
        id = 0
        for atom_id_1 in atom_ids:
            vocab[atom_id_1] = {}
            for bond_id in bond_ids:
                vocab[atom_id_1][bond_id] = {}
                for atom_id_2 in atom_ids:
                    if atom_id_2 >= atom_id_1:
                        vocab[atom_id_1][bond_id][atom_id_2] = id
                        id += 1
        for atom_id in atom_ids:
            vocab[atom_id][999] = {}
            vocab[atom_id][999][999] = id
            id += 1
        vocab[999] = {}
        vocab[999][999] = {}
        vocab[999][999][999] = id
        self.vocab_size = id
        return vocab

    def index(self, atom_type1, atom_type2, bond_type):
        atom_type1, atom_type2 = np.sort([atom_type1, atom_type2])
        try:
            return self.vocab[atom_type1][bond_type][atom_type2]
        except Exception as e:
            print(e)
            return self.vocab_size

    def one_hot_feature_index(self, atom_type_one_hot1, atom_type_one_hot2, bond_type_one_hot):
        atom_type1, atom_type2 = np.sort([atom_type_one_hot1.index(1), atom_type_one_hot2.index(1)]).tolist()
        bond_type = bond_type_one_hot.index(1)
        return self.index([atom_type1, bond_type, atom_type2])


def smiles_to_graph(smiles, vocab, max_length=5, n_virtual_nodes=8, add_self_loop=True):
    d_atom_feats = 137
    d_bond_feats = 14
    # Canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
    # Featurize Atoms
    n_atoms = mol.GetNumAtoms()
    atom_features = []

    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_features.append(atom_featurizer_all(atom))
    atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan
    # Construct and Featurize Triplet Nodes
    ## bonded atoms
    triplet_labels = []
    virtual_atom_and_virtual_node_labels = []

    atom_pairs_features_in_triplets = []
    bond_features_in_triplets = []

    bonded_atoms = set()
    triplet_id = 0
    for bond in mol.GetBonds():
        begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
        atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
        bond_feature = bond_featurizer_all(bond)
        bond_features_in_triplets.append(bond_feature)
        bonded_atoms.add(begin_atom_id)
        bonded_atoms.add(end_atom_id)
        triplet_labels.append(vocab.index(atom_features[begin_atom_id][:N_ATOM_TYPES].index(1),
                                          atom_features[end_atom_id][:N_ATOM_TYPES].index(1),
                                          bond_feature[:N_BOND_TYPES].index(1)))
        virtual_atom_and_virtual_node_labels.append(0)
        atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[
            end_atom_id, begin_atom_id] = triplet_id
        triplet_id += 1
    ## unbonded atoms 
    for atom_id in range(n_atoms):
        if atom_id not in bonded_atoms:
            atom_pairs_features_in_triplets.append(
                [atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
            triplet_labels.append(vocab.index(atom_features[atom_id][:N_ATOM_TYPES].index(1), 999, 999))
            virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)
    # Construct and Featurize Paths between Triplets
    ## line graph paths
    edges = []
    paths = []
    line_graph_path_labels = []
    mol_graph_path_labels = []
    virtual_path_labels = []
    self_loop_labels = []
    for i in range(n_atoms):
        node_ids = atomIDPair_to_tripletId[i]
        node_ids = node_ids[~np.isnan(node_ids)]
        if len(node_ids) >= 2:
            new_edges = list(permutations(node_ids, 2))
            edges.extend(new_edges)
            new_paths = [[new_edge[0]] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [new_edge[1]] for new_edge in
                         new_edges]
            paths.extend(new_paths)
            n_new_edges = len(new_edges)
            line_graph_path_labels.extend([1] * n_new_edges)
            mol_graph_path_labels.extend([0] * n_new_edges)
            virtual_path_labels.extend([0] * n_new_edges)
            self_loop_labels.extend([0] * n_new_edges)
    # # molecule graph paths
    adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
    nx_g = nx.from_numpy_array(adj_matrix)
    paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, max_length + 1))
    for i in paths_dict.keys():
        for j in paths_dict[i]:
            path = paths_dict[i][j]
            path_length = len(path)
            if 3 < path_length <= max_length + 1:
                triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]] for pi in range(len(path) - 1)]
                path_start_triplet_id = triplet_ids[0]
                path_end_triplet_id = triplet_ids[-1]
                triplet_path = triplet_ids[1:-1]
                triplet_path = [path_start_triplet_id] + triplet_path + [VIRTUAL_PATH_INDICATOR] * (
                            max_length - len(triplet_path) - 2) + [path_end_triplet_id]
                paths.append(triplet_path)
                edges.append([path_start_triplet_id, path_end_triplet_id])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(1)
                virtual_path_labels.append(0)
                self_loop_labels.append(0)
    for n in range(n_virtual_nodes):
        for i in range(len(atom_pairs_features_in_triplets) - n):
            edges.append([len(atom_pairs_features_in_triplets), i])
            edges.append([i, len(atom_pairs_features_in_triplets)])
            paths.append([len(atom_pairs_features_in_triplets)] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [len(atom_pairs_features_in_triplets)])
            line_graph_path_labels.extend([0, 0])
            mol_graph_path_labels.extend([0, 0])
            virtual_path_labels.extend([n + 1, n + 1])
            self_loop_labels.extend([0, 0])
        atom_pairs_features_in_triplets.append(
            [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
        bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
        triplet_labels.append(vocab.index(999, 999, 999))
        virtual_atom_and_virtual_node_labels.append(n + 1)
    if add_self_loop:
        for i in range(len(atom_pairs_features_in_triplets)):
            edges.append([i, i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i])
            line_graph_path_labels.append(0)
            mol_graph_path_labels.append(0)
            virtual_path_labels.append(0)
            self_loop_labels.append(1)
    edges = np.array(edges, dtype=np.int64)
    data = (edges[:, 0], edges[:, 1])
    g = dgl.graph(data)
    g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
    g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
    g.ndata['label'] = torch.LongTensor(triplet_labels)
    g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
    g.edata['path'] = torch.LongTensor(paths)
    g.edata['lgp'] = torch.BoolTensor(line_graph_path_labels)
    g.edata['mgp'] = torch.BoolTensor(mol_graph_path_labels)
    g.edata['vp'] = torch.BoolTensor(virtual_path_labels)
    g.edata['sl'] = torch.BoolTensor(self_loop_labels)
    return g


def smiles_to_graph_tune(smiles, max_length=5, n_virtual_nodes=8, add_self_loop=True):
    d_atom_feats = 137
    d_bond_feats = 14
    # Canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
    # Featurize Atoms
    n_atoms = mol.GetNumAtoms()
    atom_features = []

    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_features.append(atom_featurizer_all(atom))  # 定义每个atom的feature
    atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan  # 构建一个atom*atom的矩阵
    # Construct and Featurize Triplet Nodes
    ## bonded atoms
    virtual_atom_and_virtual_node_labels = []

    atom_pairs_features_in_triplets = []
    bond_features_in_triplets = []

    bonded_atoms = set()
    triplet_id = 0
    for bond in mol.GetBonds():  # 遍历所有的bonds
        begin_atom_id, end_atom_id = np.sort(
            [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])  # 得到当前bond相关联的两个atom id
        atom_pairs_features_in_triplets.append(
            [atom_features[begin_atom_id], atom_features[end_atom_id]])  # 将这两个atom相应的特征拼接起来，放到atom pair特征中
        bond_feature = bond_featurizer_all(bond)  # 得到当前bond的特征
        bond_features_in_triplets.append(bond_feature)  # 将bond的特征保存下来
        bonded_atoms.add(begin_atom_id)  # 保存计算过特征的atom id
        bonded_atoms.add(end_atom_id)  # 保存计算过特征的atom id
        virtual_atom_and_virtual_node_labels.append(0)  # 添加一个label，表示是否是虚拟的节点
        atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[
            end_atom_id, begin_atom_id] = triplet_id  # 存储当前的节点id，这个id和bond id一样，存储的位置是atom*atom的矩阵
        triplet_id += 1
    ## unbonded atoms 
    for atom_id in range(n_atoms):  # 遍历所有的节点
        if atom_id not in bonded_atoms:  # 如果还有剩下没有遍历的节点（上面遍历边的时候剩下的节点）
            atom_pairs_features_in_triplets.append([atom_features[atom_id], [
                VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])  # 将这个节点和一个虚拟节点特征（全为-1）一起放入atom pair特征中
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)  # 既然是虚拟节点，则添加的边也是虚拟的
            virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)  # 添加虚拟节点对应的label（-1）

    # Construct and Featurize Paths between Triplets #得到path
    ## line graph paths
    edges = []
    paths = []
    line_graph_path_labels = []
    mol_graph_path_labels = []
    virtual_path_labels = []
    self_loop_labels = []
    for i in range(n_atoms):  # 遍历每一个atom
        node_ids = atomIDPair_to_tripletId[i]  # 得到当前atom，与这个atom相关的所有atom
        node_ids = node_ids[~np.isnan(node_ids)]
        if len(node_ids) >= 2:  # 如果当前atom有与别的atom相连
            new_edges = list(permutations(node_ids, 2))  # 构建这两个atom的边
            edges.extend(new_edges)  # 存储这个边
            new_paths = [[new_edge[0]] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [new_edge[1]] for new_edge in
                         new_edges]  # 添加从这个起始点到终点的path，这个path中间通过-10000来填充
            paths.extend(new_paths)  # 存储这个path
            n_new_edges = len(new_edges)  # 得到当前边的个数
            line_graph_path_labels.extend([1] * n_new_edges)
            mol_graph_path_labels.extend([0] * n_new_edges)
            virtual_path_labels.extend([0] * n_new_edges)
            self_loop_labels.extend([0] * n_new_edges)
    # # molecule graph paths
    adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))  # 构建当前节点的关联矩阵
    nx_g = nx.from_numpy_array(adj_matrix)  # 利用当前的关联矩阵构建图
    paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, max_length + 1))  # 得到当前节点与任意节点的路径
    for i in paths_dict.keys():  # 遍历这个路径
        for j in paths_dict[i]:
            path = paths_dict[i][j]
            path_length = len(path)  # 判断当前节点是否存在多个路径
            if 3 < path_length <= max_length + 1:  # 找到存在至少有4个节点的路径
                triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]] for pi in
                               range(len(path) - 1)]  # 找到这个路径上每个节点对应的value，也就是bond的id
                path_start_triplet_id = triplet_ids[0]
                path_end_triplet_id = triplet_ids[-1]
                triplet_path = triplet_ids[1:-1]
                # assert [path_start_triplet_id,path_end_triplet_id] not in edges
                triplet_path = [path_start_triplet_id] + triplet_path + [VIRTUAL_PATH_INDICATOR] * (
                            max_length - len(triplet_path) - 2) + [path_end_triplet_id]
                paths.append(triplet_path)
                edges.append([path_start_triplet_id, path_end_triplet_id])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(1)
                virtual_path_labels.append(0)
                self_loop_labels.append(0)
    for n in range(n_virtual_nodes):
        for i in range(len(atom_pairs_features_in_triplets) - n):
            edges.append([len(atom_pairs_features_in_triplets), i])
            edges.append([i, len(atom_pairs_features_in_triplets)])
            paths.append([len(atom_pairs_features_in_triplets)] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [len(atom_pairs_features_in_triplets)])
            line_graph_path_labels.extend([0, 0])
            mol_graph_path_labels.extend([0, 0])
            virtual_path_labels.extend([n + 1, n + 1])
            self_loop_labels.extend([0, 0])
        atom_pairs_features_in_triplets.append(
            [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
        bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
        virtual_atom_and_virtual_node_labels.append(n + 1)
    if add_self_loop:
        for i in range(len(atom_pairs_features_in_triplets)):
            edges.append([i, i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i])
            line_graph_path_labels.append(0)
            mol_graph_path_labels.append(0)
            virtual_path_labels.append(0)
            self_loop_labels.append(1)
    edges = np.array(edges, dtype=np.int64)
    data = (edges[:, 0], edges[:, 1])
    g = dgl.graph(data)
    g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
    g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
    g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
    g.edata['path'] = torch.LongTensor(paths)
    g.edata['lgp'] = torch.BoolTensor(line_graph_path_labels)
    g.edata['mgp'] = torch.BoolTensor(mol_graph_path_labels)
    g.edata['vp'] = torch.BoolTensor(virtual_path_labels)
    g.edata['sl'] = torch.BoolTensor(self_loop_labels)
    g.edata['edge_index'] = torch.LongTensor(edges)
    return g
