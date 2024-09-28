import dgl
import torch
import numpy as np
from copy import deepcopy

from torch_geometric.data import Batch

from .featurizer import smiles_to_graph
from torch_geometric.utils import add_self_loops
from torch_geometric.loader.dataloader import Collater


def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0], batch_num], axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate([[cs_num[i]] * batch_num_target[i] for i in range(len(cs_num) - 1)], axis=-1)
    return tensor_data + torch.from_numpy(add_factors).reshape(-1, 1)


class Collator_pretrain(object):
    def __init__(
            self,
            vocab,
            max_length, n_virtual_nodes, add_self_loop=True,
            candi_rate=0.15, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1,
            fp_disturb_rate=0.15, md_disturb_rate=0.15
    ):
        self.vocab = vocab
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

        self.candi_rate = candi_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.keep_rate = keep_rate

        self.fp_disturb_rate = fp_disturb_rate
        self.md_disturb_rate = md_disturb_rate

    def bert_mask_nodes(self, g):
        n_nodes = g.number_of_nodes()
        all_ids = np.arange(0, n_nodes, 1, dtype=np.int64)
        valid_ids = torch.where(g.ndata['vavn'] <= 0)[0].numpy()
        valid_labels = g.ndata['label'][valid_ids].numpy()
        probs = np.ones(len(valid_labels)) / len(valid_labels)
        unique_labels = np.unique(np.sort(valid_labels))
        for label in unique_labels:
            label_pos = (valid_labels == label)
            probs[label_pos] = probs[label_pos] / np.sum(label_pos)
        probs = probs / np.sum(probs)
        candi_ids = np.random.choice(valid_ids, size=int(len(valid_ids) * self.candi_rate), replace=False, p=probs)

        mask_ids = np.random.choice(candi_ids, size=int(len(candi_ids) * self.mask_rate), replace=False)

        candi_ids = np.setdiff1d(candi_ids, mask_ids)
        replace_ids = np.random.choice(candi_ids, size=int(len(candi_ids) * (self.replace_rate / (1 - self.keep_rate))),
                                       replace=False)

        keep_ids = np.setdiff1d(candi_ids, replace_ids)

        g.ndata['mask'] = torch.zeros(n_nodes, dtype=torch.long)
        g.ndata['mask'][mask_ids] = 1
        g.ndata['mask'][replace_ids] = 2
        g.ndata['mask'][keep_ids] = 3
        sl_labels = g.ndata['label'][g.ndata['mask'] >= 1].clone()

        # Pre-replace
        new_ids = np.random.choice(valid_ids, size=len(replace_ids), replace=True, p=probs)
        replace_labels = g.ndata['label'][replace_ids].numpy()
        new_labels = g.ndata['label'][new_ids].numpy()
        is_equal = (replace_labels == new_labels)
        while (np.sum(is_equal)):
            new_ids[is_equal] = np.random.choice(valid_ids, size=np.sum(is_equal), replace=True, p=probs)
            new_labels = g.ndata['label'][new_ids].numpy()
            is_equal = (replace_labels == new_labels)
        g.ndata['begin_end'][replace_ids] = g.ndata['begin_end'][new_ids].clone()
        g.ndata['edge'][replace_ids] = g.ndata['edge'][new_ids].clone()
        g.ndata['vavn'][replace_ids] = g.ndata['vavn'][new_ids].clone()
        return sl_labels

    def disturb_fp(self, fp):
        fp = deepcopy(fp)
        b, d = fp.shape
        fp = fp.reshape(-1)
        disturb_ids = np.random.choice(b * d, int(b * d * self.fp_disturb_rate), replace=False)
        fp[disturb_ids] = 1 - fp[disturb_ids]
        return fp.reshape(b, d)

    def disturb_md(self, md):
        md = deepcopy(md)
        b, d = md.shape
        md = md.reshape(-1)
        sampled_ids = np.random.choice(b * d, int(b * d * self.md_disturb_rate), replace=False)
        a = torch.empty(len(sampled_ids)).uniform_(0, 1)
        sampled_md = a
        md[sampled_ids] = sampled_md
        return md.reshape(b, d)

    def __call__(self, samples):
        try:
            smiles_list, fps, mds = map(list, zip(*samples))
            graphs = []
            for smiles in smiles_list:
                graphs.append(
                    smiles_to_graph(smiles, self.vocab, max_length=self.max_length,
                                    n_virtual_nodes=self.n_virtual_nodes,
                                    add_self_loop=self.add_self_loop))
            batched_graph = dgl.batch(graphs)
            mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1)
            fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1)
            batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                       batched_graph.batch_num_edges(),
                                                                       batched_graph.edata['path'][:, :])
            sl_labels = self.bert_mask_nodes(batched_graph)
            disturbed_fps = self.disturb_fp(fps)
            disturbed_mds = self.disturb_md(mds)
            return smiles_list, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds
        except:
            print('collect error')


class Collator_tune(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        smiles_list, graphs, fps, mds, labels = map(list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1)
        labels = torch.stack(labels, dim=0).reshape(len(smiles_list), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])
        return smiles_list, batched_graph, fps, mds, labels


class Collator_pharVQA(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        smiles_list, graphs, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, labels, atom_phar_target_map, random_questions = map(
            list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1)
        labels = torch.stack(labels, dim=0).reshape(len(smiles_list), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)

        return smiles_list, batched_graph, fps, mds, text, text_mask, phar_targets, labels, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions


class Collator_pharVQA_eval(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        smiles_list, graphs, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions = map(
            list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)

        return smiles_list, batched_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions


class Collator_pharVQA_pretrain(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True, vocab=None):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
        self.vocab = vocab

    def __call__(self, samples):
        smiles_list, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions = map(
            list, zip(*samples))
        graphs = []
        for smiles in smiles_list:
            graphs.append(
                smiles_to_graph(smiles, self.vocab, max_length=self.max_length, n_virtual_nodes=self.n_virtual_nodes,
                                add_self_loop=self.add_self_loop))
        batched_graph = dgl.batch(graphs)

        fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)

        return smiles_list, batched_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions


class Collator_pharVQA_rxn(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        input_smiles, output_smiles, graphs, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions = map(
            list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(input_smiles), -1)
        mds = torch.stack(mds, dim=0).reshape(len(input_smiles), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)

        return input_smiles, output_smiles, batched_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions


class Collator_pharVQA_yeild(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        smiless_r, graphs_r, fps_r, mds_r, v_phar_name_r, v_phar_atom_id_r, phar_targets_r, atom_phar_target_map_r, \
            smiless_p, graphs_p, fps_p, mds_p, v_phar_name_p, v_phar_atom_id_p, phar_targets_p, atom_phar_target_map_p, \
            labels, texts, masks, random_questions = map(list, zip(*samples))

        # for d1 batch

        batched_graph_r = dgl.batch(graphs_r)
        fps_r = torch.stack(fps_r, dim=0).reshape(len(smiless_r), -1)
        mds_r = torch.stack(mds_r, dim=0).reshape(len(smiless_r), -1)
        batched_graph_r.edata['path'][:, :] = preprocess_batch_light(batched_graph_r.batch_num_nodes(),
                                                                     batched_graph_r.batch_num_edges(),
                                                                     batched_graph_r.edata['path'][:, :])

        phar_targets_r = torch.stack(phar_targets_r)

        # for d2 batch

        batched_graph_p = dgl.batch(graphs_p)
        fps_p = torch.stack(fps_p, dim=0).reshape(len(smiless_p), -1)
        mds_p = torch.stack(mds_p, dim=0).reshape(len(smiless_p), -1)
        batched_graph_p.edata['path'][:, :] = preprocess_batch_light(batched_graph_p.batch_num_nodes(),
                                                                     batched_graph_p.batch_num_edges(),
                                                                     batched_graph_p.edata['path'][:, :])

        phar_targets_p = torch.stack(phar_targets_p)

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        mask = torch.Tensor([x is not None for x in labels])
        labels = torch.stack(labels, dim=0).reshape(len(smiless_r), -1)

        return (smiless_r, batched_graph_r, fps_r, mds_r, phar_targets_r, atom_phar_target_map_r, v_phar_atom_id_r,
                v_phar_name_r), \
            (smiless_p, batched_graph_p, fps_p, mds_p, phar_targets_p, atom_phar_target_map_p, v_phar_atom_id_p,
             v_phar_name_p), labels, text, mask, text_mask, random_questions


class Collator_pharVQA_dti(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        input_smiles, proteins, protein_lens, labels, graphs, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions = map(
            list, zip(*samples))
        mask = torch.Tensor([x is not None for x in labels])
        batch_proteins = torch.stack(proteins, dim=0).reshape(len(input_smiles), -1)
        batch_protein_lens = torch.stack(protein_lens, dim=0)
        labels = torch.stack(labels, dim=0).reshape(len(input_smiles), -1)
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(input_smiles), -1)
        mds = torch.stack(mds, dim=0).reshape(len(input_smiles), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)

        return input_smiles, batch_proteins, batch_protein_lens, labels, mask, batched_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions


class Collator_pharVQA_dta(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True, follow_batch=None, exclude_keys=None):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
        self.Data_collector = Collater(follow_batch=follow_batch, exclude_keys=exclude_keys)

    def __call__(self, samples):
        input_smiles, proteins, labels, graphs, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions, Data_list = map(
            list, zip(*samples))

        batch_proteins = torch.stack(proteins, dim=0).reshape(len(input_smiles), -1)
        labels = torch.stack(labels, dim=0).reshape(len(input_smiles), -1)
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(input_smiles), -1)
        mds = torch.stack(mds, dim=0).reshape(len(input_smiles), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)
        batch_Data = self.Data_collector(Data_list)
        if self.add_self_loop:
            ## add self loop
            edge_index, _ = add_self_loops(batch_Data.edge_index, num_nodes=batch_Data.x.size(0))
            # add features corresponding to self-loop edges.
            self_loop_attr_dim = batch_Data.edge_attr.shape[1]
            self_loop_attr = torch.zeros(batch_Data.x.size(0), self_loop_attr_dim, dtype=torch.long)
            self_loop_attr[:, 0] = 4  # bond type for self-loop edge
            edge_attr = torch.cat((batch_Data.edge_attr, self_loop_attr), dim=0)
            batch_Data.edge_index = edge_index
            batch_Data.edge_attr = edge_attr

        # batch_Data = Batch.from_data_list(Data_list)
        return input_smiles, batch_proteins, labels, batched_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions, batch_Data


class Collator_pharVQA_dta_CPI(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True, follow_batch=None, exclude_keys=None):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        input_smiles, proteins, protein_lens, labels, graphs, smile_graph, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, atom_phar_target_map, random_questions = map(
            list, zip(*samples))

        batch_proteins = torch.stack(proteins, dim=0)
        labels = torch.tensor(labels).reshape(len(input_smiles), -1)
        protein_lens = torch.tensor(protein_lens).reshape(len(input_smiles), -1)
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(input_smiles), -1)
        mds = torch.stack(mds, dim=0).reshape(len(input_smiles), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])
        batched_smiles_graph = Batch.from_data_list(smile_graph)
        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)

        # batch_Data = Batch.from_data_list(Data_list)
        return input_smiles, batch_proteins, protein_lens, labels, batched_smiles_graph, batched_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions


class Collator_pharVQA_ddi(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        smiless_d1, graphs_1, fps_1, mds_1, v_phar_name_d1, v_phar_atom_id_d1, phar_targets_d1, atom_phar_target_map_d1, smiless_d2, graphs_2, fps_2, mds_2, v_phar_name_d2, v_phar_atom_id_d2, phar_targets_d2, atom_phar_target_map_d2, labels, texts, masks, random_questions = map(
            list, zip(*samples))

        # for d1 batch
        # smiless_d1, graphs_1, fps_1, mds_1, v_phar_name_d1, v_phar_atom_id_d1, phar_targets_d1, atom_phar_target_map_d1 = d1_batch

        batched_graph_d1 = dgl.batch(graphs_1)
        fps_1 = torch.stack(fps_1, dim=0).reshape(len(smiless_d1), -1)
        mds_1 = torch.stack(mds_1, dim=0).reshape(len(smiless_d1), -1)
        batched_graph_d1.edata['path'][:, :] = preprocess_batch_light(batched_graph_d1.batch_num_nodes(),
                                                                      batched_graph_d1.batch_num_edges(),
                                                                      batched_graph_d1.edata['path'][:, :])

        phar_targets_d1 = torch.stack(phar_targets_d1)

        # for d2 batch
        # smiless_d2, graphs_2, fps_2, mds_2, v_phar_name_d2, v_phar_atom_id_d2, phar_targets_d2, atom_phar_target_map_d2 = d2_batch

        batched_graph_d2 = dgl.batch(graphs_2)
        fps_2 = torch.stack(fps_2, dim=0).reshape(len(smiless_d2), -1)
        mds_2 = torch.stack(mds_2, dim=0).reshape(len(smiless_d2), -1)
        batched_graph_d2.edata['path'][:, :] = preprocess_batch_light(batched_graph_d2.batch_num_nodes(),
                                                                      batched_graph_d2.batch_num_edges(),
                                                                      batched_graph_d2.edata['path'][:, :])

        phar_targets_d2 = torch.stack(phar_targets_d2)

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        mask = torch.Tensor([x is not None for x in labels])
        labels = torch.stack(labels, dim=0).reshape(len(smiless_d1), -1)

        return (smiless_d1, batched_graph_d1, fps_1, mds_1, phar_targets_d1, atom_phar_target_map_d1, v_phar_atom_id_d1,
                v_phar_name_d1), \
            (smiless_d2, batched_graph_d2, fps_2, mds_2, phar_targets_d2, atom_phar_target_map_d2, v_phar_atom_id_d2,
             v_phar_name_d2), labels, text, mask, text_mask, random_questions


class Collator_pharVQA_retroCap(object):
    def __init__(self, src_pad, tgt_pad, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad

    def __call__(self, data):
        """Build mini-batch tensors:
        :param sep: (int) index of src seperator
        :param pads: (tuple) index of src and tgt padding
        """
        # Sort a data list by caption length
        # data.sort(key=lambda x: len(x[0]), reverse=True)
        p_smiles, r_smiles, src, tgt, alignment, graphs, fps, mds, v_phar_name, v_phar_atom_id, texts, masks, phar_targets, \
            atom_phar_target_map, random_questions = zip(*data)
        max_src_length = max([len(t) for t in src])
        max_tgt_length = max([len(t) for t in tgt])

        pg_num = [i.number_of_nodes() for i in graphs]
        pg_edge_num = [i.number_of_edges() for i in graphs]

        anchor = torch.zeros([])

        # Pad_sequence
        new_src = anchor.new_full((max_src_length, len(data)), self.src_pad, dtype=torch.long)
        new_tgt = anchor.new_full((max_tgt_length, len(data)), self.tgt_pad, dtype=torch.long)
        new_alignment = anchor.new_zeros((len(data), max_tgt_length - 1, max_src_length), dtype=torch.float)

        for i in range(len(data)):
            new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
            new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
            new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(data), -1)
        mds = torch.stack(mds, dim=0).reshape(len(data), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        text = torch.stack(texts, dim=0)
        text_mask = torch.stack(masks, dim=0)
        phar_targets = torch.stack(phar_targets)

        return p_smiles, r_smiles, new_src, new_tgt, new_alignment, (pg_num, pg_edge_num), \
            batched_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions
