import os
from copy import deepcopy

import torch
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import ChemicalFeatures
from tqdm import tqdm
from rdkit import RDConfig
from transformers import AutoTokenizer

from src.trainer.evaluator import Evaluator

phar_Dict = {'Donor': 0, 'Acceptor': 1, 'NegIonizable': 2, 'PosIonizable': 3, 'Aromatic': 4, 'Hydrophobe': 5,
             'LumpedHydrophobe': 6}


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, phar_loss_fn, align_loss_fn, evaluator, phar_evaluator,
                 result_tracker, summary_writer, device, model_name, label_mean=None, label_std=None, ddp=False,
                 local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.phar_loss_fn = phar_loss_fn
        self.align_loss_fn = align_loss_fn
        self.evaluator = evaluator
        self.phar_evaluator = phar_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')
        self.save_path = args.save_path

    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, text, text_mask, phar_targets, labels, atom_phar_target_map, v_phar_atom_id,
         v_phar_name, random_questions) = batched_data
        ecfp, md, g, labels, text, text_mask, phar_targets = ecfp.to(self.device), md.to(self.device), g.to(
            self.device), labels.to(self.device), text.to(self.device), text_mask.to(self.device), phar_targets.to(
            self.device)
        # print(torch.cuda.is_available())
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune_kpgt(g, ecfp, md)
        return predictions, labels

    def _forward_epoch_pharVQA(self, prompt_model, batched_data, eval=False, epoch=0, batch_index=0):
        (smiles, g, ecfp, md, text, text_mask, phar_targets, labels, atom_phar_target_map, v_phar_atom_id,
         v_phar_name, random_questions) = batched_data
        ecfp, md, g, labels, text, text_mask, phar_targets = ecfp.to(self.device), md.to(self.device), g.to(
            self.device), labels.to(self.device), text.to(self.device), text_mask.to(self.device), phar_targets.to(
            self.device)
        g_1 = deepcopy(g)
        batch_size = batched_data[1].batch_size

        molecules_phar_prompt, atten = prompt_model.forward_tune(g, ecfp, md, text, text_mask)
        molecule_repr = prompt_model.get_graph_feat(g_1, ecfp, md)

        # for prompt_layer in prompt_model.prompt_aggregation_model:
        #     molecules_phar_prompt, atten_prompt = prompt_layer(molecule_repr, molecule_repr, molecules_phar_prompt)

        # molecules_phar_prompt, atten_prompt = prompt_model.prompt_aggregation_model(molecules_phar_prompt,
        #                                                                             molecules_phar_prompt)
        # molecules_phar_prompt = molecules_phar_prompt.transpose(-1, -2)
        # molecules_phar_prompt = molecules_phar_prompt.mean(dim=1)
        molecules_prompt = prompt_model.prompt_projection_model(molecules_phar_prompt.reshape(batch_size, -1))
        if self.args.prompt_mode == 'add':
            molecules_prompt = molecules_prompt + molecule_repr
        elif self.args.prompt_mode == 'cat':
            molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
        pred = prompt_model.predictor(molecules_prompt)

        pred_phar_num = prompt_model.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(-1)
        phar_y = phar_targets.to(torch.float64)

        if self.args.noise_question == 'True':
            phar_target_mx = torch.zeros((batch_size, atten[0].size()[-2], len(atten))).to(self.device)
        else:
            phar_target_mx = torch.zeros((batch_size, atten[0].size()[-2], len(atten))).to(self.device)
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            for mol_index, mol in enumerate(mols):
                random_question = random_questions[mol_index]
                mol_phar_mx = atom_phar_target_map[mol_index]
                for bond_index, bond in enumerate(mol.GetBonds()):
                    begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    for question in random_question:
                        phar_target_mx[mol_index, bond_index, phar_Dict[question]] = mol_phar_mx[
                            [begin_atom_id, end_atom_id], phar_Dict[question]].sum(dim=0).to(
                            torch.bool).to(torch.float32)

        if eval is False:
            atten = torch.stack(atten, dim=2).sum(dim=-1)
            return pred, labels, pred_phar_num, phar_y, atten, phar_target_mx
        else:
            phar_target_mx = torch.zeros((batch_size, atten[0].size()[-2], len(atten)))
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            batch_bond_atom_map = []
            for mol_index, mol in enumerate(mols):
                random_question = random_questions[mol_index]
                mol_phar_mx = atom_phar_target_map[mol_index]
                bond_atom_map = torch.zeros(len(mol.GetBonds())).tolist()
                for bond_index, bond in enumerate(mol.GetBonds()):
                    begin_atom_id, end_atom_id = np.sort(
                        [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    # phar_target_mx[mol_index, bond_index] = mol_phar_mx[
                    #     [begin_atom_id, end_atom_id], phar_Dict[random_question[0]]].sum(dim=0).to(
                    #     torch.bool).to(torch.float32)
                    bond_atom_map[bond_index] = [begin_atom_id, end_atom_id]
                    for question in random_question:
                        phar_target_mx[mol_index, bond_index, phar_Dict[question]] += mol_phar_mx[
                            [begin_atom_id, end_atom_id], phar_Dict[question]].sum(dim=0).to(
                            torch.bool).to(torch.float32)
                batch_bond_atom_map.append(bond_atom_map)
            atten = torch.stack(atten).transpose(0, 1)
            for text_idx, text_token in enumerate(text):
                random_question = random_questions[text_idx]
                mol_phar_mx = atom_phar_target_map[text_idx]
                bond_atom_map = batch_bond_atom_map[text_idx]
                for question_index in range(atten.size()[1]):
                    question_name = random_question[question_index]
                    text_token_mask = text_mask[text_idx, question_index]
                    q_text_VQAindex = text_token[question_index, :text_token_mask.count_nonzero()]
                    q_text_VQA = self.tokenizer.convert_ids_to_tokens(q_text_VQAindex)

                    value_mt = atten[text_idx, question_index, :mols[text_idx].GetNumBonds(),
                               :text_token_mask.count_nonzero()]
                    target_mt = phar_target_mx[text_idx, :mols[text_idx].GetNumBonds(),
                                question_index].reshape(-1, 1).to(torch.bool)
                    min_value = value_mt.min()
                    max_value = value_mt.max()
                    norm_value_mt = (value_mt - min_value) / (max_value - min_value)

                    norm_min_value = 0
                    norm_max_value = 1
                    target_tensor = norm_min_value * torch.ones_like(target_mt, dtype=torch.float)
                    target_tensor[target_mt] = norm_max_value
                    # tensor_np = torch.cat((norm_value_mt, target_tensor), dim=-1).numpy()
                    # plt.figure(figsize=(20, 20))
                    # ax = sns.heatmap(tensor_np, cmap='coolwarm', linewidths=0.1, linecolor='white')
                    # ax.set_xticklabels(q_text_VQA + ['label'], rotation=30, fontsize=16)
                    # ax.set_yticklabels(range(mols[text_idx].GetNumBonds()), rotation=0, fontsize=16)
                    if not os.path.exists(
                            f"{self.args.save_path}/attention_map_softmax_{self.args.softmax}/{question_name}"):
                        os.makedirs(f"{self.args.save_path}/attention_map_softmax_{self.args.softmax}/{question_name}")
                    # plt.savefig(
                    #     f'{args.save_path}/attention_map_softmax_{args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_hotmap.png')
                    # plt.close()

                    mol = mols[text_idx]
                    for atom in mol.GetAtoms():
                        atom.SetAtomMapNum(atom.GetIdx())
                    target_idx = mol_phar_mx[:, question_index]

                    target_atom_num = target_idx.nonzero(as_tuple=True)[0].tolist()
                    bond_atom_list_all = []
                    for target_atom_index in target_atom_num:
                        bond_atom_list = []
                        for bond_index, bond in enumerate(target_tensor.nonzero()[:, 0]):
                            bond_atom_index = bond_atom_map[bond]
                            if target_atom_index in bond_atom_index:
                                bond_atom_list.append(norm_value_mt[bond])
                        #
                        if len(bond_atom_list) == 1:
                            bond_atom_tensor = torch.stack(bond_atom_list).mean(dim=0)
                        else:
                            bond_atom_tensor = torch.stack(bond_atom_list).mean(dim=0)
                        bond_atom_list_all.append(bond_atom_tensor)

                    if len(target_idx.nonzero().tolist()) == 0:
                        image = Chem.Draw.MolToImage(mol, size=(300, 300), kekulize=True)
                    else:
                        image = Chem.Draw.MolToImage(mol, size=(300, 300), kekulize=True,
                                                     highlightAtoms=target_idx.nonzero(as_tuple=True)[
                                                         0].tolist())
                    image.save(
                        f'{self.args.save_path}/attention_map_softmax_{self.args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_mol.png')

                    if len(bond_atom_list_all) == 0:
                        continue
                    else:
                        sort_texts = torch.stack(bond_atom_list_all).argsort(dim=1)
                        text_rank = []
                        for index, sort_text in enumerate(sort_texts):
                            text_rank.append([q_text_VQA[i] for i in sort_text])
                        with open(
                                f'{self.args.save_path}/attention_map_softmax_{self.args.softmax}/{question_name}/molecules_{batch_index}_{text_idx}_mol.txt',
                                'w', encoding='utf-8') as file:
                            for data_index, data_slice in enumerate(text_rank):
                                bond_num_2_atom_num = target_atom_num[data_index]
                                list_str = ', '.join(data_slice)
                                file.write('atom number' + str(bond_num_2_atom_num) + ': \t' + list_str + '\n')

            return pred, labels, pred_phar_num, phar_y, phar_target_mx, atten

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        epoch_loss = 0
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            predictions, labels, pharnum_predictions, pharnum_labels, predict_mat, label_mat = self._forward_epoch_pharVQA(
                model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std
            pre_loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            if self.args.noise_question == 'True':
                phar_loss = 0
                align_loss = 0
            else:
                phar_loss = self.phar_loss_fn(pharnum_predictions, pharnum_labels).mean()
                align_loss = self.align_loss_fn(predict_mat.view(predict_mat.size()[0], -1),
                                                label_mat.view(label_mat.size()[0], -1))
            # align_loss = torch.mean((predict_mat - label_mat) ** 2)
            loss = pre_loss + phar_loss + align_loss
            epoch_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/prediction_Loss/train', pre_loss,
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)
                self.summary_writer.add_scalar('Loss/phar_Loss/train', phar_loss,
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)
                self.summary_writer.add_scalar('Loss/alignment_Loss/train', align_loss,
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)
                self.summary_writer.add_scalar('Loss/total_Loss/train', loss,
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Loss/epoch_Loss/train', epoch_loss, epoch_idx)

    def train_epoch_kpgt(self, model, train_loader, epoch_idx):
        model.train()
        epoch_loss = 0
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std
            pre_loss = (self.loss_fn(predictions, labels) * is_labeled).mean()

            # align_loss = torch.mean((predict_mat - label_mat) ** 2)
            loss = pre_loss
            epoch_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/prediction_Loss/train', pre_loss,
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)
                self.summary_writer.add_scalar('Loss/total_Loss/train', loss,
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Loss/epoch_Loss/train', epoch_loss, epoch_idx)

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result, best_test_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_train_pharNum_result, best_val_pharNum_result, best_test_pharNum_result = 999, 999, 999
        best_epoch = 0

        for epoch in tqdm(range(1, self.args.n_epochs + 1)):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            if self.args.train_kpgt == 'True':
                self.train_epoch_kpgt(model, train_loader, epoch)
            else:
                self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                if val_loader == None:
                    if self.args.train_kpgt == 'True':
                        train_result, train_phar_result, _, _ = self.eval_kpgt(model, train_loader, epoch)
                    else:
                        train_result, train_phar_result, _, _ = self.eval(model, train_loader, epoch)
                    val_result, val_phar_result = 0, 0
                    ref_result = train_result
                    best_ref_result = best_train_result
                else:
                    if self.args.train_kpgt == 'True':
                        val_result, val_phar_result, _, _ = self.eval_kpgt(model, val_loader, epoch)
                    else:
                        val_result, val_phar_result, _, _ = self.eval(model, val_loader, epoch)
                    train_result, train_phar_result = 0, 0
                    ref_result = val_result
                    best_ref_result = best_val_result

                if test_loader == None:
                    test_result, test_phar_result = 0, 0
                else:
                    if self.args.train_kpgt == 'True':
                        test_result, test_phar_result, lables_all, predictions_all = self.eval_kpgt(model, test_loader,
                                                                                                    epoch)
                    else:
                        test_result, test_phar_result, lables_all, predictions_all = self.eval(model, test_loader,
                                                                                               epoch)

                if self.result_tracker.update(np.mean(best_ref_result), np.mean(ref_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_train_pharNum_result = train_phar_result
                    best_val_pharNum_result = val_phar_result
                    best_test_pharNum_result = test_phar_result

                    best_epoch = epoch

                if self.summary_writer is not None:
                    if self.args.metric == 'spear,pear':
                        for index, metric in enumerate(['spear', 'pear']):
                            # self.summary_writer.add_scalar(f'Results/train_{metric}', train_result[index], epoch)
                            self.summary_writer.add_scalar(f'Results/val_{metric}', val_result[index], epoch)
                            self.summary_writer.add_scalar(f'Results/test_{metric}', test_result[index], epoch)

                    else:
                        # self.summary_writer.add_scalar(f'Results/train_{self.args.metric}', train_result, epoch)
                        self.summary_writer.add_scalar(f'Results/val_{self.args.metric}', val_result, epoch)
                        self.summary_writer.add_scalar(f'Results/test_{self.args.metric}', test_result, epoch)

                    self.summary_writer.add_scalar('Results/train_pharNum', train_phar_result, epoch)
                    self.summary_writer.add_scalar('Results/val_pharNum', val_phar_result, epoch)
                    self.summary_writer.add_scalar('Results/test_pharNum', test_phar_result, epoch)

                    self.summary_writer.add_scalar('Results/best_epoch', best_epoch, epoch)

                if epoch - best_epoch >= 20:
                    if (self.label_mean is not None) and (self.label_std is not None):
                        predictions_all = predictions_all * self.label_std.detach().cpu() + self.label_mean.detach().cpu()
                    np.savetxt(f'{self.args.save_path}/Results_epoch_{epoch}.txt',
                               np.concatenate((lables_all.numpy(), predictions_all.numpy()), axis=1),
                               fmt='%.4f')
                    self.save_model(model)
                    break
        self.save_model(model)
        return best_train_result, best_val_result, best_test_result, best_train_pharNum_result, best_val_pharNum_result, best_test_pharNum_result

    def eval(self, model, dataloader, epoch, eval=False):
        model.eval()
        predictions_all = []
        labels_all = []
        pharnum_predictions_all = []
        pharnum_labels_all = []
        phar_target_mx_all = []
        for batch_index, batched_data in enumerate(tqdm(dataloader)):
            predictions, labels, pharnum_predictions, pharnum_labels, predict_mat, label_mat = self._forward_epoch_pharVQA(
                model, batched_data, eval=eval, epoch=epoch, batch_index=batch_index)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
            pharnum_predictions_all.append(pharnum_predictions.detach().cpu())
            pharnum_labels_all.append(pharnum_labels.detach().cpu())

        if isinstance(self.evaluator, list):
            result = []
            for evaluator in self.evaluator:
                result.append(evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all)))
        else:
            result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))

        phar_result = self.phar_evaluator.eval(torch.cat(pharnum_labels_all), torch.cat(pharnum_predictions_all))
        return result, phar_result, torch.cat(labels_all), torch.cat(predictions_all)

    def eval_kpgt(self, model, dataloader, epoch):
        model.eval()
        predictions_all = []
        labels_all = []
        for batch_index, batched_data in enumerate(dataloader):
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())

        if isinstance(self.evaluator, list):
            result = []
            for evaluator in self.evaluator:
                result.append(evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all)))
        else:
            result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))

        return result, 0, torch.cat(labels_all), torch.cat(predictions_all)

    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path + f"/best_model.pth")


class Trainer_kpgt():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, phar_loss_fn, align_loss_fn, evaluator, phar_evaluator,
                 result_tracker, summary_writer, device_id, model_name, label_mean=None, label_std=None, ddp=False,
                 local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.phar_loss_fn = phar_loss_fn
        self.align_loss_fn = align_loss_fn
        self.evaluator = evaluator
        self.phar_evaluator = phar_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device_id
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.save_path = args.save_path

    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        # print(torch.cuda.is_available())
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune(g, ecfp, md)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        try:
            model.train()
            epoch_loss = 0
            for batch_idx, batched_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                predictions, labels = self._forward_epoch(model, batched_data)
                is_labeled = (~torch.isnan(labels)).to(torch.float32)
                labels = torch.nan_to_num(labels)
                if (self.label_mean is not None) and (self.label_std is not None):
                    labels = (labels - self.label_mean) / self.label_std
                pre_loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
                # phar_loss = 0
                # align_loss = torch.mean((predict_mat - label_mat) ** 2)
                loss = pre_loss
                epoch_loss += loss
                # try:
                #     loss.backward()
                # except:
                #     print('backprop failed')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.lr_scheduler.step()
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('Loss/total_Loss/train', loss,
                                                   (epoch_idx - 1) * len(train_loader) + batch_idx + 1)

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/epoch_Loss/train', epoch_loss, epoch_idx)
        except:
            print('Exception during training')

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result, best_test_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_epoch = 0

        for epoch in tqdm(range(1, self.args.n_epochs + 1)):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            try:
                self.train_epoch(model, train_loader, epoch)
            except:
                print('training failed')

            # self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:

                if val_loader == None:
                    try:
                        train_result = self.eval(model, train_loader, epoch)
                    except:
                        print('eval train error')
                    val_result, val_phar_result = 0, 0
                    ref_result = train_result
                    best_ref_result = best_train_result
                else:
                    try:
                        val_result = self.eval(model, val_loader, epoch)
                    except:
                        print('eval val error')
                    train_result, train_phar_result = 0, 0
                    ref_result = val_result
                    best_ref_result = best_val_result

                if test_loader == None:
                    test_result, test_phar_result = 0, 0
                else:
                    try:
                        test_result = self.eval(model, test_loader, epoch)
                    except:
                        print('eval test error')
                if self.result_tracker.update(np.mean(best_ref_result), np.mean(ref_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result

                    best_epoch = epoch
                if self.summary_writer is not None:
                    if self.args.metric == 'spear,pear':
                        for index, metric in enumerate(['spear', 'pear']):
                            # self.summary_writer.add_scalar(f'Results/train_{metric}', train_result[index], epoch)
                            self.summary_writer.add_scalar(f'Results/val_{metric}', val_result[index], epoch)
                            self.summary_writer.add_scalar(f'Results/test_{metric}', test_result[index], epoch)

                    else:
                        # self.summary_writer.add_scalar(f'Results/train_{self.args.metric}', train_result, epoch)
                        self.summary_writer.add_scalar(f'Results/val_{self.args.metric}', val_result, epoch)
                        self.summary_writer.add_scalar(f'Results/test_{self.args.metric}', test_result, epoch)
                    # self.summary_writer.add_scalar('Results/train', train_result, epoch)
                    # self.summary_writer.add_scalar('Results/val', val_result, epoch)
                    # self.summary_writer.add_scalar('Results/test', test_result, epoch)
                    self.summary_writer.add_scalar('Results/best_epoch', best_epoch, epoch)

                if epoch - best_epoch >= 20:
                    self.save_model(model)
                    break

        self.save_model(model)
        return best_train_result, best_val_result, best_test_result

    def eval(self, model, dataloader, epoch):
        model.eval()
        predictions_all = []
        labels_all = []
        for batch_index, batched_data in enumerate(dataloader):
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())

        if isinstance(self.evaluator, list):
            result = []
            for evaluator in self.evaluator:
                result.append(evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all)))
        else:
            result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))

        return result

    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path + f"/best_model.pth")
