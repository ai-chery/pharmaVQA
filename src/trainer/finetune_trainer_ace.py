import os
from copy import deepcopy

import torch
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from tqdm import tqdm
from rdkit import RDConfig
from transformers import AutoTokenizer

phar_Dict = {'Donor': 0, 'Acceptor': 1, 'NegIonizable': 2, 'PosIonizable': 3, 'Aromatic': 4, 'Hydrophobe': 5,
             'LumpedHydrophobe': 6}


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, phar_loss_fn, align_loss_fn, evaluator, phar_evaluator,
                 result_tracker, summary_writer, device_id, model_name, r2_evaluator=None, label_mean=None,
                 label_std=None, ddp=False,
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
        # self.tokenizer = AutoTokenizer.from_pretrained('../pretrained/scibert_scivocab_uncased')
        self.save_path = args.save_path
        self.r2_evaluator = r2_evaluator

    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, text, text_mask, phar_targets, labels) = batched_data
        # print(torch.cuda.is_available())
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune(g, ecfp, md)
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
        molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
        pred = prompt_model.predictor(molecules_prompt)

        pred_phar_num = prompt_model.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(-1)
        phar_y = phar_targets.to(torch.float64)

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
        atten = torch.stack(atten, dim=2).sum(dim=-1)

        if eval is False:
            return pred, labels, pred_phar_num, phar_y, atten, phar_target_mx
        else:
            phar_target_mx = torch.zeros((batch_size, atten[0].size()[-2], len(atten)))
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            for mol_index, mol in enumerate(mols):
                random_question = random_questions[mol_index]
                mol_phar_mx = atom_phar_target_map[mol_index]
                for bond_index, bond in enumerate(mol.GetBonds()):
                    begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    phar_target_mx[mol_index, bond_index] = mol_phar_mx[
                        [begin_atom_id, end_atom_id], phar_Dict[random_question[0]]].sum(dim=0).to(
                        torch.bool).to(torch.float32)
            atten = torch.stack(atten).transpose(0, 1)
            tokened_VQAtext_all = []
            for text_idx, text_token in enumerate(text):
                tokened_VQAtext = []
                for question_index in range(atten.size()[1]):
                    text_token_mask = text_mask[text_idx, question_index]
                    q_text_VQAindex = text_token[question_index, :text_token_mask.count_nonzero()]
                    q_text_VQA = self.tokenizer.convert_ids_to_tokens(q_text_VQAindex)

                    # if epoch % 10 == 0:
                    #     tensor_np = torch.cat((atten[text_idx, question_index, :mols[text_idx].GetNumBonds(),
                    #                            :text_token_mask.count_nonzero()].detach().cpu(),
                    #                            phar_target_mx[text_idx, :mols[text_idx].GetNumBonds()]),
                    #                           dim=-1).numpy()
                    #     fig = plt.figure(figsize=(30, 30))
                    #     plt.imshow(tensor_np, cmap='jet')
                    #     plt.xticks(range(len(q_text_VQA)), q_text_VQA)
                    #     plt.colorbar()
                    #     plt.savefig(
                    #         f'{self.save_path}/molecules_{batch_index}_{text_idx}_question_{question_index}_hotmap_epoch_{epoch}.png')
                    #     plt.close()

                    tokened_VQAtext.append(q_text_VQA)
                tokened_VQAtext_all.append(tokened_VQAtext)

            return pred, labels, pred_phar_num, phar_y, phar_target_mx, atten, tokened_VQAtext_all

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
            # phar_loss = 0
            phar_loss = self.phar_loss_fn(pharnum_predictions, pharnum_labels).mean()
            align_loss = self.align_loss_fn(predict_mat.view(predict_mat.size()[0], -1),
                                            label_mat.view(label_mat.size()[0], -1))
            # align_loss = torch.mean((predict_mat - label_mat) ** 2)
            loss = pre_loss + self.args.alpha * phar_loss + self.args.beta * align_loss
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

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result, best_test_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_epoch = 0

        for epoch in tqdm(range(1, self.args.n_epochs + 1)):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:

                if val_loader == None:
                    train_result, train_phar_result, train_r2_result, _, _ = self.eval(model, train_loader, epoch)
                    val_result, val_phar_result, val_r2_result = 0, 0, 0
                    ref_result = train_result
                    best_ref_result = best_train_result
                else:
                    val_result, val_phar_result, val_r2_result, _, _ = self.eval(model, val_loader, epoch)
                    train_result, train_phar_result, train_r2_result = 0, 0, 0
                    ref_result = val_result
                    best_ref_result = best_val_result

                if test_loader == None:
                    test_result, test_phar_result, test_r2_result, lables_all, predictions_all = 0, 0, 0, 0, 0
                else:
                    test_result, test_phar_result, test_r2_result, lables_all, predictions_all = self.eval(model,
                                                                                                           test_loader,
                                                                                                           epoch)

                if self.result_tracker.update(np.mean(best_ref_result), np.mean(ref_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result

                    best_train_pharNum_result = train_phar_result
                    best_val_pharNum_result = val_phar_result
                    best_test_pharNum_result = test_phar_result

                    best_train_r2_result = train_r2_result
                    best_val_r2_result = val_r2_result
                    best_test_r2_result = test_r2_result

                    best_epoch = epoch
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('Results/train', train_result, epoch)
                    self.summary_writer.add_scalar('Results/val', val_result, epoch)
                    self.summary_writer.add_scalar('Results/test', test_result, epoch)

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
        return best_train_result, best_val_result, best_test_result, \
            best_train_pharNum_result, best_val_pharNum_result, best_test_pharNum_result, \
            best_train_r2_result, best_val_r2_result, best_test_r2_result

    def eval(self, model, dataloader, epoch):
        model.eval()
        predictions_all = []
        labels_all = []
        pharnum_predictions_all = []
        pharnum_labels_all = []
        phar_target_mx_all = []
        for batch_index, batched_data in enumerate(dataloader):
            predictions, labels, pharnum_predictions, pharnum_labels, predict_mat, label_mat = self._forward_epoch_pharVQA(
                model, batched_data, eval=False, epoch=epoch, batch_index=batch_index)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
            pharnum_predictions_all.append(pharnum_predictions.detach().cpu())
            pharnum_labels_all.append(pharnum_labels.detach().cpu())

        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        r2_result = self.r2_evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))

        phar_result = self.phar_evaluator.eval(torch.cat(pharnum_labels_all), torch.cat(pharnum_predictions_all))
        return result, phar_result, r2_result, torch.cat(labels_all), torch.cat(predictions_all)
    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path + f"/best_model.pth")


class Trainer_kpgt():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator,
                 result_tracker, summary_writer, device_id, model_name, r2_evaluator=None, label_mean=None,
                 label_std=None, ddp=False,
                 local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device_id
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.save_path = args.save_path
        self.r2_evaluator = r2_evaluator

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/total_Loss/train', loss,
                                               (epoch_idx - 1) * len(train_loader) + batch_idx + 1)

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Loss/epoch_Loss/train', epoch_loss, epoch_idx)

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result, best_test_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_epoch = 0

        for epoch in tqdm(range(1, self.args.n_epochs + 1)):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                if val_loader == None:
                    train_result, train_r2_result = self.eval(model, train_loader, epoch)
                    val_result, val_phar_result, val_r2_result = 0, 0, 0
                    ref_result = train_result
                    best_ref_result = best_train_result
                else:
                    val_result, val_r2_result = self.eval(model, val_loader, epoch)
                    train_result, train_phar_result, train_r2_result = 0, 0, 0
                    ref_result = val_result
                    best_ref_result = best_val_result

                if test_loader == None:
                    test_result, test_phar_result, test_r2_result = 0, 0, 0
                else:
                    test_result, test_r2_result = self.eval(model, test_loader, epoch)

                if self.result_tracker.update(np.mean(best_ref_result), np.mean(ref_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result

                    best_train_r2_result = train_r2_result
                    best_val_r2_result = val_r2_result
                    best_test_r2_result = test_r2_result

                    best_epoch = epoch
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('Results/train', train_result, epoch)
                    self.summary_writer.add_scalar('Results/val', val_result, epoch)
                    self.summary_writer.add_scalar('Results/test', test_result, epoch)

                    self.summary_writer.add_scalar('Results/train_r2', train_r2_result, epoch)
                    self.summary_writer.add_scalar('Results/val_r2', val_r2_result, epoch)
                    self.summary_writer.add_scalar('Results/test_r2', test_r2_result, epoch)

                    self.summary_writer.add_scalar('Results/best_epoch', best_epoch, epoch)

                if epoch - best_epoch >= 20:
                    break
        return best_train_result, best_val_result, best_test_result, \
            best_train_r2_result, best_val_r2_result, best_test_r2_result

    def eval(self, model, dataloader, epoch):
        model.eval()
        predictions_all = []
        labels_all = []
        for batch_index, batched_data in enumerate(dataloader):
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())

        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        r2_result = self.r2_evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        return result, r2_result
