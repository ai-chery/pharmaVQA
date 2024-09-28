from copy import deepcopy

import torch
import numpy as np
from rdkit import Chem
from sklearn.metrics import f1_score
from tqdm import tqdm

phar_Dict = {'Donor': 0, 'Acceptor': 1, 'NegIonizable': 2, 'PosIonizable': 3, 'Aromatic': 4, 'Hydrophobe': 5,
             'LumpedHydrophobe': 6}


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, phar_loss_fn, reg_evaluator,
                 result_tracker, summary_writer, device, ddp=False, local_rank=1):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.phar_loss_fn = phar_loss_fn
        self.reg_evaluator = reg_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0

    def _forward_epoch(self, model, batched_data):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        sl_labels = sl_labels.to(self.device)
        disturbed_fps = disturbed_fps.to(self.device)
        disturbed_mds = disturbed_mds.to(self.device)
        sl_predictions, fp_predictions, md_predictions = model(batched_graph, disturbed_fps, disturbed_mds)
        mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask'] >= 1].cpu().numpy()
        return mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds

    def _forward_epoch_pharVQA(self, prompt_model, batched_data, eval=False, epoch=0, batch_index=0):
        (smiles, g, ecfp, md, text, text_mask, phar_targets, atom_phar_target_map, v_phar_atom_id,
         v_phar_name, random_questions) = batched_data
        ecfp, md, g, text, text_mask, phar_targets = ecfp.to(self.device), md.to(self.device), g.to(
            self.device), text.to(self.device), text_mask.to(self.device), phar_targets.to(self.device)
        # g_1 = deepcopy(g)
        # batch_size = batched_data[1].batch_size

        molecules_phar_prompt, atten = prompt_model.module.forward_tune(g, ecfp, md, text, text_mask)
        # molecule_repr = prompt_model.get_graph_feat(g_1, ecfp, md)
        # molecules_prompt = prompt_model.prompt_projection_model(molecules_phar_prompt.reshape(batch_size, -1))
        # molecules_prompt = torch.cat((molecules_prompt, molecule_repr), dim=-1)
        # pred = prompt_model.predictor(molecules_prompt)
        # pred = labels
        pred_phar_num = prompt_model.module.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(-1)
        phar_y = phar_targets.to(torch.float64)
        return pred_phar_num, phar_y

    def train_epoch(self, model, train_loader, epoch, runs):
        model.train()
        epoch_loss = 0
        for batch_idx, batched_data in enumerate(train_loader):
            try:
                self.optimizer.zero_grad()
                pred_phar_num, phar_y = self._forward_epoch_pharVQA(model, batched_data)
                phar_loss = self.phar_loss_fn(pred_phar_num, phar_y).mean()
                loss = phar_loss
                epoch_loss += loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.n_updates += 1
                self.lr_scheduler.step()
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('Loss/loss_tot', loss.item(), self.n_updates)

                if self.n_updates == self.args.n_steps:
                    if self.local_rank == 0:
                        self.save_model(model)
                    break
            except Exception as e:
                print(e)
            else:
                continue

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Loss/epoch_mean_loss', epoch_loss.item()/len(train_loader), self.n_updates)
        return epoch_loss

    def train_runs(self, model, train_loader, epoch, runs):
        model.train()
        epoch_loss = 0
        for batch_idx, batched_data in enumerate(train_loader):
            try:
                self.optimizer.zero_grad()
                pred_phar_num, phar_y = self._forward_epoch_pharVQA(model, batched_data)
                phar_loss = self.phar_loss_fn(pred_phar_num, phar_y).mean()
                loss = phar_loss
                epoch_loss += loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.n_updates += 1
                self.lr_scheduler.step()
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('Loss/loss_tot', loss.item(), self.n_updates)

                if self.n_updates == self.args.n_steps:
                    if self.local_rank == 0:
                        self.save_model(model)
                    break
            except Exception as e:
                print(e)
            else:
                continue

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('Loss/epoch_loss', epoch_loss.item(), self.n_updates)

    def fit(self, model, train_loader):
        runs = 0
        for epoch in tqdm(range(1, 1001)):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            break_flag = self.train_epoch(model, train_loader, epoch, runs)
            if self.n_updates >= self.args.n_steps:
                break
            if break_flag:
                break

    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path + f"/{self.args.config}.pth")
