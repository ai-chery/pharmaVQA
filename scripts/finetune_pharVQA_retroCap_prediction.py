import os
import sys

from rdkit import Chem
from torch.utils.data import DataLoader

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
sys.path.append(os.path.dirname(path))
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

import pickle
import numpy as np
from datetime import datetime

from retroCap.utils.model_utils import validate
from retroCap.utils.loss_utils import LabelSmoothingLoss

from src.data.finetune_dataset import pharVQADataset_retro, phar_Dict
from src.data.collator import Collator_pharVQA_retroCap

from src.model.retro_model import retro_model
from src.model_config import config_dict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:2', help='device GPU/CPU')
    parser.add_argument('--batch_size_trn', type=int, default=32, help='raw train batch size')
    parser.add_argument('--batch_size_val', type=int, default=32, help='val/test batch size')

    parser.add_argument('--rerank_num', type=int, default=1, help='')
    parser.add_argument('--with_align', type=str, default='True', choices=['True', 'False'], help='')
    parser.add_argument('--data_dir', type=str, default='../datasets/RXNprediction/USPTO-50k/', help='base directory')
    parser.add_argument('--save_dir', type=str, default='./results/retro', help='intermediate directory')

    parser.add_argument('--checkpoint_dir', type=str, default='pharVQA_checkpoint_diff_lr', help='checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint model file')

    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument('--num_questions', type=int, default=7)
    parser.add_argument("--config", type=str, default='base')

    parser.add_argument("--encoder_path", type=str, default='../pretrained/base/base.pth')
    # graph

    ####
    parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
    parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
    parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
    parser.add_argument('--d_ff', type=int, default=2048, help='')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--known_class', type=str, default='False', choices=['True', 'False'],
                        help='with reaction class known/unknown')
    parser.add_argument('--shared_vocab', type=str, default='False', choices=['True', 'False'],
                        help='whether sharing vocab')
    parser.add_argument('--shared_encoder', type=str, default='False', choices=['True', 'False'],
                        help='whether sharing encoder')
    parser.add_argument('--phar_lr', type=float, default=0.00003, help='maximum epoch')
    parser.add_argument('--retro_lr', type=float, default=0.0005, help='maximum epoch')

    parser.add_argument('--max_epoch', type=int, default=5000, help='maximum epoch')
    parser.add_argument('--max_step', type=int, default=5000000, help='maximum steps')
    parser.add_argument('--lr_per_epoch', type=int, default=5, help='validation steps frequency')
    parser.add_argument('--save_per_epoch', type=int, default=20, help='checkpoint saving steps frequency')
    parser.add_argument('--verbose', type=str, default='True', choices=['True', 'False'])

    args = parser.parse_args()
    return args


def anneal_prob(step, k=2, total=150000):
    step = np.clip(step, 0, total)
    min_, max_ = 1, np.exp(k * 1)
    return (np.exp(k * step / total) - min_) / (max_ - min_)


def load_checkpoint(args, model):
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model'])
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
    optimizer = checkpoint['optim']
    step = checkpoint['epoch']
    return step, optimizer, model.to(args.device)


def build_data(args, config, train=True, augment=False):
    known_class = args.known_class == 'True'
    shared_vocab = args.shared_vocab == 'True'

    if train:
        dataset = pharVQADataset_retro(mode='train', root_path=args.data_dir,
                                       known_class=known_class,
                                       shared_vocab=shared_vocab, augment=augment, rerank_num=args.rerank_num,
                                       question_num=7)
        dataset_val = pharVQADataset_retro(mode='valid', root_path=args.data_dir,
                                           known_class=known_class,
                                           shared_vocab=shared_vocab, rerank_num=args.rerank_num,
                                           question_num=7)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        collator = Collator_pharVQA_retroCap(src_pad, tgt_pad, config['path_length'])
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=True, drop_last=True,
                                collate_fn=collator)
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                              collate_fn=collator)
        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos, dataset.d_fps, dataset.d_mds
    else:
        dataset = pharVQADataset_retro(mode='test', root_path=args.data_dir,
                                       known_class=known_class,
                                       shared_vocab=shared_vocab, rerank_num=args.rerank_num,
                                       question_num=7)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        collator = Collator_pharVQA_retroCap(src_pad, tgt_pad, config['path_length'])
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                               collate_fn=collator)
        return test_iter, dataset, dataset.d_fps, dataset.d_mds


def build_model(args, config, d_fps, d_mds, vocab_itos_src, vocab_itos_tgt):
    src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
    tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]

    model = retro_model(args, config, d_fps=d_fps, d_mds=d_mds,
                        encoder_num_layers=args.encoder_num_layers,
                        decoder_num_layers=args.decoder_num_layers,
                        d_model=args.d_model, heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
                        vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
                        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
                        device=args.device)

    return model.to(args.device)


def validate(args, model, val_iter, pad_idx=1):
    pred_token_list, gt_token_list, pred_infer_list, gt_infer_list = [], [], [], []
    model.eval()
    for batch in tqdm(val_iter):
        p_smiles, r_smiles, src, tgt, gt_context_alignment, product_graph_num, product_graph, fps, mds, text, text_mask, phar_targets, \
            atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions = batch
        src, tgt, gt_context_alignment = src.to(args.device), tgt.to(args.device), gt_context_alignment.to(
            args.device)
        product_graph, fps, mds, text, text_mask, phar_targets = product_graph.to(args.device), fps.to(
            args.device), mds.to(args.device), text.to(args.device), text_mask.to(args.device), phar_targets.to(
            args.device)

        # Infer:
        with torch.no_grad():
            generative_scores, context_scores, pred_phar_num, pharVQA_atten = model(src, tgt, fps, mds, text, text_mask,
                                                                                    product_graph,
                                                                                    product_graph_num)
        # Token accuracy:
        pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
        _, pred_token_label = pred_token_logit.topk(1, dim=-1)
        gt_token_label = tgt[1:].view(-1)
        pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
        gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

    pred_tokens = torch.cat(pred_token_list).view(-1)
    gt_tokens = torch.cat(gt_token_list).view(-1)

    return (pred_tokens == gt_tokens).float().mean().item()


def update_check(accuracy, best_acc):
    if accuracy > best_acc:
        best_acc = accuracy
    return best_acc
    pass


def main(args):
    torch.backends.cudnn.benchmark = True
    print(args.device)

    g = torch.Generator()
    g.manual_seed(args.seed)

    log_path = args.checkpoint_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_file_name = os.path.join(log_path, datetime.now().strftime("%D:%H:%M:%S").replace('/', ':') + '.txt')
    with open(log_file_name, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    config = config_dict[args.config]
    train_iter, val_iter, vocab_itos_src, vocab_itos_tgt, d_fps, d_mds = build_data(args, config, train=True,
                                                                                    augment=True)
    model = build_model(args, config, d_fps, d_mds, vocab_itos_src, vocab_itos_tgt)

    del model.embedding_model.md_predictor
    del model.embedding_model.fp_predictor
    del model.embedding_model.node_predictor

    model.embedding_model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(f'{args.encoder_path}', map_location='cpu').items()},
        strict=False)
    for name, param in model.embedding_model.model.state_dict().items():
        model.embedding_model.graph_model_feat.state_dict()[name].copy_(param)

    best_accuracy = 0.0
    global_step = 0
    if args.checkpoint:
        global_step, opt, model = load_checkpoint(args, model)
        phar_lr = opt['param_groups'][0]['lr']
        retro_lr = opt['param_groups'][1]['lr']
        eps = opt['param_groups'][0]['eps']

        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'embedding_model' in n], 'lr': args.phar_lr},
            {'params': [p for n, p in model.named_parameters() if 'embedding_model' not in n], 'lr': args.retro_lr}
        ]
        optimizer = optim.Adam(param_groups, eps=eps)
    else:

        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'embedding_model' in n], 'lr': args.phar_lr},
            {'params': [p for n, p in model.named_parameters() if 'embedding_model' not in n], 'lr': args.retro_lr}
        ]
        optimizer = optim.Adam(param_groups, eps=1e-5)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    phar_loss_fn = nn.MSELoss(reduction="sum")
    align_loss_fn = LabelSmoothingLoss(reduction='sum')
    criterion_context_align = LabelSmoothingLoss(reduction='sum')
    criterion_tokens = LabelSmoothingLoss(ignore_index=model.embedding_tgt.word_padding_idx,
                                          reduction='sum', apply_logsoftmax=False)

    model.embedding_model.text_model.requires_grad_(False)
    model.embedding_model.model.requires_grad_(False)

    print('Begin:')
    for epoch in range(args.max_epoch):
        if args.verbose == 'True':
            header = 'Train Epoch: [{}]'.format(global_step + epoch)
            print_freq = 50
            progress_bar = tqdm(train_iter, miniters=print_freq, desc=header)
        else:
            progress_bar = train_iter

        loss_history_all, loss_history_token, loss_history_phar, loss_history_atten, loss_history_align = [], [], [], [], []
        for i, batch in enumerate(progress_bar):
            # if global_step > args.max_step:
            #     print('Finish training.')
            #     break
            model.train()
            p_smiles, r_smiles, src, tgt, gt_context_alignment, product_graph_num, product_graph, fps, mds, text, text_mask, phar_targets, \
                atom_phar_target_map, v_phar_atom_id, v_phar_name, random_questions = batch
            src, tgt, gt_context_alignment = src.to(args.device), tgt.to(args.device), gt_context_alignment.to(
                args.device)
            product_graph, fps, mds, text, text_mask, phar_targets = product_graph.to(args.device), fps.to(
                args.device), mds.to(args.device), text.to(args.device), text_mask.to(args.device), phar_targets.to(
                args.device)

            torch.cuda.empty_cache()
            # Forward

            generative_scores, context_scores, pred_phar_num, pharVQA_atten = model(src, tgt, fps, mds, text, text_mask,
                                                                                    product_graph,
                                                                                    product_graph_num)

            # Loss for phar prediction
            phar_y = phar_targets.to(torch.float64)
            phar_loss = phar_loss_fn(pred_phar_num, phar_y)

            # Loss for attention
            batch_size = len(p_smiles)
            phar_target_mx = torch.zeros((batch_size, pharVQA_atten[0].size()[-2], len(pharVQA_atten))).to(
                args.device)
            mols = [Chem.MolFromSmiles(s) for s in p_smiles]
            for mol_index, mol in enumerate(mols):
                random_question = random_questions[mol_index]
                mol_phar_mx = atom_phar_target_map[mol_index]
                for bond_index, bond in enumerate(mol.GetBonds()):
                    begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    phar_target_mx[mol_index, bond_index] = mol_phar_mx[
                        [begin_atom_id, end_atom_id], phar_Dict[random_question[0]]].sum(dim=0).to(
                        torch.bool).to(torch.float32)
            pharVQA_atten = torch.stack(pharVQA_atten, dim=2).sum(dim=-1)

            align_loss = align_loss_fn(pharVQA_atten.view(pharVQA_atten.size()[0], -1),
                                       phar_target_mx.view(phar_target_mx.size()[0], -1)).to(torch.float64)

            # Loss for language modeling:
            pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
            gt_token_label = tgt[1:].view(-1)

            # Compute all loss:
            loss_token = criterion_tokens(pred_token_logit, gt_token_label)

            # for context_score in context_scores:
            # Loss for context alignment:
            loss_context_align = 0
            if args.with_align == 'True':
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])

                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])

                loss_context_align = criterion_context_align(pred_context_align_logit,
                                                             gt_context_align_label)

            loss = loss_token + loss_context_align + phar_loss + align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history_all.append(loss.item())
            loss_history_token.append(loss_token.item())
            if args.with_align == 'True':
                loss_history_align.append(loss_context_align.item())
            else:
                loss_history_align.append(loss_context_align)
            loss_history_phar.append(phar_loss.item())
            loss_history_atten.append(align_loss.item())

            progress_bar.set_description(
                f'loss={loss.item():.4f}, phar_lr={optimizer.param_groups[0]["lr"]:.6f},retro_lr={optimizer.param_groups[1]["lr"]:.6f}, mean_loss={sum(loss_history_all) / (i + 1)}')

        if (epoch + 1) % args.lr_per_epoch == 0:
            scheduler.step()
        print_line = "[Time {} Epoch {}] Total Loss {} NLL-Loss {} Align-Loss {} phar-Loss {} atten-Loss {}".format(
            datetime.now().strftime("%D:%H:%M:%S").replace('/', ':'), global_step + epoch + 1,
            round(np.mean(loss_history_all), 4), round(np.mean(loss_history_token), 4),
            round(np.mean(loss_history_align), 4), round(np.mean(loss_history_phar), 4),
            round(np.mean(loss_history_atten), 4))
        print(print_line)
        with open(log_file_name, 'a+') as f:
            f.write(print_line)
            f.write('\n')

        # validation
        accuracy = validate(args, model, val_iter, model.embedding_tgt.word_padding_idx)
        print_line = 'Validation accuracy: {}'.format(round(accuracy, 4))
        print(print_line)
        with open(log_file_name, 'a+') as f:
            f.write(print_line)
            f.write('\n')

        if (epoch + 1) % args.save_per_epoch == 0:
            checkpoint_path = args.checkpoint_dir + '/model_{}.pt'.format(epoch + 1)
            torch.save({'model': model.state_dict(), 'epoch': global_step + epoch + 1, 'optim': optimizer.state_dict()},
                       checkpoint_path)
            print('Checkpoint saved to {}'.format(checkpoint_path))
        # save best
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_checkpoint_path = args.checkpoint_dir + '/model_best.pt'
            torch.save({'model': model.state_dict(), 'epoch': global_step + epoch + 1, 'optim': optimizer.state_dict()},
                       best_checkpoint_path)
            print('Best checkpoint saved to {}'.format(best_checkpoint_path))


if __name__ == '__main__':

    args = parse_args()
    print(args)
    with open('args.pk', 'wb') as f:
        pickle.dump(args, f)

    if args.known_class == 'True':
        args.checkpoint_dir = args.checkpoint_dir + '_typed'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '_untyped'

    if args.with_align == 'True':
        args.checkpoint_dir = args.checkpoint_dir + '_with_align'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '_without_align'

    args.proj_path = args.save_dir
    print(f"os.getcwd:{args.proj_path}")
    args.checkpoint_dir = os.path.join(args.proj_path, args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    # args.data_dir = os.path.join('/'.join(args.proj_path.split('/')), args.data_dir)
    main(args)
