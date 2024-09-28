import sys

sys.path.append('..')
from time import strftime

from torch.utils.tensorboard import SummaryWriter

from src.data.pretrain_dataset import pharVQADataset
from src.trainer.pretrain_trainer import Trainer
from src.model.atten_module import MultiHeadedAttention

from src.utils import set_random_seed
import argparse
import torch
import os
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES

from src.data.collator import Collator_pharVQA_pretrain
from src.model.light import PharVQA
from src.trainer.scheduler import PolynomialDecayLR

from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict

import warnings

# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
# warnings.filterwarnings("ignore")
# torch.cuda.device_count()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
local_rank = int(os.environ['LOCAL_RANK'])

# device = 'cpu'
def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training pharVQA")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_steps", type=int, default=100000 )
    parser.add_argument("--save_path", type=str, default='../save/')

    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, default='../pretrained/base/base_VQA_7question.pth')
    parser.add_argument("--dataset", type=str, default='chembl29')
    parser.add_argument("--data_path", type=str, default='../datasets/pretrain')

    parser.add_argument('--num_questions', type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00003, help='model learning rate for training (default: 0.00003)')

    parser.add_argument("--n_threads", type=int, default=1)
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()
    return args


def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)


def get_attention_layer(num_layers, heads, d_model, dropout=0.0, device='cpu'):
    multihead_attn_modules = nn.ModuleList(
        [MultiHeadedAttention(heads, d_model, d_model, dropout=dropout)
         for _ in range(num_layers)])
    return multihead_attn_modules.to(device)


def finetune(args):
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    # local_rank = 2

    # distribution
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')

    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    print(local_rank)

    print(f'pretrain on dataset {args.dataset}')
    args.save_path = f'../save/{args.dataset}/question_num{args.num_questions}/'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print('processing data...')
    collator = Collator_pharVQA_pretrain(config['path_length'], vocab=vocab)
    train_dataset = pharVQADataset(root_path=args.data_path, dataset=args.dataset,
                                   question_num=args.num_questions, device=device)
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset),
                              batch_size=config['batch_size'] // args.n_devices,
                              num_workers=args.n_threads,
                              worker_init_fn=seed_worker, drop_last=True, collate_fn=collator)
    # Model Initialization
    model = PharVQA(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)

    model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                              n_layers=2, predictor_drop=0.0, device=device,
                                              d_hidden_feats=config['d_g_feats'])

    # Finetuning Setting
    # freeze pretrained text model
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}', map_location='cpu').items()},
        strict=False)
    # for name, param in model.model.state_dict().items():
    #     model.graph_model_feat.state_dict()[name].copy_(param)
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    del model.graph_model_feat

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    # model.module.text_model.requires_grad_(False)
    # model.module.graph_model_feat.requires_grad_(False)

    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=20000, tot_updates=200000, lr=config['lr'], end_lr=1e-9,
                                     power=1)

    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))

    phar_loss_fn = MSELoss(reduction='none')
    phar_evaluator = Evaluator('chembl29', 'rmse', args.num_questions)
    result_tracker = Result_Tracker('rmse')
    starttime = strftime("%Y-%m-%d_%H-%M-%S")
    if local_rank == 0:
        summary_writer = SummaryWriter(
            f"tensorboard/pretrain-{args.dataset}/questionNum{args.num_questions}/lr{args.lr}/st{starttime}",
            comment=starttime)
    else:
        summary_writer = None

    trainer = Trainer(args, optimizer, lr_scheduler, phar_loss_fn, phar_evaluator, result_tracker,
                      summary_writer,
                      device=device, local_rank=local_rank)
    trainer.fit(model, train_loader)
    if local_rank == 0:
        summary_writer.close()


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    finetune(args)
