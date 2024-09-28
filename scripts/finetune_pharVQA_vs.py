import sys
from time import strftime

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from src.model.atten_module import MultiHeadedAttention
from src.model.encoder import TransformerEncoder

from src.utils import set_random_seed, LabelSmoothingLoss
import argparse
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset, pharVQADataset
from src.data.collator import Collator_tune, Collator_pharVQA
from src.model.light import LiGhTPredictor as LiGhT, PharVQA, LiGhT_graph_model
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.finetune_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict

import warnings

# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
warnings.filterwarnings("ignore")


# torch.cuda.device_count()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument('--train_kpgt', type=str, default="False")
    parser.add_argument('--prompt_mode', type=str, default='cat')
    parser.add_argument('--normalize', type=str, default='True')

    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--num_runs", type=int, default=1)

    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--save_path", type=str, default='../save/')

    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, default='../pretrained/base/base.pth')
    parser.add_argument("--dataset", type=str, default='HPK1')
    parser.add_argument("--data_path", type=str, default='../datasets/vs/')
    parser.add_argument("--dataset_type", type=str, default='regression')
    parser.add_argument("--metric", type=str, default='spear,pear')
    parser.add_argument("--split", type=str, default='scaffold-0')

    parser.add_argument('--num_questions', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size for training (default: 32)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00003, help='model learning rate for training (default: 0.00003)')

    parser.add_argument("--n_threads", type=int, default=1)
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


def get_attention_layer(num_layers, heads, d_model_kv, d_model_q, dropout=0.0, device='cpu'):
    multihead_attn_modules = nn.ModuleList(
        [MultiHeadedAttention(heads, d_model_kv, d_model_q, dropout=dropout)
         for _ in range(num_layers)])

    encoder = TransformerEncoder(num_layers=num_layers,
                                 d_model=d_model_q, heads=heads,
                                 d_ff=d_model_q, dropout=dropout,
                                 attn_modules=multihead_attn_modules)
    return encoder.to(device)


def finetune(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    args.save_path = f'../save/{args.dataset}/question_num{args.num_questions}/{args.split}/{args.seed}/train_kpgt_{args.train_kpgt}/prompt_{args.prompt_mode}_normalize_{args.normalize}/'
    print(f'dataset moleculesNet on {args.dataset} split {args.split}')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    collator = Collator_pharVQA(config['path_length'])
    train_dataset = pharVQADataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type,
                                   question_num=args.num_questions,
                                   split_name=f'{args.split}', split='train', device=device)
    val_dataset = pharVQADataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type,
                                 question_num=args.num_questions,
                                 split_name=f'{args.split}', split='val', device=device)
    test_dataset = pharVQADataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type,
                                  question_num=args.num_questions,
                                  split_name=f'{args.split}', split='test', device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads,
                              worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads,
                            worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads,
                             worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
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

    if args.train_kpgt == 'True':
        model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                        n_tasks=train_dataset.n_tasks,
                                        n_layers=2,
                                        predictor_drop=0.0,
                                        device=device, d_hidden_feats=config['d_g_feats'])
    elif args.train_kpgt == 'False' :
        model.prompt_linear_model = get_predictor(d_input_feats=config['d_g_feats'], n_tasks=1, \
                                                  n_layers=2, predictor_drop=0.0, device=device,
                                                  d_hidden_feats=config['d_g_feats'])
        if args.prompt_mode == 'add':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'] * 3,
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3,
                                            n_tasks=train_dataset.n_tasks,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
        elif args.prompt_mode == 'cat':
            model.prompt_projection_model = get_predictor(d_input_feats=(args.num_questions) * config['d_g_feats'],
                                                          n_tasks=config['d_g_feats'],
                                                          n_layers=2, predictor_drop=0.0, device=device,
                                                          d_hidden_feats=config['d_g_feats'])

            model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 4,
                                            n_tasks=train_dataset.n_tasks,
                                            n_layers=2,
                                            predictor_drop=0.0,
                                            device=device, d_hidden_feats=config['d_g_feats'])
    print(model.predictor)
    # Finetuning Setting
    # freeze pretrained text model
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}', map_location='cpu').items()},
        strict=False)
    for name, param in model.model.state_dict().items():
        model.graph_model_feat.state_dict()[name].copy_(param)

    model.text_model.requires_grad_(False)
    model.model.requires_grad_(False)

    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer,
                                     warmup_updates=args.n_epochs * len(train_dataset) // args.batch_size // 10,
                                     tot_updates=args.n_epochs * len(train_dataset) // args.batch_size, lr=args.lr,
                                     end_lr=1e-9, power=1)

    if args.dataset_type == 'classification':
        loss_fn = BCEWithLogitsLoss(reduction='none')
    else:
        loss_fn = MSELoss(reduction='none')

    if args.normalize == 'True':
        mean = train_dataset.mean.numpy()
        std = train_dataset.std.numpy()
    else:
        mean = None
        std = None
    evaluator_spear = Evaluator(args.dataset, 'spear', train_dataset.n_tasks, mean=mean,
                                std=std)
    evaluator_pear = Evaluator(args.dataset, 'pear', train_dataset.n_tasks, mean=mean,
                               std=std)
    phar_loss_fn = MSELoss(reduction='none')
    phar_evaluator = Evaluator(args.dataset, 'rmse', args.num_questions)

    align_loss_fn = LabelSmoothingLoss(reduction='sum', smoothing=0.5)

    result_tracker = Result_Tracker(args.metric)
    starttime = strftime("%Y-%m-%d_%H-%M-%S")
    summary_writer = SummaryWriter(
        f"tensorboard/finetune-{args.dataset}/questionNum{args.num_questions}/split{args.split}/lr{args.lr}/st{starttime}",
        comment=starttime)
    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, phar_loss_fn, align_loss_fn,
                      [evaluator_spear, evaluator_pear], phar_evaluator,
                      result_tracker,
                      summary_writer,
                      device=device,
                      model_name='pharVQA_vs',
                      label_mean=train_dataset.mean.to(device) if mean is not None else None,
                      label_std=train_dataset.std.to(device) if std is not None else None)
    best_train, best_val, best_test, best_train_pharNum, best_val_pharNum, best_test_pharNum = trainer.fit(model,
                                                                                                           train_loader,
                                                                                                           val_loader,
                                                                                                           test_loader)
    # print(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")
    print(f'train: {best_train}, val: {best_val}, test: {best_test} ')
    print(f"Phar num RMSE, val: {best_val_pharNum:.3f}, test: {best_test_pharNum:.3f}")

    return best_test


if __name__ == '__main__':
    args = parse_args()
    scaffold_results_list = []
    for split in ['scaffold-0', 'scaffold-1', 'scaffold-2']:
        args.split = split
        results_list = []
        for runs in range(args.num_runs):
            args.seed = args.seed + runs
            set_random_seed(args.seed)
            results = finetune(args)
            results_list.append([args.split, runs, results])
            print(f'Dataset {args.dataset} on {args.split} at seed {args.seed} results is: {results}')
    print(results_list)
    df = pd.DataFrame(results_list)
    df.to_csv(
        f'{args.save_path}/{args.dataset}_questionNum{args.num_questions}_{args.split}_finetune.csv',
        mode='a', index=False, header=False
    )
