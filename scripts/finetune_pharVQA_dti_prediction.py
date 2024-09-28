import os, pickle
import random
import sys

import numpy as np
import torch
from dgl import load_graphs

from torch.utils.data import DataLoader

import warnings
from argparse import Namespace

sys.path.append('../')
from DTI.args import args
from DTI.utils import seed_set, save_print_log, get_model_args
from DTI.trainer import Trainer
from src.data.collator import Collator_pharVQA_dti
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import pharVQADataset_dti
from src.model.dti_model import phar_dti_model
from src.model_config import config_dict

warnings.filterwarnings("ignore")


def load_args(path: str,
              current_args: Namespace = None,
              ):
    """
    Loads a model args.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :return: The loaded args.
    """
    # Load args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    model_ralated_args = get_model_args()

    if current_args is not None:
        for key, value in vars(args).items():
            if key in model_ralated_args:
                setattr(current_args, key, value)
    else:
        current_args = args
    return current_args


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    vocab=None,
                    config=None,
                    d_fps=None,
                    d_mds=None,
                    cuda: bool = None,
                    n_word=None,
                    load_model=True):
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MPNN.
    """

    # args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    model = phar_dti_model(current_args, config, vocab, d_fps, d_mds, n_word=n_word)
    if load_model == False:

        del model.embedding_model.md_predictor
        del model.embedding_model.fp_predictor
        del model.embedding_model.node_predictor

        model.embedding_model.text_model.requires_grad_(False)
        model.embedding_model.model.requires_grad_(False)
        return model
    else:
        # Load model and args
        state = torch.load(path, map_location=lambda storage, loc: storage)
        args, loaded_state_dict = state['args'], state['state_dict']
        model_ralated_args = get_model_args()

        if current_args is not None:
            for key, value in vars(args).items():
                if key in model_ralated_args:
                    setattr(current_args, key, value)
        else:
            current_args = args

        model_state_dict = model.state_dict()

        # Skip missing parameters and parameters of mismatched size
        pretrained_state_dict = {}
        for param_name in loaded_state_dict.keys():
            new_param_name = param_name
            if new_param_name not in model_state_dict:
                print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
            elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
                print(f'Pretrained parameter "{param_name}" '
                      f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                      f'model parameter of shape {model_state_dict[new_param_name].shape}.')
            else:
                # debug(f'Loading pretrained parameter "{param_name}".')
                pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
        # Load pretrained weights
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)

        if cuda:
            print('Moving model to cuda')
            model = model.cuda()

        return model


def load_dataset(args):
    graphs, label_dict = load_graphs(os.path.join(args.root_path, f"{args.dataset}/{args.dataset}_5.pkl"))
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    test_size = len(graphs) - train_size - val_size

    all_idx = list(range(len(graphs)))
    random.shuffle(all_idx)
    train_idx = all_idx[:train_size]
    val_idx = all_idx[train_size:train_size + val_size]
    test_idx = all_idx[train_size + val_size:]
    train_dataset = pharVQADataset_dti(root_path=args.root_path, dataset=args.dataset, question_num=args.num_questions,
                                       use_idxs=train_idx, device=args.device)
    val_dataset = pharVQADataset_dti(root_path=args.root_path, dataset=args.dataset, question_num=args.num_questions,
                                     use_idxs=val_idx, device=args.device)
    test_dataset = pharVQADataset_dti(root_path=args.root_path, dataset=args.dataset, question_num=args.num_questions,
                                      use_idxs=test_idx, device=args.device)

    return train_dataset, val_dataset, test_dataset, train_dataset.n_word


def run_training(args):
    seed_set(args.seed)
    config = config_dict['base']
    train_dataset, valid_dataset, test_dataset, n_word = load_dataset(args)

    collator = Collator_pharVQA_dti(config['path_length'])
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                              collate_fn=collator,
                              drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                            collate_fn=collator,
                            drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True,
                             collate_fn=collator,
                             drop_last=False)

    option = args.__dict__
    print(f'Building model')
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = load_checkpoint(args.checkpoint_path, current_args=args, config=config, vocab=vocab,
                            d_fps=train_dataset.d_fps,
                            d_mds=train_dataset.d_mds, n_word=n_word, load_model=args.load_model)

    model = model.to(args.device)
    # model.from_pretrain(args.checkpoint_path)
    trainer = Trainer(args, model)

    test_auc = trainer.train(train_loader,val_loader,test_loader)
    return test_auc


def cross_validate(args):
    """k-fold cross validation"""
    # Initialize relevant variables
    init_seed = args.seed
    root_save_dir = args.exp_path
    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        print('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        args.exp_path = os.path.join(root_save_dir, 'seed_{}'.format(args.seed))
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
        model_scores = run_training(args)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    save_print_log('==' * 20, root_save_dir)
    for fold_num, scores in enumerate(all_scores):
        msg = 'Seed {} ==> {} = {}'.format(init_seed + fold_num, args.metric, scores)
        save_print_log(msg, root_save_dir)

    # Report scores across models
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)  # average score for each model across tasks
    msg = 'Overall test {} = {} +/- {}'.format(args.metric, mean_score, std_score)
    save_print_log(msg, root_save_dir)
    return mean_score, std_score


if __name__ == '__main__':
    cross_validate(args)
