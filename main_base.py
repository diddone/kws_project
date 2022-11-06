import os
from typing import Tuple, Union, List, Callable, Optional
from tqdm import tqdm
from itertools import islice
import dataclasses

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torchaudio
from IPython import display as display_

from collections import defaultdict
from IPython.display import clear_output
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from thop import profile
import tempfile
import wandb

from data.data import get_dataloaders
from melspecs import get_melspecs
from base_config import BaseConfig
from models import CRNN
import utils
from train_distil import train_distillation
from train import train_base
import argparse
from pathlib import Path
from datetime import datetime


WANDB_GROUP = 'Base'
PARAMS = {
    'num_epochs': 200,
    'learning_rate': 3e-4,
    'alpha': 0.9,
    'temp': 20,
    'hidden_size': 12,
    'cnn_out_channels': 3,
    'gru_num_layers': 2,
    'kernel_size': (5, 24),
    'stride': (3, 15),
    'dropout': 0.,
}


def main(args, params={}):
    utils.fix_seed(args.seed)

    save_dir = Path(f'save/{WANDB_GROUP}') / datetime.now().strftime(r"%m%d_%H%M%S")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    config = BaseConfig(**params)
    config.use_wandb = args.use_wandb

    train_loader, val_loader = get_dataloaders(config.keyword)
    melspec_train, melspec_val = get_melspecs(config)

    base_model = CRNN(config).to(config.device)

    print(base_model)

    opt = torch.optim.Adam(
        base_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.num_epochs, eta_min=5e-5)
        if args.use_sched
        else None
    )

    exp_name = f'{WANDB_GROUP}_sched={args.use_sched}'

    if config.use_wandb:
        wandb.init(project='dla-kws', config=dataclasses.asdict(config), name=exp_name, group='base')
        wandb.watch(base_model)

    history = train_base(base_model, train_loader, val_loader, melspec_train, melspec_val, opt, config)

    print('FINAL_METRIC', history[-1])
    save_path = str(save_dir / (exp_name + '.tar'))
    utils.save_model(base_model, save_path)
    wandb.finish()

    return history

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="KWS")

    parser.add_argument(
        "-s",
        "--seed",
        default=911,
        type=int,
        help="Seed (default: 911)",
    )

    parser.add_argument(
        "-sc",
        "--use_sched",
        default=False,
        type=bool,
        help="Use scheduler (default: False)",
    )

    parser.add_argument(
        "-w",
        "--use_wandb",
        default=True,
        type=bool,
        help="Use wandb to log data",
    )

    args = parser.parse_args()
    params = {
        'num_epochs' : 30
    }

    for lr in [3e-4, 5e-4]:
        for seed in [99, 911, 119, 9, 123, 42]:
            params['learning_rate'] = lr
            args.seed = seed
            main(args, params)