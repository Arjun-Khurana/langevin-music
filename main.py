"""Main file for running experiments.

Set up virtualenv with `pipenv install`, and run with `pipenv run python main.py`.
"""

from langevin_music import main
import itertools
import os

from langevin_music.network.common import PositionalEncoding
from langevin_music.dataset.chorales import RANGES
from langevin_music.dataset.modules import ChoraleSeqDataModule
from langevin_music.network.transformer import MusicTransformer

import json
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import pytorch_lightning.loggers as pl_loggers
from torch.nn.utils import rnn 

def hyperparameter_search(data_module, max_epochs, embedding_dim_values, dropout_values):
    best_val_acc = 0
    best_hyperparams = None
    batch_size = 8
    gpus = 1
    save_data = []
    
    hyperparams_list = list(itertools.product(embedding_dim_values, dropout_values))
    for version, hyperparams in enumerate(hyperparams_list):
        embedding_dim, dropout = hyperparams
        print(f"Training with embedding_dim={embedding_dim}, dropout={dropout}")

        
        # data_module = ChoraleSeqDataModule(batch_size)
        # data = data_module(batch_size)
        epoch_step = 2
        epochs = epoch_step
        model_data = []
        while epochs <= max_epochs:
            if epochs > 2:
                last_checkpoint = os.listdir(f'./logs/default/version_{version}/checkpoints')[0]
                last_checkpoint = os.path.join(f'./logs/default/version_{version}/checkpoints', last_checkpoint)
                print(f'loading from {last_checkpoint}')
                model = MusicTransformer.load_from_checkpoint(last_checkpoint)
            else:
                model = MusicTransformer(embedding_dim=embedding_dim, dropout=dropout)

            trainer = pl.Trainer(
                max_epochs=epoch_step, 
                progress_bar_refresh_rate=20,
                truncated_bptt_steps = 32,
                logger=pl_loggers.TensorBoardLogger("logs/"),
                log_every_n_steps=1,
                gpus = gpus
            )
            trainer.split_idx = 0
            trainer.fit(model, data_module)

            val_acc = model.valid_acc.compute().item()

            model_data.append({
                'epochs': epochs,
                'val_acc': val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_hyperparams = hyperparams

            print(f"Validation accuracy: {val_acc}")

            if len(model_data) > 2 and abs(model_data[-2]['val_acc'] - model_data[-1]['val_acc']) < 0.001:
                print(f'val accuracy converged to: {model_data[-1]["val_acc"]}')
                break

            epochs += epoch_step

        save_data.append({
            'hyperparams': hyperparams,
            'model_data': model_data
        })
        with open('save_data.json', 'w') as f:
            json.dump(save_data, f, indent=4)


    print(f"Best hyperparameters found: embedding_dim={best_hyperparams[0]}, dropout={best_hyperparams[1]}")
    return best_hyperparams

if __name__ == '__main__':
    max_epochs = 10
    embedding_dim_values = [128]
    dropout_values = [0.3]

    data_module = ChoraleSeqDataModule(batch_size = 8)
    best_hyperparams = hyperparameter_search(data_module, max_epochs, embedding_dim_values, dropout_values)

