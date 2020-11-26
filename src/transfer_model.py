import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .utils import evaluate, imsave, LossTracker


def fine_tune(model, dataloaders, epochs=10, **kwargs):
    """Study perf of transfer with limited new data from target domain"""
    pass


def transfer(model, dataloaders, epochs=10, **kwargs):

    train_dl, val_dl = dataloaders

    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.msd.parameters())

    compute_loss = lambda t: loss_criterion(model(t[0]), t[1]).item()
    tr_losses, val_losses = [sum(map(compute_loss, val_dl)) /len(val_dl)], \
        [sum(map(compute_loss, val_dl)) / len(val_dl)]
    
    plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    current_batch = 0
    for e in range(epochs):
        if current_batch > 1e3: break

        for i, (input_, target) in enumerate(train_dl):
            current_batch = i + e*len(train_dl)
            print(f"\rBatch [{current_batch+1:03d}/{epochs*len(train_dl):04d}]", end=' '*5)

            output = model.net(input_)
            loss = loss_criterion(output, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with evaluate(model):                
                val_loss = sum(map(compute_loss, val_dl)) /len(val_dl)
                tr_losses.append(loss.item())
                val_losses.append(val_loss)

            if not current_batch % 20:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.plot(np.arange(len(val_losses)), tr_losses, label='Training', color=plt_colors[0])
                ax.plot(np.arange(len(val_losses)), val_losses, label='Validation', color=plt_colors[1])

                ax.legend(loc='upper right')
                ax.set_xlabel('batch')
                ax.set_xticks(np.linspace(0, len(val_losses), 13).astype('int16'))
                ax.set_ylabel('MSE Loss')
                ax.set_ylim([1e-6,None])
                ax.yaxis.grid()
                plt.savefig('outputs/transfer_loss_W_d80.png', bbox_inches='tight')
                plt.close(fig)

            if current_batch > 500: break
        