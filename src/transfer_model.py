import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from tqdm import tqdm

from .utils import evaluate, imsave


def fine_tune(model, dataloaders, epochs=10, **kwargs):
    """Study perf of transfer with limited new data from target domain"""
    pass


def shuffle_weights(model):
    """Channel-wise shuffling of a models weights"""
    
    with torch.no_grad():
        for p in model.parameters():
            # Filter out biases and final layer
            if len(p.size()) < 4: continue
            
            for c in range(p.size(1)):
                rand_idx = torch.tensor(random.sample(range(p[:,c].numel()), p[:,c].numel()))
                p.data[:,c] = p.data[:,c].flatten()[rand_idx].view(p[:,c].size())


def plot_weights_dist(module, state_dicts, title=None, filename=None):
    from matplotlib.collections import LineCollection, PolyCollection

    lines, polys = [], []
    with evaluate(module):
        for state_dict in state_dicts:
            module.load_state_dict(state_dict)
            
            parameters = list(filter(lambda t: len(t.size())> 3, module.parameters()))

            param_norm = list(map(lambda t: t.norm(2).mean().item() /t.numel(), parameters))
            param_mean = list(map(lambda t: t.mean().item(), parameters))
            param_std = list(map(lambda t: t.std().item(), parameters))
            
            polys.append(np.concatenate([np.zeros((1,2)), np.stack([np.arange(len(parameters)), param_norm], axis=1), np.array([[len(parameters)-1,0]])], axis=0))
            lines.append(np.stack([np.arange(len(parameters)), param_norm], axis=1))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-40)
    ax.xaxis.set_pane_color((1.0,) *4)
    ax.yaxis.set_pane_color((1.0,) *4)
    ax.zaxis.set_pane_color((1.0,) *4)

    line_collection = LineCollection(lines, colors=('k',) *len(lines))
    poly_collection = PolyCollection(polys, facecolors=('tab:orange',) *len(lines))
    poly_collection.set_alpha(.3)

    ax.add_collection3d(line_collection, zs=np.arange(len(lines)), zdir='y')
    ax.add_collection3d(poly_collection, zs=np.arange(len(lines)), zdir='y')

    ax.set_xlabel('Layer depth')
    ax.set_xlim3d(0, len(parameters))
    ax.set_ylabel('Epochs')
    ax.set_ylim3d(-1, len(lines)+1)
    ax.set_zlabel('Kernel norm')
    ax.set_zlim3d(0, .1)
    ax.grid(False)

    if title is not None: ax.set_title(title)
    if filename is None: filename = 'model_weights.png'

    plt.savefig(f'outputs/{filename}')
    plt.close()


def transfer(model, dataloaders, epochs=10, **kwargs):

    train_dl, val_dl = dataloaders

    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.msd.parameters())

    compute_loss = lambda t: loss_criterion(model(t[0]), t[1]).item()
    tr_losses, val_losses = [sum(map(compute_loss, val_dl)) /len(val_dl)], \
        [sum(map(compute_loss, val_dl)) / len(val_dl)]
    
    plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    max_batch = 500
    current_batch = 0
    for e in range(epochs):
        if current_batch > max_batch: break

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
                ax.set_ylim([1e-6,2e-3])
                ax.yaxis.grid()
                plt.savefig(kwargs.get('filename', 'outputs/transfer_loss.png'), bbox_inches='tight')
                plt.close(fig)

            if current_batch > max_batch: break

    print('')
        