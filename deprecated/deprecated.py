import argparse
import glob
import os
import random
import sys
from datetime import datetime

import astra
import foam_ct_phantom
import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.test_model as test
import src.utils as utils
from src import astra_sim
from src.image_dataset import MultiOrbitDataset, data_augmentation
from src.train_model import TVRegularization, train
from src.transfer_model import transfer
from src.unet_regr_model import UNetRegressionModel
from src.utils import (ValSampler, _nat_sort, evaluate, imsave,
                       load_projections, mimsave)


def compute_stats_wrt_sample_size():

    target_ims, input_ims = utils.load_phantom_ds()
    # target_ims, input_ims = utils.load_walnuts_ds()

    dataset = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False)
    full_dl = DataLoader(dataset, batch_size=64)

    n_samples = np.logspace(1, np.log(len(dataset))/np.log(10), 40).astype('int16')

    mean_in, mean_out = 0, 0
    var_in, var_out = 0, 0
    
    torch.set_grad_enabled(False)
    for input_ims, target_ims in full_dl:
        mean_in += input_ims.mean().item()
        mean_out += target_ims.mean().item()

        var_in += input_ims.std().pow(2).item()
        var_out += target_ims.std().pow(2).item()

    true_mean_in = mean_in /len(full_dl)
    true_mean_out = mean_out /len(full_dl)
    true_std_in = np.sqrt(var_in /len(full_dl))
    true_std_out = np.sqrt(var_out /len(full_dl))
    
    mean_in, mean_out, var_in, var_out = [np.zeros(n_samples.shape) for _ in range(4)]

    for i in range(len(n_samples)):
        part_dl = DataLoader(dataset, batch_size=10, sampler=ValSampler(len(dataset), n_samples=n_samples[i]))

        for input_ims, target_ims in part_dl:
            mean_in[i] += input_ims.mean().item()
            mean_out[i] += target_ims.mean().item()

            var_in[i] += input_ims.std().pow(2).item()
            var_out[i] += target_ims.std().pow(2).item()

        mean_in[i] = mean_in[i] /len(part_dl)
        mean_out[i] = mean_out[i] /len(part_dl)

        var_in[i] = var_in[i] /len(part_dl)
        var_out[i] = var_out[i] /len(part_dl)

    fig, axes = plt.subplots(ncols=2, figsize=(20,10))
    axes[0].plot([n_samples[0], n_samples[-1]], [true_mean_in, true_mean_in], '--r')
    axes[0].plot([n_samples[0], n_samples[-1]], [true_mean_out, true_mean_out], '--r')
    axes[1].plot([n_samples[0], n_samples[-1]], [true_std_in, true_std_in], '--r')
    axes[1].plot([n_samples[0], n_samples[-1]], [true_std_out, true_std_out], '--r')

    axes[0].plot(n_samples, mean_in, label='mean_in')
    axes[0].plot(n_samples, mean_out, label='mean_out')
    axes[1].plot(n_samples, np.sqrt(var_in), label='std_in')
    axes[1].plot(n_samples, np.sqrt(var_out), label='std_out')
    
    axes[0].set_title('MEAN'), axes[1].set_title('STD')
    axes[0].legend(); axes[1].legend()
    plt.suptitle('Stats convergence for PhantomRadial dataset')
    plt.savefig('outputs/stat_convergence.png')