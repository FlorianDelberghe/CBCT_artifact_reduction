import argparse
import glob
import os
import random
import sys
from datetime import datetime
from typing import Iterable, List, Tuple

import numpy as np
import torch
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.image_dataset import MultiOrbitDataset
from src.models import UNetRegressionModel
from src.train_model import *
from src.utils import ValSampler, _nat_sort
from SVCCA import cca_core
from SVCCA.pwcca import compute_pwcca


class MSDForwardLogger():

    def __init__(self, layers, n_samples, sample_rate):
        self.layers = layers
        self._representation = torch.zeros(len(layers), n_samples *sample_rate **2, device='cuda')
        self.sample_rate = sample_rate
        self.current_ind = 0
    
    def update_representation(self, layers_out):
        for i, layer_out in enumerate(layers_out):
            slc = [slice(i,i+1,1), slice(self.current_ind, self.current_ind +len(layer_out) *self.sample_rate **2, 1)]
            self._representation[slc] = layer_out.reshape(1, len(layer_out) *self.sample_rate **2)
            
        self.current_ind += len(layer_out) *self.sample_rate **2

    def get_representation(self, group_sz=5):
        return [self._representation[i:i+group_sz].cpu().numpy() for i in range(0, len(self._representation), group_sz)]

class UNetForwardLogger():

    def __init__(self, layers, n_samples, sample_rate):
        # (layers, channels, samples)
        self._representation = [torch.zeros(self.get_layer_out_size(layer), n_samples *sample_rate **2, device='cuda') 
                               for layer in layers]
        self.sample_rate = sample_rate
        self.current_ind = np.zeros(len(layers), dtype='int32')

    @staticmethod
    def get_layer_out_size(layer):
        if hasattr(layer, 'mpconv'):
            return layer.mpconv[1].conv[3].weight.size(1)

        elif isinstance(layer.conv, nn.Conv2d):
            return layer.conv.weight.size(1)

        return layer.conv.conv[3].weight.size(1)

    def update_layer_representation(self, layer_ind, layer_out): 
        
        slc = [slice(None), 
               slice(self.current_ind[layer_ind], self.current_ind[layer_ind] +len(layer_out) *self.sample_rate **2, 1)]
               
        self._representation[layer_ind][slc] = torch.flatten(torch.transpose(layer_out, 0,1), 1)

        self.current_ind[layer_ind] += len(layer_out) *self.sample_rate **2

    def get_representation(self):
        return [rep.cpu().numpy() for rep in self._representation]


def hook_MSD(model, layers, n_samples, sample_rate):

    def hook_closure(layers, logger):
        def hook_fn(self, input_, output):
            sample_int = (output.size(-2) //sample_rate, output.size(-1) //sample_rate)
            slc = lambda i: (slice(None), slice(i,i+1,1),
                             slice(sample_int[0]//2, sample_rate*sample_int[0], sample_int[0]),
                             slice(sample_int[1]//2, sample_rate*sample_int[1], sample_int[1]))

            logger.update_representation([output[slc(i)] for i in layers]) 

        return hook_fn

    msd_logger = MSDForwardLogger(layers, n_samples, sample_rate)
    model.hook_handles = [model.msd.msd_block.register_forward_hook(hook_closure(layers, msd_logger))]

    return msd_logger

def hook_UNet(model, layers, n_samples, sample_rate):

    def hook_closure(layer, layer_ind, logger):
        def hook_fn(self, input_, output):
            sample_int = (output.size(-2) //sample_rate, output.size(-1) //sample_rate)
            slc = (slice(None), slice(None), 
                   slice(sample_int[0]//2, sample_rate*sample_int[0], sample_int[0]),
                   slice(sample_int[1]//2, sample_rate*sample_int[1], sample_int[1]))
                                
            logger.update_layer_representation(layer_ind, output[slc])

        return hook_fn

    unet_logger = UNetForwardLogger(layers, n_samples, sample_rate)    
    model.hook_handles = []
    for i, layer in enumerate(layers):
        model.hook_handles.append(layer.register_forward_hook(hook_closure(layer, i, unet_logger)))

    return unet_logger

def hook_model(model, *args, **kwargs):

    if isinstance(model, UNetRegressionModel):
        return hook_UNet(model, *args, **kwargs)

    elif isinstance(model, MSDRegressionModel):
        return hook_MSD(model, *args, **kwargs)

    else:
        raise ValueError(f"Unknown model class: {model.__class__.__name__}")


def get_model_representation(models, patches, layers, sample_rate=10): 

    # patches are in batches need to get the sampler's size and not the dataloader's
    loggers = [hook_model(model, layers[i], len(patches.sampler), sample_rate) for i, model in enumerate(models)]

    with utils.evaluate(*models):
        for patch, _ in patches: [model(patch) for model in models]

    [hook_handle.remove() for model in models for hook_handle in model.hook_handles]
    return  [logger.get_representation() for logger in loggers]


def get_patches(dataset, n_patches=100, batch_size=8) -> Iterable[torch.Tensor]: 

    return DataLoader(dataset, batch_size, sampler=ValSampler(len(dataset), n_patches), drop_last=False)

    
def get_layers(model): return [model.msd.inc,
                               model.msd.down1, model.msd.down2, model.msd.down3, model.msd.down4,
                               model.msd.up1, model.msd.up2, model.msd.up3, model.msd.up4,
                            #    model.msd.outc
                               ]


def get_svcca_matrix(reps1, reps2, max_dims=None):

    cca_matrix = np.zeros((len(reps1), len(reps2)))

    for i, rep1 in enumerate(reps1):
        for j, rep2 in enumerate(reps2):
            n_dims = min(len(rep1), len(rep2)) if max_dims is not None else min(max_dims, min(len(rep1), len(rep2)))

            rep1, rep2 = rep1 -rep1.mean(axis=1, keepdims=True), rep2 -rep2.mean(axis=1, keepdims=True)

            U1, s1, V1 = np.linalg.svd(rep1, full_matrices=False)
            U2, s2, V2 = np.linalg.svd(rep2, full_matrices=False)

            svacts1 = np.dot(s1[:n_dims]*np.eye(n_dims), V1[:n_dims])
            # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
            svacts2 = np.dot(s2[:n_dims]*np.eye(n_dims), V2[:n_dims])
            # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)
            
            cca_coefs = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-18, verbose=False)['cca_coef1']
            cca_matrix[i, j] = cca_coefs.mean()

    return cca_matrix
    

def get_pwcca_dist(reps1, reps2, *, xlabel=None, ylabel=None, xticklabels=None, title=None, **kwargs):
    assert len(reps1) == len(reps2)
    
    pwcca_dists = np.zeros(len(reps1))

    for i, (rep1, rep2) in enumerate(zip(reps1, reps2)):

            n_dims = min(len(rep1), len(rep2))

            rep1, rep2 = rep1 -rep1.mean(axis=1, keepdims=True), rep2 -rep2.mean(axis=1, keepdims=True)

            pwcca_coef, *_ = compute_pwcca(rep1, rep2, epsilon=1e-7)
            pwcca_dists[i] = 1-pwcca_coef

    return pwcca_dists


if __name__ == '__main__':

    GPU_ID = 1
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
            torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
        flush=True)

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 80, 'width': 1,
                    'dilations': [1, 2, 4, 8, 16], 'loss': 'L2'}

    target_ims, input_ims = utils.load_phantom_ds()
    cv_split_generator = utils.split_data_CV(input_ims, target_ims, frac=(1/7, 2/7))

    (test_set, val_set, train_set) = next(cv_split_generator)
    train_ds = MultiOrbitDataset(*train_set, device='cuda')
    val_ds = MultiOrbitDataset(*val_set, device='cuda')
    test_ds = MultiOrbitDataset(*test_set, device='cuda')

    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # model_params['width'] = 32
    models = [MSDRegressionModel(**model_params) for _ in range(6)]

    [model.set_normalization(DataLoader(test_ds, batch_size=100, sampler=ValSampler(len(test_ds), min(len(test_ds), 2000))))
        for model in models]

    [model.msd.load_state_dict(
        torch.load(sorted(glob.glob(f'model_weights/MSD_phantoms/MSD_d80_P_scratch_CV{cv}*/model*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for cv, model in zip(['01', '03', '05'], models[:3])]
    [model.msd.load_state_dict(
        torch.load(sorted(glob.glob(f'model_weights/MSD_phantoms/MSD_d80_P_transfer_CV01_CV{cv}*/model*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for cv, model in zip(['01', '03', '05'], models[3:])]

    layers = [range(1,81) for model in models]

    reps = get_model_representation(models, get_patches(test_ds, 10), layers, n_samples=10, sample_rate=10)[0]

    ax.plot(get_pwcca_dist(reps[0], reps[1]))
    sys.exit()


    # dists_scratch = np.stack([get_pwcca_dist(reps[0], reps[1]),
    #                   get_pwcca_dist(reps[1], reps[2]),
    #                   get_pwcca_dist(reps[0], reps[2])], axis=0)

    # dists_transfer = np.stack([get_pwcca_dist(reps[3], reps[4]),
    #                   get_pwcca_dist(reps[4], reps[5]),
    #                   get_pwcca_dist(reps[3], reps[5])], axis=0)

    # ax.plot(dists_scratch.mean(0), label=f'UNet_f8 (scratch) ', c=colors[0])
    # ax.fill_between(range(dists_scratch.shape[1]), 
    #                     dists_scratch.mean(0) -dists_scratch.std(0),
    #                     dists_scratch.mean(0) +dists_scratch.std(0), color=colors[0], alpha=.2)

    # ax.plot(dists_transfer.mean(0), label=f'UNet_f8 (transfer)', c=colors[1])
    # ax.fill_between(range(dists_transfer.shape[1]), 
    #                     dists_transfer.mean(0) -dists_transfer.std(0),
    #                     dists_transfer.mean(0) +dists_transfer.std(0), color=colors[1], alpha=.2)

    dists_CV = np.stack([get_pwcca_dist(reps[0], reps[3]),
                        get_pwcca_dist(reps[1], reps[4]),
                        get_pwcca_dist(reps[2], reps[5])], axis=0)

    for i in range(3):
        ax.plot(dists_CV[i], label=f'CV{i*2+1:0>d}', c=colors[i])
        # ax.fill_between(range(dists_CV.shape[1]), 
        #                     dists_CV.mean(0) -dists_CV.std(0),
        #                     dists_CV.mean(0) +dists_CV.std(0), color=colors[i], alpha=.2)

    #            
    # for i, d in enumerate([16, 32, 64]):   
    #     model_params['width'] = d
    #     models = [UNetRegressionModel(**model_params) for _ in range(3)]

    #     [model.set_normalization(DataLoader(test_ds, batch_size=100, sampler=ValSampler(len(test_ds), min(len(test_ds), 2000))))
    #      for model in models]

    #     [model.msd.load_state_dict(
    #         torch.load(sorted(glob.glob(f'model_weights/UNet_baseline/UNet_f{d}_W_CV{cv}*/best*.h5'), key=_nat_sort)[-1], map_location='cpu'))
    #         for cv, model in zip(['01', '13', '25'], models)]
            
    #     layers = [get_layers(model) for model in models]
    #     # layers = [range(1,d+1) for _ in models]T

    #     reps = get_model_representation(models, test_ds, layers, n_samples=10, sample_rate=10)

    #     dists = np.stack([get_pwcca_dist(reps_f8[0], reps[0],),
    #                       get_pwcca_dist(reps_f8[0], reps[1],),
    #                       get_pwcca_dist(reps_f8[0], reps[2],)], axis=0)

    #     ax.plot(dists.mean(0), c=colors[i], label=f'UNet_f{d}')
    #     ax.fill_between(range(dists.shape[1]), 
    #                        dists.mean(0) -dists.std(0),
    #                        dists.mean(0) +dists.std(0), color=colors[i], alpha=.2)
            
    ax.set_xlabel('Layers')
    ax.set_xticks(range(9))
    ax.set_xticklabels(['inconv', 'down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4'], rotation=30)
    ax.set_ylabel('PWCCA distance')
    plt.legend()
    plt.savefig('outputs/PWCCA.png')
    plt.close() 
