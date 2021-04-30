from tqdm import trange, tqdm

import numpy as np
import torch
from matplotlib import cm
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.image_dataset import MultiOrbitDataset
from src.models import UNetRegressionModel
from src.train_model import *
from src.test_model import set_rcParams
from src.transfer_model import shuffle_weights, mean_var_init
from src.utils import ValSampler, _nat_sort
from SVCCA import cca_core
from SVCCA.pwcca import compute_pwcca

""""Uses code based on https://arxiv.org/abs/1706.05806, https://github.com/google/svcca"""


class MSDForwardLogger():
    """Logs forward propagated activation maps in MS-D Nets"""

    def __init__(self, layers, n_samples, sample_rate):
        self.layers = layers
        self._representation = torch.zeros(len(layers), n_samples *sample_rate **2, device='cuda')
        # number of samples per dimension for each activation map, gathers sample_rate^2 per neuron activation (2D)
        self.sample_rate = sample_rate
        self.current_ind = 0
    
    def update_representation(self, layers_out):
        for i, layer_out in enumerate(layers_out):
            slc = [slice(i,i+1,1), slice(self.current_ind, self.current_ind +len(layer_out) *self.sample_rate **2, 1)]
            self._representation[slc] = layer_out.reshape(1, len(layer_out) *self.sample_rate **2)
            
        self.current_ind += len(layer_out) *self.sample_rate **2

    def get_representation(self, group_sz=5):
        """Return representation of all layers to be used for cca analysis"""
        return [self._representation[i:i+group_sz].cpu().numpy() for i in range(0, len(self._representation), group_sz)]


class UNetForwardLogger():
    """Logs forward propagated activation maps in UNets"""

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
        """Return representation of all layers to be used for cca analysis"""
        return [rep.cpu().numpy() for rep in self._representation]


def hook_MSD(model, layers, n_samples, sample_rate):
    """Hooks logger to the forward method of the layers"""

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
    """Hooks logger to the forward method of the layers"""

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


def get_patches(dataset, n_patches=100, batch_size=8): 

    return DataLoader(dataset, batch_size, sampler=ValSampler(len(dataset), n_patches), drop_last=False)


def get_unet_layers(model):
    """UNet layers to be hooked"""

    return [model.msd.inc,
            model.msd.down1, model.msd.down2, model.msd.down3, model.msd.down4,
            model.msd.up1, model.msd.up2, model.msd.up3, model.msd.up4,
            ]


def get_svcca_matrix(reps1, reps2, max_dims=None):
    """Computes the SVCCA similarity matrix for reps1 and reps2"""

    cca_matrix = np.zeros((len(reps1), len(reps2)))

    for i, rep1 in enumerate(reps1):
        for j, rep2 in enumerate(reps2):
            # dimensionality used for the computation
            n_dims = min(len(rep1), len(rep2)) if max_dims is None else min(max_dims, min(len(rep1), len(rep2)))

            rep1, rep2 = rep1 -rep1.mean(axis=1, keepdims=True), rep2 -rep2.mean(axis=1, keepdims=True)

            U1, s1, V1 = np.linalg.svd(rep1, full_matrices=False)
            U2, s2, V2 = np.linalg.svd(rep2, full_matrices=False)

            # use directions that explain 99% of variance
            explained_var = max( ((s1 **2 / (s1**2).sum()) > .01).sum(), ((s2 **2 / (s2**2).sum()) > .01).sum() )
            n_dims = min(n_dims, int(explained_var))

            svacts1 = np.dot(s1[:n_dims]*np.eye(n_dims), V1[:n_dims])
            # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
            svacts2 = np.dot(s2[:n_dims]*np.eye(n_dims), V2[:n_dims])
            # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)
            
            cca_coefs = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-18, verbose=False)['cca_coef1']
            cca_matrix[i, j] = cca_coefs.mean()

    return cca_matrix
    

def get_pwcca_dist(reps1, reps2):
    """Computes the PWCCA distance for reps1 and reps2"""
    
    assert len(reps1) == len(reps2)
    
    pwcca_dists = np.zeros(len(reps1))

    for i, (rep1, rep2) in enumerate(zip(reps1, reps2)):

            rep1, rep2 = rep1 -rep1.mean(axis=1, keepdims=True), rep2 -rep2.mean(axis=1, keepdims=True)

            pwcca_coef, *_ = compute_pwcca(rep1, rep2, epsilon=1e-7)
            pwcca_dists[i] = 1-pwcca_coef

    return pwcca_dists
