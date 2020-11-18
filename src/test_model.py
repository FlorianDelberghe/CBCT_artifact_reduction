import argparse
import glob
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity
from skimage.filters import threshold_multiotsu
from imageio import imread
from msd_pytorch import MSDRegressionModel
from torch.utils.data import DataLoader
from torch.utils.data import dataset
from tqdm import tqdm

from . import utils
from .image_dataset import ImageDataset, data_augmentation
from .unet_regr_model import UNetRegressionModel
from .utils import ValSampler, imsave, evaluate


def test(model, dataset):
    """Computes metrics ['MSE', 'SSIM', 'DSC'] for provifded model and sample of dataset"""

    te_dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), n_samples=100))

    metrics = np.empty((3, len(te_dl)))
    with evaluate(model):
        for i, (input_, target) in enumerate(te_dl):
            pred = model(input_)

            metrics[:,i] = mse(pred, target), ssim(pred, target), dsc(pred, target)                   

    return metrics.mean(axis=1), metrics.std(axis=1)


def noise_robustness(model, dataset, noise='gaussian', **kwargs):
    
    if noise.lower() == 'gaussian':
        noise_range = kwargs.get('noise_range', np.linspace(0, .05, 20, ))
        add_noise = gauss_noise
    elif noise.lower() == 'shot':
        noise_range = kwargs.get('noise_range', np.linspace(0, .1, 20))
        add_noise = shot_noise
    else:
        raise ValueError(f"Wrong noise argument: '{noise}'")
    
    te_dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), 100))

    metrics = np.empty((3, len(noise_range)))
    with evaluate(model):
        for j in range(len(noise_range)):
            metrics_ = np.empty((3, len(te_dl)))
            for i, (input_, target) in enumerate(te_dl):

                input_ = add_noise(input_, noise_range[j])
                pred = model(input_)

                metrics_[:,i] = mse(pred, target), ssim(pred, target), dsc(pred, target)             

            metrics[:,j] = metrics_.mean(axis=1)

    fig, ax = plt.subplots()
    ax.plot(noise_range, metrics[1], label='DSC', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    ax.plot(noise_range, metrics[2], label='SSIM', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])

    ax2 = ax.twinx()
    ax2.plot(noise_range, metrics[0], label='MSE', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])

    ax.legend(loc='upper right'); ax2.legend(loc='upper right')
    ax.set_ylim([0, None])
    ax2.set_ylim([0, None])
    # ax2.set_ylim([0, 1.1*metrics[0].max()])
    plt.savefig('outputs/noise_robustness.png')
    plt.close(fig)


def compare_models(model1, model2, dataset):
    
    te_dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), n_samples=100))

    metrics = np.empty((6, len(te_dl)))
    with evaluate(model1), evaluate(model2):
        for i, (input_, target) in enumerate(te_dl):
            pred1 = model1(input_)
            pred2 = model2(input_)

            metrics[:2,i] = mse(pred1, target), mse(pred2, target)
            metrics[2:4,i] = ssim(pred1, target), ssim(pred2, target)
            metrics[4:6,i] = dsc(pred1, target), dsc(pred2, target)                        

        print("Model1 \n\tMSE:  {mse_mean:.4e} \n\tSSIM: {ssim_mean:.4f} \n\tDSC:  {dsc_mean:.4f}".format(
            mse_mean=metrics[0].mean(), ssim_mean=metrics[2].mean(), dsc_mean=metrics[4].mean()
        ))
        print("Model2 \n\tMSE:  {mse_mean:.4e} \n\tSSIM: {ssim_mean:.4f} \n\tDSC:  {dsc_mean:.4f}".format(
            mse_mean=metrics[1].mean(), ssim_mean=metrics[3].mean(), dsc_mean=metrics[5].mean()
        ))


# ===== Noise Functions ===== #
def add_noise(x, which='gaussian'):
    pass


def gauss_noise(x, mean=0, std=.1):

    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Wrong input type, is {type(x)}")

    if mean == std == 0:
        return x

    return x.add(torch.empty(x.size(), device=x.device).normal_(mean, std))

def shot_noise(x, pix_frac=.001):

    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Wrong input type, is {type(x)}")

    if pix_frac == 0:
        return x

    n_pix = int(x[0,0].numel() *pix_frac)
    pix_coords = (torch.tensor(random.choices(range(x.size(-2)), k=n_pix)),
                  torch.tensor(random.choices(range(x.size(-1)), k=n_pix)))

    split_id = random.randint(n_pix //2 -n_pix //5, n_pix//2 +n_pix //5)

    x[:,:,pix_coords[0][:split_id], pix_coords[1][:split_id]] = x.mean() +x.std() *4
    x[:,:,pix_coords[0][split_id:], pix_coords[1][split_id:]] = x.mean() -x.std() *4

    return x

# ===== Metrics ===== #
def compute_metric(pred, target, metric='MSE'):

    if metric.lower() == 'mse':
        return mse(pred, target)
    elif metric.lower() == 'rmse':
        return rmse(pred, target)
    elif metric.lower() == 'norm_mse':
        return norm_mse(pred, target)
    elif metric.lower() == 'dsc':
        return dsc(pred, target)
    elif metric.lower() == 'ssim':
        return ssim(pred, target)
    else:
        raise ValueError(f"Unknown metric: '{metric}'")


def mse(x, y):
    """MSE loss x: image, y: target_image"""
    if isinstance(x, torch.Tensor):
        with torch.no_grad(): 
            return ((x-y) **2).mean().item()

    return ((x-y) **2).mean()

def rmse(x, y):
    return np.sqrt(mse(x, y))

def norm_mse(x, y):
    # normalize wrt to target image
    d_range = y.max() - y.min()
    if isinstance(x, torch.Tensor):
        with torch.no_grad(): 
            return (((x-y) /d_range) **2).mean().item()

    return (((x-y) /d_range)**2).mean()

def dsc(x, y, n_classes=3):

    def otsu(image, n_classes):
        thresholds = threshold_multiotsu(image, classes=n_classes)

        return np.digitize(image, bins=thresholds).astype('uint8')        
    
    def compute_dsc(i, seg_x, seg_y):
        return 2 * np.logical_and((seg_x == i), (seg_y == i)).sum() /((seg_x == i).sum() + (seg_y == i).sum())


    if isinstance(x, torch.Tensor): x = x.cpu().squeeze(1).numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().squeeze(1).numpy()

    if x.ndim == 3:
        dsc_coeffs = np.empty(len(x), dtype='float32')

        for i in range(len(x)):
            seg_x, seg_y = otsu(x[i], n_classes), otsu(y[i], n_classes)
            dsc_coeffs[i] = np.array([compute_dsc(i, seg_x, seg_y) for i in np.unique(seg_y)]).mean()
        
        return dsc_coeffs.mean()

    if x.ndim == 2:
        seg_x, seg_y = otsu(x, n_classes), otsu(y, n_classes)

        return np.array([compute_dsc(i, seg_x, seg_y) for i in np.unique(seg_y)]).mean()


def ssim(x, y):

    if isinstance(x, torch.Tensor): x = x.cpu().squeeze(1).numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().squeeze(1).numpy()

    d_range = y.max() -y.min()
    
    # Batch images
    if x.ndim == 3:
        ssim = sum(map(lambda i: structural_similarity(x[i], y[i], data_range=d_range), range(len(x))))

        return ssim / len(x)

    elif x.ndim == 2:
        return structural_similarity(x, y, data_range=d_range)

    else:
        raise ValueError(f"Number of dims is not compatible with ssim function is {x.ndims}, must be '4' (Tensors), '2','3' (np.ndarray)")

    # dynamic range of input values (aquired on 12bits flat panel detector)
    # L = 2 **12 -1
    # L = max(x.max()-x.min(), y.max()-y.min())
    # k1, k2 = 0.01, 0.03
    # c1, c2 = (k1*L) **2, (k2*L) **2
    # c3 = c2 /2
    
    # mean_x, mean_y = x.mean(), y.mean()
    # (std_x, cov_xy), (_, std_y) = np.cov(x.ravel(), y.ravel())
    
    # luminance = (2*mean_x*mean_y +c1) / (mean_x **2 + mean_y**2 +c1)
    # contrast = (2*std_x*std_y +c2) / (std_x **2 + std_y**2 +c2)
    # structure = (cov_xy + c3) / (std_x*std_y + c3)

    # return luminance * contrast * structure
        
    
