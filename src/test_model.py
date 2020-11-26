import glob
import os
import pathlib
import random
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from skimage.filters import threshold_multiotsu
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from . import utils
from .image_dataset import ImageDataset
from .unet_regr_model import UNetRegressionModel
from .utils import ValSampler, evaluate, imsave


def test(model, dataloader):
    """Computes metrics ['MSE', 'SSIM', 'DSC'] for provifded model and sample of dataset"""

    metrics = np.empty((3, len(dataloader)))
    with evaluate(model):
        for i, (input_, target) in enumerate(dataloader):
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


def compare_models(models, dataloader, names=None, title=None):
    
    n_models = len(models)
    names = [f'Model{i+1}' for i in range(n_models)] if names is None else names

    metrics = np.empty((3 *n_models, len(dataloader)))
    with evaluate(*models):
        for i, (input_, target) in enumerate(dataloader):
            preds = [model(input_) for model in models]

            metrics[:n_models,i] = [mse(pred, target) for pred in preds]
            metrics[n_models:2*n_models,i] = [ssim(pred, target) for pred in preds]
            metrics[2*n_models:,i] = [dsc(pred, target) for pred in preds]                    

    fig, axes = plt.subplots(ncols=3, figsize=(15,5))
    
    metric_names = ['MSE', 'SSIM', 'DSC']
    for i in range(3):
        axes[i].boxplot(metrics[n_models*i:n_models*i+n_models].T, positions=np.arange(n_models))
        axes[i].set_xticklabels(names)
        axes[i].set_title(metric_names[i])

    plt.suptitle(title)
    plt.savefig('outputs/model_comparison.png')


def plot_metrics_evolution(model, state_dicts, dataset, title=None,
                           filename='outputs/metrics_evolution.png'):

    n_samples = 100
    metrics = np.zeros((3, len(state_dicts), n_samples))
    # Loading the state_dicts from files
    if isinstance(state_dicts[0], (str, pathlib.Path)):
        state_dicts = (torch.load(state_dict) for state_dict in state_dicts)
    te_dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), n_samples=n_samples))

    with evaluate(model):
        for i, state_dict in enumerate(state_dicts):
            model.msd.load_state_dict(state_dict)
            metrics[:,i,:] = np.array(
                [[mse(model(input_), target), ssim(model(input_), target), dsc(model(input_), target)]
                 for input_, target in te_dl]).T

    metric_names = ['MSE', 'SSIM', 'DSC']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(10,10))
    ax2 = ax.twinx()

    xticks = np.arange(metrics.shape[1])
    line1 = ax.plot(xticks, metrics[0].mean(1), c=colors[0])
    line2 = ax2.plot(xticks, metrics[1].mean(1), c=colors[1])
    line3 = ax2.plot(xticks, metrics[2].mean(1), c=colors[2])

    ax.plot(xticks, metrics[0].mean(1) +metrics[0].std(1)/10, c=colors[0], alpha=.3)
    ax.plot(xticks, metrics[0].mean(1) -metrics[0].std(1)/10, c=colors[0], alpha=.3)
    ax2.plot(xticks, metrics[1].mean(1) +metrics[1].std(1)/10, c=colors[1], alpha=.3)
    ax2.plot(xticks, metrics[1].mean(1) -metrics[1].std(1)/10, c=colors[1], alpha=.3)
    ax2.plot(xticks, metrics[2].mean(1) +metrics[2].std(1)/10, c=colors[2], alpha=.3)
    ax2.plot(xticks, metrics[2].mean(1) -metrics[2].std(1)/10, c=colors[2], alpha=.3)
    
    plt.legend(line1+line2+line3, metric_names)
    ax.set_xticks(xticks[::2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax2.set_ylabel('Similarity Metrics')
    ax2.set_ylim([0,1])
    ax2.set_yticks(np.linspace(0,1,11))
    ax2.yaxis.grid()
    ax.set_title(title)
    plt.savefig(filename)
            

def pred_test_sample(model, dataset, filename='outputs/sample_pred.png'):

    te_dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), 10))
      
    with evaluate(model):
        preds = torch.cat([torch.cat([sample, model(sample), truth], dim=-2)
                            for sample, truth in te_dl], dim=-1)

    imsave(filename, preds.cpu().squeeze().numpy())


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
        
    
