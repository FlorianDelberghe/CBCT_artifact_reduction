import glob
import os
import random
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from skimage.filters import threshold_multiotsu
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .utils import ValSampler, evaluate, imsave

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

@contextmanager
def set_rcParams(**kwargs):
    
    plt_params = {'figure.figsize': (10,10),
                  'xtick.labelsize': 24,
                  'ytick.labelsize': 24,
                  'axes.titlesize': 30,
                  'axes.labelsize': 24}

    if kwargs is not None: plt_params.update(kwargs)
    default_rcParams = dict(map(lambda key: (key, plt.rcParams[key]), plt_params.keys()))

    plt.rcParams.update(plt_params)

    try: yield
    
    except: raise

    finally: 
        plt.rcParams.update(default_rcParams)
    
def _to_iterable(obj):
    """Makes sure object can be used as an iterable, allows support for for loops with list or single arguments inputs"""

    if isinstance(obj, Iterable): 
        return obj

    return [obj]


# ======================================= #
# =============== Metrics =============== #
# ======================================= #

def eval_init_metrics(metrics_names, dataset, n_samples=50):

    dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), n_samples), drop_last=False)

    metrics = _to_iterable(metrics_names)
    metrics_array = np.array([compute_metric(input_, target, metric) for metric in metrics for input_, target in dl]
                             ).reshape((len(metrics), n_samples))

    return metrics_array.mean(1)

def eval_metrics(models, metrics_names, dataset, n_samples=50):

    dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), n_samples), drop_last=False)

    models = _to_iterable(models)
    metrics = _to_iterable(metrics_names)

    metrics_arrays = []
    for model in tqdm(models):
        with evaluate(model):
            metrics_arrays.append(
                np.array([compute_metric(model(input_), target, metric)
                          for metric in metrics for input_, target in dl]
                         ).reshape((len(metrics), n_samples)))

    return np.stack(metrics_arrays, axis=0)

def get_metrics_stats(metrics):
    """metrics np.ndarray of shape [models, metrics, samples]"""
    return metrics.mean(-1), metrics.std(-1)

@set_rcParams()
def plot_metrics(metrics, model_names, metric_names, filename='metrics.png'):
    global colors
    
    fig, axes = plt.subplots(1, len(metric_names), figsize=(40,10))
    
    for i, name in enumerate(metric_names):
        axes[i].boxplot(metrics[:,i].T)
        axes[i].set_title(name)
        axes[i].set_xticklabels(model_names, rotation=30)

    plt.savefig(Path('outputs/') / filename)

@set_rcParams()
def plot_metrics_CV(metrics, model_names, metric_names, ref_metrics=None, filename='metrics_CV.png'):
    global colors
        
    fig, axes = plt.subplots(1, len(metric_names), figsize=(len(metric_names) *10,10))

    if ref_metrics is not None:
        for i, name in enumerate(metric_names):
            axes[i].plot([-.5, len(model_names)-.5], (ref_metrics[i],) *2, '--k', linewidth=4, alpha=.5)

    for j in range(len(metrics)):
        for i, name in enumerate(metric_names):

            axes[i].boxplot(metrics[j][:, i].T, positions=np.arange(len(metrics[j]))+(j-1)*.15, widths=.1,
                            boxprops={'color': colors[j]}, whiskerprops={'color': colors[j]},
                            capprops={'color': colors[j]}, medianprops={'color': 'k'})

            if j == len(metrics)-1:
                axes[i].set_title(name)
                axes[i].set_xticks(np.arange(len(metrics[j])))
                axes[i].set_xticklabels(model_names, rotation=30)

    plt.savefig(Path('outputs/') / filename)
        
def get_metrics_table(metrics, models_names, metrics_names, stdout=None):
    """metrics np.ndarray of shape [models, metrics, samples]"""
    print(metrics.shape)
    
    if stdout is not None:
        default_stdout = sys.stdout
        sys.stdout = stdout

    print(r'\begin{tabular}', end='')
    print('{'+ '{}'.format(' '.join(['l'] + ['c' for _ in range(len(models_names))])) +'}', end=' \\\\\n')
    print(' & ' + ' & '.join(models_names))

    for i, metric in enumerate(metrics_names):
        print(f'{metric.upper():4s} & ', end='')
        print(' & '.join([f'${metrics[j,i].mean():.4e} \\pm {metrics[j,i].std():.4e}$'
                          for j in range(len(metrics))]), end=' \\\\\n')

    print(r'\end{tabular}')


def compute_metric_evolution(model, metrics, state_dicts, dataset, n_samples=50):

    metrics_te = np.zeros((len(metrics), len(state_dicts)+1, n_samples))
    state_dicts = (torch.load(state_dict, map_location='cpu') for state_dict in state_dicts)

    dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), n_samples), drop_last=False)

    with evaluate(model):
        metrics_te[:,0] = np.array([[compute_metric(model(input_), target, metric) for metric in metrics]
                                   for input_, target in dl]).T

        for i, state_dict in enumerate(state_dicts):
            model.msd.load_state_dict(state_dict)

            metrics_te[:,i+1] = np.array([[compute_metric(model(input_), target, metric) for metric in metrics]
                                   for input_, target in dl]).T

    return metrics_te


# ======================================= #
# ============ Representation =========== #
# ======================================= #






































def noise_robustness(model, dataloader, noise='gaussian', **kwargs):
    
    if noise.lower() == 'gaussian':
        noise_range = kwargs.get('noise_range', np.linspace(0, .05, 20, ))
        add_noise = gauss_noise
    elif noise.lower() == 'shot':
        noise_range = kwargs.get('noise_range', np.linspace(0, .1, 20))
        add_noise = shot_noise
    else:
        raise ValueError(f"Wrong noise argument: '{noise}'")

    metrics = np.empty((3, len(noise_range)))
    with evaluate(model):
        for j in range(len(noise_range)):
            metrics_ = np.empty((3, len(dataloader)))
            for i, (input_, target) in enumerate(dataloader):

                input_ = add_noise(input_, std=noise_range[j])
                pred = model(input_)

                metrics_[:,i] = mse(pred, target), ssim(pred, target), dsc(pred, target)             

            metrics[:,j] = metrics_.mean(axis=1)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(10,10))
    ax2 = ax.twinx()

    line1 = ax.plot(noise_range, metrics[0], c=colors[0])
    line2 = ax2.plot(noise_range, metrics[1], c=colors[1])
    line3 = ax2.plot(noise_range, metrics[2], c=colors[2])

    plt.legend(line1+line2+line3, ('MSE', 'SSIM', 'DSC'), loc='upper right')
    ax.set_xlabel('Noise range')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Similarity Metrics')
    ax.set_ylim([0, None])
    ax2.set_ylim([0, 1])
    ax2.yaxis.grid()
    ax.set_title(kwargs.get('title', None))
    plt.savefig('outputs/noise_robustness.png')
    plt.close(fig)




@set_rcParams()
def plot_metrics_evolution(metrics, metrics_names, title=None, filename='outputs/metrics_evolution.png'):
    global colors

    fig, ax = plt.subplots(figsize=(10,10))
    ax2 = ax.twinx()

    xticks = np.arange(metrics.shape[1])
    line1 = ax.plot(xticks, metrics[0].mean(1) *metrics[3].mean(1).max() /metrics[0].mean(1).max(), c=colors[0])
    line2 = ax2.plot(xticks, metrics[1].mean(1), c=colors[1])
    line3 = ax2.plot(xticks, metrics[2].mean(1), c=colors[2])
    line4 = ax.plot(xticks, metrics[3].mean(1) , c=colors[3])

    ax.fill_between(xticks, metrics[0].mean(1) -metrics[0].std(1), metrics[0].mean(1) +metrics[0].std(1), color=colors[0], alpha=.2)
    ax.fill_between(xticks, metrics[3].mean(1) -metrics[3].std(1), metrics[3].mean(1) +metrics[3].std(1), color=colors[3], alpha=.2)
    ax2.fill_between(xticks, metrics[1].mean(1) -metrics[1].std(1), metrics[1].mean(1) +metrics[1].std(1), color=colors[1], alpha=.2)
    ax2.fill_between(xticks, metrics[2].mean(1) -metrics[2].std(1), metrics[2].mean(1) +metrics[2].std(1), color=colors[2], alpha=.2)
    
    plt.legend(line1+line2+line3+line4, metrics_names)
    ax.set_xticks(xticks[::2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax2.set_ylabel('Similarity Metrics')
    ax2.set_ylim([0,1])
    ax2.set_yticks(np.linspace(0,1,11))
    ax2.yaxis.grid()
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()


def compare_loss(*folders, names=None):

    # loss_files = [next(Path(folder).glob('losses.txt') ) for folder in folders]
    loss_files = [next(folder.glob('losses.txt') ) for folder in folders]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(15,15))

    for i, loss_file in enumerate(loss_files):

        with loss_file.open('rt') as f:
            for j, line in enumerate(f):

                loss_name = line.strip()
                iters = list(map(int, next(f).strip().split(',')))
                loss = list(map(float, next(f).strip().split(',')))

                plt.plot(iters, loss, color=colors[i], label=loss_name, linestyle=['dashed', None][j%2])

    lines = plt.gca().get_lines()
    if names is None:
        plt.legend(lines[:2], ['training loss', 'validation loss'])
    else:
        plt.legend(lines[:2] + lines[1::2], ['training loss', 'validation loss'] + names)

    plt.xlabel('Batch')
    plt.ylim([0,9e-4])
    plt.ylabel('Loss')
    plt.savefig('outputs/loss.png')
            
@utils.set_seed
def pred_test_sample(model, dataset, filename='outputs/sample_pred.png'):

    te_dl = DataLoader(dataset, batch_size=1, sampler=ValSampler(len(dataset), 10))
      
    with evaluate(model):
        preds = torch.cat([torch.cat([sample, model(sample), truth], dim=-2)
                            for sample, truth in te_dl], dim=-1)

    imsave(filename, preds.cpu().squeeze().numpy())


def SVCCA_analysis(neurons, inputs):
    pass


# ===== Noise Functions ===== #
# def add_noise(x, which='gaussian'):
#     pass


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
def compute_metric(pred, target, metric):

    metrics_fns = dict(map(lambda fn: (fn.__name__, fn), 
                           [mse, rmse, norm_mse, dsc, ssim, psnr]))
    
    return metrics_fns[metric.lower()](pred, target)
    
def mse(x, y):
    """MSE loss x: image, y: target_image"""

    def _mse(x, y): return ((x-y) **2).mean()

    if isinstance(x, torch.Tensor):
        with torch.no_grad(): 
            return _mse(x, y).item()

    return _mse(x, y)

def rmse(x, y):
    return np.sqrt(mse(x, y))

def norm_mse(x, y):

    # normalize wrt to target image
    def _norm_mse(x, y): return (((x-y) /(y.max()-y.min())) **2).mean()

    if isinstance(x, torch.Tensor):
        with torch.no_grad(): 
            return _norm_mse(x, y).item()

    return _norm_mse(x, y)

def dsc(x, y, n_classes=3):
    
    def otsu(image, n_classes):
        try:
            thresholds = threshold_multiotsu(image, classes=n_classes)
        except ValueError:
            # When output is constant can't apply otsu thresholding
            thresholds = np.ones(1)

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

def psnr(x, y, max_I=None):

    def log_10(a): return np.log(a) /np.log(10)

    if isinstance(x, torch.Tensor): x = x.cpu().squeeze(1).numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().squeeze(1).numpy()
    
    # Max value of target image
    if max_I is None: max_I = y.max()

    return 10 *log_10(max_I **2 /mse(x, y))
        
    
