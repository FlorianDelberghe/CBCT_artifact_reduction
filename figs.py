import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread, mimsave
from matplotlib import cm, rc
from matplotlib.ticker import (AutoMinorLocator, FixedLocator, FuncFormatter,
                               MultipleLocator)
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from cca import *
from src.image_dataset import (MultiOrbitDataset, PoissonNoise, RandomHFlip,
                               TransformList)
from src.models import UNetRegressionModel
from src.test_model import *
from src.utils import _nat_sort
from src.utils.nn import TrSampler, ValSampler, evaluate


def get_models():
    """Easy way to get all the models and their names for tests"""

    model_params = {'c_in': 1, 'c_out': 1,
                    'width': 1, 'dilations': [1,2,4,8,16]}

    model_d30 = MSDRegressionModel(depth=30, **model_params)
    model_d80 = MSDRegressionModel(depth=80, **model_params) 
    
    del model_params['width']
    model_f8 = UNetRegressionModel(width=8, **model_params)
    model_f16 = UNetRegressionModel(width=16, **model_params)
    model_f32 = UNetRegressionModel(width=32, **model_params)
    model_f64 = UNetRegressionModel(width=64, **model_params)

    models = (model_d30, model_d80) + (model_f8, model_f16, model_f32, model_f64)
    model_names = [f'MSD_d{d}' for d in [30, 80]] + [f'UNet_f{f}' for f in [8, 16, 32, 64]]

    return models, model_names


# ==================== #
# === Metrics Eval === #
# ==================== #

def eval_metrics_CV():
    """Computes metrics over different inits with cross validation, creates box plots usefull to check generalization/robustness"""

    target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialNoisy/')
    # target_ims, input_ims = utils.load_walnut_ds()

    ds = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    models_dir = Path('/media/beta/florian/model_weights')
    metrics = ['SSIM', 'DSC', 'PSNR']

    models, model_names = get_models()
    [model.set_normalization(norm_dl) for model in models]

    print(eval_init_metrics(metrics, ds))
    metrics_te = []
    for cv in ['01', '03', '05']:
        # if not cv == '01': continue
        task = 'meanVarInit' 

        [model.msd.load_state_dict(
            torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for model, d in zip(models[:2], [30, 80])]

        [model.msd.load_state_dict(
            torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for model, f in zip(models[-4:], [8, 16, 32, 64])] 

        metrics_te.append(eval_metrics(models, metrics, ds, 50))

    plot_metrics_CV(metrics_te, model_names, metrics, ref_metrics=eval_init_metrics(metrics, ds), filename=f'metrics_P_CV{cv}_best.png')


@set_rcParams()
def eval_metrics_samples():
    """Evaluates models for different training setups, here with changing amounts of training samples"""

    global colors

    target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialNoisy/')
    # target_ims, input_ims = utils.load_walnut_ds()

    ds = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    models_dir = Path('/media/beta/florian/model_weights')
    metrics = ['SSIM', 'DSC', 'PSNR']

    models, model_names = get_models()
    [model.set_normalization(norm_dl) for model in models]
    
    for cv in ['01', '03', '05']:
        metrics_samples = []
        for n_samples in [16, 128, 1024]:
            task = '_shuffle'

            [model.msd.load_state_dict(
                torch.load(sorted(models_dir.glob(f'models_Pn_CV{cv}/MSD_d{d}_Pn{n_samples}{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
            for model, d in zip(models[:2], [30, 80])]

            [model.msd.load_state_dict(
                torch.load(sorted(models_dir.glob(f'models_Pn_CV{cv}/UNet_f{f}_Pn{n_samples}{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
            for model, f in zip(models[-4:], [8, 16, 32, 64])]  

            metrics_samples.append(eval_metrics(models, metrics, ds))

        task = 'shuffle'
        [model.msd.load_state_dict(
            torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for model, d in zip(models[:2], [30, 80])]

        [model.msd.load_state_dict(
            torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for model, f in zip(models[-4:], [8, 16, 32, 64])] 

        metrics_samples.append(eval_metrics(models, metrics, ds))

        
        fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) *10,10))
        cmap = cm.get_cmap('coolwarm', 4)(range(4))

        box_plot_handles = [None for _ in range(len(metrics_samples))]
        for j in range(len(metrics_samples)):
            for i, name in enumerate(metrics):                
                if i == 1:
                    box_plot_handles[j] = axes[i].boxplot(metrics_samples[j][:, i].T, positions=np.array([0,1, 3,4,5,6])-.21+ j*.14, widths=.1,
                                                        boxprops={'color': cmap[j], 'linewidth': 2}, whiskerprops={'color': cmap[j], 'linewidth': 2},
                                                        capprops={'color': cmap[j], 'linewidth': 2}, medianprops={'color': 'k'},
                                                        showfliers=False)["boxes"][0]
                else:
                    axes[i].boxplot(metrics_samples[j][:, i].T, positions=np.array([0,1, 3,4,5,6])+(j-1)*.15, widths=.1,
                                    boxprops={'color': cmap[j], 'linewidth': 2}, whiskerprops={'color': cmap[j], 'linewidth': 2},
                                    capprops={'color': cmap[j], 'linewidth': 2}, medianprops={'color': 'k'},
                                    showfliers=False)

        metrics_samples = np.array(metrics_samples)
        for i, ax in enumerate(axes):
            for j in range(metrics_samples.shape[1]):
                # print(np.median(metrics_samples[:,j,i], -1))
                ax.plot(np.array([-.21,-.07,.07,.21]) +np.array([0,1, 3,4,5,6])[j] +.05, np.median(metrics_samples[:,j,i], -1), '-k', linewidth=.7, alpha=.7)

        ref_metrics = eval_init_metrics(metrics, ds)
        for i, name in enumerate(metrics):
            axes[i].plot([-.5, len(model_names)+.5], (ref_metrics[i],) *2, '--k', linewidth=4, alpha=.5)
            axes[i].set_title(name)
            axes[i].set_xticks(np.array([0,1, 3,4,5,6]))
            axes[i].set_xticklabels(model_names, rotation=30)

        axes[1].legend(box_plot_handles, ['16', '128', '1024', f'{4*709}    '], loc='lower right')
        plt.savefig(Path('outputs/') / f'metrics_P_CV{cv}_samples_shuffle_best.png')


@set_rcParams()
def fig_convergence():
    """Study convergence by observing how models converge from the first to the last epoch"""

    global colors

    target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialNoisy/')

    ds = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    models_dir = Path('/media/beta/florian/model_weights')
    metrics = ['SSIM', 'DSC', 'PSNR']

    models, model_names = get_models()  
    [model.set_normalization(norm_dl) for model in models]

    msd_cmap = cm.get_cmap('Purples', 6)(range(3,6,1))
    unet_cmap = cm.get_cmap('Oranges', 6)(range(3,6,1))

    for cv in ['01', '03', '05']:

        n_samples = 16
        mean_metrics = []

        ref_metrics = eval_init_metrics(metrics, ds)
        
        for e in range(5):
            fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics)*7.5,7.5))

            for i, name in enumerate(metrics):
                axes[i].plot([-.5, len(model_names)+.5], (ref_metrics[i],) *2, '--k', linewidth=3, alpha=.5)
                axes[i].set_title(name)
                axes[i].set_xticks(np.array([0,1, 3,4,5,6]))
                axes[i].set_xticklabels(model_names, rotation=30)

            for k, task in enumerate(['', '_transfer', '_shuffle']):
                if k == 0: legend_handles = []

                # Load models' state dicts after the first epoch
                [model.msd.load_state_dict(
                    torch.load(sorted(models_dir.glob(f'models_Pn_CV{cv}/MSD_d{d}_Pn{n_samples}{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[e], map_location='cpu'))
                for model, d in zip(models[:2], [30, 80])]

                [model.msd.load_state_dict(
                    torch.load(sorted(models_dir.glob(f'models_Pn_CV{cv}/UNet_f{f}_Pn{n_samples}{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[e], map_location='cpu'))
                for model, f in zip(models[-4:], [8, 16, 32, 64])]  

                mean_metrics.append(np.array(eval_metrics(models, metrics, ds)).mean(-1))

                for j in range(len(models)):
                    for i in range(len(metrics)):
                        cmap = msd_cmap if j < 2 else unet_cmap  
                        # axes[i].scatter(np.array([0,1, 3,4,5,6])[j] -.21 +.21*k, mean_metrics_1e[j,i], color=cmap[k], linewidth=5)
                        if e == 0: 
                            axes[i].arrow(np.array([0,1, 3,4,5,6])[j] -.21 +.21*k, mean_metrics[k][j,i], 0, 0,
                                                length_includes_head=False, width=.19, head_width=.3, 
                                                head_length=[.05,.05,1.5][i], color=cmap[k], alpha=.8)
                        else: 
                            axes[i].arrow(np.array([0,1, 3,4,5,6])[j] -.21 +.21*k, mean_metrics[k][j,i], 0, mean_metrics[e*3+k][j,i]-mean_metrics[k][j,i],
                                                length_includes_head=False, width=.19, head_width=.3, 
                                                head_length=[.05,.05,1.5][i], color=cmap[k], alpha=.8)

            cmap = cm.get_cmap('gray', 6)(range(2,-1,-1))
            legend_handles = [axes[2].arrow(0,0,0,0, width=0, head_width=0, color=cmap[i], alpha=.8) for i in range(3)]
            axes[1].legend(legend_handles, ['scratch', 'transfer', 'shuffle'], loc='lower left')

            for i in range(len(axes)):
                axes[i].set_ylim([[0,1],[0,1],[15,40]][i])

            plt.savefig(f'outputs/metrics_delta_Pn{n_samples}_CV{cv}_{e+1}e.png', bbox_inches='tight')

        frames = sorted(Path('outputs/').glob(f'metrics_delta_Pn{n_samples}_CV{cv}_[0-9]e.png'), key=_nat_sort)
        print(frames)
        frames = [imread(frame) for frame in frames]

        mimsave('outputs/convergence.gif', np.array(frames), fps=3)
            
        sys.exit()


@set_rcParams()
def memory_generalization():
    """Generalization of models to the walnut dataset while training on the phatom one"""

    target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialNoisy/')
    # target_ims, input_ims = utils.load_walnut_ds()

    ds = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    models_dir = Path('/media/beta/florian/model_weights')
    metrics = ['SSIM', 'DSC', 'PSNR']

    models, model_names = get_models()
    [model.set_normalization(norm_dl) for model in models]

    cv = '01'
    task = 'transfer_CV01'

    msd_cmap = cm.get_cmap('Purples', 4)(range(2,4,1))
    unet_cmap = cm.get_cmap('Oranges', 8)(range(4,8,1))
    cmap = np.concatenate([msd_cmap, unet_cmap], 0)


    metrics_phantoms = np.zeros((len(metrics), len(models), 4))

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_baseline/MSD_d{d}_W_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_baseline/UNet_f{f}_W_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_phantoms[...,0] = eval_metrics(models, metrics, ds, 4).mean(-1).T


    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_phantoms[...,1] = eval_metrics(models, metrics, ds, 4).mean(-1).T


    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_phantoms[...,2] = eval_metrics(models, metrics, ds, 4).mean(-1).T


    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[-1], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[-1], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_phantoms[...,3] = eval_metrics(models, metrics, ds, 4).mean(-1).T

    fig, axes = plt.subplots(1,3, figsize=(25,7.5))

    for i, ax in enumerate(axes):
        ax.fill_between(range(4), metrics_phantoms[i].min(0), metrics_phantoms[i].max(0), color='k', alpha=.2)
        # for j in range(metrics_phantoms.shape[1]):
        #     ax.plot(metrics_phantoms[i,j].mean(1), c=cmap[j], linestyle='dashed')

    models, model_names = get_models()
    target_ims, input_ims = utils.load_walnut_ds()

    ds = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  
    [model.set_normalization(norm_dl) for model in models]

    ref_metrics = eval_init_metrics(metrics, ds)
    for i, (ax, ref) in enumerate(zip(axes, ref_metrics)):
        ax.plot([0, 3], (ref,) *2, c='k', alpha=.7, linestyle='dashed', linewidth=3)


    metrics_walnuts = np.zeros((len(metrics), len(models), 4))

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_baseline/MSD_d{d}_W_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_baseline/UNet_f{f}_W_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_walnuts[...,0] = eval_metrics(models, metrics, ds, 4).mean(-1).T


    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_walnuts[...,1] = eval_metrics(models, metrics, ds, 4).mean(-1).T


    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_walnuts[...,2] = eval_metrics(models, metrics, ds, 4).mean(-1).T


    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[-1], map_location='cpu'))
    for model, d in zip(models[:2], [30, 80])]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/model_*.h5'), key=_nat_sort)[-1], map_location='cpu'))
    for model, f in zip(models[-4:], [8, 16, 32, 64])] 

    metrics_walnuts[...,3] = eval_metrics(models, metrics, ds, 4).mean(-1).T
    
    for i, ax in enumerate(axes):
        for j in range(metrics_phantoms.shape[1]):
            if i == 1:
                ax.plot(metrics_walnuts[i,j], c=cmap[j], linewidth=2, 
                    label=['MSD_d30', 'MSD_d80', 'UNet_f8', 'UNet_f16', 'UNet_f32', 'UNet_f64'][j])
            else:
                ax.plot(metrics_walnuts[i,j], c=cmap[j], linewidth=2)

        ax.set_title(metrics[i])
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['random init', 'first epoch', 'best epoch', 'last epoch'], rotation=20)

    axes[1].legend(loc='lower right')
    plt.savefig(f'outputs/generalization_{task}_CV{cv}.png', bbox_inches='tight')


def eval_metrics_mat():
    """Plots result of all metrics wrt to different init setups as a matrix"""    

    target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialNoisy/')

    ds = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    models_dir = Path('/media/beta/florian/model_weights')
    metrics = ['SSIM', 'DSC', 'PSNR']

    models, model_names = get_models()
    [model.set_normalization(norm_dl) for model in models]

    metrics_mat, metrics_std_te = np.zeros((15,6)), np.zeros((15,6))

    fig, ax = plt.subplots(figsize=(5*6,5))
    tasks = ['scratch', 'transfer', 'shuffle', 'mean-var']

    ref_metrics = [0.7825,  0.965, 33.1472258 ]
    print(ref_metrics)

    for i, task in enumerate(['scratch', 'transfer_CV01', 'shuffle', 'meanVarInit']):
        metrics_te = []
        for cv in ['01', '03', '05']:

            [model.msd.load_state_dict(
                torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
            for model, d in zip(models[:2], [30, 80])]

            [model.msd.load_state_dict(
                torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_{task}_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
            for model, f in zip(models[-4:], [8, 16, 32, 64])] 

            metrics_te.append(eval_metrics(models, metrics, ds, 50))
        
        metrics_te = np.array(metrics_te)
        for j in range(len(metrics)):
            if j == 2:
                metrics_mat[j*5+i+1] = metrics_te.mean((0,-1))[:,j] /38
                metrics_std_te[j*5+i+1] = metrics_te.std((0,-1))[:,j] /38
            else:
                metrics_mat[j*5+i+1] = metrics_te.mean((0,-1))[:,j]
                metrics_std_te[j*5+i+1] = metrics_te.std((0,-1))[:,j]


    metrics_mat[metrics_mat == 0] = np.nan
    metrics_std_te[metrics_mat == 0] = np.nan
    ax.matshow(metrics_mat, cmap='twilight', aspect=1/6, vmin=.64, vmax=1)

    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color='white')

    for (i, j), z in np.ndenumerate(metrics_mat):
        if i % 5 == 0: continue
        if i < 10: 
            if z < ref_metrics[i//5]: 
                ax.text(j, i, f'{z:0.3f}', ha='center', va='center', c='red')
            else:
                if z >= metrics_mat[i//5*5+1:i//5*5+5,j].max(): 
                    ax.text(j, i, f'{z:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z > .92)], fontweight='bold')
                else: 
                    ax.text(j, i, f'{z:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z > .92)])
        else:
            if z >= metrics_mat[i//5*5+1:i//5*5+5,j].max(): 
                    ax.text(j, i, f'{z*38:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z*38 > 35)], fontweight='bold')
            else:
                ax.text(j, i, f'{z*38:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z*38 > 35)])


    ax.xaxis.tick_top()
    ax.set_xticks(range(6))
    fontproperties = {'weight' : 'bold', 'size' : 10}
    ax.set_xticklabels(model_names, fontproperties)

    ax.tick_params(axis='both', which='minor', labelsize=8)


    # ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(FixedLocator([0,5,10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{metrics[pos]:<7s}'))
    # ax.set_yticklabels(ax.yaxis.get_majorticklabels(), fontdict={'weight' : 'bold', 'size' : 15})

    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, pos: f'{tasks[pos%4-1]}'))
    # ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, pos: print(pos)))
    plt.savefig('outputs/metrics_table.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(5*6,5))
    ax.matshow(metrics_mat, cmap='twilight', aspect=1/6, vmin=.64, vmax=1)

    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color='white')

    for (i, j), z in np.ndenumerate(metrics_mat):
        if i % 5 == 0: continue
        if i < 10: 
            if z < ref_metrics[i//5]: 
                ax.text(j, i, f'{z:0.3f} -+ {metrics_std_te[i,j]:0.3f}', ha='center', va='center', c='red')
            else:
                if z >= metrics_mat[i//5*5+1:i//5*5+5,j].max(): 
                    ax.text(j, i, f'{z:0.3f} -+ {metrics_std_te[i,j]:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z > .92)], fontweight='bold')
                else: 
                    ax.text(j, i, f'{z:0.3f} -+ {metrics_std_te[i,j]:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z > .92)])
        else:
            if z >= metrics_mat[i//5*5+1:i//5*5+5,j].max(): 
                    ax.text(j, i, f'{z*38:0.3f} -+ {metrics_std_te[i,j]*38:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z*38 > 35)], fontweight='bold')
            else:
                ax.text(j, i, f'{z*38:0.3f} -+ {metrics_std_te[i,j]*38:0.3f}', ha='center', va='center', c=[(0,0,0,1), (1,1,1,1)][int(not z*38 > 35)])

    ax.xaxis.tick_top()
    ax.set_xticks(range(6))
    fontproperties = {'weight' : 'bold', 'size' : 10}
    ax.set_xticklabels(model_names, fontproperties)

    ax.tick_params(axis='both', which='minor', labelsize=8)


    # ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(FixedLocator([0,5,10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{metrics[pos]:<7s}'))
    # ax.set_yticklabels(ax.yaxis.get_majorticklabels(), fontdict={'weight' : 'bold', 'size' : 15})

    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, pos: f'{tasks[pos%4-1]}'))
    
    plt.savefig('outputs/metrics_table_std.png', dpi=300, bbox_inches='tight')


# ==================== #
# ======= CCA ======== #
# ==================== #

@set_rcParams()
def svcca_over_training():
    """Evolution of SVCCA similarity matrix over training"""

    target_ims, input_ims = utils.load_phantom_ds('PhantomsRadialNoisy')

    ds = MultiOrbitDataset(input_ims, target_ims, data_augmentation=False, vert_sym=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))

    patches = get_patches(ds, 50) 

    for width in [80]:

        model_params = {'c_in': 1, 'c_out': 1, 'width': 1, 'depth': width, 'dilations': [1,2,4,8,16]}            

        fig, axes = plt.subplots(1,4, figsize=(22,5))

        ref_model, comp_model = MSDRegressionModel(**model_params), MSDRegressionModel(**model_params)
        ref_model.set_normalization(norm_dl)
        comp_model.set_normalization(norm_dl)

        init_state_dict = torch.load(sorted(models_dir.glob(f'MSD_baseline/MSD_d{width}_W_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
        first_state_dict = torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{width}_P_scratch_CV01_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu')
        best_state_dict = torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{width}_P_scratch_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
        last_state_dict = torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{width}_P_scratch_CV01_*/model_*.h5'), key=_nat_sort)[-1], map_location='cpu')

        ref_model.msd.load_state_dict(best_state_dict)

        for i, state_dict in enumerate([init_state_dict, first_state_dict, best_state_dict, last_state_dict]):

            comp_model.msd.load_state_dict(state_dict)
            if i == 0:
                # shuffle_weights(comp_model.msd)
                pass

            # reps = get_model_representation([ref_model, comp_model], patches, 
            #                                 [get_unet_layers(ref_model), get_unet_layers(comp_model)], sample_rate=10)

            reps = get_model_representation([ref_model, comp_model], patches, 
                                            [range(1,81), range(1,81)], sample_rate=10)
        
            mat = axes[i].matshow(get_svcca_matrix(*reps), vmin=.2, vmax=1)
            if i == 0: axes[i].set_ylabel('BEST')
            axes[i].set_xlabel(['INIT', 'FIRST', 'BEST', 'LAST'][i])
            axes[i].set_title(['(a)', '(b)', '(c)', '(d)'][i], loc='left')

        # axes[0].set_yticks(rang)
        for ax in axes:
            ax.set_xticks([])
            ax.yaxis.tick_right()
            # ax.set_yticks(range(9))
            # ax.set_yticklabels(unet_layers)

        cax = fig.add_axes([.931, 0.1, .02, 0.8])
        plt.colorbar(mat, cax=cax)
        plt.savefig(f'outputs/svcca_training_MSD_d{width}_transfer_CV01.png', bbox_inches='tight')



@set_rcParams()
def effect_init():

    from matplotlib.transforms import Bbox
    
    width, depth = 16, 1
    task = 'scratch'
    model_params = {'c_in': 1, 'c_out': 1, 'width': width, 'depth': 1, 'dilations': [1,2,4,8,16]}
    models = [UNetRegressionModel(**model_params) for _ in range(3)]    
    [model.set_normalization(ds) for model in models]

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_baseline/UNet_f{width}_W_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
    ) for model in models]

    layers = [get_unet_layers(model) for model in models[:2]]
    reps_init = get_model_representation(models[:2], patches, layers, sample_rate=10)

    fig, axes = plt.subplots(2,3, sharex=True ,figsize=(20,10))    

    axes[0,0].matshow(get_svcca_matrix(*reps_init))  
    axes[0,0].set_ylabel('CV01 pre'); axes[0,0].set_xlabel('CV03/05 pre')
   

    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{width}_P_shuffle_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
    ) for model, cv in zip(models, ['01','03','05'])]

    layers = [get_unet_layers(model) for model in models]
    reps_best = get_model_representation(models, patches, layers, sample_rate=10)

    axes[0,1].matshow(get_svcca_matrix(reps_best[0], reps_best[1]))
    axes[0,2].matshow(get_svcca_matrix(reps_best[0], reps_best[2]))
    axes[0,1].set_ylabel('CV01 transfer'); axes[0,1].set_xlabel('CV03 transfer')
    axes[0,2].set_xlabel('CV05 transfer')


    [model.msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{width}_P_scratch_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
    ) for model, cv in zip(models, ['01','03','05'])]

    layers = [get_unet_layers(model) for model in models]
    reps_best = get_model_representation(models, patches, layers, sample_rate=10)

    axes[1,0].matshow(get_svcca_matrix(reps_best[0], reps_best[1]))
    axes[1,1].matshow(get_svcca_matrix(reps_best[0], reps_best[2]))

    axes[1,0].set_ylabel('CV01 scratch'); axes[1,0].set_xlabel('CV03 scratch')
    axes[1,1].set_xlabel('CV05 scratch')


    models[0].msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{width}_P_transfer_CV01_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
    )
    models[1].msd.load_state_dict(
        torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{width}_P_scratch_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
    )

    layers = [get_unet_layers(model) for model in models[:2]]
    reps = get_model_representation(models[:2], patches, layers, sample_rate=10)

    axes[1,2].matshow(get_svcca_matrix(*reps))
    axes[1,2].set_ylabel('CV01 scratch'); axes[1,2].set_xlabel('CV01 transfer')

    for i, ax in enumerate(axes.flat):
        ax.set_title(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'][i]+' '*22)
        ax.set_xticks([])
        if i in [1,3]:
            ax.set_yticks([])
        else:
            ax.yaxis.tick_right()
            ax.set_yticks(range(9))
            ax.set_yticklabels(unet_layers)

    plt.savefig(f'outputs/svcca_effects_init_UNet_f{width}.png', bbox_inches='tight')
    sys.exit()
    

def sub_model_hypothesis():
    
    width, depth = 1, 1
    fig, ax = plt.subplots()

    for i, width in enumerate([8, 16, 32, 64]):
        model_params = {'c_in': 1, 'c_out': 1, 'width': width, 'depth': depth, 'dilations': [1,2,4,8,16]}
        models = [MSDRegressionModel(**model_params) for _ in range(3)]
        [model.set_normalization(ds) for model in models]

        [model.msd.load_state_dict(
            torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{width}_P_transfer_CV01_CV0{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
        ) for model, cv in zip(models, [1,3,5])]

        layers = [range(1, depth+1) for model in models]
        reps = get_model_representation(models, patches, layers, sample_rate=10)
        

        dists = [get_pwcca_dist(reps[0], reps[1]), get_pwcca_dist(reps[0], reps[2])]

        ax.plot(np.array(dists).mean(0), c=colors[i], label=f'UNet_f{width}')
        ax.fill_between(range(len(dists[0])), np.array(dists).min(0), np.array(dists).max(0), color=colors[i], alpha=.5)

    ax.set_ylabel('PWCCA distance')
    # ax.set_xticks(range(9))
    # ax.set_xticklabels(unet_layers)
    plt.legend()
    plt.savefig('outputs/sub_model_UNet.png')    


    # model_params = {'c_in': 1, 'c_out': 1, 'width': 8, 'depth': 1, 'dilations': [1,2,4,8,16]}
    # ref_models = [UNetRegressionModel(**model_params) for _ in range(3)]
    # [model.set_normalization(ds) for model in ref_models]

    # [model.msd.load_state_dict(
    #     torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f8_P_scratch_CV0{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
    # ) for model, cv in zip(ref_models, [1,3,5])]

    # layers = [get_unet_layers(model) for model in ref_models]
    # ref_reps = get_model_representation(ref_models, patches, layers, sample_rate=10)

    # fig, ax = plt.subplots()
    # dists = [get_pwcca_dist(ref_reps[0], ref_reps[1]), get_pwcca_dist(ref_reps[0], ref_reps[2])]
    # ax.plot(np.array(dists).mean(0), c='k', label=f'UNet_f8')
    # ax.fill_between(range(len(dists[0])), np.array(dists).min(0), np.array(dists).max(0), color='k', alpha=.5)


    # for i, width in enumerate([16, 32, 64]):
    #     model_params = {'c_in': 1, 'c_out': 1, 'width': width, 'depth': depth, 'dilations': [1,2,4,8,16]}
    #     models = [UNetRegressionModel(**model_params) for _ in range(3)]
    #     [model.set_normalization(ds) for model in models]

    #     [model.msd.load_state_dict(
    #         torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{width}_P_scratch_CV0{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu')
    #     ) for model, cv in zip(models, [1,3,5])]

    #     layers = [get_unet_layers(model) for model in [ref_models[0]]+models]
    #     reps = get_model_representation([ref_models[0]]+models, patches, layers, sample_rate=10)       

    #     dists = [get_pwcca_dist(reps[0], reps[1]), get_pwcca_dist(reps[0], reps[2]), get_pwcca_dist(reps[0], reps[3])]

    #     ax.plot(np.array(dists).mean(0), c=colors[i+1], label=f'UNet_f{width}')
    #     ax.fill_between(range(len(dists[0])), np.array(dists).min(0), np.array(dists).max(0), color=colors[i+1], alpha=.5)

    
    # ax.set_ylim([0,.45])
    # ax.set_yticks([.1,.2,.3,.4])
    # ax.set_ylabel('PWCCA distance')
    # ax.set_xticks(range(9))
    # ax.set_xticklabels(unet_layers)
    # plt.legend()
    # plt.savefig('outputs/sub_model_UNet_comp_ref8.png')   
    

def svcca_convergence():
    """Animation of SVCCA similarity over training"""

    def training_generator(model, ds, epochs=10):

        
        dl = DataLoader(ds, batch_size=6, shuffle=True)
        optimizer = torch.optim.Adam(model.msd.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        gamma =  1e-2 **(1/(epochs-2))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        for e in trange(epochs):
            for i, (in_slc, target_slc) in enumerate(dl):

                out_slc = model(in_slc)
                loss = criterion(out_slc, target_slc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not (i % 40): yield model

                if i >= 400: continue

            if e > 1: scheduler.step()

    
    target_ims, input_ims = utils.load_phantom_ds()
    test_set, val_set, train_set = next(utils.split_data_CV(input_ims, target_ims, frac=(1/7, 2/7)))

    transforms = TransformList([RandomHFlip()])
    ds = MultiOrbitDataset(*train_set, device='cuda', transforms=transforms)

    patches = get_patches(MultiOrbitDataset(*test_set, device='cuda', vert_sym=False), 50)

    which = 3
    models, model_names = get_models()
    model, model_name = models[which], model_names[which]
    del models, model_names

    model.set_normalization(norm_dl)
    reps = []

    # model.msd.load_state_dict(torch.load(sorted(models_dir.glob(f'MSD_baseline/MSD_d80_W_CV01_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    # shuffle_weights(model.msd)

    save_path = Path('/media/beta/florian/figs/UNet_f16_scratch/')
    save_path.mkdir(parents=True, exist_ok=True)

    for i, model_sample in enumerate(training_generator(model, ds, epochs=10)):
        torch.save(model_sample.msd.state_dict(), save_path / f'state_dict_{i}.h5')

    state_dicts = sorted(save_path.glob('*.h5'), key=_nat_sort)

    import copy 
    comp_model = copy.deepcopy(model)

    model.msd.load_state_dict(torch.load(state_dicts[-1], map_location='cpu'))

    for i, state_dict in tqdm(enumerate(state_dicts), total=len(state_dicts)):
        comp_model.msd.load_state_dict(torch.load(state_dict, map_location='cpu'))

        reps = get_model_representation([model, comp_model], patches, [get_unet_layers(model), get_unet_layers(comp_model)])
        # reps = get_model_representation([model, comp_model], patches, [range(1,81), range(1,81)])
        svcca_mat = get_svcca_matrix(*reps)

        fig, ax = plt.subplots(figsize=(5,5))
        mat = ax.matshow(svcca_mat, cmap='viridis', vmin=.2, vmax=1)

        cax = fig.add_axes([.931, 0.1, .02, 0.8])
        plt.colorbar(mat, cax=cax)
        ax.set_yticks(range(9))
        ax.set_yticklabels(unet_layers)
        ax.xaxis.set_visible(False)

        Path('outputs/svcca_frames').mkdir(exist_ok=True)
        plt.savefig(Path('outputs/svcca_frames') / f'svcca_mat_{i}.png', bbox_inches='tight')
        plt.close()

    frames = sorted(Path('outputs/svcca_frames').glob('*.png'), key=_nat_sort)
    frames = [imread(frame) for frame in frames]

    mimsave('outputs/SVCCA_mat_convergence_UNet_scratch.gif', np.stack(frames, 0), fps=30)

    sys.exit()


def main():
    # Sets available GPU if not already set in env vars
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
          flush=True)

    # ===== Metrics ===== #

    # eval_metrics_CV()
    # eval_metrics_mat()
    # eval_metrics_samples()
    # fig_convergence()
    # memory_generalization()
    # dense_metric_eval()
    # metrics_evolution()
    # plot_weights_evolution()
    # plot_losses()
    # plot_appendix()
    # eval_test_samples()


    # ===== CCA ===== #
    global colors, models_dir, unet_layers
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    models_dir = Path('/media/beta/florian/model_weights/')
    unet_layers = ['in', 'd1', 'd2', 'd3', 'd4', 'u1', 'u2', 'u3', 'u4', 'out'][:-1]

    # svcca_over_training()
    # pwcca_over_training()
    # effect_init()
    # sub_model_hypothesis()
    # svcca_convergence()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, nargs='?', default=0,
                        help='GPU to run on')
    args = parser.parse_args()

    GPU_ID = args.gpu

    main()
