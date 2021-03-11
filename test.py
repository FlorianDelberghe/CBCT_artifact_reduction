#%%
import argparse
import glob
import os
import pickle
import random
import sys
from pathlib import Path
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.transfer_model as transfer
from src.image_dataset import MultiOrbitDataset, PoissonNoise
from src.models import UNetRegressionModel
from src.test_model import *
from src.utils import _nat_sort
from src.utils.nn import ValSampler

#%%
# Sets arguments here to be used as a notebook
# if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
#     torch.cuda.set_device(GPU_ID)

# print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
#         flush=True)


#%%
def plot_weights_evolution():
    model_params = {'c_in': 1, 'c_out': 1, 'depth': 80, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}

    model = MSDRegressionModel(**model_params)

    transfer.plot_weights_dist(model.msd, 
                               list(map(lambda f: torch.load(f, map_location='cpu'), sorted(Path(
                                   'model_weights/MSD_phantoms/').glob('MSD_d80_P_scratch_CV01*/model_*.h5'), key=_nat_sort))),
                      title='norm fW', filename='weight_dist_scratch.png')

    transfer.plot_weights_dist(model.msd, 
                               list(map(lambda f: torch.load(f, map_location='cpu'), sorted(Path(
                                   'model_weights/MSD_phantoms/').glob('MSD_d80_P_transfer_CV01_CV01*/model_*.h5'), key=_nat_sort))),
                      title='norm fW', filename='weight_dist_trans.png')

# plot_weights_evolution()


#%%
def plot_losses():
#     for _ in utils.split_data_CV(*(list(range(7)),) *2, frac=(0, 6/7)): pass
    # phantoms_LOOCV = [f'val_Phantom{i+1}' for i in [6,3,5,0,1,2,4]]    
    # phantoms_LOICV = [f'tr_Phantom{i+1}' for i in [4,6,3,5,0,1,2]]

    folders = sorted(list(Path('model_weights/UNet_phantoms').glob('UNet_f64_P_*CV01*')), key=_nat_sort)
    folder_names = list(map(lambda f: f.name[:-11], folders))
    print(folders)
    # folders = [folder for _, folder in sorted(zip(phantoms_LOICV, folders))]

    compare_loss(*folders, names=folder_names)

# plot_losses()

#%%
def metrics_evolution():
    
    model_params = {'c_in': 1, 'c_out': 1, 'width': 1,
                    'dilations': [1, 2, 4, 8, 16], 'loss': 'L2'}

    model_d30 = MSDRegressionModel(depth=30, **model_params)
    model_d80 = MSDRegressionModel(depth=80, **model_params) 
    
    del model_params['width']
    model_f8 = UNetRegressionModel(width=8, **model_params)
    model_f16 = UNetRegressionModel(width=16, **model_params)
    model_f32 = UNetRegressionModel(width=32, **model_params)
    model_f64 = UNetRegressionModel(width=64, **model_params)

    models_dir = Path('model_weights/')
    model_names = [f'MSD_d{d}' for d in [30, 80]] + [f'UNet_f{f}' for f in [8, 16, 32, 64]]

    state_dicts = [[sorted(models_dir.glob(f'MSD_phantoms/{model}_P_scratch_CV{cv}_*/model_*.h5'), key=_nat_sort) 
    for cv in ['01', '03', '05']] for model in model_names[:2]] \
        + [[sorted(models_dir.glob(f'UNet_phantoms/{model}_P_scratch_CV{cv}_*/model_*.h5'), key=_nat_sort) 
        for cv in ['01', '03', '05']] for model in model_names[2:]]

    target_ims, input_ims = utils.load_walnut_ds()
    dataset_generator = utils.split_data_CV(input_ims, target_ims, frac=(1/7, 2/7), verbose=False)
    test_set, val_set, *_ = next(dataset_generator)

    ds = MultiOrbitDataset(*test_set, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 1000))  

    models = (model_d30, model_d80) + (model_f8, model_f16, model_f32, model_f64)
    [model.set_normalization(norm_dl) for model in models]

    metrics = ['MSE', 'SSIM', 'DSC', 'PSNR']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1,4, figsize=(40,10))

    # for i, model_name in enumerate(model_names):
    #     metrics_CV = pickle.load(open(f"outputs/computed_metrics/metrics_te_W_{model_name}.pkl", 'rb'))['metrics_values']

    #     for j, metric in enumerate(metrics):
    #         axes[j].plot(metrics_CV[j].mean((0,-1)), color=colors[i])
    #         axes[j].fill_between(range(metrics_CV[j].shape[1]), metrics_CV[j].mean((0,-1)) -metrics_CV[j].std((0,-1)),
    #                              metrics_CV[j].mean((0,-1)) +metrics_CV[j].std((0,-1)), 
    #                              color=colors[i], alpha=.2)

    for i, (model, state_dicts_CV) in tqdm(enumerate(zip(models, state_dicts)), total=len(models)):
        metrics_CV = []
        for state_dicts_epochs in state_dicts_CV:
            metrics_CV.append(compute_metric_evolution(model, metrics, state_dicts_epochs, ds, n_samples=50))

        metrics_CV = np.stack(metrics_CV, axis=1)
        pickle.dump(
            {'name': model_names[i], 'metrics': metrics, 'metrics_values': metrics_CV,
             'dims': "[metrics, CV, epochs, samples]"},
            open(f"outputs/computed_metrics/metrics_te_P_scratch_{model_names[i]}.pkl", 'wb'))

        for j, metric in enumerate(metrics):
            axes[j].plot(metrics_CV[j].mean((0,-1)), color=colors[i])
            axes[j].fill_between(range(metrics_CV[j].shape[1]), metrics_CV[j].mean((0,-1)) -metrics_CV[j].std((0,-1)),
                                 metrics_CV[j].mean((0,-1)) +metrics_CV[j].std((0,-1)), 
                                 color=colors[i], alpha=.2)

    for i, metric in enumerate(metrics):
        axes[i].set_xlabel('Epoch')
        axes[i].set_title(metric)
        if i == 3: axes[i].legend(model_names)

    plt.savefig('outputs/metrics_evolution.png')
    

def eval_metrics_CV():

    target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialSmall/')
    # target_ims, input_ims = utils.load_walnut_ds()
    dataset_generator = utils.split_data_CV(input_ims, target_ims, frac=(1/7, 2/7), verbose=False)
    test_set, *_ = next(dataset_generator)

    ds = MultiOrbitDataset(*test_set, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    model_params = {'c_in': 1, 'c_out': 1,
                    'width': 1, 'dilations': [1, 2, 4, 8, 16]}

    models_dir = Path('model_weights')
    metrics = ['SSIM', 'DSC', 'PSNR']

    model_d30 = MSDRegressionModel(depth=30, **model_params)
    model_d80 = MSDRegressionModel(depth=80, **model_params) 
    
    del model_params['width']
    model_f8 = UNetRegressionModel(width=8, **model_params)
    model_f16 = UNetRegressionModel(width=16, **model_params)
    model_f32 = UNetRegressionModel(width=32, **model_params)
    model_f64 = UNetRegressionModel(width=64, **model_params)

    models = (model_d30, model_d80) + (model_f8, model_f16, model_f32, model_f64)
    model_names = [f'MSD_d{d}' for d in [30, 80]] + [f'UNet_f{f}' for f in [8, 16, 32, 64]]

    [model.set_normalization(norm_dl) for model in models]

    metrics_te = []
    for cv in ['01', '03', '05']:
        [model.msd.load_state_dict(
            torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_P_scratch_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for model, d in zip(models[:2], [30, 80])]

        [model.msd.load_state_dict(
            torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_P_scratch_CV{cv}_*/best_*.h5'), key=_nat_sort)[0], map_location='cpu'))
        for model, f in zip(models[-4:], [8, 16, 32, 64])]   

        metrics_te.append(eval_metrics(models, metrics, ds))

    plot_metrics_CV(metrics_te, model_names, metrics, ref_metrics=eval_init_metrics(metrics, ds))


def plot_pwcca_dist_over_training(model, state_dicts, ref_state_dict, patches, title=None, filename='pwcca_dist_dist.png'):
    from cca import get_layers, get_model_representation, get_pwcca_dist

    with evaluate(model):        
        model.msd.load_state_dict(ref_state_dict)
        ref_rep = get_model_representation([model], patches, [get_layers(model)])[0]

        # models = (model.msd.load_stat_dict(state_dict) for state_dict in state_dicts)
        reps = []
        for state_dict in state_dicts:
            model.msd.load_state_dict(state_dict)
            reps.extend(get_model_representation([model], patches, [get_layers(model)]))

    fig, ax = plt.subplots()
    cmap = cm.get_cmap('viridis', len(reps))
    
    for rep, c in zip(reps, cmap(np.linspace(0,1,len(reps)))):
        ax.plot(get_pwcca_dist(ref_rep, rep), c=c)


    plt.savefig(Path('outputs/') / filename)

def test():

    eval_metrics_CV()
    # metrics_evolution()
    # plot_weights_evolution()
    sys.exit()

    from cca import get_patches, get_layers, get_model_representation, get_pwcca_dist, get_svcca_matrix

    models_dir = Path('model_weights')

    # target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialSmall/')
    target_ims, input_ims = utils.load_walnut_ds()
    dataset_generator = utils.split_data_CV(input_ims, target_ims, frac=(1/7, 2/7), verbose=False)
    test_set, *_ = next(dataset_generator)

    ds = MultiOrbitDataset(*test_set, data_augmentation=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    model_params = {'c_in': 1, 'c_out': 1,
                    'width': 1, 'dilations': [1, 2, 4, 8, 16]}

    model_d30 = MSDRegressionModel(depth=30, **model_params)
    model_d80 = MSDRegressionModel(depth=80, **model_params) 
    
    del model_params['width']
    model_f8 = UNetRegressionModel(width=8, **model_params)
    model_f16 = UNetRegressionModel(width=16, **model_params)
    model_f32 = UNetRegressionModel(width=32, **model_params)
    model_f64 = UNetRegressionModel(width=64, **model_params)

    models = (model_d30, model_d80) + (model_f8, model_f16, model_f32, model_f64)
    model_names = [f'MSD_d{d}' for d in [30, 80]] + [f'UNet_f{f}' for f in [8, 16, 32, 64]]

    [model.set_normalization(norm_dl) for model in models]

    patches = get_patches(ds, 10)

    reps_CV = []
    with  utils.nn.evaluate(*models):        
        for cv in ['01', '03', '05']:
            [model.msd.load_state_dict(
                torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d{d}_W_CV{cv}_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu'))
            for model, d in zip(models[:2], [30, 80])]

            [model.msd.load_state_dict(
                torch.load(sorted(models_dir.glob(f'UNet_phantoms/UNet_f{f}_W_CV{cv}_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu'))
            for model, f in zip(models[-4:], [8, 16, 32, 64])]   
            
            layers = [range(1,31), range(1,81)] + [get_layers(model) for model in models[-4:]]

            reps_CV.append(get_model_representation(models, patches, layers))

    # print(np.array(reps_CV).shape)

    fig, ax = plt.subplots()
    for j in range(6):

        pwcca_dist = [get_pwcca_dist(reps_CV[0][j], reps_CV[1][j]),
                      get_pwcca_dist(reps_CV[0][j], reps_CV[2][j])]

        ax.plot(np.mean(pwcca_dist, axis=0), color=colors[j])
            
    plt.savefig('outputs/temp.png')
    sys.exit()

    for i, reps in enumerate(reps_CV):


            # fig, axes = plt.subplots(len(reps), len(reps), figsize=(50,50))

            # for i in range(len(reps)):
            #     for j in trange(i, len(reps)):

            #         cca_matrix = get_svcca_matrix(reps[i], reps[j], 10)
            #         axes[i,j].matshow(cca_matrix)

            #         if i == 0:
            #             axes[i,j].set_xlabel(model_names[j])    
            #             axes[i,j].xaxis.set_label_position('top') 

            #         if j == 0: axes[i,j].set_ylabel(model_names[i])

            #     plt.savefig(f'outputs/SVCCA_CV{cv}.png')
                    
    # mat = ax.matshow(cca_matrix)
    # cbar = ax.figure.colorbar(mat, ax=ax)
    
    # target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialSmall/')
        target_ims, input_ims = utils.load_walnut_ds()
    dataset_generator = utils.split_data_CV(input_ims, target_ims, frac=(1/7, 2/7), verbose=False)
    test_set, *_ = next(dataset_generator)

    ds = MultiOrbitDataset(*test_set, vert_sym=False)
    norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    model_params = {'c_in': 1, 'c_out': 1,
                    'width': 64, 'dilations': [1, 2, 4, 8, 16]}

    models_dir = Path('model_weights')
    model = UNetRegressionModel(**model_params)
    ref_state_dict = torch.load(next(models_dir.glob('UNet_baseline/UNet_f64_W_CV01_*/best_*.h5')), map_location='cpu')
    state_dicts = list(map(lambda s: torch.load(s, map_location='cpu'), 
                           sorted(models_dir.glob('UNet_phantoms/UNet_f64_P_scratch_CV01_*/model_*.h5'), key=_nat_sort)))

    model.set_normalization(norm_dl)

    patches = get_patches(ds)
    plot_pwcca_dist_over_training(model, state_dicts, ref_state_dict, patches)


















    # ds = MultiOrbitDataset(*test_set, data_augmentation=False)
    # norm_dl = DataLoader(ds, batch_size=50, sampler=ValSampler(len(ds), 500))  

    # model_params = {'c_in': 1, 'c_out': 1,
    #                 'width': 1, 'dilations': [1, 2, 4, 8, 16]}

    # models_dir = Path('model_weights')
    # metrics = ['MSE', 'SSIM', 'DSC', 'PSNR']

    # model_1 = MSDRegressionModel(depth=80, **model_params) 
    # model_2 = MSDRegressionModel(depth=80, **model_params) 
    # model_3 = MSDRegressionModel(depth=80, **model_params) 
    
    # models = (model_1, model_2, model_3)
    # model_names = [f'MSD_d80_{d}' for d in ['scratch', 'transfer' ,'shuffle']]

    # [model.set_normalization(norm_dl) for model in models]

    # metrics_te = []
    # for cv in ['01', '03', '05']:
    #     [model.msd.load_state_dict(
    #         torch.load(sorted(models_dir.glob(f'MSD_phantoms/MSD_d80_P_{d}_CV{cv}_*/model_*.h5'), key=_nat_sort)[0], map_location='cpu'))
    #     for model, d in zip(models, ['scratch', 'transfer_CV01' ,'shuffle'])]

    #     metrics_te.append(eval_metrics(models, metrics, ds))

    # plot_metrics_CV(metrics_te, model_names, metrics, ref_metrics=eval_init_metrics(metrics, ds), filename='metrics__P_tr.png')


def main():
    # Sets available GPU if not already set in env vars
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
          flush=True)

    test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, nargs='?', default=2,
                        help='GPU to run astra sims on')
    args = parser.parse_args()

    GPU_ID = args.gpu

    main()


# %%
