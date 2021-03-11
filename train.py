import argparse
import os
import sys
from datetime import datetime

import torch
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.image_dataset import (GaussianNoise, ImageDataset, MultiOrbitDataset,
                               PoissonNoise, RandomHFlip, TransformList)
from src.models import UNetRegressionModel
from src.train_model import *
from src.utils import _nat_sort
from src.utils.nn import ValSampler
from src.transfer_model import shuffle_weights, mean_var_init


def train_model_CV():  

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 1, 'width': 64,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    batch_size = 8
    transforms = TransformList([RandomHFlip(), GaussianNoise(0,0.02)])

    target_ims, input_ims = utils.load_phantom_ds(folder_path='PhantomsRadialSmall/')
    # target_ims, input_ims = utils.load_walnut_ds()
    cv_split_generator = utils.split_data_CV(input_ims, target_ims, frac=(1/7, 2/7))

    for i, (_, val_set, train_set) in enumerate(cv_split_generator):
        if i % 2 or i // 2 not in [0,1,2]: continue
        
        model = UNetRegressionModel(**model_params)
        # model.msd.load_state_dict(
        #     torch.load(next(Path('model_weights/UNet_baseline/').glob(f'UNet_f64_W_CV01_*/best_*.h5')), map_location='cpu'))
        # shuffle_weights(model.msd)
        # mean_var_init(model.msd)

        train_ds = MultiOrbitDataset(*train_set, device='cuda', transforms=transforms)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ds = MultiOrbitDataset(*val_set, device='cuda')
        val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=ValSampler(len(val_ds)))

        train_params = {'epochs': 5, 'lr': 2e-3, 'regularization': None, 'cutoff_epoch': 1}
        if MODEL_NAME is not None:
            train_params['save_folder'] = f"model_weights/{MODEL_NAME}_CV{i+1:0>2d}_{datetime.now().strftime('%m%d%H%M%S')}"

        model.set_normalization(DataLoader(train_ds, batch_size=100, sampler=ValSampler(len(train_ds), min(len(train_ds), 5000))))
        train(model, (train_dl, val_dl), nn.MSELoss(), **train_params)


def train_model():

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 30, 'width': 1,
                    'dilations': [1, 2, 4, 8, 16], 'loss': 'L2'}
    batch_size = 32

    model = MSDRegressionModel(**model_params)
    # model= UNetRegressionModel(**model_params)

    target_ims, input_ims = utils.load_phantom_ds('PhantomsRadialSmall/')
    # test_set, val_set, train_set = split_data(input_ims, target_ims, 7/42)
    _, val_set, train_set = next(utils.split_data_CV(input_ims, target_ims, frac=(0, 2/7)))
    
    train_ds = MultiOrbitDataset(*train_set, device='cuda', transforms=RandomHFlip())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = MultiOrbitDataset(*val_set, device='cuda')
    val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=ValSampler(len(val_ds)))
    
    train_params = {'epochs': 30, 'lr': 2e-3, 'regularization': None, 'cutoff_epoch': 5}
    if MODEL_NAME is not None:
        train_params['save_folder'] = f"model_weights/{MODEL_NAME}_{datetime.now().strftime('%m%d%H%M%S')}"

    model.set_normalization(DataLoader(train_ds, batch_size=100, sampler=ValSampler(len(train_ds), min(len(train_ds), 5000))))
    train(model, (train_dl, val_dl), nn.MSELoss(), **train_params)


def main():
    # Sets available GPU if not already set in env vars
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
          flush=True)

    train_model_CV()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, nargs='?', default=0,
                        help='GPU to run astra sims on')
    parser.add_argument('-s', '--seed', type=int, nargs='?', default=0,
                        help='Seed for sampling')
    parser.add_argument('-n', '--model_name', type=str, nargs='?', default=None,
                        help='Name of the model')
    args = parser.parse_args()

    GPU_ID = args.gpu
    SEED = args.seed
    MODEL_NAME = args.model_name

    main()
