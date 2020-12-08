import argparse
import glob
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.image_dataset import MultiOrbitDataset
from src.transfer_model import transfer, shuffle_weights
from src.unet_regr_model import UNetRegressionModel
from src.utils import ValSampler, _nat_sort, evaluate


def transfer_model():

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 30, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    model = MSDRegressionModel(**model_params)
    # model.net.load_state_dict(torch.load('model_weights/MSD_d30_fW_e20.h5'))

    target_ims, input_ims = utils.load_phantom_ds()
    test_set, val_set, train_set = utils.split_data(input_ims, target_ims)

    test_ds, val_ds, train_ds = \
        MultiOrbitDataset(*test_set, device='cuda'), \
        MultiOrbitDataset(*val_set, device='cuda'), \
        MultiOrbitDataset(*train_set, device='cuda')
    
    # test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, sampler=ValSampler(len(val_ds), fixed_samples=False))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    model.set_normalization(DataLoader(train_ds, batch_size=50, sampler=ValSampler(len(train_ds), 700)))
    
    transfer(model, (train_dl, val_dl), filename='outputs/transfer_loss_W2P_scratch.png')

    model.net.load_state_dict(torch.load('model_weights/MSD_d30_fW_e20.h5'))
    transfer(model, (train_dl, val_dl), filename='outputs/transfer_loss_W2P_transfer.png')

    model.net.load_state_dict(torch.load('model_weights/MSD_d30_fW_e20.h5'))
    shuffle_weights(model.msd)
    transfer(model, (train_dl, val_dl), filename='outputs/transfer_loss_W2P_shuffle.png')


def main():
    # Sets available GPU if not already set in env vars
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
          flush=True)
    
    transfer_model()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, nargs='?', default=0,
                        help='GPU to run astra sims on')
    args = parser.parse_args()

    GPU_ID = args.gpu

    main()
