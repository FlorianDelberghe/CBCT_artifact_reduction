import argparse
import glob
import os
import random
import sys
from datetime import datetime

import torch
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.image_dataset import MultiOrbitDataset
from src.train_model import *
from src.utils import ValSampler, _nat_sort
import src.models as models


def train_model():  

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 30, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    batch_size = 64

    target_ims, input_ims = utils.load_phantom_ds()
    # cv_split_generator = utils.split_data_CV(input_ims, target_ims, frac=(1/4, 2/4))
    # test_set, val_set, train_set = utils.split_data(input_ims, target_ims)

    # for i, (test_set, val_set, train_set) in enumerate(cv_split_generator):
    for i in range(3):
        test_set, val_set, train_set = next(utils.split_data_CV(input_ims, target_ims, frac=(1/4, (2-i)/4)))
        if val_set is None: val_set = train_set

        model = MSDRegressionModel(**model_params)
        model.msd.load_state_dict(
            torch.load(sorted(glob.glob('model_weights/MSD_d30_fW_1127195514/model*.h5'), key=_nat_sort)[-1]))

        train_ds = MultiOrbitDataset(*train_set, device='cuda')
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ds = MultiOrbitDataset(*val_set, device='cuda')
        val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=ValSampler(len(val_ds)))

        train_params = {'epochs': 50, 'lr': 2e-3}
        if MODEL_NAME is not None:
            train_params['save_folder'] = f"model_weights/{MODEL_NAME}_nP{i+1:0>2d}_{datetime.now().strftime('%m%d%H%M%S')}"

        model.set_normalization(DataLoader(train_ds, batch_size=100, sampler=ValSampler(len(train_ds), min(len(train_ds), 5000))))
        train(model, (train_dl, val_dl), nn.MSELoss(), **train_params)

    sys.exit()
    # model = MSDRegressionModel(**model_params)
    # model.msd.load_state_dict(
    #     torch.load(sorted(glob.glob('model_weights/MSD_d30_fW_1127195514/model*.h5'), key=_nat_sort)[-1]))

    # from src.transfer_model import shuffle_weights
    # shuffle_weights(model.msd)

    
    # train_ds = MultiOrbitDataset(*train_set, device='cuda')
    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_ds = MultiOrbitDataset(*val_set, device='cuda')
    # val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=ValSampler(len(val_ds)))
    
    kwargs = {}
    if MODEL_NAME is not None:
        kwargs['save_folder'] = f"model_weights/{MODEL_NAME}_{datetime.now().strftime('%m%d%H%M%S')}"

    model.set_normalization(DataLoader(train_ds, batch_size=100, sampler=ValSampler(len(train_ds), min(len(train_ds), 5000))))
    # train(model, (train_dl, val_dl), nn.MSELoss(), 50, lr=2e-3, **kwargs)
    cross_validation_train()


def main():
    # Sets available GPU if not already set in env vars
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
          flush=True)

    train_model()


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
