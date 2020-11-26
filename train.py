import argparse
import glob
import os
import random
from datetime import datetime

import torch
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from src.image_dataset import MultiOrbitDataset
from src.train_model import TVRegularization, train
from src.unet_regr_model import UNetRegressionModel
from src.utils import ValSampler, _nat_sort
                    


def train_model():    
    target_ims, input_ims = utils.load_walnut_ds()

    random.seed(SEED)
    val_id = random.randrange(len(target_ims))
    print(f"Using sample {val_id} as validation")

    input_val, target_val = [input_ims.pop(val_id)], [target_ims.pop(val_id)]
    input_tr, target_tr = input_ims, target_ims

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 30, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    model = MSDRegressionModel(**model_params)
    # model.msd.load_state_dict(
    #     torch.load(sorted(glob.glob('model_weights/MSD_d30_Walnuts_scratch_1120175735/best*.h5'), key=_nat_sort)[-1]))

    batch_size = 32
    train_dl = DataLoader(MultiOrbitDataset(input_tr, target_tr, device='cuda'), batch_size=batch_size, shuffle=True)
    val_ds = MultiOrbitDataset(input_val, target_val, device='cuda')
    val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=ValSampler(len(val_ds)))
    
    kwargs = {}
    if MODEL_NAME is not None:
        kwargs['save_folder'] = f"model_weights/{MODEL_NAME}_{datetime.now().strftime('%m%d%H%M%S')}"

    train(model, (train_dl, val_dl), nn.MSELoss(), 20, lr=2e-3, **kwargs)


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
