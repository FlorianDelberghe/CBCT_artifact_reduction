import argparse
import glob
import os
import random
import sys
from datetime import datetime

import astra
import foam_ct_phantom
import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread
from msd_pytorch import MSDRegressionModel
from torch import nn
from torch.utils.data import DataLoader

import src.utils as utils
from src import astra_sim
from src.image_dataset import MultiOrbitDataset, data_augmentation
from src.test_model import add_noise, compute_metric, noise_robustness, test
from src.train_model import TVRegularization, train
from src.transfer_model import transfer
from src.unet_regr_model import UNetRegressionModel
from src.utils import (ValSampler, _nat_sort, evaluate, imsave,
                       load_projections, mimsave)


def test_astra_sim():

    scanner_params = astra_sim.FleX_ray_scanner()

    # ===== Import projections data and reconstruct volume w/ FDK ===== #
    projections, dark_field, flat_field, vecs = load_projections(DATA_PATH+f'Walnuts/Walnut{WALNUT_ID}', orbit_id=ORBIT_ID)
    projections = (projections - dark_field) / (flat_field - dark_field)
    projections = -np.log(projections)
    projections = np.ascontiguousarray(projections)

    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, 
                                                  proj_vecs=vecs, gpu_id=GPU_ID, 
                                                  rec_shape=(501, 501, 501))
    reconstruction = (reconstruction -reconstruction.min()) / (reconstruction.max() -reconstruction.min())
    utils.save_vid(f'outputs/reconst_fdk_pos{ORBIT_ID}.avi', np.swapaxes(reconstruction[...,None], 0,1))

    radial_slices = astra_sim.radial_slice_sampling(reconstruction, np.linspace(0, np.pi, 360, endpoint=False))
    utils.save_vid('outputs/reconst_fdk_radial.avi', radial_slices[...,None])

    axial_slices = astra_sim.radial2axial(radial_slices)
    utils.save_vid('outputs/reconst_fdk_radial2axial.avi', axial_slices[...,None])

    # ===== Import ground truth volume data and simulate projection and reconstruction ===== #
    fdk_files = sorted(glob.glob(DATA_PATH+f'Walnuts/Walnut{WALNUT_ID}/Reconstructions/fdk_pos{ORBIT_ID}_*.tiff'))
    agd_files = sorted(glob.glob(DATA_PATH+f'Walnuts/Walnut{WALNUT_ID}/Reconstructions/full_AGD_50*.tiff'))
    fdk_volume = np.stack([imread(file) for file in fdk_files], axis=0)
    agd_volume = np.stack([imread(file) for file in agd_files], axis=0)

    utils.save_vid(f'outputs/true_fdk_pos{ORBIT_ID}.avi', 
                   (np.swapaxes(fdk_volume[..., None], 0,1) - fdk_volume.min()) / (fdk_volume.max() - fdk_volume.min()))
    utils.save_vid(f'outputs/true_fdk_pos{ORBIT_ID}_radial.avi', 
                   astra_sim.radial_slice_sampling(fdk_volume, np.linspace(0, np.pi, 360, endpoint=False))[..., None] / fdk_volume.max())
    utils.save_vid('outputs/true_agd.avi',
                   (np.swapaxes(agd_volume[..., None], 0,1) - agd_volume.min()) / (agd_volume.max() - agd_volume.min()))

    vecs = astra_sim.create_scan_geometry(scanner_params, 1200, elevation=0)
    projections = astra_sim.create_CB_projection(agd_volume, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID)

    utils.save_vid('outputs/agd_proj.avi', projections[...,None] /projections.max())
    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID, rec_shape=(501,501,501))
    reconstruction = (np.transpose(reconstruction, (0,2,1)) -reconstruction.min()) / (reconstruction.max() -reconstruction.min())
    utils.save_vid(f'outputs/reconst_fdk_pos{ORBIT_ID}_sim.avi', (np.swapaxes(reconstruction[...,None], 0,1)))
    utils.save_vid(f'outputs/reconst_fdk_pos{ORBIT_ID}_sim_radial.avi', 
                   astra_sim.radial_slice_sampling(reconstruction, np.linspace(0, np.pi, 360, endpoint=False))[...,None])

    # ===== Creates phatom and simulate projection and reconstruction ===== #
    phantom = astra_sim.create_ct_foam_phantom()

    projections = astra_sim.create_CB_projection(phantom, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID)
    utils.save_vid('outputs/phantom_proj.avi', projections[...,None] /projections.max())
    
    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID, rec_shape=(501,501,501))
    reconstruction = (reconstruction -reconstruction.min()) / (reconstruction.max() -reconstruction.min())
    utils.save_vid('outputs/phantom_reconst_fdk.avi',(np.swapaxes(reconstruction[...,None], 0,1)))
    utils.save_vid('outputs/phantom_reconst_fdk_radial.avi', 
                   astra_sim.radial_slice_sampling(reconstruction, np.linspace(0, np.pi, 360, endpoint=False))[...,None])


def train_model(seed=0):
    
    model_params = {'c_in': 1, 'c_out': 1, 'depth': 30, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    model = MSDRegressionModel(**model_params)
    # model.net.load_state_dict(torch.load('model_weights/radial_msd_depth80_it5_epoch59_copy.pytorch')['state_dict'])
    # regularization = TVRegularization(scaling=1e-3)
    regularization = None

    # agd_ims, fdk_ims = utils.load_phantom_ds()
    agd_ims, fdk_ims = utils.load_walnut_ds()
    random.seed(seed)
    test_id = random.randrange(len(agd_ims))
    print(f"Using sample {test_id} as validation")
    input_val, target_val = [fdk_ims.pop(test_id)], [agd_ims.pop(test_id)]
    # train_id = random.randrange(len(agd_ims))
    # input_tr, target_tr = [fdk_ims.pop(train_id)], [agd_ims.pop(train_id)]
    input_tr, target_tr = fdk_ims, agd_ims

    batch_size = 32
    train_dl = DataLoader(MultiOrbitDataset(input_tr, target_tr, device='cuda'), batch_size=batch_size, shuffle=True)
    val_ds = MultiOrbitDataset(input_val, target_val, device='cuda')
    val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=ValSampler(len(val_ds)))
    
    kwargs = {}
    # kwargs = {'save_folder': 
    #           f"model_weights/MSD_d80_walnuts_finetuned_{datetime.now().strftime('%m%d%H%M%S')}"}
    train(model, (train_dl, val_dl), nn.MSELoss(), 20, regularization, lr=2e-3, **kwargs)


def test_model():

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 80, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    model_80 = MSDRegressionModel(**model_params)
    state_dicts = sorted(glob.glob('model_weights/MSD_d80_walnuts_finetuned_1114125135/best*.h5'), key=_nat_sort)
    model_80.msd.load_state_dict(torch.load(state_dicts[-1]))

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 30, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    model_30 = MSDRegressionModel(**model_params)
    state_dicts = sorted(glob.glob('model_weights/MSD_d30_walnuts1113135028/best*.h5'), key=_nat_sort)
    model_30.msd.load_state_dict(torch.load(state_dicts[-1]))

    agd_ims, fdk_ims = utils.load_walnut_ds()
    # agd_ims, fdk_ims = utils.load_phantom_ds()
    random.seed(0)
    test_id = random.randrange(len(agd_ims))
    input_te, target_te = [fdk_ims.pop(test_id)], [agd_ims.pop(test_id)]
    
    te_ds = MultiOrbitDataset(input_te, target_te, data_augmentation=False)
    te_dl = DataLoader(te_ds, batch_size=8, sampler=ValSampler(len(te_ds)))

    model_80.set_normalization(te_dl)
    model_30.set_normalization(te_dl)

    mean, std = test(model_80, te_ds)
    print(f"Model d80 \n\tMSE: {mean[0]:.4e} +-{std[0]:.4e}, \n\tSSIM: {mean[1]:.4f} +-{std[1]:.4e}, \n\tDSC: {mean[2]:.4f} +-{std[2]:.4e}")

    mean, std = test(model_30, te_ds)
    print(f"Model d30 \n\tMSE: {mean[0]:.4e} +-{std[0]:.4e}, \n\tSSIM: {mean[1]:.4f} +-{std[1]:.4e}, \n\tDSC: {mean[2]:.4f} +-{std[2]:.4e}")

    
    sys.exit()

    model.msd.load_state_dict(torch.load(state_dicts[-1]))
    with evaluate(model):
        for i, (input_, target) in enumerate(te_dl):
            pred = model(input_)
            print(f"MSE: {mse(pred, target):.4e}, SSIM: {ssim(pred, target):.4f}, DSC: {dsc(pred, target):.4f}")

            imsave(f'outputs/test_pred_{i+1}.tif', np.clip(np.concatenate([input_[0,0].cpu().numpy(), pred[0,0].cpu().numpy(), target[0,0].cpu().numpy()], axis=-1), 0 ,None))

    plt.figure(figsize=(10,10))
    plt.plot(epochs, losses)
    plt.savefig('outputs/training.png')


def transfer_model():

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 30, 'width': 1,
                        'dilations': [1,2,4,8,16], 'loss': 'L2'}
    model = MSDRegressionModel(**model_params)
    # state_dict = torch.load('model_weights/radial_msd_depth80_it5_epoch59_copy.pytorch')['state_dict']
    # model.net.load_state_dict(state_dict)

    agd_ims, fdk_ims = utils.load_walnut_ds()
    random.seed(0)
    val_id = random.randrange(len(agd_ims))
    input_val, target_val = [fdk_ims.pop(val_id)], [agd_ims.pop(val_id)]

    # train_id = random.randrange(len(agd_ims))
    # input_tr, target_tr = [fdk_ims.pop(train_id)], [agd_ims.pop(train_id)]
    input_tr, target_tr = fdk_ims, agd_ims

    val_ds = MultiOrbitDataset(input_val, target_val, device='cuda')
    train_dl = DataLoader(MultiOrbitDataset(input_tr, target_tr, device='cuda'), batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, sampler=ValSampler(len(val_ds)))

    model.set_normalization(train_dl)

    transfer(model, (train_dl, val_dl))


def main():
    # Sets available GPU if not already set in env vars
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}", 
          flush=True)

    # test_astra_sim()
    # train_model()
    # test_model()
    transfer_model()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, nargs='?', default=0,
                        help='GPU to run astra sims on')
    parser.add_argument('--nut_id', type=int, nargs='?', default=7, 
                        help='Which walnut to import')
    parser.add_argument('--orbit_id', type=int, nargs='?', default=2,
                        choices=(1,2,3),
                        help='Orbit to load projection from')
    args = parser.parse_args()

    GPU_ID = args.gpu
    WALNUT_ID = args.nut_id
    ORBIT_ID = args.orbit_id
    DATA_PATH = '/data/fdelberghe/'

    main()
