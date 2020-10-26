import argparse
import glob
import os
import sys
from functools import reduce

import astra
import foam_ct_phantom
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import test
import torch
from imageio import imread
from msd_pytorch import MSDRegressionModel
from torch import mode

from src.unet_regr_model import UNetRegressionModel
from src import astra_sim
import src.utils as utils
from src.utils import (FleX_ray_scanner, imsave, load_projections, mimsave,
                       to_astra_coords)


def test_astra_sim():

    scanner_params = FleX_ray_scanner()

    # ===== Import projections data and reconstruct volume w/ FDK ===== #
    projections, dark_field, flat_field, vecs = load_projections(DATA_PATH+f'Walnut{WALNUT_ID}', orbit_id=ORBIT_ID)
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
    fdk_files = sorted(glob.glob(DATA_PATH+f'Walnut{WALNUT_ID}/Reconstructions/fdk_pos{ORBIT_ID}_*.tiff'))
    agd_files = sorted(glob.glob(DATA_PATH+f'Walnut{WALNUT_ID}/Reconstructions/full_AGD_50*.tiff'))
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


def test_msd_net():
    
    # Sets available GPU if not already set in env vars
    if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
        torch.cuda.set_device(globals().get('GPU_ID', -1))

    print(f"Running on GPU:{torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Full volume
    fdk_files = sorted(glob.glob(DATA_PATH+f'Walnut{WALNUT_ID}/Reconstructions/fdk_pos{ORBIT_ID}_*.tiff'))
    agd_files = sorted(glob.glob(DATA_PATH+f'Walnut{WALNUT_ID}/Reconstructions/full_AGD_50*.tiff'))

    # Radial slices
    test_fdk = sorted(glob.glob(f'/data/fdelberghe/WalnutsRadial/Walnut7/fdk_pos{ORBIT_ID}*.tiff'))
    test_agd = sorted(glob.glob('/data/fdelberghe/WalnutsRadial/Walnut7/iterative*.tiff'))

    images_fdk = np.stack([imread(file) for file in fdk_files], axis=0)
    images_agd = np.stack([imread(file) for file in agd_files], axis=0)

    model_params = {'c_in': 1, 'c_out': 1, 'depth': 80, 'width': 1,
                    'dilations': [1,2,4,8,16], 'loss': 'L2'}

    model = MSDRegressionModel(**model_params)
    model.load('model_weights/radial_msd_depth80_it5_epoch59_copy.pytorch')

    # model = UNetRegressionModel(**model_params, reflect=True, conv3d=False)
    # model.load('model_weights/radial_unet_depth80_it5_epoch9.pytorch')

    mean_in, std_in = images_fdk.mean(), images_fdk.std()
    mean_out, std_out = images_agd.mean(), images_agd.std()

    model.scale_in.weight.data.fill_(1 / std_in)
    model.scale_in.bias.data.fill_(-mean_in / std_in)
    model.scale_out.weight.data.fill_(std_out)
    model.scale_out.bias.data.fill_(mean_out)

    test_tensor = torch.from_numpy(images_fdk[:,250]).view((1,1, *images_fdk[:,250].shape))
    print(test_tensor.size(), test_tensor.min().item(), test_tensor.max().item(), test_tensor.std().item())

    with torch.no_grad():
        out_tensor = model.net(test_tensor.cuda())

    print(out_tensor.size(), out_tensor.min().item(), out_tensor.max().item(), out_tensor.std().item())
    print(np.sqrt(((out_tensor[0,0].detach().cpu().numpy() - images_agd[:,250]) **2).mean()))

    test_tensor = (test_tensor - test_tensor.min()) / (test_tensor.max() - test_tensor.min())
    imsave('outputs/test_in.png', test_tensor[0,0].numpy())
    out_tensor = (out_tensor - out_tensor.min()) / (out_tensor.max() - out_tensor.min())
    imsave('outputs/test_out.png', out_tensor[0,0].detach().cpu().numpy())


def main():

    test_astra_sim()
    test_msd_net()


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

    DATA_PATH = '/data/fdelberghe/Walnuts/'
    GPU_ID = args.gpu
    WALNUT_ID = args.nut_id
    ORBIT_ID = args.orbit_id

    main()
