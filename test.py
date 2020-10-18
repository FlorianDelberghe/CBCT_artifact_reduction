import argparse
import glob
import os
import sys

import astra
import foam_ct_phantom
import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread
from msd_pytorch import MSDRegressionModel

from src import astra_sim
import src.utils as utils
from src.utils import (FleX_ray_scanner, imsave, load_projections, mimsave,
                       to_astra_coords)


def main():

    scanner_params = FleX_ray_scanner()
    projections, dark_field, flat_field, vecs = load_projections(DATA_PATH+f'Walnut{WALNUT_ID}', orbit_id=ORBIT_ID)
    projections = (projections - dark_field) / (flat_field - dark_field)
    projections = -np.log(projections)
    projections = np.ascontiguousarray(projections)

    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, 
                                                  proj_vecs=vecs, gpu_id=GPU_ID, 
                                                  rec_shape=(501, 501, 501))

    reconstruction = (reconstruction -reconstruction.min()) / (reconstruction.max() -reconstruction.min())
    utils.save_vid('outputs/reconst_fdk.avi', np.swapaxes(reconstruction[...,None], 0,1))

    radial_slices = astra_sim.radial_slice_sampling(reconstruction, np.linspace(0, np.pi, 360, endpoint=False))
    utils.save_vid('outputs/reconst_fdk_radial.avi', radial_slices[...,None])

    radial_slices = astra_sim.radial2axial(radial_slices)
    utils.save_vid('outputs/reconst_fdk_radial2axial.avi', radial_slices[...,None])

    fdk_files = sorted(glob.glob(DATA_PATH+f'Walnut{WALNUT_ID}/Reconstructions/fdk_pos{ORBIT_ID}_*.tiff'))
    agd_files = sorted(glob.glob(DATA_PATH+f'Walnut{WALNUT_ID}/Reconstructions/full_AGD_50*.tiff'))

    fdk_volume = np.stack([imread(file) for file in fdk_files], axis=0)
    utils.save_vid('outputs/true_fdk.avi', np.swapaxes(fdk_volume[...,None], 0,1) /fdk_volume.max())
    agd_volume = np.stack([imread(file) for file in agd_files], axis=0)
    utils.save_vid('outputs/true_agd.avi', np.swapaxes(agd_volume[...,None], 0,1) /agd_volume.max())
     
    vecs = astra_sim.create_scan_geometry(scanner_params, 1200)
    projections = astra_sim.create_CB_projection(agd_volume, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID)
    utils.save_vid('outputs/proj_sim.avi', projections[...,None] /projections.max())
    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID, rec_shape=(550,501,501))
    utils.save_vid('outputs/reconst_fdk_sim.avi',(np.swapaxes(reconstruction[...,None], 0,1) -reconstruction.min()) / (reconstruction.max() -reconstruction.min()))
    

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
