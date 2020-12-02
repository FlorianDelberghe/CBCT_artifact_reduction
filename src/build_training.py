import glob
import os
import re
import shutil
import sys
from pathlib import Path
import time

import foam_ct_phantom
import numpy as np
from imageio import imread, imsave
from natsort import natsorted
from scipy.interpolate import RegularGridInterpolator

from . import astra_sim, utils
from .astra_sim import FleX_ray_scanner, radial_slice_sampling


def build_foam_phantom_dataset(folder_path, n_phantoms, spheres_per_unit=10000, GPU_ID=0):
    
    # Choose to reset dataset path or not
    if os.path.exists(folder_path):
        print("Path already exists, delete?")
        choice = input()
        if choice.lower()[0] == 'y':
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        else:
            print("Will overwrite files in folder")

    shape = (501,) *3
    n_rad_slices = 360
    temp_dir = '_temp_phatoms/'

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        
    scanner_params = FleX_ray_scanner()
    orbit_elevation = [-15,0,15]
    theta_range = np.linspace(0, np.pi, n_rad_slices, endpoint=False)

    for i in range(n_phantoms):
        foam_ct_phantom.FoamPhantom.generate(os.path.join(temp_dir, 'foam_phantom.h5'), 
                                             seed=i, nspheres_per_unit=spheres_per_unit)
        
        phantom = foam_ct_phantom.FoamPhantom(os.path.join(temp_dir, 'foam_phantom.h5'))
        geom = foam_ct_phantom.VolumeGeometry(*shape, voxsize=3/shape[0])

        phantom.generate_volume(os.path.join(temp_dir, 'gen_phantom.h5'), geom)
        phantom_volume = foam_ct_phantom.load_volume(os.path.join(temp_dir, 'gen_phantom.h5'))

        if not os.path.exists(os.path.join(folder_path, f'phantom{i+1}/')):
                os.makedirs(os.path.join(folder_path, f'phantom{i+1}/'))

        rad_slices = astra_sim.radial_slice_sampling(phantom_volume, theta_range)
        
        for j in range(len(rad_slices)):
            print(f"\rSaving phantom_n{i+1:0>3d}_s{j+1:0>4d}.tif", end=' '*5)
            imsave(os.path.join(folder_path, f'phantom{i+1}/', f"phantom_true_n{i+1:0>3d}_s{j+1:0>4d}.tif"), rad_slices[j].astype('float32'))
        print('')

        # Simulates FDK reconstruction for all 3 orbits
        for orbit in [1,2,3]:
            vecs = astra_sim.create_scan_geometry(scanner_params, 1200, elevation=orbit_elevation[orbit-1])
            projections = astra_sim.create_CB_projection(phantom_volume, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID)
            fdk_phantom_volume = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID, rec_shape=(501,501,501))

            fdk_rad_slices = astra_sim.radial_slice_sampling(fdk_phantom_volume, theta_range)

            for j in range(len(rad_slices)):
                print(f"\rSaving phantom_n{i+1:0>3d}_s{j+1:0>4d}.tif", end=' '*5)
                imsave(os.path.join(folder_path, f'phantom{i+1}/', f"phantom_fdk_n{i+1:0>3d}_o{orbit:d}_s{j+1:0>4d}.tif"), fdk_rad_slices[j].astype('float32'))
            print('')

    # Removes temp phatom files
    shutil.rmtree(temp_dir)


def build_fast_walnut_dataset():
    
    DATA_PATH = Path('/data/fdelberghe/')
    save_folder = DATA_PATH/'FastWalnuts2/'

    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Lists subdirs
    walnuts_folders = natsorted(list(filter(lambda p: re.search(r'/Walnut\d+$', p.as_posix()),
                                            DATA_PATH.glob('Walnuts/*'))))
    for folder in walnuts_folders:
        print(f"Loading {folder}/...")

        agd_images = natsorted(folder.glob('Reconstructions/full_AGD_50*.tif'))
        fdk_images = [natsorted(folder.glob(f'Reconstructions/fdk_pos{orbit}*.tiff')) 
                      for orbit in (1,2,3)]

        agd_volume = np.stack([imread(file) for file in agd_images], axis=0)
        fdk_volumes = [np.stack([imread(file) for file in fdk_orbit], axis=0) for fdk_orbit in fdk_images]

        n_theta = int(np.round(np.sqrt(2) *501))
        theta_range = np.linspace(0, np.pi, n_theta, endpoint=False)    

        adg_rad_slices = radial_slice_sampling(agd_volume, theta_range)

        # makes save folder
        (save_folder/folder.name).mkdir(parents=True, exist_ok=True)

        for i in range(len(adg_rad_slices)):
            print(f"\rSaving agd_s{i+1:0>3d}.tif", end=' '*5)
            imsave((save_folder/folder.name/ f'agd_s{i+1:0>3d}.tif'), adg_rad_slices[i])
        print('')

        for j, fdk_volume in enumerate(fdk_volumes):
            fdk_rad_slices = radial_slice_sampling(fdk_volume, theta_range)

            for i in range(len(fdk_rad_slices)):
                print(f"\rSaving fdk_orbit{j+1:0>2d}_s{i+1:0>3d}.tif", end=' '*5)
                imsave((save_folder/folder.name/ f'fdk_orbit{j+1:0>2d}_s{i+1:0>3d}.tif'), fdk_rad_slices[i])
            print('')


def build_phantom_dataset(gpu_id=2):
    """Builds dataset with anthropomorphic phantoms"""
    
    DATA_PATH = Path('/data/fdelberghe/')
    save_folder = DATA_PATH/'PhantomsRadial/'

    phantom_folders = natsorted(DATA_PATH.glob('AxialPhantoms/*'))

    scanner_params = FleX_ray_scanner()
    scanner_trajs = [astra_sim.create_scan_geometry(scanner_params, n_projs=1200, elevation=el) for el in [-15, 0, 15]]
    
    theta_range = np.linspace(0, np.pi, int(np.round(np.sqrt(2) *501)), endpoint=False)

    for folder in [phantom_folders[0], phantom_folders[-1]]:
        print(f"Loading {folder}/...")        

        axial_ims = sorted(folder.glob('*.tif'), key=utils._nat_sort)
        input_volume = np.stack([imread(file) for file in axial_ims], axis=0).astype('float32')
        # Estimates the air density from mean intensity at the edges of the volume
        dark_field = np.mean([input_volume[0].mean(), input_volume[:,0].mean(), input_volume[:,:,0].mean(), input_volume[:,:,-1].mean()])
        input_volume = (input_volume -dark_field) /(input_volume.max() -dark_field)
        
        interp_shape = (501, 501)
        # interp the volume in a box the size of the largest axis
        max_in_dim = max(input_volume.shape)

        # Creates grid center on volume center regardless of volume shape
        z_gr = np.linspace(-input_volume.shape[0] /interp_shape[0] /max_in_dim *501,
                          input_volume.shape[0] /interp_shape[0] /max_in_dim *501,
                          input_volume.shape[0])
        x_gr, y_gr = [np.linspace(-input_volume.shape[j] /interp_shape[1] /max_in_dim *501,
                                  input_volume.shape[j] /interp_shape[1] /max_in_dim *501,
                                  input_volume.shape[j]) for j in range(1,3)]

        interp = RegularGridInterpolator((z_gr, x_gr, y_gr), input_volume, fill_value=0, bounds_error=False)

        z_gr = np.linspace(-1.0, 1.0, interp_shape[0])
        xy_gr = np.linspace(-1.0, 1.0, interp_shape[1])

        z_rad = np.vstack((z_gr,) *interp_shape[1]).T

        (save_folder/folder.name).mkdir(parents=True, exist_ok=True)

        for j in range(len(theta_range)):
            x_rad = np.vstack((xy_gr * np.cos(theta_range[j]),) *interp_shape[0])
            y_rad = np.vstack((xy_gr * -np.sin(theta_range[j]),) *interp_shape[0])

            rad_slices_input = interp(np.vstack((z_rad.flatten(), x_rad.flatten(), y_rad.flatten())).T
                                ).reshape(interp_shape)

            print(f"\rSaving {folder.name}/CT_target_s{j+1:0>3d}", end=' '*5)  
            imsave((save_folder/folder.name/ f'CT_target_s{j+1:0>4d}.tif'), rad_slices_input.astype('float32'))
        print('')

        # Voxel size for whole cube volume within scan FoV
        vox_sz = scanner_params.source_origin_dist /(scanner_params.source_detector_dist /min(scanner_params.FoV) +.5) /max_in_dim

        for i, scanner_traj in enumerate(scanner_trajs):
            projections = astra_sim.create_CB_projection(input_volume, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz, gpu_id=gpu_id)
            reconstructed_volume = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz *max_in_dim/501, gpu_id=gpu_id)

            rad_slices_CB = radial_slice_sampling(reconstructed_volume, theta_range)
            
            for j in range(len(theta_range)):    
                print(f"\rSaving {folder.name}/CB_source_s{j+1:0>3d}", end=' '*5)     
                imsave((save_folder/folder.name/ f'CB_source_orbit{i+1:0>2d}_s{j+1:0>4d}.tif'), rad_slices_CB[j])
            print('')
   

if __name__ == '__main__':
    build_phantom_dataset()
