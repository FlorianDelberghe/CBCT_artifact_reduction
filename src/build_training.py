import glob
import os
import shutil

import foam_ct_phantom
import numpy as np
from imageio import imread, imsave
from scipy.interpolate import RegularGridInterpolator

from . import astra_sim, utils
from .astra_sim import FleX_ray_scanner, radial_slice_sampling
from .utils import _nat_sort


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
    
    DATA_PATH = '/data/fdelberghe/'
    save_folder = '/data/fdelberghe/FastWalnuts2/'

    os.makedirs(save_folder, exist_ok=True)
    
    # Lists subdirs
    for folder in sorted(glob.glob(os.path.join(DATA_PATH, 'Walnuts/Walnut*'))):
        print(f"Loading: {folder}...")
        
        os.makedirs(os.path.join(save_folder, folder.split('/')[-1]), exist_ok=True)

        agd_images = sorted(glob.glob(os.path.join(folder, 'Reconstructions/full_AGD_50*.tiff')))
        fdk_images = [sorted(glob.glob(os.path.join(folder, f'Reconstructions/fdk_pos{orbit}*.tiff'))) 
                      for orbit in range(1, 4)]

        agd_volume = np.stack([imread(file) for file in agd_images], axis=0)
        fdk_volumes = [np.stack([imread(file) for file in fdk_orbit], axis=0) for fdk_orbit in fdk_images]

        n_theta = int(np.round(np.sqrt(2) *501))
        theta_range = np.linspace(0, np.pi, n_theta, endpoint=False)    

        adg_rad_slices = radial_slice_sampling(agd_volume, theta_range)

        for i in range(len(adg_rad_slices)):
            print(f"\rSaving agd_s{i+1:0>3d}.tif", end=' '*5)
            imsave(os.path.join(save_folder, folder.split('/')[-1], f"agd_s{i+1:0>3d}.tif"), adg_rad_slices[i])
        print('')

        for j, fdk_volume in enumerate(fdk_volumes):
            fdk_rad_slices = radial_slice_sampling(fdk_volume, theta_range)

            for i in range(len(fdk_rad_slices)):
                print(f"\rSaving fdk_orbit{j+1:0>2d}_s{i+1:0>3d}.tif", end=' '*5)
                imsave(os.path.join(save_folder, folder.split('/')[-1], f"fdk_orbit{j+1:0>2d}_s{i+1:0>3d}.tif"), fdk_rad_slices[i])
            print('')


def build_phantom_dataset():
    """Builds dataset with anthropomorphic phantoms"""
    
    DATA_PATH = '/data/fdelberghe/'
    save_folder = os.path.join(DATA_PATH, 'PhantomsRadial/')
    os.makedirs(save_folder, exist_ok=True)

    phantom_folders = [f'PT{i}/'for i in [1, 5, 6, 8, 10]]

    phantom_input_paths = (
        sorted(glob.glob(os.path.join(DATA_PATH + f"Phantoms/{folder}/input*.tif")), key=_nat_sort)
        for folder in phantom_folders)

    scanner_params = FleX_ray_scanner()
    scanner_traj = astra_sim.create_scan_geometry(scanner_params, n_projs=1200)
    
    theta_range = np.linspace(0, np.pi, int(np.round(np.sqrt(2) *501)), endpoint=False)

    for i, in_paths in enumerate(phantom_input_paths):
        input_volume = np.stack([imread(file) for file in in_paths], axis=0).astype('float32')[::-1]
        input_volume /= input_volume.max()

        # Interpolation of original volume on radial slices 
        interp_shape = (501, 501)
        z_gr = np.linspace(-input_volume.shape[0] /interp_shape[0] /2, 
                           input_volume.shape[0] /interp_shape[0] /2,
                           input_volume.shape[0])
        xy_gr = np.linspace(-1.0, 1.0, input_volume.shape[1])

        interp = RegularGridInterpolator((z_gr, xy_gr, xy_gr), input_volume, fill_value=0, bounds_error=False)
        rad_slices_input = np.empty((len(theta_range), *interp_shape), dtype='float32')
        
        z_gr = np.linspace(-1.0, 1.0, interp_shape[0])
        xy_gr = np.linspace(-1.0, 1.0, interp_shape[1])

        z_rad = np.vstack((z_gr,) *interp_shape[1]).T

        for j in range(len(theta_range)):
            x_rad = np.vstack((xy_gr * np.cos(theta_range[j]),) *interp_shape[0])
            y_rad = np.vstack((xy_gr * -np.sin(theta_range[j]),) *interp_shape[0])

            rad_slices_input[j] = interp(np.vstack((z_rad.flatten(), x_rad.flatten(), y_rad.flatten())).T
                                ).reshape(rad_slices_input.shape[-2:])


        # Voxel size for whole cube volume within scan FoV
        vox_sz = scanner_params.source_origin_dist / (scanner_params.source_detector_dist/min(scanner_params.FoV) +.5) /max(input_volume.shape[-2:])
        projections = astra_sim.create_CB_projection(input_volume, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz)

        reconstructed_volume = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz *1001/501)

        rad_slices_CB = radial_slice_sampling(reconstructed_volume, theta_range)

        os.makedirs(os.path.join(save_folder, phantom_folders[i]), exist_ok=True)
        for j in range(len(theta_range)):    
            print(f"\rSaving {phantom_folders[i]} s{j+1:0>3d}", end=' '*5)        
            imsave(os.path.join(save_folder, phantom_folders[i], f'CB_source_s{j+1:0>4d}.tif'), rad_slices_CB[j])
            imsave(os.path.join(save_folder, phantom_folders[i], f'CT_target_s{j+1:0>4d}.tif'), rad_slices_input[j])

