import os
import re
import shutil
import sys
import time
from pathlib import Path

import foam_ct_phantom
import nibabel as nib
import numpy as np
from imageio import imread, imsave
from natsort import natsorted
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange

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


# def build_phantom_dataset(gpu_id=2):
#     """Builds dataset with anthropomorphic phantoms"""
    
#     DATA_PATH = Path('/data/fdelberghe/')
#     save_folder = DATA_PATH/'PhantomsRadial5/'

#     phantom_folders = natsorted(DATA_PATH.glob('AxialPhantoms/*'))

#     scanner_params = FleX_ray_scanner()
#     scanner_trajs = [astra_sim.create_scan_geometry(scanner_params, n_projs=1200, elevation=el) for el in [-15, 0, 15]]
    
#     theta_range = np.linspace(0, np.pi, int(np.round(np.sqrt(2) *501)), endpoint=False)

#     for folder in [phantom_folders[0], phantom_folders[2]]:
#         print(f"Loading {folder}/...")        

#         axial_ims = sorted(folder.glob('*.tif'), key=utils._nat_sort)
#         input_volume = np.stack([imread(file) for file in axial_ims], axis=0).astype('float32')
#         # Estimates the air density from mean intensity at the edges of the volume
#         dark_field = np.mean([input_volume[0].mean(), input_volume[:,0].mean(), input_volume[:,:,0].mean(), input_volume[:,:,-1].mean()])
#         input_volume = (input_volume -dark_field) /(input_volume.max() -dark_field)
        
#         interp_shape = (501, 501)
#         # interp the volume in a box the size of the largest axis
#         max_in_dim = max(input_volume.shape)

#         # Creates grid center on volume center regardless of volume shape
#         z_gr = np.linspace(-input_volume.shape[0] /interp_shape[0] /max_in_dim *501,
#                           input_volume.shape[0] /interp_shape[0] /max_in_dim *501,
#                           input_volume.shape[0])
#         x_gr, y_gr = [np.linspace(-input_volume.shape[j] /interp_shape[1] /max_in_dim *501,
#                                   input_volume.shape[j] /interp_shape[1] /max_in_dim *501,
#                                   input_volume.shape[j]) for j in range(1,3)]

#         interp = RegularGridInterpolator((z_gr, x_gr, y_gr), input_volume, fill_value=0, bounds_error=False)

#         z_gr = np.linspace(-1.0, 1.0, interp_shape[0])
#         xy_gr = np.linspace(-1.0, 1.0, interp_shape[1])

#         z_rad = np.vstack((z_gr,) *interp_shape[1]).T

#         (save_folder/folder.name).mkdir(parents=True, exist_ok=True)

#         for j in range(len(theta_range)):
#             x_rad = np.vstack((xy_gr * np.cos(theta_range[j]),) *interp_shape[0])
#             y_rad = np.vstack((xy_gr * -np.sin(theta_range[j]),) *interp_shape[0])

#             rad_slices_input = interp(np.vstack((z_rad.flatten(), x_rad.flatten(), y_rad.flatten())).T
#                                 ).reshape(interp_shape)

#             print(f"\rSaving {folder.name}/CT_target_s{j+1:0>3d}", end=' '*5)  
#             imsave((save_folder/folder.name/ f'CT_target_s{j+1:0>4d}.tif'), rad_slices_input.astype('float32'))
#         print('')

#         # Voxel size for whole cube volume within scan FoV
#         vox_sz = scanner_params.source_origin_dist /(scanner_params.source_detector_dist /min(scanner_params.FoV) +.5) /max_in_dim

#         for i, scanner_traj in enumerate(scanner_trajs):
#             projections = astra_sim.create_CB_projection(input_volume, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz, gpu_id=gpu_id)
#             reconstructed_volume = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz *max_in_dim/501, gpu_id=gpu_id)

#             rad_slices_CB = radial_slice_sampling(reconstructed_volume, theta_range)
            
#             for j in range(len(theta_range)):    
#                 print(f"\rSaving {folder.name}/CB_source_s{j+1:0>3d}", end=' '*5)     
#                 imsave((save_folder/folder.name/ f'CB_source_orbit{i+1:0>2d}_s{j+1:0>4d}.tif'), rad_slices_CB[j])
#             print('')
   

def build_large_phantom_dataset(gpu_id=1):
    """Builds dataset with anthropomorphic phantoms too large to fit in memory"""
    import astra

    @utils.timeit
    def create_half_CB_projection(ct_volume, scanner_params, proj_vecs, voxel_size=.1, **kwargs):
        """
            Args:
            -----
                ct_volume (np.ndarray): [z,x,y] axis order at import
                scanner_params (class): scanner class to get relevant metadata from
                proj_vecs (np.ndarray): vecs of the scan trajectory
                voxel_size (float): voxel size in mm

                --optional--
                gpu_id (int): GPU to run astra on, can be set in globals(), defaults to -1 otherwise

            Returns:
            --------
                projections (np.ndarray): projection data
        """

        astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', -1)))

        source_height = proj_vecs[0,2]
        split_line = int(np.round(len(ct_volume) /2 +source_height/voxel_size))
        print(len(ct_volume), split_line)
        z_dims = list(map(lambda x: x*len(ct_volume)/2*voxel_size, [-1, -1+2*split_line/len(ct_volume), 1]))
        
        full_projections = None #np.empty((len(proj_vecs), *scanner_params.detector_binned_size))
        print(z_dims)

        for i in range(2):
            half_volume = ct_volume[:split_line+1] if not i else ct_volume[split_line:]

            # [y,x,z] axis order for size, [x,y,z] for volume shape, ...why?
            vol_geom = astra.creators.create_vol_geom(*np.transpose(half_volume, (1,2,0)).shape,
                *[sign*size/2*voxel_size for size in half_volume.shape[-2:][::-1] for sign in [-1,1]],
                *z_dims[i:i+2],
            )        

            # [z,x,y] axis order for volume data
            proj_id = astra.data3d.create('-vol', vol_geom, data=half_volume)    
            proj_geom = astra.create_proj_geom('cone_vec', 
                                            *scanner_params.detector_binned_size, 
                                            proj_vecs)

            projections_id, projections = astra.creators.create_sino3d_gpu(
                proj_id, proj_geom, vol_geom)

            astra.data3d.delete([proj_id, projections_id])

            if not i:
                full_projections = projections
                # utils.save_vid('outputs/half_projection_top.avi', np.transpose(projections, (1,0,2))[..., None])
            else:
                full_projections += projections
                # utils.save_vid('outputs/half_projection_bottom.avi', np.transpose(projections, (1,0,2))[..., None])
                # sys.exit()

        # from [rows,proj_slc,cols] to [proj_slc,rows,cols]
        return np.transpose(full_projections, (1,0,2))

    
    DATA_PATH = Path('/data/fdelberghe/')
    save_folder = DATA_PATH/'PhantomsRadial/'

    phantom_folders = natsorted(DATA_PATH.glob('AxialPhantoms/*'))

    scanner_params = FleX_ray_scanner()
    scanner_trajs = [astra_sim.create_scan_geometry(scanner_params, n_projs=1200, elevation=el) for el in [-12, 0, 12]]
    
    theta_range = np.linspace(0, np.pi, int(np.round(np.sqrt(2) *501)), endpoint=False)

    for folder in [phantom_folders[1], phantom_folders[-1]]:
        print(f"Loading {folder}/...")        

        axial_ims = sorted(folder.glob('*.tif'), key=utils._nat_sort)
        input_volume = np.stack([imread(file) for file in axial_ims], axis=0).astype('float32')
        # Estimates the air density from mean intensity at the edges of the volume
        dark_field = np.mean([input_volume[0].mean(), input_volume[:,0].mean(), input_volume[:,:,0].mean(), input_volume[:,:,-1].mean()])
        input_volume = (input_volume -dark_field) /(input_volume.max() -dark_field)
        
        interp_shape = (501, 501)
        # # interp the volume in a box the size of the largest axis
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

        for i in range(len(scanner_trajs)):
            projections = create_half_CB_projection(input_volume, scanner_params, proj_vecs=scanner_trajs[i], voxel_size=vox_sz, gpu_id=gpu_id)
            reconstructed_volume = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=scanner_trajs[i], voxel_size=vox_sz *max_in_dim/501, gpu_id=gpu_id)

            rad_slices_CB = radial_slice_sampling(reconstructed_volume, theta_range)
                
            for j in range(len(theta_range)):    
                print(f"\rSaving {folder.name}/CB_source_s{j+1:0>3d}", end=' '*5)     
                imsave((save_folder/folder.name/ f'CB_source_orbit{i+1:0>2d}_s{j+1:0>4d}.tif'), rad_slices_CB[j])
            print('')


def get_HR_phantoms():

    """High res volumes of 4 of the phantoms"""
    DATA_PATH = Path('/data/fdelberghe/')
    save_folder = DATA_PATH/'PhantomsRadial5/'

    phantom_folders = sorted(DATA_PATH.glob('AxialPhantoms/*'), key=utils._nat_sort)

    for folder in phantom_folders:

        axial_ims = sorted(folder.glob('*.tif'), key=utils._nat_sort)
        input_volume = np.stack([imread(file) for file in axial_ims], axis=0).astype('float32')
        # Estimates the air density from mean intensity at the edges of the volume
        dark_field = np.mean([input_volume[0].mean(), input_volume[:,0].mean(), input_volume[:,:,0].mean(), input_volume[:,:,-1].mean()])
        input_volume = (input_volume -dark_field) /(input_volume.max() -dark_field)

        yield input_volume

def get_usb_phantom():
    
    DATA_PATH = Path('/data/maureen_shares/GE-EvolutionHD-phantoms/')
    phantom_folders = [DATA_PATH / folder for folder in ['usb1/DICOM/PA0/ST0/SE1/',
                                                         'usb1/DICOM/PA0/ST0/SE5/',
                                                         'usb2/DICOM/PA0/ST0/SE3/',
                                                         'usb2/DICOM/PA0/ST0/SE1/',
                                                         'usb2/DICOM/PA0/ST0/SE7/',
                                                         'usb2/DICOM/PA0/ST0/SE5/',
                                                         'usb1/DICOM/PA0/ST0/SE3/']]

    for folder in phantom_folders:
        print(f"Loading {folder}/...")     

        axial_ims = sorted(folder.glob('*'), key=utils._nat_sort)

        # Slices saved in str sort order revert to right order [1, 10, 100, 101, ..., 109, 11, 110, ...] -> [0, 1, 2, 3, ...]
        str_sorted_idx = list(zip(sorted([f'{i+1}' for i in range(len(axial_ims))]), list(range(len(axial_ims)))))
        str_sorted_idx = sorted(str_sorted_idx, key=lambda x: int(x[0]))
        str_sorted_idx = list(zip(*str_sorted_idx))[1]

        input_volume = np.stack([imread(axial_ims[i]) for i in str_sorted_idx], axis=0).astype('float32')
        input_volume = np.clip(input_volume, -1024, None)

        # Estimates the air density from mean intensity at the edges of the volume
        dark_field = np.mean([input_volume[0].mean()])
        input_volume = np.clip((input_volume -dark_field) /(input_volume.max() -dark_field), 0, None)

        yield input_volume


def build_phantom_dataset(input_volume, save_folder, scaling=1, gpu_id=0):

    scanner_params = FleX_ray_scanner()
    scanner_trajs = [astra_sim.create_scan_geometry(scanner_params, n_projs=1200, elevation=el) for el in [-15, 0, 15]]
    
    theta_range = np.linspace(0, np.pi, int(np.round(np.sqrt(2) *501)), endpoint=False)        
    
    interp_shape = (501, 501)
    # interp the volume in a box the size of the largest axis
    max_in_dim = max(input_volume.shape)

    # Creates grid center on volume center regardless of volume shape
    z_gr = np.linspace(-input_volume.shape[0] /interp_shape[0] /max_in_dim *501 *scaling,
                        input_volume.shape[0] /interp_shape[0] /max_in_dim *501 *scaling,
                        input_volume.shape[0])
    x_gr, y_gr = [np.linspace(-input_volume.shape[j] /interp_shape[1] /max_in_dim *501 *scaling,
                                input_volume.shape[j] /interp_shape[1] /max_in_dim *501 *scaling,
                                input_volume.shape[j]) for j in range(1,3)]

    interp = RegularGridInterpolator((z_gr, x_gr, y_gr), input_volume, fill_value=0, bounds_error=False)

    z_gr = np.linspace(-1.0, 1.0, interp_shape[0])
    xy_gr = np.linspace(-1.0, 1.0, interp_shape[1])

    z_rad = np.vstack((z_gr,) *interp_shape[1]).T

    save_folder.mkdir(parents=True, exist_ok=True)

    for j in trange(len(theta_range), desc=f"Saving {save_folder.name}/CT_target"):
        x_rad = np.vstack((xy_gr * np.cos(theta_range[j]),) *interp_shape[0])
        y_rad = np.vstack((xy_gr * -np.sin(theta_range[j]),) *interp_shape[0])

        rad_slices_input = interp(np.vstack((z_rad.flatten(), x_rad.flatten(), y_rad.flatten())).T
                                    ).reshape(interp_shape)

        imsave((save_folder/ f'CT_target_s{j+1:0>4d}.tif'), rad_slices_input.astype('float32'))
    print('')

    # Voxel size for whole cube volume within scan FoV
    vox_sz = scanner_params.source_origin_dist /(scanner_params.source_detector_dist /min(scanner_params.FoV) +.5) /max_in_dim

    for i, scanner_traj in enumerate(scanner_trajs):
        projections = astra_sim.create_CB_projection(input_volume, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz *scaling, gpu_id=gpu_id)
        utils.save_vid(f'outputs/projections.avi', projections[...,None])
        reconstructed_volume = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz *max_in_dim/501, gpu_id=gpu_id)

        rad_slices_CB = radial_slice_sampling(reconstructed_volume, theta_range)
        
        for j in trange(len(theta_range), desc=f"Saving {save_folder.name}/CB_source_orbit{i+1:0>2d}"):    
            imsave(save_folder / f'CB_source_orbit{i+1:0>2d}_s{j+1:0>4d}.tif', rad_slices_CB[j])
        print('')


def build_abdo_dataset(gpu_id=0):
    
    import astra
    astra.astra.set_gpu_index(gpu_id)

    data_path = Path('/data/maureen_shares/BoneMetastases_NIfTI_BodyRef')
    nii_filenames = sorted(data_path.glob('*.nii'), key=utils._nat_sort)

    save_path = Path('/data/fdelberghe/AbdoScans')    

    scanner_params = FleX_ray_scanner()
    scanner_trajs = [astra_sim.create_scan_geometry(scanner_params, n_projs=1200, elevation=el) for el in [-15, 0, 15]]
    
    theta_range = np.linspace(0, np.pi, int(np.round(np.sqrt(2) *501)), endpoint=False)        
    
    interp_shape = (501, 501)
    
    for i, filename in enumerate(nii_filenames):
        if i <= 5: continue
        nii_file = nib.load(filename)
        input_volume = np.transpose(nii_file.get_fdata(), (2,1,0))[::-1]
        input_volume /= input_volume.max()

        # input_volume = 1/ (1+np.exp(-(input_volume -1600)*np.log(4)/400))
        # utils.save_vid(f'outputs/projections.avi', input_volume[::-1,...,None])

        # [x,y,z] scaling factors
        scaling = np.diagonal(nii_file.affine)[:3]
        volume_size = list(map(lambda size,scale: size*scale, input_volume.shape, scaling[::-1]))
        
        # interp the volume in a box the size of the largest axis
        max_in_size = max(volume_size) *1.2

        # Creates grid center on volume center regardless of volume shape
        z_gr = np.linspace(-volume_size[0] /max_in_size,
                           volume_size[0] /max_in_size,
                           input_volume.shape[0])
        x_gr, y_gr = [np.linspace(-volume_size[j] /max_in_size,
                                  volume_size[j] /max_in_size,
                                  input_volume.shape[j]) for j in range(1, 3)]

        interp = RegularGridInterpolator((z_gr, x_gr, y_gr), input_volume, fill_value=0, bounds_error=False)

        z_gr = np.linspace(-1.0, 1.0, interp_shape[0])
        xy_gr = np.linspace(-1.0, 1.0, interp_shape[1])

        z_rad = np.vstack((z_gr,) *interp_shape[1]).T
        
        save_folder = save_path / f'Volume{i+1}'
        save_folder.mkdir(parents=True, exist_ok=True)

        for j in trange(len(theta_range), desc=f"Saving {save_folder.name}/CT_target"):
            x_rad = np.vstack((xy_gr * np.cos(theta_range[j]),) *interp_shape[0])
            y_rad = np.vstack((xy_gr * -np.sin(theta_range[j]),) *interp_shape[0])

            rad_slice = interp(np.vstack((z_rad.flatten(), x_rad.flatten(), y_rad.flatten())).T
                                        ).reshape(interp_shape)

            imsave((save_folder/ f'CT_target_s{j+1:0>4d}.tif'), rad_slice.astype('float32'))
        print('')

        vox_sz = scanner_params.source_origin_dist /(scanner_params.source_detector_dist /min(scanner_params.FoV) +.5)
        print(vox_sz)

        for i, scanner_traj in enumerate(scanner_trajs):

            # ============================== #
            # [y,x,z] axis order for size, [x,y,z] for volume shape, why...?
            vol_geom = astra.creators.create_vol_geom(*np.transpose(input_volume, (1,2,0)).shape,
                *[sign*volume_size[k]/2/vox_sz for k in [2,1,0] for sign in [-1,1]]
            )
            
            # [z,x,y] axis order for volume data
            proj_id = astra.data3d.create('-vol', vol_geom, data=input_volume)    
            proj_geom = astra.create_proj_geom('cone_vec', 
                                            *scanner_params.detector_effective_size, 
                                            scanner_traj)

            projections_id, projections = astra.creators.create_sino3d_gpu(
                proj_id, proj_geom, vol_geom)

            astra.data3d.delete(projections_id)

            # from [rows,proj_slc,cols] to [proj_slc,rows,cols]
            projections = np.transpose(projections, (1,0,2))
            # ============================== #

            # projections = astra_sim.create_CB_projection(input_volume, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz, gpu_id=gpu_id)
            utils.save_vid(f'outputs/projections.avi', projections[...,None])
            reconstructed_volume = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=scanner_traj, voxel_size=vox_sz /501, gpu_id=gpu_id)

            rad_slices_CB = radial_slice_sampling(reconstructed_volume, theta_range)
            
            for j in trange(len(theta_range), desc=f"Saving {save_folder.name}/CB_source_orbit{i+1:0>2d}"):    
                imsave(save_folder / f'CB_source_orbit{i+1:0>2d}_s{j+1:0>4d}.tif', rad_slices_CB[j].astype('float32'))
            print('')

        

if __name__ == '__main__':

    save_folders = [Path(f'/data/fdelberghe/PhantomsRadialSmall/Phantom{i}') for i in range(1,8)]
    for ct_volume, save_folder in zip(get_usb_phantom(), save_folders):
        build_phantom_dataset(ct_volume, save_folder, scaling=.8)
    
