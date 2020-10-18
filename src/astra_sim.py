import glob
import os
import sys
import time

import astra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import foam_ct_phantom

from . import nesterov_gradient
from .utils import to_astra_coords, from_astra_coords, time_function


def rotate_astra_vec_geom(vecs, theta):
    """Rotates the scanning vector geometry by theta rad"""

    rot_mat = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [            0,              0, 1]])

    return np.concatenate([vecs[:, i:i+3] @ rot_mat.T for i in range(0, 12, 3)], axis=1)


def create_scan_geometry(scan_params, n_projs=360, elevation=0):
        
    scan_angles = np.linspace(0, 2*np.pi, n_projs, endpoint=False)

    src = np.array([(scan_params.source_origin_dist * np.sin(theta),
                     scan_params.source_origin_dist * -np.cos(theta),
                     elevation) 
                     for theta in scan_angles])

    d = np.array([(scan_params.origin_detector_dist * -np.sin(theta),
                   scan_params.origin_detector_dist * np.cos(theta),
                   -elevation /scan_params.source_origin_dist *scan_params.origin_detector_dist)
                  for theta in scan_angles])
   
    # col pixel spacing vector
    u = np.array([(scan_params.pixel_binned_size * np.cos(theta),
                   scan_params.pixel_binned_size * np.sin(theta),
                   0)
                  for theta in scan_angles])

    # row pixel spacing vector
    v = np.array([(0, 0, scan_params.pixel_binned_size) for _ in scan_angles])

    return np.concatenate((src, d, u, v), axis=1).astype('float32')


@time_function
def create_CB_projection(ct_volume, scanner_params, voxel_size=.1, proj_vecs=None, n_projs=180, **kwargs):
    """
        ARGS:
            ct_volume (np.ndarray): [z, x, y] axis order at import
            scanner_params (class):
            voxel_size (float): voxel size in mm
            n_projs (int):
            gpu_id (int):

        RETURNS:
            projections (np.ndarray): projection data
    """

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', 0)))

    # [y,x,z] axis order
    vol_geom = astra.creators.create_vol_geom(*np.transpose(ct_volume, (2,1,0)).shape,
        *[sign*size/2*voxel_size for size in np.transpose(ct_volume, (2,1,0)).shape for sign in [-1,1]]
    )
    # [z,y,x] axis order
    proj_id = astra.data3d.create('-vol', vol_geom, data=np.transpose(ct_volume, (0,2,1)))

    angles = np.linspace(0, 2 * np.pi, num=n_projs, endpoint=False)
    
    if proj_vecs is not None:
        proj_geom = astra.create_proj_geom('cone_vec', *scanner_params.detector_binned_size, proj_vecs)
                
    else:
        proj_geom = astra.create_proj_geom(
            'cone', *(scanner_params.pixel_binned_size,) *2, 
            *scanner_params.detector_binned_size, angles,
            scanner_params.source_origin_dist, scanner_params.origin_detector_dist)

    projections_id, projections = astra.creators.create_sino3d_gpu(
        proj_id, proj_geom, vol_geom)

    astra.data3d.delete(projections_id)

    projections /= projections.max()

    # from [rows,proj_slc,cols] to [proj_slc,rows,cols]
    return np.transpose(projections, (1,0,2))


@time_function
def FDK_reconstruction(projections, scanner_params, proj_vecs=None, voxel_size=.1, rec_shape=501, **kwargs):
    """Uses FDK method to reconstruct CT volume from CB projections
        
        Args:
        -----
            projections (np.ndarray): [proj_slc, rows, cols] CBCT projections
            scanner_params (class): class containing scanner data
            proj_vecs (np.ndarray): vects describing the scanning used for reconstruction
            voxel_size (float): size of the reconstructed volumes
            rec_shape (int/tuple): shape of the reconstructed volume tuple with 3 dims [z,x,y] or int if isotropic

            --optional--
            gpu_id (int): GPU for astra to use if not set globaly, defaults to 0
            
        Returns:
        --------
            recontstruction (np.ndarray): [z,x,y] recontructed CT volume
    """

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', 0)))

    # from [proj_slc,rows,cols] to [rows,proj_slc,cols]
    projections = to_astra_coords(projections)    

    if proj_vecs is not None:
        # Gets scanning geometry from vec param
        proj_geom = astra.create_proj_geom('cone_vec', *scanner_params.detector_binned_size, proj_vecs)

    else:
        # Assumes scanning geometry from scanner params
        angles = np.linspace(0, 2 *np.pi, num=projections.shape[1], endpoint=False)
        proj_geom = astra.create_proj_geom(
            'cone', *(scanner_params.pixel_binned_size,) *2, 
            *scanner_params.detector_binned_size, angles,
            scanner_params.source_origin_dist, scanner_params.origin_detector_dist)
    
    projections_id = astra.data3d.create('-sino', proj_geom, projections)

    # [z,x,y] to [y,x,z] axis transposition
    reconstructed_shape = tuple([rec_shape[i] for i in [2,1,0]]) if isinstance(rec_shape, tuple) else (rec_shape,) *3
    voxel_size = 2 *scanner_params.pixel_binned_size * scanner_params.source_origin_dist / scanner_params.source_detector_dist
    vol_geom = astra.creators.create_vol_geom(*reconstructed_shape,
        *[sign*size/2*voxel_size for size in reconstructed_shape for sign in [-1, 1]]
    )
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

    alg_cfg = astra.astra_dict('FDK_CUDA')
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    alg_cfg['option'] = {'ShortScan': False}
    algorithm_id = astra.algorithm.create(alg_cfg)

    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data3d.get(reconstruction_id)

    # Free ressources
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(projections_id)
    astra.data3d.delete(reconstruction_id)

    return reconstruction


@time_function
def AGD_reconstruction(projections, scanner_params, proj_vecs=None, voxel_size=.1, 
                       rec_shape=501, n_iter=50, **kwargs):
    """Uses FDK method to reconstruct CT volume from CB projections
        
        Args:
        -----
            projections (np.ndarray): [proj_slc, rows, cols] CBCT projections
            scanner_params (class): class containing scanner data
            proj_vecs (np.ndarray): vects describing the scanning used for reconstruction
            voxel_size (float): size of the reconstructed volumes
            rec_shape (int/tuple): shape of the reconstructed volume tuple with 3 dims [z,x,y] or int if isotropic
            n_iter (int): number of gradient descent steps

            --optional--
            gpu_id (int): GPU for astra to use if not set globaly, defaults to 0
            
        Returns:
        --------
            recontstruction (np.ndarray): [z,x,y] recontructed CT volume
    """

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', 0)))

    projections = to_astra_coords(projections)    

    if proj_vecs is not None:
        # Gets scanning geometry from vec param
        proj_geom = astra.create_proj_geom('cone_vec', *scanner_params.detector_binned_size, proj_vecs)

    else:
        # Assumes scanning geometry from scanner params
        angles = np.linspace(0, 2 *np.pi, num=projections.shape[1], endpoint=False)
        proj_geom = astra.create_proj_geom(
            'cone', *(scanner_params.pixel_binned_size,) *2, 
            *scanner_params.detector_binned_size, angles,
            scanner_params.source_origin_dist, scanner_params.origin_detector_dist)
    
    projections_id = astra.data3d.create('-sino', proj_geom, projections)

    reconstructed_shape = rec_shape if isinstance(rec_shape, tuple) else (rec_shape,) *3
    voxel_size = 2 *scanner_params.pixel_binned_size * scanner_params.source_origin_dist / scanner_params.source_detector_dist
    vol_geom = astra.creators.create_vol_geom(*reconstructed_shape,
        *[sign*size/2*voxel_size for size in reconstructed_shape for sign in [-1, 1]]
    )
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)

    astra.plugin.register(nesterov_gradient.AcceleratedGradientPlugin)
    cfg_agd = astra.astra_dict('AGD-PLUGIN')
    cfg_agd['ProjectionDataId'] = projections_id
    cfg_agd['ReconstructionDataId'] = reconstruction_id
    cfg_agd['ProjectorId'] = projector_id
    cfg_agd['option'] = {'MinConstraint': 0}
    alg_id = astra.algorithm.create(cfg_agd)

    astra.algorithm.run(alg_id, n_iter)

    reconstruction = astra.data3d.get(reconstruction_id)

    # release memory allocated by ASTRA structures
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(projections_id)
    astra.data3d.delete(reconstruction_id)

    return reconstruction


def generate_radial_slices(projections, scanner_params, proj_vecs, rec_shape=501, **kwargs):
    """"""
    raise NotImplementedError("")


@time_function
def radial_slice_sampling(ct_volume, theta_range):
    """Samples horizontal slices in a CT volume at angles in theta_range
        Args:
        -----
            ct_volume (np.ndarray): image volume to be sliced, axis as [z,x,y] isotropic volumes (for now)
            theta_range (np.ndarray/list): angle for the volume to be sliced at

        Returns:
        --------
            rad_slices (np.ndarray): extracted slices, [slc,z,xy], 
                dim[2] vector in the xy plane (cos(theta),sin(theta))
    """

    # Cast range of slices to np.ndarray if possible
    if isinstance(theta_range, list):
        theta_range = np.array(theta_range)

    n_z = ct_volume.shape[0]

    z_gr = np.linspace(-1.0, 1.0, n_z)
    interp = RegularGridInterpolator((z_gr,) *3, ct_volume, fill_value=0)

    x_rad, y_rad, z_rad = [np.zeros((n_z, n_z)) for _ in range(3)]

    rad_slices = np.empty((theta_range.shape[0], *ct_volume.shape[:2]), dtype='float32')
    z_rad = np.vstack((z_gr,)*n_z)    

    for i in range(theta_range.shape[0]):
        x_rad, y_rad = np.meshgrid(z_gr * np.cos(theta_range[i]), z_gr * -np.sin(theta_range[i]),)

        rad_slices[i] = interp(np.vstack((z_rad.flatten(),
                                          x_rad.T.flatten(),
                                          y_rad.flatten())).T
                               ).reshape((n_z,)*2).T
        
    return rad_slices


def create_ct_foam_phantom(shape=501, filepath='small_foam_phantom.h5'):

    # (nx, ny, nx) for phatom volume geometry
    shape = shape if isinstance(shape, tuple) else (shape,) *3

    phantom = foam_ct_phantom.FoamPhantom(filepath)
    geom = foam_ct_phantom.VolumeGeometry(*shape, voxsize=3/501)
    phantom.generate_volume('test_phantom.h5', geom)

    # [z,y,x] axis order
    phantom_volume = foam_ct_phantom.load_volume('test_phantom.h5')

    return phantom_volume


@time_function
def radial2axial(rad_slices):

    n_rad = rad_slices.shape[0]
    n_x = rad_slices.shape[1]
    n_z = rad_slices.shape[2]

    # We want to describe the voxels of the radial slices in polar coordinates. Therefore, we define a theta to describe the angles of the radial slices.
    theta = np.linspace(0, np.pi, n_rad+1, endpoint=True)

    # we add the first radial slice (theta = 0) to the end to be the data for theta = pi
    rad_slices = np.concatenate((rad_slices,(rad_slices[0,:,:]).reshape(1,n_x,n_x)),axis=0)
    x_gr = np.linspace(-1.0, 1.0, n_x)

    # set up radial and angular coordinates of the x-y grid (needs to be done once for all z slices)
    polar_cords = np.zeros((n_x**2, 2))

    for i in range(n_x*n_x):

        # x-coordinate
        x = x_gr[np.mod(i, n_x)]

        # y-coordinate
        y = x_gr[i//n_x]

        if x == 0:
            polar_cords[i, 0] = np.pi/2  # angular
            polar_cords[i, 1] = y  # radial 

        if y == 0:
            polar_cords[i, 0] = 0 # angular
            polar_cords[i, 1] = x # radial

        elif x != 0 and y != 0:
            polar_cords[i, 0] = np.arctan(y/x)   # angular  
            polar_cords[i, 1] = np.sign(y)*np.sqrt(x**2 + y**2)  # radial
       
        if polar_cords[i,0] < 0:
            polar_cords[i,0] =  polar_cords[i,0] + np.pi # Keep angle positive
                      
    axial_slices = np.zeros([n_z, n_x, n_x])
    
    for z in range(n_z): 
        interpolator = RegularGridInterpolator((theta, x_gr), rad_slices[:,z,:], 
                                               bounds_error=False, fill_value=0)
        axial_slices[z,:,:] = interpolator(polar_cords).reshape(1,n_x,n_x, order='F')       
           
    return axial_slices[:,:,::-1]
