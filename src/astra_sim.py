import astra
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import foam_ct_phantom

from . import nesterov_gradient
from .utils import timeit


class FleX_ray_scanner():
    """Class for storage of the CBCT scanners intrinsic parameters, sizes in mm"""

    def __init__(self, name='FleX_ray_scanner',
                 detector_size=(1944, 1536),
                 pixel_size=74.8e-3,
                 bin_factor=2,
                 source_origin_dist=66,
                 source_detector_dist=199):


        self.name = name
        # (n_rows, n_cols)
        self.detector_size = detector_size
        # Isotropic pixels
        self.pixel_size = pixel_size

        self.bin_factor = int(bin_factor)
        self.detector_binned_size = tuple(map(lambda x: int(x/bin_factor), self.detector_size))
        self.pixel_binned_size = bin_factor * self.pixel_size

        self.source_origin_dist = source_origin_dist
        self.source_detector_dist = source_detector_dist
        self.origin_detector_dist = self.source_detector_dist - self.source_origin_dist


        self.FoV = tuple(map(lambda x: x * self.pixel_size, self.detector_size))


def rotate_astra_vec_geom(vecs, theta):
    """Rotates the scanning vector geometry by theta rad in XY plane"""

    rot_mat = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [            0,              0, 1]])

    return np.concatenate([vecs[:, i:i+3] @ rot_mat.T for i in range(0, 12, 3)], axis=1)


def create_scan_geometry(scan_params, n_projs, elevation=0):
    """Creates vectors for free scanning geometry with CW rotation of the X-ray sources around the sample

        Args:
        -----
            scan_params (class): scanner class to get relevant metadata from
            n_projs (int): number of projections, regularly sampled over [0, 2*np.pi[
            elevations (float): height of the source from midplane during the scan, (in mm)

        Returns:
        --------
            vecs (np.ndarray): projection vectors characterizing the scan goemetry
    """
        
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


@timeit
def create_CB_projection(ct_volume, scanner_params, proj_vecs, voxel_size=.1, **kwargs):
    """
        Args:
        -----
            ct_volume (np.ndarray): [z,x,y] axis order at import
            scanner_params (class): scanner class to get relevant metadata from
            proj_vecs (np.ndarray): vecs of the scan trajectory
            voxel_size (float): voxel size in mm

            --optional--
            gpu_id (int): GPU to run astra on, can be set in globals(), defaults twqo -1 otherwise

        Returns:
        --------
            projections (np.ndarray): projection data
    """

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', -1)))
    
    # [y,x,z] axis order for size, [x,y,z] for volume shape, why...?
    vol_geom = astra.creators.create_vol_geom(*np.transpose(ct_volume, (1,2,0)).shape,
        *[sign*size/2*voxel_size for size in np.transpose(ct_volume, (2,1,0)).shape for sign in [-1,1]]
    )

    # [z,x,y] axis order for volume data
    proj_id = astra.data3d.create('-vol', vol_geom, data=ct_volume)    
    proj_geom = astra.create_proj_geom('cone_vec', 
                                       *scanner_params.detector_binned_size, 
                                       proj_vecs)

    projections_id, projections = astra.creators.create_sino3d_gpu(
        proj_id, proj_geom, vol_geom)

    astra.data3d.delete(projections_id)

    # from [rows,proj_slc,cols] to [proj_slc,rows,cols]
    return np.transpose(projections, (1,0,2))


@timeit
def FDK_reconstruction(projections, scanner_params, proj_vecs, voxel_size=.1, rec_shape=501, **kwargs):
    """Uses FDK method to reconstruct CT volume from CB projections
        
        Args:
        -----
            projections (np.ndarray): [proj_slc, rows, cols] CBCT projections
            scanner_params (class): class containing scanner data
            proj_vecs (np.ndarray): vects describing the scanning used for reconstruction
            voxel_size (float): size of the voxels in the reconstructed volume
            rec_shape (int/tuple): shape of the reconstructed volume tuple with 3 dims [z,x,y] or int if isotropic

            --optional--
            gpu_id (int): GPU for astra to use if not set globaly, defaults to -1
            
        Returns:
        --------
            reconstruction (np.ndarray): [z,x,y] recontructed CT volume
    """

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', -1)))

    # from [proj_slc,rows,cols] to [rows,proj_slc,cols]
    projections = np.transpose(projections, (1,0,2))
    proj_geom = astra.create_proj_geom('cone_vec', *scanner_params.detector_binned_size, proj_vecs)    
    projections_id = astra.data3d.create('-sino', proj_geom, projections)

    # [z,x,y] to [y,x,z] axis transposition
    reconstructed_shape = tuple([rec_shape[i] for i in [2,1,0]]) if isinstance(rec_shape, tuple) else (rec_shape,) *3
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
    astra.data3d.delete([projections_id, reconstruction_id])

    return reconstruction


@timeit
def AGD_reconstruction(projections, scanner_params, proj_vecs, voxel_size=.1, rec_shape=501, n_iter=50, **kwargs):
    """Uses AGD method to reconstruct CT volume from CB projections
        
        Args:
        -----
            projections (np.ndarray): [proj_slc, rows, cols] CBCT projections
            scanner_params (class): class containing scanner data
            proj_vecs (np.ndarray): vects describing the scanning used for reconstruction
            voxel_size (float): size of the voxels in the reconstructed volume
            rec_shape (int/tuple): shape of the reconstructed volume tuple with 3 dims [z,x,y] or int if isotropic

            --optional--
            gpu_id (int): GPU for astra to use if not set globaly, defaults to -1
            
        Returns:
        --------
            reconstruction (np.ndarray): [z,x,y] recontructed CT volume
    """

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', -1)))

    # from [proj_slc,rows,cols] to [rows,proj_slc,cols]
    projections = np.transpose(projections, (1,0,2))
    # Gets scanning geometry from vec param
    proj_geom = astra.create_proj_geom('cone_vec', *scanner_params.detector_binned_size, proj_vecs)    
    projections_id = astra.data3d.create('-sino', proj_geom, projections)

    # [z,x,y] to [y,x,z] axis transposition
    reconstructed_shape = tuple([rec_shape[i] for i in [2,1,0]]) if isinstance(rec_shape, tuple) else (rec_shape,) *3
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
    astra.data3d.delete([projections_id, reconstruction_id])
    
    return reconstruction


@timeit
def radial_slice_sampling(ct_volume, theta_range):
    """Samples horizontal slices in a CT volume at angles in theta_range
        Args:
        -----
            ct_volume (np.ndarray): image volume to be sliced, axis as [z,x,y] (XY plane must be of isotropic dims)
            theta_range (np.ndarray/list): angle for the volume to be sliced at

        Returns:
        --------
            rad_slices (np.ndarray): extracted slices, [slc,z,xy], 
                dim[2] vector in the xy plane (cos(theta),sin(theta))
    """

    n_z = ct_volume.shape[0]
    n_xy = ct_volume.shape[1]

    z_gr = np.linspace(-1.0, 1.0, n_z)
    xy_gr = np.linspace(-1.0, 1.0, n_xy)

    interp = RegularGridInterpolator((z_gr, xy_gr, xy_gr), ct_volume, fill_value=0)
    rad_slices = np.empty((len(theta_range), *ct_volume.shape[:2]), dtype='float32')

    z_rad = np.vstack((z_gr,) *n_xy).T

    for i in range(len(theta_range)):
        x_rad = np.vstack((xy_gr * np.cos(theta_range[i]),) *n_z)
        y_rad = np.vstack((xy_gr * -np.sin(theta_range[i]),) *n_z)

        rad_slices[i] = interp(np.vstack((z_rad.flatten(), x_rad.flatten(), y_rad.flatten())).T
                               ).reshape(rad_slices.shape[-2:])
    
    return rad_slices

@timeit
def create_ct_foam_phantom(shape=501, seed=0, gen_new=False):
    """"""
    if gen_new:
        foam_ct_phantom.FoamPhantom.generate('large_foam_phantom.h5', seed, nspheres_per_unit=10000)
    
    # (nx, ny, nx) for phatom volume geometry
    shape = shape if isinstance(shape, tuple) else (shape,) *3

    phantom = foam_ct_phantom.FoamPhantom('large_foam_phantom.h5')
    geom = foam_ct_phantom.VolumeGeometry(*shape, voxsize=3/shape[0])
    phantom.generate_volume('test_phantom.h5', geom)

    # [z,y,x] axis order
    phantom_volume = foam_ct_phantom.load_volume('test_phantom.h5')

    return phantom_volume


@timeit
def radial2axial(rad_slices):
    """From https://github.com/Jomigi/Cone_angle_artifact_reduction code.radial2axial.py"""

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
