from tqdm import tqdm, trange
from pathlib import Path, PurePath
from struct import iter_unpack, unpack

import astra
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from imageio import imread
from pydicom import dcmread
from scipy.interpolate import RectBivariateSpline, interp2d, griddata

import src.utils as utils
from src import astra_sim
from src.utils import imsave, load_projections, mimsave

"""
Prototype code to reconstruct the LDCT dataset in ASTRA, 

was unable to fx issues with FDK reconstruction for helical trajectories
"""


class DicomStack():
    """Class for handling of dicom stack by DicomReader"""

    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, slc):
        # returns slices from the ordered sequence of input images
        if isinstance(slc, slice):
            return (dcmread(self.filenames[i]) for i in range(slc.start, slc.stop, slc.step))

        return dcmread(self.filenames[slc])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def metadata_iterator(self):
        return (pydicom.filereader.read_file_meta_info(filename) for filename in self.filenames)


class DicomReader():
    """Handling of dicom projection for reconstruction of LDCT dataset"""

    def __init__(self, serie_path):
        self.serie_path = self._init_serie_path(serie_path)
        self.slice_paths = sorted(list(self.serie_path.glob('*.dcm')))
        self.slice_stack = DicomStack(self.slice_paths)

        # Usefull metadata for reconstruction 
        self.dicom_metadata_codes = {
            'DataCollectionDiameter': (0x0018, 0x0090),
            'ExposureTime': (0x0018, 0x1150),
            'SpiralPitchFactor': (0x0018, 0x9311),
            'RescaleIntercept': (0x0028,0x1052),
            'RescaleSlope': (0x0028,0x1053),
            'NumberofDetectorRows': (0x7029,0x1010),
            'NumberofDetectorColumns': (0x7029,0x1011),
            'DetectorElementTransverseSpacing': (0x7029,0x1002), # col width
            'DetectorElementAxialSpacing': (0x7029,0x1006), # row width
            'DetectorShape': (0x7029, 0x100B),
            'Rows': (0x0028, 0x0010),
            'Columns': (0x0028, 0x0011),
            'DetectorFocalCenterAngularPosition': (0x7031, 0x1001),  # phy_0
            'DetectorFocalCenterAxialPosition': (0x7031, 0x1002),  # z_0
            'DetectorFocalCenterRadialDistance': (0x7031, 0x1003),  # rho_0
            'ConstantRadialDistance': (0x7031, 0x1031),
            'DetectorCentralElement': (0x7031, 0x1033),
            'SourceAngularPositionShift': (0x7033, 0x100B),  # delta_phy_0
            'SourceAxialPositionShift': (0x7033, 0x100C),  # delta_z_0
            'SourceRadialPositionShift': (0x7033, 0x100D),  # delta_rho_0
            'FlyingFocalSpotMode': (0x7033, 0x100E),
            'NumberofSourceAngularSteps': (0x7033, 0x1013),
            'TypeofProjectionData': (0x7037, 0x1009),
            'TypeofProjectionGeometry': (0x7037, 0x100A),
            'DarkFieldCorrectionFlag': (0x7039,0x1005),
            'FlatFieldCorrectionFlag': (0x7039,0x1006),
            'LogFlag': (0x7039,0x1006),
            'WaterAttenuationCoefficient': (0x7041,0x1001),
        }

        # Gets metadata from the first slice, discards pixel data
        self.serie_metadata = self.get_metadata()

        # additional values for easier computations
        self.serie_metadata['max_fan_angle'] = self.NumberofDetectorColumns /2 *self.DetectorElementTransverseSpacing /self.ConstantRadialDistance
        self.serie_metadata['max_cone_angle'] = np.arctan(self.NumberofDetectorRows /2 *self.DetectorElementAxialSpacing /self.ConstantRadialDistance)
    
    def __len__(self):
        return len(self.slice_stack)

    def __getitem__(self, idx):
        return self.slice_stack[idx]

    def __getattr__(self, name):
        attr = self.serie_metadata[name]
        
        # Handles the dict value depending on its format
        if name == 'WaterAttenuationCoefficient':
            return float(str(attr)[2:-1])

        if not isinstance(attr, bytes):
            return attr

        # Value as binary int
        if len(attr) == 2:
            return self._decode_int(attr)

        # Values as binary float or sequence of floats
        if not len(attr) % 4:
            return self._decode_float(attr, not (len(attr)//4)==1)

        raise ValueError(f"Couldn't decode object: '{name}' with value '{attr}'")

    @staticmethod
    def _init_serie_path(path):        
        path = Path(path) if isinstance(path, (str, PurePath)) else path

        if isinstance(path, Path):
            if not path.exists() or not path.is_dir():
                raise ValueError(f"serie_path does not exist or is not a directory")

            return path

        else:
            raise ValueError(f"serie_path must be path like object, is: {type(path)}")

    def get_metadata(self):
        """Initializes metadata to that of the first slice"""
        first_slice = self.slice_stack[0]

        return dict((attr_name, first_slice[tag].value) for attr_name, tag in self.dicom_metadata_codes.items())
    
    @staticmethod        
    def _decode_int(value):
        """Decodes byte object to uint16 (big-endian encoding)"""
        return unpack('<H', value)[0]
    
    @staticmethod
    def _decode_float(value, multiple_values=False):
        """Decodes byte object to float32 (big-endian encoding)"""
        if multiple_values:
            return tuple(val[0] for val in iter_unpack('<f', value))

        return unpack('<f', value)[0]

    @staticmethod
    def _to_cartesian_coords(vecs):
        """Point position in cylindrical coordinates (phy,z,rho) to (x,y,z)"""

        return np.stack([-vecs[:,2]*np.sin(vecs[:,0]),
                         vecs[:,2]*np.cos(vecs[:,0]),
                         vecs[:,1]], axis=1)


    def get_projections_and_geometry(self, slc=None):

        def overlap_mask(detector_shape, spiral_pitch_factor):
            """WIP: mask to weigh projection overlap"""
            if .5 > spiral_pitch_factor <= 1: raise ValueError("Spiral pitch must be in [0.5,1[")
            
            overlap_mask = np.ones(detector_shape)
            n_overlap_rows = (1-spiral_pitch_factor) *detector_shape[0]

            overlap_mask[:int(n_overlap_rows)//2] = 0

            # Half value for rows that fully overlap
            overlap_mask[:int(n_overlap_rows)] = overlap_mask[-int(n_overlap_rows):] = .5
            # Scaled value for partially overlapping rows
            overlap_mask[int(n_overlap_rows)+1] = overlap_mask[-int(n_overlap_rows)-1] \
                 = n_overlap_rows -np.ceil(n_overlap_rows)
            overlap_mask[0] = overlap_mask[-1] = 1 -(n_overlap_rows -np.ceil(n_overlap_rows))

            return overlap_mask

        # initialize interpolator for cylindrical to flat detector ray interpolation
        virtual_projection = self.interp_on_virtual_detector()

        if slc is None: slc = slice(0, len(self.slice_stack), 4)
        n_proj = (slc.stop -1 -slc.start) //slc.step +1

        # Initialize empty volumes
        source_position = np.empty((n_proj, 3), dtype='float32')
        source_delta = np.empty((n_proj, 3), dtype='float32')
        sinogram = np.empty((n_proj, self.NumberofDetectorRows, self.NumberofDetectorColumns), dtype='float32')
        overlap_weighting_mask = overlap_mask(sinogram[0].shape, self.SpiralPitchFactor)

        for i, slc in tqdm(enumerate(self.slice_stack[slc]), total=n_proj):  
            # (phy_0, z_0, rho_0)
            source_position[i] = list(map(lambda x: self._decode_float(x.value), 
                                        (slc[0x7031, 0x1001], slc[0x7031, 0x1002], slc[0x7031, 0x1003])))
            # FFS position delta
            source_delta[i] = list(map(lambda x: self._decode_float(x.value), 
                                        (slc[0x7033, 0x100B], slc[0x7033, 0x100C], slc[0x7033, 0x100D])))

            # Can change between slices
            rescale_slope, rescale_intercept = slc[0x0028,0x1053].value, slc[0x0028,0x1052].value
            
            sinogram[i] = virtual_projection((slc.pixel_array.T *rescale_slope +rescale_intercept) *overlap_weighting_mask) 
  
        # (columns, rows)
        detector_central_elem = self.DetectorCentralElement        
        # row_width, col_width = self.DetectorElementAxialSpacing, self.DetectorElementTransverseSpacing
        row_width, col_width = self.DetectorElementAxialSpacing, self.virtual_transverse_spacing

        # vector from central element to center of detector
        delta_center = (self.NumberofDetectorRows/2 -detector_central_elem[1], 
                        self.NumberofDetectorColumns/2 -detector_central_elem[0])

        # position of central element
        detector_position = np.stack([source_position[:,0] +np.pi, source_position[:,1],
                                      self.ConstantRadialDistance -source_position[:,2]], axis=1)

        # Adds FFS delta_position
        source_position += source_delta

        # position of detector center
        detector_position[:,0] += delta_center[1] *col_width /self.ConstantRadialDistance # assume tan(\theta) \approx \theta & cos(\theta) \approx 1
        detector_position[:,1] -= delta_center[0] *row_width

        # col, row unit vectors
        u = np.stack([col_width*np.cos(source_position[:,0]),
                      col_width*np.sin(source_position[:,0]),
                      np.zeros(len(source_position))], axis=1)
        v = np.concatenate([np.zeros((len(source_position),2)), 
                            np.full((len(source_position),1), -row_width)], axis=1)
        
        proj_vecs = np.concatenate([self._to_cartesian_coords(source_position), 
                                    self._to_cartesian_coords(detector_position), u, v], axis=1)

        return np.clip(sinogram, 0, None), proj_vecs

 
    def interp_on_virtual_detector(self):
        """Closure to initialize interpolator params"""

        detector_radius = self.ConstantRadialDistance
        n_rows, n_cols = self.NumberofDetectorRows, self.NumberofDetectorColumns

        detector_half_size = (n_rows/2 * self.DetectorElementAxialSpacing,
                              n_cols/2 * self.DetectorElementTransverseSpacing)
        virtual_detector_half_size = (detector_radius *np.tan(self.max_cone_angle) / np.cos(self.max_fan_angle),
                                      detector_radius *np.tan(self.max_fan_angle))

        # sets params of the virtual flat detector
        self.serie_metadata['virtual_detector_height'] = virtual_detector_half_size[0]
        self.serie_metadata['virtual_detector_half_size'] = virtual_detector_half_size[1]
        self.serie_metadata['virtual_axial_spacing'] = virtual_detector_half_size[0] *2 /n_rows
        self.serie_metadata['virtual_transverse_spacing'] = virtual_detector_half_size[1] *2 /n_cols

        true_col_gr, true_row_gr = np.meshgrid(np.linspace(-detector_half_size[1], detector_half_size[1], n_cols),
                                               np.linspace(-detector_half_size[0], detector_half_size[0], n_rows))

        proj_col_gr = detector_radius *np.tan(true_col_gr /detector_radius)
        proj_row_gr = np.tan(true_row_gr /detector_radius) * np.sqrt(proj_col_gr**2 + detector_radius**2)

        virtual_col_gr, virtual_row_gr = np.meshgrid(np.linspace(-virtual_detector_half_size[1], virtual_detector_half_size[1], n_cols), 
                                                     np.linspace(-detector_half_size[0], detector_half_size[0], n_rows))
        # virtual_col_gr, virtual_row_gr = np.meshgrid(np.linspace(-virtual_detector_half_size[1], virtual_detector_half_size[1], n_cols), 
        #                                              np.linspace(-virtual_detector_half_size[0], virtual_detector_half_size[0], n_rows))

        backproj_col_gr = detector_radius *np.arctan(virtual_col_gr /detector_radius)
        backproj_row_gr = detector_radius *np.arctan(virtual_row_gr /np.sqrt(backproj_col_gr**2 + detector_radius**2))

        # col_dist = in-plane dist from detector pixel to source
        col_dist, row_dist = np.meshgrid(np.sqrt(virtual_col_gr[0]**2 + detector_radius**2), virtual_row_gr[:,0])
        relative_ray_path_length = (np.sqrt(row_dist**2 + col_dist**2) /detector_radius).astype('float32')

        # fan angle, cone angle
        psy = np.stack((np.linspace(-self.max_fan_angle, self.max_fan_angle, n_cols),) *n_rows, axis=0)
        kappa = np.stack((np.linspace(-self.max_cone_angle, self.max_cone_angle, n_rows),) *n_cols, axis=1)
        
        def interpolate(proj_slice):  
            """Interpolator, TODO test both methods and both ray weighting scheme"""
            
            interp = RectBivariateSpline(true_row_gr[:,0], true_col_gr[0], proj_slice)
            virtual_proj = interp(backproj_row_gr, backproj_col_gr, grid=False).reshape(proj_slice.shape)

            return virtual_proj -np.log(np.cos(psy)*np.cos(kappa))
            
            # virtual_proj = griddata(np.stack((proj_row_gr.flatten(), proj_col_gr.flatten()), axis=1),
            #                         proj_slice.flatten(),
            #                         np.stack((virtual_row_gr.flatten(), virtual_col_gr.flatten()), axis=1), 
            #                         fill_value=0).reshape(proj_slice.shape)

            # return virtual_proj *relative_ray_path_length
 
        return interpolate
        

def iterative_FDK_reconstruction(projections, scanner_params, proj_vecs, voxel_size=.1, rec_shape=501, vol_center=0, **kwargs):

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', -1)))

    sample_size = 8000

    # [z,x,y] to [y,x,z] axis transposition
    vol_center = tuple([vol_center[i] for i in [2,1,0]]) if isinstance(vol_center, tuple) else (vol_center,) *3
    reconstructed_shape = tuple([rec_shape[i] for i in [2,1,0]]) if isinstance(rec_shape, tuple) else (rec_shape,) *3

    vol_geom = astra.creators.create_vol_geom(*reconstructed_shape,
        *[center+sign*size/2*voxel_size for center, size in zip(vol_center, reconstructed_shape) for sign in [-1, 1]]
    )
    
    for i in trange(0, len(projections), sample_size):

        # from [proj_slc,rows,cols] to [rows,proj_slc,cols]
        proj_geom = astra.create_proj_geom('cone_vec', *scanner_params.detector_effective_size, proj_vecs[i:i+sample_size])    
        projections_id = astra.data3d.create('-sino', proj_geom, np.transpose(projections[i:i+sample_size], (1,0,2)))       

        if not i:
            reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
        else:
            reconstruction_id = astra.data3d.create('-vol', vol_geom, data=astra.data3d.get(reconstruction_id))

        alg_cfg = astra.astra_dict('FDK_CUDA')
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        alg_cfg['option'] = {'ShortScan': False}
        algorithm_id = astra.algorithm.create(alg_cfg)

        astra.algorithm.run(algorithm_id)

        # Free ressources
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(projections_id)

    reconstruction = astra.data3d.get(reconstruction_id)
    astra.data3d.delete(reconstruction_id)

    return reconstruction

def FBP_reconstruction(projections, scanner_params, proj_vecs, voxel_size=.1, rec_shape=501, vol_center=0, **kwargs):

    astra.astra.set_gpu_index(globals().get('GPU_ID', kwargs.get('gpu_id', -1)))

    # from [proj_slc,rows,cols] to [rows,proj_slc,cols]
    projections = np.transpose(projections, (1,0,2))
    proj_geom = astra.create_proj_geom('cone_vec', *scanner_params.detector_effective_size, proj_vecs)    
    projections_id = astra.data3d.create('-sino', proj_geom, projections)

    # [z,x,y] to [y,x,z] axis transposition
    vol_center = tuple([vol_center[i] for i in [2,1,0]]) if isinstance(vol_center, tuple) else (vol_center,) *3
    reconstructed_shape = tuple([rec_shape[i] for i in [2,1,0]]) if isinstance(rec_shape, tuple) else (rec_shape,) *3
    
    vol_geom = astra.creators.create_vol_geom(*reconstructed_shape,
        *[center+sign*size/2*voxel_size for center, size in zip(vol_center, reconstructed_shape) for sign in [-1, 1]]
    )
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

    alg_cfg = astra.astra_dict('BP3D_CUDA')
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)

    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data3d.get(reconstruction_id)

    # Free ressources
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete([projections_id, reconstruction_id])

    return reconstruction


def reconstruct_MDCT():

    serie_path = Path('data/N005/1.000000-Full_dose_projections-20314/')
    # serie_path = Path('data/N012/09-03-2018-06804/1.000000-Full dose projections-88808')
    # serie_path = Path('data/N296/09-03-2018-88975/1.000000-Full dose projections-84660')

    reader = DicomReader(serie_path)
    print(reader.NumberofSourceAngularSteps)
    projections, proj_vecs = reader.get_projections_and_geometry(slc=slice(0, 0+ 8*4608, 1))
    # projections /= projections.max() *reader.NumberofSourceAngularSteps *10
    # print(projections.min(), projections.max())
    
    mean_z = proj_vecs[:,2].mean()
    proj_vecs[:,2] -= mean_z
    proj_vecs[:,5] -= mean_z

    fig, ax = plt.subplots(figsize=(10,10))
    ax2 = ax.twinx()

    ax.plot(proj_vecs[:,0])
    ax.plot(proj_vecs[:,1])

    ax2.plot(proj_vecs[:,2])

    plt.savefig('outputs/helical_coords.png')

    print(reader.NumberofDetectorRows *reader.DetectorElementAxialSpacing)
    print(proj_vecs[0::4608, 2])
    print(list(map(lambda i: proj_vecs[0::4608, 2][i] -proj_vecs[0::4608, 2][i+3], range(0, len(proj_vecs[0::4608, 2])-3))))
    # print(proj_vecs[4608])
    # sys.exit()

    scanner_params = astra_sim.SiemensCT(detector_size=(reader.NumberofDetectorRows, reader.NumberofDetectorColumns),
                                         pixel_size=(reader.DetectorElementAxialSpacing, reader.virtual_transverse_spacing),
                                        #  pixel_size=(reader.DetectorElementAxialSpacing, reader.DetectorElementTransverseSpacing),
                                         source_origin_dist=reader.DetectorFocalCenterRadialDistance,
                                         source_detector_dist=reader.ConstantRadialDistance)

    scanner_params = astra_sim.SiemensCT(detector_size=(reader.NumberofDetectorRows, reader.NumberofDetectorColumns),
                                         pixel_size=(1, reader.virtual_transverse_spacing),
                                        #  pixel_size=(reader.DetectorElementAxialSpacing, reader.DetectorElementTransverseSpacing),
                                         source_origin_dist=reader.DetectorFocalCenterRadialDistance,
                                         source_detector_dist=reader.ConstantRadialDistance)
    
    folder = Path('/data/fdelberghe/Walnuts/Walnut13')
    agd_images = sorted(folder.glob('Reconstructions/full_AGD_50*.tiff'), key=utils._nat_sort)
    agd_volume = np.stack([imread(file) for file in agd_images], axis=0)
    
    # proj_vecs[:,2] *= .5/.325
    # proj_vecs[:,5] *= .5/.325

    z = np.linspace(-.1*5*32, .1*5*32, 5*720)
    phy = np.linspace(0, 5 * 2*np.pi, 5*720)

    proj_vecs = np.zeros((5*720,12))

    proj_vecs[:,0], proj_vecs[:,3] = 595*np.sin(phy), -(1085-595)*np.sin(phy)
    proj_vecs[:,1], proj_vecs[:,4] = -595*np.cos(phy), (1085-595)*np.cos(phy)
    proj_vecs[:,2], proj_vecs[:,5] = z, z

    proj_vecs[:,6] = reader.virtual_transverse_spacing*np.cos(phy)
    proj_vecs[:,7] = reader.virtual_transverse_spacing*np.sin(phy)

    proj_vecs[:,11] = 1

    projections = astra_sim.create_CB_projection(agd_volume, scanner_params, proj_vecs, voxel_size=.5)
    # projections[:,0] = .75; projections[:,-2] = .25; projections[:,-1] = 0
    utils.save_vid(f'outputs/projections.avi', projections[...,None])
    # sys.exit()

    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs,
                                                  voxel_size=.5, rec_shape=501, vol_center=(0,0,0), gpu_id=GPU_ID)

    water_attenuation_coeff = reader.WaterAttenuationCoefficient
    reconstruction = 1000 *(reconstruction -water_attenuation_coeff) /water_attenuation_coeff
    print(reconstruction.min(), reconstruction.max())

    # Path('data/N005/reconstruction/').mkdir(exist_ok=True)
    # for i in range(len(reconstruction)):
    #     imsave(f'data/N005/reconstruction/slice_{i:0>4}.tif', reconstruction[i])

    radial_slices = astra_sim.radial_slice_sampling(np.clip(reconstruction, -1500, 2500), np.linspace(0, 2*np.pi, 360, endpoint=False))
    
    utils.save_vid(f'outputs/LDCT_reconstruction_radial.avi', radial_slices[...,None])
    utils.save_vid(f'outputs/LDCT_reconstruction_axial.avi', reconstruction[...,None])

    # for i in range(len(radial_slices)):
    #     imsave(f'data/N005/reconstruction/slice_radial_{i:0>4}.tif', radial_slices[i])