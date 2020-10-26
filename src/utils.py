import glob
import os
import time

import imageio
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_GRAY2RGB


class FleX_ray_scanner():
    """Class for storage of the CBCT scanners intrinsic parameters, sizes in mm"""

    def __init__(self, name='FleX_ray_scanner'):

        self.name = name

        # (n_rows, n_cols)
        self.detector_size = (1944, 1536)
        # self.detector_size = self.detector_size[::-1]
        self.detector_binned_size = tuple(map(lambda x: int(x/2), self.detector_size))

        # Isotropic pixels
        self.pixel_size = 74.8 * 1e-3
        self.pixel_binned_size = 2 * self.pixel_size

        self.source_origin_dist = 66
        self.source_detector_dist = 199
        self.origin_detector_dist = self.source_detector_dist - self.source_origin_dist

        self.FoV = tuple(map(lambda x: x * self.pixel_size, self.detector_size))


def to_astra_coords(volume):
    """[z, x, y] -> [x, z, y]"""
    return np.transpose(volume, (1,0,2))

def from_astra_coords(volume):
    """[x, z, y] -> [z, x, y] same axis flipping to and from"""
    return to_astra_coords(volume)

def flip_trans(image):
    """Re-orients projections correctly"""
    return np.transpose(np.flipud(image))


def load_projections(walnut_path, orbit_id, angular_sub_sampling=1):
    """Loads all CB projections from one walnut/orbit pair with associated info for reconstruction"""

    print(f"Loading: {walnut_path}...")

    proj_files = sorted(glob.glob(os.path.join(walnut_path, 'Projections', f'tubeV{orbit_id}', 'scan_*.tif')))  
    flat_files = sorted(glob.glob(os.path.join(walnut_path, 'Projections', f'tubeV{orbit_id}', 'io*.tif')))
    dark_file = os.path.join(walnut_path, 'Projections', f'tubeV{orbit_id}', 'di000000.tif')
    vec_file = os.path.join(walnut_path, 'Projections', f'tubeV{orbit_id}', 'scan_geom_corrected.geom')

    projections = np.stack([flip_trans(imageio.imread(file)) for file in proj_files], axis=0)

    dark_field = flip_trans(imageio.imread(dark_file))
    flat_field = np.mean(np.stack([flip_trans(imageio.imread(file)) for file in flat_files], axis=0), axis=0)

    # first and last are projections under the same angle
    projs_idx = range(0, 1200, angular_sub_sampling)
    vecs = np.loadtxt(vec_file)

    # Projections acquired in reverse order of vecs
    projections = projections[projs_idx][::-1]
    vecs = vecs[projs_idx]

    return projections, dark_field, flat_field, vecs


def rescale_before_saving(func):
    """Decorator for imageio.imsave & imageio.mimsave functions, rescales float 
    images to uint8 range and changes type to uint8 to prevent warning messages"""

    def rescale(*args, **kwargs):

        if 'im' in kwargs.keys() or 'ims' in kwargs.keys():
            im = kwargs.pop('im', kwargs['ims'])
        else:
            im = args[1]

        if im.min() >= 0:
            if im.max() <= 1:
                # Normalizing to [0,1] then rescaling to uint
                # im = ((im-im.min()) / (im.max()-im.min())).astype('uint8')

                # # Resclaing [0,1] floats to uint8
                im = (im *255).astype('uint8')

            elif im.max() <= 255:
                # Matches type to avoid warnings
                im = im.astype('uint8')
            
        elif im.min() >= -1 and im.max() <= 1:
            # Normalizing to [0,1] then rescaling to uint
            im = ((im-im.min()) / (im.max()-im.min())).astype('uint8')

            # Rescaling [-1,1] floats to uint8
            # im = ((im +1) *127.5).astype('uint8')
        
        uri = kwargs.pop('uri', args[0])

        return func(uri, im, **kwargs)

    return rescale

@rescale_before_saving
def imsave(*args, **kwargs):
    return imageio.imsave(*args, **kwargs)

@rescale_before_saving
def mimsave(*args, **kwargs):
    return imageio.mimsave(*args, **kwargs)


def save_vid(filename, image_stack, codec='MJPG', fps=30.0, **kwargs):
    """Saves image stack as a video file (better compression than gifs)
        Args:
        -----
            filename (str): path to savefile
            image_stack (np.ndarray): image stack in [z/t,x,y,c] format
            codec (str): opencv fourcc compatible codec, 4 char long str
    """

    fourcc = VideoWriter_fourcc(*codec)
    out = VideoWriter(filename, fourcc, fps, image_stack.shape[1:3][::-1], **kwargs)

    # Rescale for video compatible format
    if not image_stack.dtype is np.uint8:
        image_stack = np.clip(image_stack *255, 0,255).astype('uint8')

    for i in range(image_stack.shape[0]):
        if image_stack.shape[-1] == 1:
            out.write(cvtColor(image_stack[i], COLOR_GRAY2RGB))

        elif image_stack.shape[-1] == 3:
            out.write(image_stack[i])

        else:
            raise ValueError(f"Wrong number of channels for frame, should be 1 or 3, is {image_stack[i].shape[-1]}")

    out.release()


def time_function(func):
    """Decorator to monitor computing time"""

    f_name = ' '.join(func.__name__.split('_'))

    def time_wrapper(*args, **kwargs):
        start_time = time.time()
        print(f'Starting {f_name}...', end=' ', flush=True)
        return_values = func(*args, **kwargs)
        print(f"Done! (took {time.time() -start_time:.2f}s)")
        
        return return_values

    return time_wrapper
