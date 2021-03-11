import glob
import os
from pathlib import Path

import imageio
import numpy as np
from cv2 import COLOR_GRAY2RGB, VideoWriter, VideoWriter_fourcc, cvtColor

from . import _nat_sort


def load_projections(walnut_path, orbit_id, angular_sub_sampling=1):
    """Loads all CB projections from one walnut/orbit pair with associated info for reconstruction"""

    def flip_trans(image):
        """Re-orients projections correctly"""
        return np.transpose(np.flipud(image))


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
        
        def normalize(image):
            if image.max()-image.min() == 0:
                return np.zeros_like(image)

            return (image-image.min()) / (image.max()-image.min())


        im = kwargs.pop('im', kwargs.pop('ims', args[1]))
        uri = kwargs.pop('uri', args[0])

        if uri.split('.')[-1] in ['tif', 'tiff']:
            return func(uri, im, **kwargs)

        # Normalizing to [0,1] then rescaling to uint
        im = (normalize(im) *255).astype('uint8')

        return func(uri, im, **kwargs)

    return rescale

@rescale_before_saving
def imsave(*args, **kwargs):
    return imageio.imsave(*args, **kwargs)

@rescale_before_saving
def mimsave(*args, **kwargs):
    return imageio.mimsave(*args, **kwargs)


def save_vid(filename, image_stack, codec='MJPG', fps=30, **kwargs):
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
        image_stack = ((image_stack-image_stack.min()) / (image_stack.max()-image_stack.min()) *255).astype('uint8')
        # image_stack = np.clip(image_stack *255, 0,255).astype('uint8')

    for i in range(image_stack.shape[0]):
        if image_stack.shape[-1] == 1:
            out.write(cvtColor(image_stack[i], COLOR_GRAY2RGB))

        elif image_stack.shape[-1] == 3:
            out.write(image_stack[i])

        else:
            raise ValueError(f"Wrong number of channels for frame, should be 1 or 3, is {image_stack[i].shape[-1]}")

    out.release()


def load_walnut_ds():

    ds_path = '/data/fdelberghe/FastWalnuts2/'
    walnut_folders = [folder for folder in sorted(os.listdir(ds_path), key=_nat_sort) 
                      if os.path.isdir(os.path.join(ds_path, folder))]

    walnuts_agd_paths = [sorted(glob.glob(os.path.join(ds_path + f"{folder}/agd_*.tif")), key=_nat_sort) 
                         for folder in walnut_folders]
    walnuts_fdk_paths = [
        [sorted(glob.glob(os.path.join(ds_path + f"{folder}/fdk_orbit{orbit_id:02d}*.tif")),
                key=_nat_sort) for orbit_id in [1,2,3]] for folder in walnut_folders]

    return walnuts_agd_paths, walnuts_fdk_paths


def load_foam_phantom_ds():

    ds_path = '/data/fdelberghe/FoamPhantoms/'
    phantom_folders = [folder for folder in sorted(os.listdir(ds_path), key=_nat_sort) 
                      if os.path.isdir(os.path.join(ds_path, folder))]

    phantom_agd_paths = [
        sorted(glob.glob(os.path.join(ds_path + f"{folder}/phantom_true_*.tif")), key=_nat_sort)
        for folder in phantom_folders]

    phantom_fdk_paths = [
        [sorted(glob.glob(os.path.join(ds_path + f"{folder}/phantom_fdk_*_o{orbit_id}*.tif")), key=_nat_sort)
         for orbit_id in [1,2,3]] for folder in phantom_folders]

    return phantom_agd_paths, phantom_fdk_paths


def load_phantom_ds(folder_path='PhantomsRadial/'):

    ds_path = Path('/data/fdelberghe/') /folder_path

    phatom_folders = sorted(ds_path.glob('*/'), key=_nat_sort)

    phantom_truth_paths = [sorted(folder.glob('CT_target_*.tif'), key=_nat_sort) for folder in phatom_folders]
    phantom_fdk_paths = [[sorted(folder.glob(f'CB_source_orbit{orbit_id:0>2d}_*.tif'), key=_nat_sort)
                          for orbit_id in [1,2,3]] for folder in phatom_folders]

    return phantom_truth_paths, phantom_fdk_paths