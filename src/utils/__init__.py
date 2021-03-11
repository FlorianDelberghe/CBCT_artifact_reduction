import glob
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import PurePath ,Path

import imageio
import numpy as np
import torch
from cv2 import COLOR_GRAY2RGB, VideoWriter, VideoWriter_fourcc, cvtColor


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

        # if im.min() >= 0:
        #     if im.max() <= 1:                
        #         # Resclaing [0,1] floats to uint8
        #         im = (im *255).astype('uint8')

        #     elif im.max() <= 255:
        #         # Matches type to avoid warnings
        #         im = im.astype('uint8')
            
        # elif im.min() >= -1 and im.max() <= 1:
        #     # Rescaling [-1,1] floats to uint8
        #     im = ((im +1) *127.5).astype('uint8')
        
        # return func(uri, im, **kwargs)

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


def timeit(func):
    """Decorator to monitor computing time"""

    f_name = ' '.join(func.__name__.split('_'))

    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        print(f'Starting {f_name}...', end='', flush=True)
        return_values = func(*args, **kwargs)
        print(f"\tDone! (took {time.time()-start_time:.2f}s)", flush=True)
        
        return return_values

    return timeit_wrapper


@contextmanager
def evaluate(*models):
    """Context manager to evaluate models disables grad computation and sets models to eval"""

    try:
        torch.set_grad_enabled(False)
        [model.eval() for model in models]
        yield None

    finally:
        torch.set_grad_enabled(True)
        [model.train() for model in models]
    

#TODO: unit test for _nat_sort
def _nat_sort(path):
    if isinstance(path, PurePath):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", path.as_posix())]
    
    return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", path)]


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


def set_seed(func):
    """Decorator to set random seed before function calls for deterministic behavior"""

    def _set_seed_wrapper(*args, seed=0, **kwargs):
        random.seed(seed)
        return func(*args, **kwargs)
    
    return _set_seed_wrapper

@set_seed
def split_data(input_ims, target_ims, frac=7/42, verbose=True):

    n_test = n_val = max(1, int(np.round(len(target_ims) *frac)))
    
    zipped_ims = random.sample([
        (i, input_im, target_im) for i, (input_im, target_im) in enumerate(zip(input_ims, target_ims))
    ], len(target_ims))

    ids_te, *test_set = tuple(zip(*zipped_ims[:n_test]))
    ids_val, *val_set = tuple(zip(*zipped_ims[n_test:n_test+n_val]))
    ids_tr, *train_set = tuple(zip(*zipped_ims[n_test+n_val:]))

    if verbose:
        print(f"Sample indices for test: {ids_te}, validation: {ids_val}, training: {ids_tr}")

    return test_set, val_set, train_set

@set_seed
def split_data_CV(input_ims, target_ims, frac=1/4, verbose=True):
    
    if isinstance(frac, (tuple, list)):
        n_test = int(np.round(len(target_ims) *frac[0]))
        n_val = max(1, int(np.round(len(target_ims) *frac[1])))

    else:
        n_test = n_val = max(1, int(np.round(len(target_ims) *frac)))

    zipped_ims = random.sample([
        (i, input_im, target_im) for i, (input_im, target_im) in enumerate(zip(input_ims, target_ims))
    ], len(target_ims))

    if n_test <= 0:
        ids_te, test_set = None, None
    else:
        ids_te, *test_set = tuple(zip(*zipped_ims[:n_test]))
        zipped_ims = zipped_ims[n_test:]

    for i in range(len(zipped_ims)):
        val_idx = set([(j+i) %len(zipped_ims) for j in range(n_val)])
        tr_idx = set(range(len(zipped_ims))) -val_idx

        ids_val, *val_set = tuple(zip(*[zipped_ims[j] for j in val_idx]))
        ids_tr, *train_set = tuple(zip(*[zipped_ims[j] for j in tr_idx]))

        if verbose:
            print(f"Sample indices for test: {ids_te}, validation: {ids_val}, training: {ids_tr}")

        yield test_set, val_set, train_set


class ValSampler(torch.utils.data.Sampler):
    """Samplers to avoid going through the entire validation dataset each time"""
        
    def __init__(self, dataset_len, n_samples=100, fixed_samples=True):
        self.dataset_len = dataset_len
        self.n_samples = int(dataset_len *n_samples) if 0 < n_samples <= 1 else n_samples
        self.fixed_samples = fixed_samples
        
        # Returns the same samples for every iter
        if fixed_samples:
            self.samples = random.sample(range(self.dataset_len), self.n_samples)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        if self.fixed_samples:
            return iter(self.samples)
            
        return iter(random.sample(range(self.dataset_len), self.n_samples))


class BatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, sub_sampling):
        raise NotImplementedError



