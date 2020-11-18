import glob
import os
import random
import re
import time
from contextlib import contextmanager

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import COLOR_GRAY2RGB, VideoWriter, VideoWriter_fourcc, cvtColor


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
        print(f'Starting {f_name}...', end=' ', flush=True)
        return_values = func(*args, **kwargs)
        print(f"Done! (took {time.time()-start_time:.2f}s)", flush=True)
        
        return return_values

    return timeit_wrapper


@contextmanager
def evaluate(model):
    """Context manager to evaluate models disables grad computation and sets model to eval"""

    try:
        torch.set_grad_enabled(False)
        model.eval()
        yield None

    finally:
        torch.set_grad_enabled(True)
        model.train()
    

_nat_sort = lambda s: [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", s)]


def load_walnut_ds():

    DATA_PATH = '/data/fdelberghe/FastWalnuts2/'
    walnut_folders = [folder for folder in sorted(os.listdir(DATA_PATH), key=_nat_sort) 
                      if os.path.isdir(os.path.join(DATA_PATH, folder))]

    walnuts_agd_paths = [sorted(glob.glob(os.path.join(DATA_PATH + f"{folder}/agd_*.tif")), key=_nat_sort) 
                         for folder in walnut_folders]
    walnuts_fdk_paths = [
        [sorted(glob.glob(os.path.join(DATA_PATH + f"{folder}/fdk_orbit{orbit_id:02d}*.tif")),
                key=_nat_sort) for orbit_id in [1, 2, 3]] for folder in walnut_folders]

    return walnuts_agd_paths, walnuts_fdk_paths


def load_foam_phantom_ds():

    DATA_PATH = '/data/fdelberghe/'
    phantom_folders = [folder for folder in sorted(os.listdir(os.path.join(DATA_PATH, 'FoamPhantoms/')), key=_nat_sort) 
                      if os.path.isdir(os.path.join(DATA_PATH, f'FoamPhantoms/{folder}'))]

    phantom_agd_paths = [
        sorted(glob.glob(os.path.join(DATA_PATH + f"FoamPhantoms/{folder}/phantom_true_*.tif")), key=_nat_sort)
        for folder in phantom_folders]

    phantom_fdk_paths = [
        [sorted(glob.glob(os.path.join(DATA_PATH + f"FoamPhantoms/{folder}/phantom_fdk_*_o{orbit_id}*.tif")), key=_nat_sort)
         for orbit_id in [1, 2, 3]] for folder in phantom_folders]

    return phantom_agd_paths, phantom_fdk_paths


def load_phantom_ds():

    DATA_PATH = '/data/fdelberghe/'
    phantom_folders = [folder for folder in sorted(os.listdir(os.path.join(DATA_PATH, 'Phantoms/')), key=_nat_sort) 
                      if os.path.isdir(os.path.join(DATA_PATH, f'Phantoms/{folder}'))]

    phantom_target_paths = [
        sorted(glob.glob(os.path.join(DATA_PATH + f"Phantoms/{folder}/target*.tif")), key=_nat_sort)
        for folder in phantom_folders]

    phantom_input_paths = [
        sorted(glob.glob(os.path.join(DATA_PATH + f"Phantoms/{folder}/input*.tif")), key=_nat_sort)
         for folder in phantom_folders]

    return phantom_target_paths, phantom_input_paths


class ValSampler(torch.utils.data.Sampler):
    """Samplers to avoid going through the entire validation dataset each time"""
        
    def __init__(self, dataset_len, n_samples=100):
        self.dataset_len = dataset_len
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        return iter(random.sample(range(self.dataset_len-1), self.n_samples))


class BatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, sub_sampling):
        raise NotImplementedError


class LossTracker():

    def __init__(self, **kwargs):
        """Function gets losses to track as {'loss_name': loss_init_value,}"""
        self.progress = [0]
        self.losses = dict(
            ((k, v) if isinstance(v, list) else (k, [v]) for k, v in kwargs.items())
            )

    def update(self, *losses, **named_losses):

        # Same order as __init__
        if len(losses) > 0:
            self.progress.append(self.progress[-1]+1)
            for key, loss in zip(self.losses, losses):
                self.losses[key].append(loss)
            # One method or the other
            return

        # Random order
        if len(named_losses) > 0:
            self.progress.append(self.progress[-1]+1)
            for name, loss in named_losses.item():
                self.losses[name].append(loss)
            return
        
        raise ValueError("No valid value to update losses")

    def plot(self, **kwargs):
        plt.figure(figsize=(10,10))

        for key, loss in self.losses.items():
            if len(self.progress) == len(loss):
                plt.plot(self.progress, loss, label=key)
            else:
                plt.plot(list(range(len(loss))), loss, label=key)

        plt.xlabel(kwargs.get('xlabel', 'Epoch'))
        plt.ylabel(kwargs.get('ylabel', 'Loss'))
        plt.title(kwargs.get('title', ''))
        plt.xticks(kwargs.get('xticks', 
                              np.linspace(0, len(self.progress)-1, min(16, len(self.progress))).astype('int16')))
        plt.legend()
        plt.savefig(kwargs.get('filename', 'outputs/training_losses.png'))


