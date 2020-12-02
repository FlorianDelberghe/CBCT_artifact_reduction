import re
from pathlib import Path

import numpy as np
from imageio import imsave
from tqdm import trange


def get_stack_size(file):

    with hx_file.open(mode='rt') as f:
        # Discards the first lines of the file
        for _ in zip(f, range(5)): pass

        line = next(f)

        # (width, height, n_slices)
        stack_shape = list(map(int, re.findall(r'\s\d+\s', line)))
        stack_size = list(map(float, re.findall(r'\s\d+\.\d+\s', line)))

        return stack_shape, stack_size


def get_slices(file, stack_shape):

    blocksize = 2
    slc_len = stack_shape[1]*stack_shape[0]
    
    for i in trange(stack_shape[2], position=0): 
        # decoding parameters: `<` for little endian, `H` for unsigned short (2 bytes)
        slc = np.fromfile(file, dtype=np.dtype('<H'), count=slc_len, offset=blocksize*i*slc_len)

        yield slc.reshape(stack_shape[:2][::-1])


DATA_PATH = Path('/data/fdelberghe/')
IMPORT_PATH = Path('/data/maureen_shares/')

phantoms = [('Phantom1', 'Phantom1_rar'), ('Phantom3', '226'), ('Phantom6', 'VUMC2.0_rar'), ('Phantom7', '261')]

for phantom, file_name in phantoms:
    vol_file = IMPORT_PATH / f'{file_name}.vol'
    hx_file = IMPORT_PATH / f'{file_name}.hx'

    stack_shape, stack_size = get_stack_size(hx_file)

    save_path = DATA_PATH / 'AxialPhantoms' / phantom
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {phantom}...")
    for i, slc in enumerate(get_slices(vol_file, stack_shape)):
        imsave(save_path / f'sliceZ_{i:0>4d}.tif', slc)


