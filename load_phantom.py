import re
from pathlib import Path
from struct import unpack_from

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

    # 16 bits little endian decoding
    blocksize = 2

    with file.open(mode='rb') as f:

        for _ in trange(stack_shape[2], position=0):
            slc = np.empty((stack_shape[1]*stack_shape[0],), dtype='uint16')

            for i in range(len(slc)):
                # decoding parameters: `<` for little endian, `H` for unsigned short (2 bytes)     
                slc[i] = unpack_from("<H", f.read(blocksize))[0]

            yield slc


DATA_PATH = Path('/data/fdelberghe/')
phantoms = ['Phantom1_rar', 'Phantom3', 'Phantom6', 'Phantom7']

for phantom in phantoms[1:]:
    vol_file = DATA_PATH / 'Phantoms' / f'{phantom}.vol'
    hx_file = DATA_PATH / 'Phantoms' / f'{phantom}.hx'

    stack_shape, stack_size = get_stack_size(hx_file)

    save_path = DATA_PATH / 'AxialPhantoms' / phantom
    save_path.mkdir(parents=True, exist_ok=True)

    for i, slc in enumerate(get_slices(vol_file, stack_shape)):
        imsave(save_path / f'sliceZ_{i:0>4d}.tif', slc.reshape((stack_shape[1], stack_shape[0])))



