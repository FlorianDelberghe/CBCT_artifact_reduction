import argparse
import glob

import numpy as np
from imageio import imread

import src.utils as utils
from src import astra_sim
from src.utils import imsave, load_projections, mimsave


def test_astra_sim():
    """Various tests for import, reconstruction and synthetic simulation of CBCT process"""

    scanner_params = astra_sim.FleX_ray_scanner()

    # ===== Import projections data and reconstruct volume w/ FDK ===== #
    projections, dark_field, flat_field, vecs = load_projections(DATA_PATH+f'Walnuts/Walnut{WALNUT_ID}', orbit_id=ORBIT_ID)
    projections = (projections - dark_field) / (flat_field - dark_field)
    projections = -np.log(projections)
    projections = np.ascontiguousarray(projections)

    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, 
                                                  proj_vecs=vecs, gpu_id=GPU_ID, 
                                                  rec_shape=(501, 501, 501))
    reconstruction = (reconstruction -reconstruction.min()) / (reconstruction.max() -reconstruction.min())
    utils.save_vid(f'outputs/reconst_fdk_pos{ORBIT_ID}.avi', np.swapaxes(reconstruction[...,None], 0,1))

    radial_slices = astra_sim.radial_slice_sampling(reconstruction, np.linspace(0, np.pi, 360, endpoint=False))
    utils.save_vid('outputs/reconst_fdk_radial.avi', radial_slices[...,None])

    axial_slices = astra_sim.radial2axial(radial_slices)
    utils.save_vid('outputs/reconst_fdk_radial2axial.avi', axial_slices[...,None])

    # ===== Import ground truth volume data and simulate projection and reconstruction ===== #
    fdk_files = sorted(glob.glob(DATA_PATH+f'Walnuts/Walnut{WALNUT_ID}/Reconstructions/fdk_pos{ORBIT_ID}_*.tiff'))
    agd_files = sorted(glob.glob(DATA_PATH+f'Walnuts/Walnut{WALNUT_ID}/Reconstructions/full_AGD_50*.tiff'))
    fdk_volume = np.stack([imread(file) for file in fdk_files], axis=0)
    agd_volume = np.stack([imread(file) for file in agd_files], axis=0)

    utils.save_vid(f'outputs/true_fdk_pos{ORBIT_ID}.avi', 
                   (np.swapaxes(fdk_volume[..., None], 0,1) - fdk_volume.min()) / (fdk_volume.max() - fdk_volume.min()))
    utils.save_vid(f'outputs/true_fdk_pos{ORBIT_ID}_radial.avi', 
                   astra_sim.radial_slice_sampling(fdk_volume, np.linspace(0, np.pi, 360, endpoint=False))[..., None] / fdk_volume.max())
    utils.save_vid('outputs/true_agd.avi',
                   (np.swapaxes(agd_volume[..., None], 0,1) - agd_volume.min()) / (agd_volume.max() - agd_volume.min()))

    vecs = astra_sim.create_scan_geometry(scanner_params, 1200, elevation=0)
    projections = astra_sim.create_CB_projection(agd_volume, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID)

    utils.save_vid('outputs/agd_proj.avi', projections[...,None] /projections.max())
    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID, rec_shape=(501,501,501))
    reconstruction = (reconstruction -reconstruction.min()) / (reconstruction.max() -reconstruction.min())

    fdk_radial = astra_sim.radial_slice_sampling(fdk_volume, np.linspace(0, np.pi, 360, endpoint=False))
    reconstruction_radial = astra_sim.radial_slice_sampling(reconstruction, np.linspace(0, np.pi, 360, endpoint=False))

    utils.save_vid(f'outputs/reconst_fdk_pos{ORBIT_ID}_sim.avi', (np.swapaxes(reconstruction[...,None], 0,1)))

    # ===== Creates phatom and simulate projection and reconstruction ===== #
    phantom = astra_sim.create_ct_foam_phantom()

    projections = astra_sim.create_CB_projection(phantom, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID)
    utils.save_vid('outputs/phantom_proj.avi', projections[...,None] /projections.max())
    
    reconstruction = astra_sim.FDK_reconstruction(projections, scanner_params, proj_vecs=vecs, gpu_id=GPU_ID, rec_shape=(501,501,501))
    reconstruction = (reconstruction -reconstruction.min()) / (reconstruction.max() -reconstruction.min())
    utils.save_vid('outputs/phantom_reconst_fdk.avi',(np.swapaxes(reconstruction[...,None], 0,1)))
    utils.save_vid('outputs/phantom_reconst_fdk_radial.avi', 
                   astra_sim.radial_slice_sampling(reconstruction, np.linspace(0, np.pi, 360, endpoint=False))[...,None])


def main():    
    test_astra_sim()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, nargs='?', default=0,
                        help='GPU to run astra sims on')
    parser.add_argument('--walnut_id', type=int, nargs='?', default=9,
                        help='GPU to run astra sims on')
    parser.add_argument('--orbit_id', type=int, nargs='?', default=2,
                        choices=(1,2,3), help='GPU to run astra sims on')
    args = parser.parse_args()

    GPU_ID = args.gpu
    WALNUT_ID = args.walnut_id
    ORBIT_ID = args.orbit_id
    DATA_PATH = '/data/fdelberghe/'

    main()
