# Cone Angle Artifact Reduction using symmetry-aware deep learning

This repository contains code that was used to perform the experiments described in the paper "Efficient High Cone-Angle Artifact Reduction in Circular Cone-Beam CT using Symmetry-Aware Deep Learning with Dimension Reduction" by Minnema et al.

## Getting started
We recommend installing Conda or Miniconda for Python3 to set up the required packages to run this code. A virtual environment in (Mini)conda can be created and activated with:

```
env_name=*my_name*
conda create --name $env_name python=3.6
conda activate $env_name
```

To get the source code and install the required pacakges

```
git clone https://github.com/Jomigi/Cone_angle_artifact_reduction.git
cd Cone_angle_artifact_reduction
python setup.py
```

## Walnut Data set
Download the [Walnut data set](https://zenodo.org/record/2686726#.Xz0faFozaV4) from zenodo.


## Code description
The code in this repository is split into three parts: CTreconstruction, CNN, and Radial2Cartesian.

### CT reconstuction
The script `generate_training_data.py` contains all code that is necessary to reproduce the reconstruction of cone-beam CT scans as performed in the paper. This includes the computation of FDK reconstruction and iterative ground truth reconstruction and the extraction of radial slices from these volumes via interpolation. Note that as described in the paper, this step is repeated 24 times with rotated scan geometries and only the radial slices close to 0° and 90° are extracted in repetition. The script takes up to one day to run for a single Walnut. To run `generate_training_data.py`, customize the variables in the section "`user defined settings`" and run:

```
python generate_training_data
```

### CNN
This includes all code necessary to train an MS-D Net or a U-Net to reduce cone-angle artifacts in cone-beam CT scans. To train the networks, specify the datapaths in train.py and then run:

```
python train.py
```

This folder also contains the validation scheme (validation.sh) that was used to optimize the depth and dilations of MS-D Net as well as the number of epochs to train both CNNs.

### Radial2Cartesian
This folder contain a single script which was used to performed to Radial-to-Cartesian re-sampling step.

To run this code:
```
python radial2axial.py
```

## License
The code is licensed under the MIT license - see the LICENSE.md file for more details.
