# Cone Angle Artifact Reduction 

This repository contains code for experiments based on the paper "Efficient High Cone-Angle Artifact Reduction in Circular 
Cone-Beam CT using Symmetry-Aware Deep Learning with Dimension Reduction" by Minnema et al. 
Cone-Beam CT produces a lot of artefacts when reconstructed with simple backprojection derived techniques such as FDK reconstruction.
The goal of this project is to find a deep learning based method for efficiently removing high cone angle artefacts in CBCT images.
A large focus should be put on the transfer learning task to go from the walnut validated scans to real human heads or antropomorphic 
phantoms.

### Task 1:

Explore transfer learning task from synthetic foam CT phantom to the walnuts dataset (and back).


### Task 2:

Explore transfer learning from walnut trained models to simulated or real human head scans.


## Structure:

```
test: main file to run experiments from  

    |src: main modules containing all the usefull functions to be called from the root dir  

        |utils: Utility functions for for example IO operation, useful classes, and decorators  
        |astra_sim: Projection and reconstruction functions using ASTRA toolbox
        |build_training: Building training data for phantoms and walnuts  
        |train_model: Functions for training models  
```