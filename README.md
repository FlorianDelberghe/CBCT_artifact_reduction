# Cone Angle Artifact Reduction 

## Background  

This repository contains code for experiments based on the paper "Efficient High Cone-Angle Artifact Reduction in 
Circular Cone-Beam CT using Symmetry-Aware Deep Learning with Dimension Reduction" by Minnema et al. 
Cone-Beam CT produces a lot of artefacts when reconstructed with simple backprojection derived techniques such as FDK 
reconstruction.


The goal of this project is to use the knowledge learned from the artifact reduction task on the Walnuts dataset and explore 
transfer learning from this image domain to more antropomorphic looking CBCT scans +(phantoms and real head scans).

### Task 1:

Explore transfer learning task from Walnuts dataset to antropomorphic phatoms.

    * How much new knowledge do we need  
        -> No conclusive results (4 is too few volumes)  

    * What are the differences compared to learning from scratch  
        -> Faster rate of convergence but similar performances  
        -> Slight differences in robustness

    * Does information about the source domain remains after tranfer  
        -> Some does (need to quantify)



### Task 2:

Validate the simulated CBCT projection method and build synthetic CBCT images from real human head CT for the second transfer 
learning task.

14.12.2020:  
CBCT simulation from true CT looks qualitatively (and quantitatively?) realistic for the purpose of producing synthetic 
CBCT scans for our experiments (tested on Walnut dataset). -> Acquiring high-res CT volume could provide good data for 
the transfer learning experiments.  

21.12.2020: **(ongoing)**  
Using the [LDCT dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026), we can acquire a 
large amount of human head & neck scans that can be used to upgrade our limited dataset. Axial resolution is quite 
limiting however (5mm), to solve that, we try to reconstruct high resolution images from the included projection data 
to get ~1mm axial resolution.  


## Structure:
```
train: Model training  
test: Test models and produce figures  
transfer: Transfer learning experiments  
astra_demo: Demo of basic astra operation, reconstruction, projection and simulated reconstruction on volumes  
load_phantoms: Loads .vol binary files into numpy array for saving of radial slices  

    |src: main modules containing all the usefull functions to be called from the root dir  

        |utils: Utility functions for IO operation, useful classes, and decorators  
        |astra_sim: Projection and reconstruction functions using ASTRA toolbox  
        |image_dataset: Data structures for import and training  
        |build_training: Building training data for phantoms and walnuts  
        |train_model: Functions for training models  
        |test_model: Functions to test model perfs and produce figures  
        |transfer_model: Code for the transfer experiments  
        |models: MSD based UNet model and other experimental CNNs
        |nesterov_gradient: Useful class and methods for astra iterative reconstruction  
    
    |model_weights: Weights for pretrained MSDNet  

```