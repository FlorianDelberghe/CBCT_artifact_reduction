# Cone Angle Artifact Reduction 

## Background  

This repository contains code for experiments based on the paper "Efficient High Cone-Angle Artifact Reduction in Circular 
Cone-Beam CT using Symmetry-Aware Deep Learning with Dimension Reduction" by Minnema et al. 
Cone-Beam CT produces a lot of artefacts when reconstructed with simple backprojection derived techniques such as FDK reconstruction.


The goal of this project is to use the knowledge learned from the artifact reduction task on the Walnuts dataset and explore 
transfer learning from this image domain to more antropomorphic looking CBCT scans +(phantoms and real head scans).

### Task 1:

Explore transfer learning task from Walnuts dataset to antropomorphic phatoms.

    * How much new knowledge do we need  
    * What are the differences compared to learning from scratch  
    * Does information about the source domain remains after tranfer  


### Task 2:

Validate the simulated CBCT projection method and build synthetic CBCT images from real human head CT for the second transfer 
learning task.


## Structure:
**TODO: Structure has changed**
```
test: main file to run experiments from  

    |src: main modules containing all the usefull functions to be called from the root dir  

        |utils: Utility functions for for example IO operation, useful classes, and decorators  
        |astra_sim: Projection and reconstruction functions using ASTRA toolbox
        |build_training: Building training data for phantoms and walnuts  
        |train_model: Functions for training models  
```