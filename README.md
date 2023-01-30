# KudaNN: A parallelized implementation of the KNN algorithm using the CUDA framework

This work was done in the context of my end-of-semester HPC mini-project. The goal is to optimize the KNN algorithm -which is a greedy algorithm to run sequentially as it requires at minimum n_train*n_predict iterations- using the CUDA framework.  

## Prerequisits:
A CUDA capable GPU  
The CUDA programming suite (learn more about setting up a CUDA development environment [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/))

## How to use
Put the source and header files in your project directory.  
Include the "kudann.cuh" header file.  
Call the the kudaNN kernel function on your data sets using the desired configuration (make sure to have as many threads as there are training/prediction sample pairs, so n_train*n_predict threads).
