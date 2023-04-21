# UHRNet
UHRNet：A Deep Learning-Based Method for Accurate 3D Reconstruction from a Single Fringe-Pattern
###########################
In this paper, we propose a deep learning-based method for accurate 3D reconstruction for a single fringe-pattern. We use unet's encoding and decoding structure as baackbone and design Multi-level Conv block and Fusion block to enhance the ability of feature extraction and detail reconstruction of the network. Wang et al. 's   dataset was used as our training set validation set and test set. The link to the data set is left at the end. The datasets' ground-truth is determined through PSP measurements on plaster sculptures of varying sizes and shapes. To increase the number of samples in the datasets, the sculptures were randomly moved and rotated multiple times. The experiment uses a desktop computer with an Intel Core i7-9700k processor, a 32-GB RAM, and a Nvidia GeForce GTX 2080Ti. The code for training is written in Pytorch and utilizes the Adaptive Moment Estimation (Adam) optimizer. 

UHRNet is implemented in [PyTorch](https://pytorch.org/), please install PyTorch first following the official instruction.
- Python 3.9
- PyTorch
- Torchvision
- Pillow 
- numpy
- CUDA
- sklearn

##Dataset
https://figshare.com/s/c09f17ba357d040331e4
We  divided the dataset into training set, validation set and test set according to the ratio of 80%, 10% and 10%
