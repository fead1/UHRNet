# UHRNet：A Deep Learning-Based Method for Accurate 3D Reconstruction from a Single Fringe-Pattern
UHRNet CNN implementation (pytorch)
## Introduction
In this paper, we propose a deep learning-based method for accurate 3D reconstruction for a single fringe-pattern. We use unet's encoding and decoding structure as baackbone and design Multi-level Conv block and Fusion block to enhance the ability of feature extraction and detail reconstruction of the network. Wang et al. 's dataset was used as our training set validation set and test set. The link to the data set is left at the end. The test set contains 153 patterns, and our method's average RMSE is only 0.443(mm) and an average SSIM is 0.9978 on the test set.

For more details, please refer to our paper:https://arxiv.org/abs/2304.14503

**Frame of UHRNet**
 
- UHRNet structure
![UHRNet](https://raw.githubusercontent.com/fead1/UHRNet/main/Network%20structure/UHRNet%20structure.png)
- Muti-Level Conv Block structure
![Muti-Level Conv Block](https://raw.githubusercontent.com/fead1/UHRNet/main/Network%20structure/Muti-Level%20Conv%20Block.png)
- High-resolution Fusion Block structrure
![High-resolution Fusion Block](https://raw.githubusercontent.com/fead1/UHRNet/main/Network%20structure/High-resolution%20Fusion%20Block.png)

## Main Results
-   **Prediction evalution of  three networks on test set**

|Model|RMSE(mm)|SSIM|Param(M)|Speed(s)|
|---|---|---|---|---|
|our method|0.433|0.9978|30.33|0.0224|
|hNet|1.330|0.9767|8.63|0.0093|
|ResUNet|0.685|0.9931|32.44|0.0105|

-   **3D height map reconstructed by our method**

1. single object in the field of view

![demo](https://raw.githubusercontent.com/fead1/UHRNet/main/3D%20height%20map%20reconstructed%20by%20our%20method/single%20object%20in%20the%20field%20of%20view.png)

2. two ioslated object in the field of view

![demo](https://raw.githubusercontent.com/fead1/UHRNet/main/3D%20height%20map%20reconstructed%20by%20our%20method/two%20ioslated%20object%20in%20the%20field%20of%20view.png)

3. two overlapping object in the field of view

![demo](https://raw.githubusercontent.com/fead1/UHRNet/main/3D%20height%20map%20reconstructed%20by%20our%20method/two%20overlapping%20object%20in%20the%20field%20of%20view.png)

4. three overlapping object in the field of view

![demo](https://raw.githubusercontent.com/fead1/UHRNet/main/3D%20height%20map%20reconstructed%20by%20our%20method/three%20overlapping%20object%20in%20the%20field%20of%20view.png)

## Our Environment

- Python 3.9.7
- pytorch 1.5.0
- CUDA 11.3
- Numpy 1.23.3
## Pretrained model and Dataset
- Pretrained model(UHRNet):
Link：https://pan.baidu.com/s/1QS5ftR2Ww2n6enVeVlf-yQ 
Password：1234
According the link given above to download the weights to the UHRNet folder to run the pre-trained model
- Dataset:
[Single-input dual-output 3D shape reconstruction (figshare.com)](https://figshare.com/s/c09f17ba357d040331e4)[1]
This dataset contains 1532 fringe-patterns and corresponding 3D height map, which are divided into training set, test set and validation set according to the ratio of 80%, 10% and 10%

## Citation
 [1] A. Nguyen, O. Rees and Z. Wang, "Learning-based 3D imaging from single structured-light image,"  _Graphical Models,_ vol. 126, 2023.




