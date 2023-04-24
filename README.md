# UHRNet：A Deep Learning-Based Method for Accurate 3D Reconstruction from a Single Fringe-Pattern
UHRNet CNN implementation (pytorch)
## Introduction
In this paper, we propose a deep learning-based method for accurate 3D reconstruction for a single fringe-pattern. We use unet's encoding and decoding structure as baackbone and design Multi-level Conv block and Fusion block to enhance the ability of feature extraction and detail reconstruction of the network. Wang et al. 's dataset was used as our training set validation set and test set. The link to the data set is left at the end. The test set contains 153 patterns, and our method's average RMSE is only 0.443(mm) and an average SSIM is 0.9978 on the test set.

For more details, please refer to our paper:

**Frame of UHRNet**
 
- UHRNet structure
[UHRNet]()
- Muti-Level Conv Block structure
[Muti-Level Conv Block]()
- Fusion Block structrure
[Fusion Block]()

## Main Results
-   **Prediction evalution of  three networks on test set**

|Model|RMSE(mm)|SSIM|Param(M)|Speed(s)|
|---|---|---|---|---|
|our method|0.433|0.9978|30.33|0.0224|
|hNet|1.330|0.9767|8.63|0.0093|
|RESUNet||||

-   **3D height map reconstructed by our method**

1. single object in the field of view

[demo]()

2. two ioslated object in the field of view

[demo]()

3. two overlapping objectin the field of view

[demo]()

4. three overlapping objectin the field of view

[demo]()


