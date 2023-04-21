import argparse
import logging
import os
import random
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,TensorDataset
import torch
import torch.nn as nn
from UHRNet import UHRNet
from sklearn.model_selection import train_test_split
from SSIM import SSIM
import matplotlib.pyplot as plt


#####################Dataset download address：https://figshare.com/s/c09f17ba357d040331e4

fringepath = 'the path of f1_80.npy'
gt = 'the path of Z.npy'
pic = torch.from_numpy(np.load(fringepath).swapaxes(2,3).swapaxes(1,2))
true = torch.from_numpy(np.load(gt).swapaxes(2,3).swapaxes(1,2))
x_train, x_split, y_train, y_split = train_test_split(pic, true, test_size=0.2, random_state=0)
val, test, valgt, testgt = train_test_split(x_split, y_split, test_size=0.5, random_state=1)

path = 'D:\pythonProject\\UHRNet\\UHRNet_weight.pth'
checkpoint = torch.load(path)
loss1 = nn.MSELoss()
loss2 = SSIM()
loss1_ = 0
loss2_ = 0

########################################
# Drop the background part as an invalid point
def trans(input, gt):
    a = 0
    for i in range(352):
        for j in range(640):
            if gt[0,0,i,j] <= -100:
                input[0,0,i,j] = gt[0,0,i,j]
                a +=1
    rat = 352*640/(352*640-a)
    return input,rat
############################

len = test.shape[0]
test0 = TensorDataset(test, testgt)
testdata = DataLoader(test0, batch_size=1, shuffle=False)
net = UHRNet().cuda()
net.load_state_dict(checkpoint['state_dict'])
net.eval()

with torch.no_grad():
    i = 0
    for data in testdata:
        inp, gt = data
        inp, gt = inp.cuda(), gt.cuda()
        out = net(inp)
        out, rat = trans(out, gt)
#The range of data set should be changed consistent with the true height of the object firstly,
# and n in MSEloss was changed from the number of all image pixels to the number of effective partial pixels
        loss10 = torch.sqrt(loss1(out/6, gt/6)*rat)
        loss20= loss2(out+105, gt+105)
        loss1_ += loss10.item()
        loss2_ +=loss20.item()
        print(i)
        i = i+1
        print(loss10)
        print(loss20)

    print(loss1_/len)
    print(loss2_/len)