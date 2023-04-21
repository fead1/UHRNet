import numpy as np

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
import matplotlib.pyplot as plt

fringepattern = torch.from_numpy(np.load('fringe_pattern.npy'))
gt = np.load('height_map.npy')
path = 'UHRNet_weight.pth'
checkpoint = torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Drop the background part as an invalid point mask
def trans(input, gt):
    for i in range(352):
        for j in range(640):
            if gt[i,j] <= -100:
                input[i,j] = gt[i,j]
    return input
############################

net = UHRNet().to(device)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
with torch.no_grad():
    fringepattern = fringepattern.to(device)
    out = net(fringepattern.unsqueeze(0).unsqueeze(0))
    out = out.detach().cpu().numpy()[0,0]
    out = trans(out, gt)

plt.subplot(121)
plt.imshow(out, cmap='jet')
plt.subplot(122)
plt.imshow(gt, cmap='jet')
plt.show()

