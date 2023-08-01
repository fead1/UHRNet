import torch.nn as nn
from torch import cat as cat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torchvision.datasets
import math
from torchsummary import summary



' Branch1 block '
class Branch1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt1 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bt1(x1)
        return x1

' Branch2 block '
class Branch2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch2, self).__init__()
        self.conv2_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_1 = nn.BatchNorm2d(out_ch)
        self.rl2_1 = nn.LeakyReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.bt2_2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):


        x2 = self.conv2_1(x)
        x2 = self.bt2_1(x2)
        x2 = self.rl2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bt2_2(x2)
        return x2

' Branch3 block '
class Branch3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch3, self).__init__()
        self.conv3_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_1 = nn.BatchNorm2d(out_ch)
        self.rl3_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.bt3_2 = nn.BatchNorm2d(out_ch)
        self.rl3_2 = nn.LeakyReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.bt3_3 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x3 = self.conv3_1(x)
        x3 = self.bt3_1(x3)
        x3 = self.rl3_1(x3)
        x3 = self.conv3_2(x3)
        x3 = self.bt3_2(x3)
        x3 = self.rl3_2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.bt3_3(x3)
        return x3

' Branch4 block '
class Branch4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch4, self).__init__()
        self.conv4_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_1 = nn.BatchNorm2d(out_ch)
        self.rl4_1 = nn.LeakyReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.bt4_2 = nn.BatchNorm2d(out_ch)
        self.rl4_2 = nn.LeakyReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2 , dilation=2)
        self.bt4_3 = nn.BatchNorm2d(out_ch)
        self.rl4_3 = nn.LeakyReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=5, dilation=5)
        self.bt4_4 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x4 = self.conv4_1(x)
        x4 = self.bt4_1(x4)
        x4 = self.rl4_1(x4)
        x4 = self.conv4_2(x4)
        x4 = self.bt4_2(x4)
        x4 = self.rl4_2(x4)
        x4 = self.conv4_3(x4)
        x4 = self.bt4_3(x4)
        x4 = self.rl4_3(x4)
        x4 = self.conv4_4(x4)
        x4 = self.bt4_4(x4)
        return x4

' Residual block with Inception module '
class ResB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResB, self).__init__()
        self.branch0 = Branch0(in_ch, out_ch)
        self.branch1 = Branch1(in_ch, out_ch // 4)
        self.branch2 = Branch2(in_ch, out_ch // 4)
        self.branch3 = Branch3(in_ch, out_ch // 4)
        self.branch4 = Branch4(in_ch, out_ch // 4)
        self.rl = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = cat((x1, x2, x3, x4), dim=1)
        x6 =  x0 + x5
        x7= self.rl(x6)
        return  x7

' Downsampling block '
class DownB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownB, self).__init__()
        self.res = ResB(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x1 = self.res(x)
        x2 = self.pool(x1)
        return x2, x1

' Upsampling block '
class UpB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpB, self).__init__()
        self.up = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding = 1, output_padding = 1)
        self.res = ResB(out_ch*2, out_ch)
    def forward(self, x, x_):
        x1 = self.up(x)
        x2 = cat((x1 , x_), dim=1)
        x3 = self.res(x2)
        return x3

' Output layer '
class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
    def forward(self, x):
        x1 = self.conv(x)
        return x1

class cbl(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(cbl, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt = nn.BatchNorm2d(out_ch)
        self.lu = nn.LeakyReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bt(x)
        return self.lu(x)
' Architecture of Res-UNet '
class UHRNet(nn.Module):
    def __init__(self):
        super(UHRNet, self).__init__()
        self.down1 = DownB(1, 64)
        self.down2 = DownB(64, 128)
        self.down3 = DownB(128, 256)
        self.down4 = DownB(256, 512)
        self.res = ResB(512, 1024)
        # self.chan = ChannelAttention(1024)
        self.up1 = UpB(1024, 512)
        self.up2 = UpB(512, 256)
        self.up3 = UpB(256, 128)
        self.up4 = UpB(128, 64)
        self.outc = Outconv(64, 1)
        self.drop = nn.Dropout(0.3)
        self.up1_ = nn.ConvTranspose2d(in_channels=128, out_channels=32,kernel_size=3, stride=2, padding=1, output_padding=1)        # self.branch1 = third_branch(8, 512)
        self.up2_ = nn.ConvTranspose2d(in_channels=256, out_channels=32,kernel_size=3, stride=4, output_padding=1)
        self.up3_ = nn.ConvTranspose2d(in_channels=512, out_channels=32,kernel_size=3, stride=8, output_padding=5)
        self.cov1 = cbl(96, 64)
        self.cov1_ = cbl(128,64)

        self.down0_ = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.up2__ = nn.ConvTranspose2d(in_channels=256, out_channels=64,kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3__ = nn.ConvTranspose2d(in_channels=512, out_channels=64,kernel_size=3, stride=4, output_padding=1)

        self.cov2 = cbl(192, 128)
        self.cov2_ = cbl(256, 128)

        self.down0__ = nn.Conv2d(in_channels=64, out_channels=128, stride=4, kernel_size=4)
        self.down1_ = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.up3___ = nn.ConvTranspose2d(in_channels=512, out_channels=128,kernel_size=3, stride=2, padding=1, output_padding=1)

        self.cov3 = cbl(384, 256)
        self.cov3_ = cbl(512, 256)
       
    def forward(self, x):
        x1, x1_ = self.down1(x)

        x2, x2_ = self.down2(x1)
        x3, x3_ = self.down3(x2)
        x4, x4_ = self.down4(x3)
        x4 = self.drop(x4)
        x5  = self.res(x4)


        x6 = self.up1(x5, x4_)
        x3__ = cat((self.down0__(x1_),self.down1_(x2_),self.up3___(x4_)),dim=1)
        x3__ = self.cov3(x3__)
        x3__ = cat((x3_, x3__), dim=1)
        x3__ = self.cov3_(x3__)
        x7 = self.up2(x6, x3__)
        x2__ = cat((self.down0_(x1_),self.up2__(x3_),self.up3__(x4_)), dim=1)
        x2__ = self.cov2(x2__)
        x2__ = cat((x2_, x2__), dim=1)
        x2__ = self.cov2_(x2__)
        x8 = self.up3(x7, x2__)

        x1__ = cat((self.up3_(x4_), self.up2_(x3_), self.up1_(x2_)),dim=1)
        x1__ = self.cov1(x1__)
        x1__ = cat((x1_, x1__),dim=1)
        x1__ = self.cov1_(x1__)
        x9 = self.up4(x8, x1__)

        x10 = self.outc(x9)
      
        return x10

model = UHRNet()
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
