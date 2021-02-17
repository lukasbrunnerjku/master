import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import sys

sys.path.append('./')
from utils import count_params, autopad 
from convolutions import (DeformConv, SpatiallyConv, 
    DepthwiseConv, FlattenedConv, GroupedConv, ShuffledGroupedConv)

# ------ globals ------

ConvBase = nn.Conv2d

# --- basic blocks ----

class Conv(nn.Module):
    def __init__(self, chi, cho, k=1, s=1, p=None, g=1, act=nn.ReLU(), affine=False):  
        super().__init__()
        # define class ConvBase to be one of the different 
        self.conv = ConvBase(chi, cho, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cho, affine=affine)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpSample(nn.Module):
    def __init__(self, chi, cho, k=1, s=3, p=None, g=1, sf=2):
        super().__init__() 
        # learned upsampling, avoids checkerboard artifacts
        self.sf = sf
        self.conv = Conv(chi, cho, k, s, p, g=g)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.sf, mode='nearest'))

class Bottleneck(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, g=1, e=0.5):
        super().__init__()
        chh = int(cho * e)
        self.conv1 = Conv(chi, chh, 1, 1, 0, g=g)
        self.conv2 = Conv(chh, chh, k, s, p, g=g)
        self.conv3 = Conv(chh, cho, 1, 1, 0, g=g)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return residual + self.conv3(self.conv2(self.conv1(x)))
    
class BasicBlock(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, g=1):
        super().__init__()
        self.conv1 = Conv(chi, cho, k, s, p, g=g)
        self.conv2 = Conv(cho, cho, k, 1, p, g=g)
    
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return residual + self.conv2(self.conv1(x)) 
    
