import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import ops
from torch.cuda.amp import autocast
import sys

try:
    sys.path.append('./')
    from utils import (count_params, autopad, fuse_conv_and_bn, 
        time_synchronized)
except ImportError:
    from .utils import (count_params, autopad, fuse_conv_and_bn, 
        time_synchronized)

BASE = nn.Conv2d
GROUPS = 1

class DeformConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        k = k if isinstance(k, tuple) else (k, k) 
        p = autopad(k, p)
        self.conv = ops.DeformConv2d(chi, cho, k, s, p, 
            dilation=dilation, groups=groups, bias=bias)
        # for each group we need output channels of 2 to get for each kernel weight
        # offset position in x, y (2 channels) and we have to know that for every 
        # pixel in the convolution output, thus we use same kernel size and padding!!
        self.offset = nn.Conv2d(chi, groups * 2 * k[0] * k[1], k, s, p, 
            dilation=dilation, groups=groups, bias=True)
        self._init_offset()

    def _init_offset(self):
        # as the original paper suggests initialize offsets with zeros
        # thus we start with a standard convolution
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)

    @autocast(enabled=False)
    def forward(self, x):
        return self.conv(x, self.offset(x))
    
    def fuse(self, bn):
        self.conv = fuse_conv_and_bn(self.conv, bn)
        return self

class SpatiallyConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        # decreases complexity, but kernel space limited
        k = k if isinstance(k, tuple) else (k, k)
        p = p if isinstance(p, tuple) else (p, p)
        s = s if isinstance(s, tuple) else (s, s)
        p = autopad(k, p)
        self.conv1 = nn.Conv2d(chi, chi, (k[0], 1), (s[0], 1), (p[0], 0), 
            dilation=dilation, groups=groups, bias=True)
        self.conv2 = nn.Conv2d(chi, cho, (1, k[1]), (1, s[1]), (0, p[1]), 
            dilation=dilation, groups=groups, bias=bias)  
        
    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    def fuse(self, bn):
        self.conv2 = fuse_conv_and_bn(self.conv2, bn)
        return self

class DepthwiseConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        p = autopad(k, p)
        # decreases complexity, smaller networks can be wider,
        # each filter soley has access to a single input channel
        # and we keep the number of input channels at first
        self.conv1 = nn.Conv2d(chi, chi, k, s, p, 
            dilation=dilation, groups=chi, bias=True)
        # learn channelwise(inter group) correlation with 1x1 convolutions
        self.conv2 = nn.Conv2d(chi, cho, 1, 1, 0, 
            dilation=dilation, groups=1, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    def fuse(self, bn):
        self.conv2 = fuse_conv_and_bn(self.conv2, bn)
        return self

class FlattenedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=1, bias=True):
        super().__init__() 
        # paper claims importance of bias term!
        k = k if isinstance(k, tuple) else (k, k)
        p = p if isinstance(p, tuple) else (p, p)
        s = s if isinstance(s, tuple) else (s, s)
        p = autopad(k, p)
        # lateral, kernel: C x 1 x 1
        self.conv1 = nn.Conv2d(chi, cho, 1, 1, 0, groups=1, bias=True)
        # vertical, kernel: 1 x Y x 1
        self.conv2 = nn.Conv2d(cho, cho, (k[0], 1), (s[0], 1), (p[0], 0), 
            dilation=dilation, groups=cho, bias=True)
        # horizontal, kernel: 1 x 1 x X,
        # last term can omit bias e.g. if batchnorm is done anyway afterwards 
        self.conv3 = nn.Conv2d(cho, cho, (1, k[1]), (1, s[1]), (0, p[1]), 
            dilation=dilation, groups=cho, bias=bias)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))
    
    def fuse(self, bn):
        self.conv3 = fuse_conv_and_bn(self.conv3, bn)
        return self

class GroupedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=8, bias=True):
        super().__init__()  # typically groups are 2, 4, 8, 16
        p = autopad(k, p)
        # decreases complexity, the idea of grouped convolutions is that 
        # the correlation between feature channels is sparse anyway,
        # and here we will be even more sparse since we only allow 
        # intra channel group correlation, 
        # use grouped convolution also for 1x1 convolutions (see ShuffleNet)
        # which are then called pointwise grouped convolutions
        self.conv = nn.Conv2d(chi, cho, k, s, p, 
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)
    
    def fuse(self, bn):
        self.conv = fuse_conv_and_bn(self.conv, bn)
        return self

class ShuffledGroupedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, dilation=1, groups=8, bias=True):
        super().__init__()  # typically groups are 2, 4, 8, 16
        self.cho = cho
        self.groups = groups
        p = autopad(k, p)
        # decreases complexity, the idea of grouped convolutions is that 
        # the correlation between feature channels is sparse anyway,
        # and here we will be even more sparse since we only allow 
        # intra channel group correlation, 
        # use grouped convolution also for 1x1 convolutions (see ShuffleNet)
        # which are then called pointwise grouped convolutions
        self.conv = nn.Conv2d(chi, cho, k, s, p, 
            dilation=dilation, groups=groups, bias=bias)
        # example forward pass: 
        # assume g=2 groups and 6 channels (=> n=3 group size) 
        # 111222 (entries of the two groups are 1 and 2 respectively) 
        # reshaping to (2, 6): 
        # 111
        # 222
        # transposing:
        # 12
        # 12
        # 12
        # flattening:
        # 121212 => shuffled!

    def forward(self, x):
        x = self.conv(x)
        if self.groups != 1:
            # x has g * n output channels with g beeing the number 
            # of groups; to shuffel reshape to (g, n)
            x = x.reshape(x.size(0), self.groups, 
                int(self.cho / self.groups), x.size(-2), x.size(-1))
            # then transpose in the (g, n) dimensions
            x = torch.transpose(x, 1, 2)
            # finally flatten dimension (n, g) => channels shuffled!
            x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        return x 
    
    def fuse(self, bn):
        if self.groups != 1:  # here conv weights and bn weights don't match 
            s = self.conv.weight.shape  # remember original shape: cho, chi, k, k
            # shuffle conv weight in 'cho' dimension to match 'bn'
            x = self.conv.weight.reshape(self.conv.out_channels, -1)  # cho, chi*k^2 
            x = x.reshape(self.groups, int(self.cho / self.groups), -1)  # g, n, chi*k^2 
            x = torch.transpose(x, 1, 0)  # n, g, chi*k^2 
            x = x.reshape(self.conv.out_channels, -1)  # cho, chi*k^2 but shuffled
            self.conv.weight = x.reshape(*s)  # reshape copies, re-assign
            
            self.conv = fuse_conv_and_bn(self.conv, bn)  # now weigths match
            
            # shuffle conv weight in 'cho' dimension back to initial order
            x = self.conv.weight.reshape(self.conv.out_channels, -1)  # cho, chi*k^2
            x = x.reshape(int(self.cho / self.groups), self.groups, -1)  # n, g, chi*k^2 
            x = torch.transpose(x, 1, 0)  # g, n, chi*k^2 
            x = x.reshape(self.conv.out_channels, -1)  # cho, chi*k^2
            self.conv.weight = x.reshape(*s)  # reshape copies, re-assign
        else:  # straight forward case
            self.conv = fuse_conv_and_bn(self.conv, bn)
        return self
    
class Conv(nn.Module):
    def __init__(self, chi, cho, k=1, s=1, p=None, d=1, g=GROUPS, act=True, affine=True): 
        super().__init__()
        if chi % g != 0 or cho % g != 0:
            g = 1
            print(f'Channel {chi} or {cho} not divisible by groups: {g}; using groups=1')
        p = autopad(k, p)
        self.conv = BASE(chi, cho, k, s, p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cho, affine=affine)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def fuseforward(self, x):
        return self.act(self.conv(x))
    
    def fuse(self):  # merge batchnorm and convolution for inference speed up
        if not isinstance(self.conv, nn.Conv2d):
            self.conv = self.conv.fuse(self.bn)  # each custom BASE has own fuse method
        else:
            self.conv = fuse_conv_and_bn(self.conv, self.bn)
        delattr(self, 'bn')  # remove batchnorm
        self.forward = self.fuseforward  # update forward

def test_conv(Base, nruns=100, device='cuda'):
    # will do basic sanity check, we want to get same spatial 
    # dimension with custom convolutions as in standard module
    # to swap convolution types conveniently
    chi, cho, k, s = 8, 32, 3, 1
    
    x = torch.randn(16, chi, 512, 512)
    conv = Base(chi, cho, k, s, autopad(k))
    conv_ = nn.Conv2d(chi, cho, k, s, autopad(k))
    
    if 'cuda' in device: 
        assert torch.cuda.is_available()
        conv.cuda().train()
        conv_.cuda().train()
        x = x.cuda()
        
        if torch.backends.cudnn.benchmark:
            # have to do warm up iterations for fair comparison
            print('benchmark warm up...')
            for _ in range(50):
                _ = conv(x)
    else:
        conv.cpu().train()
        conv_.cpu().train()
        nruns=1
    
    p = count_params(conv)
    p_ = count_params(conv_)
    # relative number of parameter change in brackets w.r.t. nn.conv2d
    print(f'Number of parameters: {p} ({p / p_ * 100:.2f}%)')
    
    # ensure same behaviour as standard module
    out = conv(x)
    out_ = conv_(x)
    assert out.shape == out_.shape, f'Shape missmatch, should be {out_.shape} but is {out.shape}'
    
    # g0 = torch.randn_like(out)
    # performance test without feature/target loading
    # because that would require a significant amount of overhead
    start = time_synchronized()
    for _ in range(nruns):
        out = conv(x)
        for param in conv.parameters():
            param.grad = None
        out.mean().backward()  # out.backward(g0)
    end = time_synchronized()
    
    print(f'Forward + Backward time: {(end - start) * 1000 / nruns:.3f}ms')
        
def test_fuse(Module, nruns=10, device='cuda'):
    # Module has batchnorm and more bundled together
    chi, cho, k, s = 8, 32, 3, 1
    
    module = Module(chi, cho, k, s)
    assert hasattr(module, 'fuse')
    
    x = torch.randn(16, chi, 512, 512)
    
    if 'cuda' in device: 
        assert torch.cuda.is_available()
        module.cuda().train()
        x = x.cuda()
    else:
        module.cpu().train()
        nruns=1
    
    optimizer = optim.Adam(module.parameters())
    
    # traing batchnorm and convolution: 
    for _ in range(nruns):
        optimizer.zero_grad(set_to_none=True)  # version >=1.7
        out = module(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
    
    # test fuse function:
    x = torch.randn(1, chi, 512, 512, device=device)
    module.eval()  # MUST BE in eval mode to work!!!
    
    with torch.no_grad():  # compare module outputs
        unfused = module(x)
        module.fuse()
        #import pdb; pdb.set_trace()
        fused = module(x)  # merged batchnorm
        d = torch.norm(unfused - fused).div(unfused.norm()).item()
        print('fuse relative error: %.8f' % d)
        
def test_fuse_conv_and_bn():
    x = torch.randn(16, 3, 256, 256)
    rn18 = torchvision.models.resnet18(pretrained=True)
    rn18.eval()
    net = torch.nn.Sequential(
        rn18.conv1,
        rn18.bn1
    )
    y1 = net.forward(x)
    fusedconv = fuse_conv_and_bn(net[0], net[1])
    y2 = fusedconv.forward(x)
    d = (y1 - y2).norm().div(y1.norm()).item()
    print('fuse relative error: %.8f' % d)
    
def main():
    global BASE, GROUPS
    
    test_fuse_conv_and_bn()
    
    for Base in [nn.Conv2d, DeformConv, SpatiallyConv, 
        DepthwiseConv, FlattenedConv, GroupedConv, ShuffledGroupedConv]:
        BASE = Base
        if 'group' in Base.__name__.lower():
            GROUPS = 8
        else:
            GROUPS = 1
        
        print(f'BASE: {BASE.__name__}, GROUPS: {GROUPS}')
        test_conv(Base, device='cuda')
        #test_fuse(Conv, device='cuda')
    
if __name__ == '__main__':
    from utils import init_torch_seeds
    
    # init_torch_seeds(seed=1234)
    main()
    
    '''
    ----- on cpu  -----
    BASE: Conv2d, GROUPS: 1
    Number of parameters: 2336 (100.00%)
    Forward + Backward time: 329.352ms
    BASE: DeformConv, GROUPS: 1
    Number of parameters: 3650 (156.25%)
    Forward + Backward time: 55058.459ms
    BASE: SpatiallyConv, GROUPS: 1
    Number of parameters: 1000 (42.81%)
    Forward + Backward time: 574.737ms
    BASE: DepthwiseConv, GROUPS: 1
    Number of parameters: 368 (15.75%)
    Forward + Backward time: 558.899ms
    BASE: FlattenedConv, GROUPS: 1
    Number of parameters: 544 (23.29%)
    Forward + Backward time: 1656.760ms
    BASE: GroupedConv, GROUPS: 8
    Number of parameters: 320 (13.70%)
    Forward + Backward time: 585.382ms
    BASE: ShuffledGroupedConv, GROUPS: 8
    Number of parameters: 320 (13.70%)
    Forward + Backward time: 682.220ms
    
    ----- on cuda -----
    BASE: Conv2d, GROUPS: 1
    Number of parameters: 2336 (100.00%)
    Forward + Backward time: 8.753ms
    BASE: DeformConv, GROUPS: 1
    Number of parameters: 3650 (156.25%)
    Forward + Backward time: 70.436ms
    BASE: SpatiallyConv, GROUPS: 1
    Number of parameters: 1000 (42.81%)
    Forward + Backward time: 20.321ms
    BASE: DepthwiseConv, GROUPS: 1
    Number of parameters: 368 (15.75%)
    Forward + Backward time: 23.472ms
    BASE: FlattenedConv, GROUPS: 1
    Number of parameters: 544 (23.29%)
    Forward + Backward time: 40.704ms
    BASE: GroupedConv, GROUPS: 8
    Number of parameters: 320 (13.70%)
    Forward + Backward time: 14.369ms
    BASE: ShuffledGroupedConv, GROUPS: 8
    Number of parameters: 320 (13.70%)
    Forward + Backward time: 16.996ms
    
    fuse relative error: 0.00000030
    BASE: Conv2d, GROUPS: 1
    fuse relative error: 0.00000015
    BASE: DeformConv, GROUPS: 1
    fuse relative error: 0.00000015
    BASE: SpatiallyConv, GROUPS: 1
    fuse relative error: 0.00000011
    BASE: DepthwiseConv, GROUPS: 1
    fuse relative error: 0.00000009
    BASE: FlattenedConv, GROUPS: 1
    fuse relative error: 0.00000013
    BASE: GroupedConv, GROUPS: 8
    fuse relative error: 0.00000016
    BASE: ShuffledGroupedConv, GROUPS: 8
    fuse relative error: 0.00000015
    '''
