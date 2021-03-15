import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import math
import sys

try:
    sys.path.append('./')
    from utils import (initialize_weights, model_info, time_synchronized, 
        profile, profile_training, init_torch_seeds) 
    from convs import Conv
    import convs
except ImportError:
    from .utils import (initialize_weights, model_info, time_synchronized, 
        profile, profile_training, init_torch_seeds) 
    from .convs import Conv
    from . import convs  

class UpSample(nn.Module):
    def __init__(self, chi, cho, k=1, s=3, p=None, d=1, g=convs.GROUPS, sf=2):
        super().__init__()  # avoids checkerboard artifacts
        self.sf = sf
        self.conv = Conv(chi, cho, k, s, p, d=d, g=g)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.sf, mode='nearest'))

class Bottleneck(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, d=1, g=convs.GROUPS, e=0.5):
        super().__init__()
        chh = int(cho * e)
        self.conv1 = Conv(chi, chh, 1, 1, 0, d=d, g=g)
        self.conv2 = Conv(chh, chh, k, s, p=d, d=d, g=g)
        self.conv3 = Conv(chh, cho, 1, 1, 0, d=d, g=g, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.relu(residual + self.conv3(self.conv2(self.conv1(x))))
    
class BasicBlock(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, d=1, g=convs.GROUPS):
        super().__init__()
        self.conv1 = Conv(chi, cho, k, s, p=d, d=d, g=g)
        self.conv2 = Conv(cho, cho, k, 1, p=d, d=d, g=g, act=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.relu(residual + self.conv2(self.conv1(x)))

class Root(nn.Module):
    def __init__(self, chi, cho, k, residual):
        super().__init__()
        self.conv = Conv(chi, cho, 1, 1, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        if self.residual:
            x += children[0]
        return self.relu(x)
    
class Tree(nn.Module):
    def __init__(self, levels, block, chi, cho, s=1,
                 level_root=False, root_dim=0, root_k=1,
                 dilation=1, root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * cho
        if level_root:
            root_dim += chi
        if levels == 1:
            self.tree1 = block(chi, cho, s=s, d=dilation)
            self.tree2 = block(cho, cho, s=1, d=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, chi, cho, s=s, root_dim=0,
                root_k=root_k, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, cho, cho, root_dim=root_dim + cho,
                root_k=root_k, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, cho, root_k, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if s > 1:
            self.downsample = nn.MaxPool2d(s, stride=s)
        if chi != cho:
            self.project = Conv(chi, cho, k=1, s=1, act=False)
                
    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super().__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = Conv(3, channels[0], k=7, s=1, p=3)
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], s=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
            level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
            level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
            level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
            level_root=True, root_residual=residual_root)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, chi, cho, convs, s=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(Conv(chi, cho, k=3, s=s if i == 0 else 1,
                p=dilation, d=dilation))
            chi = cho
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y

def dla34(**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    return model

def dla60(**kwargs):  # DLA-60
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    return model

class IDAUp(nn.Module):
    def __init__(self, node_k, out_dim, channels, up_factors):
        super().__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = Conv(c, out_dim, k=1, s=1)
                    
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                up = UpSample(out_dim, out_dim, k=3, 
                    s=1, p=None, d=1, g=out_dim, sf=2)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            

        for i in range(1, len(channels)):
            node = Conv(out_dim * 2, out_dim, k=node_k, s=1)
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y

class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super().__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x

def fill_head_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DLASeg(nn.Module):
    def __init__(self, base_name, heads, down_ratio=4, head_conv=256):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        for head_key in self.heads:
            num_channels = self.heads[head_key]
            head = nn.Sequential(
              nn.Conv2d(channels[self.first_level], head_conv,
                kernel_size=3, padding=1, bias=True),
              nn.ReLU(inplace=True),
              nn.Conv2d(head_conv, num_channels, 
                kernel_size=1, stride=1, 
                padding=0, bias=True))
            if 'hm' in head_key:
                head[-1].bias.data.fill_(-2.19)
            else:
                fill_head_weights(head)
            self.__setattr__(head_key, head)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret
    
    def fuse(self):  # fuse Conv2d() + BatchNorm2d() layers
        # used to speed up inference
        print('Fusing layers... ')
        self.eval()  # MUST BE in eval mode to work!
        for m in self.children():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.fuse()
        self.info()
        return self
    
    def info(self, verbose=False, img_size=512):  # print model information
        model_info(self, verbose, img_size)

def centernet(heads, num_layers=34, head_conv=256, down_ratio=4, pretrained=False):
    assert pretrained == False, 'Not supporting pretrained backbone in new version.'
    # example: object detection
    # heads = {'cpt_hm': num_classes, 'cpt_off': 2, 'wh': 2}
    model = DLASeg('dla{}'.format(num_layers), heads,
        down_ratio=down_ratio, head_conv=head_conv)
    initialize_weights(model)
    return model

if __name__ == '__main__':
    init_torch_seeds(seed=1234)
    
    # from convs import print_structure
    # print_structure(centernet(heads={'cpt_hm': 30, 'cpt_off': 2, 'wh': 2}))
    
    from convs import (DeformConv, SpatiallyConv, 
        DepthwiseConv, FlattenedConv, GroupedConv, ShuffledGroupedConv)
    
    for Base in [nn.Conv2d, DeformConv, SpatiallyConv, 
        DepthwiseConv, FlattenedConv, GroupedConv, ShuffledGroupedConv]:
        
        # change 'BASE' class for 'Conv' wrapper class
        convs.BASE = Base
        if 'group' in Base.__name__.lower():
            convs.GROUPS = 8
        else:
            convs.GROUPS = 1
            
        print(f'BASE: {convs.BASE.__name__}, GROUPS: {convs.GROUPS}')
        model = centernet(heads={'cpt_hm': 30, 'cpt_off': 2, 'wh': 2})
        model.info()  # summary
        
        try:
            profile(model)  # timing
            model.fuse()  # fuse and print summary again
            profile(model)  # fuse timing
            profile_training(model)  # forward + backward timing/memory
        except Exception as e:
            print(e)
            
    """
    PyTorch version 1.6.0
    CUDA version 10.2
    cuDNN version 7605
    cuDNN deterministic False
    cuDNN benchmark True
    BASE: Conv2d, GROUPS: 1
    Model Summary: 260 layers, 17.9M parameters, 17.9M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 9.625ms (cuda)
    Fusing layers... 
    Model Summary: 260 layers, 17.9M parameters, 17.9M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 9.121ms (cuda)
    BASE: DeformConv, GROUPS: 1
    Model Summary: 362 layers, 19.0M parameters, 19.0M gradients, 30.6 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 21.795ms (cuda)
    Fusing layers... 
    Model Summary: 362 layers, 19.0M parameters, 19.0M gradients, 30.6 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 21.822ms (cuda)
    BASE: SpatiallyConv, GROUPS: 1
    Model Summary: 362 layers, 16.6M parameters, 16.6M gradients, 58.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 13.423ms (cuda)
    Fusing layers... 
    Model Summary: 362 layers, 16.6M parameters, 16.6M gradients, 58.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 13.519ms (cuda)
    BASE: DepthwiseConv, GROUPS: 1
    Model Summary: 362 layers, 3.88M parameters, 3.88M gradients, 23.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 12.050ms (cuda)
    Fusing layers... 
    Model Summary: 362 layers, 3.88M parameters, 3.88M gradients, 23.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 11.684ms (cuda)
    BASE: FlattenedConv, GROUPS: 1
    Model Summary: 413 layers, 3.86M parameters, 3.86M gradients, 24.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 17.506ms (cuda)
    Fusing layers... 
    Model Summary: 413 layers, 3.86M parameters, 3.86M gradients, 24.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 17.178ms (cuda)
    BASE: GroupedConv, GROUPS: 8
    Model Summary: 311 layers, 17.9M parameters, 17.9M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 11.208ms (cuda)
    Fusing layers... 
    Model Summary: 311 layers, 17.9M parameters, 17.9M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 11.097ms (cuda)
    BASE: ShuffledGroupedConv, GROUPS: 8
    Model Summary: 311 layers, 17.9M parameters, 17.9M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 9.485ms (cuda)
    Fusing layers... 
    Model Summary: 311 layers, 17.9M parameters, 17.9M gradients, 62.1 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 9.421ms (cuda)
    """
    