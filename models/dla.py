import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import math
import sys

sys.path.append('./')
from utils import count_params, autopad 
from convolutions import (DeformConv, SpatiallyConv, 
    DepthwiseConv, FlattenedConv, GroupedConv, ShuffledGroupedConv)

BASE = ShuffledGroupedConv
GROUPS = 8

#TODO ReLU inplace! look at yolov5
class Conv(nn.Module):
    def __init__(self, chi, cho, k=1, s=1, p=None, d=1, g=GROUPS, act=nn.ReLU(), affine=True):  
        super().__init__()
        self.conv = BASE(chi, cho, k, s, autopad(k, p), 
            dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cho, affine=affine)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def fuse_forward(self, x):  # TODO
        return self.act(self.conv(x))

class UpSample(nn.Module):
    def __init__(self, chi, cho, k=1, s=3, p=None, d=1, g=GROUPS, sf=2):
        super().__init__() 
        # learned upsampling, avoids checkerboard artifacts
        self.sf = sf
        self.conv = Conv(chi, cho, k, s, p, d=d, g=g)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.sf, mode='nearest'))

class Bottleneck(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, d=1, g=GROUPS, e=0.5):
        super().__init__()
        chh = int(cho * e)
        self.conv1 = Conv(chi, chh, 1, 1, 0, d=d, g=g)
        self.conv2 = Conv(chh, chh, k, s, p=d, d=d, g=g)
        self.conv3 = Conv(chh, cho, 1, 1, 0, d=d, g=g, act=nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.relu(residual + self.conv3(self.conv2(self.conv1(x))))
    
class BasicBlock(nn.Module):
    def __init__(self, chi, cho, k=3, s=1, p=None, d=1, g=GROUPS):
        super().__init__()
        self.conv1 = Conv(chi, cho, k, s, p=d, d=d, g=g)
        self.conv2 = Conv(cho, cho, k, 1, p=d, d=d, g=g, act=nn.Identity())
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.relu(residual + self.conv2(self.conv1(x)))

class Root(nn.Module):
    def __init__(self, chi, cho, k, residual):
        super().__init__()
        self.conv = Conv(chi, cho, 1, 1, act=nn.Identity())
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
            self.project = Conv(chi, cho, k=1, s=1, act=nn.Identity())
                
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

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
            stride=1, padding=0, bias=True)

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
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            return x
        
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

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

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
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
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

def centernet(heads, num_layers=34, head_conv=256, down_ratio=4):
    # example: object detection
    # heads = {'cpt_hm': num_classes, 'cpt_off': 2, 'wh': 2}
    model = DLASeg('dla{}'.format(num_layers), heads,
        down_ratio=down_ratio, head_conv=head_conv)
    return model


if __name__ == '__main__':
    model = centernet(heads={'cpt_hm': 30, 'cpt_off': 2, 'wh': 2})
    print(f'BASE: {BASE.__name__}, GROUPS: {GROUPS}')
    print(f'Parameters: {count_params(model) / 10**6:0.3}M')
    assert torch.cuda.is_available()
    model.cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    model.eval()
    x = torch.randn(1, 3, 512, 512).cuda()
    n = 10  # average over 'n' runs
    with torch.no_grad():
        start.record()
        for _ in range(n):
            out = model(x)
        end.record()
    torch.cuda.synchronize()
    
    print(f'Elapsed forward time (inference): {start.elapsed_time(end) / n:.3f}ms')
    
    print(f'In: {x.size()}')
    for head_key, head in out.items():
        print(f'{head_key} Out: {head.size()}')
        
    """ In: torch.Size([1, 3, 512, 512])
    BASE: Conv2d, GROUPS: 1
    Parameters: 18.5M
    Elapsed forward time (inference): 28.570ms

    BASE: DeformConv, GROUPS: 1
    Parameters: 19.5M
    Elapsed forward time (inference): 37.897ms

    BASE: SpatiallyConv, GROUPS: 1
    Parameters: 17.1M
    Elapsed forward time (inference): 35.122ms

    BASE: DepthwiseConv, GROUPS: 1
    Parameters: 4.38M
    Elapsed forward time (inference): 29.335ms

    BASE: FlattenedConv, GROUPS: 1
    Parameters: 4.38M
    Elapsed forward time (inference): 33.086ms

    BASE: GroupedConv, GROUPS: 8
    Parameters: 3.18M
    Elapsed forward time (inference): 32.019ms

    BASE: ShuffledGroupedConv, GROUPS: 8
    Parameters: 3.18M
    Elapsed forward time (inference): 34.249ms
    """
    