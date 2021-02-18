import math
from pathlib import Path
from copy import deepcopy
import yaml
import thop
import time 

import numpy as np
import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)

class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        #import pdb; pdb.set_trace()
        return torch.cat(x, self.d)

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            # different channels will kick in while training one after the other
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    # MixConv partitions channels into groups and
    # apply different kernel size to each group
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch, head: bool):  # model_dict, input channels, use heads?
    gd, gw = d['depth_multiple'], d['width_multiple']
    
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    
    if head:
        print('Build Yolov5 Backbone + Head')
        d = d['backbone'] + d['head']
    else:
        print('Build Yolov5 Backbone')
        d = d['backbone']
        
    for i, (f, n, m, args) in enumerate(d):  
        # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, 
            MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        # attach index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def init_torch_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
            
def model_info(model, verbose=False, img_size=512):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = thop.profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p / 10**6:0.3}M parameters, {n_g / 10**6:0.3}M gradients{fs}")

class Yolov5(nn.Module):
    def __init__(self, cfg='models/yolov5s.yaml', ch=3, head=True):  
        # yaml config file, input channels, head=False => backbone only!
        super().__init__()
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        self.model, self.save = parse_model(deepcopy(self.yaml), [ch], head=head)
        # Init weights, biases
        initialize_weights(self)

    def forward(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # from earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]      

            if profile:
                # FLOPS
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                # FLOPS, Number of Parameters, Time (ms), Module Type
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x
        
    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)
        
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

if __name__ == '__main__':
    # import sys
    # sys.path.append('./')
    
    for cfg in ['models/yolov5s.yaml', 'models/yolov5m.yaml', 
                'models/yolov5l.yaml', 'models/yolov5x.yaml']:
        print(f'{10*"-"} {cfg} {10*"-"}')
        yolo = Yolov5(cfg, head=True).cuda().eval()
        model_info(yolo)

        x = torch.randn((1, 3, 512, 512), device='cuda')
        y = yolo(x, profile=False)
        #y = yolo(x, profile=True)
        print(f'In/Out: {x.size()} --> {y.size()}')

    """
    ---------- models/yolov5s.yaml ----------
    Build Yolov5 Backbone + Head
    Model Summary: 278 layers, 7.05M parameters, 7.05M gradients, 10.4 GFLOPS
           0.5      3520       0.2ms Focus                                   
           0.6     18560       0.1ms Conv                                    
           0.6     18816       0.7ms C3                                      
           0.6     73984       0.1ms Conv                                    
           1.3    156928       1.4ms C3                                      
           0.6    295424       0.1ms Conv                                    
           1.3    625152       1.3ms C3                                      
           0.6   1180672       0.2ms Conv                                    
           0.3    656896       0.4ms SPP                                     
           0.6   1182720       0.7ms C3                                      
           0.1    131584       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
           0.7    361984       0.7ms C3                                      
           0.1     33024       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
           0.7     90880       0.7ms C3                                      
           0.3    147712       0.1ms Conv                                    
           0.0         0       0.0ms Concat                                  
           0.6    296448       0.7ms C3                                      
           0.3    590336       0.2ms Conv                                    
           0.0         0       0.0ms Concat                                  
           0.6   1182720       0.7ms C3                                      
    9.1ms total
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 512, 16, 16])
    ---------- models/yolov5m.yaml ----------
    Build Yolov5 Backbone + Head
    Model Summary: 386 layers, 21.0M parameters, 21.0M gradients, 32.2 GFLOPS
           0.7      5280       0.2ms Focus                                   
           1.4     41664       0.2ms Conv                                    
           2.1     65280       1.4ms C3                                      
           1.4    166272       0.2ms Conv                                    
           5.2    629760       2.2ms C3                                      
           1.4    664320       0.2ms Conv                                    
           5.1   2512896       2.5ms C3                                      
           1.4   2655744       0.3ms Conv                                    
           0.8   1476864       0.4ms SPP                                     
           2.1   4134912       1.0ms C3                                      
           0.2    295680       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
           2.4   1182720       1.0ms C3                                      
           0.2     74112       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
           2.4    296448       1.0ms C3                                      
           0.7    332160       0.2ms Conv                                    
           0.0         0       0.0ms Concat                                  
           2.1   1035264       1.0ms C3                                      
           0.7   1327872       0.3ms Conv                                    
           0.0         0       0.0ms Concat                                  
           2.1   4134912       1.0ms C3                                      
    13.7ms total
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 768, 16, 16])
    ---------- models/yolov5l.yaml ----------
    Build Yolov5 Backbone + Head
    Model Summary: 494 layers, 46.6M parameters, 46.6M gradients, 73.0 GFLOPS
           0.9      7040       0.3ms Focus                                   
           2.4     73984       0.3ms Conv                                    
           5.1    156928       1.5ms C3                                      
           2.4    295424       0.3ms Conv                                    
          13.2   1611264       2.9ms C3                                      
           2.4   1180672       0.3ms Conv                                    
          13.2   6433792       3.0ms C3                                      
           2.4   4720640       0.4ms Conv                                    
           1.3   2624512       0.4ms SPP                                     
           5.1   9971712       1.2ms C3                                      
           0.3    525312       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
           5.6   2757632       1.3ms C3                                      
           0.3    131584       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
           5.7    690688       1.3ms C3                                      
           1.2    590336       0.3ms Conv                                    
           0.0         0       0.0ms Concat                                  
           5.1   2495488       1.3ms C3                                      
           1.2   2360320       0.3ms Conv                                    
           0.0         0       0.0ms Concat                                  
           5.1   9971712       1.3ms C3                                      
    16.9ms total
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 1024, 16, 16])
    ---------- models/yolov5x.yaml ----------
    Build Yolov5 Backbone + Head
    Model Summary: 602 layers, 87.2M parameters, 87.2M gradients, 139.0 GFLOPS
           1.2      8800       0.3ms Focus                                   
           3.8    115520       0.4ms Conv                                    
          10.1    309120       1.7ms C3                                      
           3.8    461440       0.3ms Conv                                    
          26.9   3285760       3.9ms C3                                      
           3.8   1844480       0.3ms Conv                                    
          26.9  13125120       3.7ms C3                                      
           3.8   7375360       0.5ms Conv                                    
           2.1   4099840       0.4ms SPP                                     
          10.1  19676160       1.5ms C3                                      
           0.4    820480       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
          10.9   5332480       1.5ms C3                                      
           0.4    205440       0.1ms Conv                                    
           0.0         0       0.0ms torch.nn.modules.upsampling.Upsample    
           0.0         0       0.0ms Concat                                  
          10.9   1335040       1.5ms C3                                      
           1.9    922240       0.3ms Conv                                    
           0.0         0       0.0ms Concat                                  
          10.1   4922880       1.5ms C3                                      
           1.9   3687680       0.4ms Conv                                    
           0.0         0       0.0ms Concat                                  
          10.1  19676160       1.5ms C3                                      
    20.4ms total
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 1280, 16, 16])
    """