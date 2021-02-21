import torch
from torch import nn
import numpy as np
import thop
import time
from copy import deepcopy
from torchvision import ops

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def autopad(k, p=None):
    if p is None:  # pad s.t. same spatial shape after convolution
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    if isinstance(conv, nn.Conv2d):
        convclass = nn.Conv2d
    elif isinstance(conv, ops.DeformConv2d):
        convclass = ops.DeformConv2d
    else:
        raise NotImplementedError
        
    fusedconv = convclass(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
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

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def profile(model, device='cuda', img_size=512, nruns=100, verbose=False):
    x = torch.randn(1, 3, img_size, img_size)
    print(f'Input size: {x.size()}')

    if torch.cuda.is_available() and 'cuda' in device: 
        model.cuda().eval()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        x = x.cuda()

        start.record()
        for _ in range(nruns):
            out = model(x)
        end.record()
        torch.cuda.synchronize()
        print(f'Forward time: {start.elapsed_time(end) / nruns:.3f}ms (cuda)')
    else:
        model.cpu().eval()
        start = time_synchronized()
        for _ in range(nruns):
            out = model(x)
        end = time_synchronized()  # seconds
        print(f'Forward time: {(end - start) / 1000 / nruns:.3f}ms (cpu)')
    
    if verbose:
        if isinstance(out, dict):
            for head_key, head in out.items():
                print(f'{head_key} output: {head.size()}')
        elif isinstance(out, list) or isinstance(out, tuple):
            print('output:', end=' ')
            for head in out:
                print(head.size(), end=', ')
            print('')
        else:
            print('output:', out.size())

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

def init_torch_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
            
def model_info(model, verbose, img_size):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:
        # assume model expects RGB image if not otherwise specified
        input_channels = getattr(model, 'input_channels', 3)
        img = torch.randn(1, input_channels, img_size, img_size, 
            device=next(model.parameters()).device)
        # macs ... multiply-add computations
        # flops ... floating point operations
        macs, _ = thop.profile(deepcopy(model), inputs=(img,), verbose=False)
        flops = macs / 1E9 * 2  # each mac = 2 flops (addition + multiplication)
        fs = ', %.1f GFLOPs' % (flops)
    except ImportError:
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p / 10**6:0.3}M parameters, {n_g / 10**6:0.3}M gradients{fs}")
