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
    return time.time()  # seconds

def print_structure(model):
    for m in model.children():
        print(m)

def profile(model, device='cuda', img_size=512, nruns=100, verbose=False):
    x = torch.randn(1, 3, img_size, img_size)
    print(f'Input size: {x.size()}')

    if torch.cuda.is_available() and 'cuda' in device: 
        model.cuda().eval()
        x = x.cuda()

        if torch.backends.cudnn.benchmark:
            # have to do warm up iterations for fair comparison
            print('benchmark warm up...')
            for _ in range(50):
                _ = model(x)
        
        start = time_synchronized()
        for _ in range(nruns):
            o = model(x)
        end = time_synchronized()
        print(f'Forward time: {(end - start) * 1000 / nruns:.3f}ms (cuda)')
    else:
        model.cpu().eval()
        start = time_synchronized()
        for _ in range(nruns):
            o = model(x)
        end = time_synchronized()  # seconds
        print(f'Forward time: {(end - start) * 1000 / nruns:.3f}ms (cpu)')
    
    if verbose:
        if isinstance(o, dict):
            for head_key, head in o.items():
                print(f'{head_key} output: {head.size()}')
        elif isinstance(o, list) or isinstance(o, tuple):
            print('output:', end=' ')
            for head in o:
                print(head.size(), end=', ')
            print('')
        else:
            print('output:', o.size())
            
def profile_training(model, img_size=512, nruns=100):
    x = torch.randn(16, 3, img_size, img_size)
    print(f'Input size: {x.size()}')

    assert torch.cuda.is_available()
    model.cuda().train()
    x = x.cuda()
    
    o = model(x)
    if isinstance(o, list) or isinstance(o, tuple):
        g0 = [torch.rand_like(item) for item in o]
    elif isinstance(o, dict):
        g0 = [torch.rand_like(item) for item in o.values()]
    else:
        g0 = [torch.rand_like(o)]
    
    if torch.backends.cudnn.benchmark:
        # have to do warm up iterations for fair comparison
        print('benchmark warm up forward...')
        for _ in range(50):
            o = model(x)

        print('benchmark warm up backward...')
        for _ in range(50):
            o = model(x)
            for param in model.parameters():
                param.grad = None
            o = o.values() if isinstance(o, dict) else ([o] if isinstance(o, torch.Tensor) else o)         
            for i, v in enumerate(o):
                v.backward(g0[i], retain_graph=i < len(o) - 1)

    print(f'run through forward pass for {nruns} runs...')
    start = time_synchronized()
    for _ in range(nruns):
        o = model(x)
    end = time_synchronized()
    fwd_time = end - start  # fwd only
    
    print(f'run through forward and backward pass for {nruns} runs...')
    torch.cuda.reset_peak_memory_stats(device='cuda')
    start = time_synchronized()
    for _ in range(nruns):
        o = model(x)
        for param in model.parameters():
            param.grad = None
        o = o.values() if isinstance(o, dict) else ([o] if isinstance(o, torch.Tensor) else o)          
        for i, v in enumerate(o):
            v.backward(g0[i], retain_graph=i < len(o) - 1)
    end = time_synchronized()
    mem = torch.cuda.max_memory_reserved(device='cuda')  # bytes
    bwd_time = end - start  # fwd + bwd
    bwd_time = (bwd_time - fwd_time)  # bwd only

    print(f'Forward time: {fwd_time * 1000 / nruns:.3f}ms (cuda)')
    print(f'Backward time: {bwd_time * 1000 / nruns:.3f}ms (cuda)')
    print(f'Maximum of managed memory: {mem / 10**9}GB')

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
        
    print('PyTorch version {}'.format(torch.__version__))
    print('CUDA version {}'.format(torch.version.cuda))
    print('cuDNN version {}'.format(cudnn.version()))
    print('cuDNN deterministic {}'.format(cudnn.deterministic))
    print('cuDNN benchmark {}'.format(cudnn.benchmark))

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
