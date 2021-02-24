import math
import torch
from torch import nn
import numpy as np
import thop
import time
from copy import deepcopy
from torchvision import ops
from contextlib import contextmanager

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

@contextmanager
def torch_distributed_zero_first(rank: int):
    """
    Make all processes in distributed training wait for master to do something.
    
    when a process encounters a barrier it will block, a process is blocked by a 
    barrier until all processes have encountered a barrier, upon which the 
    barrier is lifted for all processes
    """
    if rank != 0:
        print(f'rank: {rank} encounter barrier 1')
        torch.distributed.barrier()
    yield
    if rank == 0:
        print(f'rank: {rank} encounter barrier 2')
        torch.distributed.barrier()

class ModelEMA:
    """
    When training a model, it is often beneficial to maintain moving averages 
    of the trained parameters. Evaluations that use averaged parameters 
    sometimes produce significantly better results than the final trained values!
    """
    def __init__(self, model, decay=0.9999, updates=0):
        model = deepcopy(model.module if is_parallel(model) else model).eval()
        self.ema = model
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():  # Update EMA parameters
            self.updates += 1
            d = self.decay(self.updates)
            
            # model state_dict
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict() 
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    # note: inplace operation allow concurrent lockless updates
                    v -= (1. - d) * (v - msd[k].detach())

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)  # Update EMA attributes

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

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
            
def model_info(model, verbose=False, img_size=512):
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
