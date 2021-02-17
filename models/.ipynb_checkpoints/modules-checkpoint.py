import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import ops

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------ convolutions -------

def autopad(k, p=None):
    if p is None:  # pad s.t. same spatial shape after convolution
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class DeformConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, groups=1, bias=True):
        super().__init__() 
        k = k if isinstance(k, tuple) else (k, k) 
        p = autopad(k, p)
        self.convolution = ops.DeformConv2d(chi, cho, k, s, p, 
            groups=groups, bias=bias)
        # for each group we need output channels of 2 to get for each kernel weight
        # offset position in x, y (2 channels) and we have to know that for every 
        # pixel in the convolution output, thus we use same kernel size and padding!!
        self.offset = nn.Conv2d(chi, groups * 2 * k[0] * k[1], k, s, p, 
            groups=groups, bias=True)
        self._init_offset()

    def _init_offset(self):
        # as the original paper suggests initialize offsets with zeros
        # thus we start with a standard convolution
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)

    def forward(self, x):
        return self.convolution(x, self.offset(x))

class SpatiallyConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, groups=1, bias=True):
        super().__init__() 
        # decreases complexity, but kernel space limited
        k = k if isinstance(k, tuple) else (k, k)
        p = p if isinstance(p, tuple) else (p, p)
        p = autopad(k, p)
        self.conv1 = nn.Conv2d(chi, chi, (k[0], 1), s, (p[0], 0), 
            groups=1, bias=bias)
        self.conv2 = nn.Conv2d(chi, cho, (1, k[1]), s, (0, p[1]), 
            groups=1, bias=bias)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))

class DepthwiseConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, groups=1, bias=True):
        super().__init__() 
        p = autopad(k, p)
        # decreases complexity, smaller networks can be wider,
        # each filter soley has access to a single input channel
        # and we keep the number of input channels at first
        self.conv1 = nn.Conv2d(chi, chi, k, s, p, groups=chi, bias=bias)
        # learn channelwise(inter group) correlation with 1x1 convolutions
        self.conv2 = nn.Conv2d(chi, cho, 1, 1, 0, groups=1, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class FlattenedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, groups=1, bias=True):
        super().__init__() 
        # paper claims importance of bias term!
        k = k if isinstance(k, tuple) else (k, k)
        p = p if isinstance(p, tuple) else (p, p)
        p = autopad(k, p)
        # lateral, kernel: C x 1 x 1
        self.conv1 = nn.Conv2d(chi, cho, 1, 1, 0, groups=1, bias=True)
        # vertical, kernel: 1 x Y x 1
        self.conv2 = nn.Conv2d(cho, cho, (k[0], 1), s, (p[0], 0), 
                               groups=cho, bias=True)
        # horizontal, kernel: 1 x 1 x X,
        # last term can omit bias e.g. if batchnorm is done anyway afterwards 
        self.conv3 = nn.Conv2d(cho, cho, (1, k[1]), s, (0, p[1]), 
                               groups=cho, bias=bias)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))

class GroupedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, groups=8, bias=True):
        super().__init__()  # typically groups are 2, 4, 8, 16
        p = autopad(k, p)
        # decreases complexity, the idea of grouped convolutions is that 
        # the correlation between feature channels is sparse anyway,
        # and here we will be even more sparse since we only allow 
        # intra channel group correlation, 
        # use grouped convolution also for 1x1 convolutions (see ShuffleNet)
        # which are then called pointwise grouped convolutions
        self.conv = nn.Conv2d(chi, cho, k, s, p, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)

class ShuffledGroupedConv(nn.Module):
    def __init__(self, chi, cho, k, s=1, p=None, groups=8, bias=True):
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
        self.conv = nn.Conv2d(chi, cho, k, s, p, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        # x has g * n output channels with g beeing the number 
        # of groups; to shuffel reshape to (g, n)
        x = x.reshape(x.size(0), self.groups, 
            int(self.cho / self.groups), x.size(-2), x.size(-1))
        # then transpose in the (g, n) dimensions
        x = torch.transpose(x, 1, 2)
        # finally flatten dimension (g, n) => channels shuffled!
        x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        
        # note: assume g=2 groups and 6 channels (=> n=3) 
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
        return x 

def test_conv(Conv, n=10, benchmark=False):
    print(f'{10*"-"} Test {Conv.__name__} {10*"-"}')
    
    # will select optimal algorithm depending on input size,
    # note: for varying input sizes (inference) this will 
    # eventually slow down code when kept with non default value True
    # (still can use scaling and padding to keep resolution for inference)
    if benchmark:
        # when set to True the first run will be slower since different
        # algorithms are tested, thus run here once
        torch.backends.cudnn.benchmark = True

    # setup input and loss
    b, c, h, w = 16, 32, 128, 128
    chi, cho, k = c, 2 * c, 3
    s, p = 1, None
    p = autopad(k , p)
    
    loss_fn = nn.MSELoss()
    
    # will do basic sanity check, we want to get same spatial 
    # dimension with custom convolutions as in standard module
    # to swap convolution types conveniently
    assert torch.cuda.is_available()
    conv = Conv(chi, cho, k, s, p).cuda()
    conv_ = nn.Conv2d(chi, cho, k, s, p).cuda()
    
    p = count_params(conv)
    p_ = count_params(conv_)
    # relative number of parameter change in brackets w.r.t. nn.conv2d
    print(f'Number of parameters: {p} ({p / p_ * 100:.2f}%)')
    
    optimizer = optim.Adam(conv.parameters())
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    x = torch.randn(b, c, h, w).cuda()
    # ensure same behaviour as standard module
    out = conv(x)
    out_ = conv_(x)
    assert out.shape == out_.shape, f'Shape missmatch, should be {out_.shape} but is {out.shape}'
    y = torch.randn_like(out)
    
    # performance test without feature/target loading
    # because that would require a significant amount of overhead
    start.record()
    for _ in range(n):
        #optimizer.zero_grad(set_to_none=True)  # version >=1.7
        for param in conv.parameters():
            param.grad = None
        out = conv(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    end.record()
    torch.cuda.synchronize()
    print(f'Elapsed forward + backward time: {start.elapsed_time(end) / n:.3f}ms')
    
    if benchmark:  # reset to default
        torch.backends.cudnn.benchmark = False

# ------ basic building blocks -------

class Focus(nn.Module):
    def __init__(self, chi, cho):  
        super().__init__()
        # e.g. conv2d with bn and ReLU
        self.conv = Conv

    def forward(self, x):  # x(b, c, w, h) -> y(b, 4c, w/2, h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], 
            x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))

class UpSample(nn.Module):
    def __init__(self, scale_factor: float):
        super().__init__() 
        # learned upsampling, avoids checkerboard artifacts
        self.scale_factor = scale_factor
        self.conv_block = conv_block
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv_block(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, chi, cho, shortcut=True, groups=1, e=0.5):
        super().__init__()
        chh = int(cho * e)  # hidden channels
        self.conv1 = Conv(chi, chh, 1, groups=groups)
        self.conv2 = Conv(chh, cho, 3, groups=groups)
        self.conv3 = Conv(chi, chh, 1, groups=groups)
        self.add = shortcut and chi == cho

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

if __name__ == '__main__':
    # to get detailed cuda errors use:
    # CUDA_LAUNCH_BLOCKING=1 python myscript.py
    # OR
    # export CUDA_LAUNCH_BLOCKING=1
    # OR
    # run on cpu!
    for Conv in 
    test_conv(nn.Conv2d)  # sanity check of test function
    test_conv(DeformConv)
    test_conv(SpatiallyConv)
    test_conv(DepthwiseConv)
    test_conv(FlattenedConv)
    test_conv(GroupedConv)
    test_conv(ShuffledGroupedConv)
    
    '''
    ---------- Test Conv2d ----------
    Number of parameters: 18496 (100.00%)
    Elapsed forward + backward time: 3.342ms
    ---------- Test DeformConv ----------
    Number of parameters: 23698 (128.12%)
    Elapsed forward + backward time: 19.018ms
    ---------- Test SpatiallyConv ----------
    Number of parameters: 9312 (50.35%)
    Elapsed forward + backward time: 3.258ms
    ---------- Test DepthwiseConv ----------
    Number of parameters: 2432 (13.15%)
    Elapsed forward + backward time: 2.473ms
    ---------- Test FlattenedConv ----------
    Number of parameters: 2624 (14.19%)
    Elapsed forward + backward time: 4.105ms
    ---------- Test GroupedConv ----------
    Number of parameters: 2368 (12.80%)
    Elapsed forward + backward time: 6.615ms
    ---------- Test ShuffledGroupedConv ----------
    Number of parameters: 2368 (12.80%)
    Elapsed forward + backward time: 7.009ms
    '''

    # with affine=False we won't learn mean and variance shift
    # thus we aim at always normalizing to zero mean, unit variance activations
    bn = nn.BatchNorm2d(64, affine=False)
    
    # performance guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    # try out torch.backends.cudnn.benchmark = True for possibly more efficient algorithms
    # pin_memory=True for DataLoader to increase speed when transfering to GPU
    # verison >=1.7 use optimizer.zero_grad(set_to_none=True) will set grads to None
    # for Pointwise operations (elementwise addition, multiplication, math functions)
    # use @torch.jit.script decorator for function with many pointwise operations, 
    # this won't spawn multiple kernels (memory copy overhead)
    # avoid unnecessary CPU-GPU synchronization with calls like: print(cuda_tensor),
    # cuda_tensor.item(), memory copies: tensor.cuda(), cuda_tensor.cpu(), 
    # instead create tensors directly on the target device e.g. 
    # torch.rand(size, device=torch.device('cuda'))
    
    # cuda guide: https://pytorch.org/docs/stable/notes/cuda.html
    # get memory consumption currently or max peak since start of script with:
    # torch.cuda.memory_allocated(device) or
    # torch.cuda.max_memory_allocated(device)
    # to allow TF32 mode on matmul, cuDNN, these flags default to True,
    # normally we operate on 32FP with 1bit sign, 8 exponent, 23 mantissa
    # with TF32 mode we do operation with 1bit sign, 8 exponent, 10 mantissa
    # torch.backends.cuda.matmul.allow_tf32 = True and
    # torch.backends.cudnn.allow_tf32 = True
    # note: on NVIDIA GPUs since Ampere architecture (V100 has only Volta)
    # NVIDIA shown similar performance for FP32 an TF32 trained networks!
