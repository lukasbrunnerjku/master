import torch
from torch import nn
from torch import optim
from torchvision import models
import sys

try:
    sys.path.append('./')
    from yolo import Yolov5
    from ghostnet import ghostnet
except ImportError:
    from .yolo import Yolov5
    from .ghostnet import ghostnet

def get_backbone(model_builder, pretrained=False):
    # model_builder ... either class or function
    name = model_builder.__name__
    if 'resnet' in name:
        model = model_builder(pretrained=pretrained)
        modules = list(model.children())[:-2]
        backbone = nn.Sequential(*modules)
    elif 'vgg' in name:
        backbone = model_builder(pretrained=pretrained).features
    elif 'shufflenet' in name:
        model = model_builder(pretrained=pretrained)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
    elif 'mobilenet' in name:
        model = model_builder(pretrained=pretrained)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
    elif name == 'Yolov5':
        cfg = './yolov5s.yaml'
        backbone = model_builder(cfg)
    elif name.lower() == 'ghostnet':
        model = model_builder()
        modules = list(model.children())[:-4]
        backbone = nn.Sequential(*modules)
    else:
        raise NotImplementedError
    
    x = torch.randn(1, 3, 512, 512)
    backbone.eval()
    out = backbone(x)  # b x c x h x w
    backbone.out_channels = out.size(1)  
    return backbone  
    
def main():
    from utils import (init_torch_seeds, model_info, profile, 
        profile_training)
    
    init_torch_seeds(seed=1234)
    
    # analyze backbone characterstics of different models
    model_builders = [
        models.resnet18, 
        models.resnet50,  
        models.vgg16, 
        models.shufflenet_v2_x2_0, 
        models.mobilenet_v2,
        Yolov5,
        ghostnet,
    ][-2:]
    
    for model_builder in model_builders:
        print(f'{10*"-"} {model_builder.__name__} {10*"-"}')
        model = get_backbone(model_builder, pretrained=False)
        model_info(model, verbose=False, img_size=512) 
        profile(model, verbose=True, amp=True)
        profile_training(model, amp=True)
    
    '''
    PyTorch version 1.6.0
    CUDA version 10.2
    cuDNN version 7605
    cuDNN deterministic False
    cuDNN benchmark True
    ---------- resnet18 ----------
    Model Summary: 66 layers, 11.2M parameters, 11.2M gradients, 19.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 2.913ms (cuda)
    output: torch.Size([1, 512, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 18.971ms (cuda)
    Backward time: 48.689ms (cuda)
    Maximum of managed memory: 4.049600512GB
    ---------- resnet50 ----------
    Model Summary: 149 layers, 23.5M parameters, 23.5M gradients, 42.9 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 7.414ms (cuda)
    output: torch.Size([1, 2048, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 70.931ms (cuda)
    Backward time: 152.753ms (cuda)
    Maximum of managed memory: 14.621343744GB
    ---------- vgg16 ----------
    Model Summary: 32 layers, 14.7M parameters, 14.7M gradients, 160.5 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 8.386ms (cuda)
    output: torch.Size([1, 512, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 103.586ms (cuda)
    Backward time: 274.922ms (cuda)
    Maximum of managed memory: 14.66957824GB
    ---------- shufflenet_v2_x2_0 ----------
    Model Summary: 204 layers, 5.34M parameters, 5.34M gradients, 6.2 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 9.068ms (cuda)
    output: torch.Size([1, 2048, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 21.453ms (cuda)
    Backward time: 46.293ms (cuda)
    Maximum of managed memory: 7.304380416GB
    ---------- mobilenet_v2 ----------
    Model Summary: 210 layers, 2.22M parameters, 2.22M gradients, 3.3 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 6.598ms (cuda)
    output: torch.Size([1, 1280, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 24.545ms (cuda)
    Backward time: 61.306ms (cuda)
    Maximum of managed memory: 9.284091904GB
    ---------- Yolov5 ----------
    Build Yolov5 Backbone + Head
    Model Summary: 278 layers, 7.05M parameters, 7.05M gradients, 10.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 9.292ms (cuda)
    output: torch.Size([1, 512, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 20.856ms (cuda)
    Backward time: 46.989ms (cuda)
    Maximum of managed memory: 5.125439488GB
    ---------- ghostnet ----------
    Model Summary: 402 layers, 2.67M parameters, 2.67M gradients, 1.5 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 14.874ms (cuda)
    output: torch.Size([1, 960, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 22.233ms (cuda)
    Backward time: 67.145ms (cuda)
    Maximum of managed memory: 6.876561408GB
    
    #### using amp=True profiling option ###
    >> automatic mixed precision 
    
    PyTorch version 1.7.1
    CUDA version 11.0
    cuDNN version 8005
    cuDNN deterministic False
    cuDNN benchmark True
    ---------- resnet18 ----------
    Model Summary: 66 layers, 11.2M parameters, 11.2M gradients, 19.0 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 7.977ms (cuda)
    output: torch.Size([1, 512, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 12.957ms (cuda)
    Backward time: 27.266ms (cuda)
    Maximum of managed memory: 5.291114496GB
    ---------- resnet50 ----------
    Model Summary: 149 layers, 23.5M parameters, 23.5M gradients, 42.9 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 21.184ms (cuda)
    output: torch.Size([1, 2048, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 44.892ms (cuda)
    Backward time: 64.629ms (cuda)
    Maximum of managed memory: 7.778336768GB
    ---------- vgg16 ----------
    Model Summary: 32 layers, 14.7M parameters, 14.7M gradients, 160.5 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 9.048ms (cuda)
    output: torch.Size([1, 512, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 56.132ms (cuda)
    Backward time: 116.881ms (cuda)
    Maximum of managed memory: 11.794382848GB
    ---------- shufflenet_v2_x2_0 ----------
    Model Summary: 204 layers, 5.34M parameters, 5.34M gradients, 6.2 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 34.367ms (cuda)
    output: torch.Size([1, 2048, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 36.438ms (cuda)
    Backward time: 56.044ms (cuda)
    Maximum of managed memory: 5.43162368GB
    ---------- mobilenet_v2 ----------
    Model Summary: 210 layers, 2.22M parameters, 2.22M gradients, 3.3 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 14.494ms (cuda)
    output: torch.Size([1, 1280, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 31.873ms (cuda)
    Backward time: 56.273ms (cuda)
    Maximum of managed memory: 4.691329024GB
    ---------- Yolov5 ----------
    Build Yolov5 Backbone + Head
    Model Summary: 278 layers, 7.05M parameters, 7.05M gradients, 10.4 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 52.058ms (cuda)
    output: torch.Size([1, 512, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 73.105ms (cuda)
    Backward time: 94.685ms (cuda)
    Maximum of managed memory: 3.024093184GB
    ---------- ghostnet ----------
    Model Summary: 402 layers, 2.67M parameters, 2.67M gradients, 1.5 GFLOPs
    Input size: torch.Size([1, 3, 512, 512])
    benchmark warm up...
    Forward time: 63.157ms (cuda)
    output: torch.Size([1, 960, 16, 16])
    Input size: torch.Size([16, 3, 512, 512])
    benchmark warm up forward...
    benchmark warm up backward...
    run through forward pass for 100 runs...
    run through forward and backward pass for 100 runs...
    Forward time: 99.156ms (cuda)
    Backward time: 19.497ms (cuda)
    Maximum of managed memory: 3.71195904GB
    '''

if __name__ == '__main__':
    main()
