import torch
import numpy as np

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def autopad(k, p=None):
    if p is None:  # pad s.t. same spatial shape after convolution
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
