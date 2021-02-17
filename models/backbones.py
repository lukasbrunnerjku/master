import torch
from torch import nn
from torch import optim
from torchvision import models

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_backbone(model_class, pretrained=False):
    name = model_class.__name__
    if 'resnet' in name:
        model = model_class(pretrained=pretrained)
        modules = list(model.children())[:-2]
        backbone = nn.Sequential(*modules)
    elif 'vgg' in name:
        backbone = model_class(pretrained=pretrained).features
    elif ('shufflenet' in name) or ('mobilenet' in name):
        model = model_class(pretrained=pretrained)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
    else:
        raise NotImplementedError
    
    x = torch.randn(1, 3, 512, 512)
    backbone.eval()
    out = backbone(x)  # b x c x h x w
    backbone.out_channels = out.size(1)  
    return backbone

def analyze_model(model_class, n=10, benchmark=False):
    torch.manual_seed(1234)
    rng_state = torch.get_rng_state()
    
    # will select optimal algorithm depending on input size,
    # note: for varying input sizes (inference) this will 
    # eventually slow down code when kept with non default value True
    # (still can use scaling and padding to keep resolution for inference)
    if benchmark:
        # when set to True the first run will be slower since different
        # algorithms are tested, thus run here once
        torch.backends.cudnn.benchmark = True
        
    print(f'{10*"-"} {model_class.__name__} {10*"-"}')
    model = get_backbone(model_class, pretrained=False)
    
    print(f'Parameters: {count_params(model) / 10**6:0.3}M')
    assert torch.cuda.is_available()
    model.cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    model.eval()
    x = torch.randn(1, 3, 512, 512).cuda()
    with torch.no_grad():
        start.record()
        for _ in range(n):
            out = model(x)
        end.record()
    torch.cuda.synchronize()
    print(f'Elapsed forward time (inference): {start.elapsed_time(end) / n:.3f}ms')
    print(f'In/Out: {x.size()} --> {out.size()}')
    print(f'Down ratio: {x.size(-1) / out.size(-1)}')

    x = torch.randn(16, 3, 512, 512).cuda()
    out = model(x)
    y = torch.randn_like(out)
    
    model.train()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    start.record()
    for _ in range(n):
        #optimizer.zero_grad(set_to_none=True)  # version >=1.7
        for param in model.parameters():
            param.grad = None
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    end.record()
    torch.cuda.synchronize()
    print(f'Elapsed forward + backward time: {start.elapsed_time(end) / n:.3f}ms')
    
    if benchmark:  # reset to default
        torch.backends.cudnn.benchmark = False
    
    # reset RNG to have same input over test runs
    torch.set_rng_state(rng_state)
    
def dry_run(model, n=10):
    assert torch.cuda.is_available()
    model.cuda()
    model.eval()
    for _ in range(n):
        x = torch.randn(1, 3, 512, 512).cuda()
        _ = model(x)
    
def main():
    # analyze backbone characterstics of different models
    model_classes = [
        models.resnet18, 
        models.resnet50,  
        models.vgg16, 
        models.shufflenet_v2_x2_0, 
        models.mobilenet_v2,
    ]
    
    # without a dry run model performance of first model would be 
    # significantly worse and not compareable!
    dry_run(model_classes[0]())
    
    for model_class in model_classes:
        analyze_model(model_class)
    
    '''
    ---------- resnet18 ----------
    Parameters: 11.2M
    Elapsed forward time (inference): 2.346ms
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 512, 16, 16])
    Down ratio: 32.0
    Elapsed forward + backward time: 74.531ms
    ---------- resnet50 ----------
    Parameters: 23.5M
    Elapsed forward time (inference): 6.409ms
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 2048, 16, 16])
    Down ratio: 32.0
    Elapsed forward + backward time: 307.028ms
    ---------- vgg16 ----------
    Parameters: 14.7M
    Elapsed forward time (inference): 8.864ms
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 512, 16, 16])
    Down ratio: 32.0
    Elapsed forward + backward time: 483.552ms
    ---------- shufflenet_v2_x2_0 ----------
    Parameters: 5.34M
    Elapsed forward time (inference): 8.893ms
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 2048, 16, 16])
    Down ratio: 32.0
    Elapsed forward + backward time: 133.615ms
    ---------- mobilenet_v2 ----------
    Parameters: 2.22M
    Elapsed forward time (inference): 6.694ms
    In/Out: torch.Size([1, 3, 512, 512]) --> torch.Size([1, 1280, 16, 16])
    Down ratio: 32.0
    Elapsed forward + backward time: 94.504ms
    '''

if __name__ == '__main__':
    main()
