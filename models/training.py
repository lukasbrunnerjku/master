import tempfile
import torch
from torch import nn
from torch import optim
import sys
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import numpy as np
import math

from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter 
from torch.utils import data

### NEW ###
"""
mixed precision training

distributed parallelism = single process per GPU
instead of only multi-threading multi-GPU usage

exponentially moving average on model weights
"""

sys.path.append('./')
from utils import (torch_distributed_zero_first, ModelEMA, is_parallel,
    time_synchronized)

def setup(rank, world_size):
    # A free port on the machine that will host the process with rank 0.
    os.environ['MASTER_ADDR'] = 'localhost'
    # IP address of the machine that will host the process with rank 0.
    os.environ['MASTER_PORT'] = '12355'
    # The total number of processes, so master knows how many workers to wait for.
    os.environ['WORLD_SIZE'] = str(world_size)
    # Rank of each process, so they will know whether it is the master of a worker.
    os.environ['RANK'] = str(rank)
    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def train(epoch, model, optimizer, scaler, ema, dataloader, 
    loss_fn, writer, rank, world_size, accumulate):
    param = next(model.parameters())
    device = param.device  # torch.device
    cuda = param.is_cuda  # bool
    model.train()
    
    pbar = enumerate(dataloader)
    if rank == 0:
        pbar = tqdm(pbar, total=len(dataloader), desc='train')
        running_loss = 0.0
        running_corrects = 0
        num_samples = 0
        pf = ''  # progress bar postfix
        
    for i, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with autocast(enabled=cuda):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss *= world_size  # for DDP world_size > 1

        if rank == 0:
            mem = torch.cuda.memory_reserved() / 1E9 if cuda else 0
            pf += f'mem: {mem:.3f}GB, '
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            num_samples += inputs.size(0)
            
            pf += f'loss: {running_loss / num_samples:.4f}, ' 
            pf += f'acc: {running_corrects.double() / num_samples:.4f}, ' 
            pf += f'epoch: {epoch}'
            pbar.set_postfix_str(pf)

        scaler.scale(loss).backward()  # accumulates scaled gradients

        if (i + 1) % accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema:  # update exponentially moving average model weights
                ema.update(model)

@torch.no_grad()
def val(epoch, model, dataloader, loss_fn, writer, world_size, best_acc):
    param = next(model.parameters())
    device = param.device  # torch.device
    cuda = param.is_cuda  # bool
    model.eval()
    
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=len(dataloader), desc='val')
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0
    pf = ''  # progress bar postfix
        
    for i, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with autocast(enabled=cuda):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss *= world_size  # for DDP world_size > 1

        mem = torch.cuda.memory_reserved() / 1E9 if cuda else 0
        pf += f'mem: {mem:.3f}GB, '

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        num_samples += inputs.size(0)

        pf += f'loss: {running_loss / num_samples:.4f}, ' 
        pf += f'acc: {running_corrects.double() / num_samples:.4f}, '
        pf += f'epoch: {epoch}'
        pbar.set_postfix_str(pf)

    epoch_acc = running_corrects.double() / num_samples
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        CHECKPOINT_PATH = tempfile.gettempdir() + '/model.checkpoint'
        # All processes should see same parameters as they all start 
        # from same random parameters and gradients are synchronized
        # in backward passes. Therefore, saving it in one process is
        # sufficient.
        # model state_dict
        msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(msd, CHECKPOINT_PATH)
    
    return best_acc

def main(rank, world_size):
    # append to option in real world application
    cuda = True
    seed = 42
    best_acc = 0.0
    epochs = 5
    start_epoch = 0
    accumulate = 2
    gpus = (0, 1)
    data_dir = '/mnt/data/hymenoptera_data'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpus])
    if torch.cuda.is_available() and cuda:
        torch.cuda.manual_seed_all(seed)
        if world_size > 1:
            setup(rank, world_size)
        device = torch.device(rank)
        torch.cuda.set_device(rank)
    else:
        device = torch.device('cpu')
        cuda = False
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'Running on rank {rank}/ device {torch.cuda.current_device()}.')
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_tf)
    
    # distributed samplers will split up dataset and each process will get
    # a part of the whole data exclusively 
    sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None 
    
    # set shuffle to False if samplers are used, sampler will do that
    # if DistributedSampler has shuffle=True
    train_dl = data.DataLoader(train_ds, batch_size=4, 
        shuffle=(sampler is None), sampler=sampler, num_workers=4,
        drop_last=True, pin_memory=True, persistent_workers=False)
    
    # will do validation and testing on main process only to get 
    # statistics over the whole val/test set!
    val_dl = data.DataLoader(val_ds, batch_size=1, 
        shuffle=False, sampler=None, num_workers=4,
        drop_last=True, pin_memory=True, persistent_workers=False)
    
    class_names = train_ds.classes                                 
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    writer = SummaryWriter() if rank == 0 else None
    
    if world_size:  # batch statistics across multipe devices 
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model = model.to(device)
    
    ema = ModelEMA(model) if rank == 0 else None  # move to device before
    
    if world_size > 1:  # move to device before
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        ema.updates = start_epoch * len(train_dl) // accumulate
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler(enabled=cuda)

    since = time_synchronized()
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}, Rank: {rank}')
        
        if world_size > 1:  # so shuffle works properly in DDP mode
            sampler.set_epoch(epoch)
        
        dist.barrier()  # ??
        
        train(epoch, model, optimizer, scaler, ema, train_dl, 
            loss_fn, writer, rank, world_size, accumulate)
        
        with torch_distributed_zero_first(rank):
            best_acc = val(epoch, model, val_dl, loss_fn, writer, world_size, best_acc)
          
    if rank == 0:
        time_elapsed = time_synchronized() - since
        print(f'Training complete: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val acc: {best_acc:4f}')
    
    # "scaler": scaler.state_dict()
    # scaler.load_state_dict(checkpoint["scaler"])

    """
    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly s.t. models stored on GPU 0
    # are loaded into GPU 1 and so on
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))
    """
    if world_size > 1: 
        cleanup()

def run(main_fn, world_size):
    mp.spawn(main_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == '__main__':
    world_size = 2
    if world_size > 1:
        run(main, world_size=world_size)
    else:
        main(rank=0, world_size=1)
    