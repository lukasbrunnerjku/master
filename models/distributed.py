import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter 

# note: export CUDA_VISIBLE_DEVICES=0,1 to enable mutli-gpu usage

# don't create tensorboard logger in 
# if __name__ == '__main__':, must be on workers

SEED = 42

""" data sync with all_gather and similar functions
def gather_list_and_concat(list_of_nums):
    tensor = torch.Tensor(list_of_nums).cuda()
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

>>> res = gather_list_and_concat([1, 2])   # in node 1
>>> res = gather_list_and_concat([3, 4])  # in node 2
>>> res
torch.Tensor([1,2,3,4])
"""

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
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # if we do it like that loss values produced will
    # be equal for each process/gpu combination,
    # for real world example we would have to use
    # the distributed sampler to split whole data in distinct
    # subset exclusively available to a single process respectively
    
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    if rank in [1]:  # with seeds same as rank 0
        writer = SummaryWriter()

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)
        loss = loss_fn(outputs, labels)
        if rank in [1]:
            writer.add_scalar('loss', loss.item(), epoch)
        loss.backward()
        optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == '__main__':
    run_demo(demo_basic, world_size=2)
    