import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model = nn.Linear(10, 10).cuda(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    outputs = ddp_model(torch.randn(20, 10).cuda(rank))
    loss = outputs.sum()
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    import sys
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    main(rank, world_size)

