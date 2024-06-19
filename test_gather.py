import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def allgather(tensor, group=None):
    if group is None:
        group = dist.group.WORLD

    world_size = dist.get_world_size(group)
    allgather_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]

    dist.all_gather(allgather_tensor, tensor, group=group)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)

    return allgather_tensor


def run(rank, size):
    print(f"Process {rank} starting")

    # Set the device for this process
    torch.cuda.set_device(rank)

    # Initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=size)

    # Create a tensor and print its initial value
    tensor = torch.tensor([rank], dtype=torch.float32).cuda()
    print(f"Process {rank} tensor before allgather: {tensor}")

    # Test allgather
    gathered_tensor = allgather(tensor)
    print(f"Process {rank} tensor after allgather: {gathered_tensor}")

    # Test without allgather
    gathered_tensor_noop = tensor * 2  # Simple operation for comparison
    print(f"Process {rank} tensor after noop: {gathered_tensor_noop}")

    # Finalize the process group
    dist.barrier()
    dist.destroy_process_group()
    print(f"Process {rank} finished")


def init_process(rank, size, fn):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)

    fn(rank, size)


if __name__ == "__main__":
    size = 2
    mp.spawn(init_process, args=(size, run), nprocs=size, join=True)
