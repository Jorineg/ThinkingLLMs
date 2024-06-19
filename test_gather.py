import torch
from accelerate import Accelerator

def allgather(tensor, group=None):
    """smantic sugar for torch.distributed.all_gather.

    Args:
        tensor: (bs, ...)
        group:

    Returns:
        All gathered tensor (world_size, bs, ...)
    """
    if group is None:
        group = torch.distributed.group.WORLD

    allgather_tensor = [torch.zeros_like(tensor) for _ in range(group.size())]
    torch.distributed.all_gather(allgather_tensor, tensor, group=group)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)
    return allgather_tensor


def main():
    accelerator = Accelerator()
    rank = accelerator.process_index

    print(f"Process {rank} starting")
    
    # Create a tensor and print its initial value
    tensor = torch.tensor([rank], dtype=torch.float32).to(accelerator.device)
    print(f"Process {rank} tensor before allgather: {tensor}")
    
    # Test allgather
    gathered_tensor = allgather(tensor)  # (world_size, bs, ...)
    print(f"Process {rank} tensor after allgather: {gathered_tensor}")
    
    # Test without allgather
    gathered_tensor_noop = tensor * 2  # Simple operation for comparison
    print(f"Process {rank} tensor after noop: {gathered_tensor_noop}")
    
    print(f"Process {rank} finished")

if __name__ == "__main__":
    main()
