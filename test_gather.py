import torch
from accelerate import Accelerator

def allgather(tensor, accelerator):
    # Ensure tensor is on the correct device
    tensor = tensor.to(accelerator.device)
    
    # Perform the all_gather operation
    gathered_tensors = accelerator.gather(tensor)
    
    # Reshape the gathered tensors
    gathered_tensor = torch.cat(gathered_tensors, dim=0)
    
    return gathered_tensor

def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    size = accelerator.num_processes
    
    print(f"Process {rank} starting")
    
    # Create a tensor and print its initial value
    tensor = torch.tensor([rank], dtype=torch.float32).to(accelerator.device)
    print(f"Process {rank} tensor before allgather: {tensor}")
    
    # Test allgather
    gathered_tensor = allgather(tensor, accelerator)
    print(f"Process {rank} tensor after allgather: {gathered_tensor}")
    
    # Test without allgather
    gathered_tensor_noop = tensor * 2  # Simple operation for comparison
    print(f"Process {rank} tensor after noop: {gathered_tensor_noop}")
    
    print(f"Process {rank} finished")

if __name__ == "__main__":
    main()
