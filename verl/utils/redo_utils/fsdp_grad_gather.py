import torch
import torch.distributed as dist


def get_flat_grads(model):
    """Flatten and concatenate all gradients from model parameters."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads) if grads else torch.tensor([], device=next(model.parameters()).device)


def gather_full_grad(model, param_indices=None, memory_efficient=True, chunk_size=1000000):
    """
    Gather all FSDP-sharded gradients to rank 0 and return the (optionally subsampled) gradient vector (on rank 0).
    
    Args:
        model: The model to gather gradients from
        param_indices: 1D LongTensor or list of indices to select from the flattened gradient vector
        memory_efficient: If True, use a memory-efficient approach that gathers gradients in chunks
        chunk_size: Size of chunks to use when memory_efficient=True
        
    Returns:
        On rank 0: The full gradient vector
        On other ranks: None
    """
    try:
        flat_grad = get_flat_grads(model)
        if param_indices is not None:
            flat_grad = flat_grad[param_indices]
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = flat_grad.device
        
        # If only one process, just return the gradient
        if world_size == 1:
            return flat_grad
            
        # For small gradients, use the original approach
        if flat_grad.numel() < chunk_size or not memory_efficient:
            # Gather sizes from all ranks
            local_size = torch.tensor([flat_grad.numel()], dtype=torch.long, device=device)
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(sizes, local_size)
            sizes = [s.item() for s in sizes]
            max_size = max(sizes)

            # Pad local grad if needed
            if flat_grad.numel() < max_size:
                padded = torch.zeros(max_size, device=device)
                padded[:flat_grad.numel()] = flat_grad
            else:
                padded = flat_grad

            # Gather all grads
            gathered = [torch.zeros(max_size, device=device) for _ in range(world_size)]
            dist.all_gather(gathered, padded)

            if rank == 0:
                # Remove padding and concatenate
                all_grads = [g[:sizes[i]] for i, g in enumerate(gathered)]
                full_grad = torch.cat(all_grads)
                return full_grad
            else:
                return None
        else:
            # Memory-efficient approach: gather gradients in chunks
            # First, gather sizes to know how much to expect from each rank
            local_size = torch.tensor([flat_grad.numel()], dtype=torch.long, device=device)
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(sizes, local_size)
            sizes = [s.item() for s in sizes]
            
            # Only rank 0 needs to allocate memory for the full gradient
            if rank == 0:
                full_grad = torch.zeros(sum(sizes), device=device)
                offset = 0
                
                # Process each rank's gradient in chunks
                for r in range(world_size):
                    r_size = sizes[r]
                    received = 0
                    
                    while received < r_size:
                        # Determine chunk size for this iteration
                        curr_chunk_size = min(chunk_size, r_size - received)
                        
                        # Create buffer for receiving
                        chunk_buffer = torch.zeros(curr_chunk_size, device=device)
                        
                        # If processing own rank's data, copy directly
                        if r == rank:
                            chunk_buffer.copy_(flat_grad[received:received+curr_chunk_size])
                        else:
                            # Receive chunk from rank r
                            if r_size > 0:
                                dist.broadcast(chunk_buffer, src=r)
                        
                        # Add to full gradient
                        full_grad[offset+received:offset+received+curr_chunk_size] = chunk_buffer
                        received += curr_chunk_size
                    
                    offset += r_size
                
                return full_grad
            else:
                # Other ranks send their gradients in chunks
                sent = 0
                local_size = flat_grad.numel()
                
                while sent < local_size:
                    # Determine chunk size for this iteration
                    curr_chunk_size = min(chunk_size, local_size - sent)
                    
                    # Send chunk to rank 0
                    chunk_to_send = flat_grad[sent:sent+curr_chunk_size]
                    dist.broadcast(chunk_to_send, src=rank)
                    sent += curr_chunk_size
                
                return None
    except Exception as e:
        print(f"Error in gather_full_grad: {e}")
        if dist.get_rank() == 0:
            # Return a dummy tensor on error
            return torch.zeros(1, device=next(model.parameters()).device)
        else:
            return None
