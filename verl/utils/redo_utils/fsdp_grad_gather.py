import torch
import torch.distributed as dist


def get_flat_grads(model):
    """Flatten and concatenate all gradients from model parameters."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads) if grads else torch.tensor([], device=next(model.parameters()).device)


def gather_full_grad(model, param_indices=None):
    """
    Gather all FSDP-sharded gradients to rank 0 and return the (optionally subsampled) gradient vector (on rank 0).
    param_indices: 1D LongTensor or list of indices to select from the flattened gradient vector.
    """
    flat_grad = get_flat_grads(model)
    if param_indices is not None:
        flat_grad = flat_grad[param_indices]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = flat_grad.device

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
