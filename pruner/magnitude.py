import torch

def mag(input_tensor, weight_tensor, sparsity_ratio, prune_n):
    prune_m = prune_n * 2
    result_tensor = weight_tensor

    sparsity = min(max(0.0, sparsity_ratio), 1.0)
    if sparsity == 1.0:
        result_tensor.zero_()
        return torch.zeros_like(result_tensor)
    elif sparsity == 0.0:
        return torch.ones_like(result_tensor)

    num_elements = weight_tensor.numel()

    num_zeros = round(num_elements * sparsity)
    importance = result_tensor.abs()
    threshold = torch.kthvalue(importance.reshape(-1), num_zeros)[0]
    mask = importance > threshold

    result_tensor.mul_(mask)

    return result_tensor