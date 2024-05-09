import torch
import gc

def wanda(input_tensor, weight_tensor, sparsity_ratio, prune_n):
    prune_m = prune_n * 2
    result_tensor = weight_tensor
    l2_norm_tensor = torch.norm(input_tensor, p=2, dim=0) / input_tensor.shape[0] 
    l2_norm_tensor = l2_norm_tensor.unsqueeze(1).expand_as(weight_tensor.T)
    wanda_tensor = torch.abs(weight_tensor) * l2_norm_tensor.T

    if prune_n != 0:
        weight_mask = (torch.zeros_like(wanda_tensor)==1)
        for ii in range(wanda_tensor.shape[1]):
            if ii % prune_m == 0:
                tmp = wanda_tensor[:,ii:(ii+4)].float()
                weight_mask.scatter_(1,ii+torch.topk(tmp, 2, dim=1, largest= False)[1], True) #smallest in tmp만 출력한다! [0]은 숫자 [1]은 위치를 나타낸다!
    else:
        thresh = torch.sort(wanda_tensor.flatten().cuda())[0][int(wanda_tensor.numel()*sparsity_ratio)].cpu()
        weight_mask = (wanda_tensor<=thresh)
    
    result_tensor[weight_mask] = 0
    gc.collect()
    torch.cuda.empty_cache()
    return result_tensor