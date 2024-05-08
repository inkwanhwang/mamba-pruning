import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from io import BytesIO
from datasets import load_dataset
import random
import easydict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
import gc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_param', type=str, default='130m')
parser.add_argument('--model_address', type=str, default='./mamba_ssm/models/state-spaces_mamba-130m/')
parser.add_argument('--store_address', type=str, default='./mamba_ssm/models/test/')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--nsamples',type=int, default=64)
parser.add_argument('--seqlen', type=int, default=2048)
parser.add_argument('--prune_in_proj', type=bool, default= True)
parser.add_argument('--prune_conv1d',type= bool, default= True)
parser.add_argument('--prune_x_proj',type= bool, default= True)
parser.add_argument('--prune_dt_proj',type= bool, default= True)
parser.add_argument('--prune_A_log',type= bool, default= False)
parser.add_argument('--prune_out_proj',type= bool, default= True)
parser.add_argument('--sparsity_ratio',type=float, default= 0.5)
parser.add_argument('--device_num', type= str, default="cuda:0")
parser.add_argument('--type', type=str, default="float32")
parser.add_argument('--prune_n', type=int, default=0)

args = parser.parse_args()

model_param = args.model_param
prune_in_proj = args.prune_in_proj
prune_conv1d = args.prune_conv1d
prune_x_proj = args.prune_x_proj
prune_dt_proj = args.prune_dt_proj
prune_A_log = args.prune_A_log
prune_out_proj = args.prune_out_proj
seed = args.seed
nsamples = args.nsamples
seqlen = args.seqlen
ssm_state = None
sparsity_ratio = args.sparsity_ratio
device_num = args.device_num
prune_n = args.prune_n
type = args.type
import ipdb
ipdb.set_trace()
if type == "float32":
    dtype = torch.float32
elif type == "bfloat16":
    dtype = torch.bfloat16


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = torch.LongTensor([])
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader = torch.cat((trainloader, inp), dim=0)

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)

# We will use n:m wanda pruning
def wanda(input_tensor, weight_tensor, sparsity_ratio, prune_n):
    prune_m = prune_n * 2
    result_tensor = weight_tensor
    l2_norm_tensor = torch.norm(input_tensor, p=2, dim=0) / input_tensor.shape[0]
    l2_norm_tensor = l2_norm_tensor.unsqueeze(1).expand_as(weight_tensor.T)
    wanda_tensor = torch.abs(weight_tensor) * l2_norm_tensor.T

    if prune_n != 0:
        weight_mask = torch.zeros_like(wanda_tensor, dtype = torch.bool)
        for ii in range(wanda_tensor.shape[1]):
            if ii % prune_m == 0:
                tmp = wanda_tensor[:,ii:(ii+4)].float()
                weight_mask.scatter_(1,ii+torch.topk(tmp, 2, dim=1, largest= False)[1], True) #smallest in tmp만 출력한다! [0]은 숫자 [1]은 위치를 나타낸다!
    else:
        thresh = torch.sort(wanda_tensor.flatten().cuda())[0][int(wanda_tensor.numel()*sparsity_ratio)].cpu()
        weight_mask = (wanda_tensor<=thresh)
    

    result_tensor[weight_mask] = 0
    # Garbage collect & Cache memory manage
    gc.collect()
    torch.cuda.empty_cache()
    return result_tensor

def wanda_A(l2_norm_tensor, weight_tensor, sparsity_ratio, prune_n):
    prune_m = prune_n * 2
    result_tensor = weight_tensor
    l2_norm_tensor = l2_norm_tensor ** (1/2)
    wanda_tensor = torch.abs(weight_tensor) * l2_norm_tensor

    if prune_n != 0:
        weight_mask = torch.zeros_like(wanda_tensor, dtype = torch.bool)
        for ii in range(wanda_tensor.shape[1]):
            if ii % prune_m == 0:
                tmp = wanda_tensor[:,ii:(ii+4)].float()
                weight_mask.scatter_(1,ii+torch.topk(tmp, 2, dim=1, largest= False)[1], True) #smallest in tmp만 출력한다! [0]은 숫자 [1]은 위치를 나타낸다!
    else:
        thresh = torch.sort(wanda_tensor.flatten().cuda())[0][int(wanda_tensor.numel()*sparsity_ratio)].cpu()
        weight_mask = (wanda_tensor<=thresh)
    

    result_tensor[weight_mask] = 0

    # gc.collect()
    # torch.cuda.empty_cache()
    return result_tensor

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

def mag_A(l2_norm_tensor, weight_tensor, sparsity_ratio, prune_n):
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

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        print(x)
        print(x.shape)
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(os.path.expanduser(args.model_address), device=device_num, dtype=dtype)
device = torch.device(device_num)

#c4 dataset 불러오기
print("loading calibdation data")
dataloader, _ = get_loaders("c4",nsamples=nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
print("dataset loading complete")

dataloader1, dataloader2 = dataloader.chunk(2, dim=0)

# print(dataloader.shape)

input_ids = dataloader.to(device)
model_weights = model.state_dict()

if model_param == "2.8b":
    layer_num = 64
    d_model = 2560
elif model_param == "130m":
    layer_num = 24
    d_model = 768
elif model_param == "370m":
    layer_num = 48
    d_model = 1024
elif model_param == "790m":
    layer_num = 48
    d_model = 1536
elif model_param == "1.4b":
    layer_num = 48
    d_model = 2048

d_state = 16
dt_rank = model_weights['backbone.layers.0.mixer.x_proj.weight'].shape[0] - d_state*2
d_inner = d_model * 2

mixer_in_proj_weights = ['backbone.layers.{0}.mixer.in_proj.weight'.format(i) for i in range(layer_num)]
mixer_out_proj_weights = ['backbone.layers.{0}.mixer.out_proj.weight'.format(i) for i in range(layer_num)]
mixer_A_log_weights=['backbone.layers.{0}.mixer.A_log'.format(i) for i in range(layer_num)]         
mixer_conv1d_weights=['backbone.layers.{0}.mixer.conv1d.weight'.format(i) for i in range (layer_num)]
mixer_conv1d_bias=['backbone.layers.{0}.mixer.conv1d.bias'.format(i) for i in range (layer_num)]
mixer_dt_proj_weights=['backbone.layers.{0}.mixer.dt_proj.weight'.format(i) for i in range(layer_num)]
mixer_dt_proj_bias=['backbone.layers.{0}.mixer.dt_proj.bias'.format(i) for i in range(layer_num)]
mixer_x_proj_weights=['backbone.layers.{0}.mixer.x_proj.weight'.format(i) for i in range(layer_num)]
norm_weights = ['backbone.layers.{0}.norm.weight'.format(i) for i in range(layer_num)]
mixer_D = ['backbone.layers.{0}.mixer.D'.format(i) for i in range(layer_num)]

embedding_layer = nn.Embedding.from_pretrained(model_weights['backbone.embedding.weight'])
embedded_input = embedding_layer(input_ids)

hidden_state = embedded_input
prev_residual = torch.empty(nsamples,seqlen,d_model).to(device=device_num)
# print(prev_residual.shape)
norm_cls = RMSNorm(hidden_size=d_model, device=device, dtype=dtype)

for i in range(layer_num):
# for i in range(1):
    with torch.no_grad():
        residual = hidden_state + prev_residual

        norm_cls.weight = nn.Parameter(model_weights[norm_weights[i]])
        normalized_hidden_input = norm_cls(hidden_state)

        # print(model_weights[mixer_in_proj_weights[i]].shape)
        # print(model_weights[mixer_in_proj_weights[i]])
        if prune_in_proj == True:
            model_weights[mixer_in_proj_weights[i]] = wanda(rearrange(normalized_hidden_input,"b l d -> (b l) d"), model_weights[mixer_in_proj_weights[i]],sparsity_ratio, prune_n)
        # print(wanda_in_proj.shape)
        # print(wanda_in_proj)

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        xz = normalized_hidden_input @ model_weights[mixer_in_proj_weights[i]].T

        # print(xz.shape)
        # print(xz)
        x, z = xz.chunk(2, dim=2)
        # print(x.shape)
        # print(z.shape)

        # print(model_weights[mixer_conv1d_weights[i]].shape)
        # print(model_weights[mixer_conv1d_bias[i]].shape)   

        model_weights[mixer_conv1d_weights[i]] = rearrange(model_weights[mixer_conv1d_weights[i]], "b l d -> (b l) d")
        
        # print(tmp_conv1d_weights.shape)
        # print(tmp_conv1d_weights)
        if prune_conv1d == True:
            model_weights[mixer_conv1d_weights[i]] = wanda(rearrange(x, "b l d -> (b l) d"), model_weights[mixer_conv1d_weights[i]].T, sparsity_ratio, prune_n)
            model_weights[mixer_conv1d_weights[i]] = model_weights[mixer_conv1d_weights[i]].T

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        # print(wanda_conv1d.shape)
        # print(wanda_conv1d)

        x = causal_conv1d_fn(
            x=rearrange(x, "b l d -> b d l"),
            weight=model_weights[mixer_conv1d_weights[i]],
            bias=model_weights[mixer_conv1d_bias[i]],
            activation="silu",
        )

        # print(x.shape)
        x = rearrange(x, "b d l -> b l d")

        # print(model_weights[mixer_x_proj_weights[i]].shape)
        # print(model_weights[mixer_x_proj_weights[i]])
        if prune_x_proj == True:
            model_weights[mixer_x_proj_weights[i]] = wanda(rearrange(x, "b l d -> (b l) d"), model_weights[mixer_x_proj_weights[i]], sparsity_ratio, prune_n)

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()
        # print(wanda_x_proj.shape)
        # print(wanda_x_proj)

        x_dbl = x @ model_weights[mixer_x_proj_weights[i]].T
        
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)

        # print(dt.shape)
        # print(B.shape)
        # print(C.shape)

        # print(model_weights[mixer_dt_proj_weights[i]].shape)
        # print(model_weights[mixer_dt_proj_weights[i]])
        if prune_dt_proj == True:
            model_weights[mixer_dt_proj_weights[i]] = wanda(rearrange(dt, "b l d -> (b l) d"), model_weights[mixer_dt_proj_weights[i]], sparsity_ratio, prune_n)
        
        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        # print(wanda_dt_proj.shape)
        # print(wanda_dt_proj)
        dt = dt @ model_weights[mixer_dt_proj_weights[i]].T
        # print(dt.shape)

        A = -torch.exp(model_weights[mixer_A_log_weights[i]].float())
        # debug #

        prev_h = torch.zeros((d_model * 2,16)).to(device)
        for a in range (dt.shape[0]):

            # for b in range(1):
            tmp_x = dt[a, :].unsqueeze(0)
            tmp_dt = dt[a, :].unsqueeze(0)
            tmp_B = B[a, :].unsqueeze(0)
            tmp_deltaA = torch.exp(tmp_dt.unsqueeze(-1) * A)
            tmp_deltaB_u = (tmp_dt.unsqueeze(-1) * tmp_B.unsqueeze(2)) * (tmp_x.unsqueeze(-1))
            tmp_h = tmp_deltaA * (prev_h.unsqueeze(0)) + tmp_deltaB_u
            
            if a == 0:
                h = torch.sum(tmp_h**2,dim=(0,1))
            else:
                h += torch.sum(tmp_h**2,dim=(0,1))
            # print(tmp_x.shape)
            # print(tmp_dt.shape)
            # print(tmp_B.shape)
            # print(tmp_deltaA.shape)
            # print(tmp_deltaB_u.shape)
            # print(tmp_h.shape)

            if a%20 == 0:
                print("{}/{}".format(a,dt.shape[0]))
            del tmp_x, tmp_dt, tmp_B, tmp_deltaA, tmp_deltaB_u, tmp_h
            gc.collect()
            torch.cuda.empty_cache()

        print(h.shape)




        dt = rearrange(dt, "b l d -> b d l", l=seqlen)
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()

        # print(dt.shape)
        # print(B.shape)
        # print(C.shape)


        print(A.shape)
        print(A)
        if prune_A_log == True:
            A = wanda_A(h, A, sparsity_ratio, prune_n)
            print(A.shape)
            print(A)
            model_weights[mixer_A_log_weights[i]] = torch.log(-A)
            # print(model_weights[mixer_A_log_weights[i]])


        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        y = selective_scan_fn(
            rearrange(x, "b l d -> b d l", l = seqlen),
            dt,
            A,
            B,
            C,
            model_weights[mixer_D[i]],
            z=rearrange(z, "b l d -> b d l", l = seqlen),
            delta_bias= model_weights[mixer_dt_proj_bias[i]].float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        
        # print(y.shape)
        y = rearrange(y, "b d l -> b l d")
        # print(y.shape)
        # print(model_weights[mixer_out_proj_weights[i]].shape)
        # print(model_weights[mixer_out_proj_weights[i]])
        if prune_out_proj == True:
            model_weights[mixer_out_proj_weights[i]] = wanda(rearrange(y,"b l d -> (b l) d"), model_weights[mixer_out_proj_weights[i]], sparsity_ratio, prune_n)
        # print(wanda_out_proj.shape)
        # print(wanda_out_proj)

        out = y @ model_weights[mixer_out_proj_weights[i]].T
        # print(out.shape)
        hidden_state = out
        prev_residual = residual

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        print('layer {} pruning complete'.format(i))


model.save_pretrained(os.path.expanduser(args.store_address))