import os
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
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_param', type=str, default='2.8b')
parser.add_argument('--model_address', type=str, default='../mamba-pruning/models/state-spaces/mamba-2.8b')
parser.add_argument('--store_address', type=str, default='../mamba-pruning/pruned_models')
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

if type == "float32":
    dtype = torch.float32
elif type == "bfloat16":
    dtype = torch.bfloat16


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
 
def sparsegpt(input_tensor, weight_tensor, sparsity_ratio, prune_n, blocksize=128, percdamp=.01, device = "cuda:0", A=False, dtype = torch.float32, nsamples = nsamples):
    prune_m = prune_n * 2

    rows = weight_tensor.shape[0]
    columns = weight_tensor.shape[1]

    H = torch.zeros(input_tensor.shape[0], input_tensor.shape[0]).to(device)
    if torch.isnan(input_tensor).any() == True:

        print("Is there any NaN's in input_tensor?{}".format(torch.isnan(input_tensor).any()))
        print(input_tensor)
    # tmp_H = torch.zeros(input_tensor.shape[1], input_tensor.shape[1]).to(device)
    # print(H.shape)
    # print(input_tensor.shape)
    input_tensors = torch.split(input_tensor, split_size_or_sections=int(input_tensor.shape[0]/nsamples), dim=0)
    tmp = 0

    for inp in input_tensors:

        inp = inp.t()
        inp = math.sqrt(2/(tmp+1)) * inp.float()
        H *= tmp/(tmp + 1)
        tmp += 1
        H += inp.matmul(inp.t())
        
        
    # H = H.double()
    dead = torch.diag(H) == 0
    H[dead,dead] = 1
    weight_tensor[:,dead] = 0

    # Losses = torch.zeros(rows,columns)
    
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    # print("damp")
    # print(damp)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)

    if dtype == torch.bfloat16:
        Hinv = H.bfloat16()
    else:
        Hinv = H


    mask = None

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        W1 = weight_tensor[:,i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        # Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]


        if prune_n == 0:
            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1,-1))**2)
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel()*sparsity_ratio)]
                mask1 = tmp <= thresh
        else:
            mask1 = torch.zeros_like(W1) == 1

        for i in range(count):
            w = W1[:,i]
            d = Hinv1[i,i]

            if prune_n != 0 and i % prune_m == 0:
                tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            
            q = w.clone()
            if A == False:
                q[mask1[:,i]] = 0    
            else:
                q[mask1[:,i]] = 0  
                # q[mask1[:,i]] = float('-inf') 

            Q1[:, i] = q
            # Losses1[:, i] = (w - q) ** 2 / d ** 2
            q[mask1[:,i]] = 0 
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        weight_tensor[:, i1:i2] = Q1
        # print(Losses.shape)
        # print(Losses1.shape)
        # Losses += torch.sum(Losses1, 1) / 2      

        weight_tensor[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    torch.cuda.synchronize()
    del H, Hinv, mask1, mask, tmp, w, d, q, err1, Err1, Q1, W1
    gc.collect()
    torch.cuda.empty_cache()


    return weight_tensor

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(os.path.expanduser(args.model_address), device=device_num, dtype=dtype)
device = torch.device(device_num)

#c4 dataset 불러오기
print("loading calibdation data")
dataloader, _ = get_loaders("c4",nsamples=nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
print("dataset loading complete")

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
prev_residual = torch.empty(nsamples,seqlen,d_model).to(device=device)
print(prev_residual.shape)
norm_cls = RMSNorm(hidden_size=d_model, device=device, dtype=dtype)

for i in range(layer_num):
# for i in range(3):
    with torch.no_grad():
        residual = hidden_state + prev_residual

        norm_cls.weight = nn.Parameter(model_weights[norm_weights[i]]).to(device)
        normalized_hidden_input = norm_cls(hidden_state)

        # import ipdb
        # ipdb.set_trace()
        input_tensor=rearrange(normalized_hidden_input,"b l d -> (b l) d")
        weight_tensor= model_weights[mixer_in_proj_weights[i]]
        # print(model_weights[mixer_in_proj_weights[i]].shape)
        # print(model_weights[mixer_in_proj_weights[i]])
        if prune_in_proj == True:
            model_weights[mixer_in_proj_weights[i]] = sparsegpt(input_tensor, weight_tensor,sparsity_ratio=sparsity_ratio,prune_n= prune_n, device=device, dtype = dtype)
        # print(model_weights[mixer_in_proj_weights[i]].shape)
        # print(model_weights[mixer_in_proj_weights[i]])

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))

        import ipdb
        ipdb.set_trace()
        gc.collect()
        torch.cuda.empty_cache()

        xz = normalized_hidden_input @ model_weights[mixer_in_proj_weights[i]].T

        # print(xz.shape)
        # print(xz)
        x, z = xz.chunk(2, dim=2)
        # print(x.shape)
        # print(z.shape)

        # print(model_weights[mixer_conv1d_weights[i]].shape)
        # print(model_weights[mixer_conv1d_weights[i]])

        model_weights[mixer_conv1d_weights[i]] = rearrange(model_weights[mixer_conv1d_weights[i]], "b l d -> (b l) d")
        
        if prune_conv1d == True:
            model_weights[mixer_conv1d_weights[i]] = sparsegpt(input_tensor=rearrange(x, "b l d -> (b l) d"), weight_tensor=model_weights[mixer_conv1d_weights[i]].T, sparsity_ratio=sparsity_ratio, prune_n=prune_n, device=device, dtype = dtype)
            model_weights[mixer_conv1d_weights[i]] = model_weights[mixer_conv1d_weights[i]].T

        # print(model_weights[mixer_conv1d_weights[i]].shape)
        # print(model_weights[mixer_conv1d_weights[i]])

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()


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
            model_weights[mixer_x_proj_weights[i]] = sparsegpt(input_tensor=rearrange(x, "b l d -> (b l) d"), weight_tensor=model_weights[mixer_x_proj_weights[i]], sparsity_ratio=sparsity_ratio, prune_n=prune_n, device=device, dtype = dtype)

        # print(model_weights[mixer_x_proj_weights[i]].shape)
        # print(model_weights[mixer_x_proj_weights[i]])

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        x_dbl = x @ model_weights[mixer_x_proj_weights[i]].T
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)

        # print(dt.shape)
        # print(B.shape)
        # print(C.shape)

        # print(model_weights[mixer_dt_proj_weights[i]].shape)
        # print(model_weights[mixer_dt_proj_weights[i]])

        if prune_dt_proj == True:
            model_weights[mixer_dt_proj_weights[i]] = sparsegpt(input_tensor=rearrange(dt, "b l d -> (b l) d"), weight_tensor=model_weights[mixer_dt_proj_weights[i]], sparsity_ratio=sparsity_ratio, prune_n=prune_n, device=device, dtype = dtype)
        
        # print(model_weights[mixer_dt_proj_weights[i]].shape)
        # print(model_weights[mixer_dt_proj_weights[i]])
        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        dt = dt @ model_weights[mixer_dt_proj_weights[i]].T
        # print(dt.shape)

        # print(dt.shape)
        # print(B.shape)
        # print(C.shape)

        A = -torch.exp(model_weights[mixer_A_log_weights[i]].float())


        # prev_h = torch.zeros((1,1,5120,16)).to(device)
        # for a in range (dt.shape[0]):
        # # for a in range(1):
            
        #     tmp_x = dt[a, :].unsqueeze(0).bfloat16()
        #     tmp_dt = dt[a, :].unsqueeze(0).bfloat16()
        #     tmp_B = B[a, :].unsqueeze(0).bfloat16()
        #     tmp_deltaA = torch.exp(tmp_dt.unsqueeze(-1) * A).bfloat16()
        #     tmp_deltaB_u = (tmp_dt.unsqueeze(-1) * tmp_B.unsqueeze(2)) * (tmp_x.unsqueeze(-1))
        #     DN = tmp_dt.shape[2]*tmp_B.shape[2]
        #     tmp_wf = torch.zeros([tmp_dt.shape[1], DN]).bfloat16().to(device)
        #     for b in range(dt.shape[1]):  
        #     # for b in range(1):
        #         tmp_h = tmp_deltaA[:,b].unsqueeze(1) * (prev_h) + tmp_deltaB_u[:,b].unsqueeze(1)
        #         prev_h = tmp_h
        #         tmp_wf[b,:] += tmp_h.squeeze().flatten()
        #         del tmp_h
        #         gc.collect()
        #         torch.cuda.empty_cache()
        #     if a == 0:
        #         inp = math.sqrt(2) * tmp_wf.t()
        #         inp = inp.bfloat16().cpu()
        #         inpt = inp.bfloat16().T.cpu()
        #         # print(inp.device)
        #         hessian = inp @ inpt
        #     else:
        #         hessian *= (a) / (a+1)
        #         inp = math.sqrt(2/a) * tmp_wf.t()
        #         inp = inp.bfloat16().cpu()
        #         inpt = inp.bfloat16().T.cpu()
        #         hessian += inp @ inpt
        #     print(hessian.shape)
        #     print(inp.shape)
        #     # print(tmp_x.shape)
        #     # print(tmp_dt.shape)
        #     # print(tmp_B.shape)
        #     # print(tmp_deltaA.shape)
        #     # print(tmp_deltaB_u.shape)
        #     # print(tmp_h[0,0].shape)

        #     if a%20 == 0:
        #         print("{}/{}".format(a,dt.shape[0]))
        #     del tmp_x, tmp_dt, tmp_B, tmp_deltaA, tmp_deltaB_u, inp
        #     gc.collect()
        #     torch.cuda.empty_cache()

        # print(hessian.shape)
        # hessian = hessian.cuda('cuda:0')

        print("before A matrix")
        print(A.shape)
        print(A)
        print(torch.min(A))
        print(torch.max(A))

        if prune_A_log == True:
            # A = model_weights[mixer_A_log_weights[i]].bfloat16()
            A = A.bfloat16()
            dt_act = F.softplus(dt)
            # A = sparsegpt(input_tensor=rearrange(x, "b l d -> (b l) d"), weight_tensor=A.T, sparsity_ratio=sparsity_ratio, prune_n=prune_n, device=device, A=True, dtype = dtype)
            A = sparsegpt(input_tensor=rearrange(dt_act, "b l d -> (b l) d"), weight_tensor=A.T, sparsity_ratio=sparsity_ratio, prune_n=prune_n, device=device, A=True, dtype = dtype)
            A = A.T
            positive_mask = A > 0
            positive_count = positive_mask.sum().item()
            print("sum of positive num in A: {}".format(positive_count))
            # model_weights[mixer_A_log_weights[i]] = A
            eigenvalues, eigenvectors = torch.linalg.eig(A)

            print("고유값:", eigenvalues)
            print("고유벡터:", eigenvectors)
            # A = -torch.exp(A).float()
            A = A.float()
            print("pruned A")
            print(A.shape)
            print(A)
            # print(A.dtype)
            print(torch.min(A))
            print(torch.max(A))
            # model_weights[mixer_A_log_weights[i]] = torch.log(-A)
            # model_weights[mixer_A_log_weights[i]] = A
            # print("A")
            # print(model_weights[mixer_A_log_weights[i]])
        # print("A matrix")
        # print(-torch.exp(model_weights[mixer_A_log_weights[i]]))
        # print("Is there any NaN's?")
        # print(torch.isnan(model_weights[mixer_A_log_weights[i]]).any())
        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()

        dt = rearrange(dt, "b l d -> b d l", l=seqlen)
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()
        print(torch.min(dt))
        print(torch.max(dt))

        y = selective_scan_fn(
            rearrange(x, "b l d -> b d l", l = seqlen),
            dt,
            A,
            B,
            C,
            model_weights[mixer_D[i]].float(),
            z=rearrange(z, "b l d -> b d l", l = seqlen),
            delta_bias= model_weights[mixer_dt_proj_bias[i]].float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        
        # print(y.shape)
        y = rearrange(y, "b d l -> b l d")
        # print("\n")
        # print(torch.min(y))
        # print(torch.max(y))
        # print("\n")
        # print(y.shape)
        print(model_weights[mixer_out_proj_weights[i]].shape)
        print(model_weights[mixer_out_proj_weights[i]])
        if prune_out_proj == True:
            model_weights[mixer_out_proj_weights[i]] = sparsegpt(input_tensor=rearrange(y,"b l d -> (b l) d"), weight_tensor=model_weights[mixer_out_proj_weights[i]], sparsity_ratio=sparsity_ratio, prune_n=prune_n,device=device, dtype = dtype)
        print(model_weights[mixer_out_proj_weights[i]].shape)
        print(model_weights[mixer_out_proj_weights[i]])

        out = y @ model_weights[mixer_out_proj_weights[i]].T
        # print(out.shape)
        hidden_state = out
        prev_residual = residual

        # print(torch.cuda.memory_allocated(device))
        # print(torch.cuda.memory_reserved(device))
        gc.collect()
        torch.cuda.empty_cache()
        print("\n")
        print('layer {} pruning complete'.format(i))
        print("\n")

print("pruning complete")
model.save_pretrained(os.path.expanduser(args.store_address))