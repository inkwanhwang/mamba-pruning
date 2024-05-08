import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, MambaForCausalLM
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from datasets import load_dataset
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from config import load_config
import gc
from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# Argument parsing using YAML
config = load_config(conf_url = 'config.yaml')

model_param = config.params.model_param
nsamples = config.params.nsamples
seqlen = config.params.seqlen
sparsity_ratio = config.params.sparsity_ratio
dtype = config.params.dtype
seed = config.params.seed
device_num = config.params.device_num

prune_in_proj = config.prune.prune_in_proj
prune_conv1d = config.prune.prune_conv1d
prune_x_proj = config.prune.prune_x_proj
prune_dt_proj = config.prune.prune_dt_proj
prune_A_log = config.prune.prune_A_log
prune_out_proj = config.prune.prune_out_proj
ssm_state = config.prune.ssm_state
prune_n = config.prune.prune_n

if dtype == "float32":
    dtype = torch.float32
elif dtype == "bfloat16":
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
    set_seed(seed)
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
    
def wanda(input_tensor, weight_tensor, sparsity_ratio, prune_n, A = False):
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

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
if model_param != "7b":
    model = MambaLMHeadModel.from_pretrained(os.path.expanduser(config.paths.model_address), device=device_num, dtype=dtype)
else:
    model = MambaForCausalLM.from_pretrained("tri-ml/mamba-7b-rw").bfloat16().to(device_num)
    
device = torch.device(device_num)

# c4 dataset
print("loading calibdation data")
dataloader, _ = get_loaders("c4",nsamples=nsamples,seed=seed,seqlen=seqlen,tokenizer=tokenizer)
print("dataset loading complete")
input_ids = dataloader.to(device)
model_weights = model.state_dict()

if model_param == "130m":
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
elif model_param == "2.8b":
    layer_num = 64
    d_model = 2560
elif model_param == "7b":
    layer_num = 64
    d_model = 4096

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

if model_param != "7b":
    embedding_layer = nn.Embedding.from_pretrained(model_weights['backbone.embedding.weight'])
else:
    embedding_layer = nn.Embedding.from_pretrained(model_weights['backbone.embeddings.weight'])
embedded_input = embedding_layer(input_ids)

hidden_state = embedded_input
prev_residual = torch.empty(nsamples,seqlen,d_model).to(device=device_num)
print(prev_residual.shape)
norm_cls = RMSNorm(hidden_size=d_model, device=device, dtype=dtype)

for i in range(layer_num):
    with torch.no_grad():
        residual = hidden_state + prev_residual

        norm_cls.weight = nn.Parameter(model_weights[norm_weights[i]])
        normalized_hidden_input = norm_cls(hidden_state)

        if prune_in_proj == True:
            model_weights[mixer_in_proj_weights[i]] = wanda(rearrange(normalized_hidden_input,"b l d -> (b l) d"), model_weights[mixer_in_proj_weights[i]],sparsity_ratio, prune_n)

        gc.collect()
        torch.cuda.empty_cache()

        xz = normalized_hidden_input @ model_weights[mixer_in_proj_weights[i]].T
        x, z = xz.chunk(2, dim=2)

        model_weights[mixer_conv1d_weights[i]] = rearrange(model_weights[mixer_conv1d_weights[i]], "b l d -> (b l) d")
        
        if prune_conv1d == True:
            model_weights[mixer_conv1d_weights[i]] = wanda(rearrange(x, "b l d -> (b l) d"), model_weights[mixer_conv1d_weights[i]].T, sparsity_ratio, prune_n)
            model_weights[mixer_conv1d_weights[i]] = model_weights[mixer_conv1d_weights[i]].T

        gc.collect()
        torch.cuda.empty_cache()

        x = causal_conv1d_fn(
            x=rearrange(x, "b l d -> b d l"),
            weight=model_weights[mixer_conv1d_weights[i]],
            bias=model_weights[mixer_conv1d_bias[i]],
            activation="silu",
        )
        x = rearrange(x, "b d l -> b l d")
        if prune_x_proj == True:
            model_weights[mixer_x_proj_weights[i]] = wanda(rearrange(x, "b l d -> (b l) d"), model_weights[mixer_x_proj_weights[i]], sparsity_ratio, prune_n)
        gc.collect()
        torch.cuda.empty_cache()

        x_dbl = x @ model_weights[mixer_x_proj_weights[i]].T
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)

        if prune_dt_proj == True:
            model_weights[mixer_dt_proj_weights[i]] = wanda(rearrange(dt, "b l d -> (b l) d"), model_weights[mixer_dt_proj_weights[i]], sparsity_ratio, prune_n)
        
        gc.collect()
        torch.cuda.empty_cache()

        dt = dt @ model_weights[mixer_dt_proj_weights[i]].T
        A = -torch.exp(model_weights[mixer_A_log_weights[i]].float())

        if prune_A_log == True:
            dt_act = F.softplus(dt)
            A = wanda(rearrange(dt_act, "b l d -> (b l) d"), A.T, sparsity_ratio, prune_n, A=True)
            A = A.T
            model_weights[mixer_A_log_weights[i]] = torch.log(-A)
        dt = rearrange(dt, "b l d -> b d l", l=seqlen)
        B = rearrange(B, "b l dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "b l dstate -> b dstate l", l=seqlen).contiguous()

        gc.collect()
        torch.cuda.empty_cache()

        y, _ = selective_scan_fn(
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
        y = rearrange(y, "b d l -> b l d")
        if prune_out_proj == True:
            model_weights[mixer_out_proj_weights[i]] = wanda(rearrange(y,"b l d -> (b l) d"), model_weights[mixer_out_proj_weights[i]], sparsity_ratio, prune_n)
        out = y @ model_weights[mixer_out_proj_weights[i]].T
        hidden_state = out
        prev_residual = residual

        gc.collect()
        torch.cuda.empty_cache()
        
        print('layer {} pruning complete'.format(i))

print("\nSaving model...")
model.save_pretrained(os.path.expanduser(config.paths.store_address))