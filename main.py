import os
from transformers import AutoTokenizer, MambaForCausalLM
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from config import load_config
import gc
from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from lib.data import get_loaders
from pruner.wanda import wanda
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
try:
    from mamba.mamba_ssm.ops.triton.layernorm import RMSNorm
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

# Set seeds for reproducibility
np.random.seed(seed)
torch.random.manual_seed(seed)

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
if model_param != "7b":
    model = MambaLMHeadModel.from_pretrained(os.path.expanduser(config.paths.model_address), device_num, dtype)
else:
    model = MambaForCausalLM.from_pretrained("tri-ml/mamba-7b-rw").bfloat16().to(device_num)
    
device = torch.device(device_num)
# Load and process c4 dataset
dataloader, _ = get_loaders("c4",nsamples=nsamples,seed=seed,seqlen=seqlen,tokenizer=tokenizer)
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

mixer_in_proj_weights = [f'backbone.layers.{i}.mixer.in_proj.weight' for i in range(layer_num)]
mixer_out_proj_weights = [f'backbone.layers.{i}.mixer.out_proj.weight' for i in range(layer_num)]
mixer_A_log_weights=[f'backbone.layers.{i}.mixer.A_log' for i in range(layer_num)]         
mixer_conv1d_weights=[f'backbone.layers.{i}.mixer.conv1d.weight' for i in range (layer_num)]
mixer_conv1d_bias=[f'backbone.layers.{i}.mixer.conv1d.bias' for i in range (layer_num)]
mixer_dt_proj_weights=[f'backbone.layers.{i}.mixer.dt_proj.weight' for i in range(layer_num)]
mixer_dt_proj_bias=[f'backbone.layers.{i}.mixer.dt_proj.bias' for i in range(layer_num)]
mixer_x_proj_weights=[f'backbone.layers.{i}.mixer.x_proj.weight' for i in range(layer_num)]
norm_weights = [f'backbone.layers.{i}.norm.weight' for i in range(layer_num)]
mixer_D = [f'backbone.layers.{i}.mixer.D' for i in range(layer_num)]
embedding_layer = nn.Embedding.from_pretrained(model_weights['backbone.embedding.weight']) if model_param != "7b" else nn.Embedding.from_pretrained(model_weights['backbone.embeddings.weight'])
embedded_input = embedding_layer(input_ids)

hidden_state = embedded_input
prev_residual = torch.empty(nsamples,seqlen,d_model).to(device=device_num)
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

        x = causal_conv1d_fn(x = rearrange(x, "b l d -> b d l"), weight = model_weights[mixer_conv1d_weights[i]], bias = model_weights[mixer_conv1d_bias[i]], activation = "silu",)
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
            return_last_state=ssm_state is not None,)
        y = rearrange(y, "b d l -> b l d")
        if prune_out_proj == True:
            model_weights[mixer_out_proj_weights[i]] = wanda(rearrange(y,"b l d -> (b l) d"), model_weights[mixer_out_proj_weights[i]], sparsity_ratio, prune_n)
        out = y @ model_weights[mixer_out_proj_weights[i]].T
        gc.collect()
        torch.cuda.empty_cache()
        
        hidden_state = out
        prev_residual = residual

        print(f'layer {i} pruning complete', i)

print("\nSaving model...")
model.save_pretrained(os.path.expanduser(config.paths.store_address))
print("\nPruned Successfully!")