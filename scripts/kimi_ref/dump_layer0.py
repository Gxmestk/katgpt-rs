#!/usr/bin/env python3
"""
Dump Python reference intermediate states for layer 0 of Kimi-K3-0.40B.

This mirrors the Rust kimi_k3_layer0_dump test, allowing a step-by-step
comparison to isolate the divergence.
"""

import sys
import os
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import fla_stub

# Inject fla stubs
fla_mod = types.ModuleType("fla")
fla_modules = types.ModuleType("fla.modules")
fla_ops = types.ModuleType("fla.ops")
fla_ops_kda = types.ModuleType("fla.ops.kda")
fla_ops_utils = types.ModuleType("fla.ops.utils")
fla_ops_utils_index = types.ModuleType("fla.ops.utils.index")
fla_utils = types.ModuleType("fla.utils")

fla_modules.FusedRMSNormGated = fla_stub.FusedRMSNormGated
fla_modules.ShortConvolution = fla_stub.ShortConvolution
fla_ops_kda.chunk_kda = fla_stub.chunk_kda
fla_ops_kda.fused_recurrent_kda = fla_stub.fused_recurrent_kda
fla_ops_utils_index.prepare_cu_seqlens_from_mask = fla_stub.prepare_cu_seqlens_from_mask
fla_ops_utils_index.prepare_lens_from_mask = fla_stub.prepare_lens_from_mask

def tensor_cache(fn):
    _cache = {}
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in _cache:
            _cache[key] = fn(*args, **kwargs)
        return _cache[key]
    return wrapper

fla_utils.tensor_cache = tensor_cache
sys.modules["fla"] = fla_mod
sys.modules["fla.modules"] = fla_modules
sys.modules["fla.ops"] = fla_ops
sys.modules["fla.ops.kda"] = fla_ops_kda
sys.modules["fla.ops.utils"] = fla_ops_utils
sys.modules["fla.ops.utils.index"] = fla_ops_utils_index
sys.modules["fla.utils"] = fla_utils

import torch
import numpy as np
from safetensors.torch import load_file
import importlib.util
import importlib.machinery

MODEL_DIR = os.path.abspath(os.environ.get(
    "KIMI_K3_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "kimi-k3-0.40b"),
))
print(f"Model dir: {MODEL_DIR}")

# Load modeling code as package
MODEL_PKG = "kimi_model"
pkg_spec = importlib.machinery.ModuleSpec(MODEL_PKG, None, is_package=True)
pkg_spec.submodule_search_locations = [MODEL_DIR]
pkg_mod = importlib.util.module_from_spec(pkg_spec)
sys.modules[MODEL_PKG] = pkg_mod

for submod in ["configuration_kimi_k3", "modeling_kimi_linear"]:
    full = f"{MODEL_PKG}.{submod}"
    path = os.path.join(MODEL_DIR, f"{submod}.py")
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)

KimiLinearConfig = sys.modules[f"{MODEL_PKG}.configuration_kimi_k3"].KimiLinearConfig
KimiK3Config = sys.modules[f"{MODEL_PKG}.configuration_kimi_k3"].KimiK3Config
modeling = sys.modules[f"{MODEL_PKG}.modeling_kimi_linear"]

# Load config
config = KimiK3Config.from_pretrained(os.path.join(MODEL_DIR, "config.json"))
tc = config.text_config
tc._attn_implementation = "eager"
config._attn_implementation = "eager"

# Load weights
raw = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
sd = {}
for name, tensor in raw.items():
    if name.startswith("language_model."):
        sd[name[len("language_model."):]] = tensor
    elif name.startswith("vision_tower.") or name.startswith("mm_projector."):
        continue

# Build model
model = modeling.KimiLinearForCausalLM(tc)
model.load_state_dict(sd, strict=False)
model.eval()

# Output dir
out_dir = os.path.join(SCRIPT_DIR, "ref_output", "py_layer0")
os.makedirs(out_dir, exist_ok=True)

def save_raw(name, tensor_1d):
    """Save a 1D tensor as raw f32 LE binary."""
    arr = tensor_1d.detach().float().cpu().numpy().astype(np.float32)
    arr.tofile(os.path.join(out_dir, name))

# ── Step-by-step layer 0 forward ──────────────────────────────────────────
d = tc.hidden_size
token_id = tc.bos_token_id  # 1

# Embedding
embed = model.model.embed_tokens(torch.tensor([[token_id]]))  # [1, 1, 1024]
hidden = embed[0, 0, :]  # [1024]
save_raw("00_embed.bin", hidden)
print(f"00_embed: range [{hidden.min():.4f}, {hidden.max():.4f}]")

# Layer 0
layer0 = model.model.layers[0]
prefix_sum = hidden.clone()

# Step 1: block_residual is empty → hidden_states = prefix_sum
# (In Python, this is a no-op when block_residual.shape[1] == 0)
hidden_states = prefix_sum.clone()
save_raw("01_before_attn_res.bin", hidden_states)
print(f"01_before_attn_res: range [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

# Step 2: Block boundary (0 % 4 == 0) → push prefix_sum, set prefix_sum = None
block_residual = hidden.new_zeros(1, 0, d)  # [1, 0, 1024]
block_residual = torch.cat([block_residual, prefix_sum.view(1, 1, d)], dim=1)  # [1, 1, 1024]
prefix_sum = None

# Step 3: input_layernorm
normed = layer0.input_layernorm(hidden_states.unsqueeze(0).unsqueeze(0))  # [1, 1, 1024]
save_raw("02_after_input_layernorm.bin", normed[0, 0, :])
print(f"02_after_input_layernorm: range [{normed.min():.4f}, {normed.max():.4f}]")

# Step 3b: KDA attention
attn_out = layer0.self_attn(
    hidden_states=normed,
    attention_mask=None,
    cache_params=None,
    output_attentions=False,
    use_cache=False,
)  # [1, 1, 1024]
save_raw("03_after_kda_attn.bin", attn_out[0, 0, :])
print(f"03_after_kda_attn: range [{attn_out.min():.4f}, {attn_out.max():.4f}]")

# Step 4: prefix_sum = attn_out (since prefix_sum was None)
prefix_sum = attn_out[0, 0, :].clone()
save_raw("04_prefix_sum_after_attn.bin", prefix_sum)
print(f"04_prefix_sum_after_attn: range [{prefix_sum.min():.4f}, {prefix_sum.max():.4f}]")

# Step 5: apply_attn_res for MLP
# _apply_attn_res(prefix_sum.view(-1, d), block_residual, mlp_res_proj, mlp_res_norm)
mlp_mixed = modeling._apply_attn_res(
    prefix_sum.view(1, d),
    block_residual.view(1, 1, d),  # [1, num_blocks=1, d]
    layer0.mlp_res_proj,
    layer0.mlp_res_norm,
).view(1, 1, d)
save_raw("05_after_mlp_attn_res.bin", mlp_mixed[0, 0, :])
print(f"05_after_mlp_attn_res: range [{mlp_mixed.min():.4f}, {mlp_mixed.max():.4f}]")

# Step 6: post_attention_layernorm
normed2 = layer0.post_attention_layernorm(mlp_mixed)
save_raw("06_after_post_attn_layernorm.bin", normed2[0, 0, :])
print(f"06_after_post_attn_layernorm: range [{normed2.min():.4f}, {normed2.max():.4f}]")

# Step 6b: Dense FFN
ffn_out = layer0.mlp(normed2)  # [1, 1, 1024]
save_raw("07_after_dense_ffn.bin", ffn_out[0, 0, :])
print(f"07_after_dense_ffn: range [{ffn_out.min():.4f}, {ffn_out.max():.4f}]")

# Step 7: prefix_sum = ffn_out (since prefix_sum was None after boundary)
final = ffn_out[0, 0, :].clone()
save_raw("08_final_layer0_out.bin", final)
print(f"08_final_layer0_out: range [{final.min():.4f}, {final.max():.4f}]")

print(f"\nPython layer 0 intermediates saved to {out_dir}/")
