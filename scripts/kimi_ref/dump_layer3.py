#!/usr/bin/env python3
"""Dump layer 3 intermediates for MLA divergence isolation."""

import sys, os, types
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fla_stub

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

import torch, numpy as np
from safetensors.torch import load_file
import importlib.util, importlib.machinery

MODEL_DIR = os.path.abspath(os.environ.get("KIMI_K3_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "kimi-k3-0.40b")))

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
KimiK3Config = sys.modules[f"{MODEL_PKG}.configuration_kimi_k3"].KimiK3Config
modeling = sys.modules[f"{MODEL_PKG}.modeling_kimi_linear"]

config = KimiK3Config.from_pretrained(os.path.join(MODEL_DIR, "config.json"))
tc = config.text_config
tc._attn_implementation = "eager"
config._attn_implementation = "eager"
raw = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
sd = {name[len("language_model."):]: t for name, t in raw.items() if name.startswith("language_model.")}
model = modeling.KimiLinearForCausalLM(tc)
model.load_state_dict(sd, strict=False)
model.eval()

out_dir = os.path.join(SCRIPT_DIR, "ref_output", "py_layer3")
os.makedirs(out_dir, exist_ok=True)
def save_raw(name, t):
    if t.dim() > 1: t = t.reshape(-1)
    t.detach().float().cpu().numpy().astype(np.float32).tofile(os.path.join(out_dir, name))

with torch.no_grad():
    d = tc.hidden_size
    token_id = tc.bos_token_id
    embed = model.model.embed_tokens(torch.tensor([[token_id]]))

    # Run layers 0-2 first (KDA layers — these now match)
    hidden = embed
    block_residual = hidden.new_zeros(1 * 1, 0, d)

    for idx in range(3):
        layer = model.model.layers[idx]
        hidden, block_residual = layer(
            hidden, attention_mask=None, past_key_values=None,
            output_attentions=False, use_cache=False, block_residual=block_residual)

    # Now at layer 3 (MLA)
    layer3 = model.model.layers[3]
    prefix_sum = hidden[0, 0, :].clone()

    # Before attn-res
    save_raw("l3_00_prefix_sum.bin", prefix_sum)

    # Apply attn-res (block_residual has 1 entry from layer 0)
    hidden_states = modeling._apply_attn_res(
        prefix_sum.view(1, d),
        block_residual.view(1, block_residual.shape[1], d),
        layer3.self_attention_res_proj,
        layer3.self_attention_res_norm,
    ).view(1, 1, d)
    save_raw("l3_01_after_self_attn_res.bin", hidden_states[0, 0, :])

    # Block boundary: 3 % 4 != 0 → no push

    # Input layernorm
    normed = layer3.input_layernorm(hidden_states)
    save_raw("l3_02_after_input_layernorm.bin", normed[0, 0, :])

    # MLA attention
    attn_out = layer3.self_attn(
        hidden_states=normed, attention_mask=None, position_ids=None,
        past_key_values=None, output_attentions=False, use_cache=False)
    save_raw("l3_03_after_mla_attn.bin", attn_out[0, 0, :])

    # prefix_sum += attn_out
    prefix_sum = prefix_sum + attn_out[0, 0, :]
    save_raw("l3_04_prefix_sum_after_attn.bin", prefix_sum)

    # MLP attn-res
    mlp_mixed = modeling._apply_attn_res(
        prefix_sum.view(1, d),
        block_residual.view(1, block_residual.shape[1], d),
        layer3.mlp_res_proj,
        layer3.mlp_res_norm,
    ).view(1, 1, d)
    save_raw("l3_05_after_mlp_attn_res.bin", mlp_mixed[0, 0, :])

    # Post-attention layernorm
    normed2 = layer3.post_attention_layernorm(mlp_mixed)
    save_raw("l3_06_after_post_attn_layernorm.bin", normed2[0, 0, :])

    # MoE FFN
    ffn_out = layer3.block_sparse_moe(normed2)
    save_raw("l3_07_after_moe_ffn.bin", ffn_out[0, 0, :])

    # Final: prefix_sum += ffn_out
    final = prefix_sum + ffn_out[0, 0, :]
    save_raw("l3_08_final_layer3_out.bin", final)

    print("Python layer 3 intermediates saved.")
    for f in sorted(os.listdir(out_dir)):
        arr = np.fromfile(os.path.join(out_dir, f), dtype=np.float32)
        print(f"  {f:40s}: [{arr.min():.4f}, {arr.max():.4f}], mean={arr.mean():.4f}")
