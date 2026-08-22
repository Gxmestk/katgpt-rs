#!/usr/bin/env python3
"""
Dump KDA internal states for layer 0 of Kimi-K3-0.40B.

Hooks into the KDA forward to dump:
- q_proj, k_proj, v_proj outputs (before shortconv)
- After shortconv + silu
- g_raw (kernel gate), beta
- The KDA kernel output (before output norm + o_proj)
- After output norm + o_proj
"""

import sys
import os
import types

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
            _cache[key] = fn(*args, fn.__name__)
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
from einops import rearrange
from safetensors.torch import load_file
import importlib.util
import importlib.machinery

MODEL_DIR = os.path.abspath(os.environ.get(
    "KIMI_K3_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "kimi-k3-0.40b"),
))

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
sd = {}
for name, tensor in raw.items():
    if name.startswith("language_model."):
        sd[name[len("language_model."):]] = tensor
    elif name.startswith("vision_tower.") or name.startswith("mm_projector."):
        continue

model = modeling.KimiLinearForCausalLM(tc)
model.load_state_dict(sd, strict=False)
model.eval()

out_dir = os.path.join(SCRIPT_DIR, "ref_output", "py_kda_internals")
os.makedirs(out_dir, exist_ok=True)

def save_raw(name, tensor):
    if tensor.dim() > 1:
        tensor = tensor.reshape(-1)
    arr = tensor.detach().float().cpu().numpy().astype(np.float32)
    arr.tofile(os.path.join(out_dir, name))

with torch.no_grad():
    d = tc.hidden_size
    token_id = tc.bos_token_id

    embed = model.model.embed_tokens(torch.tensor([[token_id]]))
    hidden = embed[0, 0, :]

    layer0 = model.model.layers[0]
    normed = layer0.input_layernorm(hidden.view(1, 1, d))
    h = normed[0, 0, :]  # [1024] — the KDA input

    kda = layer0.self_attn  # KimiDeltaAttention

    # ── Manually run the KDA forward to capture intermediates ───────────
    batch_size, q_len, _ = h.view(1, 1, d).shape

    q_proj_states = kda.q_proj(h.view(1, 1, d))  # [1, 1, 256]
    k_proj_states = kda.k_proj(h.view(1, 1, d))
    v_proj_states = kda.v_proj(h.view(1, 1, d))

    save_raw("kda_00_q_proj.bin", q_proj_states)
    save_raw("kda_00_k_proj.bin", k_proj_states)
    save_raw("kda_00_v_proj.bin", v_proj_states)

    # ShortConv (B=1, T=1, D=256 → step path with zero cache)
    q_conv, _ = kda.q_conv1d(x=q_proj_states, cache=None, output_final_state=False)
    k_conv, _ = kda.k_conv1d(x=k_proj_states, cache=None, output_final_state=False)
    v_conv, _ = kda.v_conv1d(x=v_proj_states, cache=None, output_final_state=False)

    save_raw("kda_01_q_conv.bin", q_conv)
    save_raw("kda_01_k_conv.bin", k_conv)
    save_raw("kda_01_v_conv.bin", v_conv)

    # g_raw (kernel gate)
    g = kda.f_b_proj(kda.f_a_proj(h.view(1, 1, d)))  # [1, 1, 256]
    save_raw("kda_02_g_raw.bin", g)

    # beta
    beta = kda.b_proj(h.view(1, 1, d)).float()  # [1, 1, 8]
    save_raw("kda_03_beta_raw.bin", beta)

    # Rearrange to head format
    head_dim = kda.head_dim  # 32
    head_k_dim = kda.head_k_dim  # 32
    num_heads = kda.num_heads  # 8

    q = rearrange(q_conv, "... (h d) -> ... h d", d=head_k_dim)
    k = rearrange(k_conv, "... (h d) -> ... h d", d=head_k_dim)
    v = rearrange(v_conv, "... (h d) -> ... h d", d=head_dim)
    g_heads = rearrange(g, "... (h d) -> ... h d", d=head_dim)

    # Run KDA kernel (chunk mode since no cache)
    o, _ = fla_stub.chunk_kda(
        q=q, k=k, v=v, g=g_heads, beta=beta,
        A_log=kda.A_log,
        dt_bias=kda.dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=kda.gate_lower_bound,
        transpose_state_layout=True,
    )
    # o: [1, 1, 8, 32]

    save_raw("kda_04_kernel_out.bin", o)

    # Output gate (full rank)
    g_out = kda.g_proj(h.view(1, 1, d))  # [1, 1, 256]
    g_out_heads = rearrange(g_out, "... (h d) -> ... h d", d=head_dim)
    save_raw("kda_05_g_out.bin", g_out)

    # FusedRMSNormGated
    o_normed = kda.o_norm(o, g_out_heads)
    save_raw("kda_06_after_onorm.bin", o_normed)

    # o_proj
    o_flat = rearrange(o_normed, "b t h d -> b t (h d)")
    final = kda.o_proj(o_flat)  # [1, 1, 1024]
    save_raw("kda_07_after_oproj.bin", final)

    print("KDA internals saved.")
    for f in sorted(os.listdir(out_dir)):
        arr = np.fromfile(os.path.join(out_dir, f), dtype=np.float32)
        print(f"  {f:35s}: [{arr.min():.6f}, {arr.max():.6f}], mean={arr.mean():.6f}")
