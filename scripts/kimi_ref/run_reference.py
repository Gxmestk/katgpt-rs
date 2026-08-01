#!/usr/bin/env python3
"""
Kimi-K3-0.40B reference logits generator for G1 correctness gate.

This script loads the actual Kimi-K3 model using its own modeling code,
but replaces fla's triton-dependent operations with pure-PyTorch equivalents
(see fla_stub.py). The result is reference logits that can be compared against
the Rust katgpt-rs forward pass.

Usage:
    /path/to/python run_reference.py

Output:
    ref_logits_bos.npy  — logits for BOS token (id=1), shape [vocab_size]
    ref_hidden_bos.npy  — final hidden state before lm_head, shape [hidden_size]
"""

import sys
import os
import types

# ── Step 1: Build fake fla module tree from fla_stub ──────────────────────────
# The modeling code imports:
#   from fla.modules import FusedRMSNormGated, ShortConvolution
#   from fla.ops.kda import chunk_kda, fused_recurrent_kda
#   from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
#   from fla.utils import tensor_cache
# We inject our pure-PyTorch replacements BEFORE importing the modeling code.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import fla_stub

# Build the fake module tree
fla_mod = types.ModuleType("fla")
fla_modules = types.ModuleType("fla.modules")
fla_ops = types.ModuleType("fla.ops")
fla_ops_kda = types.ModuleType("fla.ops.kda")
fla_ops_utils = types.ModuleType("fla.ops.utils")
fla_ops_utils_index = types.ModuleType("fla.ops.utils.index")
fla_utils = types.ModuleType("fla.utils")

# Populate modules
fla_modules.FusedRMSNormGated = fla_stub.FusedRMSNormGated
fla_modules.ShortConvolution = fla_stub.ShortConvolution
fla_ops_kda.chunk_kda = fla_stub.chunk_kda
fla_ops_kda.fused_recurrent_kda = fla_stub.fused_recurrent_kda
fla_ops_utils_index.prepare_cu_seqlens_from_mask = fla_stub.prepare_cu_seqlens_from_mask
fla_ops_utils_index.prepare_lens_from_mask = fla_stub.prepare_lens_from_mask

def tensor_cache(fn):
    """Simple memoization decorator (fla.utils.tensor_cache)."""
    _cache = {}
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in _cache:
            _cache[key] = fn(*args, **kwargs)
        return _cache[key]
    return wrapper

fla_utils.tensor_cache = tensor_cache

# Register the module tree
sys.modules["fla"] = fla_mod
sys.modules["fla.modules"] = fla_modules
sys.modules["fla.ops"] = fla_ops
sys.modules["fla.ops.kda"] = fla_ops_kda
sys.modules["fla.ops.utils"] = fla_ops_utils
sys.modules["fla.ops.utils.index"] = fla_ops_utils_index
sys.modules["fla.utils"] = fla_utils

# ── Step 2: Import torch + load the model ─────────────────────────────────────
import torch
import numpy as np
from safetensors.torch import load_file

MODEL_DIR = os.environ.get(
    "KIMI_K3_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "kimi-k3-0.40b"),
)
MODEL_DIR = os.path.abspath(MODEL_DIR)
print(f"Model dir: {MODEL_DIR}")

# The modeling file uses relative imports (.configuration_kimi_k3),
# so we need to load it as a package module. We register the model dir
# as a package named `kimi_model`.
import importlib.util
import importlib.machinery

MODEL_PKG = "kimi_model"
# Create a package spec pointing at the model directory
pkg_spec = importlib.machinery.ModuleSpec(MODEL_PKG, None, is_package=True)
pkg_spec.submodule_search_locations = [MODEL_DIR]
pkg_mod = importlib.util.module_from_spec(pkg_spec)
sys.modules[MODEL_PKG] = pkg_mod

# Now load the config and modeling modules as submodules of the package
for submod in ["configuration_kimi_k3", "modeling_kimi_linear"]:
    full = f"{MODEL_PKG}.{submod}"
    path = os.path.join(MODEL_DIR, f"{submod}.py")
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)

KimiLinearConfig = sys.modules[f"{MODEL_PKG}.configuration_kimi_k3"].KimiLinearConfig
KimiK3Config = sys.modules[f"{MODEL_PKG}.configuration_kimi_k3"].KimiK3Config
KimiLinearForCausalLM = sys.modules[f"{MODEL_PKG}.modeling_kimi_linear"].KimiLinearForCausalLM

# ── Step 3: Load config ───────────────────────────────────────────────────────
config_path = os.path.join(MODEL_DIR, "config.json")
config = KimiK3Config.from_pretrained(config_path)
text_config = config.text_config
print(f"Config loaded: hidden_size={text_config.hidden_size}, "
      f"layers={text_config.num_hidden_layers}, vocab={text_config.vocab_size}")

# Set attention implementation to eager (CPU, no flash attention)
text_config._attn_implementation = "eager"
config._attn_implementation = "eager"

# ── Step 4: Build model from config ───────────────────────────────────────────
print("Building model from config...")
model = KimiLinearForCausalLM(text_config)
model.eval()
print(f"Model param count: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 5: Load weights from safetensors ─────────────────────────────────────
print("Loading safetensors weights...")
safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
raw_tensors = load_file(safetensors_path)

# The safetensors uses "language_model." prefix for the text model.
# The KimiLinearForCausalLM expects keys without that prefix (it IS the language model).
# Map: "language_model.model.X" → "model.X", "language_model.lm_head.X" → "lm_head.X"
state_dict = {}
for name, tensor in raw_tensors.items():
    if name.startswith("language_model."):
        new_name = name[len("language_model."):]
        state_dict[new_name] = tensor
    elif name.startswith("vision_tower.") or name.startswith("mm_projector."):
        continue  # skip vision components

print(f"  {len(state_dict)} tensors mapped (vision components skipped)")

# Load state dict
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
if unexpected:
    print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

# ── Step 6: Run forward pass on BOS token ─────────────────────────────────────
print("\nRunning forward pass on BOS token (id=1)...")
bos_token_id = text_config.bos_token_id
input_ids = torch.tensor([[bos_token_id]], dtype=torch.long)

with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=False)

logits = outputs.logits  # [1, 1, vocab_size]
hidden = outputs.hidden_states[-1] if outputs.hidden_states else None

# If hidden_states not returned, get it from the model directly
if hidden is None:
    with torch.no_grad():
        model_out = model.model(input_ids=input_ids, use_cache=False)
        hidden = model_out.last_hidden_state

logits_flat = logits[0, 0, :].float().numpy()
hidden_flat = hidden[0, 0, :].float().numpy()

print(f"Logits shape: {logits_flat.shape}")
print(f"Logits range: [{logits_flat.min():.4f}, {logits_flat.max():.4f}]")
print(f"Hidden shape: {hidden_flat.shape}")
print(f"Hidden range: [{hidden_flat.min():.4f}, {hidden_flat.max():.4f}]")
print(f"NaN count: {np.isnan(logits_flat).sum()}, Inf count: {np.isinf(logits_flat).sum()}")

# Top-5 tokens
top5_idx = np.argsort(logits_flat)[::-1][:5]
print(f"Top-5 tokens: {top5_idx.tolist()}")
print(f"Top-5 logits: {[logits_flat[i] for i in top5_idx]}")

# ── Step 7: Save reference outputs + per-layer hidden states ────────────────
out_dir = os.path.join(SCRIPT_DIR, "ref_output")
os.makedirs(out_dir, exist_ok=True)

# Save as .npy (for Python/numpy comparison)
logits_npy = os.path.join(out_dir, "ref_logits_bos.npy")
hidden_npy = os.path.join(out_dir, "ref_hidden_bos.npy")
np.save(logits_npy, logits_flat)
np.save(hidden_npy, hidden_flat)

# Save as raw f32 LE binary (for Rust bytemuck::cast_slice comparison)
logits_bin = os.path.join(out_dir, "ref_logits_bos.bin")
logits_flat.astype(np.float32).tofile(logits_bin)

# Also copy the raw binary to the model dir where the Rust test expects it
model_dir = MODEL_DIR
logits_model_dir = os.path.join(model_dir, "ref_logits_bos.npy")
logits_flat.astype(np.float32).tofile(logits_model_dir)

print(f"\nReference logits (.npy) saved to: {logits_npy}")
print(f"Reference logits (.bin) saved to: {logits_bin}")
print(f"Reference logits (raw f32) copied to: {logits_model_dir}")
print(f"Reference hidden saved to: {hidden_npy}")

# ── Step 8: Dump per-layer hidden states for divergence isolation ─────────
# This runs the model again with hooks to capture the output of each decoder
# layer, enabling a layer-by-layer diff against the Rust implementation.
print("\nDumping per-layer hidden states for divergence isolation...")

layer_outputs = {}

def make_hook(idx):
    def hook(module, input, output):
        # output is either a tensor (no attn_res) or (prefix_sum, block_residual)
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        layer_outputs[idx] = h[0, 0, :].float().numpy().copy()
    return hook

hooks = []
for idx, layer in enumerate(model.model.layers):
    h = layer.register_forward_hook(make_hook(idx))
    hooks.append(h)

# Also hook the embedding output + final norm
embed_hook_data = {}
def embed_hook(module, input, output):
    embed_hook_data["embed"] = output[0, 0, :].float().numpy().copy()
hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

norm_hook_data = {}
def norm_hook(module, input, output):
    norm_hook_data["final_norm"] = output[0, 0, :].float().numpy().copy()
hooks.append(model.model.norm.register_forward_hook(norm_hook))

with torch.no_grad():
    model(input_ids=input_ids, use_cache=False)

for h in hooks:
    h.remove()

# Save per-layer outputs
layer_dir = os.path.join(out_dir, "ref_layers")
os.makedirs(layer_dir, exist_ok=True)
np.save(os.path.join(layer_dir, "layer_embed.bin.npy"), embed_hook_data["embed"])
for idx in sorted(layer_outputs.keys()):
    path = os.path.join(layer_dir, f"layer_{idx}_out.bin.npy")
    np.save(path, layer_outputs[idx])
    # Also save raw f32 for Rust comparison
    raw_path = os.path.join(layer_dir, f"layer_{idx}_out.bin")
    layer_outputs[idx].astype(np.float32).tofile(raw_path)
np.save(os.path.join(layer_dir, "layer_final_norm.bin.npy"), norm_hook_data["final_norm"])
norm_hook_data["final_norm"].astype(np.float32).tofile(
    os.path.join(layer_dir, "layer_final_norm.bin"))
# Also save embed as raw
embed_hook_data["embed"].astype(np.float32).tofile(
    os.path.join(layer_dir, "layer_embed.bin"))

print(f"Per-layer hidden states saved to: {layer_dir}/")
for idx in sorted(layer_outputs.keys()):
    h = layer_outputs[idx]
    print(f"  layer {idx}: range [{h.min():.4f}, {h.max():.4f}], mean {h.mean():.4f}")

print("DONE.")
