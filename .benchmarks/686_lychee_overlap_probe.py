#!/usr/bin/env python3
"""
Issue 694 PoC probe: adjacent-layer same-head top-k attention-index overlap
on Kimi-K3-0.40B real weights (CPU, float32, eager attention).

ARCHITECTURE DISCOVERY (documented in the bench note):
  Kimi-K3-0.40B is a HYBRID model. Per config.json linear_attn_config:
    kda_layers       [1,2,3,5,6,7] (1-indexed) -> layer_idx 0,1,2,4,5,6 = KimiDeltaAttention (linear attn)
    full_attn_layers [4,8]         (1-indexed) -> layer_idx 3,7         = KimiMLAAttention (full attn)
  Only TWO full-attention layers exist and they are NOT adjacent (span 4).
  The paper's adjacent-layer chain mechanism is therefore structurally
  inapplicable on this model; we measure the ONLY measurable full-attn chain:
  layer 3 -> layer 7 same-head top-k overlap.

CPU EXECUTION STRATEGY (no triton on macOS):
  1. Pre-register minimal stub modules for the exact `fla.*` import names in
     modeling_kimi_linear.py (the fla kernels are triton-only and do not
     build on macOS). The stubs provide constructor parity ONLY for the two
     module classes KimiDeltaAttention.__init__ instantiates
     (ShortConvolution, FusedRMSNormGated); their forwards are never called.
  2. Replace KimiDeltaAttention.forward with a pure-torch reimplementation of
     the fla KDA semantics (validated against ref_logits_bos.npy — the BOS
     logits row produced by the reference implementation ships with the
     model dir; the probe reproduces it at cosine 0.999999 with matching
     argmax):
       - ShortConvolution: causal depthwise conv1d (left-pad W-1 prefill /
         cache of last W inputs at decode; conv over cache+new, keep the last
         T outputs), silu activation.
       - q/k L2-normalized (use_qk_l2norm_in_kernel=True), beta sigmoid
         (use_beta_sigmoid_in_kernel=True, scale=1.0).
       - per-step per-dim log decay g = -exp(A_log) * softplus(g_proj + dt_bias)
         (no lower_bound; gate_lower_bound absent from config -> safe_gate False).
       - delta-rule recurrence S <- S*exp(g); S <- S + beta*k (x) (v - k.S);
         o = (q*scale).S  (scale = head_dim**-0.5 = 32**-0.5)
       - output gate: rmsnorm(o) * o_norm.weight * sigmoid(g_proj_full)
         (use_full_rank_gate=True; FusedRMSNormGated eps from config).
  3. Wrap modeling_kimi_linear.eager_attention_forward to stash per-layer
     attention probs (last query row) into a capture dict.

Determinism: fixed prompts, greedy decode, CPU float32; the probe asserts
bit-identical tables across two full runs (--determinism-check).

Usage:
  uv run --python /opt/homebrew/bin/python3 --with einops \
      python 686_lychee_overlap_probe.py [--smoke] [--determinism-check] [--json-out PATH]
"""

import argparse
import json
import os
import sys
import types

import torch  # noqa: E402  (must precede the fla stub classes below)
import torch.nn.functional as F  # noqa: E402

MODEL_DIR = "/Users/katopz/git/katgpt-rs/data/kimi-k3-0.40b"
SEED = 0
BLOCK = 64
RAW_KS = (512, 1024, 2048)
BLK_KS = (8, 16, 32)  # top-N 64-token blocks == 512/1024/2048 tokens
N_STEPS = 48
SMOKE_STEPS = 8

# ----------------------------------------------------- fla package stubs (CPU)
# The fla (flash-linear-attention) kernels are triton-only (no macOS build).
# We replace KimiDeltaAttention.forward with a pure-torch implementation, so
# the ONLY thing the remote code needs from fla is constructor parity for
# the two module classes KimiDeltaAttention.__init__ instantiates. We pre-
# register minimal stub modules for the exact `fla.*` import names in
# modeling_kimi_linear.py (lines 54-62) and never import the real package
# (which pulls triton at module level).


class _ShortConvolutionStub(torch.nn.Conv1d):
    """Constructor parity with fla.modules.ShortConvolution (depthwise causal
    conv, weight [D,1,W], no bias). Forward is never called on the probe path
    (_short_conv_silu below re-implements it in pure torch)."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **kw):
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )
        self.hidden_size = hidden_size
        self.activation = activation


class _FusedRMSNormGatedStub(torch.nn.Module):
    """Constructor parity with fla.modules.FusedRMSNormGated: elementwise
    affine weight + eps + activation. Forward is never called on the probe
    path (_rms_norm_sigmoid_gate re-implements it in pure torch)."""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kw):
        super().__init__()
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))


def _fla_unavailable(*a, **k):
    raise RuntimeError("fla stub: kernel path not available on the CPU probe path")


def _tensor_cache_stub(fn):
    return fn


def _install_fla_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[name] = m
        parent, _, leaf = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], leaf, m)
        return m

    mod("fla")
    mod("fla.modules", FusedRMSNormGated=_FusedRMSNormGatedStub, ShortConvolution=_ShortConvolutionStub)
    mod("fla.modules.conv")
    mod("fla.ops")
    mod("fla.ops.kda", chunk_kda=_fla_unavailable, fused_recurrent_kda=_fla_unavailable)
    mod("fla.ops.utils")
    mod(
        "fla.ops.utils.index",
        prepare_cu_seqlens_from_mask=_fla_unavailable,
        prepare_lens_from_mask=_fla_unavailable,
    )
    mod("fla.utils", tensor_cache=_tensor_cache_stub)


torch.manual_seed(SEED)

# ------------------------------------------------------- pure-torch KDA layer


def _short_conv_silu(conv, x, cache=None):
    """Causal depthwise conv1d + silu, faithful to fla ShortConvolution.

    conv: nn.Conv1d subclass, weight [D,1,W], no bias, groups=D.
    x: [B,T,D]. cache: [B,D,W] holding the last W inputs (oldest first) or None.
    Returns (out [B,T,D], new_cache [B,D,W]).

    Semantics: out[t] = silu( sum_j w[0,j] * x[t-W+1+j] )  (w[W-1] hits x[t]).
    """
    B, T, D = x.shape
    W = conv.kernel_size[0]
    xt = x.transpose(1, 2)  # [B,D,T]
    if cache is not None:
        hist = torch.cat([cache, xt], dim=-1)  # [B,D,W+T]
    else:
        hist = F.pad(xt, (W - 1, 0))  # [B,D,W-1+T]
    out = F.conv1d(hist, conv.weight, groups=D)  # [B,D,T+1] w/ cache, [B,D,T] w/o
    out = out[..., -T:]  # with cache: drop the already-produced oldest output
    out = F.silu(out)
    new_cache = hist[..., -W:].detach()
    return out.transpose(1, 2), new_cache


def _kda_recurrent(q, k, v, g, beta, s0, scale):
    """KDA delta-rule recurrence, pure torch, float32.

    q,k,v: [B,T,H,D]; g: [B,T,H,D] per-step per-dim log-decay; beta: [B,T,H].
    State S: [B,H,D,V] (D=key dim, V=value dim).
    Mirrors fla naive_recurrent_kda exactly (G=1 here, no repeat_interleave).
    """
    B, T, H, D = q.shape
    V = v.shape[-1]
    q = q * scale
    S = torch.zeros(B, H, D, V, dtype=torch.float32) if s0 is None else s0.to(torch.float32).clone()
    o = torch.empty(B, T, H, V, dtype=torch.float32)
    for t in range(T):
        S = S * g[:, t].exp().unsqueeze(-1)  # decay over key dim
        kt = k[:, t]  # [B,H,D]
        vt = v[:, t]  # [B,H,V]
        bt = beta[:, t]  # [B,H]
        corr = vt - torch.einsum("bhd,bhdv->bhv", kt, S)  # v - k.S
        S = S + bt.unsqueeze(-1).unsqueeze(-1) * kt.unsqueeze(-1) * corr.unsqueeze(-2)
        o[:, t] = torch.einsum("bhd,bhdv->bhv", q[:, t], S)
    return o, S


def _rms_norm_sigmoid_gate(o, weight, gg, eps):
    """FusedRMSNormGated(activation='sigmoid'): rmsnorm(o) * weight * sigmoid(gg)."""
    rms = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    return rms * weight * torch.sigmoid(gg)


def kda_forward_pure(self, hidden_states, attention_mask=None, cache_params=None, **kwargs):
    """Pure-torch replacement for KimiDeltaAttention.forward.

    Probe path assumptions: batch=1, no padding (attention_mask None on the
    linear-attention path when the prompt is unpadded), float32 CPU.
    """
    B, T, _ = hidden_states.shape
    conv_q = conv_k = conv_v = None
    rec = None
    if cache_params is not None:
        if cache_params.conv_states[self.layer_idx] is not None:
            conv_q, conv_k, conv_v = cache_params.conv_states[self.layer_idx]
        rec = cache_params.recurrent_states[self.layer_idx]

    H = self.num_heads
    D = self.head_dim

    qp = self.q_proj(hidden_states)
    kp = self.k_proj(hidden_states)
    vp = self.v_proj(hidden_states)
    q, conv_q = _short_conv_silu(self.q_conv1d, qp, conv_q)
    k, conv_k = _short_conv_silu(self.k_conv1d, kp, conv_k)
    v, conv_v = _short_conv_silu(self.v_conv1d, vp, conv_v)

    q = q.view(B, T, H, D)
    k = k.view(B, T, H, D)
    v = v.view(B, T, H, D)

    g_in = self.f_b_proj(self.f_a_proj(hidden_states)).view(B, T, H, D)
    beta_raw = self.b_proj(hidden_states).float()  # [B,T,H]

    q32 = F.normalize(q.float(), dim=-1)
    k32 = F.normalize(k.float(), dim=-1)
    beta = torch.sigmoid(beta_raw)

    a_log = self.A_log.detach().float().view(1, 1, H, 1)
    dt_bias = self.dt_bias.detach().float().view(1, 1, H, D)
    g_dec = -a_log * F.softplus(g_in.float() + dt_bias)  # [B,T,H,D]

    o, rec_out = _kda_recurrent(q32, k32, v.float(), g_dec, beta, rec, scale=D**-0.5)

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = rec_out
        cache_params.conv_states[self.layer_idx] = (conv_q, conv_k, conv_v)

    if self.use_full_rank_gate:
        gg = self.g_proj(hidden_states)
    else:
        gg = self.g_b_proj(self.g_a_proj(hidden_states))
    gg = gg.view(B, T, H, D).float()
    o = _rms_norm_sigmoid_gate(o.float(), self.o_norm.weight.detach().float(), gg, self.o_norm.eps)
    o = o.reshape(B, T, H * D)
    return self.o_proj(o.to(hidden_states.dtype))


# ------------------------------------------------------------- model loading


def load_model():
    _install_fla_stubs()  # BEFORE the remote-code import inside from_pretrained
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model = model.to("cpu").eval()
    model.language_model.config._attn_implementation = "eager"

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    lin_mod = sys.modules[type(model.language_model).__module__]

    cap = {"probs": {}}
    orig_eager = lin_mod.eager_attention_forward

    def eager_probe(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, probs = orig_eager(module, query, key, value, attention_mask, scaling, dropout, **kw)
        cap["probs"].setdefault(module.layer_idx, []).append(
            probs.detach()[:, :, -1, :].float().clone()
        )
        return out, probs

    lin_mod.eager_attention_forward = eager_probe

    kda_cls = None
    layer_types = []
    for layer in model.language_model.model.layers:
        layer_types.append("KDA" if layer.is_linear_attn else "MLA")
        if layer.is_linear_attn and kda_cls is None:
            kda_cls = type(layer.self_attn)
    kda_cls.forward = kda_forward_pure
    return model, tok, cap, layer_types


# ------------------------------------------------------------ prompt recipe

FILLER = [
    "The old library stood quiet under the pale morning light.",
    "Gardeners swept the stone paths before the visitors arrived.",
    "A small bronze bell hung above the wooden entrance door.",
    "Records of the harvest were kept in leather-bound notebooks.",
    "The river bent slowly around the eastern hillside.",
    "Children traced the carved figures on the courtyard wall.",
    "Merchants unloaded crates of dried fruit near the harbor.",
    "An apprentice copied letters by the narrow window.",
    "The clockmaker adjusted a spring no larger than a seed.",
    "Fog settled over the fields shortly after sunset.",
    "Two shepherds argued politely about the price of wool.",
    "The baker's oven kept its heat through the long evening.",
]
NEEDLE_TMPL = "The magic number found in the book is {n}."
NEEDLES = (48371, 90215, 65932)
DEPTHS = (0.25, 0.50, 0.75)
QUESTION = "\n\nQuestion: What is the magic number found in the book? Answer:"
HEADER = "The following is a long passage from an old book."
TARGET_CTX = 3650


def build_prompts(tok):
    """Fixed 9 prompts: 3 needle depths x 3 needle values. Fully deterministic."""
    flen = [len(tok(f, add_special_tokens=False)["input_ids"]) for f in FILLER]
    q_len = len(tok(QUESTION, add_special_tokens=False)["input_ids"])
    hdr = len(tok(HEADER, add_special_tokens=False)["input_ids"])
    prompts = []
    for depth in DEPTHS:
        for needle in NEEDLES:
            needle_txt = NEEDLE_TMPL.format(n=needle)
            n_len = len(tok(needle_txt, add_special_tokens=False)["input_ids"])
            budget = TARGET_CTX - hdr - q_len - n_len - 10
            # deterministic filler stream (cycled), split at the depth fraction
            total, i = 0, 0
            while True:
                nxt = flen[i % len(flen)]
                if total + nxt > budget:
                    break
                total += nxt
                i += 1
            target_a = int(round(total * depth))
            plen, j, part_a_seq = 0, 0, []
            while plen + flen[j % len(flen)] <= target_a:
                part_a_seq.append(j % len(flen))
                plen += flen[j % len(flen)]
                j += 1
            part_b_seq = []
            plen_b = 0
            while plen_b + flen[j % len(flen)] <= total - plen:
                part_b_seq.append(j % len(flen))
                plen_b += flen[j % len(flen)]
                j += 1
            part_a = " ".join(FILLER[i] for i in part_a_seq)
            part_b = " ".join(FILLER[i] for i in part_b_seq)
            text = f"{HEADER} {part_a} {needle_txt} {part_b}{QUESTION}"
            ids = tok(text, add_special_tokens=False)["input_ids"]
            pre = tok(f"{HEADER} {part_a} ", add_special_tokens=False)["input_ids"]
            pre_n = tok(f"{HEADER} {part_a} {needle_txt}", add_special_tokens=False)["input_ids"]
            prompts.append(
                {
                    "depth": depth,
                    "needle": needle,
                    "text": text,
                    "n_tokens": len(ids),
                    "needle_span": (len(pre), len(pre_n)),
                }
            )
    return prompts


# ------------------------------------------------------------- capture loop


def run_prompt(model, tok, prompt, n_steps, cap):
    ids = tok(prompt["text"], add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long)
    cap["probs"].clear()
    with torch.no_grad():
        out = model.language_model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1].argmax(-1)
        gen = []
        step_rows = []
        for _s in range(n_steps):
            out = model.language_model(
                input_ids=next_tok.unsqueeze(0), past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            rows = {l: r[0][0].clone() for l, r in cap["probs"].items()}  # [H, ctx]
            cap["probs"].clear()
            step_rows.append(rows)
            next_tok = out.logits[:, -1].argmax(-1)
            gen.append(int(next_tok))
    return {
        "step_rows": step_rows,
        "gen_ids": gen,
        "ctx": len(ids),
        "needle_span": prompt["needle_span"],
    }


# --------------------------------------------------------------- statistics


def topk_sets(row, k):
    return [torch.topk(row[h], k).indices for h in range(row.shape[0])]


def block_topk_sets(row, nblk_k, block=BLOCK):
    H, ctx = row.shape
    nblk = ctx // block
    bs = row[:, : nblk * block].view(H, nblk, block).sum(-1)  # [H, nblk] mass
    return [torch.topk(bs[h], nblk_k).indices for h in range(H)]


def overlap_of(a, b, k):
    return len(set(a.tolist()) & set(b.tolist())) / k


def _pearson(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _spearman(a, b):
    ra = torch.tensor(a, dtype=torch.float64).argsort().argsort().double()
    rb = torch.tensor(b, dtype=torch.float64).argsort().argsort().double()
    return _pearson(ra.tolist(), rb.tolist())


def full_run(model, tok, prompts, n_steps, cap):
    """One full probe pass -> summary dict."""
    runs = [run_prompt(model, tok, p, n_steps, cap) for p in prompts]
    H = runs[0]["step_rows"][0][3].shape[0]

    raw_acc = {k: [[] for _ in range(H)] for k in RAW_KS}
    blk_acc = {k: [[] for _ in range(H)] for k in BLK_KS}
    step_ov = {k: [] for k in RAW_KS}
    depth_acc = {str(d): {k: [] for k in RAW_KS} for d in DEPTHS}
    cross = []
    within7 = []
    within3 = []
    t2t = {3: [], 7: []}  # step-to-step (time-axis) same-head overlap, k=1024
    prev_sets = None
    needle_mass = {3: [[] for _ in range(H)], 7: [[] for _ in range(H)]}
    gen_texts = []

    for run, prompt in zip(runs, prompts):
        span = run["needle_span"]
        prev_sets = None  # reset per prompt (step-to-step must not cross contexts)
        for rows in run["step_rows"]:
            r3, r7 = rows[3], rows[7]
            for k in RAW_KS:
                s3, s7 = topk_sets(r3, k), topk_sets(r7, k)
                ovs = [overlap_of(s3[h], s7[h], k) for h in range(H)]
                for h in range(H):
                    raw_acc[k][h].append(ovs[h])
                step_ov[k].append(sum(ovs) / H)
                depth_acc[str(prompt["depth"])][k].extend(ovs)
            for k in BLK_KS:
                s3, s7 = block_topk_sets(r3, k), block_topk_sets(r7, k)
                for h in range(H):
                    blk_acc[k][h].append(overlap_of(s3[h], s7[h], k))
            s3, s7 = topk_sets(r3, 1024), topk_sets(r7, 1024)
            cross.append([[overlap_of(s3[a], s7[b], 1024) for b in range(H)] for a in range(H)])
            within7.append([[overlap_of(s7[a], s7[b], 1024) for b in range(H)] for a in range(H)])
            within3.append([[overlap_of(s3[a], s3[b], 1024) for b in range(H)] for a in range(H)])
            if prev_sets is not None:
                cur = {3: s3, 7: s7}
                for l in (3, 7):
                    t2t[l].append([overlap_of(prev_sets[l][h], cur[l][h], 1024) for h in range(H)])
            prev_sets = {3: s3, 7: s7}
            for l, r in ((3, r3), (7, r7)):
                mass = r[:, span[0] : span[1]].sum(-1)
                for h in range(H):
                    needle_mass[l][h].append(float(mass[h]))
        gen_texts.append(tok.decode(run["gen_ids"]))

    def head_stats(acc):
        means = [round(float(torch.tensor(v).mean()), 6) for v in acc]
        t = sorted(means)
        n = len(t)
        med = t[n // 2] if n % 2 else round(0.5 * (t[n // 2 - 1] + t[n // 2]), 6)
        return {
            "per_head": means,
            "median": med,
            "q1": t[n // 4],
            "q3": t[(3 * n) // 4],
            "min": t[0],
            "max": t[-1],
            "n_ge_0.5": sum(m >= 0.5 for m in means),
            "n_ge_0.7": sum(m >= 0.7 for m in means),
            "n_ge_0.9": sum(m >= 0.9 for m in means),
        }

    out = {"heads": H, "steps": n_steps, "prompts": len(prompts)}
    for k in RAW_KS:
        out[f"raw{k}"] = head_stats(raw_acc[k])
    for k in BLK_KS:
        out[f"blk{k}"] = head_stats(blk_acc[k])

    mean_mat = [
        [round(float(torch.tensor([m[a][b] for m in cross]).mean()), 6) for b in range(H)]
        for a in range(H)
    ]
    same = [mean_mat[h][h] for h in range(H)]
    best_other = [max(mean_mat[h][b] for b in range(H) if b != h) for h in range(H)]

    def offdiag_mean(mats):
        vals = [m[a][b] for m in mats for a in range(H) for b in range(H) if a != b]
        return round(sum(vals) / len(vals), 6)

    out["cross_head"] = {
        "k": 1024,
        "mean_matrix": mean_mat,
        "same_head": same,
        "best_other": best_other,
        "margin": [round(s - b, 6) for s, b in zip(same, best_other)],
        "within_L7_offdiag": offdiag_mean(within7),
        "within_L3_offdiag": offdiag_mean(within3),
    }
    out["time_axis"] = {
        "note": "same-head step-to-step overlap (the axis flashmemory already amortizes)",
        "k": 1024,
        "L3_mean": round(float(torch.tensor([v for m in t2t[3] for v in m]).mean()), 6),
        "L7_mean": round(float(torch.tensor([v for m in t2t[7] for v in m]).mean()), 6),
        "L3_per_head": [
            round(float(torch.tensor([m[h] for m in t2t[3]]).mean()), 6) for h in range(H)
        ],
        "L7_per_head": [
            round(float(torch.tensor([m[h] for m in t2t[7]]).mean()), 6) for h in range(H)
        ],
    }
    ov = out["raw1024"]["per_head"]
    out["needle_mass_corr"] = {}
    for l in (3, 7):
        nm = [round(float(torch.tensor(v).mean()), 6) for v in needle_mass[l]]
        out["needle_mass_corr"][f"L{l}"] = {
            "needle_mass_per_head": nm,
            "pearson": round(_pearson(nm, ov), 4),
            "spearman": round(_spearman(nm, ov), 4),
        }
    out["step_trend_raw1024"] = {
        "first8": round(sum(step_ov[1024][:8]) / max(1, len(step_ov[1024][:8])), 6),
        "last8": round(sum(step_ov[1024][-8:]) / max(1, len(step_ov[1024][-8:])), 6),
    }
    out["per_depth_raw1024"] = {
        d: round(float(torch.tensor(v[1024]).mean()), 6) if v[1024] else None
        for d, v in depth_acc.items()
    }
    out["gen_texts"] = gen_texts
    out["ctx_tokens"] = [r["ctx"] for r in runs]
    out["needle_spans"] = [list(r["needle_span"]) for r in runs]
    return out


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 prompts x 8 steps")
    ap.add_argument("--determinism-check", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    model, tok, cap, layer_types = load_model()
    print("layer types:", layer_types, file=sys.stderr)

    import numpy as np

    ref_path = os.path.join(MODEL_DIR, "ref_logits_bos.npy")
    with torch.no_grad():
        bos = model.language_model(input_ids=torch.tensor([[1]]), use_cache=False)
    logits = bos.logits[0, -1].float()
    if os.path.exists(ref_path):
        # ref_logits_bos.npy is a RAW float32 dump (no npy header): 655360 bytes
        # = 163840 f32 = exactly one vocab-sized logits row.
        ref = torch.from_numpy(np.fromfile(ref_path, dtype=np.float32).copy())
        cos = F.cosine_similarity(logits, ref, dim=0).item()
        mad = float((logits - ref).abs().max())
        print(
            f"[ref-check] cos={cos:.6f} maxabsdiff={mad:.5f} argmax={int(logits.argmax())} ref_argmax={int(ref.argmax())}",
            file=sys.stderr,
        )

    prompts = build_prompts(tok)
    if args.smoke:
        prompts = prompts[:2]
        n_steps = SMOKE_STEPS
    else:
        n_steps = N_STEPS
    for p in prompts:
        print(
            f"prompt depth={p['depth']} needle={p['needle']} ctx={p['n_tokens']} span={p['needle_span']}",
            file=sys.stderr,
        )

    s1 = full_run(model, tok, prompts, n_steps, cap)
    if args.determinism_check:
        s2 = full_run(model, tok, prompts, n_steps, cap)
        j1, j2 = json.dumps(s1, sort_keys=True), json.dumps(s2, sort_keys=True)
        print(f"[determinism] identical={j1 == j2}", file=sys.stderr)
        s1["determinism_identical"] = j1 == j2

    print(json.dumps(s1, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(s1, f, indent=2)


if __name__ == "__main__":
    main()
