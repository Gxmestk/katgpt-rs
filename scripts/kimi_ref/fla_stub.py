"""
Pure-PyTorch replacements for fla's triton-dependent modules.

Every implementation here is derived from the fla source code (the triton
kernels themselves, not just their docstrings). This allows the Kimi-K3 model
to run on CPU without CUDA/triton, producing reference logits for G1
correctness verification.

Source citations:
- fused_recurrent_kda: fla/ops/kda/fused_recurrent.py (triton kernel inner loop)
- FusedRMSNormGated:   fla/modules/fused_norm_gate.py (IS_RMS_NORM=True path)
- ShortConvolution:    fla/modules/conv/short_conv.py (step() path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# softplus (matching fla.ops.utils.softplus)
# ─────────────────────────────────────────────────────────────────────────────

def softplus(x):
    """torch.nn.functional.softplus — the exact function fla uses internally."""
    return F.softplus(x)


# ─────────────────────────────────────────────────────────────────────────────
# fused_recurrent_kda: pure-PyTorch from triton kernel source
# ─────────────────────────────────────────────────────────────────────────────

def fused_recurrent_kda(
    q, k, v, g, beta,
    A_log=None, dt_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    lower_bound=None,
    state_v_first=False,
    cu_seqlens=None,
    **kwargs,
):
    """
    Pure-PyTorch implementation of fla.ops.kda.fused_recurrent_kda.

    Implements the same recurrence as the triton kernel (state layout [K, V]):
        For each timestep t, per head hv:
        1. L2-normalize q, k (if use_qk_l2norm_in_kernel)
        2. Scale q by `scale`
        3. Compute per-channel decay: gk = -exp(A_log[hv]) * softplus(g + dt_bias)
        4. Decay state: S *= exp(gk)[:, None]   (per-row)
        5. Delta rule: v_new = (v - S^T @ k) * sigmoid(beta)
        6. Update: S += k[:, None] * v_new[None, :]
        7. Readout: o = S^T @ q
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]

    if initial_state is not None:
        S = initial_state.clone().float()  # [B, HV, K, V]
    else:
        S = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)

    out = torch.zeros(B, T, HV, V, dtype=v.dtype, device=v.device)

    for b in range(B):
        for t in range(T):
            for hv in range(HV):
                q_t = q[b, t, hv, :].float()
                k_t = k[b, t, hv, :].float()
                v_t = v[b, t, hv, :].float()

                if use_qk_l2norm_in_kernel:
                    q_t = q_t / torch.sqrt(torch.sum(q_t * q_t) + 1e-6)
                    k_t = k_t / torch.sqrt(torch.sum(k_t * k_t) + 1e-6)
                q_t = q_t * scale

                if use_gate_in_kernel:
                    g_t = g[b, t, hv, :].float()
                    A_val = A_log[hv].float()
                    if dt_bias is not None:
                        bias = dt_bias[hv * K: (hv + 1) * K].float()
                        g_t = g_t + bias
                    if lower_bound is not None:
                        gk = lower_bound * torch.sigmoid(torch.exp(A_val) * g_t)
                    else:
                        gk = -torch.exp(A_val) * softplus(g_t)
                else:
                    gk = g[b, t, hv, :].float()

                decay = torch.exp(gk)
                S[b, hv] = S[b, hv] * decay.unsqueeze(1)

                Sk = S[b, hv]
                temp = k_t @ Sk
                v_new = v_t - temp

                beta_t = beta[b, t, hv].float()
                if use_beta_sigmoid_in_kernel:
                    beta_t = torch.sigmoid(beta_t)
                    if allow_neg_eigval:
                        beta_t = beta_t * 2.0
                v_new = v_new * beta_t

                S[b, hv] = Sk + k_t.unsqueeze(1) * v_new.unsqueeze(0)
                out[b, t, hv, :] = (q_t @ S[b, hv]).to(v.dtype)

    return out, (S if output_final_state else None)


def chunk_kda(*args, **kwargs):
    """Chunk mode not needed for single-token inference. Delegate to fused_recurrent."""
    return fused_recurrent_kda(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# FusedRMSNormGated: IS_RMS_NORM=True, ACTIVATION="sigmoid"
# ─────────────────────────────────────────────────────────────────────────────

class FusedRMSNormGated(nn.Module):
    """
    RMS normalization gated by sigmoid(g).

    From triton kernel (IS_RMS_NORM=True):
        var = mean(x^2, dim=-1, keepdim=True)
        x_hat = x * rsqrt(var + eps)
        y = x_hat * weight * sigmoid(g)
    """

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5,
                 activation="swish", device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        factory_kwargs = {"device": device, "dtype": dtype}
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        x_shape = x.shape
        x_flat = x.reshape(-1, x_shape[-1]).float()
        g_flat = g.reshape(-1, g.shape[-1]).float()

        var = x_flat.pow(2).mean(dim=-1, keepdim=True)
        x_hat = x_flat * torch.rsqrt(var + self.eps)
        y = x_hat * self.weight.float()

        if self.activation == "sigmoid":
            y = y * torch.sigmoid(g_flat)
        elif self.activation in ("swish", "silu"):
            y = y * g_flat * torch.sigmoid(g_flat)

        return y.reshape(x_shape).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# ShortConvolution: depthwise causal Conv1d with SiLU + cache
# ─────────────────────────────────────────────────────────────────────────────

class ShortConvolution(nn.Conv1d):
    """
    Causal depthwise 1D convolution with SiLU activation and cache.

    step() path (single-token inference):
        cache.roll(shifts=-1, dims=-1)
        cache[:, :, -1] = x
        y = sum(cache * weight, dim=-1)
        y = silu(y)
    """

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu",
                 backend="triton", device=None, dtype=None, **kwargs):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.activation = activation
        self.backend = backend

    def forward(self, x, residual=None, mask=None, cache=None,
                output_final_state=False, cu_seqlens=None, **kwargs):
        B, T, D = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        if B * T == N:
            return self._step(x, residual, cache, output_final_state, cu_seqlens)

        # Prefill: causal Conv1d (left-pad only)
        W = self.kernel_size[0]
        x_t = x.transpose(1, 2)
        x_padded = F.pad(x_t, (W - 1, 0))
        y = F.conv1d(x_padded, self.weight, self.bias, groups=self.groups)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        y = y.transpose(1, 2)
        return y, (cache if output_final_state else None)

    def _step(self, x, residual, cache, output_final_state, cu_seqlens):
        B, T, D = x.shape
        W = self.kernel_size[0]

        if cache is None:
            cache = x.new_zeros(B, D, W)

        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = x.reshape(-1, D) if cu_seqlens is not None else x[:, 0, :]

        w = self.weight.squeeze(1)
        y = (cache * w.unsqueeze(0)).sum(dim=-1)

        if self.activation in ("silu", "swish"):
            y = F.silu(y)

        y = y.unsqueeze(1)

        if residual is not None:
            y = y + residual

        return y, (cache if output_final_state else None)


# ─────────────────────────────────────────────────────────────────────────────
# fla.ops.utils.index stubs
# ─────────────────────────────────────────────────────────────────────────────

def prepare_cu_seqlens_from_mask(attention_mask):
    lens = attention_mask.sum(dim=-1)
    cu_seqlens = torch.zeros(
        attention_mask.shape[0] + 1,
        dtype=torch.int32,
        device=attention_mask.device,
    )
    for i, l in enumerate(lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + l
    return cu_seqlens


def prepare_lens_from_mask(attention_mask):
    return attention_mask.sum(dim=-1).tolist()
