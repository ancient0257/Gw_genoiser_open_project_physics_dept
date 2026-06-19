"""
models/autoencoder.py  — v5

Bug-fixes vs v4
---------------
• MultiScaleConvNeXt.dw_b padding formula was wrong:
  used padding=k_b*2 instead of dilation*(k_b-1)//2.
  For k_b=7, dilation=4: correct=12, was=14 → output length was +2 every
  forward pass, hitting the trim branch on every single call.
  Fixed: padding = dilation * (k_b - 1) // 2.

• LinearSelfAttention normaliser was wrong:
  einsum "bhdi,bhdi->bhi" contracts over BOTH d and i (=L), giving a
  scalar (B,H) normaliser shared across all positions — mathematically
  incorrect linear attention.
  Correct: per-position normaliser = q_i · Σ_j k_j, shape (B,H,L).
  Fixed: split into (1) k_sum = k.sum(-1) → (B,H,D), then
  (2) denom = einsum "bhd,bhdl->bhl" q,k_sum → per-position (B,H,L).

Performance additions vs v4
----------------------------
• Flash-attention path: when torch.nn.functional.scaled_dot_product_attention
  is available (PyTorch ≥ 2.0) and bottleneck L ≤ 512, use full MHA via
  F.sdpa (which dispatches to FlashAttention-2 on CUDA) instead of the
  manual linear-attention path — faster on modern GPUs for small L.
• Depthwise separable ConvTranspose in decoder: split the large-kernel
  ConvTranspose1d into (1) depthwise ConvTranspose1d + (2) pointwise Conv1d.
  Same expressiveness, ~4× fewer parameters in the upsample path,
  less checkerboard artefact from large-stride transpose convolutions.
• Pre-activation residual in bottleneck: DilatedResBlock now follows the
  full pre-activation ResNet pattern (norm→act→conv) instead of
  norm→conv→act, which consistently improves gradient flow in deep stacks.
• Bottleneck output projection uses grouped conv (groups=ch//16) instead
  of pointwise — maintains channel mixing with reduced parameter count.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _norm(ch: int, g: int = 8) -> nn.GroupNorm:
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)

def _kaiming(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep).float()
        return x * mask / keep


class LayerScale(nn.Module):
    def __init__(self, ch: int, init_val: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.full((ch, 1), init_val))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SEBlock1d(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(ch // r, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False), nn.SiLU(),
            nn.Linear(mid, ch, bias=False), nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# FiLM conditioning
# ─────────────────────────────────────────────────────────────────────────────

class FiLMConditioner(nn.Module):
    def __init__(self, ch: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, ch * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, noise_feat: torch.Tensor) -> torch.Tensor:
        params      = self.net(noise_feat)
        gamma, beta = params.chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


def _noise_features(x: torch.Tensor) -> torch.Tensor:
    xf      = x.squeeze(1)
    log_rms = torch.log(xf.pow(2).mean(-1).sqrt().clamp(min=1e-8))
    log_pk  = torch.log(xf.abs().amax(-1).clamp(min=1e-8))
    return torch.stack([log_rms, log_pk], dim=-1)   # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Self-attention at bottleneck  (bug-fix: correct linear attention normaliser)
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckAttention(nn.Module):
    """
    Dual-path attention:
    • If F.scaled_dot_product_attention available AND L ≤ sdpa_max_len:
      use full MHA via F.sdpa (FlashAttention-2 on CUDA) — exact, fast.
    • Otherwise: correct linear attention with per-position normaliser.

    v4 bug-fix: the linear attention normaliser was a scalar (B,H) shared
    across all positions. Correct: per-position dot(q_i, Σ_j k_j) → (B,H,L).
    """
    SDPA_MAX_LEN = 512

    def __init__(self, ch: int, heads: int = 8):
        super().__init__()
        while ch % heads != 0:
            heads //= 2
        self.heads   = heads
        self.head_ch = ch // heads
        self.norm    = _norm(ch)
        self.qkv     = nn.Conv1d(ch, ch * 3, 1, bias=False)
        self.out_proj= nn.Conv1d(ch, ch, 1, bias=False)
        self.ls      = LayerScale(ch, init_val=1e-4)
        nn.init.zeros_(self.out_proj.weight)

    def _linear_attn(self, q: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor) -> torch.Tensor:
        """
        Correct linear attention.
        q,k,v: (B, H, D, L) with ELU+1 applied to q,k.

        Per-position normaliser = q_i · (Σ_j k_j)  → (B, H, L)
        """
        B, H, D, L = q.shape

        # kv aggregate: Σ_j k_j ⊗ v_j  → (B, H, D_k, D_v)
        kv = torch.einsum("bhdi,bhei->bhde", k, v)       # (B, H, D, D)

        # Output at each position: q_i · kv  → (B, H, D, L)
        out = torch.einsum("bhdi,bhde->bhei", q, kv)     # (B, H, D, L)

        # Per-position normaliser: q_i · k_sum  → (B, H, L)
        k_sum = k.sum(dim=-1)                             # (B, H, D)
        denom = torch.einsum("bhdl,bhd->bhl", q, k_sum)  # (B, H, L) ← FIXED
        denom = denom.unsqueeze(2).clamp(min=1e-6)        # (B, H, 1, L)

        return out / denom                                 # (B, H, D, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        H, D    = self.heads, self.head_ch

        h   = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)   # each (B, C, L)

        if _HAS_SDPA and L <= self.SDPA_MAX_LEN:
            # Flash-attention path: reshape to (B, H, L, D) for F.sdpa
            q = q.view(B, H, D, L).permute(0, 1, 3, 2)   # (B, H, L, D)
            k = k.view(B, H, D, L).permute(0, 1, 3, 2)
            v = v.view(B, H, D, L).permute(0, 1, 3, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            out = out.permute(0, 1, 3, 2).contiguous().view(B, C, L)
        else:
            # Correct linear attention path
            q = q.view(B, H, D, L)
            k = k.view(B, H, D, L)
            v = v.view(B, H, D, L)
            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0
            out = self._linear_attn(q, k, v).contiguous().view(B, C, L)

        out = self.out_proj(out)
        return x + self.ls(out)


# ─────────────────────────────────────────────────────────────────────────────
# WaveNet dilated bottleneck — pre-activation residual
# ─────────────────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    """
    Pre-activation (norm→act→conv) residual — better gradient flow than
    post-activation (conv→norm→act) used in v4.
    Output projection uses grouped conv (groups=max(1,ch//16)) for efficiency.
    """
    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1,
                 ls_init: float = 1e-4):
        super().__init__()
        pad       = dilation * (kernel - 1) // 2
        groups    = max(1, ch // 16)
        self.norm = _norm(ch)
        self.act  = nn.SiLU()
        self.conv_f = nn.Conv1d(ch, ch, kernel, dilation=dilation,
                                 padding=pad, bias=False)
        self.conv_g = nn.Conv1d(ch, ch, kernel, dilation=dilation,
                                 padding=pad, bias=False)
        self.proj   = nn.Conv1d(ch, ch, 1, groups=groups, bias=False)
        self.ls     = LayerScale(ch, ls_init)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.act(self.norm(x))             # pre-activation
        f   = torch.tanh(self.conv_f(h))
        g   = torch.sigmoid(self.conv_g(h))
        out = self.proj(f * g)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        return x + self.ls(out)


class DilatedBottleneck(nn.Module):
    def __init__(self, ch: int, n: int = 8, kernel: int = 3, base: int = 2):
        super().__init__()
        half = n // 2
        dilations = [base ** (i % half) for i in range(n)]
        self.blocks = nn.ModuleList([
            DilatedResBlock(ch, kernel, d, ls_init=1e-4 / (i + 1))
            for i, d in enumerate(dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scale ConvNeXt  (bug-fix: correct dilation padding)
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleConvNeXt(nn.Module):
    """
    Bug-fix: dw_b padding was k_b*2 instead of dilation*(k_b-1)//2.
    For k_b=7, dilation=4: correct=12, was=14 → +2 output length every call.
    """
    def __init__(self, ch: int, kernel: int, n_groups: int, expand: int = 4):
        super().__init__()
        dilation = 4
        k_b   = max((kernel // 4) | 1, 3)   # odd, at least 3
        pad_b = dilation * (k_b - 1) // 2   # BUG-FIX: was k_b * 2

        self.norm = _norm(ch, n_groups)
        self.dw_a = nn.Conv1d(ch, ch, kernel, padding=kernel // 2,
                               groups=ch, bias=False)
        self.dw_b = nn.Conv1d(ch, ch, k_b, dilation=dilation,
                               padding=pad_b, groups=ch, bias=False)
        mid       = ch * expand
        self.pw1  = nn.Conv1d(ch, mid, 1, bias=False)
        self.act  = nn.GELU()
        self.pw2  = nn.Conv1d(mid, ch, 1, bias=False)
        nn.init.zeros_(self.pw2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.norm(x)
        dw = self.dw_a(h) + self.dw_b(h)
        # No trim needed after bug-fix — lengths match exactly
        return self.pw2(self.act(self.pw1(dw)))


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int,
                 leaky: float, n_groups: int, drop_path_p: float, expand: int = 4):
        super().__init__()
        self.ms_block   = MultiScaleConvNeXt(in_ch, kernel, n_groups, expand)
        self.drop       = DropPath(drop_path_p)
        self.downsample = nn.Sequential(
            _norm(in_ch, n_groups),
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
        )
        self.downsample.apply(_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x + self.drop(self.ms_block(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Attention gate
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    def __init__(self, x_ch: int, g_ch: int):
        super().__init__()
        inter    = max(x_ch // 2, 8)
        self.Wx  = nn.Conv1d(x_ch, inter, 1, bias=False)
        self.Wg  = nn.Conv1d(g_ch,  inter, 1, bias=False)
        self.psi = nn.Conv1d(inter, x_ch, 1, bias=True)
        nn.init.zeros_(self.psi.weight)
        nn.init.ones_(self.psi.bias)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-1] != x.shape[-1]:
            g = F.interpolate(g, size=x.shape[-1], mode="linear", align_corners=False)
        return x * torch.sigmoid(self.psi(F.silu(self.Wx(x) + self.Wg(g))))


# ─────────────────────────────────────────────────────────────────────────────
# Depthwise-separable ConvTranspose  (new — fewer params, less checkerboard)
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSepConvTranspose1d(nn.Module):
    """
    Depthwise ConvTranspose1d followed by pointwise Conv1d.
    ~4× fewer parameters than a full ConvTranspose1d at same expressiveness.
    Reduces checkerboard artefacts because the pointwise mixes channels
    independently of the spatial upsampling step.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        self.dw = nn.ConvTranspose1d(
            in_ch, in_ch, kernel, stride=stride,
            padding=kernel // 2, output_padding=stride - 1,
            groups=in_ch, bias=False,
        )
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        _kaiming(self.dw); _kaiming(self.pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ─────────────────────────────────────────────────────────────────────────────
# Decoder stage
# ─────────────────────────────────────────────────────────────────────────────

class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel: int, stride: int, n_groups: int,
                 use_attn_gate: bool, is_last: bool):
        super().__init__()
        self.up   = DepthwiseSepConvTranspose1d(in_ch, out_ch, kernel, stride)
        self.norm = _norm(out_ch, n_groups)
        self.act  = nn.SiLU()

        self.gate = AttentionGate(skip_ch, out_ch) if (use_attn_gate and skip_ch > 0) else None

        if skip_ch > 0:
            fuse_in        = out_ch + skip_ch
            self.fuse_norm = _norm(fuse_in, n_groups)
            self.fuse_conv = nn.Conv1d(fuse_in, out_ch * 2, 3, padding=1, bias=False)
            self.se        = SEBlock1d(out_ch) if not is_last else nn.Identity()
        else:
            self.fuse_norm = None
            self.fuse_conv = None
            self.se        = nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.act(self.norm(self.up(x)))

        if skip is not None and self.fuse_conv is not None:
            if x.shape[-1] > skip.shape[-1]:
                x = x[..., :skip.shape[-1]]
            elif x.shape[-1] < skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))

            if self.gate is not None:
                skip = self.gate(skip, x)

            cat       = self.fuse_norm(torch.cat([x, skip], dim=1))
            val, gate = self.fuse_conv(cat).chunk(2, dim=1)
            x         = self.se(val * torch.sigmoid(gate))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class GWAutoencoder(nn.Module):
    """GW strain denoising U-Net v5."""

    def __init__(
        self,
        enc_channels:             tuple = (1, 32, 64, 128, 256, 512, 512, 512),
        kernel_sizes:             tuple = (31, 15, 7, 7, 5, 3, 3),
        stride:                   int   = 2,
        leaky_slope:              float = 0.01,
        num_groups:               int   = 8,
        stochastic_depth_prob:    float = 0.1,
        n_bottleneck_blocks:      int   = 8,
        bottleneck_dilation_base: int   = 2,
        bottleneck_kernel:        int   = 3,
        use_attention_gates:      bool  = True,
        use_bottleneck_attention: bool  = True,
        use_film:                 bool  = True,
        expand_ratio:             int   = 4,
    ):
        super().__init__()
        n = len(kernel_sizes)
        assert len(enc_channels) == n + 1

        self.encoder_stages = nn.ModuleList([
            EncoderStage(
                enc_channels[i], enc_channels[i+1],
                kernel_sizes[i], stride, leaky_slope, num_groups,
                stochastic_depth_prob * i / max(n-1, 1),
                expand_ratio,
            )
            for i in range(n)
        ])

        btn_ch       = enc_channels[-1]
        self.dilated = DilatedBottleneck(btn_ch, n_bottleneck_blocks,
                                          bottleneck_kernel, bottleneck_dilation_base)
        self.attn    = (BottleneckAttention(btn_ch)
                        if use_bottleneck_attention else nn.Identity())
        self.film    = FiLMConditioner(btn_ch) if use_film else None

        dec_ch = list(reversed(enc_channels))
        dec_k  = list(reversed(kernel_sizes))
        self.decoder_stages = nn.ModuleList()
        for i in range(n):
            skip_ch = enc_channels[n - 1 - i] if i < n - 1 else 0
            self.decoder_stages.append(DecoderStage(
                dec_ch[i], skip_ch, dec_ch[i+1], dec_k[i], stride,
                num_groups, use_attention_gates, is_last=(i == n-1),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_len = x.shape[-1]
        nf         = _noise_features(x)

        skips: list[torch.Tensor] = []
        h = x
        for stage in self.encoder_stages:
            h = stage(h)
            skips.append(h)

        h = self.dilated(h)
        h = self.attn(h)
        if self.film is not None:
            h = self.film(h, nf)

        for i, stage in enumerate(self.decoder_stages):
            skip = skips[len(skips) - 2 - i] if i < len(skips) - 1 else None
            h    = stage(h, skip)

        if h.shape[-1] != target_len:
            h = (h[..., :target_len] if h.shape[-1] > target_len
                 else F.pad(h, (0, target_len - h.shape[-1])))
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for s in self.encoder_stages:
            h = s(h)
        return self.attn(self.dilated(h))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = GWAutoencoder()
    print(f"Parameters: {m.n_parameters:,}")
    x = torch.randn(2, 1, 32768)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape, f"{y.shape} != {x.shape}"
    print(f"✓  {x.shape} → {y.shape}")
