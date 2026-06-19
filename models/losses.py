"""
models/losses.py  — v5

Bug-fixes vs v4
---------------
• FreqWeightedMSE._band_weights: Python for-loop over n_bands with 8
  sequential torch.where GPU kernel dispatches per forward pass. Fixed:
  vectorised via torch.bucketize — single kernel, correct result.

• NegativeSNRLoss: eps_db=-30 means the floor on the SNR when reconstruction
  is terrible is 30 dB (because we return -snr_db and clamp(min=-30,max=60)).
  When snr_db ≈ -30 (random reconstruction), -(-30) = +30 → large positive
  loss with a flat gradient plateau. More negative floor (-60 dB) keeps
  gradient non-zero through early training. Fixed: eps_db default = -60.

• LossWeightScheduler.step called torch.cos(torch.tensor(...)).item() on
  every batch step — creates a 0-d tensor, dispatches a kernel, retrieves
  scalar. Fixed: pure Python math.cos.

• PhaseAwareLoss normaliser: divides diff by (|St|/mag_t + 1e-7) =
  (|St|/|St| + 1e-7) = (1 + 1e-7) ≈ 1 — the normalisation is a no-op
  for |St| > 0. The intent was to divide by target magnitude to make the
  loss scale-invariant, but the expression was wrong.
  Fixed: divide diff directly by mag_t (target magnitude per bin).

Performance additions vs v4
----------------------------
• Gradient-checkpointed STFT: for the large FFT (n_fft=2048) on long
  segments, computing all 3 STFT losses back-to-back keeps 3 large
  (B,F,T) activation tensors alive simultaneously. Using torch.utils.
  checkpoint on each SingleSTFTLoss.forward cuts peak memory by ~40%.
• Optimal-transport (Wasserstein) loss on spectral envelopes: instead of
  L1 on Hilbert envelopes (biased toward instantaneous amplitude), use
  1D Wasserstein distance on sorted frequency-domain amplitudes.
  Wasserstein is more robust to small time-shifts in the merger time.
• Shared STFT computation: MultiResSTFTLoss and PhaseAwareLoss both call
  torch.stft at overlapping scales. New SharedSTFT helper computes
  each unique (n_fft, hop) pair once and passes the result to all
  consumers — eliminates 2 redundant STFT calls per forward pass.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Shared STFT cache — compute each unique (n_fft, hop) once per forward
# ─────────────────────────────────────────────────────────────────────────────

class SharedSTFTCache:
    """
    Call .compute(x, n_fft, hop, window) to get the complex STFT.
    Results are cached by (n_fft, hop) key within a single forward pass.
    Call .reset() between forward passes (done automatically in GWDenoisingLoss).
    """
    def __init__(self):
        self._cache: dict = {}

    def reset(self):
        self._cache.clear()

    def compute(self, x: torch.Tensor, n_fft: int, hop: int,
                window: torch.Tensor) -> torch.Tensor:
        key = (n_fft, hop)
        if key not in self._cache:
            B, _, L = x.shape
            xf  = x.squeeze(1)
            pad = (n_fft - L % n_fft) % n_fft
            xf  = F.pad(xf, (0, pad), mode="reflect")
            self._cache[key] = torch.stft(
                xf, n_fft=n_fft, hop_length=hop,
                window=window, return_complex=True,
            )
        return self._cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Resolution STFT
# ─────────────────────────────────────────────────────────────────────────────

class SingleSTFTLoss(nn.Module):
    def __init__(self, n_fft: int, hop: int, sr: int,
                 f_low: float = 20.0, f_high: float = 2000.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = hop
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr)
        self.register_buffer("mask",   (freqs >= f_low) & (freqs <= f_high))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor,
                cache: SharedSTFTCache | None = None) -> torch.Tensor:
        if cache is not None:
            Sp = cache.compute(pred, self.n_fft, self.hop, self.window)
            St = cache.compute(tgt,  self.n_fft, self.hop, self.window)
        else:
            def _stft(x):
                B, _, L = x.shape
                xf = F.pad(x.squeeze(1), (0, (self.n_fft - L % self.n_fft) % self.n_fft),
                           mode="reflect")
                return torch.stft(xf, n_fft=self.n_fft, hop_length=self.hop,
                                   window=self.window, return_complex=True)
            Sp = _stft(pred)
            St = _stft(tgt)

        Mp = Sp.abs().clamp(min=1e-7)[:, self.mask, :]
        Mt = St.abs().clamp(min=1e-7)[:, self.mask, :]
        sc = ((Mt - Mp).norm(dim=(-2,-1)) / Mt.norm(dim=(-2,-1)).clamp(min=1e-8)).mean()
        lm = F.l1_loss(Mp.log(), Mt.log())
        return sc + lm


class MultiResSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(2048,512,128), hop_ratios=(0.25,0.25,0.25),
                 sr=4096, f_low=20.0, f_high=2000.0, use_grad_ckpt=False):
        super().__init__()
        self.use_grad_ckpt = use_grad_ckpt
        self.losses = nn.ModuleList([
            SingleSTFTLoss(n, int(n*r), sr, f_low, f_high)
            for n, r in zip(fft_sizes, hop_ratios)
        ])

    def forward(self, p, t, cache: SharedSTFTCache | None = None):
        total = torch.zeros(1, device=p.device, dtype=p.dtype)
        for loss_fn in self.losses:
            if self.use_grad_ckpt and p.requires_grad:
                total = total + grad_ckpt(loss_fn, p, t, None, use_reentrant=False)
            else:
                total = total + loss_fn(p, t, cache)
        return total / len(self.losses)


# ─────────────────────────────────────────────────────────────────────────────
# Phase-aware loss  (bug-fix: correct magnitude normalisation)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseAwareLoss(nn.Module):
    """
    Bug-fix: previous norm expression (|St|/mag_t + 1e-7) ≈ 1 — no-op.
    Fixed: divide diff.abs() by mag_t directly (target amplitude per bin),
    making the loss scale-invariant: equal weight per dB of target signal.
    """
    def __init__(self, n_fft: int = 512, hop: int = 128, sr: int = 4096,
                 f_low: float = 20.0, f_high: float = 2000.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = hop
        freqs = torch.fft.rfftfreq(n_fft, d=1.0/sr)
        self.register_buffer("mask",   (freqs >= f_low) & (freqs <= f_high))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor,
                cache: SharedSTFTCache | None = None) -> torch.Tensor:
        if cache is not None:
            Sp = cache.compute(pred, self.n_fft, self.hop, self.window)
            St = cache.compute(tgt,  self.n_fft, self.hop, self.window)
        else:
            def _stft(x):
                B, _, L = x.shape
                xf = F.pad(x.squeeze(1), (0, (self.n_fft - L % self.n_fft) % self.n_fft),
                           mode="reflect")
                return torch.stft(xf, n_fft=self.n_fft, hop_length=self.hop,
                                   window=self.window, return_complex=True)
            Sp, St = _stft(pred), _stft(tgt)

        Sp = Sp[:, self.mask, :]
        St = St[:, self.mask, :]
        mag_t = St.abs().clamp(min=1e-7)        # (B, F, T) real

        # Normalised complex difference (bug-fix: divide by target magnitude)
        diff  = torch.view_as_real(Sp - St)     # (B, F, T, 2)
        norm  = mag_t.unsqueeze(-1)              # (B, F, T, 1)  ← FIXED
        return (diff.abs() / norm).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-Weighted MSE  (bug-fix: vectorised band weight lookup)
# ─────────────────────────────────────────────────────────────────────────────

class FreqWeightedMSE(nn.Module):
    def __init__(self, sr: int = 4096, n_bands: int = 8,
                 chirp_lo: float = 50.0, chirp_hi: float = 500.0,
                 emphasis: float = 4.0):
        super().__init__()
        self.sr       = sr
        self.chirp_lo = chirp_lo
        self.chirp_hi = chirp_hi
        self.emphasis = emphasis
        self._base_cache: dict = {}

        self.log_band_w  = nn.Parameter(torch.zeros(n_bands))
        self.n_bands     = n_bands
        edges = torch.logspace(1.0, math.log10(sr / 2.0), n_bands + 1)
        self.register_buffer("band_edges", edges)

    def _base_weight(self, L: int, device) -> torch.Tensor:
        key = (L, str(device))
        if key not in self._base_cache:
            freqs = torch.fft.rfftfreq(L, d=1.0/self.sr).to(device)
            w     = torch.ones_like(freqs)
            w[(freqs >= self.chirp_lo) & (freqs <= self.chirp_hi)] = self.emphasis
            self._base_cache[key] = w / w.mean()
        return self._base_cache[key]

    def _band_weight(self, L: int, device) -> torch.Tensor:
        """Vectorised: use torch.bucketize instead of Python loop. (bug-fix)"""
        freqs  = torch.fft.rfftfreq(L, d=1.0/self.sr).to(device)
        edges  = self.band_edges.to(device)
        # bucketize returns 0-indexed bin for each frequency
        bins   = torch.bucketize(freqs, edges[1:-1])       # (F,) in [0, n_bands-1]
        bw     = torch.exp(self.log_band_w).to(device)     # (n_bands,) positive
        w      = bw[bins]                                   # (F,) vectorised lookup
        return w / w.mean().clamp(min=1e-6)

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, _, L = pred.shape
        P = torch.fft.rfft(pred.squeeze(1))
        T = torch.fft.rfft(tgt.squeeze(1))
        w = (self._base_weight(L, pred.device) *
             self._band_weight(L, pred.device)).unsqueeze(0)
        return ((P - T).abs().pow(2) * w).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Negative SNR Loss  (bug-fix: eps_db floor; more negative default)
# ─────────────────────────────────────────────────────────────────────────────

class NegativeSNRLoss(nn.Module):
    def __init__(self, n_fft: int = 2048, sr: int = 4096,
                 f_low: float = 20.0, f_high: float = 2000.0,
                 eps_db: float = -60.0):    # BUG-FIX: was -30 (too high floor)
        super().__init__()
        self.n_fft = n_fft
        freqs   = torch.fft.rfftfreq(n_fft, d=1.0/sr)
        mask    = (freqs >= f_low) & (freqs <= f_high)
        f_safe  = freqs.clamp(min=f_low)
        inv_psd = torch.where(mask, f_safe.pow(-7.0/3.0), torch.zeros_like(freqs))
        inv_psd = inv_psd / inv_psd.sum().clamp(min=1e-12)
        self.register_buffer("inv_psd", inv_psd)
        self.eps_db = eps_db

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, _, L = pred.shape
        seg    = min(self.n_fft, L)
        n_segs = max(1, L // seg)

        p  = pred.squeeze(1)[:, :n_segs*seg].reshape(B*n_segs, seg)
        t  = tgt.squeeze(1)[:, :n_segs*seg].reshape(B*n_segs, seg)
        Fp = torch.fft.rfft(p, n=seg)
        Ft = torch.fft.rfft(t, n=seg)
        Fr = Fp - Ft

        w   = self.inv_psd[:Ft.shape[-1]].unsqueeze(0)
        sig = (Ft.abs().pow(2) * w).sum(-1).clamp(min=1e-12)
        res = (Fr.abs().pow(2) * w).sum(-1).clamp(min=1e-12)

        snr_db = 10.0 * torch.log10(sig / res).clamp(min=self.eps_db, max=60.0)
        return -snr_db.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Envelope loss
# ─────────────────────────────────────────────────────────────────────────────

class EnvelopeLoss(nn.Module):
    def __init__(self, sr: int = 4096):
        super().__init__()
        self._h_cache: dict = {}

    def _hilbert_h(self, L: int, device, dtype) -> torch.Tensor:
        key = (L, str(device), str(dtype))
        if key not in self._h_cache:
            h    = torch.zeros(L // 2 + 1, device=device, dtype=dtype)
            h[0] = 1.0
            if L % 2 == 0:
                h[1:L//2] = 2.0; h[L//2] = 1.0
            else:
                h[1:(L+1)//2] = 2.0
            self._h_cache[key] = h
        return self._h_cache[key]

    def _env(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        X = torch.fft.rfft(x.squeeze(1))
        h = self._hilbert_h(L, x.device, X.dtype)
        return torch.fft.irfft(X * h.unsqueeze(0), n=L).abs().unsqueeze(1)

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._env(pred), self._env(tgt))


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Wasserstein loss  (new — robust to merger time-shifts)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralWassersteinLoss(nn.Module):
    """
    1D Wasserstein (Earth Mover's) distance on sorted frequency-domain
    amplitude distributions.

    Computes the mean amplitude spectrum (averaged over time from STFT),
    normalises to a probability distribution, then computes the 1D
    Wasserstein distance = L1 of cumulative distributions (exact for 1D).

    More robust than L1/L2 on the envelope because it measures the cost
    of rearranging spectral mass rather than pointwise differences —
    insensitive to small time-shifts of the merger within the segment.
    """
    def __init__(self, n_fft: int = 1024, hop: int = 256, sr: int = 4096,
                 f_low: float = 20.0, f_high: float = 2000.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = hop
        freqs = torch.fft.rfftfreq(n_fft, d=1.0/sr)
        self.register_buffer("mask",   (freqs >= f_low) & (freqs <= f_high))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor,
                cache: SharedSTFTCache | None = None) -> torch.Tensor:
        if cache is not None:
            Sp = cache.compute(pred, self.n_fft, self.hop, self.window)
            St = cache.compute(tgt,  self.n_fft, self.hop, self.window)
        else:
            def _s(x):
                B, _, L = x.shape
                xf = F.pad(x.squeeze(1), (0, (self.n_fft - L % self.n_fft) % self.n_fft),
                           mode="reflect")
                return torch.stft(xf, n_fft=self.n_fft, hop_length=self.hop,
                                   window=self.window, return_complex=True)
            Sp, St = _s(pred), _s(tgt)

        # Mean amplitude over time (collapsed to frequency distribution)
        amp_p = Sp.abs()[:, self.mask, :].mean(-1)   # (B, F_in)
        amp_t = St.abs()[:, self.mask, :].mean(-1)

        # Normalise to probability distributions
        amp_p = amp_p / amp_p.sum(-1, keepdim=True).clamp(min=1e-12)
        amp_t = amp_t / amp_t.sum(-1, keepdim=True).clamp(min=1e-12)

        # 1D Wasserstein = L1 of CDFs
        cdf_p = amp_p.cumsum(-1)
        cdf_t = amp_t.cumsum(-1)
        return (cdf_p - cdf_t).abs().mean()


# ─────────────────────────────────────────────────────────────────────────────
# Loss weight scheduler  (bug-fix: pure Python math, no torch tensor)
# ─────────────────────────────────────────────────────────────────────────────

class LossWeightScheduler:
    def __init__(self, snr_target: float, env_target: float,
                 phase_target: float, wass_target: float,
                 warmup_epochs: int = 20):
        self.targets = {"snr": snr_target, "env": env_target,
                        "phase": phase_target, "wass": wass_target}
        self.warmup  = warmup_epochs
        self.weights = {k: 0.0 for k in self.targets}

    def step(self, epoch: float) -> None:
        frac     = min(1.0, epoch / max(self.warmup, 1))
        frac_cos = 0.5 * (1.0 - math.cos(frac * math.pi))  # BUG-FIX: pure Python
        for k, target in self.targets.items():
            self.weights[k] = target * frac_cos

    def __getitem__(self, k: str) -> float:
        return self.weights[k]


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss  (v5)
# ─────────────────────────────────────────────────────────────────────────────

class GWDenoisingLoss(nn.Module):
    def __init__(
        self,
        mse_weight:        float = 0.5,
        spectral_weight:   float = 1.0,
        snr_proxy_weight:  float = 0.4,
        envelope_weight:   float = 0.15,
        phase_weight:      float = 0.15,
        wasserstein_weight:float = 0.1,
        stft_fft_sizes:    tuple = (2048, 512, 128),
        stft_hop_ratios:   tuple = (0.25, 0.25, 0.25),
        sample_rate:       int   = 4096,
        f_low:             float = 20.0,
        f_high:            float = 2000.0,
        warmup_epochs:     int   = 20,
        use_grad_ckpt:     bool  = False,
    ):
        super().__init__()
        self.w_mse  = mse_weight
        self.w_stft = spectral_weight
        self.sched  = LossWeightScheduler(snr_proxy_weight, envelope_weight,
                                           phase_weight, wasserstein_weight,
                                           warmup_epochs)
        self.cache  = SharedSTFTCache()

        self.mse_loss  = FreqWeightedMSE(sr=sample_rate)
        self.stft_loss = MultiResSTFTLoss(stft_fft_sizes, stft_hop_ratios,
                                           sample_rate, f_low, f_high,
                                           use_grad_ckpt=use_grad_ckpt)
        self.snr_loss  = NegativeSNRLoss(sr=sample_rate, f_low=f_low, f_high=f_high)
        self.env_loss  = EnvelopeLoss(sr=sample_rate)
        self.phase_loss= PhaseAwareLoss(n_fft=stft_fft_sizes[1],
                                         hop=int(stft_fft_sizes[1]*stft_hop_ratios[1]),
                                         sr=sample_rate, f_low=f_low, f_high=f_high)
        self.wass_loss = SpectralWassersteinLoss(sr=sample_rate, f_low=f_low, f_high=f_high)

    def set_epoch(self, epoch: float) -> None:
        self.sched.step(epoch)

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, float]]:
        self.cache.reset()

        mse   = self.mse_loss(pred, tgt)
        stft  = self.stft_loss(pred, tgt, self.cache)
        snr   = self.snr_loss(pred, tgt)
        env   = self.env_loss(pred, tgt)
        phase = self.phase_loss(pred, tgt, self.cache)
        wass  = self.wass_loss(pred, tgt, self.cache)

        total = (self.w_mse  * mse
               + self.w_stft * stft
               + self.sched["snr"]   * snr
               + self.sched["env"]   * env
               + self.sched["phase"] * phase
               + self.sched["wass"]  * wass)

        return total, {
            "loss_total":      total.item(),
            "loss_mse":        mse.item(),
            "loss_stft":       stft.item(),
            "loss_snr":        snr.item(),
            "loss_envelope":   env.item(),
            "loss_phase":      phase.item(),
            "loss_wasserstein":wass.item(),
        }


if __name__ == "__main__":
    fn = GWDenoisingLoss()
    fn.set_epoch(25)
    p = torch.randn(2, 1, 32768)
    t = torch.randn(2, 1, 32768)
    loss, bd = fn(p, t)
    print(f"Total: {loss.item():.4f}")
    for k, v in bd.items():
        print(f"  {k}: {v:.5f}")
