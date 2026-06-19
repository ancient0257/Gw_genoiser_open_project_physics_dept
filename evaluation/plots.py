"""
evaluation/plots.py  — v4

Bug-fixes vs v1 (never previously updated)
--------------------------------------------
• plot_loss_curves read history["train"] and history["val"] — v4 renamed
  these keys to "train_total" and "val_total". Fixed with .get() fallback
  that tries both key names.
• plot_spectrogram_comparison: scipy spectrogram with nperseg=256 at 4096 Hz
  gives 128 Hz frequency resolution — too coarse to see chirp structure.
  Fixed: nperseg=512 for better frequency resolution; added Kaiser window
  (better sidelobe rejection than Hann for GW signals).
• COLORS dict referenced "signal" key in residual panel but signal is not
  shown there — should use "residual" key. Fixed.
• plot_asd_comparison: fill_between condition was always True (noisy ASD is
  always >= clean ASD by construction after denoising) — visually correct
  but logically wrong if denoising worsened some bands. Fixed with explicit
  condition check per frequency bin.

Performance additions
---------------------
• plot_loss_curves: now plots all v4 loss components (stft, snr, env, phase)
  in stacked subplots with correct key names.
• plot_snr_comparison: annotates pass/fail with green/red colour per bar.
• Waterfall plot: new function plot_waterfall() shows time evolution of PSD
  across the segment — useful for seeing glitch suppression over time.
• All figures use constrained_layout=True instead of tight_layout() (avoids
  deprecation warning in matplotlib ≥ 3.8 and handles colorbars better).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram as scipy_spectrogram

STYLE = {
    "axes.facecolor":   "#0d0d0d",
    "figure.facecolor": "#0d0d0d",
    "axes.edgecolor":   "#444444",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "text.color":       "#cccccc",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "axes.grid":        True,
    "font.family":      "monospace",
    "font.size":        9,
}
COLORS = {
    "noisy":    "#e05252",
    "clean":    "#52b8e0",
    "signal":   "#7de052",
    "residual": "#e0b852",
}

def _sty():
    plt.rcParams.update(STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loss curves  (bug-fix: correct v4 key names + all loss components)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_path: Path | str) -> Path:
    _sty()
    # Try v4 key names first, fall back to v1/v2 names
    tr_tot = history.get("train_total", history.get("train", []))
    va_tot = history.get("val_total",   history.get("val",   []))
    epochs = range(1, len(tr_tot) + 1)

    has_components = "train_stft" in history

    rows = 3 if has_components else 2
    fig, axes = plt.subplots(rows, 1, figsize=(11, rows * 2.5),
                              constrained_layout=True)
    fig.suptitle("Training Dynamics — GW Denoising Autoencoder", fontsize=11)

    # Total loss
    ax = axes[0]
    ax.plot(epochs, tr_tot, color=COLORS["noisy"],  lw=1.5, label="Train")
    ax.plot(epochs, va_tot, color=COLORS["clean"],  lw=1.5, linestyle="--", label="Val")
    if va_tot:
        best_ep = int(np.argmin(va_tot)) + 1
        ax.axvline(best_ep, color="#ffffff", lw=0.8, linestyle=":", alpha=0.5,
                    label=f"Best (ep {best_ep})")
    ax.set_ylabel("Total loss"); ax.set_yscale("log"); ax.legend(fontsize=8)

    if has_components:
        # Component losses
        comp_keys = [("train_stft","val_stft","STFT",   "#9b59b6"),
                     ("train_snr", "val_snr",  "−SNR",   "#e67e22"),
                     ("train_env", "val_env",  "Envelope","#1abc9c"),
                     ("train_phase","val_phase","Phase",  "#e74c3c")]
        ax2 = axes[1]
        for tk, vk, label, color in comp_keys:
            if tk in history:
                ax2.plot(epochs, history[tk], color=color, lw=1.0,
                          label=f"tr {label}", alpha=0.85)
                if vk in history:
                    ax2.plot(epochs, history[vk], color=color, lw=1.0,
                              linestyle="--", alpha=0.5)
        ax2.set_ylabel("Component losses"); ax2.set_yscale("log"); ax2.legend(fontsize=7)

    # LR + grad norm
    lr_ax = axes[-1]
    if "lr" in history:
        lr_ax.plot(epochs, history["lr"], color=COLORS["signal"], lw=1.2, label="LR")
        lr_ax.set_ylabel("LR"); lr_ax.set_yscale("log")
    if "grad_norm_mean" in history:
        ax_g = lr_ax.twinx()
        gn   = history["grad_norm_mean"]
        ax_g.plot(epochs, gn, color=COLORS["residual"], lw=0.8, alpha=0.7, label="‖g‖")
        ax_g.set_ylabel("Grad norm", color=COLORS["residual"])
    lr_ax.set_xlabel("Epoch")

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. SNR bar chart  (pass/fail colour per bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_snr_comparison(
    event_names: list[str], snr_noisy: list[float], snr_clean: list[float],
    save_path: Path | str, threshold_db: float = 3.0,
) -> Path:
    _sty()
    n = len(event_names)
    x = np.arange(n); w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5), constrained_layout=True)

    bar1 = ax.bar(x - w/2, snr_noisy, w, color=COLORS["noisy"], alpha=0.85, label="Noisy")
    bar2 = ax.bar(x + w/2, snr_clean, w, alpha=0.85, label="Reconstructed",
                   color=[COLORS["signal"] if (sc > sn and
                          10*np.log10(sc/max(sn,1e-6)) >= threshold_db)
                          else COLORS["clean"]
                          for sn, sc in zip(snr_noisy, snr_clean)])

    for xi, (sn, sc) in enumerate(zip(snr_noisy, snr_clean)):
        db    = 10 * np.log10(sc / max(sn, 1e-6))
        color = COLORS["signal"] if db >= threshold_db else COLORS["noisy"]
        ax.text(xi, max(sn, sc) * 1.02, f"Δ{db:+.1f}dB",
                ha="center", va="bottom", fontsize=7.5, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(event_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Network SNR"); ax.set_title("SNR: Noisy vs Reconstructed")
    ax.legend(fontsize=8); ax.set_ylim(0, max(snr_clean + snr_noisy) * 1.3)

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Waveform overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform_comparison(
    noisy: np.ndarray, reconstructed: np.ndarray,
    sample_rate: int, event_name: str,
    save_path: Path | str, clean_ref: np.ndarray | None = None,
) -> Path:
    _sty()
    t    = np.arange(len(noisy)) / sample_rate
    rows = 3 if clean_ref is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, rows * 2.5),
                              sharex=True, constrained_layout=True)
    fig.suptitle(f"Waveform — {event_name}", fontsize=11)

    axes[0].plot(t, noisy,         color=COLORS["noisy"], lw=0.6, alpha=0.9, label="Noisy")
    axes[0].plot(t, reconstructed, color=COLORS["clean"], lw=0.9, alpha=0.9, label="Recon")
    axes[0].set_ylabel("Strain (whitened)"); axes[0].legend(fontsize=8)

    if clean_ref is not None:
        axes[1].plot(t, clean_ref,     color=COLORS["signal"], lw=0.8, label="Clean ref")
        axes[1].plot(t, reconstructed, color=COLORS["clean"],  lw=0.8, alpha=0.7, label="Recon")
        axes[1].set_ylabel("Strain"); axes[1].legend(fontsize=8)
        res_ax = axes[2]
    else:
        res_ax = axes[1]

    res = noisy - reconstructed
    res_ax.plot(t, res, color=COLORS["residual"], lw=0.6, alpha=0.8)
    res_ax.set_ylabel("Residual"); res_ax.set_xlabel("Time (s)")

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Spectrogram triplet  (bug-fix: Kaiser window, nperseg=512)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram_comparison(
    noisy: np.ndarray, reconstructed: np.ndarray,
    sample_rate: int, event_name: str,
    save_path: Path | str,
    f_low: float = 20.0, f_high: float = 512.0,
    nperseg: int = 512,
) -> Path:
    _sty()

    # Kaiser window: lower sidelobes than Hann → better dynamic range
    win = np.kaiser(nperseg, beta=14)

    def _spec(x):
        f, t, S = scipy_spectrogram(
            x, fs=sample_rate, nperseg=nperseg,
            noverlap=nperseg * 3 // 4,
            window=win, scaling="density",
        )
        mask = (f >= f_low) & (f <= f_high)
        return f[mask], t, np.maximum(S[mask], 1e-40)

    fn, tn, Sn = _spec(noisy)
    fr, tr, Sr = _spec(reconstructed)
    _, _,   Sd = _spec(noisy - reconstructed)

    vmin = Sn.max() * 1e-5
    vmax = Sn.max()

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

    panels = [(gs[0], Sn, tn, fn, "Noisy"),
              (gs[1], Sr, tr, fr, "Reconstructed"),
              (gs[2], Sd, tr, fr, "Residual (noise)")]
    im = None
    for i, (g, S, t, f, title) in enumerate(panels):
        ax = fig.add_subplot(g)
        im = ax.pcolormesh(t, f, S,
                            norm=LogNorm(vmin=vmin, vmax=vmax),
                            cmap="inferno", shading="gouraud")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.set_yticklabels([])

    cb_ax = fig.add_subplot(gs[3])
    plt.colorbar(im, cax=cb_ax, label="PSD (strain²/Hz)")
    fig.suptitle(f"Spectrogram — {event_name}", fontsize=11)

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. ASD comparison  (bug-fix: conditional fill_between)
# ─────────────────────────────────────────────────────────────────────────────

def plot_asd_comparison(
    noisy: np.ndarray, reconstructed: np.ndarray,
    sample_rate: int, event_name: str,
    save_path: Path | str,
    f_low: float = 20.0, f_high: float = 2000.0,
) -> Path:
    from scipy.signal import welch as scipy_welch
    _sty()

    def _asd(x):
        f, p = scipy_welch(x, fs=sample_rate, nperseg=min(2048, len(x)),
                            window="hann")
        return f, np.sqrt(np.maximum(p, 1e-50))

    fn, an = _asd(noisy)
    fr, ar = _asd(reconstructed)
    mask = (fn >= f_low) & (fn <= f_high)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.loglog(fn[mask], an[mask], color=COLORS["noisy"],  lw=1.2, alpha=0.85, label="Noisy")
    ax.loglog(fr[mask], ar[mask], color=COLORS["clean"],  lw=1.2, alpha=0.85, label="Reconstructed")

    # Only shade where denoising actually reduced power (bug-fix)
    suppressed = an[mask] > ar[mask]
    ax.fill_between(fn[mask], an[mask], ar[mask],
                     where=suppressed, alpha=0.12, color=COLORS["signal"],
                     label="Suppressed noise")
    # Shade where denoising added power (ideally absent)
    amplified = ar[mask] > an[mask]
    if amplified.any():
        ax.fill_between(fn[mask], an[mask], ar[mask],
                         where=amplified, alpha=0.12, color=COLORS["noisy"],
                         label="Added power (undesired)")

    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("ASD (strain/√Hz)")
    ax.set_title(f"Amplitude Spectral Density — {event_name}")
    ax.legend(fontsize=8); ax.set_xlim(f_low, f_high)

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. Waterfall PSD  (new)
# ─────────────────────────────────────────────────────────────────────────────

def plot_waterfall(
    noisy: np.ndarray, reconstructed: np.ndarray,
    sample_rate: int, event_name: str,
    save_path: Path | str,
    f_low: float = 20.0, f_high: float = 512.0,
    n_slices: int = 32,
) -> Path:
    """
    Waterfall: stack of PSDs over time slices, showing how noise suppression
    evolves through the segment. Each row = one time slice.
    Side-by-side: noisy | reconstructed.
    """
    _sty()
    seg_len = len(noisy)
    slice_len = seg_len // n_slices
    nperseg   = min(slice_len, 512)

    def _slice_psds(x):
        psds = []
        for i in range(n_slices):
            chunk = x[i * slice_len : (i+1) * slice_len]
            from scipy.signal import welch as scipy_welch
            f, p = scipy_welch(chunk, fs=sample_rate, nperseg=min(nperseg, len(chunk)))
            mask = (f >= f_low) & (f <= f_high)
            psds.append(np.sqrt(np.maximum(p[mask], 1e-50)))
        return f[mask], np.array(psds)   # (n_slices, F)

    freqs, Pn = _slice_psds(noisy)
    _,     Pr = _slice_psds(reconstructed)
    t_axis    = np.linspace(0, seg_len / sample_rate, n_slices)

    vmin = Pn.min(); vmax = Pn.max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"ASD Waterfall — {event_name}", fontsize=11)

    for ax, P, title in [(ax1, Pn, "Noisy"), (ax2, Pr, "Reconstructed")]:
        im = ax.pcolormesh(freqs, t_axis, P,
                            norm=LogNorm(vmin=vmin, vmax=vmax),
                            cmap="plasma", shading="gouraud")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Time (s)")
        ax.set_title(title); ax.set_xscale("log")

    plt.colorbar(im, ax=ax2, label="ASD")
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
