"""
utils/snr.py  — v4

Bug-fixes vs v1 (never previously updated)
--------------------------------------------
• peak_matched_filter_snr: hp.cyclic_time_shift(hp.start_time) no longer
  exists in PyCBC ≥ 2.0 — method was removed. Fixed: use hp.prepend_zeros()
  or simply resize directly; PyCBC matched_filter handles the time shift.
• ppsd.welch() kwarg 'seg_stride' removed in PyCBC ≥ 2.0. Fixed: use
  pycbc.psd.welch(ts, seg_len=...) — stride defaults to seg_len//2.
• broadband_snr middle-20% heuristic assumes merger is centred in the
  segment — untrue for real events where coalescence time varies.
  Fixed: scan for peak-power window using a sliding RMS.
• snr_improvement_db returned 0.0 when either SNR was 0 — silently hides
  denoising that destroyed signal. Fixed: return -inf when clean SNR is 0
  and noisy SNR > 0 (genuine degradation).

New functions
-------------
• matched_filter_overlap(): faithfulness ⟨h_rec | h_true⟩ / √(⟨h_rec|h_rec⟩·⟨h_true|h_true⟩)
  Standard GW metric; 1.0 = perfect, < 0.97 = unacceptable for PE.
• psd_weighted_snr(): compute SNR using actual segment PSD (not a proxy)
  without needing a template — used when PyCBC unavailable but a PSD is known.
"""

from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Peak-window broadband SNR  (bug-fix: scan for peak, not fixed window)
# ─────────────────────────────────────────────────────────────────────────────

def broadband_snr(
    strain: np.ndarray,
    sample_rate: int,
    window_s: float = 0.5,
    noise_frac: float = 0.2,
) -> float:
    """
    Template-free SNR: peak sliding-window RMS / off-source RMS.

    Scans the segment with a `window_s`-second window to find the peak-power
    location, then uses that as the signal window. Off-source noise is taken
    from the first `noise_frac` fraction of the segment.

    Returns linear SNR.
    """
    n     = len(strain)
    w     = max(1, int(window_s * sample_rate))
    s2    = strain ** 2

    # Sliding sum via cumsum  (O(N))
    cs    = np.concatenate([[0.0], np.cumsum(s2)])
    rms   = np.sqrt((cs[w:] - cs[:-w]) / w)
    peak  = int(np.argmax(rms))

    sig_power   = float(rms[peak] ** 2)
    noise_end   = int(noise_frac * n)
    noise_power = float(np.mean(s2[:max(1, noise_end)]))

    if noise_power < 1e-30:
        return 0.0
    return float(np.sqrt(sig_power / max(noise_power, 1e-30)))


def snr_db(snr_linear: float) -> float:
    if snr_linear <= 0:
        return float("-inf")
    return 10.0 * np.log10(snr_linear)


def snr_improvement_db(snr_noisy: float, snr_clean: float) -> float:
    """
    ΔdB = 10·log10(SNR_clean / SNR_noisy)
    Returns -inf if clean SNR collapsed to 0 (genuine degradation).
    Returns 0 if noisy SNR was already 0 (undefined improvement).
    """
    if snr_noisy <= 0:
        return 0.0
    if snr_clean <= 0:
        return float("-inf")
    return 10.0 * np.log10(snr_clean / snr_noisy)


# ─────────────────────────────────────────────────────────────────────────────
# PSD-weighted SNR  (no template needed)
# ─────────────────────────────────────────────────────────────────────────────

def psd_weighted_snr(
    strain: np.ndarray,
    psd:    np.ndarray,
    freqs:  np.ndarray,
    sample_rate: int,
    f_low:  float = 20.0,
    f_high: float = 2000.0,
) -> float:
    """
    ρ = sqrt( 4 ∫ |h̃(f)|² / S(f) df )

    Uses the actual measured PSD rather than the f^{-7/3} approximation.
    More accurate than broadband_snr; doesn't require a template.
    """
    n       = len(strain)
    dt      = 1.0 / sample_rate
    h_fd    = np.fft.rfft(strain) * dt
    fft_f   = np.fft.rfftfreq(n, d=dt)

    psd_i   = np.interp(fft_f, freqs, psd, left=1.0, right=1.0)
    psd_i   = np.maximum(psd_i, 1e-50)

    mask    = (fft_f >= f_low) & (fft_f <= f_high)
    intg    = 4.0 * np.abs(h_fd[mask]) ** 2 / psd_i[mask]
    df      = fft_f[1] - fft_f[0] if len(fft_f) > 1 else 1.0
    return float(np.sqrt(np.maximum(0.0, np.trapezoid(intg, dx=df))))


# ─────────────────────────────────────────────────────────────────────────────
# Matched-filter overlap (faithfulness)
# ─────────────────────────────────────────────────────────────────────────────

def matched_filter_overlap(
    h_rec:  np.ndarray,
    h_true: np.ndarray,
    psd:    np.ndarray,
    freqs:  np.ndarray,
    sample_rate: int,
    f_low:  float = 20.0,
    f_high: float = 2000.0,
) -> float:
    """
    Faithfulness (match) between reconstructed and true waveform:
        M = max_t ⟨h_rec | h_true⟩ / sqrt(⟨h_rec|h_rec⟩ · ⟨h_true|h_true⟩)

    Maximised over time shifts via FFT (equivalent to matched filter).
    M = 1.0 → perfect reconstruction.  M < 0.97 → unacceptable for PE.

    Returns NaN if either signal has zero norm.
    """
    n    = len(h_rec)
    dt   = 1.0 / sample_rate
    df   = 1.0 / (n * dt)

    Hr   = np.fft.rfft(h_rec)
    Ht   = np.fft.rfft(h_true)
    fft_f = np.fft.rfftfreq(n, d=dt)

    psd_i = np.maximum(np.interp(fft_f, freqs, psd, left=1.0, right=1.0), 1e-50)
    mask  = (fft_f >= f_low) & (fft_f <= f_high)

    # Inner product ⟨a|b⟩ = 4·Re(∫ ã*(f)·b̃(f)/S(f) df)
    def _inner(A, B):
        return float(4.0 * np.real(np.sum(np.conj(A[mask]) * B[mask] / psd_i[mask])) * df)

    norm_rec  = _inner(Hr, Hr)
    norm_true = _inner(Ht, Ht)
    if norm_rec <= 0 or norm_true <= 0:
        return float("nan")

    # Time-maximised overlap: take abs of IFFT of the cross-spectrum
    cross = np.conj(Hr) * Ht / psd_i
    cross[~mask] = 0.0
    overlap_ts = np.fft.irfft(cross, n=n)
    peak_cross  = float(4.0 * np.max(np.abs(overlap_ts)) * df)

    return peak_cross / np.sqrt(norm_rec * norm_true)


# ─────────────────────────────────────────────────────────────────────────────
# PyCBC matched-filter SNR  (bug-fixed)
# ─────────────────────────────────────────────────────────────────────────────

def peak_matched_filter_snr(
    strain: np.ndarray,
    sample_rate: int,
    mass1: float = 30.0,
    mass2: float = 30.0,
    f_low: float = 20.0,
    approximant: str = "IMRPhenomD",
) -> float:
    """
    Peak matched-filter SNR via PyCBC.

    Bug-fixes vs v1:
    • hp.cyclic_time_shift removed — use hp directly, PyCBC handles alignment
    • ppsd.welch() 'seg_stride' kwarg removed in PyCBC ≥2.0; using positional args
    • Template resize now uses pycbc resize API correctly
    Falls back to broadband_snr if PyCBC unavailable or waveform fails.
    """
    try:
        import pycbc.types as pt
        import pycbc.filter as pf
        import pycbc.waveform as pw
        import pycbc.psd as ppsd
    except ImportError:
        log.warning("PyCBC not available — broadband SNR fallback.")
        return broadband_snr(strain, sample_rate)

    delta_t  = 1.0 / sample_rate
    duration = len(strain) / sample_rate
    delta_f  = 1.0 / duration

    ts = pt.TimeSeries(strain.astype(np.float64), delta_t=delta_t)

    try:
        hp, _ = pw.get_fd_waveform(
            approximant=approximant,
            mass1=mass1, mass2=mass2,
            delta_f=delta_f,
            f_lower=f_low,
        )
    except Exception as exc:
        log.warning(f"Waveform gen failed ({exc}) — broadband fallback.")
        return broadband_snr(strain, sample_rate)

    seg_len = len(ts)

    # Bug-fix: ppsd.welch positional API (no seg_stride kwarg in PyCBC ≥2.0)
    try:
        psd_obj = ppsd.welch(ts, seg_len=seg_len // 4)
    except TypeError:
        psd_obj = ppsd.welch(ts, seg_len // 4, seg_len // 8)

    # Resize template to freq-series matching the strain length
    tlen = seg_len // 2 + 1
    if len(hp) < tlen:
        hp.resize(tlen)
    else:
        hp = hp[:tlen]

    try:
        snr_ts = pf.matched_filter(
            hp, ts.to_frequencyseries(),
            psd=psd_obj, low_frequency_cutoff=f_low,
        )
        return float(abs(snr_ts).max())
    except Exception as exc:
        log.warning(f"Matched filter failed ({exc}) — broadband fallback.")
        return broadband_snr(strain, sample_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Noise power reduction  (unchanged, correct)
# ─────────────────────────────────────────────────────────────────────────────

def noise_power_reduction(
    noisy: np.ndarray, clean: np.ndarray,
    sample_rate: int,
    f_low: float = 30.0, f_high: float = 300.0,
) -> float:
    """NPR = 1 - ∫PSD_clean df / ∫PSD_noisy df  in [f_low, f_high]."""
    from scipy.signal import welch as scipy_welch
    nperseg = min(1024, len(noisy))
    fn, pn  = scipy_welch(noisy, fs=sample_rate, nperseg=nperseg)
    fc, pc  = scipy_welch(clean, fs=sample_rate, nperseg=nperseg)
    mask    = (fn >= f_low) & (fn <= f_high)
    pn_int  = float(np.trapezoid(pn[mask], fn[mask]))
    pc_int  = float(np.trapezoid(pc[mask], fc[mask]))
    if pn_int < 1e-30:
        return 0.0
    return float(1.0 - pc_int / pn_int)


# ─────────────────────────────────────────────────────────────────────────────
# Catalogue-aware SNR evaluation
# ─────────────────────────────────────────────────────────────────────────────

def catalogue_snr(
    event_name: str,
    strain:     np.ndarray,
    sample_rate: int,
    f_low:      float = 20.0,
    use_pycbc:  bool  = False,
) -> tuple[float, float, float]:
    """
    Compute SNR for a named GWTC event using catalogue-sourced masses.

    Returns (snr, m1, m2) where m1/m2 are from MASS_PRIORS if available,
    otherwise falls back to (30, 30) defaults.

    Dispatches to peak_matched_filter_snr (PyCBC) or psd_weighted_snr
    based on use_pycbc flag.
    """
    try:
        from data.downloader import MASS_PRIORS
        m1, m2 = MASS_PRIORS.get(event_name, (30.0, 30.0))
    except ImportError:
        m1, m2 = 30.0, 30.0

    if use_pycbc:
        snr = peak_matched_filter_snr(strain, sample_rate, m1, m2, f_low)
    else:
        snr = broadband_snr(strain, sample_rate)

    return snr, m1, m2
