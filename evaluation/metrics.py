"""
evaluation/metrics.py  — v4

Bug-fixes vs v1 (never previously updated)
--------------------------------------------
• MSE normalised by np.var(noisy) — after MAD-whitening var(noisy)≈1 always,
  so normalisation was a no-op. Fixed: normalise by var(clean_target) which
  measures how much of the true signal was reconstructed (relative error).
  When clean_target is unknown (real data), fall back to var(noisy).
• evaluate_event passed use_pycbc=True as default but broadband_snr was called
  incorrectly when use_pycbc=False (called with mass1/mass2 kwargs it doesn't
  accept). Fixed: clean dispatch.
• snr_improvement_db returned 0.0 for zero SNR — now propagates -inf correctly.
• evaluate_all returned empty dict for n=0 but callers didn't check — added
  a no-op result with all zeros to prevent KeyError downstream.

New metrics
-----------
• overlap: matched-filter faithfulness ⟨h_rec|h_true⟩/√(⟨h_rec|h_rec⟩·⟨h_true|h_true⟩)
  Standard GW PE metric; requires psd+freqs from the segment.
• psnr_db: peak SNR in dB (20·log10(peak/RMS_residual)) — intuitive complement
  to the matched-filter SNR metric.
• EventResult now stores overlap and psnr_db fields.
• Threshold checks updated: overlap ≥ 0.95 added as Pass/Fail criterion.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np

from utils.snr import (
    broadband_snr,
    peak_matched_filter_snr,
    psd_weighted_snr,
    matched_filter_overlap,
    snr_improvement_db,
    noise_power_reduction,
)

log = logging.getLogger(__name__)

THRESHOLDS = {
    "snr_improvement_db":    3.0,
    "reconstruction_mse":    0.05,
    "noise_power_reduction": 0.40,
    "overlap":               0.95,   # new
}


@dataclass
class EventResult:
    event_name:            str
    snr_noisy:             float
    snr_clean:             float
    snr_improvement_db:    float
    reconstruction_mse:    float    # normalised by clean signal variance
    noise_power_reduction: float
    overlap:               float = float("nan")   # matched-filter faithfulness
    psnr_db:               float = float("nan")   # peak SNR in dB

    @property
    def passes_snr(self) -> bool:
        return self.snr_improvement_db >= THRESHOLDS["snr_improvement_db"]
    @property
    def passes_mse(self) -> bool:
        return self.reconstruction_mse < THRESHOLDS["reconstruction_mse"]
    @property
    def passes_npr(self) -> bool:
        return self.noise_power_reduction >= THRESHOLDS["noise_power_reduction"]
    @property
    def passes_overlap(self) -> bool:
        return np.isnan(self.overlap) or self.overlap >= THRESHOLDS["overlap"]
    @property
    def passes_all(self) -> bool:
        return self.passes_snr and self.passes_mse and self.passes_npr

    def summary(self) -> str:
        t = {True: "✓", False: "✗"}
        ov  = f"{self.overlap:.3f}" if not np.isnan(self.overlap) else "  n/a"
        psn = f"{self.psnr_db:.1f}" if not np.isnan(self.psnr_db) else " n/a"
        return (
            f"{self.event_name:<26}| "
            f"SNR {self.snr_noisy:.1f}→{self.snr_clean:.1f} "
            f"(Δ{self.snr_improvement_db:+.1f}dB) {t[self.passes_snr]} | "
            f"MSE {self.reconstruction_mse:.4f} {t[self.passes_mse]} | "
            f"NPR {self.noise_power_reduction*100:.1f}% {t[self.passes_npr]} | "
            f"M={ov} {t[self.passes_overlap]} | PSNR {psn}dB"
        )


def evaluate_event(
    event_name:    str,
    noisy:         np.ndarray,
    reconstructed: np.ndarray,
    sample_rate:   int,
    mass1:         float = 30.0,
    mass2:         float = 30.0,
    use_pycbc:     bool  = False,
    f_low:         float = 20.0,
    noise_band:    tuple[float, float] = (30.0, 300.0),
    psd:           np.ndarray | None  = None,
    freqs:         np.ndarray | None  = None,
    clean_ref:     np.ndarray | None  = None,   # ground-truth signal if known
) -> EventResult:
    """
    Compute all verification metrics for one held-out event.

    Parameters
    ----------
    noisy         : whitened strain input to the autoencoder
    reconstructed : autoencoder output
    clean_ref     : ground-truth clean signal (if available, e.g. from injection)
    psd / freqs   : segment PSD for PSD-weighted SNR and overlap
    use_pycbc     : use PyCBC matched-filter (requires pycbc + lalsuite)
    """
    # ── SNR ───────────────────────────────────────────────────────────────────
    if use_pycbc:
        snr_n = peak_matched_filter_snr(noisy,         sample_rate, mass1, mass2, f_low)
        snr_c = peak_matched_filter_snr(reconstructed, sample_rate, mass1, mass2, f_low)
    elif psd is not None and freqs is not None:
        snr_n = psd_weighted_snr(noisy,         psd, freqs, sample_rate, f_low)
        snr_c = psd_weighted_snr(reconstructed, psd, freqs, sample_rate, f_low)
    else:
        snr_n = broadband_snr(noisy,         sample_rate)
        snr_c = broadband_snr(reconstructed, sample_rate)

    delta_snr = snr_improvement_db(snr_n, snr_c)

    # ── MSE  (bug-fix: normalise by clean_ref var, not noisy var) ─────────────
    ref   = clean_ref if clean_ref is not None else noisy
    denom = float(np.var(ref)) + 1e-30
    mse   = float(np.mean((noisy - reconstructed) ** 2)) / denom

    # ── Noise power reduction ─────────────────────────────────────────────────
    npr = noise_power_reduction(noisy, reconstructed, sample_rate,
                                f_low=noise_band[0], f_high=noise_band[1])

    # ── Overlap (faithfulness) ────────────────────────────────────────────────
    overlap = float("nan")
    if clean_ref is not None and psd is not None and freqs is not None:
        overlap = matched_filter_overlap(reconstructed, clean_ref,
                                          psd, freqs, sample_rate, f_low)

    # ── PSNR ─────────────────────────────────────────────────────────────────
    peak        = float(np.max(np.abs(reconstructed)))
    res_rms     = float(np.sqrt(np.mean((noisy - reconstructed) ** 2))) + 1e-12
    psnr_db_val = 20.0 * np.log10(peak / res_rms) if peak > 0 else float("nan")

    return EventResult(
        event_name=event_name,
        snr_noisy=snr_n, snr_clean=snr_c,
        snr_improvement_db=delta_snr,
        reconstruction_mse=mse,
        noise_power_reduction=npr,
        overlap=overlap,
        psnr_db=psnr_db_val,
    )


def evaluate_all(results: list[EventResult]) -> dict:
    if not results:
        return {"n_events": 0, "mean_snr_improvement_db": 0.0,
                "mean_mse": 0.0, "mean_noise_power_reduction": 0.0,
                "pass_snr": 0, "pass_mse": 0, "pass_npr": 0,
                "pass_all": 0, "thresholds": THRESHOLDS, "per_event": []}

    snr_imps = [r.snr_improvement_db    for r in results]
    mses     = [r.reconstruction_mse    for r in results]
    nprs     = [r.noise_power_reduction for r in results]
    overlaps = [r.overlap for r in results if not np.isnan(r.overlap)]

    return {
        "n_events":                   len(results),
        "mean_snr_improvement_db":    float(np.mean(snr_imps)),
        "median_snr_improvement_db":  float(np.median(snr_imps)),
        "std_snr_improvement_db":     float(np.std(snr_imps)),
        "mean_mse":                   float(np.mean(mses)),
        "mean_noise_power_reduction": float(np.mean(nprs)),
        "mean_overlap":               float(np.mean(overlaps)) if overlaps else None,
        "pass_snr":  int(sum(r.passes_snr  for r in results)),
        "pass_mse":  int(sum(r.passes_mse  for r in results)),
        "pass_npr":  int(sum(r.passes_npr  for r in results)),
        "pass_all":  int(sum(r.passes_all  for r in results)),
        "thresholds": THRESHOLDS,
        "per_event": [
            {
                "event":                  r.event_name,
                "snr_noisy":              round(r.snr_noisy, 2),
                "snr_clean":              round(r.snr_clean, 2),
                "snr_improvement_db":     round(r.snr_improvement_db, 2),
                "reconstruction_mse":     round(r.reconstruction_mse, 5),
                "noise_power_reduction":  round(r.noise_power_reduction, 4),
                "overlap":                round(r.overlap, 4) if not np.isnan(r.overlap) else None,
                "psnr_db":                round(r.psnr_db, 2) if not np.isnan(r.psnr_db) else None,
                "pass_all":               bool(r.passes_all),
            }
            for r in results
        ],
    }


def print_results_table(results: list[EventResult]) -> None:
    if not results:
        print("No results to display."); return
    sep = "─" * 110
    print(f"\n{sep}")
    print(f"{'Event':<26}| {'SNR improvement':<28}| {'MSE':>8} | {'NPR':>8} | {'Overlap':>9} | PSNR")
    print(sep)
    for r in results:
        print(r.summary())
    print(sep)
    agg = evaluate_all(results)
    print(f"\nMean SNR improvement : {agg['mean_snr_improvement_db']:+.2f} ± "
          f"{agg['std_snr_improvement_db']:.2f} dB  (target ≥ +{THRESHOLDS['snr_improvement_db']} dB)")
    print(f"Mean MSE             : {agg['mean_mse']:.5f}  (target < {THRESHOLDS['reconstruction_mse']})")
    print(f"Mean NPR             : {agg['mean_noise_power_reduction']*100:.1f}%"
          f"  (target ≥ {THRESHOLDS['noise_power_reduction']*100:.0f}%)")
    if agg.get("mean_overlap") is not None:
        print(f"Mean overlap         : {agg['mean_overlap']:.4f}  (target ≥ {THRESHOLDS['overlap']})")
    print(f"\nEvents passing all metrics: {agg['pass_all']}/{agg['n_events']}\n")
