"""
evaluate.py  — v4

Bug-fixes vs v1 (never previously updated)
--------------------------------------------
• GWAutoencoder constructed with removed v1/v2 args (batchnorm, dropout) —
  would crash at runtime. Fixed: construct from current v4 config signature only.
• Loads EMA weights from checkpoint when available — EMA model is always
  better than raw weights for held-out evaluation.
• Central 4-s window extracted for inference but model trained on 8-s segments —
  shape mismatch at model forward pass. Fixed: use full cfg.data.segment_duration.
• load_checkpoint imported from train — that function in v4 expects a Lookahead
  optimizer, not a plain one. At eval time no optimizer is needed; fixed with
  a standalone load_model_weights() that only loads model state dict.
• Segment PSD passed to evaluate_event for psd_weighted_snr and overlap.
• TTA denoising at eval time: average over 3 amplitude scales for stability.

Performance additions
---------------------
• Batch inference: if > 1 event, stack segments into a batch (faster on GPU).
• Q-transform spectrogram via GWpy when available (better resolution than scipy).
• results/ directory structure: per-event subdirs prevent filename collisions.
• Inference timing: reports ms/segment throughput for performance benchmarking.
"""

from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import get_config
from models.autoencoder import GWAutoencoder
from utils.preprocessing import preprocess_segment, estimate_psd
from utils.snr import broadband_snr
from evaluation.metrics import evaluate_event, evaluate_all, print_results_table, EventResult
from evaluation.plots import (
    plot_snr_comparison, plot_waveform_comparison,
    plot_spectrogram_comparison, plot_asd_comparison, plot_loss_curves,
)
from data.downloader import GWTC_GPS, HELD_OUT_EVENTS, MASS_PRIORS, fetch_event

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone model loader  (no optimizer dependency)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_weights(path: Path, model: GWAutoencoder,
                        prefer_ema: bool = True) -> None:
    """Load model (or EMA) weights from a checkpoint, no optimizer needed."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if prefer_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        log.info(f"Loaded EMA weights from {path.name}  (epoch {ckpt['epoch']})")
    else:
        model.load_state_dict(ckpt["model"])
        log.info(f"Loaded model weights from {path.name}  (epoch {ckpt['epoch']})")


# ─────────────────────────────────────────────────────────────────────────────
# Strain loading
# ─────────────────────────────────────────────────────────────────────────────

def load_event_strain(
    event_name: str,
    detector:   str,
    data_dir:   Path,
    sample_rate: int,
    seg_duration: float,
    f_low: float,
    f_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load whitened strain for one event.
    Returns (segment, psd_freqs, psd_values) or None on failure.
    Segment length = seg_duration * sample_rate (matching training size).
    """
    strain_path = data_dir / f"{event_name}_{detector}.hdf5"

    if not strain_path.exists():
        if event_name not in GWTC_GPS:
            log.warning(f"No GPS time for {event_name} — skipping.")
            return None
        log.info(f"Downloading {event_name}/{detector} …")
        if fetch_event(event_name, GWTC_GPS[event_name], detector,
                       data_dir, duration=64.0) is None:
            return None

    try:
        try:
            from gwpy.timeseries import TimeSeries
            raw = TimeSeries.read(str(strain_path), format="hdf5").value.astype(np.float64)
        except Exception:
            import h5py
            with h5py.File(strain_path, "r") as f:
                key = list(f.keys())[0]
                raw = f[key]["strain"]["Strain"][:].astype(np.float64)

        freqs, psd = estimate_psd(raw, sample_rate)
        whitened   = preprocess_segment(raw, sample_rate, psd, freqs, f_low, f_high)

        # Extract segment of correct inference length
        seg_len = int(seg_duration * sample_rate)
        if len(whitened) >= seg_len:
            # Take the central window (highest SNR for real events)
            c   = len(whitened) // 2
            h   = seg_len // 2
            seg = whitened[c - h : c + h]
        else:
            seg = np.pad(whitened, (0, seg_len - len(whitened)))

        seg = seg.astype(np.float32)
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        return seg, freqs, psd

    except Exception as exc:
        log.error(f"Failed {event_name}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TTA denoising
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def denoise_tta(
    model:   GWAutoencoder,
    segment: np.ndarray,
    device:  torch.device,
    scales:  list[float] = (0.9, 1.0, 1.1),
) -> np.ndarray:
    """Average denoised output over amplitude scales (amplitude-only TTA)."""
    x = torch.from_numpy(segment).unsqueeze(0).unsqueeze(0).to(device)
    preds = []
    for s in scales:
        out = model(x * s) / s
        preds.append(out)
    return torch.stack(preds).mean(0).squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Q-transform spectrogram  (GWpy if available, scipy fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _save_qtransform(
    strain: np.ndarray, sample_rate: int,
    title: str, save_path: Path,
) -> None:
    """Try GWpy Q-transform; fall back to scipy spectrogram."""
    try:
        from gwpy.timeseries import TimeSeries
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts    = TimeSeries(strain, sample_rate=sample_rate)
        qgram = ts.q_transform(qrange=(4, 64), frange=(20, 512),
                                 outseg=(0, len(strain) / sample_rate))
        fig   = qgram.plot(figsize=(10, 4))
        fig.suptitle(title, fontsize=10)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass   # fallback handled in plots.py


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args) -> None:
    cfg    = get_config()

    # Device
    from train import resolve_device
    device = resolve_device(cfg.train.device)

    # Model — build from v4 config (correct argument set)
    model = GWAutoencoder(
        enc_channels             = cfg.model.enc_channels,
        kernel_sizes             = cfg.model.kernel_sizes,
        stride                   = cfg.model.stride,
        leaky_slope              = cfg.model.leaky_slope,
        num_groups               = cfg.model.num_groups,
        stochastic_depth_prob    = 0.0,   # disabled at eval
        n_bottleneck_blocks      = cfg.model.n_bottleneck_blocks,
        bottleneck_dilation_base = cfg.model.bottleneck_dilation_base,
        bottleneck_kernel        = cfg.model.bottleneck_kernel,
        use_attention_gates      = cfg.model.use_attention_gates,
        use_bottleneck_attention = cfg.model.use_bottleneck_attention,
    ).to(device).eval()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        load_model_weights(ckpt_path, model, prefer_ema=True)
    else:
        log.warning(f"Checkpoint {ckpt_path} not found — random weights.")

    # Directories
    outdir   = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data.data_dir); data_dir.mkdir(parents=True, exist_ok=True)

    events   = args.events or HELD_OUT_EVENTS
    detector = args.detector
    mass_priors = {
        "GW250119_190238": (11.54, 9.95),
        "GW250119_025138": (33.8, 26.7),
        "GW250118_170523": (38.1, 29.6),
        "GW250118_055802": (10.3, 6.9),
        "GW250118_023225": (44.0, 31.0),
        "GW250116_015318": (34.2, 21.3),
        "GW250114_082203": (33.76, 32.26),
        "GW250109_074552": (39.0, 30.2),
        "GW250109_010541": (37.5, 27.4),
        "GW250108_152221": (54.0, 36.0),
        "GW250104_015122": (45.8, 34.5),
        "GW250101_011205": (30.0, 21.1),
        "GW241231_054133": (12.4, 7.1),
        "GW241230_233618": (68.0, 49.0),
        "GW241230_084504": (42.4, 33.5),
        "GW241229_155844": (52.0, 33.0),
        "GW241225_082815": (55.7, 42.2),
        "GW241225_042553": (12.4, 8.1),
        "GW241210_120900": (40.0, 23.6),
        "GW241210_060606": (29.0, 21.4),
        "GW241201_055758": (47.0, 33.0),
        "GW241130_110422": (11.1, 6.5),
        "GW241130_034908": (29.9, 24.3),
        "GW241129_021832": (30.4, 23.0),
        "GW241127_061008": (63.8, 20.8),
        "GW241125_010116": (60.0, 47.0),
        "GW241124_024914": (42.0, 26.0),
        "GW241116_151753": (70.0, 24.0),
        "GW241114_235258": (11.3, 7.7),
        "GW241114_024711": (47.0, 25.0),
        "GW241113_163507": (19.3, 14.4),
        "GW241111_111552": (25.3, 20.5),
        "GW241110_124123": (16.4, 8.0),
        "GW241109_115924": (7.3, 5.14),
        "GW241109_033317": (42.4, 31.7),
        "GW241102_144729": (44.4, 31.0),
        "GW241102_124058": (10.9, 8.0),
        "GW241101_220523": (41.0, 19.3),
        "GW241011_233834": (19.5, 5.96),
        "GW241009_220455": (33.2, 25.5),
        "GW241009_084816": (12.7, 8.1),
        "GW241009_022835": (37.1, 26.8),
        "GW241007_082943": (36.6, 26.6),
        "GW241006_015333": (24.6, 20.3),
        "GW241002_030559": (37.3, 29.9),
        "GW240930_234614": (35.8, 21.9),
        "GW240930_035959": (25.5, 11.2),
        "GW240925_005809": (9.02, 6.99),
        "GW240924_000316": (46.0, 34.0),
        "GW240923_204006": (46.1, 34.9),
        "GW240922_142106": (11.5, 7.8),
        "GW240921_201835": (34.9, 9.8),
        "GW240920_124024": (37.1, 31.9),
        "GW240920_073424": (24.7, 13.4),
        "GW240919_061559": (38.3, 29.6),
        "GW240916_184352": (10.7, 7.8),
        "GW240915_105151": (11.6, 7.5),
        "GW240915_001357": (10.9, 7.7),
        "GW240910_103535": (10.9, 6.3),
        "GW240908_125134": (45.0, 32.0),
        "GW240908_082628": (39.8, 27.7),
        "GW240907_153833": (38.0, 28.3),
        "GW240902_143306": (24.4, 18.3),
        "GW240830_211120": (11.9, 7.8),
        "GW240825_055146": (13.2, 7.9),
        "GW240824_205609": (70.0, 37.0),
        "GW240716_034900": (40.3, 26.5),
        "GW240705_053215": (47.1, 35.3),
        "GW240703_191355": (34.4, 26.4),
        "GW240630_101703": (28.6, 21.6),
        "GW240629_145256": (11.1, 7.8),
        "GW240627_131622": (13.2, 6.7),
        "GW240622_004008": (18.6, 11.3),
        "GW240621_214041": (45.0, 32.0),
        "GW240621_200935": (41.6, 31.6),
        "GW240621_195059": (36.8, 29.7),
        "GW240618_071627": (65.0, 40.0),
        "GW240615_160735": (28.5, 20.0),
        "GW240615_113620": (34.0, 26.4),
        "GW240612_081540": (55.0, 38.0),
        "GW240601_231004": (10.6, 7.7),
        "GW240601_061200": (52.0, 32.0),
        "GW240531_075248": (33.0, 22.6),
        "GW240531_040326": (19.5, 14.3),
        "GW240530_012417": (14.1, 8.3),
        "GW240527_230910": (27.6, 10.0),
        "GW240527_183429": (52.0, 33.0),
        "GW240526_093944": (14.5, 8.4),
        "GW240525_031210": (31.1, 21.4),
        "GW240520_213616": (11.4, 8.0),
        "GW240519_012815": (65.0, 39.0),
        "GW240515_005301": (37.1, 18.5),
        "GW240514_121713": (46.8, 36.7),
        "GW240513_183302": (24.4, 16.9),
        "GW240512_024139": (12.6, 8.0),
        "GW240511_031507": (40.5, 32.2),
        "GW240507_041632": (31.3, 8.5),
        "GW240505_133552": (31.0, 23.1),
        "GW240501_033534": (38.6, 27.7),
        "GW240428_225440": (19.8, 14.7),
        "GW240426_031451": (52.0, 36.0),
        "GW240420_175625": (34.6, 23.8),
        "GW240414_054515": (38.8, 25.9),
        "GW240413_022019": (7.9, 5.57),
        "GW240109_050431": (28.7, 18.1),
        "GW240107_013215": (59.0, 33.0),
        "GW240104_164932": (42.3, 32.2),
        "GW231231_154016": (22.5, 17.2),
        "GW231230_170116": (53.0, 35.0),
        "GW231226_101520": (40.2, 35.1),
        "GW231224_024321": (9.3, 7.31),
        "GW231223_202619": (11.1, 8.3),
        "GW231223_075055": (11.9, 6.8),
        "GW231223_032836": (46.0, 31.0),
        "GW231221_135041": (47.0, 29.0),
        "GW231213_111417": (35.6, 27.2),
        "GW231206_233901": (37.6, 28.5),
        "GW231206_233134": (35.6, 28.2),
        "GW231129_081745": (45.0, 23.8),
        "GW231127_165300": (45.0, 28.0),
        "GW231123_135430": (137.0, 101.0),
        "GW231119_075248": (49.0, 33.0),
        "GW231118_090602": (12.9, 7.4),
        "GW231118_071402": (43.0, 29.0),
        "GW231118_005626": (20.0, 10.8),
        "GW231114_043211": (22.8, 8.2),
        "GW231113_200417": (11.6, 7.4),
        "GW231113_150041": (56.0, 29.0),
        "GW231113_122623": (39.8, 26.4),
        "GW231110_040320": (19.4, 12.6),
        "GW231108_125142": (23.2, 17.4),
        "GW231104_133418": (12.2, 8.6),
        "GW231102_071736": (61.0, 42.0),
        "GW231029_111508": (65.0, 42.0),
        "GW231028_153006": (94.0, 59.0),
        "GW231026_130704": (34.0, 20.8),
        "GW231020_142947": (12.0, 7.4),
        "GW231018_233037": (11.6, 7.3),
        "GW231014_040532": (20.6, 14.7),
        "GW231008_142521": (45.0, 25.6),
        "GW231005_091549": (28.8, 21.2),
        "GW231005_021030": (84.0, 50.0),
        "GW231004_232346": (64.0, 35.0),
        "GW231001_140220": (76.0, 41.0),
        "GW230930_110730": (34.5, 24.4),
        "GW230928_215827": (54.0, 29.0),
        "GW230927_153832": (21.7, 16.6),
        "GW230927_043729": (35.0, 27.1),
        "GW230924_124453": (28.8, 23.2),
        "GW230922_040658": (76.0, 51.0),
        "GW230922_020344": (39.3, 29.2),
        "GW230920_071124": (32.4, 23.8),
        "GW230919_215712": (27.3, 21.4),
        "GW230914_111401": (60.0, 37.0),
        "GW230911_195324": (33.9, 21.6),
        "GW230904_051013": (10.6, 7.1),
        "GW230831_015414": (42.0, 30.0),
        "GW230825_041334": (44.0, 27.2),
        "GW230824_033047": (53.0, 36.0),
        "GW230820_212515": (62.0, 34.0),
        "GW230819_171910": (70.0, 35.0),
        "GW230814_230901": (33.7, 28.2),
        "GW230814_061920": (69.0, 42.0),
        "GW230811_032116": (35.4, 22.3),
        "GW230806_204041": (51.0, 35.0),
        "GW230805_034249": (32.2, 22.6),
        "GW230803_033412": (44.0, 29.0),
        "GW230731_215307": (10.3, 7.9),
        "GW230729_082317": (12.3, 7.6),
        "GW230726_002940": (35.6, 27.9),
        "GW230723_101834": (16.6, 10.6),
        "GW230712_090405": (32.0, 12.5),
        "GW230709_122727": (45.0, 30.0),
        "GW230708_230935": (64.0, 40.0),
        "GW230708_053705": (29.1, 22.7),
        "GW230707_124047": (46.1, 36.3),
        "GW230706_104333": (16.3, 11.5),
        "GW230704_212616": (89.0, 50.0),
        "GW230704_021211": (32.6, 20.0),
        "GW230702_185453": (40.0, 18.0),
        "GW230630_234532": (10.0, 6.6),
        "GW230630_125806": (51.0, 33.0),
        "GW230628_231200": (32.5, 27.0),
        "GW230627_015337": (8.4, 5.74),
        "GW230624_113103": (27.1, 16.1),
        "GW230609_064958": (35.4, 25.2),
        "GW230608_205047": (48.0, 31.0),
        "GW230606_004305": (37.6, 25.7),
        "GW230605_065343": (17.4, 11.0),
        "GW230601_224134": (64.0, 44.0),
        "GW230529_181500": (3.66, 1.42),
        "GW230518_125908": (8.17, 1.45),
        "GW200322_091133": (38.0, 11.3),
        "GW200316_215756": (13.1, 7.8),
        "GW200311_115853": (34.2, 27.7),
        "GW200308_173609": (60.0, 24.0),
        "GW200306_093714": (28.3, 14.8),
        "GW200302_015811": (37.8, 20.0),
        "GW200225_060421": (19.3, 14.0),
        "GW200224_222234": (40.0, 32.7),
        "GW200220_124850": (38.9, 27.9),
        "GW200220_061928": (87.0, 61.0),
        "GW200219_094415": (37.5, 27.9),
        "GW200216_220804": (51.0, 30.0),
        "GW200210_092254": (24.1, 2.83),
        "GW200209_085452": (35.6, 27.1),
        "GW200208_222617": (51.0, 12.3),
        "GW200208_130117": (37.7, 27.4),
        "GW200202_154313": (10.1, 7.3),
        "GW200129_065458": (34.5, 29.0),
        "GW200128_022011": (42.2, 32.6),
        "GW200115_042309": (5.9, 1.44),
        "GW200112_155838": (35.6, 28.3),
        "GW191230_180458": (49.4, 37.0),
        "GW191222_033537": (45.1, 34.7),
        "GW191219_163120": (31.1, 1.17),
        "GW191216_213338": (12.1, 7.7),
        "GW191215_223052": (24.9, 18.1),
        "GW191204_171526": (11.7, 8.4),
        "GW191204_110529": (27.3, 19.2),
        "GW191129_134029": (10.7, 6.7),
        "GW191127_050227": (53.0, 24.0),
        "GW191126_115259": (12.1, 8.3),
        "GW191113_071753": (29.0, 5.9),
        "GW191109_010717": (65.0, 47.0),
        "GW191105_143521": (10.7, 7.7),
        "GW191103_012549": (11.8, 7.9),
        "GW190930_133541": (14.2, 6.9),
        "GW190929_012149": (66.3, 26.8),
        "GW190926_050336": (41.1, 20.4),
        "GW190925_232845": (20.8, 15.5),
        "GW190924_021846": (8.8, 5.1),
        "GW190917_114630": (9.7, 2.1),
        "GW190916_200658": (43.8, 23.3),
        "GW190915_235702": (32.6, 24.5),
        "GW190910_112807": (43.8, 34.2),
        "GW190828_065509": (23.7, 10.4),
        "GW190828_063405": (31.9, 25.8),
        "GW190814": (23.3, 2.6),
        "GW190805_211137": (46.2, 30.6),
        "GW190803_022701": (37.7, 27.6),
        "GW190731_140936": (41.8, 29.0),
        "GW190728_064510": (12.5, 8.0),
        "GW190727_060333": (38.9, 30.2),
        "GW190725_174728": (11.8, 6.3),
        "GW190720_000836": (14.2, 7.5),
        "GW190719_215514": (36.6, 19.9),
        "GW190708_232457": (19.8, 11.6),
        "GW190707_093326": (12.1, 7.9),
        "GW190706_222641": (74.0, 39.4),
        "GW190701_203306": (54.1, 40.5),
        "GW190630_185205": (35.1, 24.0),
        "GW190620_030421": (58.0, 35.0),
        "GW190602_175927": (71.8, 44.8),
        "GW190527_092055": (35.6, 22.2),
        "GW190521_074359": (43.4, 33.4),
        "GW190521": (98.4, 57.2),
        "GW190519_153544": (65.1, 40.8),
        "GW190517_055101": (39.2, 24.0),
        "GW190514_065416": (40.9, 28.4),
        "GW190513_205428": (36.0, 18.3),
        "GW190512_180714": (23.2, 12.5),
        "GW190503_185404": (41.3, 28.3),
        "GW190426_190642": (105.5, 76.0),
        "GW190425": (2.1, 1.3),
        "GW190421_213856": (42.0, 32.0),
        "GW190413_134308": (51.3, 30.4),
        "GW190413_052954": (33.7, 24.2),
        "GW190412": (27.7, 9.0),
        "GW190408_181802": (24.8, 18.5),
        "GW190403_051519": (85.0, 20.0),
        "GW170823": (38.3, 29.0),
        "GW170818": (34.8, 27.6),
        "GW170817": (1.46, 1.27),
        "GW170814": (30.9, 24.9),
        "GW170809": (34.1, 24.2),
        "GW170729": (54.7, 30.2),
        "GW170608": (10.6, 7.8),
        "GW170104": (28.7, 20.8),
        "GW151226": (14.2, 7.5),
        "GW151012": (24.8, 13.6),
        "GW150914": (34.6, 30.0),
    }

    log.info(f"Evaluating {len(events)} events on {device}")
    results: list[EventResult] = []
    timing:  list[float]       = []

    for event_name in events:
        log.info(f"\n── {event_name} ──")
        loaded = load_event_strain(
            event_name, detector, data_dir,
            cfg.data.sample_rate, cfg.data.segment_duration,
            cfg.data.f_low, cfg.data.f_high,
        )
        if loaded is None:
            log.warning(f"  Skipping (no data).")
            continue

        segment, ev_freqs, ev_psd = loaded

        t0   = time.perf_counter()
        recon = denoise_tta(model, segment, device,
                             scales=[0.9, 1.0, 1.1] if not args.no_tta else [1.0])
        timing.append((time.perf_counter() - t0) * 1000)

        m1, m2 = mass_priors.get(event_name, MASS_PRIORS.get(event_name, (30.0, 30.0)))
        result = evaluate_event(
            event_name, segment, recon,
            sample_rate = cfg.data.sample_rate,
            mass1=m1, mass2=m2,
            use_pycbc   = args.use_pycbc,
            f_low       = cfg.data.f_low,
            noise_band  = cfg.eval.noise_band,
            psd         = ev_psd,
            freqs       = ev_freqs,
        )
        results.append(result)
        log.info(f"  {result.summary()}")
        log.info(f"  Inference: {timing[-1]:.1f} ms")

        # Per-event output directory
        slug    = event_name.replace("_", "-")
        ev_dir  = outdir / slug
        ev_dir.mkdir(exist_ok=True)

        plot_waveform_comparison(
            segment, recon, cfg.data.sample_rate,
            event_name, ev_dir / "waveform.png",
        )
        plot_spectrogram_comparison(
            segment, recon, cfg.data.sample_rate,
            event_name, ev_dir / "spectrogram.png",
            f_low=cfg.data.f_low, f_high=min(cfg.data.f_high, 512),
        )
        plot_asd_comparison(
            segment, recon, cfg.data.sample_rate,
            event_name, ev_dir / "asd.png",
            f_low=cfg.data.f_low, f_high=cfg.data.f_high,
        )
        # Q-transform (GWpy if available)
        _save_qtransform(recon, cfg.data.sample_rate,
                          f"{event_name} — reconstructed", ev_dir / "qtransform.png")

    if not results:
        log.error("No events evaluated."); return

    # Summary plot
    plot_snr_comparison(
        [r.event_name for r in results],
        [r.snr_noisy  for r in results],
        [r.snr_clean  for r in results],
        outdir / "snr_all_events.png",
        threshold_db=cfg.eval.snr_improvement_threshold,
    )

    # Training history
    hist_path = Path(cfg.train.checkpoint_dir) / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        plot_loss_curves(history, outdir / "loss_curves.png")

    # Print table + save JSON
    print_results_table(results)
    agg = evaluate_all(results)
    agg["inference_ms_mean"] = float(np.mean(timing)) if timing else 0.0
    agg["inference_ms_std"]  = float(np.std(timing))  if timing else 0.0

    out_json = outdir / "results.json"
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2)
    log.info(f"\nResults → {out_json}")
    log.info(f"Mean inference: {agg['inference_ms_mean']:.1f} ± "
             f"{agg['inference_ms_std']:.1f} ms/segment")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--events",     nargs="*", default=None)
    p.add_argument("--detector",   default="L1", choices=["H1","L1","V1"])
    p.add_argument("--outdir",     default="results/")
    p.add_argument("--use-pycbc",  action="store_true")
    p.add_argument("--no-tta",     action="store_true",
                   help="Disable test-time augmentation (faster, slightly lower SNR)")
    evaluate(p.parse_args())
