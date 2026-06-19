# Gravitational Wave Noise Filter
### 1-D Convolutional Autoencoder for LIGO Strain Denoising — v5

---

## Dataset Summary (events.csv)

| Metric | Value |
|--------|-------|
| Total events | 391 |
| Catalogues | GWTC-1, GWTC-2.1, GWTC-3, GWTC-4.1, GWTC-5.0 |
| GPS range | O1 (2015) → O5 (2025) |
| SNR range | 4.5 – 78.6 (mean 11.8) |
| With source masses | 282/391 |
| p_astro ≥ 0.99 | 214/391 |

### Catalogue breakdown

| Catalogue | Events |
|-----------|--------|
| GWTC-1-confident | 1 |
| GWTC-2.1-confident | 54 |
| GWTC-3-confident | 35 |
| GWTC-4.1 | 140 |
| GWTC-5.0 | 161 |


### Held-out test events (reserved — not used for training)

| Event | SNR | m₁ (M☉) | m₂ (M☉) | q | Catalogue | Type |
|-------|-----|---------|---------|---|-----------|------|
| GW250114_082203 | 78.6 | 33.8 | 32.3 | 0.96 | GWTC-5.0 | BBH |
| GW230814_230901 | 43.0 | 33.7 | 28.2 | 0.84 | GWTC-4.1 | BBH |
| GW170817        | 33.0 |  1.5 |  1.3 | 0.87 | GWTC-1   | BNS |
| GW231226_101520 | 34.7 | 40.2 | 35.1 | 0.87 | GWTC-4.1 | BBH |
| GW200129_065458 | 26.8 | 34.5 | 29.0 | 0.84 | GWTC-3   | BBH |
| GW190814        | 25.3 | 23.3 |  2.6 | 0.11 | GWTC-2.1 | NSBH |
| GW191216_213338 | 18.6 | 12.1 |  7.7 | 0.64 | GWTC-3   | BBH |
| GW190521_074359 | 25.9 | 43.4 | 33.4 | 0.77 | GWTC-2.1 | BBH |

---

## Architecture (v5)

```
Input strain (B, 1, 32768)   ← 8 s @ 4096 Hz
        │
   ┌────▼────────────────────────────────────────────────┐
   │  ENCODER  (7 stages, stride=2 each)                 │
   │  MultiScaleConvNeXt block (dual DW branch)          │
   │  + Strided GroupNorm projection                     │
   │  Channels: 1→32→64→128→256→512→512→512              │
   │  DropPath stochastic depth (p increases with depth) │
   └─────────────────────┬───────────────────────────────┘
                         │  bottleneck (B, 512, 256)
   ┌─────────────────────▼───────────────────────────────┐
   │  BOTTLENECK                                         │
   │  8× WaveNet DilatedResBlock (d=1,2,4,8,1,2,4,8)    │
   │    Pre-activation: norm→SiLU→gated conv→LayerScale  │
   │    RF ≈ 2 s at 4096 Hz (captures full inspiral)     │
   │  BottleneckAttention (Flash-/Linear-attention)      │
   │  FiLM conditioning (log-RMS + log-peak → γ, β)      │
   └─────────────────────┬───────────────────────────────┘
                         │
   ┌─────────────────────▼───────────────────────────────┐
   │  DECODER  (7 stages, mirrors encoder)               │
   │  DepthwiseSep ConvTranspose (less checkerboard)     │
   │  AttentionGate on each skip connection              │
   │  GLU fusion (value × sigmoid(gate))                 │
   │  SEBlock1d channel recalibration                    │
   └─────────────────────┬───────────────────────────────┘
                         │
         Reconstructed clean strain (B, 1, 32768)
```

**Combined loss:**
- `FreqWeightedMSE` — chirp-band emphasis, learnable per-band weights
- `MultiResSTFTLoss` — spectral convergence + log-magnitude L1 at 3 scales
- `NegativeSNRLoss` — directly maximise PSD-weighted SNR in dB
- `EnvelopeLoss` — Hilbert analytic envelope L1
- `PhaseAwareLoss` — complex STFT phase fidelity
- `SpectralWassersteinLoss` — OT distance on spectral amplitude distribution

**Training:**
- AdamW + Lookahead(k=5, α=0.5)
- OneCycleLR with per-group peak LRs
- EMA weights (warm-start at epoch 5)
- Curriculum SNR annealing (25→5 over 50 epochs)
- Gradient accumulation (eff. batch=64)

---

## Verification Metrics

| Metric | Target |
|--------|--------|
| SNR improvement | ≥ 3 dB on all 8 held-out events |
| Reconstruction MSE | < 0.05 (normalised by signal variance) |
| Noise power reduction | ≥ 40% in 30–300 Hz band |
| Matched-filter overlap | ≥ 0.95 (for injected events with clean ref) |

---

## Quickstart

```bash
pip install -r requirements.txt

# Download held-out test events (do this FIRST, keep separate)
python data/downloader.py --events held_out --detector L1

# Download training data: off-source noise + signal-bearing segments
python data/downloader.py --noise-segments 200 --observing-run O4 --detector L1
python data/downloader.py --events training --min-snr 8 --min-pastro 0.99 --detector L1

# Train
python train.py --epochs 200 --device auto

# Evaluate on held-out events
python evaluate.py --checkpoint checkpoints/best.pt
```

## Project Structure

```
gw_denoiser/
├── data/
│   ├── downloader.py     # GWTC GPS table (391 events), GWOSC fetch
│   └── dataset.py        # PyTorch Dataset, mixed injection strategy
├── models/
│   ├── autoencoder.py    # U-Net v5: ConvNeXt + WaveNet + Flash-attn
│   └── losses.py         # 6-component combined loss + SharedSTFT cache
├── utils/
│   ├── preprocessing.py  # Median-Welch whitening, MAD normalisation
│   └── snr.py            # PyCBC MF-SNR, psd_weighted_snr, overlap
├── evaluation/
│   ├── metrics.py        # All 4 verification metrics + overlap + PSNR
│   └── plots.py          # Loss curves, SNR bars, spectrograms, waterfall
├── train.py              # AdamW+Lookahead, OneCycleLR, EMA, curriculum
├── evaluate.py           # TTA denoising, Q-transform, per-event subdirs
├── config.py             # All hyperparameters centralised
└── requirements.txt
```
