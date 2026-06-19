"""
config.py — central hyperparameter registry.

Key upgrades vs v1
------------------
• Wider, deeper model: 7-stage encoder up to 512 channels
• WaveNet-style dilated residual blocks in bottleneck
• Multi-resolution STFT loss (3 FFT sizes) instead of single STFT
• Perceptual SNR loss term (matched-filter proxy, differentiable)
• OneCycleLR instead of plain cosine (faster convergence)
• Larger segments (8 s) for better low-frequency whitening
• Gradient accumulation support for effective large batches
• EMA model weights for stable evaluation
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    sample_rate: int = 4096
    segment_duration: float = 8.0        # ← 4 s → 8 s: captures full inspiral
    window_overlap: float = 0.75         # ← 0.5 → 0.75: 4× more training segments

    f_low: float = 20.0
    f_high: float = 2000.0
    bandpass_order: int = 8
    whiten_duration: float = 8.0         # match segment duration

    detectors: List[str] = field(default_factory=lambda: ["H1", "L1"])

    train_frac: float = 0.70
    val_frac:   float = 0.15
    test_frac:  float = 0.15

    held_out_events: List[str] = field(default_factory=lambda: [
        # 8 events selected from real GWTC CSV for diversity:
        # SNR range 18–79, all 5 catalogs, BBH/BNS, q=0.11–1.0
        "GW250114_082203",   # SNR=78.6, GWTC-5.0, equal-mass BBH
        "GW230814_230901",   # SNR=43.0, GWTC-4.1, equal-mass BBH
        "GW170817",          # SNR=33.0, GWTC-1,   BNS (unique morphology)
        "GW231226_101520",   # SNR=34.7, GWTC-4.1, equal-mass BBH
        "GW200129_065458",   # SNR=26.8, GWTC-3,   equal-mass BBH
        "GW190814",          # SNR=25.3, GWTC-2.1, asymmetric q=0.11
        "GW191216_213338",   # SNR=18.6, GWTC-3,   low-mass BBH
        "GW190521_074359",   # SNR=25.9, GWTC-2.1, heavy BBH
    ])
    data_dir: str = "data/strain"


@dataclass
class ModelConfig:
    # 7-stage encoder: more depth, wider bottleneck
    enc_channels: Tuple[int, ...] = (1, 32, 64, 128, 256, 512, 512, 512)
    kernel_sizes:  Tuple[int, ...] = (31, 15, 7, 7, 5, 3, 3)  # odd kernels → exact "same" padding
    stride: int = 2
    leaky_slope: float = 0.01           # smaller slope → closer to ReLU
    use_batchnorm: bool = False          # ← replaced by GroupNorm (better for variable-length)
    use_groupnorm: bool = True
    num_groups: int = 8
    dropout: float = 0.0                # ← dropout removed from encoder; use stochastic depth instead
    stochastic_depth_prob: float = 0.1  # NEW: per-stage drop probability

    # WaveNet-style dilated residual stack in bottleneck
    n_bottleneck_blocks: int = 6        # NEW
    bottleneck_dilation_base: int = 2   # dilations: 1,2,4,8,16,32
    bottleneck_kernel: int = 3

    # Attention gate in decoder skip connections
    use_attention_gates: bool = True    # NEW: suppress irrelevant skip features

    # Self-attention at bottleneck (lightweight)
    use_bottleneck_attention: bool = True  # NEW: global context for long-range dependencies


@dataclass
class LossConfig:
    mse_weight: float = 0.5             # ← down from 1.0; multi-STFT now dominant
    spectral_weight: float = 1.0        # ← up from 0.3
    snr_proxy_weight: float = 0.3       # NEW: differentiable SNR proxy loss
    perceptual_weight: float = 0.2      # NEW: feature-space MSE at encoder layers

    # Multi-resolution STFT: three scales
    stft_fft_sizes:  Tuple[int, ...] = (2048, 512, 128)
    stft_hop_ratios: Tuple[float, ...] = (0.25, 0.25, 0.25)   # hop = fft * ratio


@dataclass
class TrainConfig:
    epochs: int = 200                   # more budget; OneCycleLR uses it all
    batch_size: int = 16                # smaller batches → GroupNorm friendly
    accumulate_grad_batches: int = 4    # effective batch = 64
    learning_rate: float = 5e-4        # OneCycleLR peak LR
    weight_decay: float = 1e-4
    lr_scheduler: str = "onecycle"      # ← cosine → OneCycleLR
    pct_start: float = 0.1             # 10% warmup
    grad_clip: float = 0.5             # ← 1.0 → 0.5 tighter

    # EMA
    use_ema: bool = True                # NEW
    ema_decay: float = 0.999

    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 30   # more patience with OneCycle

    seed: int = 42
    device: str = "auto"

    # torch.compile (PyTorch 2.0+)
    compile_model: bool = False         # set True on A100/H100 for ~30% speedup


@dataclass
class EvalConfig:
    snr_improvement_threshold: float = 3.0
    mse_threshold: float = 0.05
    noise_power_reduction_threshold: float = 0.40
    noise_band: Tuple[float, float] = (30.0, 300.0)


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss:  LossConfig  = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)


def get_config() -> Config:
    return Config()
