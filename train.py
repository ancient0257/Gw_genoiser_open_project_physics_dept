"""
train.py  — v5

Bug-fixes vs v4
---------------
• Lookahead.step() called base_opt.step() THEN on every k-th step copied
  slow→fast weights. But OneCycleLR.step() was called on base_opt AFTER
  the Lookahead step — scheduler saw the already-overwritten (slow) LR.
  Fixed: scheduler steps on base_opt before Lookahead's slow-weight sync,
  so the LR schedule is applied to the fast weights only, which is correct.

• make_param_groups accessed model.film.parameters() without checking
  model.film is not None — AttributeError when use_film=False.
  Fixed: guard with `if model.film is not None`.

• validate_with_tta accumulated loss dict values including scheduler weight
  scalars (w_snr, w_env, w_phase) and then divided by n_batches — averaging
  loss weights makes no sense and inflates logged weight values.
  Fixed: filter out non-loss keys (those starting with "w_") before logging.

• save_checkpoint serialised Lookahead via optimizer.state_dict() which
  calls self.optimizer.state_dict() — but slow_state contains tensors that
  are not serialised as proper state (they're plain Python list of dicts of
  tensors). On reload, torch.load returns them as CPU tensors regardless of
  device, and load_state_dict tries to assign them without .to(device).
  Fixed: move slow_state tensors to the correct device in load_state_dict.

Performance additions vs v4
----------------------------
• Gradient norm EMA: instead of logging raw per-epoch mean/std, maintain
  an EMA of grad norm (β=0.98) for a smoother signal that's less noisy
  on small datasets.
• Mixed-precision loss scaling: use dynamic scaler with growth_interval=100
  (default 2000 is too slow for ~200-step epochs; frequent overflow events
  waste 50+ steps before scale recovers).
• DataLoader pin_memory_device: on CUDA, explicitly set pin_memory_device="cuda"
  for correct behaviour on multi-GPU machines.
• Warm-start EMA: instead of initialising EMA from the random initial model,
  wait until epoch > warmup_epochs to start EMA (prevents random-weight
  average polluting the shadow model during unstable early training).
"""

from __future__ import annotations
import argparse
import copy
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from config import get_config
from models.autoencoder import GWAutoencoder
from models.losses import GWDenoisingLoss

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead  (bug-fix: device-safe slow_state reload)
# ─────────────────────────────────────────────────────────────────────────────

class Lookahead:
    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k         = k
        self.alpha     = alpha
        self._step     = 0
        self.slow_state = [
            {"slow_params": [p.data.clone() for p in g["params"]]}
            for g in optimizer.param_groups
        ]

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step % self.k == 0:
            for group, slow in zip(self.optimizer.param_groups, self.slow_state):
                for p, sp in zip(group["params"], slow["slow_params"]):
                    sp.add_(p.data - sp, alpha=self.alpha)
                    p.data.copy_(sp)
        return loss

    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(**kwargs)

    def state_dict(self) -> dict:
        return {
            "optimizer":   self.optimizer.state_dict(),
            "slow_params": [[sp.cpu() for sp in s["slow_params"]]
                            for s in self.slow_state],
            "step":        self._step,
        }

    def load_state_dict(self, sd: dict, device: torch.device | None = None) -> None:
        self.optimizer.load_state_dict(sd["optimizer"])
        self._step = sd.get("step", 0)
        if "slow_params" in sd:
            for s, sps in zip(self.slow_state, sd["slow_params"]):
                for sp_dst, sp_src in zip(s["slow_params"], sps):
                    # BUG-FIX: move to correct device
                    sp_dst.copy_(sp_src.to(sp_dst.device))

    @property
    def param_groups(self):
        return self.optimizer.param_groups


# ─────────────────────────────────────────────────────────────────────────────
# EMA  (warm-start aware)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999,
                 warmup_epochs: int = 5):
        self.decay         = decay
        self.warmup_epochs = warmup_epochs
        self.shadow        = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._initialised  = False

    @torch.no_grad()
    def update(self, model: nn.Module, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            # Warm-start: copy exact weights until EMA warmup is done
            for ep, mp in zip(self.shadow.parameters(), model.parameters()):
                ep.copy_(mp.data)
            return
        if not self._initialised:
            # First real EMA update: sync shadow to current model
            for ep, mp in zip(self.shadow.parameters(), model.parameters()):
                ep.copy_(mp.data)
            self._initialised = True
            return
        for ep, mp in zip(self.shadow.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, epoch, val_loss, cfg, ema=None):
    torch.save({
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict() if hasattr(optimizer, "state_dict") else {},
        "config":    dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else str(cfg),
        **({"ema": ema.state_dict()} if ema else {}),
    }, path)
    log.info(f"  ✓ {Path(path).name}  (val={val_loss:.5f})")


def load_checkpoint(path, model, optimizer=None, device=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        if hasattr(optimizer, "load_state_dict") and isinstance(optimizer, Lookahead):
            optimizer.load_state_dict(ckpt["optimizer"], device=device)
        elif hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer"])
    log.info(f"Loaded {path}  (epoch {ckpt['epoch']}, val {ckpt['val_loss']:.6f})")
    return ckpt["epoch"], ckpt["val_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# TTA validation  (bug-fix: filter scheduler weight keys from logging)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_with_tta(model, loader, loss_fn, device, n_tta: int = 3):
    model.eval()
    totals: dict[str, float] = {}
    n_batches  = 0
    amp_on     = device.type in ("cuda", "mps")
    amp_ctx    = torch.amp.autocast(device_type=device.type, enabled=amp_on)
    scales     = [0.9, 1.0, 1.1][:n_tta]

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = []
        for s in scales:
            with amp_ctx:
                preds.append(model(x * s) / s)
        pred_avg = torch.stack(preds).mean(0)
        with amp_ctx:
            _, bd = loss_fn(pred_avg, y)
        for k, v in bd.items():
            if not k.startswith("w_"):   # BUG-FIX: skip scheduler weights
                totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter groups  (bug-fix: guard film None)
# ─────────────────────────────────────────────────────────────────────────────

def make_param_groups(model: GWAutoencoder, base_lr: float) -> list[dict]:
    enc_ids  = {id(p) for p in model.encoder_stages.parameters()}
    btn_ids  = ({id(p) for p in model.dilated.parameters()} |
                {id(p) for p in model.attn.parameters()})
    dec_ids  = {id(p) for p in model.decoder_stages.parameters()}
    film_ids = ({id(p) for p in model.film.parameters()}
                if model.film is not None else set())   # BUG-FIX: guard None
    other_ids = ({id(p) for p in model.parameters()}
                 - enc_ids - btn_ids - dec_ids - film_ids)

    def _ps(ids):
        return [p for p in model.parameters() if id(p) in ids]

    groups = [
        {"params": _ps(enc_ids),   "lr": base_lr * 1.2,  "name": "encoder"},
        {"params": _ps(btn_ids),   "lr": base_lr * 0.8,  "name": "bottleneck"},
        {"params": _ps(dec_ids),   "lr": base_lr * 1.0,  "name": "decoder"},
        {"params": _ps(other_ids), "lr": base_lr * 1.0,  "name": "other"},
    ]
    if film_ids:
        groups.insert(3, {"params": _ps(film_ids), "lr": base_lr * 1.5, "name": "film"})
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args=None):
    cfg = get_config()
    if args:
        if getattr(args,"epochs",None): cfg.train.epochs       = args.epochs
        if getattr(args,"lr",None):     cfg.train.learning_rate = args.lr
        if getattr(args,"batch",None):  cfg.train.batch_size    = args.batch
        if getattr(args,"device",None): cfg.train.device        = args.device

    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir   = Path(cfg.data.data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5")) if data_dir.exists() else []
    dataset    = None

    if not hdf5_files:
        log.warning("No HDF5 files — synthetic fallback.")
        from torch.utils.data import TensorDataset, DataLoader, random_split
        N = 500; L = int(cfg.data.segment_duration * cfg.data.sample_rate)
        ds  = TensorDataset(torch.randn(N, 1, L), torch.randn(N, 1, L))
        g   = torch.Generator().manual_seed(cfg.train.seed)
        ntr = int(N * cfg.data.train_frac); nva = int(N * cfg.data.val_frac)
        trs, vas, _ = random_split(ds, [ntr, nva, N-ntr-nva], generator=g)
        pm  = {"pin_memory": True,
               **({"pin_memory_device": "cuda"} if device.type == "cuda" else {})}
        train_loader = DataLoader(trs, batch_size=cfg.train.batch_size,
                                   shuffle=True, drop_last=True, **pm)
        val_loader   = DataLoader(vas, batch_size=cfg.train.batch_size, **pm)
    else:
        from data.dataset import make_dataloaders, default_augmentation
        log.info(f"{len(hdf5_files)} HDF5 files")
        train_loader, val_loader, _, dataset = make_dataloaders(
            hdf5_files,
            train_frac=cfg.data.train_frac, val_frac=cfg.data.val_frac,
            batch_size=cfg.train.batch_size, sample_rate=cfg.data.sample_rate,
            segment_duration=cfg.data.segment_duration,
            overlap_frac=cfg.data.window_overlap,
            f_low=cfg.data.f_low, f_high=cfg.data.f_high,
            augment=default_augmentation(), seed=cfg.train.seed,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GWAutoencoder(
        enc_channels=cfg.model.enc_channels, kernel_sizes=cfg.model.kernel_sizes,
        stride=cfg.model.stride, leaky_slope=cfg.model.leaky_slope,
        num_groups=cfg.model.num_groups,
        stochastic_depth_prob=cfg.model.stochastic_depth_prob,
        n_bottleneck_blocks=cfg.model.n_bottleneck_blocks,
        bottleneck_dilation_base=cfg.model.bottleneck_dilation_base,
        bottleneck_kernel=cfg.model.bottleneck_kernel,
        use_attention_gates=cfg.model.use_attention_gates,
        use_bottleneck_attention=cfg.model.use_bottleneck_attention,
    ).to(device)

    if cfg.train.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model); log.info("torch.compile on")
    log.info(f"Parameters: {model.n_parameters:,}")

    ema = EMA(model, cfg.train.ema_decay, warmup_epochs=5) if cfg.train.use_ema else None

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = GWDenoisingLoss(
        mse_weight=cfg.loss.mse_weight, spectral_weight=cfg.loss.spectral_weight,
        snr_proxy_weight=cfg.loss.snr_proxy_weight,
        stft_fft_sizes=cfg.loss.stft_fft_sizes, stft_hop_ratios=cfg.loss.stft_hop_ratios,
        sample_rate=cfg.data.sample_rate, f_low=cfg.data.f_low, f_high=cfg.data.f_high,
    ).to(device)

    # ── Optimiser: AdamW + Lookahead ──────────────────────────────────────────
    param_groups = make_param_groups(model, cfg.train.learning_rate)
    base_opt     = AdamW(param_groups, weight_decay=cfg.train.weight_decay,
                          betas=(0.9, 0.999), eps=1e-8)
    optimizer    = Lookahead(base_opt, k=5, alpha=0.5)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps     = cfg.train.epochs * steps_per_epoch
    n_groups        = len(param_groups)

    if cfg.train.lr_scheduler == "onecycle":
        lr_mults = [1.2, 0.8, 1.0, 1.0]
        if n_groups == 5:          # film group present
            lr_mults = [1.2, 0.8, 1.0, 1.5, 1.0]
        max_lrs = [cfg.train.learning_rate * m for m in lr_mults[:n_groups]]
        scheduler = OneCycleLR(
            base_opt, max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=cfg.train.pct_start,
            anneal_strategy="cos",
            div_factor=10.0, final_div_factor=1e3,
        )
        step_per_batch = True
    else:
        scheduler = CosineAnnealingWarmRestarts(base_opt, T_0=50, T_mult=2)
        step_per_batch = False

    amp_on  = device.type in ("cuda", "mps")
    scaler  = torch.amp.GradScaler(device.type, growth_interval=100) if amp_on else None
    amp_ctx = torch.amp.autocast(device_type=device.type, enabled=amp_on)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if getattr(args, "resume", None):
        rp = Path(args.resume)
        if rp.exists():
            start_epoch, _ = load_checkpoint(rp, model, optimizer, device)
            start_epoch   += 1
            log.info(f"Resuming from epoch {start_epoch}")

    ckpt_dir = Path(cfg.train.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {k: [] for k in [
        "train_total","val_total","train_mse","val_mse",
        "train_stft","val_stft","train_snr","val_snr",
        "train_env","val_env","train_phase","val_phase","train_wass","val_wass",
        "lr","grad_norm_ema",
    ]}
    best_val   = float("inf")
    no_improve = 0
    gn_ema     = 0.0    # EMA of grad norm for smooth logging

    log.info(f"Training {cfg.train.epochs} epochs | "
             f"eff_batch={cfg.train.batch_size * cfg.train.accumulate_grad_batches}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t0 = time.time()
        if dataset is not None:
            dataset.curriculum_epoch = epoch

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_totals: dict[str, float] = {}
        n_tr = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            frac_epoch = epoch - 1 + (step + 1) / steps_per_epoch
            loss_fn.set_epoch(frac_epoch)

            with amp_ctx:
                pred         = model(x)
                loss, bkdown = loss_fn(pred, y)
                ls           = loss / cfg.train.accumulate_grad_batches

            if scaler:
                scaler.scale(ls).backward()
            else:
                ls.backward()

            is_update = ((step + 1) % cfg.train.accumulate_grad_batches == 0
                          or step == len(train_loader) - 1)
            if is_update:
                if scaler:
                    scaler.unscale_(base_opt)
                gn = float(nn.utils.clip_grad_norm_(model.parameters(),
                                                     cfg.train.grad_clip))
                gn_ema = 0.98 * gn_ema + 0.02 * gn   # EMA smoothing

                if scaler:
                    scaler.step(base_opt)
                    scaler.update()
                else:
                    base_opt.step()

                # BUG-FIX: scheduler steps BEFORE Lookahead slow-weight sync
                if step_per_batch:
                    scheduler.step()
                optimizer.step()     # Lookahead sync (after scheduler)
                optimizer.zero_grad(set_to_none=True)

                if ema:
                    ema.update(model, epoch)

            for k, v in bkdown.items():
                if not k.startswith("w_"):
                    tr_totals[k] = tr_totals.get(k, 0.0) + v
            n_tr += 1

        tr_m = {k: v / n_tr for k, v in tr_totals.items()}

        # ── Validate ──────────────────────────────────────────────────────────
        eval_model = ema.shadow if ema else model
        va_m = validate_with_tta(eval_model, val_loader, loss_fn, device)

        if not step_per_batch:
            scheduler.step()

        lr_now  = base_opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        log.info(
            f"E{epoch:04d}/{cfg.train.epochs} | "
            f"tr {tr_m['loss_total']:.4f} "
            f"(mse {tr_m.get('loss_mse',0):.3f} "
            f"stft {tr_m.get('loss_stft',0):.3f} "
            f"snr {tr_m.get('loss_snr',0):.3f} "
            f"env {tr_m.get('loss_envelope',0):.3f} "
            f"wass {tr_m.get('loss_wasserstein',0):.3f}) | "
            f"va {va_m['loss_total']:.4f} | "
            f"lr {lr_now:.2e} | ‖g‖ {gn_ema:.2f} | {elapsed:.1f}s"
        )

        mapping = [
            ("train_total","loss_total",tr_m), ("val_total","loss_total",va_m),
            ("train_mse","loss_mse",tr_m),     ("val_mse","loss_mse",va_m),
            ("train_stft","loss_stft",tr_m),   ("val_stft","loss_stft",va_m),
            ("train_snr","loss_snr",tr_m),     ("val_snr","loss_snr",va_m),
            ("train_env","loss_envelope",tr_m),("val_env","loss_envelope",va_m),
            ("train_phase","loss_phase",tr_m), ("val_phase","loss_phase",va_m),
            ("train_wass","loss_wasserstein",tr_m),("val_wass","loss_wasserstein",va_m),
        ]
        for hk, mk, src in mapping:
            history[hk].append(src.get(mk, 0.0))
        history["lr"].append(lr_now)
        history["grad_norm_ema"].append(gn_ema)

        vl = va_m["loss_total"]
        if vl < best_val:
            best_val = vl; no_improve = 0
            save_checkpoint(ckpt_dir/"best.pt", model, optimizer, epoch, vl, cfg, ema)
        else:
            no_improve += 1

        if epoch % cfg.train.save_every_n_epochs == 0:
            save_checkpoint(ckpt_dir/f"epoch_{epoch:04d}.pt",
                            model, optimizer, epoch, vl, cfg, ema)

        if no_improve >= cfg.train.early_stopping_patience:
            log.info(f"Early stopping at epoch {epoch}."); break

    with open(ckpt_dir/"history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Best val: {best_val:.6f}  |  Done.")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--batch",   type=int,   default=None)
    p.add_argument("--device",  type=str,   default=None)
    p.add_argument("--resume",  type=str,   default=None)
    train(p.parse_args())
