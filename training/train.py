#!/usr/bin/env python3
"""
training/train.py
-----------------
Clean training pipeline for HyphAeon (PhyloAxialTransformer).
Supports rank-consistent CORAL ordinal binary cross-entropy loss (16 thresholds,
matching the production ``weights/axomeme_v1.pt`` training script in
``training/train_transformer_selection.py``) as well as smooth L1 regression,
5-stratum balanced site resampling, gradient accumulation across variable-shape
alignments, mixed precision (bfloat16 / fp16), differential head learning rate,
and cosine annealing learning rate scheduling.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aeon_core.model import (
    FocalCoralOrdinalLoss,
    PhyloAxialTransformer,
    decode_soft_ordinal_lrt,
)
from aeon_core.weights import save_safetensors
from hyphaeon.training_data import GeneTensorsDataset


def load_initial_checkpoint(model, checkpoint_path):
    """Strictly initialize model weights without resuming optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a state dictionary: {checkpoint_path}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint architecture is incompatible with the requested model: "
            f"{checkpoint_path}"
        ) from exc
    return checkpoint


def collate_single_gene(items):
    """Remove DataLoader's outer batch dimension; gene batching is always one."""
    if len(items) != 1:
        raise ValueError(f"Expected exactly one gene from DataLoader, received {len(items)}")
    return items[0]


def iter_site_indices(eligible_mask, batch_size, shuffle=True, generator=None):
    """Yield each eligible site index exactly once in chunks of ``batch_size``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    indices = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
    if shuffle and indices.numel() > 1:
        order = torch.randperm(indices.numel(), generator=generator)
        indices = indices[order]
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def sample_stratified_site_indices(
    eligible_mask,
    target_lrt,
    samples_per_gene=64,
    proportions=(0.75, 0.10, 0.08, 0.04, 0.03),
    generator=None,
):
    """
    5-stratum balanced site resampling matching the production training loop
    in ``training/train_transformer_selection.py``:
      - Stratum 0 (75%): Null / conserved (LRT == 0)
      - Stratum 1 (10%): Weak sub-significant noise (0 < LRT < 2.71)
      - Stratum 2 ( 8%): Borderline / marginal boundary (2.71 <= LRT < 6.63)
      - Stratum 3 ( 4%): Moderate positive selection (6.63 <= LRT < 15.0)
      - Stratum 4 ( 3%): Strong positive selection (LRT >= 15.0)
    """
    eligible = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
    if eligible.numel() == 0:
        return eligible

    lrt = target_lrt[eligible]
    strata = [
        eligible[lrt == 0.0],
        eligible[(lrt > 0.0) & (lrt < 2.71)],
        eligible[(lrt >= 2.71) & (lrt < 6.63)],
        eligible[(lrt >= 6.63) & (lrt < 15.0)],
        eligible[lrt >= 15.0],
    ]
    targets = [int(round(p * samples_per_gene)) for p in proportions]
    targets[0] = max(0, samples_per_gene - sum(targets[1:]))

    picked = []
    deficit = 0
    for idx_pool, n_target in reversed(list(zip(strata, targets))):
        n_want = n_target + deficit
        if idx_pool.numel() == 0:
            deficit = n_want
            continue
        if idx_pool.numel() >= n_want:
            perm = torch.randperm(idx_pool.numel(), generator=generator)[:n_want]
            picked.append(idx_pool[perm])
            deficit = 0
        else:
            rand_idx = torch.randint(
                0, idx_pool.numel(), (n_want,), generator=generator
            )
            picked.append(idx_pool[rand_idx])
            deficit = 0

    if not picked:
        return eligible
    combined = torch.cat(picked, dim=0)
    perm = torch.randperm(combined.numel(), generator=generator)
    return combined[perm]


def build_optimizer(model, lr=3e-4, weight_decay=1e-2, head_lr_mult=10.0):
    """
    Construct AdamW parameter groups matching ``train_transformer_selection.py``:
    - CORAL ordinal head parameters receive ``lr * head_lr_mult`` (default 10x)
    - Linear weights receive ``weight_decay``; biases, LayerNorms, and embeddings use 0.0
    """
    head_decay, head_no_decay = [], []
    backbone_decay, backbone_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "embedding" in name.lower()
            or "b0" in name
            or "theta_steps" in name
            or "root_token" in name
        )
        if "lrt_ordinal_head" in name:
            (head_no_decay if is_no_decay else head_decay).append(param)
        else:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)

    param_groups = [
        p
        for p in [
            {"params": backbone_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": lr * head_lr_mult, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": lr * head_lr_mult, "weight_decay": 0.0},
        ]
        if len(p["params"]) > 0
    ]
    if not param_groups:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def train_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total_loss = 0.0
    total_sites = 0

    loss_type = getattr(args, "loss", "smooth_l1")
    num_thresholds = getattr(args, "num_thresholds", 16)
    accum_steps = max(1, int(getattr(args, "accum_steps", 1)))
    stratified = bool(getattr(args, "stratified", False))
    use_bf16 = bool(getattr(args, "bf16", False) and device.type == "cuda")
    use_fp16 = bool(getattr(args, "fp16", False) and device.type == "cuda" and not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    coral_criterion = (
        FocalCoralOrdinalLoss(gamma=0.0, num_thresholds=num_thresholds).to(device)
        if loss_type == "coral"
        else None
    )

    optimizer.zero_grad()
    micro_step = 0

    for gene in loader:
        d = gene["d"].to(device)
        z = gene["z"].to(device)
        if stratified:
            sampled_indices = sample_stratified_site_indices(
                gene["eligible_mask"],
                gene["target_lrt"],
                samples_per_gene=args.batch_size,
            )
            site_batches = (
                [sampled_indices] if sampled_indices.numel() > 0 else []
            )
        else:
            site_batches = list(
                iter_site_indices(
                    gene["eligible_mask"], args.batch_size, shuffle=True
                )
            )

        for site_indices in site_batches:
            site_count = site_indices.numel()
            c = gene["c"].index_select(0, site_indices).to(device)
            a = gene["a"].index_select(0, site_indices).to(device)
            y_true = gene["target_lrt"].index_select(0, site_indices).to(device)
            batch_d = d.unsqueeze(0).expand(site_count, -1, -1)
            batch_z = z.unsqueeze(0).expand(site_count, -1, -1)

            with torch.amp.autocast(
                "cuda", enabled=(use_fp16 or use_bf16), dtype=amp_dtype
            ):
                out = model(c, a, batch_d, batch_z)
                if isinstance(out, tuple):
                    y_pred, logits = out[0], out[1]
                else:
                    logits = out
                    y_pred, _ = decode_soft_ordinal_lrt(logits)
                y_true = y_true.reshape(-1)
                if y_pred.shape != y_true.shape:
                    raise ValueError(
                        f"Target shape {tuple(y_true.shape)} is incompatible with "
                        f"decoded prediction shape {tuple(y_pred.shape)}"
                    )
                if loss_type == "coral":
                    loss = coral_criterion(logits, y_true)
                else:
                    loss = nn.functional.smooth_l1_loss(y_pred, y_true, beta=1.0)

            scaled_loss = loss / accum_steps
            scaler.scale(scaled_loss).backward()
            micro_step += 1

            if micro_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * site_count
            total_sites += site_count

    if micro_step % accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if total_sites == 0:
        raise ValueError("No eligible sites were found in any training gene")
    return total_loss / total_sites


def main():
    parser = argparse.ArgumentParser(description="Train HyphAeon Neural Selection Predictor")
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing per-gene .npz tensors"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of sites per gene micro-batch")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps across genes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak backbone learning rate")
    parser.add_argument("--head_lr_mult", type=float, default=10.0, help="LR multiplier for the CORAL ordinal head")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay on linear weights")
    parser.add_argument(
        "--loss",
        choices=("coral", "smooth_l1"),
        default="coral",
        help="Training objective: 'coral' (16-bin ordinal BCE used in production) or 'smooth_l1'",
    )
    parser.add_argument("--num_thresholds", type=int, default=16, help="Number of CORAL ordinal thresholds")
    parser.add_argument("--stratified", action="store_true", help="Use 5-stratum 75/10/8/4/3 balanced site resampling")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of axial transformer layers")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Enable BFloat16 mixed precision")
    parser.add_argument(
        "--init_checkpoint",
        help="Optional checkpoint used to initialize model weights for fine-tuning",
    )
    parser.add_argument("--output_dir", default="weights", help="Directory to save checkpoint snapshots")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GeneTensorsDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_single_gene,
    )

    model = PhyloAxialTransformer(
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        window_size=1,
        num_thresholds=args.num_thresholds,
    )
    if args.init_checkpoint:
        load_initial_checkpoint(model, args.init_checkpoint)
        print(f"[*] Initialized model weights from: {args.init_checkpoint}")
    model = model.to(device)

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        head_lr_mult=args.head_lr_mult,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.50
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.fp16 and device.type == "cuda")
    )

    print(
        f"[*] Starting training on {device} ({args.epochs} epochs, loss={args.loss}, "
        f"batch_size={args.batch_size}, accum_steps={args.accum_steps})..."
    )
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, scaler, device, args)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:2d}/{args.epochs:2d}] - Loss: {loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
        )

        if loss < best_loss:
            best_loss = loss
            ckpt_path = os.path.join(args.output_dir, "hyphaeon_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "loss_type": args.loss,
                    "num_thresholds": args.num_thresholds,
                    "embed_dim": args.embed_dim,
                    "num_layers": args.layers,
                    "num_heads": args.heads,
                    "args": vars(args),
                },
                ckpt_path,
            )
            # Self-describing safetensors artifact (weights + arch metadata in header).
            st_path = os.path.join(args.output_dir, "hyphaeon_best.safetensors")
            save_safetensors(
                model.state_dict(),
                st_path,
                arch={
                    "embed_dim": args.embed_dim,
                    "num_layers": args.layers,
                    "num_heads": args.heads,
                    "window_size": 1,
                },
                metadata={"epoch": str(epoch), "loss": f"{loss:.6f}"},
            )
            print(f"    [✓] Saved new best model to: {ckpt_path} (+ {st_path})")


if __name__ == "__main__":
    main()
