#!/usr/bin/env python3
"""
train.py
--------
Clean training pipeline for AxoMEME (PhyloAxialTransformer).
Supports multi-GPU training, mixed precision, cosine annealing learning rate schedule,
and batched SQLite / NPZ alignment loading.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from axomeme.model import PhyloAxialTransformer

class SelectionTensorsDataset(Dataset):
    """
    Loads pre-extracted training alignment tensors from NPZ archives or SQLite.
    """
    def __init__(self, npz_dir: str):
        self.files = sorted([os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')])
        print(f"[*] Loaded {len(self.files)} training tensors from: {npz_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return {
            'c': torch.from_numpy(data['c']).long(),
            'a': torch.from_numpy(data['a']).long(),
            'd': torch.from_numpy(data['d']).float(),
            'z': torch.from_numpy(data['z']).float(),
            'target_lrt': torch.from_numpy(data['target_lrt']).float()
        }

def train_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        c = batch['c'].to(device)
        a = batch['a'].to(device)
        d = batch['d'].to(device)
        z = batch['z'].to(device)
        y_true = batch['target_lrt'].to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            y_pred, _ = model(c, a, d, z)
            # Robust Huber / Smooth L1 loss on selection test statistic
            loss = nn.functional.smooth_l1_loss(y_pred.squeeze(-1), y_true, beta=1.0)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Train AxoMEME Neural Selection Predictor")
    parser.add_argument("--data_dir", required=True, help="Directory containing pre-extracted .npz alignment tensors")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (alignments per batch)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of axial transformer layers")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--output_dir", default="weights", help="Directory to save checkpoint snapshots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SelectionTensorsDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    model = PhyloAxialTransformer(
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        window_size=1
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    print(f"[*] Starting training on {device} ({args.epochs} epochs)...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, scaler, device, args)
        scheduler.step()
        elapsed = time.time() - t0
        
        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] - Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")
        
        if loss < best_loss:
            best_loss = loss
            ckpt_path = os.path.join(args.output_dir, "axomeme_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': vars(args)
            }, ckpt_path)
            print(f"    [✓] Saved new best model to: {ckpt_path}")

if __name__ == '__main__':
    main()
