"""
Physics-Informed GNN Training
===============================
Trains the LocalRiskGNN with three loss components:
  1. Collision Label Loss (BCE): Predict whether collision occurs within 2s
  2. TTC Correlation Loss (MSE): Risk should inversely correlate with TTC
  3. Momentum Loss (MSE): Risk should scale with closing speed

Usage:
  python training/train_physics_gnn.py --data training/collision_data.pt --epochs 100
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.local_risk_gnn import LocalRiskGNN


class CollisionDataset(Dataset):
    """Dataset for physics-informed GNN training."""

    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.x_t = data["x_t"]           # [N_samples, N_nodes, 7]
        self.x_t_dt = data["x_t_dt"]     # [N_samples, N_nodes, 7]
        self.edge_index = data["edge_index"]  # [2, E] - same for all
        self.labels = data["collision_labels"]  # [N_samples]
        self.ttc_values = data["ttc_values"]   # [N_samples, N_obstacles]
        self.closing_speeds = data["closing_speeds"]  # [N_samples, N_obstacles]
        self.num_samples = len(self.labels)

        print(f"  Loaded {self.num_samples} samples")
        pos = (self.labels > 0.5).sum().item()
        print(f"  Positive: {pos} ({pos/self.num_samples*100:.1f}%)")
        print(f"  Negative: {self.num_samples - pos} ({(self.num_samples-pos)/self.num_samples*100:.1f}%)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "x_t": self.x_t[idx],
            "x_t_dt": self.x_t_dt[idx],
            "label": self.labels[idx],
            "ttc": self.ttc_values[idx],
            "closing_speed": self.closing_speeds[idx],
        }


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with three physics-informed components:

    L = L_collision + alpha * L_ttc + beta * L_momentum

    L_collision: BCE between predicted robot risk and collision label
    L_ttc: MSE between predicted obstacle risks and 1/(TTC+0.5)
    L_momentum: MSE between predicted obstacle risks and normalized closing speed
    """

    def __init__(self, alpha=0.3, beta=0.1, pos_weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # Higher weight for positive (collision) samples since they're rarer
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, risk_logits, labels, ttc_values, closing_speeds):
        """
        Args:
            risk_logits: [B, N] - per-node risk logits (B=batch, N=nodes)
            labels: [B] - collision labels (0 or 1)
            ttc_values: [B, N_obs] - TTC values per obstacle
            closing_speeds: [B, N_obs] - closing speeds per obstacle
        """
        # 1. Collision Label Loss: robot node (index 0) risk vs collision label
        robot_risk = risk_logits[:, 0]  # [B]
        loss_collision = self.bce(robot_risk, labels)

        # 2. TTC Correlation Loss: obstacle risk should inversely correlate with TTC
        # Target: risk = 1/(TTC + 0.5) capped at 1.0
        obstacle_risks = torch.sigmoid(risk_logits[:, 1:])  # [B, N_obs] in [0,1]
        ttc_target = torch.clamp(1.0 / (ttc_values + 0.5), 0.0, 1.0)  # [B, N_obs]
        loss_ttc = nn.functional.mse_loss(obstacle_risks, ttc_target)

        # 3. Momentum Loss: obstacle risk should scale with closing speed
        # Normalize closing speed to [0, 1] range
        max_closing = closing_speeds.abs().max() + 1e-6
        momentum_target = torch.clamp(closing_speeds / max_closing, 0.0, 1.0)
        loss_momentum = nn.functional.mse_loss(obstacle_risks, momentum_target)

        total_loss = loss_collision + self.alpha * loss_ttc + self.beta * loss_momentum

        return total_loss, {
            "collision": loss_collision.item(),
            "ttc": loss_ttc.item(),
            "momentum": loss_momentum.item(),
            "total": total_loss.item(),
        }


def train(args):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_path = os.path.join(base_dir, args.data)
    model_path = os.path.join(base_dir, args.output)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("=" * 60)
    print("Physics-Informed GNN Training")
    print(f"Data: {data_path}")
    print(f"Output: {model_path}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"Loss weights: alpha(TTC)={args.alpha}, beta(momentum)={args.beta}")
    print("=" * 60)

    # Load dataset
    dataset = CollisionDataset(data_path)
    edge_index = dataset.edge_index  # Shared across all samples

    # Train/val split (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {n_train} | Val: {n_val}")

    # Model
    model = LocalRiskGNN(in_feats=7, hidden_dim=64, msg_dim=64, num_layers=2, dropout=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = PhysicsInformedLoss(alpha=args.alpha, beta=args.beta, pos_weight=3.0)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count}")
    print("-" * 60)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_losses = {"collision": 0, "ttc": 0, "momentum": 0, "total": 0}
        n_batches = 0

        for batch in train_loader:
            x_t = batch["x_t"]           # [B, N, 7]
            x_t_dt = batch["x_t_dt"]     # [B, N, 7]
            labels = batch["label"]      # [B]
            ttc = batch["ttc"]           # [B, N_obs]
            cs = batch["closing_speed"]  # [B, N_obs]

            B, N, F = x_t.shape
            
            # Flatten to [B*N, 7]
            x_t_flat = x_t.view(B * N, F)
            x_t_dt_flat = x_t_dt.view(B * N, F)

            # Expand edge_index: [2, E] -> [2, B*E] with offsets
            E = edge_index.size(1)
            offsets = torch.arange(B, device=x_t.device) * N
            # edge_index is [2, E], offsets is [B]
            # We want [B, 2, E] where each b has edge_index + offsets[b]
            edge_index_batched = edge_index.unsqueeze(0) + offsets.view(B, 1, 1)
            # Reshape to [2, B*E]
            edge_index_batched = edge_index_batched.transpose(0, 1).reshape(2, -1)

            # Vectorized forward pass
            risk_flat = model(x_t_flat, x_t_dt_flat, edge_index_batched, dt=0.05)  # [B*N]
            batch_risks = risk_flat.view(B, N)  # [B, N]

            loss, loss_dict = criterion(batch_risks, labels, ttc, cs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in train_losses:
                train_losses[k] += loss_dict[k]
            n_batches += 1

        for k in train_losses:
            train_losses[k] /= n_batches

        # --- Validate ---
        model.eval()
        val_losses = {"collision": 0, "ttc": 0, "momentum": 0, "total": 0}
        n_val_batches = 0
        correct = 0
        total_preds = 0

        with torch.no_grad():
            for batch in val_loader:
                x_t = batch["x_t"]
                x_t_dt = batch["x_t_dt"]
                labels = batch["label"]
                ttc = batch["ttc"]
                cs = batch["closing_speed"]

                B, N, F = x_t.shape
                x_t_flat = x_t.view(B * N, F)
                x_t_dt_flat = x_t_dt.view(B * N, F)

                E = edge_index.size(1)
                offsets = torch.arange(B, device=x_t.device) * N
                edge_index_batched = edge_index.unsqueeze(0) + offsets.view(B, 1, 1)
                edge_index_batched = edge_index_batched.transpose(0, 1).reshape(2, -1)

                risk_flat = model(x_t_flat, x_t_dt_flat, edge_index_batched, dt=0.05)
                batch_risks = risk_flat.view(B, N)

                _, loss_dict = criterion(batch_risks, labels, ttc, cs)

                # Accuracy (robot node prediction)
                robot_preds = (torch.sigmoid(batch_risks[:, 0]) > 0.5).float()
                correct += (robot_preds == labels).sum().item()
                total_preds += labels.size(0)

                for k in val_losses:
                    val_losses[k] += loss_dict[k]
                n_val_batches += 1

        for k in val_losses:
            val_losses[k] /= max(n_val_batches, 1)

        val_acc = correct / max(total_preds, 1)
        scheduler.step(val_losses["total"])

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        # Log
        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: {train_losses['total']:.4f} "
                  f"(col={train_losses['collision']:.3f} ttc={train_losses['ttc']:.3f} "
                  f"mom={train_losses['momentum']:.3f}) | "
                  f"Val: {val_losses['total']:.4f} Acc={val_acc:.2f} | "
                  f"LR={lr:.6f}")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best epoch: {best_epoch} (val loss: {best_val_loss:.4f})")
    print(f"  Model saved: {model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Physics-Informed GNN")
    parser.add_argument("--data", type=str, default="training/collision_data.pt")
    parser.add_argument("--output", type=str, default="models/trained_risk_gnn.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.3, help="TTC loss weight")
    parser.add_argument("--beta", type=float, default=0.1, help="Momentum loss weight")
    args = parser.parse_args()

    train(args)
