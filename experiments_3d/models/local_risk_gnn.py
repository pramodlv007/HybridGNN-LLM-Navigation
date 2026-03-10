"""
LocalRiskGNN: GNN for Collision Risk Prediction (3D)
======================================================
Adapted from the existing 2D LocalRiskGNN.
Pure PyTorch implementation (no torch-geometric dependency).

Features:
- Processes 7-dim node features: [is_master, pos(3), vel_dir(3)]
- 2-layer message passing for local neighborhood awareness
- Outputs per-node collision risk logits

Note: Uses random weights by default. Architecture is ready for training
with collision data from the PyBullet environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRiskGNN(nn.Module):
    """
    Lightweight GNN for collision risk prediction in 3D.

    Args:
        in_feats: Input feature dimension (default: 7)
        hidden_dim: Hidden layer dimension (default: 64)
        msg_dim: Message dimension (default: 64)
        num_layers: Number of message-passing layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        in_feats: int = 7,
        hidden_dim: int = 64,
        msg_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Node encoder: project input features to hidden dim
        self.node_encoder = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal encoder: project concatenated [x_t, x_t_dt] to hidden
        self.temporal_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # Message: concat source + target → message
            self.message_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, msg_dim),
                nn.ReLU(),
                nn.Linear(msg_dim, msg_dim),
            ))
            # Update: concat node + aggregated messages → updated node
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden_dim + msg_dim, hidden_dim),
                nn.ReLU(),
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Risk head: predict collision risk per node
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        x_t_dt: torch.Tensor,
        edge_index: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        Forward pass to compute collision risk for each node.

        Args:
            x_t: [N, 7] - Current frame features
            x_t_dt: [N, 7] - Next frame features
            edge_index: [2, E] - Edge indices [src, tgt]
            dt: Time delta (not directly used in this version)

        Returns:
            node_risk: [N] - Collision risk logits per node
        """
        N = x_t.size(0)

        # Encode current and next frame
        h_t = self.node_encoder(x_t)       # [N, hidden]
        h_t_dt = self.node_encoder(x_t_dt) # [N, hidden]

        # Temporal fusion
        h = self.temporal_encoder(torch.cat([h_t, h_t_dt], dim=-1))  # [N, hidden]

        # Message passing
        src, tgt = edge_index[0], edge_index[1]  # [E]

        for layer_idx in range(self.num_layers):
            # Compute messages
            h_src = h[src]  # [E, hidden]
            h_tgt = h[tgt]  # [E, hidden]
            msg_input = torch.cat([h_src, h_tgt], dim=-1)  # [E, 2*hidden]
            messages = self.message_layers[layer_idx](msg_input)  # [E, msg_dim]

            # Aggregate (sum)
            agg = torch.zeros(N, messages.size(1), device=h.device)
            agg.index_add_(0, tgt, messages)

            # Update node representations
            update_input = torch.cat([h, agg], dim=-1)  # [N, hidden+msg_dim]
            h = self.update_layers[layer_idx](update_input)
            h = self.norms[layer_idx](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Risk prediction
        risk_logits = self.risk_head(h).squeeze(-1)  # [N]

        return risk_logits
