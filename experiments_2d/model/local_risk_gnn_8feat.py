"""
LocalRiskGNN8Feat: Optimized GNN for Real-Time Collision Risk Prediction (8 features)

Features:
- Pure PyTorch implementation (no torch-geometric dependency)
- Processes 8-dim node features: [is_master, pos(3), vel_dir(3), speed_or_yaw(1)]
- Outputs per-node collision risk logits
- Star topology aware (robot → obstacles)
- Efficient for real-time inference (~5-10ms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRiskGNN8Feat(nn.Module):
    """
    Lightweight GNN for collision risk prediction with 8 input features.
    
    Args:
        in_feats: Input feature dimension (default: 8)
        hidden_dim: Hidden layer dimension (default: 64)
        msg_dim: Message dimension (default: 64)
        num_layers: Number of message passing layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    
    Input:
        x_t: [N, 8] - Current frame node features
        x_t_dt: [N, 8] - Next frame node features
        edge_index: [2, E] - Edge connectivity (robot -> obstacles)
        dt: scalar - Time delta between frames
    
    Output:
        node_risk: [N] - Per-node collision risk logits (higher = more dangerous)
    """
    
    def __init__(
        self,
        in_feats: int = 8,
        hidden_dim: int = 64,
        msg_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Separate encoders for different feature types
        self.pos_encoder = nn.Linear(3, hidden_dim // 2)  # Position encoding
        self.vel_encoder = nn.Linear(3, hidden_dim // 2)  # Velocity encoding
        self.extra_encoder = nn.Linear(1, hidden_dim // 4) # Extra feature encoding (speed or yaw)
        
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Edge feature encoder: relative geometry
        # Features: [rel_pos(3), rel_vel(3), dist(1), cos_angle(1)] = 8 dims
        self.edge_encoder = nn.Sequential(
            nn.Linear(8, msg_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Message passing layers
        self.msg_mlps = nn.ModuleList()
        self.node_update_mlps = nn.ModuleList()

        for _ in range(num_layers):
            # Message MLP: combines src node, dst node, and edge features
            self.msg_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * 2 + msg_dim, msg_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(msg_dim, msg_dim),
            ))
            
            # Node update MLP: combines current state + aggregated messages
            self.node_update_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim + msg_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        # Multi-head risk prediction
        self.immediate_risk_head = nn.Linear(hidden_dim, 1)
        self.shortterm_risk_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        x_t: torch.Tensor, 
        x_t_dt: torch.Tensor, 
        edge_index: torch.Tensor, 
        dt
    ) -> torch.Tensor:
        """
        Forward pass to compute collision risk for each node.
        """
        N = x_t.size(0)
        
        # Convert dt to tensor if needed
        if not torch.is_tensor(dt):
            dt = torch.tensor([float(dt)], device=x_t.device, dtype=x_t.dtype)
        
        # Extract features from node representation
        pos_t = x_t[:, 1:4]      # [N, 3] - position (x, y, z)
        vel_dir_t = x_t[:, 4:7]  # [N, 3] - velocity direction (normalized)
        extra_t = x_t[:, 7:8]    # [N, 1] - extra feature
        pos_dt = x_t_dt[:, 1:4]  # [N, 3] - next frame position
        
        # STEP 1: Encode node features separately
        pos_embed = self.pos_encoder(pos_t)       # [N, H/2]
        vel_embed = self.vel_encoder(vel_dir_t)   # [N, H/2]
        extra_embed = self.extra_encoder(extra_t) # [N, H/4]
        
        node_embed = torch.cat([pos_embed, vel_embed, extra_embed], dim=-1)  # [N, H + H/4]
        h = self.node_encoder(node_embed)  # [N, H]
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # STEP 2: Encode edge features (if edges exist)
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            
            rel_pos = pos_t[src] - pos_t[dst]  # [E, 3]
            
            robot_vel = (pos_dt[src] - pos_t[src]) / dt  # [E, 3]
            obstacle_vel = (pos_dt[dst] - pos_t[dst]) / dt  # [E, 3]
            rel_vel = robot_vel - obstacle_vel  # [E, 3]
            
            dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-6  # [E, 1]
            cos_angle = F.cosine_similarity(rel_vel, rel_pos, dim=-1).unsqueeze(-1)  # [E, 1]
            
            edge_features = torch.cat([rel_pos, rel_vel, dist, cos_angle], dim=-1)  # [E, 8]
            edge_embed = self.edge_encoder(edge_features)  # [E, msg_dim]
            
            # STEP 3: Message passing
            for layer_idx in range(self.num_layers):
                h_src = h[src]  # [E, H]
                h_dst = h[dst]  # [E, H]
                
                msg_input = torch.cat([h_src, h_dst, edge_embed], dim=-1)  # [E, 2H + msg_dim]
                messages = self.msg_mlps[layer_idx](msg_input)  # [E, msg_dim]
                
                msg_agg = torch.zeros(N, self.msg_dim, device=h.device, dtype=h.dtype)
                msg_agg.index_add_(0, dst, messages)  # [N, msg_dim]
                
                update_input = torch.cat([h, msg_agg], dim=-1)  # [N, H + msg_dim]
                h = self.node_update_mlps[layer_idx](update_input)  # [N, H]
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # STEP 4: Predict collision risk
        immediate_risk = self.immediate_risk_head(h).squeeze(-1)  # [N]
        shortterm_risk = self.shortterm_risk_head(h).squeeze(-1)  # [N]
        
        node_risk = 0.7 * immediate_risk + 0.3 * shortterm_risk  # [N]
        
        return node_risk

if __name__ == "__main__":
    print("Testing LocalRiskGNN8Feat...")
    
    num_nodes = 6
    num_edges = 5
    
    # 8 features: [is_master, pos_x, pos_y, pos_z, vel_dir_x, vel_dir_y, vel_dir_z, speed]
    x_t = torch.randn(num_nodes, 8)
    x_t[0, 0] = 1.0
    x_t[1:, 0] = 0.0
    
    x_t_dt = torch.randn(num_nodes, 8)
    x_t_dt[:, 0] = x_t[:, 0]
    
    edge_index = torch.tensor([[0] * num_edges, [1, 2, 3, 4, 5]], dtype=torch.long)
    dt = 0.01
    
    model = LocalRiskGNN8Feat(in_feats=8, hidden_dim=64, msg_dim=64, num_layers=2, dropout=0.1)
    model.eval()
    
    with torch.no_grad():
        risk_logits = model(x_t, x_t_dt, edge_index, dt)
        risk_probs = torch.sigmoid(risk_logits)
    
    print(f"Risk probabilities: {risk_probs}")
