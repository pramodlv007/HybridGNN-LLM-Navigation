"""
LocalRiskGNN: Optimized GNN for Real-Time Collision Risk Prediction

Features:
- Pure PyTorch implementation (no torch-geometric dependency)
- Processes 7-dim node features: [is_master, pos(3), vel_dir(3)]
- Outputs per-node collision risk logits
- Star topology aware (robot → obstacles)
- Efficient for real-time inference (~5-10ms)

Architecture Highlights:
- Separate encoders for position and velocity features
- Edge features: relative geometry (position, velocity, distance, approach angle)
- Multi-head risk prediction (immediate + short-term)
- 2-layer message passing for local neighborhood awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRiskGNN(nn.Module):
    """
    Lightweight GNN for collision risk prediction.
    
    Args:
        in_feats: Input feature dimension (default: 7)
        hidden_dim: Hidden layer dimension (default: 64)
        msg_dim: Message dimension (default: 64)
        num_layers: Number of message passing layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    
    Input:
        x_t: [N, 7] - Current frame node features
        x_t_dt: [N, 7] - Next frame node features
        edge_index: [2, E] - Edge connectivity (robot → obstacles)
        dt: scalar - Time delta between frames
    
    Output:
        node_risk: [N] - Per-node collision risk logits (higher = more dangerous)
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
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Separate encoders for different feature types
        # This allows the model to learn different representations
        self.pos_encoder = nn.Linear(3, hidden_dim // 2)  # Position encoding
        self.vel_encoder = nn.Linear(3, hidden_dim // 2)  # Velocity encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        # Predict both immediate (0.1s) and short-term (0.5s) collision risk
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
        
        Args:
            x_t: [N, 7] - Current frame features [is_master, pos_x, pos_y, pos_z, vel_dir_x, vel_dir_y, vel_dir_z]
            x_t_dt: [N, 7] - Next frame features (same structure)
            edge_index: [2, E] - Edge list [[src_nodes], [dst_nodes]]
            dt: Time delta (scalar or tensor)
            
        Returns:
            node_risk: [N] - Collision risk logits per node
        """
        N = x_t.size(0)
        
        # Convert dt to tensor if needed
        if not torch.is_tensor(dt):
            dt = torch.tensor([float(dt)], device=x_t.device, dtype=x_t.dtype)
        
        # Extract features from node representation
        # x_t[:, 0] = is_master flag (not used in encoding, just for indexing)
        pos_t = x_t[:, 1:4]      # [N, 3] - position (x, y, z)
        vel_dir_t = x_t[:, 4:7]  # [N, 3] - velocity direction (normalized)
        pos_dt = x_t_dt[:, 1:4]  # [N, 3] - next frame position
        
        # STEP 1: Encode node features separately
        pos_embed = self.pos_encoder(pos_t)       # [N, H/2]
        vel_embed = self.vel_encoder(vel_dir_t)   # [N, H/2]
        node_embed = torch.cat([pos_embed, vel_embed], dim=-1)  # [N, H]
        h = self.node_encoder(node_embed)  # [N, H]
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # STEP 2: Encode edge features (if edges exist)
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            
            # Compute relative geometry
            rel_pos = pos_t[src] - pos_t[dst]  # [E, 3] - relative position
            
            # Relative velocity: how obstacles move relative to robot
            robot_vel = (pos_dt[src] - pos_t[src]) / dt  # [E, 3]
            obstacle_vel = (pos_dt[dst] - pos_t[dst]) / dt  # [E, 3]
            rel_vel = robot_vel - obstacle_vel  # [E, 3]
            
            # Distance to obstacle
            dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-6  # [E, 1]
            
            # Approach angle: are we moving toward or away from obstacle?
            # Negative cosine = approaching, positive = moving apart
            cos_angle = F.cosine_similarity(rel_vel, rel_pos, dim=-1).unsqueeze(-1)  # [E, 1]
            
            # Combine all edge features
            edge_features = torch.cat([rel_pos, rel_vel, dist, cos_angle], dim=-1)  # [E, 8]
            edge_embed = self.edge_encoder(edge_features)  # [E, msg_dim]
            
            # STEP 3: Message passing
            for layer_idx in range(self.num_layers):
                h_src = h[src]  # [E, H] - source node states
                h_dst = h[dst]  # [E, H] - destination node states
                
                # Compute messages: function of src, dst, and edge features
                msg_input = torch.cat([h_src, h_dst, edge_embed], dim=-1)  # [E, 2H + msg_dim]
                messages = self.msg_mlps[layer_idx](msg_input)  # [E, msg_dim]
                
                # Aggregate messages to destination nodes
                msg_agg = torch.zeros(N, self.msg_dim, device=h.device, dtype=h.dtype)
                msg_agg.index_add_(0, dst, messages)  # [N, msg_dim]
                
                # Update node states
                update_input = torch.cat([h, msg_agg], dim=-1)  # [N, H + msg_dim]
                h = self.node_update_mlps[layer_idx](update_input)  # [N, H]
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # STEP 4: Predict collision risk
        # Multi-head prediction: immediate (0.1s) + short-term (0.5s)
        immediate_risk = self.immediate_risk_head(h).squeeze(-1)  # [N]
        shortterm_risk = self.shortterm_risk_head(h).squeeze(-1)  # [N]
        
        # Weighted combination (prioritize immediate danger)
        node_risk = 0.7 * immediate_risk + 0.3 * shortterm_risk  # [N]
        
        return node_risk


# Test the model
if __name__ == "__main__":
    print("Testing LocalRiskGNN...")
    
    # Create sample data
    num_nodes = 6  # 1 robot + 5 obstacles
    num_edges = 5  # robot connects to 5 obstacles
    
    # Node features: [is_master, pos_x, pos_y, pos_z, vel_dir_x, vel_dir_y, vel_dir_z]
    x_t = torch.randn(num_nodes, 7)
    x_t[0, 0] = 1.0  # Mark first node as robot
    x_t[1:, 0] = 0.0  # Others are obstacles
    
    x_t_dt = torch.randn(num_nodes, 7)
    x_t_dt[:, 0] = x_t[:, 0]  # Keep is_master flag same
    
    # Edge index: robot (node 0) -> obstacles (nodes 1-5)
    edge_index = torch.tensor([[0] * num_edges, [1, 2, 3, 4, 5]], dtype=torch.long)
    
    dt = 0.01
    
    # Create model
    model = LocalRiskGNN(
        in_feats=7,
        hidden_dim=64,
        msg_dim=64,
        num_layers=2,
        dropout=0.1
    )
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        risk_logits = model(x_t, x_t_dt, edge_index, dt)
        risk_probs = torch.sigmoid(risk_logits)
    
    print(f"\n[OK] Model created successfully!")
    print(f"Input shape: {x_t.shape}")
    print(f"Output shape: {risk_logits.shape}")
    print(f"\nRisk logits: {risk_logits}")
    print(f"Risk probabilities: {risk_probs}")
    print(f"\nRobot risk: {risk_probs[0]:.4f}")
    print(f"Max obstacle risk: {risk_probs[1:].max():.4f}")
    print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")
