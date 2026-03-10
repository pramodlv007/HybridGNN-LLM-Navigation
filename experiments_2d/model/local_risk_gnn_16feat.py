"""
LocalRiskGNN16Feat: Optimized GNN for Real-Time Collision Risk Prediction (16 features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRiskGNN16Feat(nn.Module):
    """
    Lightweight GNN for collision risk prediction with 16 input features.
    
    Args:
        in_feats: Input feature dimension (default: 16)
        hidden_dim: Hidden layer dimension (default: 64)
        msg_dim: Message dimension (default: 64)
        num_layers: Number of message passing layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        in_feats: int = 16,
        hidden_dim: int = 128,
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

        self.node_encoder = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(8, msg_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.msg_mlps = nn.ModuleList()
        self.node_update_mlps = nn.ModuleList()

        for _ in range(num_layers):
            self.msg_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * 2 + msg_dim, msg_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(msg_dim, msg_dim),
            ))
            
            self.node_update_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim + msg_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

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
        
        if not torch.is_tensor(dt):
            dt = torch.tensor([float(dt)], device=x_t.device, dtype=x_t.dtype)
        
        # Features 1:4 are position
        pos_t = x_t[:, 1:4]      
        pos_dt = x_t_dt[:, 1:4]  
        
        h = self.node_encoder(x_t)  # [N, H]
        h = F.dropout(h, p=self.dropout, training=self.training)
        
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
        
        immediate_risk = self.immediate_risk_head(h).squeeze(-1)  # [N]
        shortterm_risk = self.shortterm_risk_head(h).squeeze(-1)  # [N]
        
        node_risk = 0.7 * immediate_risk + 0.3 * shortterm_risk  # [N]
        
        return node_risk

if __name__ == "__main__":
    print("Testing LocalRiskGNN16Feat...")
    
    num_nodes = 6
    num_edges = 5
    
    x_t = torch.randn(num_nodes, 16)
    x_t[0, 0] = 1.0
    x_t[1:, 0] = 0.0
    
    x_t_dt = torch.randn(num_nodes, 16)
    x_t_dt[:, 0] = x_t[:, 0]
    
    edge_index = torch.tensor([[0] * num_edges, [1, 2, 3, 4, 5]], dtype=torch.long)
    dt = 0.01
    
    model = LocalRiskGNN16Feat(in_feats=16, hidden_dim=64, msg_dim=64, num_layers=2, dropout=0.1)
    model.eval()
    
    with torch.no_grad():
        risk_logits = model(x_t, x_t_dt, edge_index, dt)
        risk_probs = torch.sigmoid(risk_logits)
    
    print(f"Risk probabilities: {risk_probs}")
