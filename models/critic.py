import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import ensure_batch_agent, flatten_BN


class CriticTransformer(nn.Module):
    def __init__(self, obs_dim, output_size=1, embedding_dim=64, nhead=2,
                 dim_feedforward=128, num_layers=2, device='cpu'):
        super().__init__()
        self.device = device

        self.obs_embedding = nn.Linear(obs_dim, embedding_dim, bias=False)
        self.input_norm = nn.LayerNorm(embedding_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=0.0,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, output_size),
        )

    def forward(self, x):
        """
        x: (B,N,obs_dim)
        return: (B,N,1)
        """
        x = x.to(self.device)
        h = self.obs_embedding(x)
        h = self.input_norm(h)
        h = self.encoder(h)
        return self.out(h)

class CriticV(nn.Module):
    """
    Value function critic: V(s)
    Input:
      - flat: (B,N,D) or (N,D) or (B,D)
      - grid: (B,N,C,H,W) or (B,C,H,W) or (N,C,H,W)
    Output:
      - (B,N,1)  (your MAPPOAgent expects this)
    """
    def __init__(self, obs_dim: int, n_agents: int, hidden: int = 128,
                 cnn_channels: int = 4, cnn_feat: int = 256, device: str = "cpu"):
        super().__init__()
        self.n_agents = n_agents
        self.device = device

        # For flat: per-agent MLP then output scalar per agent
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # For grid: CNN encoder then scalar
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self._cnn_head = None
        self._cnn_feat = cnn_feat

    def _build_cnn_head(self, flat_dim: int):
        self._cnn_head = nn.Sequential(
            nn.Linear(flat_dim, self._cnn_feat),
            nn.ReLU(),
            nn.Linear(self._cnn_feat, 1),
        ).to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        x = ensure_batch_agent(obs)   # (B,N,...) or (B,1,C,H,W)

        if x.dim() == 3:  # (B,N,D)
            B, N, D = x.shape
            v_bn = self.mlp(flatten_BN(x))          # (B*N,1)
            return v_bn.reshape(B, N, 1)

        if x.dim() == 5:  # (B,N,C,H,W)
            B, N, C, H, W = x.shape
            feat = self.cnn(flatten_BN(x))          # (B*N, flat_dim)
            if self._cnn_head is None:
                self._build_cnn_head(feat.shape[-1])
            v_bn = self._cnn_head(feat)             # (B*N,1)
            return v_bn.reshape(B, N, 1)

        raise ValueError(f"Unsupported obs shape: {tuple(obs.shape)}")




class CriticMLP(nn.Module):
    def __init__(self, obs_dim, n_agents, fc1_units=128, fc2_units=128, device='cpu'):
        super().__init__()
        self.device = device
        self.in_dim = obs_dim * n_agents
        self.fc1 = nn.Linear(self.in_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, n_agents)

    def forward(self, x):
        x = x.to(self.device)
        B, N, D = x.shape
        x = x.reshape(B, N * D)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                 # (B,N)
        return x.unsqueeze(-1)          # (B,N,1)