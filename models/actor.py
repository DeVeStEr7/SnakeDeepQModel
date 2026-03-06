import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical



class ActorContinuous(nn.Module):
    def __init__(self, s_size, a_size, fc1_units=32, fc2_units=32,device='cpu'):
        super(ActorContinuous, self).__init__()
        self.fc1 = nn.Linear(s_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, a_size)
        #self.log_std = nn.Parameter(torch.full((a_size,), math.log(0.15)), requires_grad=False)
        self.log_std = nn.Parameter(torch.full((a_size,), math.log(0.15)))
        self.device = device


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def select_action(self, obs):
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        
        action = dist.rsample()
        #action_env = action.clamp(-1.0, 1.0)            
        action_env = torch.sigmoid(action)
        
        logproba = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1) 
        
        return action_env, logproba, entropy

    def select_greedy_action(self, obs):
        mean = self.forward(obs)
        return mean




class ActorDiscrete(nn.Module):
    def __init__(self, obs_dim: int, a_size: int, hidden_size: int = 128,
                 cnn_channels: int = 4, cnn_feat: int = 256, device: str = "cpu"):
        super().__init__()
        self.a_size = a_size
        self.device = device

        # MLP branch (for flat obs)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, a_size),
        )

        # CNN branch (for grid obs: (C,H,W))
        # assume obs channels = cnn_channels (e.g., 3 or 4 from SnakeEnv)
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # lazy linear head for CNN (depends on H,W); set at first forward
        self._cnn_head = None
        self._cnn_feat = cnn_feat

    def _build_cnn_head(self, flat_dim: int):
        self._cnn_head = nn.Sequential(
            nn.Linear(flat_dim, self._cnn_feat),
            nn.ReLU(),
            nn.Linear(self._cnn_feat, self.a_size),
        ).to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns logits
        - input obs could be:
            (N, D) or (B, N, D)     -> output (N,A) or (B,N,A)
            (N, C,H,W) or (B,N,C,H,W) or (B,C,H,W) -> same pattern
        """
        obs = obs.to(self.device)

        # Normalize shape to (B,N,...) then flatten to (B*N,...)
        x = ensure_batch_agent(obs)

        # Case 1: flat features
        if x.dim() == 3:  # (B,N,D)
            x_bn = flatten_BN(x)  # (B*N,D)
            logits_bn = self.mlp(x_bn)  # (B*N,A)
            B, N = x.shape[:2]
            return logits_bn.reshape(B, N, self.a_size).squeeze(0) if obs.dim() == 2 else logits_bn.reshape(B, N, self.a_size)

        # Case 2: grid features
        if x.dim() == 5:  # (B,N,C,H,W)
            x_bn = flatten_BN(x)  # (B*N,C,H,W)
            feat = self.cnn(x_bn)  # (B*N, flat_dim)
            if self._cnn_head is None:
                self._build_cnn_head(feat.shape[-1])
            logits_bn = self._cnn_head(feat)  # (B*N,A)
            B, N = x.shape[:2]
            out = logits_bn.reshape(B, N, self.a_size)
            # if original input was (B,C,H,W) we added N=1; return (B,A)
            if obs.dim() == 4:
                return out[:, 0, :]
            # if original input was (N,C,H,W), return (N,A)
            if obs.dim() == 4 and obs.shape[0] != x.shape[0]:
                return out.squeeze(0)
            return out.squeeze(0) if obs.dim() == 4 and obs.shape[0] == x.shape[1] else out

        raise ValueError(f"Unsupported obs shape: {tuple(obs.shape)}")

    def select_action(self, obs: torch.Tensor):
        logits = self.forward(obs)
        # logits could be (N,A) or (B,N,A) or (B,A)
        if logits.dim() == 3:
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, logprob, entropy
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, logprob, entropy

    def select_greedy_action(self, obs: torch.Tensor):
        logits = self.forward(obs)
        return torch.argmax(logits, dim=-1)

