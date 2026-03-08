import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
import random
import pickle
import yaml
import time
import functools
import sys
print = functools.partial(print, flush=True)

from algorithms.ppo import PPOAgent
from models import ActorDiscrete,CriticTransformer,CriticMLP,CriticV
from envs.make_env import make_env
from utils.utils import infer_obs_and_action


class PPOTrainer:
    def __init__(self, config, device):
        self.env = make_env(config)
        self.device = device

        obs_shape, n_agents, act_dim = infer_obs_and_action(self.env)
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.obs_shape = obs_shape

        timestamp = time.strftime("%m%d_%H%M%S")
        run_name = f"{config['exp_name']}_{timestamp}_seed{config['seed']}"

        print(f"env_name: {config.get('env_name')}")
        print(f"critic_type: {config['critic_type']}")
        print(f"obs_shape: {obs_shape}, n_agents={n_agents}, act_dim={act_dim}")
        print(f"run_name: {run_name}")

        # ---- build actor ----
        # If obs is flat: obs_shape=(D,)
        # If obs is grid: obs_shape=(C,H,W)
        if len(obs_shape) == 1:
            obs_dim = obs_shape[0]
            self.actor = ActorDiscrete(
                obs_dim=obs_dim,
                a_size=act_dim,
                hidden_size=config['actor_hidden_size'],
                device=self.device
            ).to(device)
        else:
            C = obs_shape[0]
            # grid input: require ActorDiscrete that supports CNN branch
            # If your ActorDiscrete is still MLP-only, this will fail early (good).
            # For the auto MLP/CNN ActorDiscrete I gave earlier, it works.
            obs_dim_placeholder = 1  # not used in CNN branch
            self.actor = ActorDiscrete(
                obs_dim=obs_dim_placeholder,
                a_size=act_dim,
                hidden_size=config['actor_hidden_size'],
                cnn_channels=int(config.get("actor_cnn_channels", C)),
                cnn_feat=int(config.get("actor_cnn_feat", 256)),
                device=self.device
            ).to(device)

        # ---- build critic ----
        critic_type = str(config['critic_type']).lower()
        if critic_type == "mlp":
            if len(obs_shape) != 1:
                raise ValueError(
                    "critic_type=mlp currently assumes flat obs (N,D). "
                    "For grid obs (N,C,H,W), switch to a CNN critic or use flat obs."
                )
            obs_dim = obs_shape[0]
            self.critic = CriticMLP(
                obs_dim=obs_dim,
                n_agents=n_agents,
                fc1_units=config['fc1_units'],
                fc2_units=config['fc2_units'],
                device=self.device
            ).to(device)
        elif critic_type == "transformer":
            if len(obs_shape) != 1:
                raise ValueError(
                    "CriticTransformer expects flat per-agent vectors (B,N,D). "
                    "For grid obs, add a CNN encoder before transformer, or use flat obs."
                )
            obs_dim = obs_shape[0]
            self.critic = CriticTransformer(
                obs_dim=obs_dim,
                embedding_dim=config['embedding_dim'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dim_feedforward=config['dim_feedforward'],
                device=self.device
            ).to(device)
        elif critic_type in ("v", "criticv", "cnn"):
            # CriticV supports both flat and grid obs; for grid it uses a CNN encoder.
            if len(obs_shape) == 1:
                obs_dim = obs_shape[0]
                cnn_channels = int(config.get("critic_cnn_channels", 4))
            else:
                C = obs_shape[0]
                obs_dim = int(config.get("critic_obs_dim_placeholder", 1))
                cnn_channels = int(config.get("critic_cnn_channels", C))

            self.critic = CriticV(
                obs_dim=obs_dim,
                n_agents=n_agents,
                hidden=int(config.get("critic_hidden", 128)),
                cnn_channels=cnn_channels,
                cnn_feat=int(config.get("critic_cnn_feat", 256)),
                device=self.device,
            ).to(device)
        else:
            raise ValueError(f"Unknown critic_type: {config['critic_type']!r}")

        self.agent = PPOAgent(
            self.actor,
            self.critic,
            pi_lr=config['pi_lr'],
            vf_lr=config['vf_lr'],
            entropy_weight=config['entropy_weight'],
            value_epochs=config['value_epochs'],
            policy_epochs=config['policy_epochs'],
            minibatch_size=config['minibatch_size'],
            lam=config['lam'],
            entropy_decay=config['entropy_decay'],
            run_name=run_name,
            entropy_min=config['entropy_min'],
            device=self.device,
            critic_type=config['critic_type'],
            gamma=config['gamma'],
            T=config['T'],
            env_name=config.get('env_name', 'unknown')
        )

    def train(self, config):
        print("Training started...")
        self.agent.train(
            env=self.env,
            seed=config['seed'],
            max_episodes=config['max_episodes'],
            goal_mean_100_reward=config['goal_mean_100_reward'],
        )
        print("Training finished.")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ppo_snake_transformer-2.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    print("===== CONFIG (yaml) =====")
    try:
        print(yaml.safe_dump(config, sort_keys=True))
    except Exception:
        print(config)
    print("=========================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = PPOTrainer(config=config,device=device)
    trainer.train(config=config)