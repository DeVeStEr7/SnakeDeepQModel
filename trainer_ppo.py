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
print = functools.partial(print, flush=True)


from algorithms.ppo import PPOAgent
from models import ActorDiscrete,CriticTransformer,CriticMLP,CriticV
from envs import make_env
from utils.utils import infer_obs_and_action


class PPOTrainer:
    def __init__(self, config, device):
        self.env = make_env(config)
        self.device = device

        obs_shape, n_agents, act_dim = infer_obs_and_action(self.env)
        self.act_dim = act_dim

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
            # grid input: require ActorDiscrete that supports CNN branch
            # If your ActorDiscrete is still MLP-only, this will fail early (good).
            # For the auto MLP/CNN ActorDiscrete I gave earlier, it works.
            obs_dim_placeholder = 1  # not used in CNN branch
            self.actor = ActorDiscrete(
                obs_dim=obs_dim_placeholder,
                a_size=act_dim,
                hidden_size=config['actor_hidden_size'],
                device=self.device
            ).to(device)

        # ---- build critic ----
        if config['critic_type'] == "mlp":
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
        else:
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
    with open("configs/ppo_snake_transformer.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = PPOTrainer(config=config,device=device)
    trainer.train(config=config)