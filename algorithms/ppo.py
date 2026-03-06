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
import functools
import time
print = functools.partial(print, flush=True)




class MAPPOAgent():
    def __init__(self, actor, critic, pi_lr=3e-4, vf_lr=1e-3, entropy_weight=0.01, value_epochs=5, policy_epochs = 5,
                minibatch_size = 256,lam = 0.95,entropy_decay = 0.9995,run_name="ppo_default",entropy_min = 0.002,
                device='cpu',critic_type='default',gamma=0.99,T=2048, env_name='simple_spread_v3'):

        self.policy = actor
        self.value = critic

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.opt_v  = optim.Adam(self.value.parameters(), lr=vf_lr)
        
        self.entropy_weight = entropy_weight
        self.value_epochs = value_epochs
        self.policy_epochs = policy_epochs
        self.lam = lam
        self.run_name = run_name
        self.entropy_min = entropy_min
        self.minibatch_size = minibatch_size
        self.entropy_decay = entropy_decay
        self.device = device
        self.critic_type = critic_type
        self.gamma = gamma
        self.T = T
        self.env_name = env_name
        
        self.out_dir = os.path.join("save_models", self.run_name)
        os.makedirs(self.out_dir, exist_ok=True)

    @torch.no_grad()
    def evaluate(self, env, n_rollouts=10):
        reward_list = []
        for _ in range(n_rollouts):
            obs, _ = env.reset()
            done = False
            reward_sum = 0.0
            while not done:
                obs_tensor = torch.from_numpy(obs).float().to(self.device)

                # obs_tensor: (N, obs_dim) or (N, C, H, W)
                logits = self.policy(obs_tensor)
                actions = torch.argmax(logits, dim=-1)  # (N,)

                obs, rews, done, _ = env.step(actions.cpu().numpy())
                reward_sum += float(np.sum(rews))
            reward_list.append(reward_sum)
        return float(np.mean(reward_list)), float(np.std(reward_list))


    def compute_gae(self,rewards, values, dones, last_v=None):
        """
        Docstring for compute_gae
        
        :param rewards: [T, n_agents]  numpy
        :param values: [T, n_agents]  numpy
        :param dones: [T]  0/1
        :param last_v: [n_agents]
        
        :return:
            adv: [T, n_agents]
            ret: [T, n_agents]
        """
        rewards = np.asarray(rewards)
        values = np.asarray(values)
        dones = np.asarray(dones).astype(np.float32) 
        T,n = rewards.shape
        if last_v is None:
            last_v = np.zeros(n, dtype=np.float32)
        else:
            last_v = np.asarray(last_v) 
        
        adv = np.zeros_like(rewards)
        gae = np.zeros(n,dtype=np.float32)
        
        for t in reversed(range(T)):
            if dones[t]:
                next_v = np.zeros(n, dtype=np.float32)
            else:
                if t == T-1:
                    next_v = last_v
                else:
                    next_v = values[t+1]
            delta  = rewards[t] + self.gamma * next_v - values[t]
            gae    = delta + self.gamma * self.lam * gae * (1.0 - dones[t])
            adv[t] = gae
        ret = adv + values
        return adv, ret

    @torch.no_grad()
    def _collect_batch(self, env):
        obs_list   = []
        act_list   = []
        logp_list  = []
        val_list   = []
        rew_list   = []
        done_list  = []

        obs, _ = env.reset()
        T_i = 0

        while T_i < self.T:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)  # (N, ...) where ... can be obs_dim or C,H,W

            # ---- policy sample ----
            logits = self.policy(obs_tensor)                 # (N, A)
            dist   = Categorical(logits=logits)
            actions = dist.sample()                          # (N,)
            logp    = dist.log_prob(actions)                 # (N,)

            actions_np = actions.cpu().numpy()

            # ---- value ----
            # critic expected: input (B, N, ...) -> output (B, N, 1)
            values = self.value(obs_tensor.unsqueeze(0)).squeeze(0).squeeze(-1)  # (N,)

            obs_next, rewards, done, _ = env.step(actions_np)

            obs_list.append(obs)
            act_list.append(actions_np)
            logp_list.append(logp.cpu().numpy())
            val_list.append(values.cpu().numpy())
            rew_list.append(rewards)   # expect (N,)
            done_list.append(done)

            T_i += 1

            if not done:
                obs = obs_next
            else:
                obs, _ = env.reset()

        # stack: obs can be (T, N, obs_dim) or (T, N, C, H, W)
        obs_arr  = np.stack(obs_list, axis=0)
        act_arr  = np.stack(act_list, axis=0)      # (T, N)
        logp_arr = np.stack(logp_list, axis=0)     # (T, N)
        val_arr  = np.stack(val_list, axis=0)      # (T, N)
        rew_arr  = np.stack(rew_list, axis=0)      # (T, N)
        done_arr = np.array(done_list)             # (T,)

        # last value
        with torch.no_grad():
            if done_arr[-1]:
                last_v = np.zeros(val_arr.shape[1], dtype=np.float32)  # (N,)
            else:
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                last_v = self.value(obs_tensor.unsqueeze(0)).squeeze(0).squeeze(-1).cpu().numpy()

        adv_np, ret_np = self.compute_gae(rew_arr, val_arr, done_arr, last_v=last_v)

        adv = torch.tensor(adv_np, device=self.device, dtype=torch.float32)
        targets = torch.tensor(ret_np, device=self.device, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(act_arr, dtype=torch.long, device=self.device)
        old_logp = torch.tensor(logp_arr, dtype=torch.float32, device=self.device)
        old_vals = torch.tensor(val_arr, dtype=torch.float32, device=self.device)

        return obs_t, actions_t, targets, old_logp, old_vals, adv, rew_arr, done_arr

    def learn(self, obs_t, actions_t, targets_t, old_logp, old_vals, adv, clip_eps=0.2):
        """
        obs_t:
          - flat: [T, N, obs_dim]
          - grid: [T, N, C, H, W]
        """
        T = obs_t.shape[0]
        n_agents = obs_t.shape[1]

        # ---- flatten first two dims, keep the rest ----
        obs_flatten = obs_t.reshape(T * n_agents, *obs_t.shape[2:])        # (T*N, obs_dim) OR (T*N, C, H, W)
        action_flatten = actions_t.reshape(T * n_agents)                   # (T*N,)
        adv_flatten = adv.reshape(T * n_agents)                            # (T*N,)
        old_logp_flat = old_logp.reshape(T * n_agents)                     # (T*N,)

        N = T * n_agents
        idx = torch.arange(N, device=self.device)
        idx_v = torch.arange(T, device=self.device)

        # ---------- policy update ----------
        for _ in range(self.policy_epochs):
            perm = idx[torch.randperm(N)]
            for start in range(0, N, self.minibatch_size):
                mb = perm[start:start + self.minibatch_size]
                s_mb = obs_flatten[mb]
                a_mb = action_flatten[mb]
                adv_mb = adv_flatten[mb]
                oldlp_mb = old_logp_flat[mb]

                logits = self.policy(s_mb)   # works for MLP (B,obs_dim) or CNN (B,C,H,W)
                dist = Categorical(logits=logits)
                newlp = dist.log_prob(a_mb)
                entropy = dist.entropy()

                ratio = torch.exp(newlp - oldlp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                loss_pi = -(torch.min(surr1, surr2)).mean() - self.entropy_weight * entropy.mean()

                self.opt_pi.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt_pi.step()

        # ---------- value update ----------
        # Here we keep your original design: value(obs_t[mb]) where input is (B, N, ...)
        # So your critic should be written to accept (B,N,...) and output (B,N,1).
        for _ in range(self.value_epochs):
            perm = idx_v[torch.randperm(T)]
            for start in range(0, T, self.minibatch_size):
                mb = perm[start:start + self.minibatch_size]

                v = self.value(obs_t[mb]).squeeze(-1)  # (B, N)
                v_old = old_vals[mb]                   # (B, N)
                v_clip = torch.clamp(v, v_old - clip_eps, v_old + clip_eps)

                target_mb = targets_t.detach()[mb]     # (B, N)
                loss_v = torch.max(F.mse_loss(v, target_mb), F.mse_loss(v_clip, target_mb))

                self.opt_v.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.opt_v.step()


    def train(self, env, seed,  max_episodes, goal_mean_100_reward):
        
        print(self.device)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        best_eval = -float("inf")
        best_std = -float("inf")
        best_mean_eval = float("inf")


        ep_rewards_hist = []
        eval_scores = []
        total_steps = 0
        batches_done = 0
        
        eval_steps = []
        eval_means = []
        eval_stds = []

        results = []  

        for episode in range(1, max_episodes + 1):
            #states_t, actions_t, log_probs, old_vals, adv, targets_t, buf = self._collect_batch(env)
            #print("start collecting:",time.time())
            obs_t, actions_t, targets, old_logp, old_vals, adv, rew_arr, done_arr = self._collect_batch(env)
            #print("end collecting",time.time())
            total_steps += len(rew_arr)

            ep_sum = 0.0
            for r, d in zip(rew_arr.sum(axis=1), done_arr):
                ep_sum += r
                if d:
                    ep_rewards_hist.append(ep_sum)
                    ep_sum = 0.0
            #print("before learning:",time.time())
            self.learn(obs_t,actions_t, targets, old_logp, old_vals, adv, clip_eps = 0.2)
            batches_done += 1
            #print("end learing",time.time())
            
            eval_mean, eval_std = self.evaluate(env, n_rollouts=10)
            #print("end evaluating:",time.time())
            eval_scores.append(eval_mean)

            mean100_train = float(np.mean(ep_rewards_hist[-100:])) 
            mean100_eval  = float(np.mean(eval_scores[-100:])) 
            results.append((total_steps, mean100_train, mean100_eval))
            #print(f"[Episode {episode}] [steps {total_steps}] train100={mean100_train:.1f}  eval100={mean100_eval:.1f}")
            print(f"[Episode {episode}] [steps {total_steps}]  "
                    f"train100={mean100_train:.5f}  "
                    f"eval_mean={eval_mean:.5f} (std={eval_std:.5f})  "
                    f"eval100={mean100_eval:.5f}  "
                    f"entropy={self.entropy_weight:.5f}")
            
            eval_steps.append(total_steps)
            eval_means.append(eval_mean)
            eval_stds.append(eval_std)
            
            if eval_mean > best_eval or (eval_mean == best_eval and best_std > eval_std):
                best_eval = float(eval_mean)
                best_std = float(eval_std)
                ckpt_path = os.path.join(self.out_dir, "best_ppo.pt")
                torch.save({
                    "policy": self.policy.state_dict(),
                    "value":  self.value.state_dict(),
                    "eval_mean": best_eval,
                    "eval_std": best_std,
                    "steps": total_steps,
                    "episode": episode,
                }, ckpt_path)
                #print(f"[BEST] {self.run_name}: saved {ckpt_path} (eval={best_eval:.1f})")

            '''recent_mean = np.mean(eval_scores[-100:]) 
            if recent_mean > best_mean_eval:
                best_mean_eval = float(recent_mean)
                ckpt_path = os.path.join(self.out_dir, "best_mean_ppo.pt")
                torch.save({
                    "policy": self.policy.state_dict(),
                    "value":  self.value.state_dict(),
                    "eval_mean": best_mean_eval,
                    "steps": total_steps,
                    "episode": episode,
                }, ckpt_path)'''
            
            
            self.entropy_weight = max(self.entropy_min, self.entropy_weight * self.entropy_decay)
            #print("end episode",time.time())
            if (len(eval_scores) >= 100 and np.mean(eval_scores[-100:]) >= goal_mean_100_reward):
                break

        save_data = {
            "steps":      np.array(eval_steps),
            "eval_means": np.array(eval_means),
            "eval_stds":  np.array(eval_stds),
            "results":    np.array(results, dtype=np.float32),
            "seed":       seed,
            "hyperparams": {
                "pi_lr": self.opt_pi.param_groups[0]["lr"],
                "vf_lr": self.opt_v.param_groups[0]["lr"],
                "entropy_weight": self.entropy_weight,
                "value_epochs": self.value_epochs
            }
        }
        with open(os.path.join(self.out_dir, "PPO_result.pkl"), "wb") as f:
            pickle.dump(save_data, f)
        
        return np.array(results, dtype=np.float32)


# Backwards-compatible alias used by trainer_ppo.py
PPOAgent = MAPPOAgent

__all__ = ["MAPPOAgent", "PPOAgent"]
