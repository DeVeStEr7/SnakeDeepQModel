import os
import csv
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from envs.snake_dqn_env import SnakeEnv

#basic DQN setup - handles the 38 dim state from the apple position, hot encoding, 5x5 vision block, etc.
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)


#plots the progress as a graph into progress.png
def save_progress_plot(episodes, rewards, apples, out_path="progress.png", window=100):
    if not episodes:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Reward")
    if len(rewards) >= window:
        ma = np.convolve(np.array(rewards), np.ones(window) / window, mode="valid")
        ax1.plot(episodes[window - 1:], ma, color="steelblue", label=f"MA{window}")
    ax1.set_ylabel("Episode Reward")
    ax1.legend()

    ax2.plot(episodes, apples, alpha=0.3, color="tomato", label="Apples eaten")
    if len(apples) >= window:
        ma2 = np.convolve(np.array(apples), np.ones(window) / window, mode="valid")
        ax2.plot(episodes[window - 1:], ma2, color="tomato", label=f"MA{window}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Apples eaten")
    ax2.legend()
    fig.suptitle("Snake DQN Training Progress")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

#grabs the moving average of the past 100 entries
def moving_avg_last(values, window=100):
    if len(values) < window:
        return None
    return float(np.mean(values[-window:]))



def main():
    #fps = 0 and render = false for fast training, but fps = 20 and render = true for actual visual
    env = SnakeEnv(fps=200, render=True)

    #uses number of state dimensions from environment and action dims are up,down,left,right
    state_dim  = SnakeEnv.STATE_DIM
    action_dim = 4

    device = torch.device("cpu")

    q = DQN(state_dim, action_dim).to(device)
    q_target = DQN(state_dim, action_dim).to(device)
    q_target.load_state_dict(q.state_dict())

    #loading previous model if available otherwise will be created later
    checkpoint_path = "best_model.pt"
    start_episode   = 0
    if os.path.exists(checkpoint_path):
        q.load_state_dict(torch.load(checkpoint_path, map_location=device))
        q_target.load_state_dict(q.state_dict())
        print(f"Resuming from {checkpoint_path}")

    #adam optimization like in class
    optimizer = optim.Adam(q.parameters(), lr=1e-4)
    #replaybuffer that keeps track of past 100,000 attempts and overwrites after that   
    replay = deque(maxlen=100_000)

    #used in bellman
    gamma = 0.99
    batch_size = 64
    #syncs every 1000 iterations
    sync_every = 1000
    #determines how much we'll allow exploration (epislon) and how fast it'll decay
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.998

    #metrics to check per episode 
    steps_total  = 0
    episode      = 0
    best_reward  = float("-inf")
    best_apples  = 0

    #logs everything to csv for graphin
    csv_path = "training_log.csv"
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["episode", "steps", "reward", "apples", "epsilon", "ma100_reward", "ma100_apples", "best_reward"])

    #tracking variables
    hist_ep      = []
    hist_rewards = []
    hist_apples  = []
    plot_every   = 100

    try:
        #training loop
        while True:
            #fresh start every episode
            state = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                env.render()
                # epsilon-greedy action selection
                #randomized decision/direction for high epsilons
                if random.random() < epsilon:
                    action = random.randint(0, action_dim - 1)
                else:
                    #otherwise base decision based on q network decision for highest q value
                    with torch.no_grad():
                        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = int(torch.argmax(q(s), dim=1).item())

                #make next step using action to get next state + reward and see if over
                next_state, reward, done = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1

                #add decision + actions to replay
                replay.append((state, action, reward, next_state, done))
                state = next_state

                #learning steps
                if len(replay) >= batch_size:
                    #sample a training batch and only learn after 64 experiences 
                    batch = random.sample(replay, batch_size)
                    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
                    #converting everything to tensors for learning
                    states_t = torch.tensor(np.array(states_b), dtype=torch.float32, device=device)
                    actions_t = torch.tensor(actions_b, dtype=torch.int64, device=device).unsqueeze(1)
                    rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device).unsqueeze(1)
                    next_states_t = torch.tensor(np.array(next_states_b), dtype=torch.float32, device=device)
                    dones_t = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)
                    #runs all states through the Q network then grabs only the q value for the action that was actually taken
                    q_values = q(states_t).gather(1, actions_t)

                    #double dqn portion for evaluation to avoid overestimation
                    with torch.no_grad():
                        #evaluates how good the action actually is
                        next_actions = q(next_states_t).argmax(1, keepdim=True)
                        next_q = q_target(next_states_t).gather(1, next_actions)
                        #bellman equation
                        target = rewards_t + gamma * next_q * (1.0 - dones_t)
                    #compares prediction vs target from bellman equation to get loss
                    loss = nn.SmoothL1Loss()(q_values, target)
                    #backprops and applies gradients to change weights
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    optimizer.step()

                #copies live weights every 1000
                steps_total += 1
                if steps_total % sync_every == 0:
                    q_target.load_state_dict(q.state_dict())

            #slightly decreases epislon via epsilon decay until it reaches min
            #lower epsilon --> less exploration
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            episode += 1
            ep_apples = env.apples_eaten

            hist_ep.append(episode)
            hist_rewards.append(ep_reward)
            hist_apples.append(ep_apples)

            #saving best model to .pt file so we can continue later
            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(q.state_dict(), checkpoint_path)
            #grab best apples episode in run so far
            if ep_apples > best_apples:
                best_apples = ep_apples

            #calculating moving averages and writing to csv
            ma100_r = moving_avg_last(hist_rewards, 100)
            ma100_a = moving_avg_last(hist_apples,  100)
            ma100_r_str = "" if ma100_r is None else f"{ma100_r:.3f}"
            ma100_a_str = "" if ma100_a is None else f"{ma100_a:.2f}"
            writer.writerow([episode, ep_steps, f"{ep_reward:.3f}", ep_apples, f"{epsilon:.3f}", ma100_r_str, ma100_a_str, f"{best_reward:.3f}"])
            csv_file.flush()

            #output to console of progress
            print(f"Episode {episode:4d}, steps in episode ={ep_steps:4d}, reward={ep_reward:6.2f},  apples={ep_apples:3d}, best_apples so far={best_apples:3d}, epsilon val={epsilon:.3f}")
            if episode % plot_every == 0:
                save_progress_plot(hist_ep, hist_rewards, hist_apples, out_path="progress.png", window=50)
                print("Saving progress to training log and making new graph")

    #incase of interruption, save current progress to plot
    except KeyboardInterrupt:
        save_progress_plot(hist_ep, hist_rewards, hist_apples, out_path="progress.png", window=50)
    finally:
        csv_file.close()
        env.close()


if __name__ == "__main__":
    main()