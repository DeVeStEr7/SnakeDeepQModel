from envs.pythonGame import SnakeEnv, SnakeConfig



# envs.py
def make_env(config):
    env_name = config.get("env_name", "")

    if env_name.lower() in ["snake", "snakeenv"]:
        # default: flat obs to match your current CriticMLP/Transformer
        obs_mode = config.get("snake_obs_mode", "flat")  # "flat" or "grid"
        render = config.get("render", False)
        render_mode = "human" if render else None

        cfg = SnakeConfig(
            obs_mode=obs_mode,
            fps=config.get("snake_fps", 10),
            max_steps=config.get("snake_max_steps", 1000),
            reward_apple=config.get("snake_reward_apple", 1.0),
            reward_death=config.get("snake_reward_death", -1.0),
            step_penalty=config.get("snake_step_penalty", 0.0),
            include_walls_channel=config.get("snake_walls_channel", True),
        )
        env = SnakeEnv(cfg=cfg, render_mode=render_mode, multi_agent=True)
        return env

