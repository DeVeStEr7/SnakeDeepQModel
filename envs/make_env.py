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
            grid_size=int(config.get("snake_grid_size", 36)),
            tile_px=int(config.get("snake_tile_px", 20)),
            obs_mode=obs_mode,
            fps=config.get("snake_fps", 10),
            max_steps=config.get("snake_max_steps", 1000),
            reward_apple=config.get("snake_reward_apple", 1.0),
            reward_death=config.get("snake_reward_death", -1.0),
            step_penalty=config.get("snake_step_penalty", 0.0),
            allow_reverse=bool(config.get("snake_allow_reverse", False)),
            include_walls_channel=config.get("snake_walls_channel", True),
            distance_reward_scale=float(config.get("snake_distance_reward_scale", 0.0)),
        )
        env = SnakeEnv(cfg=cfg, render_mode=render_mode, multi_agent=True)
        return env

    raise ValueError(f"Unknown env_name: {env_name!r}")

