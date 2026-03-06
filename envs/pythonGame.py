"""
Snake environment (Gym-like) + optional pygame visualization.

Goals:
- Provide reset()/step() API so you can plug into your PPO/MAPPO code.
- Keep a single .py file (your teammates currently have only one file).
- Support fast training (render_mode=None) and visualization (render_mode="human").
- Provide observations in either:
    - "flat"  : (1, obs_dim) float32   [default, easiest to plug into your current mappo.py]
    - "grid"  : (1, C, H, W) float32   [better for CNN, but your PPO code must handle non-flat]
- Return rewards as shape (1,) float32 to match MAPPOAgent expectations:
    env.step(actions_np) where actions_np has shape (1,) and env returns rewards shape (1,).

Action mapping (int):
0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
"""

from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# pygame is optional for training; only import when needed
try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False


@dataclass
class SnakeConfig:
    grid_size: int = 36          # board size (H=W=36), indices 0..35
    tile_px: int = 20            # render scale
    fps: int = 10                # render FPS
    max_steps: int = 5000        # safety cap per episode
    reward_apple: float = 1.0
    reward_death: float = -1.0
    step_penalty: float = 0.0    # set small negative (e.g. -0.01) if you want
    allow_reverse: bool = False  # typical snake disallows 180-degree reverse
    obs_mode: str = "flat"       # "flat" or "grid"
    include_walls_channel: bool = True


class SnakeEnv:
    """
    A minimal RL environment for Snake:
    - reset(seed=None) -> (obs, info)
    - step(action_np)  -> (obs, rewards, done, info)
    """

    def __init__(
        self,
        cfg: SnakeConfig = SnakeConfig(),
        render_mode: Optional[str] = None,   # None or "human"
        multi_agent: bool = True,            # True => obs shape (1, ...), rewards shape (1,)
    ):
        self.cfg = cfg
        self.render_mode = render_mode
        self.multi_agent = multi_agent

        self.H = self.cfg.grid_size
        self.W = self.cfg.grid_size

        # state
        self.rng = np.random.default_rng(0)
        self.snake = []            # list[(x,y)] head first, in GRID coords
        self.direction = 3          # default RIGHT
        self.apple = None           # (x,y) or None
        self.done = False
        self.steps = 0
        self.score = 0

        # pygame objects (lazy init)
        self._pg_inited = False
        self._screen = None
        self._clock = None
        self._font = None

    # -------------------------
    # Public API
    # -------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        self.done = False
        self.steps = 0
        self.score = 0

        # Initialize snake near center, length=3, heading RIGHT
        cx = self.W // 2
        cy = self.H // 2
        self.direction = 3  # RIGHT

        # head first
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]

        # place apple not on snake
        self.apple = self._spawn_apple()

        obs = self._get_obs()
        info = {"score": self.score, "steps": self.steps}
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        action: can be int, np.int64, or array-like of shape (1,)
        returns:
            obs: np.float32
            rewards: np.float32 shape (1,) if multi_agent else scalar float
            done: bool
            info: dict
        """
        if self.done:
            # If user calls step after done, keep returning done
            obs = self._get_obs()
            r = 0.0
            rew = np.array([r], dtype=np.float32) if self.multi_agent else float(r)
            return obs, rew, True, {"score": self.score, "steps": self.steps, "terminal": True}

        a = self._normalize_action(action)
        if (not self.cfg.allow_reverse) and self._is_reverse(a, self.direction):
            a = self.direction  # ignore illegal reverse
        self.direction = a

        self.steps += 1
        reward = float(self.cfg.step_penalty)

        # compute next head
        hx, hy = self.snake[0]
        nx, ny = self._next_pos(hx, hy, self.direction)

        # collision checks
        # walls: treat boundary cells (0 or H-1, 0 or W-1) as walls
        if self._is_wall(nx, ny):
            self.done = True
            reward += float(self.cfg.reward_death)
            obs = self._get_obs()
            rew = np.array([reward], dtype=np.float32) if self.multi_agent else float(reward)
            return obs, rew, True, self._info(terminal=True, death="wall")

        # self collision:
        # NOTE: moving into tail is allowed if we are not growing (because tail will be removed)
        will_grow = (self.apple is not None and (nx, ny) == self.apple)
        body_to_check = self.snake if will_grow else self.snake[:-1]
        if (nx, ny) in body_to_check:
            self.done = True
            reward += float(self.cfg.reward_death)
            obs = self._get_obs()
            rew = np.array([reward], dtype=np.float32) if self.multi_agent else float(reward)
            return obs, rew, True, self._info(terminal=True, death="self")

        # apply move
        self.snake.insert(0, (nx, ny))

        if will_grow:
            self.score += 1
            reward += float(self.cfg.reward_apple)
            self.apple = self._spawn_apple()
        else:
            self.snake.pop()  # remove tail

        # episode length cap
        if self.steps >= self.cfg.max_steps:
            self.done = True

        obs = self._get_obs()
        done = bool(self.done)
        rew = np.array([reward], dtype=np.float32) if self.multi_agent else float(reward)
        return obs, rew, done, self._info(terminal=done)

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if not _HAS_PYGAME:
            raise RuntimeError("pygame is not available in this environment.")

        self._maybe_init_pygame()

        self._screen.fill((0, 0, 0))
        self._draw_grid()
        self._draw_walls()
        self._draw_apple()
        self._draw_snake()
        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(self.cfg.fps)

    def close(self) -> None:
        if self._pg_inited:
            pygame.quit()
        self._pg_inited = False
        self._screen = None
        self._clock = None
        self._font = None

    def seed(self, seed: int) -> None:
        # Make RNG + python random deterministic
        self.rng = np.random.default_rng(int(seed))
        random.seed(int(seed))

    # -------------------------
    # Observation
    # -------------------------
    def _get_obs(self) -> np.ndarray:
        """
        Returns:
          if obs_mode == "grid":
            obs shape (1, C, H, W) float32 (or (C,H,W) if multi_agent=False)
          if obs_mode == "flat":
            obs shape (1, obs_dim) float32
        Channels:
          0: snake head
          1: snake body (excluding head)
          2: apple
          3: walls (optional)
        """
        C = 4 if self.cfg.include_walls_channel else 3
        grid = np.zeros((C, self.H, self.W), dtype=np.float32)

        # walls channel
        if self.cfg.include_walls_channel:
            grid[3, 0, :] = 1.0
            grid[3, self.H - 1, :] = 1.0
            grid[3, :, 0] = 1.0
            grid[3, :, self.W - 1] = 1.0

        # apple
        if self.apple is not None:
            ax, ay = self.apple
            grid[2, ay, ax] = 1.0  # (y,x) for image-like indexing

        # snake
        if len(self.snake) > 0:
            hx, hy = self.snake[0]
            grid[0, hy, hx] = 1.0
            for (bx, by) in self.snake[1:]:
                grid[1, by, bx] = 1.0

        if self.cfg.obs_mode == "grid":
            obs = grid
            if self.multi_agent:
                obs = obs[None, ...]  # (1,C,H,W)
            return obs.astype(np.float32)

        # "flat": flatten (C*H*W) and return (1, obs_dim)
        flat = grid.reshape(-1).astype(np.float32)
        if self.multi_agent:
            flat = flat[None, :]  # (1, obs_dim)
        return flat

    # -------------------------
    # Helpers
    # -------------------------
    def _normalize_action(self, action) -> int:
        # Accept int, np scalar, list/np array with shape (1,)
        if isinstance(action, (int, np.integer)):
            a = int(action)
        else:
            arr = np.asarray(action).reshape(-1)
            if arr.size == 0:
                a = int(self.direction)
            else:
                a = int(arr[0])
        # clamp to 0..3
        if a < 0 or a > 3:
            a = a % 4
        return a

    def _is_reverse(self, new_dir: int, old_dir: int) -> bool:
        # opposite pairs: (UP,DOWN) (LEFT,RIGHT)
        return (new_dir == 0 and old_dir == 1) or (new_dir == 1 and old_dir == 0) or \
               (new_dir == 2 and old_dir == 3) or (new_dir == 3 and old_dir == 2)

    def _next_pos(self, x: int, y: int, direction: int) -> Tuple[int, int]:
        if direction == 0:   # UP
            return x, y - 1
        if direction == 1:   # DOWN
            return x, y + 1
        if direction == 2:   # LEFT
            return x - 1, y
        # RIGHT
        return x + 1, y

    def _is_wall(self, x: int, y: int) -> bool:
        return (x <= 0) or (x >= self.W - 1) or (y <= 0) or (y >= self.H - 1)

    def _spawn_apple(self) -> Tuple[int, int]:
        # spawn in interior cells [1..W-2] x [1..H-2], not on snake
        snake_set = set(self.snake)
        for _ in range(10_000):
            x = int(self.rng.integers(1, self.W - 1))
            y = int(self.rng.integers(1, self.H - 1))
            if (x, y) not in snake_set:
                return (x, y)
        # fallback (should not happen)
        return (1, 1)

    def _info(self, terminal: bool = False, **kwargs) -> Dict[str, Any]:
        d = {
            "score": int(self.score),
            "steps": int(self.steps),
            "terminal": bool(terminal),
            "snake_len": int(len(self.snake)),
        }
        d.update(kwargs)
        return d

    # -------------------------
    # Rendering (pygame)
    # -------------------------
    def _maybe_init_pygame(self) -> None:
        if self._pg_inited:
            return
        pygame.init()
        px = self.cfg.tile_px
        self._screen = pygame.display.set_mode((self.W * px, self.H * px))
        pygame.display.set_caption("SnakeEnv")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 18)
        self._pg_inited = True

    def _draw_grid(self) -> None:
        px = self.cfg.tile_px
        # light grid lines
        for x in range(0, self.W * px, px):
            pygame.draw.line(self._screen, (40, 40, 40), (x, 0), (x, self.H * px))
        for y in range(0, self.H * px, px):
            pygame.draw.line(self._screen, (40, 40, 40), (0, y), (self.W * px, y))

    def _draw_walls(self) -> None:
        px = self.cfg.tile_px
        # draw boundary walls
        wall_color = (200, 60, 60)
        # top
        pygame.draw.rect(self._screen, wall_color, pygame.Rect(0, 0, self.W * px, px))
        # bottom
        pygame.draw.rect(self._screen, wall_color, pygame.Rect(0, (self.H - 1) * px, self.W * px, px))
        # left
        pygame.draw.rect(self._screen, wall_color, pygame.Rect(0, 0, px, self.H * px))
        # right
        pygame.draw.rect(self._screen, wall_color, pygame.Rect((self.W - 1) * px, 0, px, self.H * px))

    def _draw_apple(self) -> None:
        if self.apple is None:
            return
        px = self.cfg.tile_px
        ax, ay = self.apple
        pygame.draw.rect(
            self._screen, (60, 120, 240),
            pygame.Rect(ax * px, ay * px, px, px)
        )

    def _draw_snake(self) -> None:
        px = self.cfg.tile_px
        if not self.snake:
            return
        # head
        hx, hy = self.snake[0]
        pygame.draw.rect(
            self._screen, (80, 220, 80),
            pygame.Rect(hx * px, hy * px, px, px)
        )
        # body
        for (bx, by) in self.snake[1:]:
            pygame.draw.rect(
                self._screen, (30, 170, 30),
                pygame.Rect(bx * px, by * px, px, px)
            )

    def _draw_hud(self) -> None:
        if self._font is None:
            return
        txt = f"score={self.score}  len={len(self.snake)}  steps={self.steps}  done={self.done}"
        surf = self._font.render(txt, True, (220, 220, 220))
        self._screen.blit(surf, (10, 10))


# ------------------------------------------------------------
# Optional: keyboard play (for debugging / visualization)
# ------------------------------------------------------------
def play_with_keyboard():
    if not _HAS_PYGAME:
        raise RuntimeError("pygame is not installed/available.")

    cfg = SnakeConfig(obs_mode="flat", fps=10)
    env = SnakeEnv(cfg=cfg, render_mode="human", multi_agent=False)
    env.reset(seed=0)

    # keyboard mapping
    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
    }

    running = True
    pending_action = env.direction

    while running:
        # process events first
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    pending_action = key_to_action[event.key]
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # take one env step per frame
        obs, rew, done, info = env.step(pending_action)

        # render
        env.render()

        if done:
            # freeze a bit; press R to reset
            pass

    env.close()


# ------------------------------------------------------------
# Quick sanity test: random agent (no render)
# ------------------------------------------------------------
def quick_random_rollout(n_episodes: int = 3, seed: int = 0):
    cfg = SnakeConfig(obs_mode="flat")
    env = SnakeEnv(cfg=cfg, render_mode=None, multi_agent=True)
    env.seed(seed)
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = env.rng.integers(0, 4, size=(1,), dtype=np.int64)
            obs, rew, done, info = env.step(a)
            ep_ret += float(rew[0])
        print(f"[random] ep={ep} return={ep_ret:.3f} score={info['score']} steps={info['steps']}")


if __name__ == "__main__":
    # Choose one:
    # 1) keyboard demo with rendering:
    # play_with_keyboard()

    # 2) random rollout without rendering:
    quick_random_rollout()