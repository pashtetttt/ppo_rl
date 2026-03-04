"""
Pygame visualization for Pong training/evaluation.
Shows game frame (or observation), score, episode stats, and algorithm metrics.
"""

import pygame
import numpy as np
from collections import deque
import time


# Colors (dark theme)
BG = (18, 18, 24)
PANEL = (28, 28, 36)
TEXT = (220, 220, 230)
ACCENT = (100, 180, 255)
GREEN = (100, 220, 140)
RED = (255, 100, 100)
GOLD = (255, 200, 80)


def resize_frame(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Resize frame to (th, tw, 3) uint8. Uses numpy only to avoid loading OpenCV's SDL (conflicts with Pygame on macOS)."""
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
    # Env uses scale_obs=True so values are in [0, 1]; scale to [0, 255] for display
    if frame.dtype == np.floating or (hasattr(frame, 'dtype') and np.issubdtype(frame.dtype, np.floating)):
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    else:
        frame = np.asarray(frame, dtype=np.uint8)
    h, w = frame.shape[0], frame.shape[1]
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    for i in range(th):
        for j in range(tw):
            out[i, j] = frame[int(i * h / th), int(j * w / tw)]
    return out


class Visualization:
    """
    Pygame window for live training/eval: game view, score, episode, returns, losses.
    """

    def __init__(
        self,
        width: int = 900,
        height: int = 700,
        game_scale: int = 2,
        font_size: int = 16,
        title: str = "PPO vs REINFORCE — Atari Pong",
    ):
        pygame.init()
        pygame.display.set_caption(title)
        self.width = width
        self.height = height
        self.game_scale = game_scale
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), font_size)
        self.large_font = pygame.font.Font(pygame.font.get_default_font(), font_size + 4)
        self.recent_returns: deque = deque(maxlen=30)
        self.recent_loss: deque = deque(maxlen=50)
        self.episode_rewards: deque = deque(maxlen=100)
        self.last_frame: np.ndarray | None = None
        self.algorithm_name = "—"
        self.current_score_agent = 0
        self.current_score_opponent = 0
        self.episode = 0
        self.global_step = 0
        self.fps_target = 30
        self.running = True

    def set_algorithm(self, name: str):
        self.algorithm_name = name

    def set_scores(self, agent: int, opponent: int):
        self.current_score_agent = agent
        self.current_score_opponent = opponent

    def set_episode_step(self, episode: int, step: int):
        self.episode = episode
        self.global_step = step

    def add_return(self, r: float):
        self.recent_returns.append(r)

    def add_loss(self, loss: float):
        self.recent_loss.append(loss)

    def add_episode_reward(self, r: float):
        self.episode_rewards.append(r)

    def set_frame(self, frame: np.ndarray):
        """frame: (H, W) or (H, W, C), any size; will be scaled for display."""
        self.last_frame = frame

    def set_frame_from_obs(self, obs: np.ndarray):
        """Use first channel of stacked obs (4, 84, 84) or (84, 84, 4) as display frame."""
        if obs.ndim == 3:
            if obs.shape[0] == 4 or obs.shape[0] == 84:
                frame = obs[-1] if obs.shape[0] == 4 else obs[:, :, -1]
            else:
                frame = obs[:, :, 0] if obs.shape[2] <= 4 else obs
        else:
            frame = obs
        self.last_frame = frame

    def draw_graph(self, surface: pygame.Surface, x: int, y: int, w: int, h: int, values: deque, color: tuple, label: str):
        if not values:
            return
        vals = list(values)
        mx = max(vals) if max(vals) > 0 else 1
        mn = min(vals)
        rng = mx - mn if mx != mn else 1
        pts = []
        for i, v in enumerate(vals):
            px = x + int((i / max(len(vals) - 1, 1)) * (w - 1))
            py = y + h - 1 - int((v - mn) / rng * (h - 1))
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, 2)
        text = self.font.render(label, True, TEXT)
        surface.blit(text, (x, y - 18))

    def render(self, env=None):
        """Draw one frame. If env is provided and has render, use rgb_array for game view."""
        self.screen.fill(BG)
        w, h = self.screen.get_size()

        # Game view area (left side)
        gw, gh = 84 * self.game_scale, 84 * self.game_scale
        gx, gy = 24, 24
        game_rect = pygame.Rect(gx, gy, gw, gh)
        pygame.draw.rect(self.screen, PANEL, game_rect)
        pygame.draw.rect(self.screen, (60, 60, 80), game_rect, 2)

        if env is not None:
            try:
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, np.ndarray):
                        if frame.ndim == 2:
                            frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
                        frame = resize_frame(frame, gw, gh)
                        self.last_frame = frame
            except Exception:
                pass

        if self.last_frame is not None:
            sf = self.last_frame
            if sf.ndim == 2:
                sf = np.repeat(sf[:, :, np.newaxis], 3, axis=2)
            sf = resize_frame(sf, gw, gh)
            # make_surface expects (W, H, 3)
            surf_array = np.ascontiguousarray(np.transpose(sf, (1, 0, 2)))
            try:
                game_surf = pygame.surfarray.make_surface(surf_array)
                self.screen.blit(game_surf, (gx, gy))
            except Exception:
                pass

        # Score overlay on game
        score_text = self.large_font.render(f"{self.current_score_agent}  —  {self.current_score_opponent}", True, GOLD)
        score_rect = score_text.get_rect(center=(gx + gw // 2, gy + 22))
        self.screen.blit(score_text, score_rect)

        # Right panel: stats
        rx = gx + gw + 24
        ry = 24
        line_h = 24
        self.screen.blit(self.large_font.render(self.algorithm_name, True, ACCENT), (rx, ry))
        ry += line_h + 8
        self.screen.blit(self.font.render(f"Episode: {self.episode}", True, TEXT), (rx, ry))
        ry += line_h
        self.screen.blit(self.font.render(f"Step: {self.global_step}", True, TEXT), (rx, ry))
        ry += line_h + 12
        if self.recent_returns:
            avg_ret = np.mean(self.recent_returns)
            self.screen.blit(self.font.render(f"Avg return (last {len(self.recent_returns)}): {avg_ret:.1f}", True, GREEN), (rx, ry))
            ry += line_h
        ry += 8
        self.draw_graph(self.screen, rx, ry, 220, 80, self.recent_returns, GREEN, "Returns")
        ry += 100
        self.draw_graph(self.screen, rx, ry, 220, 60, self.recent_loss, ACCENT, "Policy loss")
        ry += 90
        if self.episode_rewards:
            self.screen.blit(self.font.render(f"Episode rewards (last 100): mean = {np.mean(self.episode_rewards):.1f}", True, TEXT), (rx, ry))

        pygame.display.flip()
        self.clock.tick(self.fps_target)

    def process_events(self) -> bool:
        """Handle quit and resize. Returns False if should exit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            if event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        return self.running

    def close(self):
        pygame.quit()


def run_visualization_loop(env, agent, vis: Visualization, max_steps: int = 5000, fps: int = 30):
    """Run one evaluation loop with visualization (no training)."""
    vis.fps_target = fps
    obs, _ = env.reset()
    vis.set_frame_from_obs(obs)
    total_reward = 0
    step = 0
    agent_score, opp_score = 0, 0
    while vis.running and step < max_steps:
        if not vis.process_events():
            break
        action, _, _ = agent.get_action(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if reward > 0:
            agent_score += 1
        elif reward < 0:
            opp_score += 1
        vis.set_scores(agent_score, opp_score)
        vis.set_episode_step(0, step)
        vis.set_frame_from_obs(next_obs)
        try:
            env.render()
        except Exception:
            pass
        vis.render(env)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            vis.add_return(total_reward)
            total_reward = 0
            agent_score, opp_score = 0, 0
        step += 1
    return step
