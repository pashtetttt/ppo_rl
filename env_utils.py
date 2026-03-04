"""
Atari Pong environment setup with 84x84 grayscale and 4-frame stacking.
Episode runs until one player reaches 21 points (handled by default ALE truncation).
"""

import gymnasium as gym
import numpy as np

# Register ALE environments (required for ale-py)
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass


def make_pong_env(
    seed: int | None = None,
    frame_skip: int = 4,
    frame_stack: int = 4,
    screen_size: int = 84,
    noop_max: int = 30,
    scale_obs: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Create Pong environment with DQN-style preprocessing:
    - Grayscale 84x84 (or screen_size)
    - Frame skip (default 4)
    - Frame stack (default 4)
    - Optional scaling to [0, 1]
    - render_mode: "human" (real Atari window) or "rgb_array" for game frames; None for training.
    """
    kwargs = dict(
        obs_type="grayscale",
        frameskip=frame_skip,
        repeat_action_probability=0.0,  # Deterministic for reproducibility
    )
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make("ALE/Pong-v5", **kwargs)

    # Atari preprocessing: noop reset, max pool over 2 frames, resize, grayscale
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=1,  # we already use frameskip in make
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=scale_obs,
    )

    # Stack last N frames so observation shape is (N, 84, 84)
    env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)

    return env


def get_obs_shape(env: gym.Env) -> tuple:
    """Return (C, H, W) for PyTorch (channel-first)."""
    obs_space = env.observation_space
    # FrameStack gives (4, 84, 84) or (84, 84, 4) depending on gym version
    s = obs_space.shape
    if len(s) == 3:
        if s[-1] < s[0]:  # (H, W, C)
            return (s[-1], s[0], s[1])
        return (s[0], s[1], s[2])  # (C, H, W)
    return s


def get_num_actions(env: gym.Env) -> int:
    return int(env.action_space.n)
