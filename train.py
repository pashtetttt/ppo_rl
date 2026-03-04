"""
Training script: REINFORCE vs PPO on Atari Pong.
Run with visualization (Pygame) to watch learning in real time.
"""

# Suppress noisy warnings before other imports
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import argparse
import numpy as np
import torch
import time

from env_utils import make_pong_env, get_obs_shape, get_num_actions
from agents import ReinforceAgent, PPOAgent, compute_gae
from visualize import Visualization


def _obs_to_array(obs) -> np.ndarray:
    """Ensure observation is numpy (handle LazyFrames)."""
    return np.asarray(obs, dtype=np.float32)


def run_reinforce(
    env,
    agent: ReinforceAgent,
    vis: Visualization | None,
    total_episodes: int,
    seed: int,
    vis_update_interval: int = 1,
) -> list[float]:
    """Train REINFORCE: one update per episode (full trajectory)."""
    episode_returns = []
    agent_score, opp_score = 0, 0
    for episode in range(total_episodes):
        obs, _ = env.reset(seed=seed + episode)
        obs = _obs_to_array(obs)
        obs_list, actions_list, log_probs_list, rewards_list, values_list = [], [], [], [], []
        episode_reward = 0
        agent_score, opp_score = 0, 0
        while True:
            if vis and episode % vis_update_interval == 0:
                vis.set_frame_from_obs(obs)
                vis.set_scores(agent_score, opp_score)
                vis.set_episode_step(episode, len(rewards_list))
                vis.render(env)
                if not vis.process_events():
                    return episode_returns
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = _obs_to_array(next_obs)
            obs_list.append(obs)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            values_list.append(value)
            episode_reward += reward
            if reward == 1:
                agent_score += 1
            elif reward == -1:
                opp_score += 1
            obs = next_obs
            if terminated or truncated:
                break
        agent.update(obs_list, actions_list, log_probs_list, rewards_list, values_list)
        episode_returns.append(episode_reward)
        if vis:
            vis.add_return(episode_reward)
            vis.add_episode_reward(episode_reward)
            vis.add_loss(0.0)
        if (episode + 1) % 10 == 0:
            print(f"  REINFORCE episode {episode + 1}/{total_episodes}  return={episode_reward:.0f}  "
                  f"(agent {agent_score} - opp {opp_score})")
    return episode_returns


def run_ppo(
    env,
    agent: PPOAgent,
    vis: Visualization | None,
    total_timesteps: int,
    seed: int,
    n_steps: int = 2048,
    n_epochs: int = 4,
    batch_size: int = 64,
    vis_update_interval: int = 4,
) -> list[float]:
    """Train PPO: collect n_steps, compute GAE, then multiple epochs of minibatch updates."""
    obs_shape = get_obs_shape(env)
    num_actions = get_num_actions(env)
    C, H, W = obs_shape
    obs_buf = np.zeros((n_steps, C, H, W), dtype=np.float32)
    actions_buf = np.zeros((n_steps,), dtype=np.int64)
    log_probs_buf = np.zeros((n_steps,), dtype=np.float32)
    rewards_buf = np.zeros((n_steps,), dtype=np.float32)
    dones_buf = np.zeros((n_steps,), dtype=np.float32)
    values_buf = np.zeros((n_steps,), dtype=np.float32)

    obs, _ = env.reset(seed=seed)
    obs = _obs_to_array(obs)
    if obs.shape[-1] == 4:
        obs = np.transpose(obs, (2, 0, 1))
    elif obs.ndim == 3 and obs.shape[0] == 4:
        pass
    else:
        obs = np.transpose(obs, (2, 0, 1))
    episode_returns = []
    global_step = 0
    next_obs = obs
    next_done = 0.0
    agent_score, opp_score = 0, 0
    episode_reward = 0

    while global_step < total_timesteps:
        for step in range(n_steps):
            global_step += 1
            if vis and global_step % vis_update_interval == 0:
                frame_obs = np.transpose(obs, (1, 2, 0)) if obs.shape[0] == 4 else obs
                vis.set_frame_from_obs(frame_obs)
                vis.set_scores(agent_score, opp_score)
                vis.set_episode_step(len(episode_returns), global_step)
                vis.render(env)
                if not vis.process_events():
                    return episode_returns
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = _obs_to_array(next_obs)
            done = terminated or truncated
            if next_obs.shape[-1] == 4:
                next_obs_ch = np.transpose(next_obs, (2, 0, 1))
            else:
                next_obs_ch = next_obs
            obs_buf[step] = obs
            actions_buf[step] = action
            log_probs_buf[step] = log_prob.cpu().numpy()
            rewards_buf[step] = reward
            dones_buf[step] = float(done)
            values_buf[step] = value.cpu().numpy()
            episode_reward += reward
            if reward == 1:
                agent_score += 1
            elif reward == -1:
                opp_score += 1
            obs = next_obs_ch
            if done:
                episode_returns.append(episode_reward)
                if vis:
                    vis.add_return(episode_reward)
                    vis.add_episode_reward(episode_reward)
                if (len(episode_returns)) % 5 == 0 and len(episode_returns) > 0:
                    print(f"  PPO step {global_step}  episodes {len(episode_returns)}  "
                          f"last return={episode_reward:.0f}  (agent {agent_score} - opp {opp_score})")
                next_obs, _ = env.reset(seed=seed + len(episode_returns))
                next_obs = _obs_to_array(next_obs)
                if next_obs.shape[-1] == 4:
                    obs = np.transpose(next_obs, (2, 0, 1))
                else:
                    obs = next_obs
                next_done = 1.0
                episode_reward = 0
                agent_score, opp_score = 0, 0
            else:
                next_done = 0.0

        with torch.no_grad():
            next_value = agent.model.get_value(
                torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
            ).cpu().numpy().item()
        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf,
            next_value, next_done,
            gamma=agent.gamma, gae_lambda=agent.gae_lambda,
        )
        # obs_buf is (n_steps, C, H, W)
        info = agent.update(
            obs_buf, actions_buf, log_probs_buf,
            advantages, returns,
            batch_size=batch_size, n_epochs=n_epochs,
        )
        if vis:
            vis.add_loss(info["policy_loss"])
        if len(episode_returns) > 0 and len(episode_returns) % 5 == 0:
            print(f"  PPO update  policy_loss={info['policy_loss']:.4f}  value_loss={info['value_loss']:.4f}")

    return episode_returns


def main():
    parser = argparse.ArgumentParser(description="PPO vs REINFORCE on Atari Pong")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["reinforce", "ppo", "both"],
                        help="Which algorithm to run")
    parser.add_argument("--visualize", action="store_true", help="Show Pygame window during training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500, help="Episodes for REINFORCE")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Total steps for PPO (Pong often needs 1–2M+)")
    parser.add_argument("--ppo-steps", type=int, default=2048, help="Steps per PPO rollout")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization (for headless)")
    args = parser.parse_args()

    use_vis = args.visualize and not args.no_viz
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Creating Pong environment (84x84 grayscale, 4-frame stack)...")
    env = make_pong_env(seed=seed, scale_obs=True)
    obs_shape = get_obs_shape(env)
    num_actions = get_num_actions(env)
    in_channels = obs_shape[0]
    print(f"Observation shape: {obs_shape}, actions: {num_actions}")

    vis = None
    if use_vis:
        vis = Visualization(width=900, height=700, game_scale=3, title="PPO vs REINFORCE — Atari Pong")

    if args.algorithm in ("reinforce", "both"):
        print("\n--- Training REINFORCE ---")
        vis_reinforce = vis
        if args.algorithm == "both" and vis:
            vis_reinforce.set_algorithm("REINFORCE")
        agent_r = ReinforceAgent(
            in_channels=in_channels,
            num_actions=num_actions,
            lr=args.lr,
            device=args.device,
        )
        returns_r = run_reinforce(
            env, agent_r, vis_reinforce,
            total_episodes=args.episodes,
            seed=seed,
            vis_update_interval=2,
        )
        print(f"REINFORCE finished. Mean return (last 50): {np.mean(returns_r[-50:]) if len(returns_r) >= 50 else np.mean(returns_r):.1f}")
        if args.algorithm == "both":
            torch.save(agent_r.model.state_dict(), "reinforce_pong.pt")

    if args.algorithm in ("ppo", "both"):
        print("\n--- Training PPO ---")
        if vis:
            vis.set_algorithm("PPO")
        env_ppo = env if args.algorithm == "ppo" else make_pong_env(seed=seed + 1, scale_obs=True)
        agent_p = PPOAgent(
            in_channels=in_channels,
            num_actions=num_actions,
            lr=args.lr,
            device=args.device,
        )
        returns_p = run_ppo(
            env_ppo, agent_p, vis,
            total_timesteps=args.timesteps,
            seed=seed + 100,
            n_steps=args.ppo_steps,
            vis_update_interval=4,
        )
        print(f"PPO finished. Mean return (last 50): {np.mean(returns_p[-50:]) if len(returns_p) >= 50 else np.mean(returns_p):.1f}")
        torch.save(agent_p.model.state_dict(), "ppo_pong.pt")
        if args.algorithm == "both":
            env_ppo.close()

    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    if vis:
        vis.close()
    print("Done.")


if __name__ == "__main__":
    main()
