"""
Evaluate a trained PPO or REINFORCE agent with Pygame visualization.
Usage: python run_eval.py --checkpoint ppo_pong.pt --algorithm ppo
"""

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import argparse
import numpy as np
import torch

from env_utils import make_pong_env, get_obs_shape, get_num_actions
from agents import PPOAgent, ReinforceAgent
from models import ActorCriticCnn
from visualize import Visualization


def _obs_to_array(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32)


def load_agent(checkpoint_path: str, algorithm: str, in_channels: int, num_actions: int, device: str):
    model = ActorCriticCnn(in_channels, num_actions).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if algorithm == "ppo":
        agent = PPOAgent(in_channels=in_channels, num_actions=num_actions, device=device)
        agent.model = model
    else:
        agent = ReinforceAgent(in_channels=in_channels, num_actions=num_actions, device=device)
        agent.model = model
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="ppo_pong.pt", help="Path to .pt checkpoint")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "reinforce"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    env = make_pong_env(seed=args.seed, scale_obs=True)
    obs_shape = get_obs_shape(env)
    num_actions = get_num_actions(env)
    in_channels = obs_shape[0]

    agent = load_agent(args.checkpoint, args.algorithm, in_channels, num_actions, args.device)
    vis = Visualization(width=900, height=700, game_scale=4, title=f"Eval — {args.algorithm.upper()} Pong")
    vis.set_algorithm(args.algorithm.upper())

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        obs = _obs_to_array(obs)
        if obs.shape[-1] == 4:
            obs = np.transpose(obs, (2, 0, 1))
        total_reward = 0
        agent_score, opp_score = 0, 0
        while True:
            action, _, _ = agent.get_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = _obs_to_array(next_obs)
            if next_obs.shape[-1] == 4:
                next_obs = np.transpose(next_obs, (2, 0, 1))
            total_reward += reward
            if reward == 1:
                agent_score += 1
            elif reward == -1:
                opp_score += 1
            vis.set_frame_from_obs(np.transpose(obs, (1, 2, 0)))
            vis.set_scores(agent_score, opp_score)
            vis.set_episode_step(ep + 1, 0)
            vis.render(env)
            if not vis.process_events():
                env.close()
                vis.close()
                return
            obs = next_obs
            if terminated or truncated:
                break
        returns.append(total_reward)
        vis.add_return(total_reward)
        print(f"Episode {ep + 1}  return={total_reward}  (agent {agent_score} - opp {opp_score})")

    print(f"Mean return over {args.episodes} episodes: {np.mean(returns):.1f}")
    env.close()
    vis.close()


if __name__ == "__main__":
    main()
