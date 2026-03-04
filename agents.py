"""
REINFORCE and PPO agents for Atari Pong.
Both use the same CNN actor-critic model; PPO adds clipping and GAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from models import ActorCriticCnn


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert env observation to channel-first float tensor (B, C, H, W)."""
    if isinstance(obs, np.ndarray):
        x = torch.from_numpy(obs).float().to(device)
    else:
        x = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 4:
        x = x.permute(0, 3, 1, 2)
    return x


# ---------------------------------------------------------------------------
# REINFORCE (vanilla policy gradient with baseline)
# ---------------------------------------------------------------------------


class ReinforceAgent:
    """
    REINFORCE with baseline (value function) for variance reduction.
    Learns from full episode returns; high variance with delayed reward.
    """

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.model = ActorCriticCnn(in_channels, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]:
        x = obs_to_tensor(obs, self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.model.get_action_and_value(x, action=None)
        a = action.item()
        if deterministic:
            probs = torch.softmax(self.model.actor(self.model.feature_extractor(x)), dim=-1)
            a = probs.argmax(dim=-1).item()
        return a, log_prob.squeeze(0), value.squeeze(0)

    def update(self, obs_list: list, actions: list, log_probs: list, rewards: list, values: list):
        """Update using full episode: R_t = sum gamma^k r_{t+k}, then policy gradient with baseline."""
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        R = 0
        for t in reversed(range(T)):
            R = rewards[t] + self.gamma * R
            returns[t] = R

        returns_t = torch.from_numpy(returns).to(self.device)
        obs_t = obs_to_tensor(np.stack(obs_list), self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(log_probs).detach()
        values_t = torch.stack(values).detach()

        advantages = returns_t - values_t
        _, new_log_prob, _, new_value = self.model.get_action_and_value(obs_t, action=actions_t)
        policy_loss = -(new_log_prob * advantages).mean()
        value_loss = F.mse_loss(new_value, returns_t)
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


# ---------------------------------------------------------------------------
# PPO (Proximal Policy Optimization with clipped objective and GAE)
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    next_done: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


class PPOAgent:
    """
    PPO with clipped surrogate objective and GAE.
    More stable and sample-efficient than REINFORCE under delayed reward.
    """

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.model = ActorCriticCnn(in_channels, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]:
        x = obs_to_tensor(obs, self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.model.get_action_and_value(x, action=None)
        a = action.item()
        if deterministic:
            probs = torch.softmax(self.model.actor(self.model.feature_extractor(x)), dim=-1)
            a = probs.argmax(dim=-1).item()
        return a, log_prob.squeeze(0), value.squeeze(0)

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs_old: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        batch_size: int = 64,
        n_epochs: int = 4,
    ) -> dict:
        """PPO update: multiple epochs over minibatches with clipped objective."""
        N = obs.shape[0]
        indices = np.arange(N)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        obs_t = obs_to_tensor(obs, self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        log_probs_old_t = torch.from_numpy(log_probs_old).float().to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)

        # Normalize advantages (per minibatch)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]
                mb_obs = obs_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_log_probs_old = log_probs_old_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                log_prob, entropy, value = self.model.get_log_prob_value(mb_obs, mb_actions)
                ratio = torch.exp(log_prob - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }
