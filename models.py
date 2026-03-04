"""
CNN feature extractor and policy/value heads for Atari (84x84, 4-channel stack).
Used by both REINFORCE and PPO agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CnnFeatureExtractor(nn.Module):
    """
    CNN that maps (B, C, 84, 84) -> (B, 512).
    Same architecture as in DQN/PPO literature for Atari.
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            self._feature_dim = self.net(dummy).shape[1]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticCnn(nn.Module):
    """
    Shared CNN backbone + separate policy (actor) and value (critic) heads.
    Policy outputs categorical distribution over actions; value outputs scalar.
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.feature_extractor = CnnFeatureExtractor(in_channels)
        dim = self.feature_extractor.feature_dim

        self.actor = nn.Sequential(
            layer_init(nn.Linear(dim, 512), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.critic(feats).squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(x)
        logits = self.actor(feats)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(feats).squeeze(-1)
        return action, log_prob, entropy, value

    def get_log_prob_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(x)
        logits = self.actor(feats)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(feats).squeeze(-1)
        return log_prob, entropy, value
