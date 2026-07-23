"""Minimal PyTorch DroQ-style trainer components.

This is intentionally project-specific: it targets normalized continuous
Gymnasium actions in ``[-1, 1]`` and the existing one-hand RoboPianist wrapper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class DroQConfig:
    observation_dim: int
    action_dim: int
    hidden_dim: int = 256
    critic_ensemble_size: int = 2
    critic_dropout: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None
    batch_size: int = 256
    utd_ratio: int = 4
    buffer_size: int = 1_000_000
    device: str = "cpu"


class ReplayBuffer:
    """Simple fixed-size replay buffer for flat observations and actions."""

    def __init__(self, observation_dim: int, action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, observation, action, reward: float, next_observation, done: bool) -> None:
        self.observations[self.position] = np.asarray(observation, dtype=np.float32)
        self.actions[self.position] = np.asarray(action, dtype=np.float32)
        self.rewards[self.position] = float(reward)
        self.next_observations[self.position] = np.asarray(next_observation, dtype=np.float32)
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(self.next_observations[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "observations": self.observations[: self.size],
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "next_observations": self.next_observations[: self.size],
            "dones": self.dones[: self.size],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(
                f"Replay buffer state size {size} exceeds buffer capacity {self.capacity}."
            )
        self.position = int(state["position"])
        self.size = size
        self.observations[:size] = state["observations"]
        self.actions[:size] = state["actions"]
        self.rewards[:size] = state["rewards"]
        self.next_observations[:size] = state["next_observations"]
        self.dones[:size] = state["dones"]


class SquashedGaussianActor(nn.Module):
    """Gaussian policy with tanh squashing for normalized actions."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(observation)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self(observation)
        return torch.tanh(mean)


class DropoutCritic(nn.Module):
    """Q-network with layer normalization and dropout, as used by DroQ-style critics."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observation, action], dim=-1)
        x = self.dropout(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        return self.q(x)


class CriticEnsemble(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 2,
        dropout: float = 0.01,
    ):
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("DroQ critic ensemble should contain at least two critics.")
        self.critics = nn.ModuleList(
            [
                DropoutCritic(observation_dim, action_dim, hidden_dim, dropout)
                for _ in range(int(ensemble_size))
            ]
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(observation, action) for critic in self.critics], dim=0)


class DroQAgent:
    """Small SAC/DroQ-style learner with dropout critics and high update ratio."""

    def __init__(self, config: DroQConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.actor = SquashedGaussianActor(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
        ).to(self.device)
        self.critics = CriticEnsemble(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
            config.critic_ensemble_size,
            config.critic_dropout,
        ).to(self.device)
        self.target_critics = CriticEnsemble(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
            config.critic_ensemble_size,
            config.critic_dropout,
        ).to(self.device)
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        target_entropy = config.target_entropy
        self.target_entropy = -float(config.action_dim) if target_entropy is None else float(target_entropy)
        self.log_alpha = torch.tensor(
            np.log(config.alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=config.auto_alpha,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=config.alpha_lr) if config.auto_alpha else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, observation, *, deterministic: bool = False) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(observation_tensor)
            else:
                action, _ = self.actor.sample(observation_tensor)
        self.actor.train()
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_observations)
            target_q_values = self.target_critics(next_observations, next_actions)
            target_q = target_q_values.min(dim=0).values
            backup = rewards + self.config.gamma * (1.0 - dones) * (
                target_q - self.alpha.detach() * next_log_probs
            )

        q_values = self.critics(observations, actions)
        critic_loss = F.mse_loss(q_values, backup.unsqueeze(0).expand_as(q_values))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(observations)
        new_q_values = self.critics(observations, new_actions).min(dim=0).values
        actor_loss = (self.alpha.detach() * log_probs - new_q_values).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss_value = 0.0
        if self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach().cpu())

        self._soft_update_targets()
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().cpu()),
            "mean_q": float(q_values.detach().mean().cpu()),
        }

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.critics.parameters(),
                self.target_critics.parameters(),
            ):
                target_parameter.data.mul_(1.0 - tau)
                target_parameter.data.add_(tau * parameter.data)

    def save(self, path: str | Path, *, replay_buffer: ReplayBuffer | None = None, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "extra": extra or {},
            "rng_state": rng_state_dict(),
        }
        if replay_buffer is not None:
            payload["replay_buffer"] = replay_buffer.state_dict()
        torch.save(payload, path)
        return path

    def save_lightweight(self, path: str | Path, *, extra: dict | None = None) -> Path:
        """Save actor-only policy state for deterministic checkpoint evaluation."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_class": "lightweight_policy",
                "algorithm": "droq",
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "extra": extra or {},
                "rng_state": rng_state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        reset_optimizers: bool = False,
    ) -> "DroQAgent":
        payload = load_droq_checkpoint(path, device=device)
        config_payload = dict(payload["config"])
        if device is not None:
            config_payload["device"] = device
        agent = cls(DroQConfig(**config_payload))
        agent.actor.load_state_dict(payload["actor"])
        if "critics" in payload:
            agent.critics.load_state_dict(payload["critics"])
        if "target_critics" in payload:
            agent.target_critics.load_state_dict(payload["target_critics"])
        if not reset_optimizers and "actor_optimizer" in payload:
            agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if not reset_optimizers and "critic_optimizer" in payload:
            agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if "log_alpha" in payload:
            agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
        if (
            not reset_optimizers
            and agent.alpha_optimizer is not None
            and payload.get("alpha_optimizer") is not None
        ):
            agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        return agent


class DroQPolicy:
    """Small adapter exposing SB3-like ``predict`` for evaluation scripts."""

    def __init__(self, agent: DroQAgent):
        self.agent = agent

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "DroQPolicy":
        return cls(DroQAgent.load(path, device=device))

    def predict(self, observation, deterministic: bool = True):
        return self.agent.act(observation, deterministic=deterministic), None


def load_droq_checkpoint(path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    """Load a local DroQ checkpoint payload.

    DroQ checkpoints may include NumPy replay-buffer arrays, so current PyTorch
    versions need ``weights_only=False`` for trusted local files.
    """

    try:
        return torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=device or "cpu")


def replay_buffer_from_checkpoint(
    payload: dict[str, Any],
    *,
    observation_dim: int,
    action_dim: int,
    fallback_capacity: int,
) -> ReplayBuffer:
    state = payload.get("replay_buffer")
    capacity = int(state.get("capacity", fallback_capacity)) if state is not None else int(fallback_capacity)
    replay_buffer = ReplayBuffer(observation_dim, action_dim, capacity)
    if state is not None:
        replay_buffer.load_state_dict(state)
    return replay_buffer


def rng_state_dict() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
