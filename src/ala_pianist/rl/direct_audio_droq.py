"""Minimal DroQ-style direct raw-audio trainer components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ala_pianist.audio.reference_bank import AudioReferenceBank
from ala_pianist.rl.droq import LOG_STD_MAX, LOG_STD_MIN


@dataclass(frozen=True)
class DirectDroQConfig:
    audio_window_size: int
    physical_dim: int
    action_dim: int = 22
    audio_latent_dim: int = 128
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    batch_size: int = 64
    utd_ratio: int = 2
    buffer_size: int = 100_000
    device: str = "cpu"


class RawAudioEncoder(nn.Module):
    """Trainable order-sensitive waveform encoder."""

    def __init__(self, audio_latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=audio_latent_dim, batch_first=True)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"Expected audio tensor [B, T], got {tuple(audio.shape)}.")
        x = self.conv(audio.unsqueeze(1))
        x = x.transpose(1, 2)
        _sequence, hidden = self.gru(x)
        return hidden[-1]


class DirectActor(nn.Module):
    def __init__(self, config: DirectDroQConfig):
        super().__init__()
        self.audio_encoder = RawAudioEncoder(config.audio_latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(config.audio_latent_dim + config.physical_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, audio: torch.Tensor, physical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.audio_encoder(audio)
        features = self.fusion(torch.cat([latent, physical], dim=-1))
        return self.mean(features), torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, audio: torch.Tensor, physical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(audio, physical)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, audio: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
        mean, _ = self(audio, physical)
        return torch.tanh(mean)


class DirectCritic(nn.Module):
    def __init__(self, config: DirectDroQConfig):
        super().__init__()
        self.audio_encoder = RawAudioEncoder(config.audio_latent_dim)
        self.q = nn.Sequential(
            nn.Linear(config.audio_latent_dim + config.physical_dim + config.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, audio: torch.Tensor, physical: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = self.audio_encoder(audio)
        return self.q(torch.cat([latent, physical, action], dim=-1))


class DirectCriticPair(nn.Module):
    def __init__(self, config: DirectDroQConfig):
        super().__init__()
        self.q1 = DirectCritic(config)
        self.q2 = DirectCritic(config)

    def forward(self, audio: torch.Tensor, physical: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.q1(audio, physical, action), self.q2(audio, physical, action)], dim=0)


class IndexedDirectReplayBuffer:
    """Replay buffer storing audio references instead of waveform windows."""

    def __init__(self, *, physical_dim: int, action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.physical = np.zeros((self.capacity, physical_dim), dtype=np.float32)
        self.next_physical = np.zeros((self.capacity, physical_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.clip_ids = np.zeros((self.capacity,), dtype=np.int32)
        self.next_clip_ids = np.zeros((self.capacity,), dtype=np.int32)
        self.audio_sample_indices = np.zeros((self.capacity,), dtype=np.int64)
        self.next_audio_sample_indices = np.zeros((self.capacity,), dtype=np.int64)
        self.position = 0
        self.size = 0

    def add(self, observation: dict, metadata: dict, action, reward: float, next_observation: dict, next_metadata: dict, done: bool) -> None:
        idx = self.position
        self.physical[idx] = np.asarray(observation["physical"], dtype=np.float32)
        self.next_physical[idx] = np.asarray(next_observation["physical"], dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)
        self.clip_ids[idx] = int(metadata["clip_id"])
        self.next_clip_ids[idx] = int(next_metadata["clip_id"])
        self.audio_sample_indices[idx] = int(metadata["audio_sample_index"])
        self.next_audio_sample_indices[idx] = int(next_metadata["audio_sample_index"])
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, bank: AudioReferenceBank, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        audio = np.stack(
            [
                bank.context_window(clip_id=int(c), center_sample=int(s))
                for c, s in zip(self.clip_ids[indices], self.audio_sample_indices[indices])
            ]
        )
        next_audio = np.stack(
            [
                bank.context_window(clip_id=int(c), center_sample=int(s))
                for c, s in zip(self.next_clip_ids[indices], self.next_audio_sample_indices[indices])
            ]
        )
        return {
            "audio": torch.as_tensor(audio, device=device),
            "physical": torch.as_tensor(self.physical[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_audio": torch.as_tensor(next_audio, device=device),
            "next_physical": torch.as_tensor(self.next_physical[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }

    def estimated_bytes(self, transitions: int | None = None) -> int:
        count = self.capacity if transitions is None else int(transitions)
        per_transition = (
            self.physical.shape[1] * 4
            + self.next_physical.shape[1] * 4
            + self.actions.shape[1] * 4
            + 4
            + 4
            + 4
            + 4
            + 8
            + 8
        )
        return int(count * per_transition)

    def state_dict(self) -> dict[str, Any]:
        size = int(self.size)
        return {
            "schema": "indexed_audio_references",
            "capacity": int(self.capacity),
            "position": int(self.position),
            "size": size,
            "physical": self.physical[:size],
            "next_physical": self.next_physical[:size],
            "actions": self.actions[:size],
            "rewards": self.rewards[:size],
            "dones": self.dones[:size],
            "clip_ids": self.clip_ids[:size],
            "next_clip_ids": self.next_clip_ids[:size],
            "audio_sample_indices": self.audio_sample_indices[:size],
            "next_audio_sample_indices": self.next_audio_sample_indices[:size],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "indexed_audio_references":
            raise ValueError(f"Unsupported direct replay schema {state.get('schema')!r}.")
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"Replay state size {size} exceeds capacity {self.capacity}.")
        self.position = int(state["position"])
        self.size = size
        self.physical[:size] = state["physical"]
        self.next_physical[:size] = state["next_physical"]
        self.actions[:size] = state["actions"]
        self.rewards[:size] = state["rewards"]
        self.dones[:size] = state["dones"]
        self.clip_ids[:size] = state["clip_ids"]
        self.next_clip_ids[:size] = state["next_clip_ids"]
        self.audio_sample_indices[:size] = state["audio_sample_indices"]
        self.next_audio_sample_indices[:size] = state["next_audio_sample_indices"]


class DirectDroQAgent:
    def __init__(self, config: DirectDroQConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.actor = DirectActor(config).to(self.device)
        self.critics = DirectCriticPair(config).to(self.device)
        self.target_critics = DirectCriticPair(config).to(self.device)
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        self.log_alpha = torch.tensor(np.log(config.alpha), dtype=torch.float32, device=self.device, requires_grad=config.auto_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr) if config.auto_alpha else None
        self.target_entropy = -float(config.action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _obs_tensors(self, observation: dict) -> tuple[torch.Tensor, torch.Tensor]:
        audio = torch.as_tensor(np.asarray(observation["audio"], dtype=np.float32), device=self.device).unsqueeze(0)
        physical = torch.as_tensor(np.asarray(observation["physical"], dtype=np.float32), device=self.device).unsqueeze(0)
        return audio, physical

    def act(self, observation: dict, *, deterministic: bool = False) -> np.ndarray:
        audio, physical = self._obs_tensors(observation)
        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(audio, physical)
            else:
                action, _ = self.actor.sample(audio, physical)
        self.actor.train()
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        audio = batch["audio"]
        physical = batch["physical"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_audio = batch["next_audio"]
        next_physical = batch["next_physical"]
        dones = batch["dones"]
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_audio, next_physical)
            target_q = self.target_critics(next_audio, next_physical, next_actions).min(dim=0).values
            backup = rewards + self.config.gamma * (1.0 - dones) * (target_q - self.alpha.detach() * next_log_probs)
        q_values = self.critics(audio, physical, actions)
        critic_loss = F.mse_loss(q_values, backup.unsqueeze(0).expand_as(q_values))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(audio, physical)
        actor_loss = (self.alpha.detach() * log_probs - self.critics(audio, physical, new_actions).min(dim=0).values).mean()
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
        with torch.no_grad():
            for parameter, target_parameter in zip(self.critics.parameters(), self.target_critics.parameters()):
                target_parameter.data.mul_(1.0 - self.config.tau)
                target_parameter.data.add_(self.config.tau * parameter.data)

    def save(self, path: str | Path, *, replay_buffer: IndexedDirectReplayBuffer | None = None, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "direct_audio_droq",
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "replay_buffer_schema": "indexed_audio_references",
            "replay_buffer": None if replay_buffer is None else replay_buffer.state_dict(),
            "replay_buffer_estimated_bytes": None
            if replay_buffer is None
            else replay_buffer.estimated_bytes(replay_buffer.size),
            "rng_state": direct_rng_state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        return path

    def save_lightweight(self, path: str | Path, *, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "direct_audio_droq",
                "checkpoint_class": "lightweight_policy",
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "extra": extra or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "DirectDroQAgent":
        try:
            payload = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location=device or "cpu")
        config_payload = dict(payload["config"])
        if device is not None:
            config_payload["device"] = device
        agent = cls(DirectDroQConfig(**config_payload))
        agent.actor.load_state_dict(payload["actor"])
        if "critics" in payload:
            agent.critics.load_state_dict(payload["critics"])
        if "target_critics" in payload:
            agent.target_critics.load_state_dict(payload["target_critics"])
        if "actor_optimizer" in payload:
            agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if "log_alpha" in payload:
            agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
        if agent.alpha_optimizer is not None and payload.get("alpha_optimizer") is not None:
            agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        return agent


def set_direct_droq_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def direct_rng_state_dict() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_direct_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    elif "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_direct_droq_checkpoint(path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=device or "cpu")


def indexed_replay_from_checkpoint(
    payload: dict[str, Any],
    *,
    physical_dim: int,
    action_dim: int,
    fallback_capacity: int,
) -> IndexedDirectReplayBuffer:
    state = payload.get("replay_buffer")
    capacity = int(state.get("capacity", fallback_capacity)) if state is not None else int(fallback_capacity)
    replay = IndexedDirectReplayBuffer(
        physical_dim=physical_dim,
        action_dim=action_dim,
        capacity=capacity,
    )
    if state is not None:
        replay.load_state_dict(state)
    return replay
