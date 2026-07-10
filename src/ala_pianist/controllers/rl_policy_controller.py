"""SB3-backed symbolic RL controller for Pipeline 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from ala_pianist.controllers.action_library import KEYSET_MIDI
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.rl.keyset_env import KEYSET_KEYS, make_keyset_observation


class RLPolicyController:
    """Load an SB3 policy and emit native 22D ALAOneHandEnv actions."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        deterministic: bool = True,
        horizon_steps: int = 32,
        keyset_midi: tuple[int, ...] = KEYSET_MIDI,
        algo: str = "SAC",
    ):
        self.model_path = None if model_path is None else Path(model_path)
        self.deterministic = deterministic
        self.horizon_steps = int(horizon_steps)
        self.keyset_midi = tuple(int(pitch) for pitch in keyset_midi)
        self.keyset_keys = tuple(pitch - 21 for pitch in self.keyset_midi)
        self.model = None
        if self.model_path is not None:
            algos = {"SAC": SAC, "PPO": PPO, "TD3": TD3}
            self.model = algos[algo.upper()].load(self.model_path)

    def action(
        self,
        env: ALAOneHandEnv,
        *,
        target_midi: int,
        step_count: int,
    ) -> np.ndarray:
        target_key = int(target_midi) - 21
        if self.model is None:
            return np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)
        obs = make_keyset_observation(
            env,
            target_key=target_key,
            keyset_keys=self.keyset_keys,
            step_count=step_count,
            horizon_steps=self.horizon_steps,
        )
        normalized_action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return rescale_normalized_action(env, normalized_action)


def rescale_normalized_action(env: ALAOneHandEnv, normalized_action) -> np.ndarray:
    spec = env.action_spec()
    low = np.asarray(spec.minimum, dtype=np.float32)
    high = np.asarray(spec.maximum, dtype=np.float32)
    normalized_action = np.clip(np.asarray(normalized_action, dtype=np.float32), -1.0, 1.0)
    native = low + ((normalized_action + 1.0) / 2.0) * (high - low)
    return native.astype(spec.dtype)
