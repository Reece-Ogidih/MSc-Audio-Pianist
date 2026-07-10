"""Hybrid Pipeline 1 controller: residual D#5 policy plus action-library fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.rl.residual_env import ResidualSingleNoteEnv, hand_joint_summary


class HybridPipeline1Controller:
    """Use residual RL for D#5 and fallback native 22D actions for other notes."""

    def __init__(
        self,
        residual_model_path: str | Path | None,
        *,
        residual_midi: int = 75,
        residual_scale: float = 0.1,
        deterministic: bool = True,
    ):
        self.residual_model_path = None if residual_model_path is None else Path(residual_model_path)
        self.residual_midi = int(residual_midi)
        self.residual_scale = float(residual_scale)
        self.deterministic = deterministic
        self.model = None
        if self.residual_model_path is not None:
            self.model = SAC.load(self.residual_model_path)

    def action(
        self,
        env: ALAOneHandEnv,
        *,
        target_midi: int,
        fallback_action,
        step_count: int,
    ) -> np.ndarray:
        if self.model is None or int(target_midi) != self.residual_midi:
            ramp = min(1.0, (step_count + 1) / 6.0)
            return np.asarray(fallback_action, dtype=env.action_spec().dtype) * ramp

        obs = _residual_observation(env, step_count=step_count)
        residual, _ = self.model.predict(obs, deterministic=self.deterministic)
        residual_env = ResidualSingleNoteEnv(
            horizon_steps=24,
            residual_scale=self.residual_scale,
            reward_mode="cleanliness",
        )
        normalized = np.clip(
            residual_env.base_normalized_action + self.residual_scale * np.asarray(residual, dtype=np.float32),
            -1.0,
            1.0,
        )
        native = residual_env.rescale_action(normalized)
        ramp = min(1.0, (step_count + 1) / 6.0)
        return native.astype(env.action_spec().dtype) * ramp


def _residual_observation(env: ALAOneHandEnv, *, step_count: int) -> np.ndarray:
    target_key = 54
    wrong_key = 50
    target_state = env.target_key_state(target_key) or 0.0
    key50_state = env.target_key_state(wrong_key) or 0.0
    max_unintended = env.max_unintended_key_state(target_key)
    pressed_count = len(env.current_pressed_keys())
    nearest = env.nearest_fingertip_to_key(target_key)
    distance = float(nearest["distance"]) if nearest is not None else 1.0
    hand_mean, hand_std = hand_joint_summary(env)
    return np.asarray(
        [
            np.clip(target_state, 0.0, 1.0),
            np.clip(key50_state, 0.0, 1.0),
            np.clip(max_unintended, 0.0, 1.0),
            np.clip(pressed_count / 88.0, 0.0, 1.0),
            min(1.0, step_count / 24.0),
            min(1.0, distance / 0.20),
            np.clip((hand_mean + hand_std) / 2.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
