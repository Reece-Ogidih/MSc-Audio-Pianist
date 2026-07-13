"""Hybrid Pipeline 1 controller: residual note policies plus action fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.rl.residual_env import (
    ResidualSingleNoteEnv,
    action_ramp,
    infer_wrong_key,
    make_residual_observation,
)


@dataclass
class ResidualPolicy:
    target_midi: int
    model_path: Path
    model: SAC
    helper_env: ResidualSingleNoteEnv


class HybridPipeline1Controller:
    """Use residual RL for configured notes and native 22D fallback otherwise."""

    def __init__(
        self,
        residual_model_path: str | Path | None = None,
        *,
        residual_midi: int = 75,
        residual_model_paths: dict[int, str | Path] | None = None,
        residual_scale: float = 0.1,
        deterministic: bool = True,
    ):
        self.residual_scale = float(residual_scale)
        self.deterministic = deterministic
        paths: dict[int, str | Path] = {}
        if residual_model_path is not None:
            paths[int(residual_midi)] = residual_model_path
        if residual_model_paths:
            paths.update({int(midi): path for midi, path in residual_model_paths.items()})
        self.policies: dict[int, ResidualPolicy] = {}
        for midi, path in paths.items():
            model_path = Path(path)
            model = SAC.load(model_path)
            helper_env = ResidualSingleNoteEnv(
                target_midi=midi,
                wrong_key=infer_wrong_key(midi),
                horizon_steps=24,
                residual_scale=self.residual_scale,
                reward_mode="cleanliness",
            )
            self.policies[midi] = ResidualPolicy(
                target_midi=midi,
                model_path=model_path,
                model=model,
                helper_env=helper_env,
            )

    def action(
        self,
        env: ALAOneHandEnv,
        *,
        target_midi: int,
        fallback_action,
        step_count: int,
    ) -> np.ndarray:
        target_midi = int(target_midi)
        if target_midi not in self.policies:
            return np.asarray(fallback_action, dtype=env.action_spec().dtype) * action_ramp(step_count)

        policy = self.policies[target_midi]
        target_key = target_midi - 21
        wrong_key = infer_wrong_key(target_midi)
        obs = make_residual_observation(
            env,
            target_key=target_key,
            wrong_key=wrong_key,
            step_count=step_count,
            horizon_steps=24,
        )
        residual, _ = policy.model.predict(obs, deterministic=self.deterministic)
        normalized = np.clip(
            policy.helper_env.base_normalized_action
            + self.residual_scale * np.asarray(residual, dtype=np.float32),
            -1.0,
            1.0,
        )
        native = policy.helper_env.rescale_action(normalized)
        return native.astype(env.action_spec().dtype) * action_ramp(step_count)
