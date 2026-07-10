"""Residual single-note RL environment around a dirty D#5 action prior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ala_pianist.baselines.calibration import TARGET_KEY, TARGET_MIDI, TARGET_NOTE
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.learning.random_search import generate_random_candidates, write_single_note_learning_midi


WRONG_KEY = 50
REWARD_MODES = ("target_travel_first", "cleanliness")

# Regenerated from the known useful random-search path when possible. This
# sentinel documents that the prior is the dirty D#5 action, not a clean teacher.
D_SHARP_5_DIRTY_BASE_ACTION = "random_search_seed_7_candidate_2"


class ResidualSingleNoteEnv(gym.Env):
    """Train residual corrections around the known dirty D#5 action."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        midi_path: str | Path = "/home/reece_dev/msc-audio-pianist/tmp/residual_dsharp5.mid",
        horizon_steps: int = 24,
        residual_scale: float = 0.1,
        reward_mode: str = "target_travel_first",
        base_action: np.ndarray | None = None,
    ):
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}.")
        self.midi_path = Path(midi_path)
        self.horizon_steps = int(horizon_steps)
        self.residual_scale = float(residual_scale)
        self.reward_mode = reward_mode
        write_single_note_learning_midi(self.midi_path)

        self._env = ALAOneHandEnv(self.midi_path)
        spec = self._env.action_spec()
        self._native_action_low = np.asarray(spec.minimum, dtype=np.float32)
        self._native_action_high = np.asarray(spec.maximum, dtype=np.float32)
        self.base_action = (
            np.asarray(base_action, dtype=np.float32)
            if base_action is not None
            else get_dirty_dsharp5_base_action(self.midi_path)
        )
        self.base_normalized_action = self.normalize_native_action(self.base_action)
        self.action_space = spaces.Box(
            low=-np.ones(spec.shape, dtype=np.float32),
            high=np.ones(spec.shape, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )
        self._step_count = 0
        self._previous_residual = np.zeros(spec.shape, dtype=np.float32)

    @property
    def wrapped_env(self) -> ALAOneHandEnv:
        return self._env

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self._env = ALAOneHandEnv(self.midi_path)
        self._env.reset()
        self._step_count = 0
        self._previous_residual = np.zeros(self.action_space.shape, dtype=np.float32)
        return self._observation(), self._info(0.0, 0.0, 0.0)

    def step(self, action):
        residual = np.asarray(action, dtype=np.float32)
        residual = np.clip(residual, -1.0, 1.0)
        normalized_action = np.clip(
            self.base_normalized_action + self.residual_scale * residual,
            -1.0,
            1.0,
        )
        native_action = self.rescale_action(normalized_action)
        ramp = min(1.0, (self._step_count + 1) / 6.0)
        timestep = self._env.step(native_action * ramp)
        self._step_count += 1

        action_penalty = 0.001 * float(np.mean(np.square(residual)))
        smoothness_penalty = 0.001 * float(np.mean(np.square(residual - self._previous_residual)))
        self._previous_residual = residual.copy()
        reward = self._reward(action_penalty, smoothness_penalty)
        terminated = bool(timestep.last())
        truncated = self._step_count >= self.horizon_steps and not terminated
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(reward, action_penalty, smoothness_penalty),
        )

    def rescale_action(self, normalized_action) -> np.ndarray:
        normalized_action = np.clip(np.asarray(normalized_action, dtype=np.float32), -1.0, 1.0)
        fraction = (normalized_action + 1.0) / 2.0
        native = self._native_action_low + fraction * (self._native_action_high - self._native_action_low)
        return native.astype(self._env.action_spec().dtype)

    def normalize_native_action(self, native_action) -> np.ndarray:
        native_action = np.asarray(native_action, dtype=np.float32)
        fraction = (native_action - self._native_action_low) / np.maximum(
            1e-6,
            self._native_action_high - self._native_action_low,
        )
        return np.clip(2.0 * fraction - 1.0, -1.0, 1.0).astype(np.float32)

    def _observation(self) -> np.ndarray:
        target_state = self._env.target_key_state(TARGET_KEY) or 0.0
        key50_state = self._env.target_key_state(WRONG_KEY) or 0.0
        max_unintended = self._env.max_unintended_key_state(TARGET_KEY)
        pressed_count = len(self._env.current_pressed_keys())
        nearest = self._env.nearest_fingertip_to_key(TARGET_KEY)
        distance = float(nearest["distance"]) if nearest is not None else 1.0
        hand_mean, hand_std = _hand_joint_summary(self._env)
        return np.asarray(
            [
                np.clip(target_state, 0.0, 1.0),
                np.clip(key50_state, 0.0, 1.0),
                np.clip(max_unintended, 0.0, 1.0),
                np.clip(pressed_count / 88.0, 0.0, 1.0),
                min(1.0, self._step_count / max(1, self.horizon_steps)),
                min(1.0, distance / 0.20),
                np.clip((hand_mean + hand_std) / 2.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _reward(self, action_penalty: float, smoothness_penalty: float) -> float:
        target_state = self._env.target_key_state(TARGET_KEY) or 0.0
        key50_state = self._env.target_key_state(WRONG_KEY) or 0.0
        max_unintended = self._env.max_unintended_key_state(TARGET_KEY)
        pressed = self._env.current_pressed_keys()
        wrong_count = len([key for key in pressed if key != TARGET_KEY])
        target_active = TARGET_KEY in pressed
        if self.reward_mode == "target_travel_first":
            return (
                10.0 * target_state
                + (4.0 if target_active else 0.0)
                - 0.5 * max_unintended
                - action_penalty
                - smoothness_penalty
            )
        return (
            10.0 * target_state
            + (5.0 if target_active else 0.0)
            + (2.0 if target_state >= 0.25 and max_unintended <= target_state + 0.02 else 0.0)
            - 8.0 * key50_state
            - 5.0 * max_unintended
            - 3.0 * wrong_count
            - action_penalty
            - smoothness_penalty
        )

    def _info(self, reward: float, action_penalty: float, smoothness_penalty: float) -> dict[str, Any]:
        target_state = self._env.target_key_state(TARGET_KEY) or 0.0
        key50_state = self._env.target_key_state(WRONG_KEY) or 0.0
        max_unintended = self._env.max_unintended_key_state(TARGET_KEY)
        pressed = tuple(self._env.current_pressed_keys())
        native_reward = self._env.current_reward()
        return {
            "target_midi": TARGET_MIDI,
            "target_note": TARGET_NOTE,
            "target_key": TARGET_KEY,
            "key50_state": float(key50_state),
            "target_key_state": float(target_state),
            "max_unintended_key_state": float(max_unintended),
            "pressed_keys": pressed,
            "wrong_pressed_key_count": len([key for key in pressed if key != TARGET_KEY]),
            "clean_target_press": pressed == (TARGET_KEY,),
            "near_clean_partial_press": bool(target_state >= 0.25 and max_unintended <= target_state + 0.02),
            "dirty_press": bool(TARGET_KEY in pressed and pressed != (TARGET_KEY,)),
            "native_reward": None if native_reward is None else float(native_reward),
            "debug_reward": float(reward),
            "action_penalty": float(action_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "reward_mode": self.reward_mode,
            "residual_scale": self.residual_scale,
            "base_action_source": D_SHARP_5_DIRTY_BASE_ACTION,
            "sustain_state": float(self._env.task.piano.sustain_state[0]),
        }


def get_dirty_dsharp5_base_action(midi_path: str | Path | None = None) -> np.ndarray:
    """Regenerate the known dirty D#5 base action from random-search seed/candidate."""

    path = Path(midi_path or "/home/reece_dev/msc-audio-pianist/tmp/residual_dsharp5_base.mid")
    write_single_note_learning_midi(path)
    env = ALAOneHandEnv(path)
    env.reset()
    candidates = generate_random_candidates(env, count=30, seed=7)
    return np.asarray(candidates[2].action, dtype=np.float32)


def evaluate_residual_action(
    *,
    residual_action: np.ndarray | None = None,
    residual_scale: float = 0.1,
    reward_mode: str = "cleanliness",
    horizon_steps: int = 24,
) -> dict:
    env = ResidualSingleNoteEnv(
        horizon_steps=horizon_steps,
        residual_scale=residual_scale,
        reward_mode=reward_mode,
    )
    env.reset()
    residual = np.zeros(env.action_space.shape, dtype=np.float32) if residual_action is None else residual_action
    metrics = []
    total_reward = 0.0
    native_reward = 0.0
    max_target = 0.0
    max_key50 = 0.0
    max_unintended = 0.0
    pressed = set()
    for _ in range(horizon_steps):
        _, reward, terminated, truncated, info = env.step(residual)
        metrics.append(info)
        total_reward += float(reward)
        native_reward += 0.0 if info["native_reward"] is None else float(info["native_reward"])
        max_target = max(max_target, float(info["target_key_state"]))
        max_key50 = max(max_key50, float(info["key50_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed.update(info["pressed_keys"])
        if terminated or truncated:
            break
    return {
        "target_midi": TARGET_MIDI,
        "target_key": TARGET_KEY,
        "max_target_key_state": max_target,
        "max_key50_state": max_key50,
        "max_unintended_key_state": max_unintended,
        "pressed_keys": sorted(pressed),
        "clean_press": pressed == {TARGET_KEY},
        "near_clean_partial": bool(max_target >= 0.25 and max_unintended <= max_target + 0.02),
        "dirty_press": bool(TARGET_KEY in pressed and pressed != {TARGET_KEY}),
        "missed": TARGET_KEY not in pressed and max_target < 0.25,
        "shaped_return": total_reward,
        "native_reward_sum": native_reward,
    }


def _hand_joint_summary(env: ALAOneHandEnv) -> tuple[float, float]:
    physics = env.env.physics
    joints = env.task._hand.joints
    qpos = np.asarray(physics.bind(joints).qpos, dtype=float)
    ranges = np.asarray(physics.bind(joints).range, dtype=float)
    denom = np.maximum(1e-6, ranges[:, 1] - ranges[:, 0])
    normalized = np.clip((qpos - ranges[:, 0]) / denom, 0.0, 1.0)
    return float(np.mean(normalized)), float(np.std(normalized))
