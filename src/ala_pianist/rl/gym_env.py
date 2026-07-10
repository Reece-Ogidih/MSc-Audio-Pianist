"""Minimal Gymnasium adapter for single-note RoboPianist debug training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ala_pianist.baselines.calibration import TARGET_KEY, TARGET_MIDI, TARGET_NOTE
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


def write_single_note_rl_midi(path: str | Path) -> Path:
    """Write the generated D#5 clip used for the first symbolic RL smoke task."""

    return write_monophonic_midi(
        [NoteEvent(TARGET_MIDI, 0.0, 1.0, 90)],
        path,
        title="single note PPO D sharp 5",
    )


class SingleNotePianoGymEnv(gym.Env):
    """Gymnasium wrapper around the public 22D one-hand RoboPianist wrapper."""

    metadata = {"render_modes": []}

    def __init__(self, midi_path: str | Path, horizon_steps: int = 32):
        super().__init__()
        self.midi_path = Path(midi_path)
        self.horizon_steps = int(horizon_steps)
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")

        self._env = ALAOneHandEnv(self.midi_path)
        spec = self._env.action_spec()
        self._native_action_low = np.asarray(spec.minimum, dtype=np.float32)
        self._native_action_high = np.asarray(spec.maximum, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(spec.shape, dtype=np.float32),
            high=np.ones(spec.shape, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self._step_count = 0
        self._last_info: dict[str, Any] = {}

    @property
    def wrapped_env(self) -> ALAOneHandEnv:
        return self._env

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self._env.reset()
        self._step_count = 0
        observation = self._observation()
        info = self._info(debug_reward=0.0, action_penalty=0.0)
        self._last_info = info
        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=self.action_space.dtype)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        native_action = self.rescale_action(action)
        timestep = self._env.step(native_action)
        self._step_count += 1

        action_penalty = 0.002 * float(np.mean(np.square(action)))
        reward = self._debug_reward(action_penalty=action_penalty)
        terminated = bool(timestep.last())
        truncated = self._step_count >= self.horizon_steps and not terminated
        observation = self._observation()
        info = self._info(debug_reward=reward, action_penalty=action_penalty)
        self._last_info = info
        return observation, float(reward), terminated, truncated, info

    def rescale_action(self, normalized_action) -> np.ndarray:
        """Map normalized `[-1, 1]` action to the public native 22D action bounds."""

        normalized_action = np.asarray(normalized_action, dtype=np.float32)
        normalized_action = np.clip(
            normalized_action,
            self.action_space.low,
            self.action_space.high,
        )
        fraction = (normalized_action + 1.0) / 2.0
        native = self._native_action_low + fraction * (
            self._native_action_high - self._native_action_low
        )
        return native.astype(self._env.action_spec().dtype)

    @property
    def native_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._native_action_low.copy(), self._native_action_high.copy()

    def _observation(self) -> np.ndarray:
        target_state = self._env.target_key_state(TARGET_KEY) or 0.0
        max_unintended = self._env.max_unintended_key_state(TARGET_KEY)
        pressed_count = len(self._env.current_pressed_keys())
        nearest = self._env.nearest_fingertip_to_key(TARGET_KEY)
        distance = float(nearest["distance"]) if nearest is not None else 1.0
        normalized_distance = min(1.0, distance / 0.20)
        phase = min(1.0, self._step_count / max(1, self.horizon_steps))
        return np.asarray(
            [
                TARGET_KEY / 87.0,
                np.clip(target_state, 0.0, 1.0),
                np.clip(max_unintended, 0.0, 1.0),
                np.clip(pressed_count / 88.0, 0.0, 1.0),
                normalized_distance,
                phase,
            ],
            dtype=np.float32,
        )

    def _debug_reward(self, *, action_penalty: float) -> float:
        target_state = self._env.target_key_state(TARGET_KEY) or 0.0
        max_unintended = self._env.max_unintended_key_state(TARGET_KEY)
        pressed = self._env.current_pressed_keys()
        clean = pressed == [TARGET_KEY]
        wrong_count = len([key for key in pressed if key != TARGET_KEY])
        target_contact = bool(self._env.key_contact_pairs(TARGET_KEY))
        return (
            5.0 * float(target_state)
            - 2.0 * float(max_unintended)
            + (5.0 if clean else 0.0)
            + (0.5 if target_contact else 0.0)
            - float(wrong_count)
            - action_penalty
        )

    def _info(self, *, debug_reward: float, action_penalty: float) -> dict[str, Any]:
        target_state = self._env.target_key_state(TARGET_KEY) or 0.0
        max_unintended = self._env.max_unintended_key_state(TARGET_KEY)
        pressed = tuple(self._env.current_pressed_keys())
        native_reward = self._env.current_reward()
        target_contact = bool(self._env.key_contact_pairs(TARGET_KEY))
        any_key_contact = bool(self._env.key_contact_pairs(None))
        clean = pressed == (TARGET_KEY,)
        near_clean = target_state >= 0.25 and max_unintended <= target_state + 0.02
        return {
            "target_midi": TARGET_MIDI,
            "target_key": TARGET_KEY,
            "target_note": TARGET_NOTE,
            "target_key_state": float(target_state),
            "max_unintended_key_state": float(max_unintended),
            "pressed_keys": pressed,
            "pressed_key_count": len(pressed),
            "wrong_pressed_key_count": len([key for key in pressed if key != TARGET_KEY]),
            "native_reward": None if native_reward is None else float(native_reward),
            "debug_reward": float(debug_reward),
            "action_penalty": float(action_penalty),
            "action_space_mode": "normalized_minus_one_to_one",
            "target_contact": target_contact,
            "any_key_contact": any_key_contact,
            "clean_target_press": clean,
            "near_clean_partial_press": near_clean,
            "sustain_state": float(self._env.task.piano.sustain_state[0]),
        }
