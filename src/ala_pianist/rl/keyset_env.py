"""Gymnasium keyset environment for symbolic MIDI-to-action training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ala_pianist.controllers.action_library import KEYSET_MIDI
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


KEYSET_KEYS = tuple(pitch - 21 for pitch in KEYSET_MIDI)
OBSERVATION_SIZE = len(KEYSET_MIDI) + 8


class KeysetPianoGymEnv(gym.Env):
    """Single-target sampled keyset env with normalized 22D actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        midi_dir: str | Path = "/home/reece_dev/msc-audio-pianist/tmp/keyset_rl",
        horizon_steps: int = 32,
        keyset_midi: tuple[int, ...] = KEYSET_MIDI,
    ):
        super().__init__()
        self.midi_dir = Path(midi_dir)
        self.midi_dir.mkdir(parents=True, exist_ok=True)
        self.horizon_steps = int(horizon_steps)
        self.keyset_midi = tuple(int(pitch) for pitch in keyset_midi)
        self.keyset_keys = tuple(pitch - 21 for pitch in self.keyset_midi)
        self._midi_paths = {
            pitch: self._write_target_midi(pitch) for pitch in self.keyset_midi
        }

        self.target_midi = self.keyset_midi[0]
        self.target_key = self.target_midi - 21
        self._env = ALAOneHandEnv(self._midi_paths[self.target_midi])
        spec = self._env.action_spec()
        self._native_action_low = np.asarray(spec.minimum, dtype=np.float32)
        self._native_action_high = np.asarray(spec.maximum, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(spec.shape, dtype=np.float32),
            high=np.ones(spec.shape, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            high=np.ones(OBSERVATION_SIZE, dtype=np.float32),
            dtype=np.float32,
        )
        self._step_count = 0
        self._previous_action = np.zeros(spec.shape, dtype=np.float32)

    @property
    def wrapped_env(self) -> ALAOneHandEnv:
        return self._env

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "target_midi" in options:
            target_midi = int(options["target_midi"])
            if target_midi not in self._midi_paths:
                raise ValueError(f"target_midi {target_midi} is not in the keyset.")
            self.target_midi = target_midi
        else:
            idx = int(self.np_random.integers(0, len(self.keyset_midi)))
            self.target_midi = self.keyset_midi[idx]
        self.target_key = self.target_midi - 21
        self._env = ALAOneHandEnv(self._midi_paths[self.target_midi])
        self._env.reset()
        self._step_count = 0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        obs = make_keyset_observation(
            self._env,
            target_key=self.target_key,
            keyset_keys=self.keyset_keys,
            step_count=self._step_count,
            horizon_steps=self.horizon_steps,
        )
        return obs, self._info(debug_reward=0.0, action_penalty=0.0, smoothness_penalty=0.0)

    def step(self, action):
        normalized = np.asarray(action, dtype=np.float32)
        normalized = np.clip(normalized, self.action_space.low, self.action_space.high)
        native_action = self.rescale_action(normalized)
        timestep = self._env.step(native_action)
        self._step_count += 1

        action_penalty = 0.001 * float(np.mean(np.square(normalized)))
        smoothness_penalty = 0.001 * float(np.mean(np.square(normalized - self._previous_action)))
        self._previous_action = normalized.copy()
        reward = self._debug_reward(
            action_penalty=action_penalty,
            smoothness_penalty=smoothness_penalty,
        )
        terminated = bool(timestep.last())
        truncated = self._step_count >= self.horizon_steps and not terminated
        obs = make_keyset_observation(
            self._env,
            target_key=self.target_key,
            keyset_keys=self.keyset_keys,
            step_count=self._step_count,
            horizon_steps=self.horizon_steps,
        )
        info = self._info(
            debug_reward=reward,
            action_penalty=action_penalty,
            smoothness_penalty=smoothness_penalty,
        )
        return obs, float(reward), terminated, truncated, info

    def rescale_action(self, normalized_action) -> np.ndarray:
        normalized_action = np.asarray(normalized_action, dtype=np.float32)
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        fraction = (normalized_action + 1.0) / 2.0
        native = self._native_action_low + fraction * (
            self._native_action_high - self._native_action_low
        )
        return native.astype(self._env.action_spec().dtype)

    def _debug_reward(self, *, action_penalty: float, smoothness_penalty: float) -> float:
        target_state = self._env.target_key_state(self.target_key) or 0.0
        max_unintended = self._env.max_unintended_key_state(self.target_key)
        pressed = self._env.current_pressed_keys()
        clean = pressed == [self.target_key]
        near_clean = target_state >= 0.25 and max_unintended <= target_state + 0.02
        wrong_count = len([key for key in pressed if key != self.target_key])
        return (
            8.0 * float(target_state)
            - 4.0 * float(max_unintended)
            + (8.0 if clean else 0.0)
            + (2.0 if near_clean else 0.0)
            - 2.0 * float(wrong_count)
            - action_penalty
            - smoothness_penalty
        )

    def _info(self, *, debug_reward: float, action_penalty: float, smoothness_penalty: float):
        target_state = self._env.target_key_state(self.target_key) or 0.0
        max_unintended = self._env.max_unintended_key_state(self.target_key)
        pressed = tuple(self._env.current_pressed_keys())
        native_reward = self._env.current_reward()
        return {
            "target_midi": self.target_midi,
            "target_key": self.target_key,
            "target_key_state": float(target_state),
            "max_unintended_key_state": float(max_unintended),
            "pressed_keys": pressed,
            "wrong_pressed_key_count": len([key for key in pressed if key != self.target_key]),
            "native_reward": None if native_reward is None else float(native_reward),
            "debug_reward": float(debug_reward),
            "action_penalty": float(action_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "clean_target_press": pressed == (self.target_key,),
            "near_clean_partial_press": bool(target_state >= 0.25 and max_unintended <= target_state + 0.02),
            "sustain_state": float(self._env.task.piano.sustain_state[0]),
        }

    def _write_target_midi(self, pitch: int) -> Path:
        path = self.midi_dir / f"keyset_target_{pitch}.mid"
        write_monophonic_midi([NoteEvent(pitch, 0.0, 1.0, 90)], path)
        return path


def make_keyset_observation(
    env: ALAOneHandEnv,
    *,
    target_key: int,
    keyset_keys: tuple[int, ...] = KEYSET_KEYS,
    step_count: int,
    horizon_steps: int,
) -> np.ndarray:
    one_hot = np.zeros(len(keyset_keys), dtype=np.float32)
    if target_key in keyset_keys:
        one_hot[keyset_keys.index(target_key)] = 1.0
    target_state = env.target_key_state(target_key) or 0.0
    max_unintended = env.max_unintended_key_state(target_key)
    pressed_count = len(env.current_pressed_keys())
    nearest = env.nearest_fingertip_to_key(target_key)
    distance = float(nearest["distance"]) if nearest is not None else 1.0
    hand_summary = _hand_joint_summary(env)
    tail = np.asarray(
        [
            target_key / 87.0,
            np.clip(target_state, 0.0, 1.0),
            np.clip(max_unintended, 0.0, 1.0),
            np.clip(pressed_count / 88.0, 0.0, 1.0),
            min(1.0, distance / 0.20),
            min(1.0, step_count / max(1, horizon_steps)),
            hand_summary[0],
            hand_summary[1],
        ],
        dtype=np.float32,
    )
    return np.concatenate([one_hot, tail]).astype(np.float32)


def _hand_joint_summary(env: ALAOneHandEnv) -> tuple[float, float]:
    physics = env.env.physics
    joints = env.task._hand.joints
    qpos = np.asarray(physics.bind(joints).qpos, dtype=float)
    ranges = np.asarray(physics.bind(joints).range, dtype=float)
    denom = np.maximum(1e-6, ranges[:, 1] - ranges[:, 0])
    normalized = np.clip((qpos - ranges[:, 0]) / denom, 0.0, 1.0)
    return float(np.mean(normalized)), float(np.std(normalized))
