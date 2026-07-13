"""Residual single-note RL environment around dirty open-loop action priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ala_pianist.baselines.calibration import (
    TARGET_KEY as D_SHARP_5_KEY,
    TARGET_MIDI as D_SHARP_5_MIDI,
    TARGET_NOTE as D_SHARP_5_NOTE,
)
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.learning.random_search import generate_random_candidates
from ala_pianist.music import NoteEvent, write_monophonic_midi


ROOT = Path("/home/reece_dev/msc-audio-pianist")
C_SHARP_5_MIDI = 73
C_SHARP_5_KEY = C_SHARP_5_MIDI - 21
REWARD_MODES = ("target_travel_first", "cleanliness")

# Regenerated from the known useful random-search path when possible. This
# sentinel documents that the prior is the dirty D#5 action, not a clean teacher.
D_SHARP_5_DIRTY_BASE_ACTION = "random_search_seed_7_candidate_2"
C_SHARP_5_DIRTY_BASE_ACTION = "csharp5_base_search_best"


class ResidualSingleNoteEnv(gym.Env):
    """Train residual corrections around a target-specific dirty base action."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        midi_path: str | Path | None = None,
        target_midi: int = D_SHARP_5_MIDI,
        wrong_key: int | None = None,
        horizon_steps: int = 24,
        residual_scale: float = 0.1,
        reward_mode: str = "target_travel_first",
        base_action: np.ndarray | None = None,
        base_action_source: str = "auto",
    ):
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}.")
        self.target_midi = int(target_midi)
        self.target_key = self.target_midi - 21
        self.target_note = note_name(self.target_midi)
        self.wrong_key = infer_wrong_key(self.target_midi) if wrong_key is None else int(wrong_key)
        self.midi_path = Path(
            midi_path
            or ROOT / "tmp" / f"residual_midi{self.target_midi}_single_note.mid"
        )
        self.horizon_steps = int(horizon_steps)
        self.residual_scale = float(residual_scale)
        self.reward_mode = reward_mode
        self.base_action_source = str(base_action_source)
        write_single_note_residual_midi(self.midi_path, target_midi=self.target_midi)

        self._env = ALAOneHandEnv(self.midi_path)
        spec = self._env.action_spec()
        self._native_action_low = np.asarray(spec.minimum, dtype=np.float32)
        self._native_action_high = np.asarray(spec.maximum, dtype=np.float32)
        self.base_action = (
            np.asarray(base_action, dtype=np.float32)
            if base_action is not None
            else get_dirty_base_action(
                self.target_midi,
                midi_path=self.midi_path,
                source=self.base_action_source,
            )
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
        ramp = action_ramp(self._step_count)
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
        return make_residual_observation(
            self._env,
            target_key=self.target_key,
            wrong_key=self.wrong_key,
            step_count=self._step_count,
            horizon_steps=self.horizon_steps,
        )

    def _reward(self, action_penalty: float, smoothness_penalty: float) -> float:
        target_state = self._env.target_key_state(self.target_key) or 0.0
        wrong_key_state = self._env.target_key_state(self.wrong_key) or 0.0
        max_unintended = self._env.max_unintended_key_state(self.target_key)
        pressed = self._env.current_pressed_keys()
        wrong_count = len([key for key in pressed if key != self.target_key])
        target_active = self.target_key in pressed
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
            - 8.0 * wrong_key_state
            - 5.0 * max_unintended
            - 3.0 * wrong_count
            - action_penalty
            - smoothness_penalty
        )

    def _info(self, reward: float, action_penalty: float, smoothness_penalty: float) -> dict[str, Any]:
        return residual_info(
            self._env,
            target_midi=self.target_midi,
            target_key=self.target_key,
            target_note=self.target_note,
            wrong_key=self.wrong_key,
            reward=float(reward),
            action_penalty=action_penalty,
            smoothness_penalty=smoothness_penalty,
            reward_mode=self.reward_mode,
            residual_scale=self.residual_scale,
            base_action_source=self.base_action_source,
        )


def write_single_note_residual_midi(path: str | Path, *, target_midi: int) -> Path:
    """Write the generated one-note clip used by residual learning."""

    return write_monophonic_midi(
        [NoteEvent(int(target_midi), 0.0, 1.0, 90)],
        path,
        title=f"single note residual {note_name(target_midi)}",
    )


def get_dirty_base_action(
    target_midi: int,
    *,
    midi_path: str | Path | None = None,
    source: str = "auto",
) -> np.ndarray:
    """Return the dirty base action for a supported single-note target."""

    target_midi = int(target_midi)
    if target_midi == D_SHARP_5_MIDI:
        return get_dirty_dsharp5_base_action(midi_path)
    if target_midi == C_SHARP_5_MIDI:
        return get_dirty_csharp5_base_action()
    raise ValueError(
        f"No dirty base action is available for MIDI {target_midi}. "
        "Run a bounded base-action discovery first."
    )


def get_dirty_dsharp5_base_action(midi_path: str | Path | None = None) -> np.ndarray:
    """Regenerate the known dirty D#5 base action from random-search seed/candidate."""

    path = Path(midi_path or ROOT / "tmp" / "residual_dsharp5_base.mid")
    write_single_note_residual_midi(path, target_midi=D_SHARP_5_MIDI)
    env = ALAOneHandEnv(path)
    env.reset()
    candidates = generate_random_candidates(env, count=30, seed=7)
    return np.asarray(candidates[2].action, dtype=np.float32)


def get_dirty_csharp5_base_action(
    summary_path: str | Path = ROOT / "experiments" / "csharp5_base_search" / "csharp5_base_search_summary.json",
) -> np.ndarray:
    """Load the discovered dirty C#5 base action from the ignored search artifact."""

    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected C#5 base-action summary at {path}. "
            "Run scripts/find_csharp5_base_action.py first."
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    action = np.asarray(summary["best"]["action"], dtype=np.float32)
    if action.shape != (22,):
        raise ValueError(f"Expected C#5 base action shape (22,), got {action.shape}.")
    return action


def make_residual_observation(
    env: ALAOneHandEnv,
    *,
    target_key: int,
    wrong_key: int,
    step_count: int,
    horizon_steps: int = 24,
) -> np.ndarray:
    target_state = env.target_key_state(target_key) or 0.0
    wrong_key_state = env.target_key_state(wrong_key) or 0.0
    max_unintended = env.max_unintended_key_state(target_key)
    pressed_count = len(env.current_pressed_keys())
    nearest = env.nearest_fingertip_to_key(target_key)
    distance = float(nearest["distance"]) if nearest is not None else 1.0
    hand_mean, hand_std = hand_joint_summary(env)
    return np.asarray(
        [
            np.clip(target_state, 0.0, 1.0),
            np.clip(wrong_key_state, 0.0, 1.0),
            np.clip(max_unintended, 0.0, 1.0),
            np.clip(pressed_count / 88.0, 0.0, 1.0),
            min(1.0, step_count / max(1, horizon_steps)),
            min(1.0, distance / 0.20),
            np.clip((hand_mean + hand_std) / 2.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def residual_info(
    env: ALAOneHandEnv,
    *,
    target_midi: int,
    target_key: int,
    target_note: str,
    wrong_key: int,
    reward: float,
    action_penalty: float,
    smoothness_penalty: float,
    reward_mode: str,
    residual_scale: float,
    base_action_source: str,
) -> dict[str, Any]:
    target_state = env.target_key_state(target_key) or 0.0
    wrong_key_state = env.target_key_state(wrong_key) or 0.0
    max_unintended = env.max_unintended_key_state(target_key)
    pressed = tuple(env.current_pressed_keys())
    native_reward = env.current_reward()
    return {
        "target_midi": int(target_midi),
        "target_note": str(target_note),
        "target_key": int(target_key),
        "wrong_key": int(wrong_key),
        "wrong_key_state": float(wrong_key_state),
        "target_key_state": float(target_state),
        "max_unintended_key_state": float(max_unintended),
        "pressed_keys": pressed,
        "wrong_pressed_key_count": len([key for key in pressed if key != target_key]),
        "clean_target_press": pressed == (target_key,),
        "near_clean_partial_press": bool(target_state >= 0.25 and max_unintended <= target_state + 0.02),
        "dirty_press": bool(target_key in pressed and pressed != (target_key,)),
        "native_reward": None if native_reward is None else float(native_reward),
        "debug_reward": float(reward),
        "action_penalty": float(action_penalty),
        "smoothness_penalty": float(smoothness_penalty),
        "reward_mode": reward_mode,
        "residual_scale": float(residual_scale),
        "base_action_source": base_action_source,
        "sustain_state": float(env.task.piano.sustain_state[0]),
    }


def evaluate_residual_action(
    *,
    residual_action: np.ndarray | None = None,
    target_midi: int = D_SHARP_5_MIDI,
    wrong_key: int | None = None,
    base_action: np.ndarray | None = None,
    base_action_source: str = "auto",
    residual_scale: float = 0.1,
    reward_mode: str = "cleanliness",
    horizon_steps: int = 24,
) -> dict:
    env = ResidualSingleNoteEnv(
        target_midi=target_midi,
        wrong_key=wrong_key,
        base_action=base_action,
        base_action_source=base_action_source,
        horizon_steps=horizon_steps,
        residual_scale=residual_scale,
        reward_mode=reward_mode,
    )
    env.reset()
    residual = np.zeros(env.action_space.shape, dtype=np.float32) if residual_action is None else residual_action
    total_reward = 0.0
    native_reward = 0.0
    max_target = 0.0
    max_wrong = 0.0
    max_unintended = 0.0
    pressed = set()
    for _ in range(horizon_steps):
        _, reward, terminated, truncated, info = env.step(residual)
        total_reward += float(reward)
        native_reward += 0.0 if info["native_reward"] is None else float(info["native_reward"])
        max_target = max(max_target, float(info["target_key_state"]))
        max_wrong = max(max_wrong, float(info["wrong_key_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed.update(info["pressed_keys"])
        if terminated or truncated:
            break
    target_key = int(target_midi) - 21
    return {
        "target_midi": int(target_midi),
        "target_key": target_key,
        "target_note": note_name(target_midi),
        "wrong_key": infer_wrong_key(target_midi) if wrong_key is None else int(wrong_key),
        "max_target_key_state": max_target,
        "max_wrong_key_state": max_wrong,
        "max_unintended_key_state": max_unintended,
        "pressed_keys": sorted(pressed),
        "clean_press": pressed == {target_key},
        "near_clean_partial": bool(max_target >= 0.25 and max_unintended <= max_target + 0.02),
        "dirty_press": bool(target_key in pressed and pressed != {target_key}),
        "missed": target_key not in pressed and max_target < 0.25,
        "shaped_return": total_reward,
        "native_reward_sum": native_reward,
    }


def action_ramp(step_count: int) -> float:
    return min(1.0, (int(step_count) + 1) / 6.0)


def infer_wrong_key(target_midi: int) -> int:
    target_midi = int(target_midi)
    if target_midi == C_SHARP_5_MIDI:
        return D_SHARP_5_KEY
    if target_midi == D_SHARP_5_MIDI:
        return 50
    return max(0, min(87, target_midi - 21 - 2))


def note_name(midi_pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    midi_pitch = int(midi_pitch)
    return f"{names[midi_pitch % 12]}{midi_pitch // 12 - 1}"


def hand_joint_summary(env: ALAOneHandEnv) -> tuple[float, float]:
    physics = env.env.physics
    joints = env.task._hand.joints
    qpos = np.asarray(physics.bind(joints).qpos, dtype=float)
    ranges = np.asarray(physics.bind(joints).range, dtype=float)
    denom = np.maximum(1e-6, ranges[:, 1] - ranges[:, 0])
    normalized = np.clip((qpos - ranges[:, 0]) / denom, 0.0, 1.0)
    return float(np.mean(normalized)), float(np.std(normalized))
