"""Thin one-hand RoboPianist wrapper for early ALA-Piano experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from dm_control import composer
from dm_env import specs

from robopianist.models.hands import HandSide
from robopianist.music import midi_file
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand


class OneHandRoboPianistEnv:
    """Right-hand RoboPianist environment with sustain hidden from callers.

    The public action has 22 dimensions: the native 20 right-hand/wrist/finger
    controls plus RoboPianist's default `forearm_tx` and `forearm_ty` controls.
    The native sustain action is always appended internally as `0.0`.
    """

    def __init__(self, midi_path: str | Path):
        self.midi_path = Path(midi_path)
        midi = midi_file.MidiFile.from_file(self.midi_path)
        self.task = PianoWithOneShadowHand(
            midi=midi,
            hand_side=HandSide.RIGHT,
            disable_fingering_reward=True,
            n_steps_lookahead=0,
            trim_silence=False,
        )
        self.env = composer.Environment(self.task, strip_singleton_obs_buffer_dim=True)
        self._native_action_spec = self.env.action_spec()
        if self._native_action_spec.shape != (23,):
            raise ValueError(
                "Expected native one-hand RoboPianist action shape (23,), got "
                f"{self._native_action_spec.shape}."
            )
        self._action_spec = specs.BoundedArray(
            shape=(22,),
            dtype=self._native_action_spec.dtype,
            minimum=np.asarray(self._native_action_spec.minimum[:-1]),
            maximum=np.asarray(self._native_action_spec.maximum[:-1]),
            name="ala_one_hand_action",
        )
        self._action_names = tuple(
            str(actuator.name) for actuator in self.task._hand.actuators
        )
        self._last_timestep = None

    def reset(self):
        self._last_timestep = self.env.reset()
        return self._last_timestep

    def step(self, action22):
        action22 = np.asarray(action22, dtype=self._action_spec.dtype)
        if action22.shape != self._action_spec.shape:
            raise ValueError(
                f"Expected action shape {self._action_spec.shape}, got {action22.shape}."
            )
        action22 = np.clip(action22, self._action_spec.minimum, self._action_spec.maximum)
        native_action = np.concatenate(
            [action22, np.asarray([0.0], dtype=self._action_spec.dtype)]
        )
        self._last_timestep = self.env.step(native_action)
        return self._last_timestep

    def action_spec(self):
        return self._action_spec

    def action_names(self) -> tuple[str, ...]:
        return self._action_names

    def action_index(self, name: str) -> int:
        return self._action_names.index(name)

    def current_target_keys(self) -> list[int]:
        timestep = self._require_timestep()
        goal = np.asarray(timestep.observation["goal"])
        frame = goal.reshape(1, self.task.piano.n_keys + 1)[0]
        return [int(key) for key in np.flatnonzero(frame[:-1] > 0.5)]

    def current_pressed_keys(self) -> list[int]:
        return [int(key) for key in np.flatnonzero(self.task.piano.activation)]

    def current_reward(self) -> float | None:
        return self._require_timestep().reward

    def raw_observation(self) -> dict[str, Any]:
        return dict(self._require_timestep().observation)

    @property
    def last_timestep(self):
        return self._last_timestep

    def _require_timestep(self):
        if self._last_timestep is None:
            raise RuntimeError("Call reset() before reading environment state.")
        return self._last_timestep
