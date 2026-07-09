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

    def piano_key_states(self) -> np.ndarray:
        return np.asarray(self.task.piano.normalized_state, dtype=float).copy()

    def target_key_state(self, key_index: int | None) -> float | None:
        if key_index is None:
            return None
        return float(self.piano_key_states()[key_index])

    def max_unintended_key_state(self, target_key: int | None) -> float:
        states = self.piano_key_states()
        if target_key is not None and 0 <= target_key < states.size:
            states = np.delete(states, target_key)
        return float(np.max(states)) if states.size else 0.0

    def fingertip_positions(self) -> dict[str, np.ndarray]:
        sites = self.task._hand.fingertip_sites
        positions = self.env.physics.bind(sites).xpos.copy()
        return {str(site.name): positions[idx].copy() for idx, site in enumerate(sites)}

    def key_press_region_position(self, key_index: int) -> np.ndarray:
        geom = self.task.piano.keys[key_index].geom[0]
        position = self.env.physics.bind(geom).xpos.copy()
        size = self.env.physics.bind(geom).size.copy()
        position[-1] += 0.5 * size[2]
        position[0] += 0.35 * size[0]
        return position

    def nearest_fingertip_to_key(self, key_index: int | None) -> dict[str, Any] | None:
        if key_index is None:
            return None
        key_position = self.key_press_region_position(key_index)
        fingertips = self.fingertip_positions()
        best_name = None
        best_position = None
        best_distance = float("inf")
        for name, position in fingertips.items():
            distance = float(np.linalg.norm(position - key_position))
            if distance < best_distance:
                best_name = name
                best_position = position
                best_distance = distance
        return {
            "fingertip": best_name,
            "distance": best_distance,
            "fingertip_position": best_position,
            "key_position": key_position,
        }

    def contact_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        data = self.env.physics.data
        model = self.env.physics.model
        for idx in range(data.ncon):
            contact = data.contact[idx]
            geom1 = model.id2name(int(contact.geom1), "geom") or f"geom_{contact.geom1}"
            geom2 = model.id2name(int(contact.geom2), "geom") or f"geom_{contact.geom2}"
            pairs.append((str(geom1), str(geom2)))
        return pairs

    def key_contact_pairs(self, key_index: int | None = None) -> list[tuple[str, str]]:
        pairs = self.contact_pairs()
        key_tokens = self._key_contact_tokens(key_index)
        return [
            pair
            for pair in pairs
            if self._pair_has_hand(pair) and any(token in " ".join(pair) for token in key_tokens)
        ]

    def note_name_for_key(self, key_index: int) -> str:
        midi_number = midi_file.key_number_to_midi_number(int(key_index))
        return midi_file.midi_number_to_note_name(midi_number)

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

    def _key_contact_tokens(self, key_index: int | None) -> tuple[str, ...]:
        if key_index is None:
            return ("key", "white", "black")
        key = self.task.piano.keys[key_index]
        geom = key.geom[0]
        return tuple(
            token
            for token in (
                str(getattr(key, "name", "")),
                str(getattr(key, "full_identifier", "")),
                str(getattr(geom, "name", "")),
                str(getattr(geom, "full_identifier", "")),
                f"_{key_index}",
            )
            if token
        )

    @staticmethod
    def _pair_has_hand(pair: tuple[str, str]) -> bool:
        text = " ".join(pair)
        return "shadow_hand" in text or "rh_" in text or "lh_" in text
