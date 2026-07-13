"""General one-hand native-goal Gymnasium environment for Pipeline 1 v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dm_control import composer

from robopianist.models.hands import HandSide
from robopianist.music import midi_file
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand

from ala_pianist.music import CurriculumClip, write_curriculum_midi


@dataclass(frozen=True)
class GeneralRewardConfig:
    """Small shaped reward used only for debug learning."""

    target_travel_weight: float = 4.0
    wrong_travel_weight: float = 2.0
    wrong_pressed_weight: float = 1.0
    action_weight: float = 0.002
    smoothness_weight: float = 0.001
    fingering_weight: float = 0.0
    native_reward_weight: float = 0.0


class GeneralOneHandGoalEnv(gym.Env):
    """SB3-compatible one-hand RoboPianist env using native goal observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        midi_path: str | Path | None = None,
        generated_midi_dir: str | Path = "tmp/general_one_hand",
        curriculum: str = "single_notes",
        midi_min: int = 73,
        midi_max: int = 75,
        seed: int = 0,
        clip_index: int = 0,
        note_count: int = 4,
        lookahead: int = 1,
        horizon_steps: int = 64,
        reward_config: GeneralRewardConfig | None = None,
        use_native_fingering_reward: bool = False,
    ):
        super().__init__()
        if lookahead < 1:
            raise ValueError("General native-goal training should use lookahead >= 1.")
        self.lookahead = int(lookahead)
        self.horizon_steps = int(horizon_steps)
        self.midi_min = int(midi_min)
        self.midi_max = int(midi_max)
        self.curriculum = str(curriculum)
        self.seed_value = int(seed)
        self.clip_index = int(clip_index)
        self.note_count = int(note_count)
        self.reward_config = reward_config or GeneralRewardConfig()
        self.use_native_fingering_reward = bool(use_native_fingering_reward)

        self.curriculum_clip: CurriculumClip | None = None
        if midi_path is None:
            generated_midi_dir = Path(generated_midi_dir)
            midi_path = generated_midi_dir / (
                f"{self.curriculum}_{self.midi_min}_{self.midi_max}_"
                f"seed{self.seed_value}_clip{self.clip_index}.mid"
            )
            self.curriculum_clip = write_curriculum_midi(
                midi_path,
                mode=self.curriculum,
                midi_min=self.midi_min,
                midi_max=self.midi_max,
                seed=self.seed_value,
                clip_index=self.clip_index,
                note_count=self.note_count,
            )
        self.midi_path = Path(midi_path)

        midi = midi_file.MidiFile.from_file(self.midi_path)
        self.task = PianoWithOneShadowHand(
            midi=midi,
            hand_side=HandSide.RIGHT,
            disable_fingering_reward=not self.use_native_fingering_reward,
            n_steps_lookahead=self.lookahead,
            trim_silence=False,
        )
        self.env = composer.Environment(self.task, strip_singleton_obs_buffer_dim=True)
        self._native_action_spec = self.env.action_spec()
        if self._native_action_spec.shape != (23,):
            raise ValueError(
                "Expected native one-hand RoboPianist action shape (23,), got "
                f"{self._native_action_spec.shape}."
            )
        self._native_low = np.asarray(self._native_action_spec.minimum[:-1], dtype=np.float32)
        self._native_high = np.asarray(self._native_action_spec.maximum[:-1], dtype=np.float32)
        self._action_names = tuple(str(actuator.name) for actuator in self.task._hand.actuators)
        self._step_count = 0
        self._previous_normalized_action = np.zeros(22, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-np.ones(22, dtype=np.float32),
            high=np.ones(22, dtype=np.float32),
            dtype=np.float32,
        )

        reset_timestep = self.env.reset()
        self._last_timestep = reset_timestep
        self._observation_names, sample_observation = self._flatten_observation(reset_timestep)
        self.observation_space = spaces.Box(
            low=np.full(sample_observation.shape, -np.inf, dtype=np.float32),
            high=np.full(sample_observation.shape, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.native_goal_shape = np.asarray(reset_timestep.observation["goal"]).shape

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._action_names

    @property
    def native_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._native_low.copy(), self._native_high.copy()

    @property
    def observation_names(self) -> tuple[str, ...]:
        return self._observation_names

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self._step_count = 0
        self._previous_normalized_action = np.zeros(22, dtype=np.float32)
        self._last_timestep = self.env.reset()
        _, observation = self._flatten_observation(self._last_timestep)
        return observation, self._info(
            shaped_reward=0.0,
            reward_components=self._reward_components(
                normalized_action=self._previous_normalized_action,
                native_reward=self._last_timestep.reward,
            ),
        )

    def step(self, action):
        normalized_action = np.asarray(action, dtype=np.float32)
        normalized_action = np.clip(normalized_action, self.action_space.low, self.action_space.high)
        native_action22 = self.rescale_action(normalized_action)
        native_action23 = np.concatenate(
            [native_action22, np.asarray([0.0], dtype=self._native_action_spec.dtype)]
        )
        self._last_timestep = self.env.step(native_action23)
        self._step_count += 1

        components = self._reward_components(
            normalized_action=normalized_action,
            native_reward=self._last_timestep.reward,
        )
        shaped_reward = self._combine_reward_components(components)
        self._previous_normalized_action = normalized_action.copy()
        _, observation = self._flatten_observation(self._last_timestep)
        terminated = bool(self._last_timestep.last())
        truncated = self._step_count >= self.horizon_steps and not terminated
        info = self._info(shaped_reward=shaped_reward, reward_components=components)
        return observation, float(shaped_reward), terminated, truncated, info

    def rescale_action(self, normalized_action) -> np.ndarray:
        """Map normalized `[-1, 1]` actions to native 22D RoboPianist bounds."""

        normalized_action = np.asarray(normalized_action, dtype=np.float32)
        normalized_action = np.clip(normalized_action, self.action_space.low, self.action_space.high)
        fraction = (normalized_action + 1.0) / 2.0
        native = self._native_low + fraction * (self._native_high - self._native_low)
        return native.astype(self._native_action_spec.dtype)

    def current_target_keys(self) -> list[int]:
        goal_frame = self._current_goal_frame()
        return [int(key) for key in np.flatnonzero(goal_frame[:-1] > 0.5)]

    def current_pressed_keys(self) -> list[int]:
        return [int(key) for key in np.flatnonzero(self.task.piano.activation)]

    def piano_key_states(self) -> np.ndarray:
        return np.asarray(self.task.piano.normalized_state, dtype=np.float32).copy()

    def max_unintended_key_state(self) -> float:
        states = self.piano_key_states()
        target_keys = set(self.current_target_keys())
        if target_keys:
            mask = np.ones(states.shape, dtype=bool)
            for key in target_keys:
                if 0 <= key < states.size:
                    mask[key] = False
            states = states[mask]
        return float(np.max(states)) if states.size else 0.0

    def _flatten_observation(self, timestep) -> tuple[tuple[str, ...], np.ndarray]:
        obs = timestep.observation
        parts: list[np.ndarray] = []
        names: list[str] = []

        def add(name: str, value) -> None:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            parts.append(array)
            names.extend([name] * array.size)

        add("goal", obs["goal"])
        add("piano/state", obs.get("piano/state", np.zeros(88, dtype=np.float32)))
        add("rh_shadow_hand/joints_pos", obs.get("rh_shadow_hand/joints_pos", []))
        add("rh_shadow_hand/position", obs.get("rh_shadow_hand/position", []))
        if "fingering" in obs:
            add("fingering", obs["fingering"])
        else:
            add("fingering", np.zeros(5, dtype=np.float32))
        add("phase", np.asarray([self._phase()], dtype=np.float32))
        return tuple(names), np.concatenate(parts).astype(np.float32)

    def _current_goal_frame(self) -> np.ndarray:
        goal = np.asarray(self._last_timestep.observation["goal"], dtype=np.float32)
        return goal.reshape(self.lookahead + 1, self.task.piano.n_keys + 1)[0]

    def _phase(self) -> float:
        return min(1.0, self._step_count / max(1, self.horizon_steps))

    def _reward_components(
        self,
        *,
        normalized_action: np.ndarray,
        native_reward: float | None,
    ) -> dict[str, float]:
        target_keys = self.current_target_keys()
        states = self.piano_key_states()
        target_state = 0.0
        if target_keys:
            target_state = float(max(states[key] for key in target_keys))
        max_unintended = self.max_unintended_key_state()
        wrong_pressed = len([key for key in self.current_pressed_keys() if key not in target_keys])
        action_magnitude = float(np.mean(np.square(normalized_action)))
        smoothness = float(np.mean(np.square(normalized_action - self._previous_normalized_action)))
        native = 0.0 if native_reward is None else float(native_reward)
        fingering = self._fingering_score(target_keys)
        return {
            "target_key_state": target_state,
            "max_unintended_key_state": max_unintended,
            "wrong_pressed_key_count": float(wrong_pressed),
            "action_magnitude": action_magnitude,
            "smoothness": smoothness,
            "native_reward": native,
            "fingering_score": fingering,
        }

    def _combine_reward_components(self, components: dict[str, float]) -> float:
        cfg = self.reward_config
        return (
            cfg.target_travel_weight * components["target_key_state"]
            - cfg.wrong_travel_weight * components["max_unintended_key_state"]
            - cfg.wrong_pressed_weight * components["wrong_pressed_key_count"]
            - cfg.action_weight * components["action_magnitude"]
            - cfg.smoothness_weight * components["smoothness"]
            + cfg.native_reward_weight * components["native_reward"]
            + cfg.fingering_weight * components["fingering_score"]
        )

    def _fingering_score(self, target_keys: list[int]) -> float:
        if not target_keys or self.curriculum_clip is None:
            return 0.0
        key_to_finger = {
            event.pitch - 21: event.fingering
            for event in self.curriculum_clip.events
            if event.fingering is not None
        }
        return 1.0 if any(key in key_to_finger for key in target_keys) else 0.0

    def _info(
        self,
        *,
        shaped_reward: float,
        reward_components: dict[str, float],
    ) -> dict[str, Any]:
        target_keys = tuple(self.current_target_keys())
        pressed_keys = tuple(self.current_pressed_keys())
        max_unintended = reward_components["max_unintended_key_state"]
        target_state = reward_components["target_key_state"]
        if target_keys and any(key in pressed_keys for key in target_keys):
            quality = "gold_demo_candidate" if max_unintended < 0.25 else "weak_demo_candidate"
        elif target_state >= 0.25:
            quality = "weak_demo_candidate"
        else:
            quality = "not_demo_candidate"
        return {
            "target_keys": target_keys,
            "pressed_keys": pressed_keys,
            "target_key_state": float(target_state),
            "max_unintended_key_state": float(max_unintended),
            "native_reward": reward_components["native_reward"],
            "shaped_reward": float(shaped_reward),
            "reward_components": dict(reward_components),
            "action_space_mode": "normalized_minus_one_to_one",
            "native_goal_shape": tuple(self.native_goal_shape),
            "lookahead": self.lookahead,
            "sustain_state": float(self.task.piano.sustain_state[0]),
            "trajectory_quality": quality,
        }
