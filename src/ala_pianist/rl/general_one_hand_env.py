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
from ala_pianist.music.sequence_generation import sequence_timing_from_profile
from ala_pianist.evaluation.unintended import unintended_penalty_components


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
    target_activation_bonus: float = 0.0
    target_activation_threshold: float = 0.9
    high_unintended_weight: float = 0.0
    high_unintended_threshold: float = 0.75
    cleanup_gate_threshold: float = 1.1
    gated_unintended_weight: float = 0.0
    gated_wrong_pressed_weight: float = 0.0
    nearby_wrong_key_weight: float = 0.0
    csharp_dsharp_key54_weight: float = 0.0
    dsharp_csharp_key52_weight: float = 0.0
    csharp_dsharp_pressed_weight: float = 0.0
    dsharp_csharp_pressed_weight: float = 0.0
    release_previous_key_weight: float = 0.0
    transition_stray_key_weight: float = 0.0
    transition_stray_pressed_weight: float = 0.0
    press_threshold: float = 0.5
    unintended_soft_threshold: float = 0.2
    unintended_travel_weight: float = 0.0
    unintended_near_press_weight: float = 0.0
    unintended_press_weight: float = 0.0
    late_release_weight: float = 0.0
    early_activation_weight: float = 0.0
    duration_weight: float = 0.0
    release_completion_release_weight: float = 0.0
    release_completion_bonus: float = 0.0
    transition_action_rate_weight: float = 0.0
    transition_saturation_weight: float = 0.0
    transition_saturation_threshold: float = 0.95


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
        midi_pitches: tuple[int, ...] | list[int] | None = None,
        pitch_sampling_weights: tuple[float, ...] | list[float] | None = None,
        sequence_pitches: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None = None,
        sequence_sampling_weights: tuple[float, ...] | list[float] | None = None,
        sequence_timing_profile: str = "legacy_curriculum",
        note_duration: float | None = None,
        note_gap: float | None = None,
        note_velocity: int | None = None,
        timing_jitter: float = 0.0,
        seed: int = 0,
        clip_index: int = 0,
        note_count: int = 4,
        lookahead: int = 1,
        horizon_steps: int = 64,
        reward_config: GeneralRewardConfig | None = None,
        use_native_fingering_reward: bool = False,
        action_mode: str = "direct",
        action_repeat: int = 1,
        ramp_steps: int = 1,
    ):
        super().__init__()
        if lookahead < 1:
            raise ValueError("General native-goal training should use lookahead >= 1.")
        self.lookahead = int(lookahead)
        self.horizon_steps = int(horizon_steps)
        self.midi_min = int(midi_min)
        self.midi_max = int(midi_max)
        self.midi_pitches = self._normalise_midi_pitches(midi_pitches)
        if self.midi_pitches is not None:
            self.midi_min = min(self.midi_pitches)
            self.midi_max = max(self.midi_pitches)
        self.pitch_sampling_weights = self._normalise_pitch_sampling_weights(
            pitch_sampling_weights,
            len(self._available_pitches()),
        )
        self.sequence_pitches = self._normalise_sequence_pitches(sequence_pitches)
        self.sequence_sampling_weights = self._normalise_pitch_sampling_weights(
            sequence_sampling_weights,
            len(self.sequence_pitches) if self.sequence_pitches is not None else 0,
        )
        self.sequence_timing = sequence_timing_from_profile(
            sequence_timing_profile,
            note_duration=note_duration,
            note_gap=note_gap,
            velocity=note_velocity,
            timing_jitter=timing_jitter,
        )
        self.sequence_timing_profile = str(sequence_timing_profile)
        self.curriculum = str(curriculum)
        self.seed_value = int(seed)
        self.clip_index = int(clip_index)
        self.note_count = int(note_count)
        self.reward_config = reward_config or GeneralRewardConfig()
        self.use_native_fingering_reward = bool(use_native_fingering_reward)
        self.action_mode = self._validate_action_mode(action_mode)
        self.action_repeat = self._validate_positive_int(action_repeat, "action_repeat")
        self.ramp_steps = self._validate_positive_int(ramp_steps, "ramp_steps")
        self._generated_midi_dir = Path(generated_midi_dir)
        self._regenerate_curriculum_on_reset = midi_path is None
        self._reset_count = 0
        self._last_reset_seed: int | None = None
        self._generated_env_cache: dict[int, tuple[Path, CurriculumClip, Any, Any]] = {}
        self._weighted_clip_indices: list[int] = []
        self._weighted_sequence_indices: list[int] = []
        self._last_nonempty_target_keys: tuple[int, ...] = ()
        self._reward_previous_target_keys: tuple[int, ...] = ()
        self._reward_future_target_keys: tuple[int, ...] = ()
        self._transition_previous_target_keys: tuple[int, ...] = ()
        self._transition_current_target_keys: tuple[int, ...] = ()
        self._transition_release_achieved = False
        self._transition_completion_awarded = False

        self.curriculum_clip: CurriculumClip | None = None
        if midi_path is None:
            self._select_generated_env(self.clip_index)
        else:
            self.midi_path = Path(midi_path)
            self._build_composer_env(self.midi_path)
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
        self._previous_policy_action = np.zeros(22, dtype=np.float32)

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
        self._previous_policy_action = np.zeros(22, dtype=np.float32)
        self._last_nonempty_target_keys = ()
        self._reward_previous_target_keys = ()
        self._reward_future_target_keys = ()
        self._transition_previous_target_keys = ()
        self._transition_current_target_keys = ()
        self._transition_release_achieved = False
        self._transition_completion_awarded = False
        if self._regenerate_curriculum_on_reset:
            if seed is not None and seed != self._last_reset_seed:
                self.seed_value = int(seed)
                self._reset_count = 0
                self._last_reset_seed = int(seed)
                self._generated_env_cache.clear()
                self._weighted_clip_indices.clear()
                self._weighted_sequence_indices.clear()
            clip_index = self.clip_index + self._reset_count
            self._select_generated_env(clip_index)
            self._reset_count += 1
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
        policy_action = np.asarray(action, dtype=np.float32)
        policy_action = np.clip(policy_action, self.action_space.low, self.action_space.high)
        total_reward = 0.0
        final_components = None
        final_shaped_reward = 0.0
        internal_steps = 0
        for normalized_action in self._expanded_actions(policy_action):
            native_action22 = self.rescale_action(normalized_action)
            native_action23 = np.concatenate(
                [native_action22, np.asarray([0.0], dtype=self._native_action_spec.dtype)]
            )
            self._last_timestep = self.env.step(native_action23)
            self._step_count += 1
            internal_steps += 1

            components = self._reward_components(
                normalized_action=normalized_action,
                native_reward=self._last_timestep.reward,
            )
            shaped_reward = self._combine_reward_components(components)
            total_reward += shaped_reward
            final_components = components
            final_shaped_reward = shaped_reward
            self._previous_normalized_action = normalized_action.copy()
            if self._last_timestep.last() or self._step_count >= self.horizon_steps:
                break
        self._previous_policy_action = policy_action.copy()
        _, observation = self._flatten_observation(self._last_timestep)
        terminated = bool(self._last_timestep.last())
        truncated = self._step_count >= self.horizon_steps and not terminated
        if final_components is None:
            final_components = self._reward_components(
                normalized_action=policy_action,
                native_reward=self._last_timestep.reward,
            )
        info = self._info(
            shaped_reward=final_shaped_reward,
            reward_components=final_components,
            internal_steps=internal_steps,
        )
        info["aggregated_shaped_reward"] = float(total_reward)
        return observation, float(total_reward), terminated, truncated, info

    def rescale_action(self, normalized_action) -> np.ndarray:
        """Map normalized `[-1, 1]` actions to native 22D RoboPianist bounds."""

        normalized_action = np.asarray(normalized_action, dtype=np.float32)
        normalized_action = np.clip(normalized_action, self.action_space.low, self.action_space.high)
        fraction = (normalized_action + 1.0) / 2.0
        native = self._native_low + fraction * (self._native_high - self._native_low)
        return native.astype(self._native_action_spec.dtype)

    def normalize_native_action(self, native_action) -> np.ndarray:
        """Map native 22D RoboPianist actions into normalized `[-1, 1]` space."""

        native_action = np.asarray(native_action, dtype=np.float32)
        native_action = np.clip(native_action, self._native_low, self._native_high)
        span = np.where(self._native_high > self._native_low, self._native_high - self._native_low, 1.0)
        normalized = 2.0 * (native_action - self._native_low) / span - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def current_target_keys(self) -> list[int]:
        goal_frame = self._current_goal_frame()
        return [int(key) for key in np.flatnonzero(goal_frame[:-1] > 0.5)]

    def future_target_keys(self) -> list[int]:
        future = []
        for frame in self._goal_frames()[1:]:
            future.extend(int(key) for key in np.flatnonzero(frame[:-1] > 0.5))
        return sorted(set(future))

    def previous_target_keys(self) -> list[int]:
        return [int(key) for key in self._last_nonempty_target_keys]

    def current_pressed_keys(self) -> list[int]:
        return [int(key) for key in np.flatnonzero(self.task.piano.activation)]

    def piano_key_states(self) -> np.ndarray:
        return np.asarray(self.task.piano.normalized_state, dtype=np.float32).copy()

    def target_key_state(self, key_index: int | None) -> float | None:
        if key_index is None:
            return None
        return float(self.piano_key_states()[int(key_index)])

    def max_unintended_key_state(self, target_key: int | None = None) -> float:
        states = self.piano_key_states()
        target_keys = {int(target_key)} if target_key is not None else set(self.current_target_keys())
        if target_keys:
            mask = np.ones(states.shape, dtype=bool)
            for key in target_keys:
                if 0 <= key < states.size:
                    mask[key] = False
            states = states[mask]
        return float(np.max(states)) if states.size else 0.0

    def fingertip_positions(self) -> dict[str, np.ndarray]:
        sites = self.task._hand.fingertip_sites
        positions = self.env.physics.bind(sites).xpos.copy()
        return {str(site.name): positions[idx].copy() for idx, site in enumerate(sites)}

    def key_press_region_position(self, key_index: int) -> np.ndarray:
        geom = self.task.piano.keys[int(key_index)].geom[0]
        position = self.env.physics.bind(geom).xpos.copy()
        size = self.env.physics.bind(geom).size.copy()
        position[-1] += 0.5 * size[2]
        position[0] += 0.35 * size[0]
        return position

    def nearest_fingertip_to_key(self, key_index: int | None) -> dict[str, Any] | None:
        if key_index is None:
            return None
        key_position = self.key_press_region_position(int(key_index))
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
        return self._goal_frames()[0]

    def _goal_frames(self) -> np.ndarray:
        goal = np.asarray(self._last_timestep.observation["goal"], dtype=np.float32)
        return goal.reshape(self.lookahead + 1, self.task.piano.n_keys + 1)

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
        cleanup_gate = 1.0 if target_state >= self.reward_config.cleanup_gate_threshold else 0.0
        nearby_wrong_key_state = self._nearby_wrong_key_state(target_keys, states)
        csharp_dsharp_key54_state = float(states[54]) if 52 in target_keys else 0.0
        dsharp_csharp_key52_state = float(states[52]) if 54 in target_keys else 0.0
        csharp_dsharp_key54_pressed = 1.0 if 52 in target_keys and 54 in self.current_pressed_keys() else 0.0
        dsharp_csharp_key52_pressed = 1.0 if 54 in target_keys and 52 in self.current_pressed_keys() else 0.0
        previous_target_keys = self._last_nonempty_target_keys
        future_target_keys = self.future_target_keys()
        transition_event = self._update_release_completion_transition(
            target_keys=tuple(int(key) for key in target_keys),
            previous_target_keys=tuple(int(key) for key in previous_target_keys),
        )
        self._reward_previous_target_keys = tuple(int(key) for key in previous_target_keys)
        self._reward_future_target_keys = tuple(int(key) for key in future_target_keys)
        unintended_sensitive = unintended_penalty_components(
            states,
            current_target_keys=target_keys,
            previous_target_keys=previous_target_keys,
            future_target_keys=future_target_keys,
            soft_threshold=self.reward_config.unintended_soft_threshold,
            press_threshold=self.reward_config.press_threshold,
        )
        release_gate = 1.0 if not target_keys and previous_target_keys else 0.0
        release_previous_key_state = 0.0
        if release_gate:
            release_previous_key_state = float(max(states[key] for key in previous_target_keys))
        transition_gate = (
            1.0
            if self._transition_current_target_keys
            and not (self._transition_release_achieved and self._transition_completion_awarded)
            else 0.0
        )
        release_completion_previous_key_state = 0.0
        if transition_gate and self._transition_previous_target_keys:
            release_completion_previous_key_state = float(
                max(states[key] for key in self._transition_previous_target_keys)
            )
        release_achieved_event = 0.0
        if (
            transition_gate
            and not self._transition_release_achieved
            and release_completion_previous_key_state < self.reward_config.unintended_soft_threshold
        ):
            self._transition_release_achieved = True
            release_achieved_event = 1.0
        release_completion_release_penalty_state = (
            0.0 if self._transition_release_achieved else release_completion_previous_key_state
        )
        second_target_completion_event = 0.0
        if (
            transition_gate
            and target_keys
            and tuple(target_keys) == self._transition_current_target_keys
            and not self._transition_completion_awarded
            and target_state >= self.reward_config.press_threshold
        ):
            self._transition_completion_awarded = True
            second_target_completion_event = 1.0
        if (
            self._transition_current_target_keys
            and target_keys
            and tuple(target_keys) != self._transition_current_target_keys
        ):
            self._clear_release_completion_transition()
            transition_gate = 0.0
        if not target_keys and not future_target_keys and self._transition_current_target_keys:
            self._clear_release_completion_transition()
        transition_completed_this_step = (
            transition_gate
            and self._transition_release_achieved
            and self._transition_completion_awarded
        )
        transition_action_rate = smoothness if transition_gate else 0.0
        transition_saturation = 0.0
        if transition_gate:
            threshold = float(self.reward_config.transition_saturation_threshold)
            denom = max(1e-6, 1.0 - threshold)
            transition_saturation = float(
                np.mean(np.maximum(0.0, np.abs(normalized_action) - threshold) / denom)
            )
        transition_episode = self.curriculum_clip is not None and len(set(self.curriculum_clip.key_indices)) > 1
        transition_stray_key_state = 0.0
        transition_stray_pressed_count = 0.0
        if transition_episode:
            stray_keys = (55, 56, 57)
            transition_stray_key_state = float(max(states[key] for key in stray_keys))
            transition_stray_pressed_count = float(
                len([key for key in self.current_pressed_keys() if key in stray_keys])
            )
        if target_keys:
            self._last_nonempty_target_keys = tuple(target_keys)
        components = {
            "target_key_state": target_state,
            "max_unintended_key_state": max_unintended,
            "wrong_pressed_key_count": float(wrong_pressed),
            "action_magnitude": action_magnitude,
            "smoothness": smoothness,
            "native_reward": native,
            "fingering_score": fingering,
            "target_activation": 1.0 if target_state >= self.reward_config.target_activation_threshold else 0.0,
            "high_unintended": max(0.0, max_unintended - self.reward_config.high_unintended_threshold),
            "cleanup_gate": cleanup_gate,
            "nearby_wrong_key_state": nearby_wrong_key_state,
            "csharp_dsharp_key54_state": csharp_dsharp_key54_state,
            "dsharp_csharp_key52_state": dsharp_csharp_key52_state,
            "csharp_dsharp_key54_pressed": csharp_dsharp_key54_pressed,
            "dsharp_csharp_key52_pressed": dsharp_csharp_key52_pressed,
            "release_gate": release_gate,
            "release_previous_key_state": release_previous_key_state,
            "release_completion_transition_gate": transition_gate,
            "release_completion_target_changed_event": float(transition_event),
            "release_completion_previous_key_state": release_completion_previous_key_state,
            "release_completion_release_penalty_state": release_completion_release_penalty_state,
            "release_completion_release_achieved_event": release_achieved_event,
            "release_completion_second_target_event": second_target_completion_event,
            "transition_action_rate": transition_action_rate,
            "transition_saturation": transition_saturation,
            "transition_stray_key_state": transition_stray_key_state,
            "transition_stray_pressed_count": transition_stray_pressed_count,
            **unintended_sensitive,
        }
        if transition_completed_this_step:
            self._clear_release_completion_transition()
        return components

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
            + cfg.target_activation_bonus * components["target_activation"]
            - cfg.high_unintended_weight * components["high_unintended"]
            - cfg.gated_unintended_weight * components["cleanup_gate"] * components["max_unintended_key_state"]
            - cfg.gated_wrong_pressed_weight * components["cleanup_gate"] * components["wrong_pressed_key_count"]
            - cfg.nearby_wrong_key_weight * components["cleanup_gate"] * components["nearby_wrong_key_state"]
            - cfg.csharp_dsharp_key54_weight * components["cleanup_gate"] * components["csharp_dsharp_key54_state"]
            - cfg.dsharp_csharp_key52_weight * components["cleanup_gate"] * components["dsharp_csharp_key52_state"]
            - cfg.csharp_dsharp_pressed_weight * components["cleanup_gate"] * components["csharp_dsharp_key54_pressed"]
            - cfg.dsharp_csharp_pressed_weight * components["cleanup_gate"] * components["dsharp_csharp_key52_pressed"]
            - cfg.release_previous_key_weight * components["release_gate"] * components["release_previous_key_state"]
            - cfg.transition_stray_key_weight * components["transition_stray_key_state"]
            - cfg.transition_stray_pressed_weight * components["transition_stray_pressed_count"]
            - cfg.unintended_travel_weight * components["unintended_continuous_travel"]
            - cfg.unintended_near_press_weight * components["unintended_near_press_barrier"]
            - cfg.unintended_press_weight * components["unintended_pressed_event_count"]
            - cfg.late_release_weight * components["late_release_travel"]
            - cfg.early_activation_weight * components["early_activation_travel"]
            - cfg.duration_weight * components["unintended_integrated_duration"]
            - cfg.release_completion_release_weight
            * components["release_completion_transition_gate"]
            * components["release_completion_release_penalty_state"]
            + cfg.release_completion_bonus * components["release_completion_second_target_event"]
            - cfg.transition_action_rate_weight * components["transition_action_rate"]
            - cfg.transition_saturation_weight * components["transition_saturation"]
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

    @staticmethod
    def _nearby_wrong_key_state(target_keys: list[int], states: np.ndarray) -> float:
        tracked_pairs = {52: 54, 54: 52}
        values = []
        for target_key in target_keys:
            wrong_key = tracked_pairs.get(int(target_key))
            if wrong_key is not None and 0 <= wrong_key < states.size:
                values.append(float(states[wrong_key]))
        return max(values, default=0.0)

    def _update_release_completion_transition(
        self,
        *,
        target_keys: tuple[int, ...],
        previous_target_keys: tuple[int, ...],
    ) -> bool:
        if not target_keys or not previous_target_keys or target_keys == previous_target_keys:
            return False
        if target_keys != self._transition_current_target_keys:
            self._transition_previous_target_keys = previous_target_keys
            self._transition_current_target_keys = target_keys
            self._transition_release_achieved = False
            self._transition_completion_awarded = False
            return True
        return False

    def _clear_release_completion_transition(self) -> None:
        self._transition_previous_target_keys = ()
        self._transition_current_target_keys = ()
        self._transition_release_achieved = False
        self._transition_completion_awarded = False

    def _info(
        self,
        *,
        shaped_reward: float,
        reward_components: dict[str, float],
        internal_steps: int = 0,
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
            "sampled_midi_pitch": self._sampled_midi_pitch(),
            "curriculum_pitches": tuple(self.curriculum_clip.pitches if self.curriculum_clip else ()),
            "curriculum_key_indices": tuple(self.curriculum_clip.key_indices if self.curriculum_clip else ()),
            "target_keys": target_keys,
            "previous_target_keys": tuple(self._reward_previous_target_keys),
            "future_target_keys": tuple(self._reward_future_target_keys),
            "pressed_keys": pressed_keys,
            "target_key_state": float(target_state),
            "max_unintended_key_state": float(max_unintended),
            "native_reward": reward_components["native_reward"],
            "shaped_reward": float(shaped_reward),
            "reward_components": dict(reward_components),
            "action_space_mode": "normalized_minus_one_to_one",
            "action_mode": self.action_mode,
            "action_repeat": self.action_repeat,
            "ramp_steps": self.ramp_steps,
            "pitch_sampling_weights": self.pitch_sampling_weights,
            "sequence_pitches": self.sequence_pitches,
            "sequence_sampling_weights": self.sequence_sampling_weights,
            "sequence_timing_profile": self.sequence_timing_profile,
            "note_duration": self.sequence_timing.note_duration,
            "note_gap": self.sequence_timing.note_gap,
            "note_velocity": self.sequence_timing.velocity,
            "timing_jitter": self.sequence_timing.timing_jitter,
            "internal_steps": int(internal_steps),
            "native_goal_shape": tuple(self.native_goal_shape),
            "lookahead": self.lookahead,
            "sustain_state": float(self.task.piano.sustain_state[0]),
            "trajectory_quality": quality,
        }

    def _expanded_actions(self, policy_action: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.action_mode == "direct":
            return (policy_action,)
        if self.action_mode == "hold":
            return tuple(policy_action.copy() for _ in range(self.action_repeat))
        if self.action_mode == "ramp_hold":
            actions = []
            ramp_count = min(self.ramp_steps, self.action_repeat)
            for idx in range(ramp_count):
                fraction = (idx + 1) / max(1, ramp_count)
                actions.append(
                    (
                        self._previous_policy_action
                        + fraction * (policy_action - self._previous_policy_action)
                    ).astype(np.float32)
                )
            while len(actions) < self.action_repeat:
                actions.append(policy_action.copy())
            return tuple(actions)
        raise RuntimeError(f"Unsupported action_mode {self.action_mode!r}.")

    def _write_generated_clip(self, clip_index: int) -> CurriculumClip:
        cache_key = int(clip_index)
        path = self._generated_midi_dir / (
            f"{self.curriculum}_{self.midi_min}_{self.midi_max}_"
            f"{self._pitch_suffix()}_"
            f"seed{self.seed_value}_clip{int(cache_key)}.mid"
        )
        return write_curriculum_midi(
            path,
            mode=self.curriculum,
            midi_min=self.midi_min,
            midi_max=self.midi_max,
            midi_pitches=self.midi_pitches,
            sequence_pitches=self.sequence_pitches,
            seed=self.seed_value,
            clip_index=int(cache_key),
            note_count=self.note_count,
            note_duration=self.sequence_timing.note_duration,
            gap=self.sequence_timing.note_gap,
            velocity=self.sequence_timing.velocity,
        )

    def _build_composer_env(self, midi_path: str | Path) -> None:
        midi = midi_file.MidiFile.from_file(Path(midi_path))
        self.task = PianoWithOneShadowHand(
            midi=midi,
            hand_side=HandSide.RIGHT,
            disable_fingering_reward=not self.use_native_fingering_reward,
            n_steps_lookahead=self.lookahead,
            trim_silence=False,
        )
        self.env = composer.Environment(self.task, strip_singleton_obs_buffer_dim=True)

    def _select_generated_env(self, clip_index: int) -> None:
        cache_key = self._curriculum_cache_key(clip_index)
        if cache_key not in self._generated_env_cache:
            clip = self._write_generated_clip(cache_key)
            midi = midi_file.MidiFile.from_file(clip.midi_path)
            task = PianoWithOneShadowHand(
                midi=midi,
                hand_side=HandSide.RIGHT,
                disable_fingering_reward=not self.use_native_fingering_reward,
                n_steps_lookahead=self.lookahead,
                trim_silence=False,
            )
            env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)
            self._generated_env_cache[cache_key] = (clip.midi_path, clip, task, env)
        self.midi_path, self.curriculum_clip, self.task, self.env = self._generated_env_cache[cache_key]

    def _curriculum_cache_key(self, clip_index: int) -> int:
        if self.curriculum == "sequence_cleanup":
            if self.sequence_sampling_weights is not None:
                if not self._weighted_sequence_indices:
                    self._weighted_sequence_indices = self._build_weighted_indices(
                        self.sequence_sampling_weights
                    )
                return self._weighted_sequence_indices[int(clip_index) % len(self._weighted_sequence_indices)]
            return int(clip_index) % max(1, len(self._available_sequences()))
        if self.curriculum == "single_notes" and self.pitch_sampling_weights is not None:
            if not self._weighted_clip_indices:
                self._weighted_clip_indices = self._build_weighted_indices(self.pitch_sampling_weights)
            return self._weighted_clip_indices[int(clip_index) % len(self._weighted_clip_indices)]
        if self.curriculum in {"single_notes", "repeated_notes", "two_note_transitions"}:
            cycle_length = len(self._available_pitches())
            return int(clip_index) % max(1, cycle_length)
        return int(clip_index)

    def _build_weighted_indices(self, weights: tuple[float, ...]) -> list[int]:
        weights = np.asarray(weights, dtype=float)
        scale = 100
        counts = np.maximum(1, np.rint(weights * scale).astype(int))
        indices: list[int] = []
        for pitch_index, count in enumerate(counts):
            indices.extend([pitch_index] * int(count))
        rng = np.random.default_rng(self.seed_value)
        rng.shuffle(indices)
        return indices

    def _sampled_midi_pitch(self) -> int | None:
        if self.curriculum_clip is None or not self.curriculum_clip.pitches:
            return None
        return int(self.curriculum_clip.pitches[0])

    def _available_pitches(self) -> tuple[int, ...]:
        if self.midi_pitches is not None:
            return self.midi_pitches
        return tuple(range(self.midi_min, self.midi_max + 1))

    def _available_sequences(self) -> tuple[tuple[int, ...], ...]:
        if self.sequence_pitches is not None:
            return self.sequence_pitches
        return ((73,), (75,), (73, 75), (75, 73))

    def _pitch_suffix(self) -> str:
        if self.midi_pitches is None:
            return "range"
        return "pitches" + "-".join(str(pitch) for pitch in self.midi_pitches)

    @staticmethod
    def _normalise_midi_pitches(
        midi_pitches: tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...] | None:
        if midi_pitches is None:
            return None
        pitches = tuple(int(pitch) for pitch in midi_pitches)
        if not pitches:
            raise ValueError("midi_pitches must contain at least one pitch.")
        if len(set(pitches)) != len(pitches):
            raise ValueError("midi_pitches must not contain duplicates.")
        return pitches

    @staticmethod
    def _normalise_sequence_pitches(
        sequence_pitches: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None,
    ) -> tuple[tuple[int, ...], ...] | None:
        if sequence_pitches is None:
            return None
        sequences = tuple(tuple(int(pitch) for pitch in sequence) for sequence in sequence_pitches)
        if not sequences:
            raise ValueError("sequence_pitches must contain at least one sequence.")
        if any(not sequence for sequence in sequences):
            raise ValueError("sequence_pitches must not contain empty sequences.")
        return sequences

    @staticmethod
    def _normalise_pitch_sampling_weights(
        pitch_sampling_weights: tuple[float, ...] | list[float] | None,
        pitch_count: int,
    ) -> tuple[float, ...] | None:
        if pitch_sampling_weights is None:
            return None
        if pitch_count <= 0:
            raise ValueError("sampling weights require at least one item.")
        weights = tuple(float(weight) for weight in pitch_sampling_weights)
        if len(weights) != int(pitch_count):
            raise ValueError("pitch_sampling_weights must match the number of available MIDI pitches.")
        if any(weight < 0.0 for weight in weights):
            raise ValueError("pitch_sampling_weights must be non-negative.")
        total = float(sum(weights))
        if total <= 0.0:
            raise ValueError("pitch_sampling_weights must contain at least one positive value.")
        return tuple(weight / total for weight in weights)

    @staticmethod
    def _validate_action_mode(action_mode: str) -> str:
        action_mode = str(action_mode)
        if action_mode not in {"direct", "hold", "ramp_hold"}:
            raise ValueError("action_mode must be one of: direct, hold, ramp_hold.")
        return action_mode

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be >= 1.")
        return value
