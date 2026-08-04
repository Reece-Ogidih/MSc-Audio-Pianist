"""Direct raw-audio + physical-state environment for Pipeline 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ala_pianist.audio.reference_bank import AudioReferenceBank
from ala_pianist.music import write_sequence_midi
from ala_pianist.music.sequence_generation import sequence_timing_from_profile
from ala_pianist.pipelines.indirect import (
    BENCHMARK_SEQUENCE_PITCHES,
    find_default_soundfont,
    render_midi_with_fluidsynth,
)
from ala_pianist.rl.general_one_hand_env import GeneralOneHandGoalEnv, GeneralRewardConfig


FORBIDDEN_OBSERVATION_FIELDS = (
    "goal",
    "fingering",
    "phase",
    "target",
    "midi",
    "timed_note",
    "controller_goal",
)
PHYSICAL_OBSERVATION_FIELDS = (
    "piano/state",
    "piano/sustain_state",
    "rh_shadow_hand/joints_pos",
    "rh_shadow_hand/position",
)
DIRECT_AUDIO_EVAL_MODES = ("correct", "zero", "mismatched")


@dataclass(frozen=True)
class DirectAudioClip:
    sequence: tuple[int, ...]
    midi_path: Path
    wav_path: Path
    clip_id: int
    variant_index: int
    velocity: int
    gain: float
    split: str = "train"


class DirectAudioGoalEnv(gym.Env):
    """Gymnasium env whose policy observation contains no symbolic target fields."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        audio_bank: AudioReferenceBank | None = None,
        clips: tuple[DirectAudioClip, ...] | None = None,
        generated_root: str | Path = "tmp/direct_audio_pipeline2",
        sequences: tuple[tuple[int, ...], ...] = BENCHMARK_SEQUENCE_PITCHES,
        sequence_sampling_weights: tuple[float, ...] | None = None,
        audio_sample_rate: int = 16_000,
        past_context_seconds: float = 0.10,
        future_context_seconds: float = 0.40,
        sequence_timing_profile: str = "aligned",
        lookahead: int = 1,
        horizon_steps: int = 64,
        reward_config: GeneralRewardConfig | None = None,
        seed: int = 13,
        variants_per_sequence: int = 1,
        sampling_split: str = "train",
    ):
        super().__init__()
        self.sequences = tuple(tuple(int(p) for p in seq) for seq in sequences)
        self.sequence_sampling_weights = _normalise_weights(sequence_sampling_weights, len(self.sequences))
        self.sampling_split = str(sampling_split)
        self.sequence_timing = sequence_timing_from_profile(sequence_timing_profile)
        self.sequence_timing_profile = str(sequence_timing_profile)
        self.lookahead = int(lookahead)
        self.horizon_steps = int(horizon_steps)
        self.reward_config = reward_config or GeneralRewardConfig()
        self.seed_value = int(seed)
        self._rng = np.random.default_rng(self.seed_value)
        self._base_env_cache: dict[int, GeneralOneHandGoalEnv] = {}
        self._active_sequence_index = 0
        self._active_clip_index = 0
        self._step_count = 0

        if audio_bank is None or clips is None:
            audio_bank, clips = build_direct_audio_reference_bank(
                generated_root=generated_root,
                sequences=self.sequences,
                sample_rate=audio_sample_rate,
                past_context_seconds=past_context_seconds,
                future_context_seconds=future_context_seconds,
                sequence_timing_profile=sequence_timing_profile,
                variants_per_sequence=variants_per_sequence,
            )
        self.audio_bank = audio_bank
        self.clips = tuple(clips)
        if not self.clips:
            raise ValueError("DirectAudioGoalEnv requires at least one audio clip.")
        self._sequence_clip_indices = _sequence_clip_indices(
            sequences=self.sequences,
            clips=self.clips,
            split=self.sampling_split,
        )

        sample_env = self._base_env_for_clip_index(0)
        sample_env.reset(seed=self.seed_value)
        physical = self._physical_observation(sample_env)
        audio = self.audio_bank.context_window(clip_id=self.clips[0].clip_id, center_sample=0)
        self.action_space = sample_env.action_space
        self.observation_space = spaces.Dict(
            {
                "audio": spaces.Box(-1.0, 1.0, shape=audio.shape, dtype=np.float32),
                "physical": spaces.Box(
                    low=np.full(physical.shape, -np.inf, dtype=np.float32),
                    high=np.full(physical.shape, np.inf, dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )
        self.physical_observation_names = self._physical_observation_names(sample_env)
        self.action_names = sample_env.action_names

    @property
    def control_timestep_seconds(self) -> float:
        return 0.05

    @property
    def simulation_timestep_seconds(self) -> float:
        env = self._base_env_for_clip_index(self._active_clip_index)
        return float(env.env.physics.model.opt.timestep)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._step_count = 0
        self._active_sequence_index, self._active_clip_index = self._sample_clip_index()
        env = self._base_env_for_clip_index(self._active_clip_index)
        env.reset(seed=None if seed is None else int(seed))
        return self._observation(), self._info(env)

    def reset_to_clip_index(self, clip_index: int, *, seed: int | None = None):
        """Reset deterministically to a known clip for evaluation.

        This selects the underlying task/audio metadata but does not expose clip
        or sequence IDs in the policy observation.
        """

        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._step_count = 0
        self._active_clip_index = int(clip_index)
        clip_sequence = tuple(self.clips[self._active_clip_index].sequence)
        try:
            self._active_sequence_index = self.sequences.index(clip_sequence)
        except ValueError as exc:
            raise ValueError(f"Clip sequence {clip_sequence} is not in the evaluation sequence set.") from exc
        env = self._base_env_for_clip_index(self._active_clip_index)
        env.reset(seed=None if seed is None else int(seed))
        return self._observation(), self._info(env)

    def step(self, action):
        env = self._base_env_for_clip_index(self._active_clip_index)
        observation, reward, terminated, truncated, info = env.step(action)
        del observation
        self._step_count += 1
        direct_info = self._info(env)
        direct_info.update(info)
        return self._observation(), float(reward), bool(terminated), bool(truncated), direct_info

    def replay_metadata(self) -> dict[str, int]:
        return {
            "clip_id": int(self.clips[self._active_clip_index].clip_id),
            "audio_sample_index": int(self.audio_bank.sample_index_for_time(self._step_count * self.control_timestep_seconds)),
        }

    def assert_no_forbidden_observation_fields(self) -> None:
        names = tuple(self.physical_observation_names)
        lowered = " ".join(names).lower()
        for token in FORBIDDEN_OBSERVATION_FIELDS:
            if token in lowered:
                raise AssertionError(f"Forbidden direct-observation token {token!r} found in {names}.")

    def _observation(self) -> dict[str, np.ndarray]:
        env = self._base_env_for_clip_index(self._active_clip_index)
        metadata = self.replay_metadata()
        return {
            "audio": self.audio_bank.context_window(
                clip_id=metadata["clip_id"],
                center_sample=metadata["audio_sample_index"],
            ).astype(np.float32, copy=False),
            "physical": self._physical_observation(env),
        }

    def observation_for_audio_mode(self, observation: dict[str, np.ndarray], *, mode: str, mismatched_clip_id: int | None = None) -> dict[str, np.ndarray]:
        """Return an inference diagnostic observation with altered audio only."""

        if mode not in DIRECT_AUDIO_EVAL_MODES:
            raise ValueError(f"Unsupported direct audio eval mode {mode!r}.")
        out = {
            "audio": np.asarray(observation["audio"], dtype=np.float32).copy(),
            "physical": np.asarray(observation["physical"], dtype=np.float32).copy(),
        }
        if mode == "correct":
            return out
        if mode == "zero":
            out["audio"] = np.zeros_like(out["audio"])
            return out
        if mismatched_clip_id is None:
            mismatched_clip_id = (self.clips[self._active_clip_index].clip_id + 1) % len(self.audio_bank)
        metadata = self.replay_metadata()
        out["audio"] = self.audio_bank.context_window(
            clip_id=int(mismatched_clip_id),
            center_sample=metadata["audio_sample_index"],
        ).astype(np.float32, copy=False)
        return out

    def _info(self, env: GeneralOneHandGoalEnv) -> dict[str, Any]:
        clip = self.clips[self._active_clip_index]
        metadata = self.replay_metadata()
        return {
            "clip_id": metadata["clip_id"],
            "logical_sequence_index": int(self._active_sequence_index),
            "audio_sample_index": metadata["audio_sample_index"],
            "sequence": clip.sequence,
            "variant_index": int(clip.variant_index),
            "split": clip.split,
            "midi_path": str(clip.midi_path),
            "wav_path": str(clip.wav_path),
            "policy_observation_fields": ("audio", "physical"),
            "physical_observation_names": self.physical_observation_names,
            "hidden_target_keys": env.current_target_keys(),
            "pressed_keys": env.current_pressed_keys(),
        }

    def _sample_clip_index(self) -> tuple[int, int]:
        sequence_index = int(self._rng.choice(len(self.sequences), p=self.sequence_sampling_weights))
        eligible_clip_indices = self._sequence_clip_indices[sequence_index]
        clip_index = int(self._rng.choice(eligible_clip_indices))
        return sequence_index, clip_index

    def _base_env_for_clip_index(self, clip_index: int) -> GeneralOneHandGoalEnv:
        clip_index = int(clip_index)
        if clip_index not in self._base_env_cache:
            clip = self.clips[clip_index]
            self._base_env_cache[clip_index] = GeneralOneHandGoalEnv(
                midi_path=clip.midi_path,
                midi_pitches=(72, 73, 74, 75, 76),
                sequence_timing_profile=self.sequence_timing_profile,
                lookahead=self.lookahead,
                horizon_steps=self.horizon_steps,
                reward_config=self.reward_config,
                action_mode="direct",
                action_repeat=1,
                seed=self.seed_value,
            )
        return self._base_env_cache[clip_index]

    @staticmethod
    def _physical_observation(env: GeneralOneHandGoalEnv) -> np.ndarray:
        timestep = env._last_timestep
        obs = timestep.observation
        parts = []
        for name in PHYSICAL_OBSERVATION_FIELDS:
            parts.append(np.asarray(obs.get(name, []), dtype=np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _physical_observation_names(env: GeneralOneHandGoalEnv) -> tuple[str, ...]:
        obs = env._last_timestep.observation
        names = []
        for name in PHYSICAL_OBSERVATION_FIELDS:
            size = np.asarray(obs.get(name, []), dtype=np.float32).size
            names.extend([name] * int(size))
        return tuple(names)


def build_direct_audio_reference_bank(
    *,
    generated_root: str | Path,
    sequences: tuple[tuple[int, ...], ...] = BENCHMARK_SEQUENCE_PITCHES,
    sample_rate: int = 16_000,
    past_context_seconds: float = 0.10,
    future_context_seconds: float = 0.40,
    sequence_timing_profile: str = "aligned",
    variants_per_sequence: int = 1,
    split: str = "train",
) -> tuple[AudioReferenceBank, tuple[DirectAudioClip, ...]]:
    root = Path(generated_root)
    midi_dir = root / "midi"
    wav_dir = root / "wav"
    midi_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    bank = AudioReferenceBank(
        sample_rate=sample_rate,
        past_context_seconds=past_context_seconds,
        future_context_seconds=future_context_seconds,
    )
    timing = sequence_timing_from_profile(sequence_timing_profile)
    clips: list[DirectAudioClip] = []
    soundfont = find_default_soundfont()
    variants_per_sequence = max(1, int(variants_per_sequence))
    for sequence in sequences:
        sequence = tuple(int(pitch) for pitch in sequence)
        for variant_index in range(variants_per_sequence):
            velocity = int(80 + 8 * (variant_index % 4))
            gain = float(0.45 + 0.05 * (variant_index % 4))
            name = "_".join(str(pitch) for pitch in sequence)
            stem = f"{name}_variant{variant_index}_v{velocity}_g{gain:.2f}".replace(".", "p")
            midi_path = midi_dir / f"{stem}.mid"
            wav_path = wav_dir / f"{stem}.wav"
            if not midi_path.exists():
                write_sequence_midi(
                    sequence,
                    midi_path,
                    midi_min=72,
                    midi_max=76,
                    timing=type(timing)(
                        note_duration=timing.note_duration,
                        note_gap=timing.note_gap,
                        velocity=velocity,
                        timing_jitter=timing.timing_jitter,
                    ),
                )
            if not wav_path.exists():
                render_midi_with_fluidsynth(
                    midi_path,
                    wav_path,
                    soundfont_path=soundfont,
                    sample_rate=44100,
                    gain=gain,
                )
            clip_id = bank.add_wav(
                wav_path,
                name=stem,
                metadata={
                    "sequence": sequence,
                    "variant_index": variant_index,
                    "velocity": velocity,
                    "gain": gain,
                    "split": split,
                    "midi_path": str(midi_path),
                },
            )
            clips.append(
                DirectAudioClip(
                    sequence=sequence,
                    midi_path=midi_path,
                    wav_path=wav_path,
                    clip_id=clip_id,
                    variant_index=variant_index,
                    velocity=velocity,
                    gain=gain,
                    split=split,
                )
            )
    return bank, tuple(clips)


def _sequence_clip_indices(
    *,
    sequences: tuple[tuple[int, ...], ...],
    clips: tuple[DirectAudioClip, ...],
    split: str = "train",
) -> tuple[tuple[int, ...], ...]:
    sequence_to_logical_index = {tuple(sequence): index for index, sequence in enumerate(sequences)}
    if len(sequence_to_logical_index) != len(sequences):
        raise ValueError("DirectAudioGoalEnv sequences must be unique.")
    grouped: list[list[int]] = [[] for _ in sequences]
    unknown_sequences = set()
    for clip_index, clip in enumerate(clips):
        if clip.split != split:
            continue
        sequence = tuple(int(pitch) for pitch in clip.sequence)
        logical_index = sequence_to_logical_index.get(sequence)
        if logical_index is None:
            unknown_sequences.add(sequence)
            continue
        grouped[logical_index].append(int(clip_index))
    missing = [sequences[index] for index, indices in enumerate(grouped) if not indices]
    if missing:
        raise ValueError(
            "Every logical sequence must have at least one eligible training audio clip. "
            f"Missing sequences for split {split!r}: {missing}."
        )
    if unknown_sequences:
        raise ValueError(
            "Audio clips include sequences that are not in the DirectAudioGoalEnv curriculum: "
            f"{sorted(unknown_sequences)}."
        )
    return tuple(tuple(indices) for indices in grouped)


def _normalise_weights(weights: tuple[float, ...] | None, count: int) -> np.ndarray | None:
    if weights is None:
        return None
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (count,):
        raise ValueError(f"Expected {count} sampling weights, got {arr.shape}.")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("Sampling weights must sum to a positive value.")
    return arr / total
