"""Shared definitions for the five-note factorial benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONDITION_IDS = (
    "droq_original",
    "droq_sensitive_v1",
    "sac_original",
    "sac_sensitive_v1",
)

FIVE_NOTE_RANGE = (72, 76)
FIVE_NOTE_SEQUENCES = (
    (72,),
    (73,),
    (74,),
    (75,),
    (76,),
    (72, 73),
    (73, 72),
    (73, 74),
    (74, 73),
    (74, 75),
    (75, 74),
    (75, 76),
    (76, 75),
)
FIVE_NOTE_WEIGHTS = (0.10, 0.10, 0.10, 0.10, 0.10) + (0.0625,) * 8

FIVE_NOTE_EVALUATION_SEQUENCES: dict[str, tuple[int, ...]] = {
    "single_72": (72,),
    "single_73": (73,),
    "single_74": (74,),
    "single_75": (75,),
    "single_76": (76,),
    "trained_72_73": (72, 73),
    "trained_73_72": (73, 72),
    "trained_73_74": (73, 74),
    "trained_74_73": (74, 73),
    "trained_74_75": (74, 75),
    "trained_75_74": (75, 74),
    "trained_75_76": (75, 76),
    "trained_76_75": (76, 75),
    "heldout_72_74": (72, 74),
    "heldout_74_72": (74, 72),
    "heldout_73_75": (73, 75),
    "heldout_75_73": (75, 73),
    "heldout_74_76": (74, 76),
    "heldout_76_74": (76, 74),
    "heldout_72_75": (72, 75),
    "heldout_75_72": (75, 72),
    "heldout_73_76": (73, 76),
    "heldout_76_73": (76, 73),
    "heldout_72_76": (72, 76),
    "heldout_76_72": (76, 72),
    "composition_72_73_74": (72, 73, 74),
    "composition_74_75_76": (74, 75, 76),
    "composition_74_73_72": (74, 73, 72),
    "composition_76_75_74": (76, 75, 74),
    "composition_ascending": (72, 73, 74, 75, 76),
    "composition_descending": (76, 75, 74, 73, 72),
    "composition_skip_up": (72, 74, 76),
    "composition_skip_down": (76, 74, 72),
    "composition_mixed_up": (72, 74, 73, 75, 76),
    "composition_mixed_down": (76, 75, 73, 74, 72),
    "composition_return_low": (72, 73, 72),
    "composition_return_high": (76, 75, 76),
    "repeat_72_72": (72, 72),
    "repeat_74_74": (74, 74),
    "repeat_76_76": (76, 76),
    "repeat_72_72_73": (72, 72, 73),
    "repeat_76_76_75": (76, 76, 75),
}

FACTOR_KEYS = {
    "condition_id",
    "algorithm",
    "reward_profile",
    "reward_parameters",
    "run_name",
    "expected_output_root",
    "algorithm_hyperparameters",
}
SENSITIVE_COEFFICIENTS = {
    "unintended_soft_threshold": 0.20,
    "press_threshold": 0.50,
    "unintended_travel_weight": 0.75,
    "unintended_near_press_weight": 0.35,
    "unintended_press_weight": 1.0,
    "late_release_weight": 0.75,
    "early_activation_weight": 0.50,
    "duration_weight": 0.25,
}


def sequence_cli(sequences: tuple[tuple[int, ...], ...] = FIVE_NOTE_SEQUENCES) -> str:
    return ";".join(",".join(str(pitch) for pitch in sequence) for sequence in sequences)


def weights_cli(weights: tuple[float, ...] = FIVE_NOTE_WEIGHTS) -> str:
    return ",".join(f"{weight:g}" for weight in weights)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_factorial_manifest(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    validate_factorial_configs(Path(path).parent, payload)
    return payload


def validate_factorial_configs(config_dir: str | Path, manifest: dict[str, Any] | None = None) -> None:
    config_dir = Path(config_dir)
    manifest = manifest or load_json(config_dir / "factorial_manifest_seed13.json")
    curriculum = load_json(config_dir / "five_note_curriculum_v1.json")
    _validate_curriculum(curriculum)
    conditions = manifest["conditions"]
    if sorted(condition["condition_id"] for condition in conditions) != sorted(CONDITION_IDS):
        raise ValueError("Factorial manifest must define exactly the four seed-13 conditions.")
    common_reference = None
    for condition in conditions:
        training = load_json(config_dir / condition["training_config"])
        _validate_training_config(training, curriculum, condition)
        common = {key: value for key, value in training.items() if key not in FACTOR_KEYS}
        if common_reference is None:
            common_reference = common
        elif common != common_reference:
            raise ValueError(f"Non-factor settings differ for {condition['condition_id']}.")


def _validate_curriculum(curriculum: dict[str, Any]) -> None:
    sequences = tuple(tuple(int(pitch) for pitch in seq) for seq in curriculum["sequence_pitches"])
    weights = tuple(float(weight) for weight in curriculum["sequence_sampling_weights"])
    if sequences != FIVE_NOTE_SEQUENCES:
        raise ValueError("Five-note curriculum does not match v1 sequence list.")
    if weights != FIVE_NOTE_WEIGHTS:
        raise ValueError("Five-note curriculum does not match v1 weights.")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("Five-note sequence weights must sum exactly to 1.0.")
    if curriculum["midi_min"] != FIVE_NOTE_RANGE[0] or curriculum["midi_max"] != FIVE_NOTE_RANGE[1]:
        raise ValueError("Five-note curriculum must use MIDI range 72-76.")
    singles = {sequence[0] for sequence in sequences if len(sequence) == 1}
    if singles != {72, 73, 74, 75, 76}:
        raise ValueError("All five single-note anchors must be present from timestep zero.")


def _validate_training_config(
    training: dict[str, Any],
    curriculum: dict[str, Any],
    condition: dict[str, Any],
) -> None:
    if training["condition_id"] != condition["condition_id"]:
        raise ValueError("Condition ID mismatch.")
    if training["algorithm"] != condition["algorithm"]:
        raise ValueError("Algorithm mismatch.")
    if training["reward_profile"] != condition["reward_profile"]:
        raise ValueError("Reward profile mismatch.")
    for key in [
        "seed",
        "timesteps",
        "midi_min",
        "midi_max",
        "lookahead",
        "action_mode",
        "action_repeat",
        "sequence_timing_profile",
        "expected_observation_dim",
        "expected_action_dim",
    ]:
        if training[key] != curriculum[key]:
            raise ValueError(f"{key} differs between curriculum and {training['condition_id']}.")
    if training["sequence_pitches"] != curriculum["sequence_pitches"]:
        raise ValueError("Sequence list mismatch.")
    if training["sequence_sampling_weights"] != curriculum["sequence_sampling_weights"]:
        raise ValueError("Sequence weights mismatch.")
    if training["resume_from_checkpoint"] is not None:
        raise ValueError("Factorial production conditions must start from scratch.")
    if training["reward_profile"] == "transition_cleanup_sensitive_v1":
        if training["reward_parameters"] != SENSITIVE_COEFFICIENTS:
            raise ValueError("Sensitive reward coefficients are not frozen at v1 values.")
