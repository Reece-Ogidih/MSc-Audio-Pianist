import importlib.util
import json
import sys
from pathlib import Path

from ala_pianist.experiments.five_note_factorial import (
    CONDITION_IDS,
    FIVE_NOTE_SEQUENCES,
    FIVE_NOTE_WEIGHTS,
    SENSITIVE_COEFFICIENTS,
    validate_factorial_configs,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
CONFIG_DIR = ROOT / "configs/five_note_factorial_1m"


def _load_script(name: str):
    script_dir = ROOT / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        path = script_dir / name
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_dir))


def _read_json(name: str):
    return json.loads((CONFIG_DIR / name).read_text())


def test_five_note_curriculum_exact_sequences_and_weights():
    curriculum = _read_json("five_note_curriculum_v1.json")
    assert tuple(tuple(seq) for seq in curriculum["sequence_pitches"]) == FIVE_NOTE_SEQUENCES
    assert tuple(curriculum["sequence_sampling_weights"]) == FIVE_NOTE_WEIGHTS
    assert sum(curriculum["sequence_sampling_weights"]) == 1.0
    assert curriculum["midi_min"] == 72
    assert curriculum["midi_max"] == 76
    assert {seq[0] for seq in curriculum["sequence_pitches"] if len(seq) == 1} == {72, 73, 74, 75, 76}
    assert curriculum["staged_introduction"] is False


def test_factorial_manifest_has_all_four_fresh_cells():
    manifest = _read_json("factorial_manifest_seed13.json")
    conditions = {condition["condition_id"]: condition for condition in manifest["conditions"]}
    assert set(conditions) == set(CONDITION_IDS)
    for condition in conditions.values():
        cfg = _read_json(condition["training_config"])
        assert cfg["resume_from_checkpoint"] is None
        assert cfg["timesteps"] == 1_000_000
        assert cfg["seed"] == 13
        assert cfg["lookahead"] == 1
        assert cfg["action_mode"] == "direct"
        assert cfg["action_repeat"] == 1
        assert cfg["sequence_timing_profile"] == "aligned"
        assert cfg["expected_observation_dim"] == 301
        assert cfg["expected_action_dim"] == 22
        assert cfg["lightweight_checkpoint_steps"] == manifest["lightweight_checkpoint_steps"]
        assert cfg["full_resumable_checkpoint_steps"] == manifest["full_resumable_checkpoint_steps"]


def test_non_factor_settings_are_identical_and_sensitive_coefficients_frozen():
    validate_factorial_configs(CONFIG_DIR)
    for name in ["droq_sensitive_v1_seed13_1m.json", "sac_sensitive_v1_seed13_1m.json"]:
        assert _read_json(name)["reward_parameters"] == SENSITIVE_COEFFICIENTS
    for name in ["droq_original_seed13_1m.json", "sac_original_seed13_1m.json"]:
        assert _read_json(name)["reward_profile"] == "transition_cleanup"
        assert _read_json(name)["reward_parameters"] == {}


def test_hex_wrappers_are_parameterised_and_do_not_embed_hex_locations():
    scripts = [
        "run_five_note_factorial_job.sh",
        "run_five_note_droq_original_1m.sh",
        "run_five_note_droq_sensitive_1m.sh",
        "run_five_note_sac_original_1m.sh",
        "run_five_note_sac_sensitive_1m.sh",
        "verify_five_note_factorial_inputs.sh",
        "smoke_test_five_note_factorial.sh",
        "status_five_note_factorial_job.sh",
        "stop_five_note_factorial_job.sh",
        "resume_five_note_factorial_job.sh",
        "evaluate_five_note_factorial_run.sh",
        "evaluate_five_note_factorial_all.sh",
        "aggregate_five_note_factorial.sh",
        "package_five_note_factorial_results.sh",
    ]
    for script in scripts:
        text = (ROOT / "scripts/hex" / script).read_text()
        assert "set -euo pipefail" in text
        assert "/homes/rgkgo20" not in text
        assert (
            "HEX_SCRATCH" in text
            or "HEX_RUN_DIR" in text
            or script.startswith("run_five_note_")
            or script.startswith("smoke_test_")
        )


def test_aggregation_pareto_and_learning_classification_helpers():
    aggregate = _load_script("aggregate_five_note_factorial.py")
    rows = [
        {
            "checkpoint_step": "1000000",
            "sequence_group": "trained_transition",
            "condition_id": "a",
            "mean_pressed_key_f1": "1.0",
            "mean_timestep_f1": "0.8",
            "mean_integrated_unintended_travel": "1.0",
            "mean_wrong_key_crossings": "0.0",
            "affected_episode_percentage": "0.0",
        },
        {
            "checkpoint_step": "1000000",
            "sequence_group": "trained_transition",
            "condition_id": "b",
            "mean_pressed_key_f1": "0.9",
            "mean_timestep_f1": "0.7",
            "mean_integrated_unintended_travel": "3.0",
            "mean_wrong_key_crossings": "1.0",
            "affected_episode_percentage": "50.0",
        },
    ]
    assert [row["condition_id"] for row in aggregate.pareto_candidates(rows)] == ["a"]
    classified = aggregate.classify_learning_curves(
        [
            {**rows[0], "checkpoint_step": "100000", "mean_timestep_f1": "0.1"},
            {**rows[0], "checkpoint_step": "1000000", "mean_timestep_f1": "0.8"},
        ]
    )
    assert classified[0]["classification"] == "still_improving"
