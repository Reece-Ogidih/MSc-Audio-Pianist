import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path("/home/reece_dev/msc-audio-pianist")


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


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_source_tree(tmp_path: Path):
    freeze = _load_script("freeze_five_note_symbolic_controller.py")
    run = tmp_path / "run"
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    checkpoint = run / "output/lightweight_checkpoints/demo/checkpoint_800000_steps.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "checkpoint_class": "lightweight_policy",
            "algorithm": "droq",
            "config": {
                "observation_dim": 301,
                "action_dim": 22,
                "hidden_dim": 256,
                "critic_ensemble_size": 2,
                "critic_dropout": 0.01,
                "actor_lr": 0.0003,
                "critic_lr": 0.0003,
                "alpha_lr": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "alpha": 0.2,
                "auto_alpha": True,
                "target_entropy": None,
                "batch_size": 256,
                "utd_ratio": 4,
                "buffer_size": 1000000,
                "device": "cpu",
            },
            "actor": {},
            "extra": {"step": 800000},
        },
        checkpoint,
    )
    training = {
        "condition_id": "droq_sensitive_v1",
        "run_name": "droq_sensitive_v1_five_note_seed13_1m",
        "algorithm": "droq",
        "reward_profile": "transition_cleanup_sensitive_v1",
        "seed": 13,
        "timesteps": 1000000,
        "midi_min": 72,
        "midi_max": 76,
        "lookahead": 1,
        "action_mode": "direct",
        "action_repeat": 1,
        "sequence_timing_profile": "aligned",
        "expected_observation_dim": 301,
        "expected_action_dim": 22,
        "curriculum": "sequence_cleanup",
        "sequence_pitches": [[72], [73], [74], [75], [76], [72, 73], [73, 72]],
        "sequence_sampling_weights": [0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25],
    }
    _write_json(run / "resolved_training_config.json", training)
    (run / "resolved_metadata.env").write_text(
        "git_commit=b5be771df058840d8c9e8bda8a48fe27f36fffe0\n",
        encoding="utf-8",
    )
    _write_json(run / "factorial_manifest_seed13.json", {})
    _write_json(run / "five_note_curriculum_v1.json", {})
    _write_json(run / "evaluation/best_checkpoint_by_metric.json", {"checkpoint_step": 800000})
    _write_json(run / "evaluation/evaluation_config.json", {"config": training})
    _write_json(run / "evaluation/evaluation_summary.json", {})
    _write_json(run / "output/demo_config.json", {"droq_config": {"observation_dim": 301, "action_dim": 22}})
    _write_json(run / "output/demo_summary.json", {})

    pd.DataFrame([{"checkpoint_step": 800000, "sequence_group": "trained_single"}]).to_csv(
        run / "evaluation/per_checkpoint_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "checkpoint_step": 800000,
                "sequence_name": "trained_73_72",
                "sequence_group": "trained_transition",
                "pressed_key_f1": 0.5,
            }
        ]
    ).to_csv(run / "evaluation/per_checkpoint_per_sequence_metrics.csv", index=False)
    pd.DataFrame([{"checkpoint_step": 800000, "algorithm": "droq"}]).to_csv(
        run / "evaluation/per_checkpoint_training_health.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "condition_id": "droq_sensitive_v1",
                "checkpoint_step": 800000,
                "trained_f1_mean": 0.79,
            }
        ]
    ).to_csv(analysis / "combined_checkpoint_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition_id": "droq_sensitive_v1",
                "checkpoint_step": 800000,
                "overall_rank": 1,
            }
        ]
    ).to_csv(analysis / "best_checkpoint_per_condition.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition_id": "droq_sensitive_v1",
                "selected_checkpoint_step": 800000,
                "selected_trained_pressed_key_f1": 0.792,
                "selected_trained_timestep_f1": 0.685,
                "selected_integrated_unintended": 1.958,
                "selected_wrong_key_press_count": 0.154,
                "selected_worst_trained_sequence": "trained_73_72",
                "selected_worst_trained_f1": 0.5,
            }
        ]
    ).to_csv(analysis / "per_condition_comparison.csv", index=False)
    return freeze, run, analysis, checkpoint


def test_sha256_file_reports_hash(tmp_path):
    freeze = _load_script("freeze_five_note_symbolic_controller.py")
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert freeze.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_selected_rows_extracts_only_800k(tmp_path):
    freeze, run, analysis, _ = _write_minimal_source_tree(tmp_path)
    rows = freeze.selected_rows(run, analysis)
    assert rows["source_per_sequence_800000"].iloc[0]["sequence_name"] == "trained_73_72"
    assert rows["per_condition_comparison"].iloc[0]["selected_checkpoint_step"] == 800000


def test_refuses_to_overwrite_existing_release(tmp_path):
    freeze, run, analysis, checkpoint = _write_minimal_source_tree(tmp_path)
    release = tmp_path / "release"
    release.mkdir()
    with pytest.raises(FileExistsError):
        freeze.freeze_release(
            release_dir=release,
            source_checkpoint=checkpoint,
            source_run_dir=run,
            analysis_dir=analysis,
            force=False,
        )


def test_manifest_required_fields_and_hash_equality_with_monkeypatched_dims(tmp_path, monkeypatch):
    freeze, run, analysis, checkpoint = _write_minimal_source_tree(tmp_path)
    monkeypatch.setattr(freeze, "checkpoint_dims", lambda path: {"observation_dim": 301, "action_dim": 22})
    monkeypatch.setattr(
        freeze,
        "env_dims",
        lambda config: {
            "observation_dim": 301,
            "action_dim": 22,
            "native_action_shape": [23],
            "native_goal_shape": [178],
            "action_names": [],
        },
    )
    manifest = freeze.freeze_release(
        release_dir=tmp_path / "release",
        source_checkpoint=checkpoint,
        source_run_dir=run,
        analysis_dir=analysis,
        force=False,
    )
    required = {
        "release_name",
        "release_version",
        "model_kind",
        "selected_step",
        "seed",
        "reward_profile",
        "midi_range",
        "action_mode",
        "action_repeat",
        "lookahead",
        "sequence_timing_profile",
        "observation_dim",
        "action_dim",
        "source_checkpoint_sha256",
        "frozen_checkpoint_sha256",
        "training_git_commit",
        "current_local_git_commit",
        "current_local_git_dirty",
        "explicit_checkpoint_statement",
    }
    assert required.issubset(manifest)
    assert manifest["source_checkpoint_sha256"] == manifest["frozen_checkpoint_sha256"]
    assert (tmp_path / "release/SHA256SUMS.txt").exists()
