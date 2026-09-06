import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def _load_script(name: str):
    script_dir = ROOT / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        path = script_dir / name
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_dir))


def test_audit_sequence_enumeration_excludes_repeated_probes():
    audit = _load_script("audit_five_note_finalist_rollouts.py")
    sequences = audit.audit_sequences()
    assert len(sequences) == 37
    assert "single_72" in sequences
    assert "trained_76_75" in sequences
    assert "heldout_72_74" in sequences
    assert "composition_ascending" in sequences
    assert not any(name.startswith("repeat_") for name in sequences)


def test_reproducible_seed_generation():
    audit = _load_script("audit_five_note_finalist_rollouts.py")
    assert audit.deterministic_seed_list(5, base_seed=13) == [13, 14, 15, 16, 17]
    assert audit.deterministic_seed_list(1, base_seed=99) == [99]


def test_correct_finalist_checkpoint_selection():
    audit = _load_script("audit_five_note_finalist_rollouts.py")
    finalists = {finalist.model_id: finalist for finalist in audit.finalist_definitions()}
    assert set(finalists) == {
        "droq_sensitive_v1_800k_frozen",
        "sac_sensitive_v1_700k",
        "sac_original_900k",
    }
    assert finalists["droq_sensitive_v1_800k_frozen"].checkpoint_step == 800000
    assert finalists["sac_sensitive_v1_700k"].checkpoint_step == 700000
    assert finalists["sac_original_900k"].checkpoint_step == 900000
    assert str(finalists["droq_sensitive_v1_800k_frozen"].checkpoint_path).startswith(
        "/home/reece_dev/msc-audio-pianist/artifacts/frozen_models/five_note_symbolic_controller_v1"
    )


def test_key_index_to_midi_mapping():
    audit = _load_script("audit_five_note_finalist_rollouts.py")
    assert audit.key_to_midi(52) == 73
    assert audit.key_to_midi(51) == 72
    assert audit.key_to_midi(42) == 63
    assert audit.note_name(73) == "C#5"
    assert audit.note_name(63) == "D#4"


def test_failure_classification_rules_from_trace():
    audit = _load_script("audit_five_note_finalist_rollouts.py")
    rows = [
        {
            "intended_key_indices": json.dumps([52]),
            "pressed_key_indices": json.dumps([52]),
            "wrong_key_threshold_crossing_events": json.dumps([]),
        },
        {
            "intended_key_indices": json.dumps([51]),
            "pressed_key_indices": json.dumps([42]),
            "wrong_key_threshold_crossing_events": json.dumps(
                [{"key_index": 42, "category": "unrelated_key_activation", "value": 1.0}]
            ),
        },
    ]
    labels = audit.failure_labels_from_trace(
        pitches=(73, 72),
        rows=rows,
        pressed_keys={52, 42},
        timestep_f1=0.5,
        action_saturation=0.0,
    )
    assert "second_target_missed" in labels
    assert "transition_incomplete" in labels
    assert "unrelated_wrong_key_press" in labels


def test_trace_video_sync_metadata_present():
    summary_path = (
        ROOT
        / "artifacts/frozen_models/five_note_symbolic_controller_v1/validation/smoke_rollout/trained_73_72/summary.json"
    )
    if not summary_path.exists():
        pytest.skip("Frozen controller smoke-rollout artifact is not present in this checkout.")
    summary = json.loads(
        summary_path.read_text()
    )
    assert summary["frame_count"] == summary["trace_rows"]
    assert summary["control_timestep"] > 0
    assert summary["fps"] == round(1.0 / summary["control_timestep"])


def test_no_mutation_of_frozen_or_raw_checkpoint_hashes():
    freeze = _load_script("freeze_five_note_symbolic_controller.py")
    manifest_path = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1/manifest.json"
    if not manifest_path.exists():
        pytest.skip("Frozen controller release artifact is not present in this checkout.")
    manifest = json.loads(manifest_path.read_text())
    frozen = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt"
    source = Path(manifest["source_checkpoint_absolute_path"])
    if not frozen.exists() or not source.exists():
        pytest.skip("Frozen/source checkpoint artifact is not present in this checkout.")
    assert freeze.sha256_file(frozen) == manifest["frozen_checkpoint_sha256"]
    assert freeze.sha256_file(source) == manifest["source_checkpoint_sha256"]
