from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ala_pianist.evaluation.long_horizon import (
    horizon_steps_for_notes,
    load_long_horizon_benchmark,
    long_horizon_metrics_from_trace,
    sequence_counts_by_length_and_archetype,
    sequence_notes,
)
from ala_pianist.evaluation.metrics import binary_key_vector


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MANIFEST = ROOT / "configs" / "long_horizon_compositional_v1.json"
PAIRWISE_MANIFEST = ROOT / "configs" / "complete_pairwise_v1.json"


def _load_evaluator():
    path = ROOT / "scripts" / "evaluate_long_horizon_compositional.py"
    spec = importlib.util.spec_from_file_location("evaluate_long_horizon_compositional", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_long_horizon_manifest_has_expected_lengths_and_no_training_sequences() -> None:
    benchmark = load_long_horizon_benchmark(MANIFEST)
    trained = {(pitch,) for pitch in benchmark.midi_pitches}
    trained.update(
        (left, right)
        for left in benchmark.midi_pitches
        for right in benchmark.midi_pitches
        if left != right
    )

    assert len(benchmark.sequences) == 20
    assert {sequence.length for sequence in benchmark.sequences} == {3, 5, 10, 20}
    assert all(sequence.pitches not in trained for sequence in benchmark.sequences)
    assert all(set(sequence.pitches).issubset(set(range(72, 77))) for sequence in benchmark.sequences)
    counts = sequence_counts_by_length_and_archetype(benchmark.sequences)
    assert len(counts) == 20
    assert all(row["sequence_count"] == 1 for row in counts)


def test_pairwise_manifest_requires_explicit_short_primitive_opt_in() -> None:
    with pytest.raises(ValueError, match="original anchor/two-note transition"):
        load_long_horizon_benchmark(PAIRWISE_MANIFEST)

    benchmark = load_long_horizon_benchmark(PAIRWISE_MANIFEST, allow_trained_short=True)

    assert benchmark.benchmark_name == "complete_pairwise_v1"
    assert len(benchmark.sequences) == 30
    assert sum(sequence.length == 1 for sequence in benchmark.sequences) == 5
    assert sum(sequence.length == 2 for sequence in benchmark.sequences) == 25
    assert any(sequence.pitches == (72, 72) for sequence in benchmark.sequences)


def test_horizon_steps_scales_with_sequence_duration() -> None:
    benchmark = load_long_horizon_benchmark(MANIFEST)
    short = next(sequence for sequence in benchmark.sequences if sequence.length == 3)
    long = next(sequence for sequence in benchmark.sequences if sequence.length == 20)

    short_steps = horizon_steps_for_notes(sequence_notes(short, benchmark.timing))
    long_steps = horizon_steps_for_notes(sequence_notes(long, benchmark.timing))

    assert short_steps > 20
    assert long_steps > short_steps


def test_long_horizon_event_metrics_measure_order_not_only_key_set() -> None:
    benchmark = load_long_horizon_benchmark(MANIFEST)
    sequence = next(sequence for sequence in benchmark.sequences if sequence.name == "len3_repeated_motif")
    notes = sequence_notes(sequence, benchmark.timing)
    times = np.arange(0.0, 1.6, 0.05)
    key_states = np.zeros((len(times), 88), dtype=np.float32)
    target_vectors = []
    pressed_vectors = []
    # Press each unique target key at least once, but miss the repeated middle
    # event's active window. Set-level F1 stays high; event hit rate catches it.
    for row, time_seconds in enumerate(times):
        active = [note.key_index for note in notes if note.onset <= time_seconds < note.offset]
        target_vectors.append(binary_key_vector(active))
        pressed = []
        if 0.0 <= time_seconds < 0.20:
            key_states[row, notes[0].key_index] = 1.0
            pressed.append(notes[0].key_index)
        if 0.80 <= time_seconds < 1.05:
            key_states[row, notes[2].key_index] = 1.0
            pressed.append(notes[2].key_index)
        pressed_vectors.append(binary_key_vector(pressed))

    metrics = long_horizon_metrics_from_trace(
        notes=notes,
        step_times=times,
        target_vectors=target_vectors,
        pressed_vectors=pressed_vectors,
        key_states=key_states,
    )

    assert metrics["pressed_key_recall"] == 1.0
    assert metrics["target_event_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["correctly_executed_prefix_length"] == 1
    assert metrics["first_target_event_failure_index"] == 1
    assert metrics["strict_whole_sequence_success"] is False


def test_long_horizon_metrics_count_unintended_wrong_presses() -> None:
    benchmark = load_long_horizon_benchmark(MANIFEST)
    sequence = benchmark.sequences[0]
    notes = sequence_notes(sequence, benchmark.timing)
    times = np.arange(0.0, 1.6, 0.05)
    key_states = np.zeros((len(times), 88), dtype=np.float32)
    target_vectors = []
    pressed_vectors = []
    wrong_key = 60
    for row, time_seconds in enumerate(times):
        active = [note.key_index for note in notes if note.onset <= time_seconds < note.offset]
        target_vectors.append(binary_key_vector(active))
        pressed = []
        for note in notes:
            if note.onset <= time_seconds < note.offset:
                key_states[row, note.key_index] = 1.0
                pressed.append(note.key_index)
        if 0.4 <= time_seconds < 0.7:
            key_states[row, wrong_key] = 1.0
            pressed.append(wrong_key)
        pressed_vectors.append(binary_key_vector(pressed))

    metrics = long_horizon_metrics_from_trace(
        notes=notes,
        step_times=times,
        target_vectors=target_vectors,
        pressed_vectors=pressed_vectors,
        key_states=key_states,
    )

    assert metrics["target_event_hit_rate"] == 1.0
    assert metrics["unintended_press_count"] >= 1
    assert metrics["unintended_timesteps_above_press_threshold"] > 0
    assert metrics["pressed_key_precision"] < 1.0


def test_explicit_audio_intervention_selection_is_exact() -> None:
    evaluator = _load_evaluator()
    sequences = ((72,), (74,), (72, 73))

    modes = evaluator._audio_modes_by_sequence(
        sequences,
        include_audio_interventions=True,
        intervention_sequences=((74,), (72, 73)),
    )

    assert modes[(72,)] == ("correct",)
    assert modes[(74,)] == ("correct", "zero", "mismatched")
    assert modes[(72, 73)] == ("correct", "zero", "mismatched")


def test_unknown_audio_intervention_sequence_fails() -> None:
    evaluator = _load_evaluator()

    with pytest.raises(ValueError, match="absent from the benchmark"):
        evaluator._audio_modes_by_sequence(
            ((72,),),
            include_audio_interventions=True,
            intervention_sequences=((76,),),
        )


def test_sequence_spec_parser_is_deterministic_and_rejects_empty() -> None:
    evaluator = _load_evaluator()

    assert evaluator._parse_sequence_specs(["72", "73, 76"]) == ((72,), (73, 76))
    with pytest.raises(ValueError, match="must not be empty"):
        evaluator._parse_sequence_specs([","])


def test_skipped_pipeline_checkpoint_is_not_required(tmp_path: Path) -> None:
    evaluator = _load_evaluator()

    audit = evaluator._audit_checkpoints(
        [],
        pipeline1_controller=tmp_path / "absent.pt",
        pipeline2_checkpoints={},
        include_pipeline1=False,
    )

    assert audit["pipeline2"] == {}
    assert audit["pipeline1_symbolic_controller"]["exists"] is False
