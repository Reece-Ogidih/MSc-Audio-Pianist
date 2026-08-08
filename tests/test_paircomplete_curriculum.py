from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from ala_pianist.music import (
    ORIGINAL_ADJACENT_PAIRS,
    assert_no_leakage,
    clean_test_sequences,
    extrapolation_sequences,
    load_frozen_test_tuples,
    midi_tuple_hash,
    missing_nonadjacent_pairs,
    ordered_pairs,
    training_distribution,
    validation_sequences,
)

ROOT = Path("/home/reece_dev/msc-audio-pianist")
spec = importlib.util.spec_from_file_location(
    "curriculum_manifest_args",
    ROOT / "scripts" / "curriculum_manifest_args.py",
)
curriculum_manifest_args = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(curriculum_manifest_args)


def test_complete_pair_enumeration_and_labels() -> None:
    all_pairs = ordered_pairs(include_repeats=True)
    distinct_pairs = ordered_pairs(include_repeats=False)

    assert len(distinct_pairs) == 20
    assert len(all_pairs) == 25
    assert sum(pair.category == "originally_seen_adjacent" for pair in all_pairs) == 8
    assert sum(pair.category == "newly_added_nonadjacent" for pair in all_pairs) == 12
    assert sum(pair.category == "repeated_note" for pair in all_pairs) == 5
    assert set(ORIGINAL_ADJACENT_PAIRS).issubset({pair.pitches for pair in all_pairs})


def test_missing_nonadjacent_pair_list_is_complete() -> None:
    assert missing_nonadjacent_pairs() == (
        (72, 74),
        (72, 75),
        (72, 76),
        (73, 75),
        (73, 76),
        (74, 72),
        (74, 76),
        (75, 72),
        (75, 73),
        (76, 72),
        (76, 73),
        (76, 74),
    )


def test_repeated_note_pairs_are_positive_gap_primitives() -> None:
    repeated = [pair for pair in ordered_pairs(include_repeats=True) if pair.category == "repeated_note"]
    assert [pair.pitches for pair in repeated] == [(72, 72), (73, 73), (74, 74), (75, 75), (76, 76)]


def test_training_distribution_is_balanced_and_excludes_frozen_splits() -> None:
    sequences, weights = training_distribution()
    frozen = load_frozen_test_tuples(ROOT / "configs" / "long_horizon_compositional_v1.json")

    assert len(sequences) == len(weights)
    assert sum(weights) == pytest.approx(1.0)
    assert_no_leakage(sequences, forbidden=frozen, label="train")
    assert_no_leakage(sequences, forbidden=validation_sequences(), label="train")
    assert_no_leakage(sequences, forbidden=clean_test_sequences(), label="train")
    assert_no_leakage(sequences, forbidden=extrapolation_sequences(), label="train")
    pitch_counts = {pitch: sum(sequence.count(pitch) for sequence in sequences) for pitch in range(72, 77)}
    assert min(pitch_counts.values()) > 0
    assert max(pitch_counts.values()) / min(pitch_counts.values()) < 1.35


def test_exact_midi_tuple_hash_leakage_detection() -> None:
    forbidden = [(72, 73, 74)]
    assert midi_tuple_hash((72, 73, 74)) == midi_tuple_hash([72, 73, 74])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage([(72, 73, 74)], forbidden=forbidden, label="unit")


def test_curriculum_manifest_args_include_horizon_and_weights(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "note_duration": 0.28,
                "note_gap": 0.12,
                "sequences": [
                    {"pitches": [72], "sampling_weight": 0.25},
                    {"pitches": [72, 76, 73], "sampling_weight": 0.75},
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text())

    horizon = curriculum_manifest_args._horizon_steps(
        payload["sequences"],
        note_duration=payload["note_duration"],
        note_gap=payload["note_gap"],
    )

    assert horizon >= 38
