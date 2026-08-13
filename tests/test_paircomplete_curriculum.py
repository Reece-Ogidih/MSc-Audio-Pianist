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
    pair_only_training_distribution,
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


def test_pair_only_distribution_contains_all_and_only_local_primitives() -> None:
    sequences, weights = pair_only_training_distribution()

    assert len(sequences) == 30
    assert sequences[:5] == tuple((pitch,) for pitch in range(72, 77))
    assert set(sequences[5:]) == {item.pitches for item in ordered_pairs(include_repeats=True)}
    assert all(len(sequence) <= 2 for sequence in sequences)
    assert sum(weights) == pytest.approx(1.0)
    assert sum(weights[:5]) == pytest.approx(0.10)
    assert sum(weights[5:]) == pytest.approx(0.90)
    assert all(weight == pytest.approx(0.02) for weight in weights[:5])
    assert all(weight == pytest.approx(0.036) for weight in weights[5:])
    assert {(pitch, pitch) for pitch in range(72, 77)}.issubset(set(sequences))


def test_pair_only_manifest_is_explicit_and_has_no_long_sequences() -> None:
    payload = json.loads((ROOT / "configs" / "pair_only_complete_v1.json").read_text())

    assert payload["name"] == "pair_only_complete_v1"
    assert payload["maximum_sequence_length"] == 2
    assert len(payload["sequences"]) == 30
    assert all(len(item["pitches"]) <= 2 for item in payload["sequences"])
    assert payload["sampling_distribution"] == {
        "anchor_mass": 0.10,
        "ordered_pair_mass": 0.90,
        "per_anchor_weight": 0.02,
        "per_ordered_pair_weight": 0.036,
    }


@pytest.mark.parametrize(
    ("script_name", "source_fragment", "semantics", "output_fragment"),
    [
        (
            "run_pipeline1_paironly_refinement.sh",
            "checkpoint_800000_steps.pt",
            "actor_weights_only_fresh_critics_optimizers_replay_buffer_rng_from_seed",
            "/workspace/runs/general_one_hand/droq",
        ),
        (
            "run_pipeline2_paironly_refinement.sh",
            "full_checkpoint_1000000_steps.pt",
            "network_optimizer_alpha_warm_start_fresh_replay_rng_from_seed",
            "/workspace/experiments/pipeline2_direct_audio",
        ),
    ],
)
def test_pair_only_hex_wrappers_pin_sources_semantics_and_checkpoints(
    script_name: str,
    source_fragment: str,
    semantics: str,
    output_fragment: str,
) -> None:
    script = (ROOT / "scripts" / "hex" / script_name).read_text()

    assert "configs/pair_only_complete_v1.json" in script
    assert source_fragment in script
    assert semantics in script
    assert output_fragment in script
    assert "100000,250000,500000" in script
    assert "--full-checkpoint-steps 500000" in script
    assert "Refusing to overwrite non-empty run directory" in script


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
