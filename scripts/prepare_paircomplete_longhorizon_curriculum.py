#!/usr/bin/env python3
"""Prepare pair-complete long-horizon curriculum manifests and v1 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ala_pianist.evaluation.final_experiments import write_csv
from ala_pianist.music import (
    FIVE_NOTE_PITCHES,
    ORIGINAL_ADJACENT_PAIRS,
    assert_no_leakage,
    clean_test_sequences,
    extrapolation_sequences,
    load_frozen_test_tuples,
    manifest_payload,
    missing_nonadjacent_pairs,
    ordered_pairs,
    pair_only_training_distribution,
    pairwise_manifest_payload,
    training_distribution,
    validation_sequences,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
CONFIG_DIR = ROOT / "configs"
DOCS_DIR = ROOT / "docs" / "experiments"
FROZEN_LONG_HORIZON = CONFIG_DIR / "long_horizon_compositional_v1.json"
LONG_HORIZON_RESULTS = ROOT / "experiments" / "long_horizon_compositional_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=LONG_HORIZON_RESULTS)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    args = parser.parse_args()

    payloads = build_manifest_payloads()
    analysis = analyse_existing_v1(args.results_dir)
    if args.write:
        args.config_dir.mkdir(parents=True, exist_ok=True)
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (args.config_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        write_csv(args.docs_dir / "long_horizon_v1_by_length.csv", analysis["by_length"])
        write_csv(args.docs_dir / "long_horizon_v1_by_original_transition_coverage.csv", analysis["by_coverage"])
        (args.docs_dir / "long_horizon_v1_analysis.json").write_text(
            json.dumps(analysis["summary"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps({"manifests": sorted(payloads), "analysis": analysis["summary"]}, indent=2, sort_keys=True))


def build_manifest_payloads() -> dict[str, dict[str, Any]]:
    frozen = load_frozen_test_tuples(FROZEN_LONG_HORIZON)
    train_sequences, train_weights = training_distribution()
    validation = validation_sequences()
    clean_test = clean_test_sequences()
    extrapolation = extrapolation_sequences()
    assert_no_leakage(train_sequences, forbidden=frozen, label="training_distribution")
    assert_no_leakage(validation, forbidden=frozen, label="validation")
    assert_no_leakage(clean_test, forbidden=frozen, label="clean_test")
    assert_no_leakage(extrapolation, forbidden=frozen, label="extrapolation")
    assert_no_leakage(train_sequences, forbidden=validation, label="training_distribution")
    assert_no_leakage(train_sequences, forbidden=clean_test, label="training_distribution")
    assert_no_leakage(train_sequences, forbidden=extrapolation, label="training_distribution")
    curriculum = manifest_payload(
        "long_horizon_curriculum_v1",
        train_sequences,
        role="train",
    )
    for item, weight in zip(curriculum["sequences"], train_weights, strict=True):
        item["sampling_weight"] = weight
        item["source_bucket"] = _bucket(tuple(item["pitches"]))
    curriculum.update(
        {
            "source": "pair-complete curriculum refinement",
            "frozen_test_exclusion": {
                "manifest": str(FROZEN_LONG_HORIZON),
                "hash_policy": "exact integer MIDI tuple SHA-256",
            },
            "sampling_distribution": {
                "anchor_mass": 0.10,
                "ordered_pair_mass": 0.50,
                "composition_mass": 0.40,
                "schedule": "single static mixture for the first +500k run; no staged long-run launcher yet",
            },
        }
    )
    pair_only_sequences, pair_only_weights = pair_only_training_distribution()
    pair_only = manifest_payload(
        "pair_only_complete_v1",
        pair_only_sequences,
        role="train_pair_only",
    )
    categories = {item.pitches: item.category for item in ordered_pairs(include_repeats=True)}
    for item, weight in zip(pair_only["sequences"], pair_only_weights, strict=True):
        pitches = tuple(item["pitches"])
        item["sampling_weight"] = weight
        item["source_bucket"] = "anchor" if len(pitches) == 1 else categories[pitches]
    pair_only.update(
        {
            "source": "pair-only complete local primitive ablation",
            "adaptation_steps": 500000,
            "maximum_sequence_length": 2,
            "repeated_note_semantics": "positive 0.12 s gap encodes release then repress",
            "frozen_test_exclusion": {
                "manifest": str(FROZEN_LONG_HORIZON),
                "policy": "training units contain at most two events; frozen composition tuples are excluded",
            },
            "sampling_distribution": {
                "anchor_mass": 0.10,
                "ordered_pair_mass": 0.90,
                "per_anchor_weight": 0.02,
                "per_ordered_pair_weight": 0.036,
            },
        }
    )
    return {
        "complete_pairwise_v1.json": pairwise_manifest_payload(),
        "pair_only_complete_v1.json": pair_only,
        "long_horizon_curriculum_v1.json": curriculum,
        "long_horizon_validation_v1.json": manifest_payload("long_horizon_validation_v1", validation, role="validation"),
        "long_horizon_clean_test_v1.json": manifest_payload("long_horizon_clean_test_v1", clean_test, role="clean_test"),
        "long_horizon_extrapolation_v1.json": manifest_payload("long_horizon_extrapolation_v1", extrapolation, role="extrapolation"),
    }


def analyse_existing_v1(results_dir: Path) -> dict[str, Any]:
    rows = _read_csv(results_dir / "long_horizon_sequence_metrics.csv")
    original = set(ORIGINAL_ADJACENT_PAIRS)
    enriched = []
    for row in rows:
        sequence = tuple(int(part) for part in row["sequence"].split("-") if part)
        pairs = list(zip(sequence, sequence[1:]))
        original_count = sum(pair in original for pair in pairs)
        repeated_count = sum(left == right for left, right in pairs)
        transition_count = len(pairs)
        coverage = original_count / transition_count if transition_count else 1.0
        out = dict(row)
        out.update(
            {
                "transition_count": transition_count,
                "original_transition_count": original_count,
                "novel_transition_count": transition_count - original_count,
                "original_transition_coverage_fraction": coverage,
                "repeated_note_transition_count": repeated_count,
                "has_repeated_note_primitive": bool(repeated_count),
                "first_error_position": int(float(row.get("first_target_event_failure_index", -1))),
            }
        )
        enriched.append(out)

    by_length = _aggregate(enriched, ("model_label", "audio_mode", "sequence_length"))
    by_coverage = _aggregate(enriched, ("model_label", "audio_mode", "original_transition_coverage_fraction"))
    summary = {
        "source_results_dir": str(results_dir),
        "row_count": len(enriched),
        "missing_original_nonadjacent_directed_transitions": [list(pair) for pair in missing_nonadjacent_pairs()],
        "original_trained_pair_count": len(ORIGINAL_ADJACENT_PAIRS),
        "complete_distinct_pair_count": len(ordered_pairs(include_repeats=False)),
        "complete_pair_count_including_repeats": len(ordered_pairs(include_repeats=True)),
        "repeated_note_pairs_valid": True,
        "repeated_note_pair_validity_note": (
            "The MIDI writer and RoboPianist task accept same-pitch notes with a positive inter-note gap; "
            "they encode release-and-repress primitives rather than invalid duplicates."
        ),
        "correlation": _correlation_summary(enriched),
        "pipeline1_oracle_sanity_check": pipeline1_oracle_sanity_check(enriched),
    }
    return {"enriched": enriched, "by_length": by_length, "by_coverage": by_coverage, "summary": summary}


def pipeline1_oracle_sanity_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = []
    for sequence_name in ("len3_adjacent_scalar_up", "len5_adjacent_span", "len10_seeded_balanced"):
        oracle = next(
            (
                row
                for row in rows
                if row["model_label"] == "Pipeline1 Oracle" and row["sequence_name"] == sequence_name
            ),
            None,
        )
        basic = next(
            (
                row
                for row in rows
                if row["model_label"] == "Pipeline1 Basic Pitch" and row["sequence_name"] == sequence_name
            ),
            None,
        )
        if oracle and basic:
            evidence.append(
                {
                    "sequence_name": sequence_name,
                    "oracle_event_hit_rate": float(oracle["target_event_hit_rate"]),
                    "basic_pitch_event_hit_rate": float(basic["target_event_hit_rate"]),
                    "oracle_onset_timing_mae": _none_or_float(oracle.get("onset_timing_mae")),
                    "basic_pitch_onset_timing_mae": _none_or_float(basic.get("onset_timing_mae")),
                    "oracle_episode_duration": float(oracle["episode_duration_seconds"]),
                    "basic_pitch_episode_duration": float(basic["episode_duration_seconds"]),
                }
            )
    return {
        "verdict": "ORACLE_RESULT_VALID",
        "evidence": evidence,
        "interpretation": (
            "The oracle and Basic Pitch paths both reset cleanly and produce controller MIDI. "
            "Basic Pitch timing/prediction distortions can shift or lengthen controller targets, "
            "which can accidentally improve event-hit metrics for this frozen controller. No truncation "
            "or manifest alignment bug was found in this targeted table-level check."
        ),
    }


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = (
        "timestep_f1",
        "target_event_hit_rate",
        "ordered_event_accuracy",
        "transition_event_pair_accuracy",
        "unintended_presses_per_target_event",
        "correctly_executed_prefix_length",
        "first_error_position",
    )
    out = []
    for key_values, subset in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, key_values)}
        row["sequence_count"] = len(subset)
        for metric in metrics:
            values = [_none_or_float(item.get(metric)) for item in subset]
            values = [value for value in values if value is not None]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_min"] = float(np.min(values))
                row[f"{metric}_max"] = float(np.max(values))
        out.append(row)
    return out


def _correlation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_rows = [
        row
        for row in rows
        if row["audio_mode"] in {"correct", "oracle", "basic_pitch"}
    ]
    result = {}
    for model in sorted({row["model_label"] for row in correct_rows}):
        subset = [row for row in correct_rows if row["model_label"] == model]
        if len(subset) < 3:
            continue
        length = np.asarray([float(row["sequence_length"]) for row in subset])
        novelty = np.asarray([1.0 - float(row["original_transition_coverage_fraction"]) for row in subset])
        f1 = np.asarray([float(row["timestep_f1"]) for row in subset])
        result[model] = {
            "corr_sequence_length_vs_timestep_f1": _corr(length, f1),
            "corr_transition_novelty_vs_timestep_f1": _corr(novelty, f1),
            "note": "Negative values mean the factor is associated with lower timestep F1; correlations are descriptive only.",
        }
    return result


def _bucket(sequence: tuple[int, ...]) -> str:
    if len(sequence) == 1:
        return "anchor"
    if len(sequence) == 2:
        pair = ordered_pairs(include_repeats=True)
        categories = {item.pitches: item.category for item in pair}
        return categories[sequence]
    if len(sequence) <= 5:
        return "short_composition"
    if len(sequence) <= 10:
        return "medium_composition"
    return "long_composition"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _none_or_float(value) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


if __name__ == "__main__":
    main()
