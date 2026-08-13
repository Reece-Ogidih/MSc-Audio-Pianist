"""Pair-complete five-note curriculum specifications."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FIVE_NOTE_PITCHES: tuple[int, ...] = (72, 73, 74, 75, 76)
ORIGINAL_ADJACENT_PAIRS: tuple[tuple[int, int], ...] = (
    (72, 73),
    (73, 72),
    (73, 74),
    (74, 73),
    (74, 75),
    (75, 74),
    (75, 76),
    (76, 75),
)


@dataclass(frozen=True)
class PrimitiveSequence:
    """A labelled local primitive used by both Pipeline 1 and Pipeline 2."""

    name: str
    pitches: tuple[int, ...]
    category: str

    @property
    def midi_tuple_hash(self) -> str:
        return midi_tuple_hash(self.pitches)


def midi_tuple_hash(pitches: Iterable[int]) -> str:
    """Stable hash for exact integer MIDI tuples."""

    text = ",".join(str(int(pitch)) for pitch in pitches)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def anchors(pitches: tuple[int, ...] = FIVE_NOTE_PITCHES) -> tuple[PrimitiveSequence, ...]:
    return tuple(
        PrimitiveSequence(name=f"anchor_{pitch}", pitches=(int(pitch),), category="anchor")
        for pitch in pitches
    )


def ordered_pairs(
    pitches: tuple[int, ...] = FIVE_NOTE_PITCHES,
    *,
    include_repeats: bool = True,
) -> tuple[PrimitiveSequence, ...]:
    """Enumerate complete ordered local pair primitives."""

    original = set(ORIGINAL_ADJACENT_PAIRS)
    out = []
    for left in pitches:
        for right in pitches:
            if left == right and not include_repeats:
                continue
            pair = (int(left), int(right))
            if left == right:
                category = "repeated_note"
            elif pair in original:
                category = "originally_seen_adjacent"
            else:
                category = "newly_added_nonadjacent"
            out.append(PrimitiveSequence(name=f"pair_{left}_{right}", pitches=pair, category=category))
    return tuple(out)


def missing_nonadjacent_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        pair.pitches
        for pair in ordered_pairs(include_repeats=False)
        if pair.category == "newly_added_nonadjacent"
    )


def load_frozen_test_tuples(path: str | Path) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(tuple(int(pitch) for pitch in item["pitches"]) for item in payload["sequences"])


def assert_no_leakage(
    sequences: Iterable[Iterable[int]],
    *,
    forbidden: Iterable[Iterable[int]],
    label: str,
) -> None:
    forbidden_hashes = {midi_tuple_hash(sequence) for sequence in forbidden}
    leaked = [tuple(int(pitch) for pitch in sequence) for sequence in sequences if midi_tuple_hash(sequence) in forbidden_hashes]
    if leaked:
        raise ValueError(f"{label} contains frozen/held-out sequence leakage: {leaked}")


def default_train_compositions() -> tuple[tuple[int, ...], ...]:
    """Deterministic training compositions over lengths 3..20.

    These are not the frozen long-horizon v1 test tuples. The finite pool keeps
    the current training infrastructure simple while covering every length.
    """

    return (
        (72, 74, 72),
        (76, 74, 76),
        (73, 76, 72, 75),
        (75, 72, 76, 73),
        (72, 74, 76, 73, 75),
        (76, 73, 72, 75, 74),
        (72, 75, 73, 76, 74, 72),
        (76, 72, 75, 73, 74, 76),
        (72, 76, 74, 73, 75, 72, 74),
        (75, 72, 74, 76, 73, 75, 72),
        (72, 74, 73, 76, 75, 72, 76, 73),
        (76, 74, 72, 75, 73, 76, 72, 74),
        (72, 75, 76, 73, 74, 72, 73, 76, 75),
        (76, 73, 75, 72, 74, 76, 75, 73, 72),
        (72, 76, 73, 74, 75, 72, 74, 76, 73, 75),
        (76, 72, 75, 74, 73, 76, 74, 72, 75, 73),
        (72, 74, 76, 75, 73, 72, 75, 76, 74, 73, 72),
        (76, 74, 72, 73, 75, 76, 73, 72, 74, 75, 76),
        (72, 75, 73, 76, 74, 72, 76, 75, 73, 74, 72, 75),
        (76, 73, 75, 72, 74, 76, 72, 73, 75, 74, 76, 73),
        (72, 76, 74, 73, 75, 72, 74, 76, 75, 73, 72, 75, 74),
        (76, 72, 73, 75, 74, 76, 73, 72, 75, 74, 73, 76, 72),
        (72, 74, 75, 73, 76, 72, 75, 74, 73, 76, 72, 74, 75, 73),
        (76, 75, 73, 72, 74, 76, 73, 75, 72, 74, 76, 75, 73, 72),
        (72, 75, 74, 76, 73, 72, 74, 75, 76, 72, 73, 75, 74, 76, 73),
        (76, 74, 73, 75, 72, 76, 75, 74, 72, 73, 76, 74, 75, 72, 73),
        (72, 76, 75, 73, 74, 72, 75, 76, 73, 74, 72, 76, 75, 74, 73, 72),
        (76, 72, 74, 75, 73, 76, 74, 72, 75, 73, 76, 72, 74, 73, 75, 76),
        (72, 74, 76, 73, 75, 72, 76, 74, 73, 75, 72, 74, 76, 75, 73, 72, 76),
        (76, 73, 72, 75, 74, 76, 72, 73, 75, 74, 76, 73, 72, 74, 75, 76, 73),
        (72, 75, 73, 74, 76, 72, 74, 75, 73, 76, 72, 75, 74, 73, 76, 72, 74, 75),
        (76, 74, 75, 72, 73, 76, 75, 74, 72, 73, 76, 74, 75, 73, 72, 76, 75, 74),
        (72, 76, 74, 75, 73, 72, 75, 74, 76, 73, 72, 76, 75, 74, 73, 72, 75, 76, 74),
        (76, 72, 75, 73, 74, 76, 73, 75, 72, 74, 76, 72, 73, 75, 74, 76, 75, 73, 72),
        (72, 74, 76, 75, 73, 72, 76, 74, 75, 73, 72, 75, 74, 76, 73, 72, 74, 75, 76, 73),
        (76, 73, 75, 74, 72, 76, 74, 73, 75, 72, 76, 75, 73, 74, 72, 76, 73, 75, 74, 72),
    )


def validation_sequences() -> tuple[tuple[int, ...], ...]:
    return (
        (72, 75, 74),
        (76, 73, 74),
        (72, 74, 75, 73, 76),
        (76, 74, 73, 75, 72),
        (72, 75, 76, 74, 73, 72, 74, 75, 73, 76),
        (76, 73, 72, 74, 75, 76, 74, 73, 75, 72),
        (72, 74, 75, 76, 73, 72, 75, 74, 76, 73, 72, 74, 76, 75, 73, 72, 75, 74, 73, 76),
        (76, 75, 73, 74, 72, 76, 73, 75, 74, 72, 76, 74, 75, 73, 72, 76, 75, 74, 73, 72),
    )


def clean_test_sequences() -> tuple[tuple[int, ...], ...]:
    return (
        (74, 76, 75),
        (73, 72, 74),
        (74, 76, 75, 73, 72),
        (72, 75, 73, 76, 74),
        (74, 76, 75, 73, 72, 74, 75, 76, 73, 72),
        (72, 74, 76, 73, 75, 72, 76, 74, 75, 73),
        (74, 76, 75, 73, 72, 74, 76, 73, 75, 72, 74, 75, 76, 73, 72, 75, 74, 76, 73, 72),
        (72, 75, 73, 76, 74, 72, 76, 75, 73, 74, 72, 75, 76, 73, 74, 72, 76, 75, 74, 73),
    )


def extrapolation_sequences() -> tuple[tuple[int, ...], ...]:
    base30 = (
        (72, 74, 76, 75, 73, 72, 75, 74, 76, 73, 72, 76, 75, 74, 73, 72, 74, 75, 76, 73, 72, 75, 74, 76, 73, 72, 74, 76, 75, 73),
        (76, 74, 72, 75, 73, 76, 72, 74, 75, 73, 76, 75, 72, 74, 73, 76, 72, 75, 74, 73, 76, 74, 72, 75, 73, 76, 75, 74, 72, 73),
        (73, 75, 72, 76, 74, 73, 72, 75, 76, 74, 73, 76, 72, 75, 74, 73, 75, 72, 76, 74, 73, 72, 75, 74, 76, 73, 75, 72, 74, 76),
        (75, 72, 74, 76, 73, 75, 74, 72, 76, 73, 75, 76, 74, 72, 73, 75, 72, 76, 74, 73, 75, 74, 72, 73, 76, 75, 72, 74, 76, 73),
        (74, 72, 75, 73, 76, 74, 75, 72, 73, 76, 74, 76, 73, 75, 72, 74, 72, 76, 75, 73, 74, 75, 72, 76, 73, 74, 72, 75, 76, 73),
    )
    return base30 + tuple(sequence + sequence[:10] for sequence in base30)


def training_distribution() -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return finite training sequence pool and normalized sampling weights."""

    anchor_items = [item.pitches for item in anchors()]
    pair_items = [item.pitches for item in ordered_pairs(include_repeats=True)]
    compositions = list(default_train_compositions())
    grouped = {
        "anchors": (anchor_items, 0.10),
        "pairs": (pair_items, 0.50),
        "compositions": (compositions, 0.40),
    }
    sequences: list[tuple[int, ...]] = []
    weights: list[float] = []
    for items, mass in grouped.values():
        for sequence in items:
            sequences.append(sequence)
            weights.append(float(mass) / len(items))
    total = sum(weights)
    return tuple(sequences), tuple(weight / total for weight in weights)


def pair_only_training_distribution() -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return anchors plus every ordered pair with a 10/90 mass split."""

    anchor_items = tuple(item.pitches for item in anchors())
    pair_items = tuple(item.pitches for item in ordered_pairs(include_repeats=True))
    sequences = anchor_items + pair_items
    weights = tuple(0.10 / len(anchor_items) for _ in anchor_items) + tuple(
        0.90 / len(pair_items) for _ in pair_items
    )
    return sequences, weights


def manifest_payload(name: str, sequences: Iterable[Iterable[int]], *, role: str) -> dict[str, Any]:
    items = []
    for index, sequence in enumerate(sequences):
        pitches = tuple(int(pitch) for pitch in sequence)
        items.append(
            {
                "name": f"{role}_{index:03d}_len{len(pitches)}",
                "pitches": list(pitches),
                "key_indices": [pitch - 21 for pitch in pitches],
                "length": len(pitches),
                "archetype": _archetype(role, len(pitches)),
                "midi_tuple_hash": midi_tuple_hash(pitches),
            }
        )
    return {
        "benchmark_name": name,
        "name": name,
        "role": role,
        "seed": 20260808,
        "midi_pitches": list(FIVE_NOTE_PITCHES),
        "sequence_timing_profile": "aligned",
        "note_duration": 0.28,
        "note_gap": 0.12,
        "velocity": 90,
        "key_index_mapping": "key_index = midi_pitch - 21",
        "sequences": items,
    }


def _archetype(role: str, length: int) -> str:
    if length <= 2:
        return role
    if length <= 5:
        return f"{role}_short"
    if length <= 10:
        return f"{role}_medium"
    if length <= 20:
        return f"{role}_long"
    return f"{role}_extrapolation"


def pairwise_manifest_payload() -> dict[str, Any]:
    primitive_items = list(anchors()) + list(ordered_pairs(include_repeats=True))
    sequences = []
    for index, item in enumerate(primitive_items):
        sequences.append(
            {
                "name": f"{item.category}_{index:03d}_{'_'.join(str(pitch) for pitch in item.pitches)}",
                "pitches": list(item.pitches),
                "key_indices": [pitch - 21 for pitch in item.pitches],
                "length": len(item.pitches),
                "archetype": item.category,
                "category": item.category,
                "midi_tuple_hash": item.midi_tuple_hash,
            }
        )
    return {
        "benchmark_name": "complete_pairwise_v1",
        "name": "complete_pairwise_v1",
        "role": "complete_pairwise_eval",
        "seed": 20260808,
        "midi_pitches": list(FIVE_NOTE_PITCHES),
        "sequence_timing_profile": "aligned",
        "note_duration": 0.28,
        "note_gap": 0.12,
        "velocity": 90,
        "key_index_mapping": "key_index = midi_pitch - 21",
        "original_adjacent_pairs": [list(pair) for pair in ORIGINAL_ADJACENT_PAIRS],
        "include_repeated_note_pairs": True,
        "anchor_count": len(anchors()),
        "distinct_ordered_pair_count": len(ordered_pairs(include_repeats=False)),
        "repeated_pair_count": len([item for item in ordered_pairs(include_repeats=True) if item.category == "repeated_note"]),
        "total_ordered_pair_count": len(ordered_pairs(include_repeats=True)),
        "primitives": [
            {
                "name": item.name,
                "pitches": list(item.pitches),
                "key_indices": [pitch - 21 for pitch in item.pitches],
                "category": item.category,
                "midi_tuple_hash": item.midi_tuple_hash,
            }
            for item in primitive_items
        ],
        "sequences": sequences,
    }
