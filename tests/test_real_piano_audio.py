from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ala_pianist.audio.real_piano import (
    construct_real_audio_benchmarks,
    discover_raw_recordings,
    preprocess_recordings,
    validate_raw_recordings,
)


def _write_note(path: Path, sample_rate: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * 0.35), dtype=np.float32) / sample_rate
    wave = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(path, wave, sample_rate)


def test_real_piano_validation_detects_required_takes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_note(raw / "midi72_take01.wav")

    rows, summary = validate_raw_recordings(raw)

    assert summary["ready"] is False
    assert any(row["status"] == "missing_takes" for row in rows)


def test_note_name_recording_filenames_map_to_fixed_c5_e5_range(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_note(raw / "C take 1.wav")
    _write_note(raw / "C# take 2.wav")
    _write_note(raw / "D# take 3.wav")
    _write_note(raw / "D take 1.wav")
    _write_note(raw / "E take 1.wav")

    recordings = discover_raw_recordings(raw)
    by_name_take = {(recording.note_name, recording.take): recording.pitch for recording in recordings}

    assert by_name_take[("C", 1)] == 72
    assert by_name_take[("C#", 2)] == 73
    assert by_name_take[("D", 1)] == 74
    assert by_name_take[("D#", 3)] == 75
    assert by_name_take[("E", 1)] == 76
    assert all(recording.filename_style == "note_name" for recording in recordings)


def test_preprocess_and_construct_benchmark_clips(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for note_name in ("C", "C#", "D", "D#", "E"):
        for take in range(1, 4):
            _write_note(raw / f"{note_name} take {take}.wav")

    rows, summary = validate_raw_recordings(raw)
    assert summary["ready"] is True
    processed_rows, processed_summary = preprocess_recordings(raw, tmp_path / "processed")
    assert processed_summary["processed_count"] == 15
    assert all(row["processed_sample_rate"] == 16000 for row in processed_rows)

    manifest = tmp_path / "mini_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark_name": "mini_real_audio_test",
                "seed": 123,
                "midi_pitches": [72, 73],
                "note_duration": 0.28,
                "note_gap": 0.12,
                "velocity": 90,
                "sequences": [
                    {"name": "anchor_72", "pitches": [72], "archetype": "anchor", "length": 1},
                    {"name": "pair_72_73", "pitches": [72, 73], "archetype": "pair", "length": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    benchmark_rows, benchmark_summary = construct_real_audio_benchmarks(
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "benchmark",
        manifests=(manifest,),
        realizations=2,
        seed=123,
    )

    assert benchmark_summary["constructed_clip_count"] == 4
    assert all(Path(row["wav_path"]).is_file() for row in benchmark_rows)
    assert all(Path(row["midi_path"]).is_file() for row in benchmark_rows)
    assert all(row["sample_rate"] == 16000 for row in benchmark_rows)
    assert all(row["midi_sha256"] for row in benchmark_rows)
