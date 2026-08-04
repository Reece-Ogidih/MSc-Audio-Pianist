from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np

from ala_pianist.evaluation.final_experiments import (
    COMPOSITIONAL_BENCHMARK_SEQUENCES,
    audit_real_audio_manifest,
    write_real_audio_manifest_template,
)
from ala_pianist.pipelines.indirect import BENCHMARK_SEQUENCE_PITCHES


def test_compositional_sequences_are_held_out_and_in_range() -> None:
    trained = {tuple(seq) for seq in BENCHMARK_SEQUENCE_PITCHES}

    assert 10 <= len(COMPOSITIONAL_BENCHMARK_SEQUENCES) <= 15
    for sequence in COMPOSITIONAL_BENCHMARK_SEQUENCES.values():
        assert tuple(sequence) not in trained
        assert len(sequence) >= 3
        assert all(72 <= pitch <= 76 for pitch in sequence)


def test_real_audio_manifest_template_and_missing_file_audit(tmp_path: Path) -> None:
    manifest = write_real_audio_manifest_template(tmp_path / "manifest.json")

    rows, summary = audit_real_audio_manifest(manifest)

    assert rows[0]["status"] == "missing_file"
    assert summary["recording_count"] == 1
    assert summary["missing_count"] == 1
    assert summary["target_pipeline2_sample_rate"] == 16000


def test_real_audio_manifest_audits_present_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "wav" / "anchor_72_take1.wav"
    wav_path.parent.mkdir()
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        samples = (0.25 * np.sin(np.linspace(0, 2 * np.pi, 1600))).astype(np.float32)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "sequence_name": "anchor_72_take1",
                        "midi_sequence": [72],
                        "wav_path": "wav/anchor_72_take1.wav",
                        "sample_rate": 16000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, summary = audit_real_audio_manifest(manifest)

    assert rows[0]["status"] == "ok"
    assert rows[0]["sample_rate"] == 16000
    assert rows[0]["duration_seconds"] > 0.0
    assert 0.0 < rows[0]["rms"] < rows[0]["peak_abs"] < 1.0
    assert summary["ok_count"] == 1
