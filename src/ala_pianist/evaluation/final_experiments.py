"""Utilities for final Audio-to-Action experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable
import wave

import numpy as np


COMPOSITIONAL_BENCHMARK_SEQUENCES: dict[str, tuple[int, ...]] = {
    "adjacent_up_72_73_74": (72, 73, 74),
    "adjacent_down_74_73_72": (74, 73, 72),
    "adjacent_up_74_75_76": (74, 75, 76),
    "adjacent_down_76_75_74": (76, 75, 74),
    "jump_return_low": (72, 74, 73),
    "jump_return_mid": (73, 75, 74),
    "jump_return_high": (76, 74, 75),
    "four_note_walk_up": (72, 73, 74, 75),
    "four_note_walk_down": (76, 75, 74, 73),
    "four_note_mixed_low": (72, 73, 75, 74),
    "four_note_mixed_high": (76, 74, 75, 73),
    "five_note_span_walk": (72, 73, 74, 75, 76),
    "five_note_nonmonotonic": (72, 76, 74, 73, 75),
}


COMPOSITIONAL_CATEGORIES: dict[str, str] = {
    "adjacent_up_72_73_74": "three_note_adjacent_run",
    "adjacent_down_74_73_72": "three_note_reversed_run",
    "adjacent_up_74_75_76": "three_note_adjacent_run",
    "adjacent_down_76_75_74": "three_note_reversed_run",
    "jump_return_low": "three_note_nonadjacent_jump",
    "jump_return_mid": "three_note_nonadjacent_jump",
    "jump_return_high": "three_note_nonadjacent_jump",
    "four_note_walk_up": "four_note_sequence",
    "four_note_walk_down": "four_note_sequence",
    "four_note_mixed_low": "four_note_sequence",
    "four_note_mixed_high": "four_note_sequence",
    "five_note_span_walk": "five_note_sequence",
    "five_note_nonmonotonic": "five_note_sequence",
}


@dataclass(frozen=True)
class RealAudioManifestEntry:
    sequence_name: str
    midi_sequence: tuple[int, ...]
    wav_path: Path
    sample_rate: int | None = None
    take_id: str | None = None
    recording_device: str | None = None
    notes: str | None = None

    @property
    def group(self) -> str:
        if len(self.midi_sequence) == 1:
            return "anchor"
        if len(self.midi_sequence) == 2:
            return "transition"
        return "composition"


def load_real_audio_manifest(path: str | Path) -> list[RealAudioManifestEntry]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["recordings"] if isinstance(payload, dict) else payload
    out = []
    for item in entries:
        out.append(
            RealAudioManifestEntry(
                sequence_name=str(item["sequence_name"]),
                midi_sequence=tuple(int(pitch) for pitch in item["midi_sequence"]),
                wav_path=(path.parent / item["wav_path"]).resolve()
                if not Path(item["wav_path"]).is_absolute()
                else Path(item["wav_path"]),
                sample_rate=None if item.get("sample_rate") is None else int(item["sample_rate"]),
                take_id=item.get("take_id"),
                recording_device=item.get("recording_device"),
                notes=item.get("notes"),
            )
        )
    return out


def audit_real_audio_manifest(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for entry in load_real_audio_manifest(path):
        row: dict[str, Any] = {
            "sequence_name": entry.sequence_name,
            "midi_sequence": "-".join(str(pitch) for pitch in entry.midi_sequence),
            "group": entry.group,
            "wav_path": str(entry.wav_path),
            "exists": entry.wav_path.exists(),
            "manifest_sample_rate": entry.sample_rate,
            "take_id": entry.take_id or "",
            "recording_device": entry.recording_device or "",
            "status": "ok",
        }
        if not entry.wav_path.exists():
            row["status"] = "missing_file"
            rows.append(row)
            continue
        try:
            stats = wav_stats(entry.wav_path)
            row.update(stats)
            if entry.sample_rate is not None and int(entry.sample_rate) != int(stats["sample_rate"]):
                row["status"] = "sample_rate_mismatch"
            if stats["peak_abs"] >= 0.999:
                row["clipping_warning"] = True
        except Exception as exc:
            row["status"] = "invalid_wav"
            row["error"] = str(exc)
        rows.append(row)
    summary = {
        "manifest_path": str(path),
        "recording_count": len(rows),
        "missing_count": sum(row["status"] == "missing_file" for row in rows),
        "invalid_count": sum(row["status"] == "invalid_wav" for row in rows),
        "sample_rate_mismatch_count": sum(row["status"] == "sample_rate_mismatch" for row in rows),
        "ok_count": sum(row["status"] == "ok" for row in rows),
        "target_pipeline2_sample_rate": 16000,
        "normalisation_policy": "mono average for multi-channel audio, deterministic resampling to 16 kHz, peak normalise only if absolute peak exceeds 1.0",
    }
    return rows, summary


def wav_stats(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_rate = int(handle.getframerate())
        frames = int(handle.getnframes())
        sample_width = int(handle.getsampwidth())
        raw = handle.readframes(frames)
    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        data = np.zeros(frames * channels, dtype=np.float32)
    if channels > 1 and data.size:
        data = data.reshape(-1, channels).mean(axis=1)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "duration_seconds": frames / sample_rate if sample_rate else 0.0,
        "peak_abs": peak,
        "rms": rms,
    }


def write_real_audio_manifest_template(path: str | Path) -> Path:
    path = Path(path)
    payload = {
        "schema": "ala_pianist_real_audio_manifest_v1",
        "target_sample_rate": 16000,
        "normalisation_policy": "mono average, deterministic resample to 16 kHz, preserve timing, no pitch extraction for Pipeline 2",
        "recordings": [
            {
                "sequence_name": "anchor_72_take1",
                "midi_sequence": [72],
                "wav_path": "wav/anchor_72_take1.wav",
                "sample_rate": None,
                "take_id": "take1",
                "recording_device": "",
                "notes": "",
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
