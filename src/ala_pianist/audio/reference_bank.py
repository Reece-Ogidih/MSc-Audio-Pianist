"""Raw-audio reference storage for direct audio-to-action policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioReference:
    """A resolved audio clip stored outside the policy observation."""

    clip_id: int
    name: str
    waveform: np.ndarray
    sample_rate: int
    metadata: dict[str, Any]

    @property
    def duration_seconds(self) -> float:
        return float(self.waveform.shape[0] / self.sample_rate)


class AudioReferenceBank:
    """Stores waveform clips and resolves deterministic context windows.

    ``clip_id`` is metadata for storage/replay lookup only. It must not be fed to
    a policy as a learnable feature.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        past_context_seconds: float = 0.10,
        future_context_seconds: float = 0.40,
    ):
        self.sample_rate = int(sample_rate)
        self.past_context_seconds = float(past_context_seconds)
        self.future_context_seconds = float(future_context_seconds)
        self._references: list[AudioReference] = []

    @property
    def window_size(self) -> int:
        total = self.past_context_seconds + self.future_context_seconds
        return int(round(total * self.sample_rate))

    @property
    def past_samples(self) -> int:
        return int(round(self.past_context_seconds * self.sample_rate))

    @property
    def future_samples(self) -> int:
        return self.window_size - self.past_samples

    def add_waveform(
        self,
        waveform,
        *,
        name: str,
        source_sample_rate: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if source_sample_rate != self.sample_rate:
            waveform = _resample_linear(waveform, int(source_sample_rate), self.sample_rate)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        clip_id = len(self._references)
        self._references.append(
            AudioReference(
                clip_id=clip_id,
                name=str(name),
                waveform=waveform.astype(np.float32, copy=False),
                sample_rate=self.sample_rate,
                metadata=dict(metadata or {}),
            )
        )
        return clip_id

    def add_wav(self, path: str | Path, *, name: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        path = Path(path)
        waveform, source_sample_rate = sf.read(path, always_2d=False)
        if np.asarray(waveform).ndim == 2:
            waveform = np.asarray(waveform, dtype=np.float32).mean(axis=1)
        return self.add_waveform(
            waveform,
            name=name or path.stem,
            source_sample_rate=int(source_sample_rate),
            metadata={**dict(metadata or {}), "path": str(path)},
        )

    def reference(self, clip_id: int) -> AudioReference:
        return self._references[int(clip_id)]

    def __len__(self) -> int:
        return len(self._references)

    def context_window(self, *, clip_id: int, center_sample: int) -> np.ndarray:
        reference = self.reference(clip_id)
        center_sample = int(center_sample)
        start = center_sample - self.past_samples
        end = center_sample + self.future_samples
        out = np.zeros(self.window_size, dtype=np.float32)
        src_start = max(0, start)
        src_end = min(reference.waveform.shape[0], end)
        if src_end <= src_start:
            return out
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        out[dst_start:dst_end] = reference.waveform[src_start:src_end]
        return out

    def sample_index_for_time(self, time_seconds: float) -> int:
        return int(round(float(time_seconds) * self.sample_rate))


def _resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if waveform.size == 0 or source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    duration = waveform.shape[0] / float(source_rate)
    target_count = max(1, int(round(duration * target_rate)))
    source_t = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False)
    target_t = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_t, source_t, waveform).astype(np.float32)
