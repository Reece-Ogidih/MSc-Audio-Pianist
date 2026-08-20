"""Audio utilities for Pipeline 1 diagnostics."""

from ala_pianist.audio.synthesis import SynthesizedClip, synthesize_monophonic_wav
from ala_pianist.audio.reference_bank import AudioReference, AudioReferenceBank
from ala_pianist.audio.transcription import (
    TranscriptionResult,
    transcribe_monophonic_wav,
    transcription_accuracy,
)
from ala_pianist.audio.transcriber import (
    AudioToMidiTranscriber,
    BasicPitchTranscriber,
    GeneratedWavPeakTranscriber,
    OracleMidiTranscriber,
    TranscriptionOutput,
)
from ala_pianist.audio.real_piano import (
    DEFAULT_PITCHES as REAL_PIANO_DEFAULT_PITCHES,
    DEFAULT_TAKES_PER_PITCH as REAL_PIANO_DEFAULT_TAKES_PER_PITCH,
    TARGET_SAMPLE_RATE as REAL_PIANO_TARGET_SAMPLE_RATE,
    construct_real_audio_benchmarks,
    preprocess_recordings,
    validate_raw_recordings,
)

__all__ = [
    "AudioToMidiTranscriber",
    "AudioReference",
    "AudioReferenceBank",
    "BasicPitchTranscriber",
    "GeneratedWavPeakTranscriber",
    "OracleMidiTranscriber",
    "REAL_PIANO_DEFAULT_PITCHES",
    "REAL_PIANO_DEFAULT_TAKES_PER_PITCH",
    "REAL_PIANO_TARGET_SAMPLE_RATE",
    "SynthesizedClip",
    "TranscriptionOutput",
    "TranscriptionResult",
    "construct_real_audio_benchmarks",
    "preprocess_recordings",
    "synthesize_monophonic_wav",
    "transcribe_monophonic_wav",
    "transcription_accuracy",
    "validate_raw_recordings",
]
