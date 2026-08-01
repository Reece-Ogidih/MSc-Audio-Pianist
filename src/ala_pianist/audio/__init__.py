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

__all__ = [
    "AudioToMidiTranscriber",
    "AudioReference",
    "AudioReferenceBank",
    "BasicPitchTranscriber",
    "GeneratedWavPeakTranscriber",
    "OracleMidiTranscriber",
    "SynthesizedClip",
    "TranscriptionOutput",
    "TranscriptionResult",
    "synthesize_monophonic_wav",
    "transcribe_monophonic_wav",
    "transcription_accuracy",
]
