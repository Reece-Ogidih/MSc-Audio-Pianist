# Basic Pitch Diagnosis

## Root cause

All 76 refined transcription rows contain the same caught exception: `RuntimeError: Basic Pitch is not available. Install basic-pitch with a supported inference runtime.` The evaluator imported the project adapter, but `BasicPitchTranscriber.transcribe()` failed while importing `basic_pitch.inference`. The Hex requirements did not contain `basic-pitch` or an inference backend. `_safe_transcribe` intentionally converted that exception into metadata and an empty prediction, after which the evaluator wrote zero-placeholder controller rows.

This was not caused by missing WAV files, FluidSynth, the soundfont, or the refined P1 checkpoint.

## Local status

The `pianist` Conda environment contains:

- Basic Pitch 0.4.0;
- ONNX Runtime 1.23.2;
- librosa 0.11.0;
- mir-eval 0.8.2;
- resampy 0.4.2.

A read-only local transcription of the existing `anchor_72.wav` completed through the bundled ONNX model and returned timed MIDI notes. TensorFlow and CoreML warnings are expected because this project deliberately uses ONNX.

## Hex repair

The Docker requirements now pin ONNX Runtime and Basic Pitch's supporting runtime packages. The Dockerfile installs `basic-pitch==0.4.0` with `--no-deps` after those dependencies, avoiding an unnecessary TensorFlow Lite installation. Its build assertion imports `basic_pitch.inference` and requires the package's `.onnx` model file to exist.

## Minimal rerun

Use `--pipeline1-condition transcribed --skip-pipeline2`. This loads the refined symbolic controller and evaluates only predicted MIDI; it does not repeat Oracle or Pipeline 2. Outputs are written under the new `basic_pitch/` repair directory.

The four required batteries contain 68 sequences total: 30 pairwise, 20 frozen long-horizon, 8 clean, and 10 extrapolation. The eight-sequence validation battery is optional and omitted initially.
