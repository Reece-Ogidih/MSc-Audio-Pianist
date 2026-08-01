# Indirect Pipeline Interface v1

## Scope

This interface starts Pipeline 1 as:

`audio waveform/file -> audio-to-MIDI transcriber -> canonical timed notes -> controller goal sequence -> frozen symbolic controller`

Initial constraints:

- right hand only;
- MIDI pitches `72-76`;
- no sustain;
- monophonic benchmark sequences;
- onset and release timing preserved rather than silently quantized.

The selected controller remains the immutable frozen 800k controller:

`artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt`

## Proposed Module Layout

| File | Purpose |
| --- | --- |
| `src/ala_pianist/music/timed_notes.py` | Canonical `TimedNote`, range/confidence/duplicate handling, conversion to controller MIDI events. |
| `src/ala_pianist/audio/transcriber.py` | Model-independent `AudioToMidiTranscriber` protocol plus oracle MIDI and generated-WAV adapters. |
| `src/ala_pianist/evaluation/transcription_metrics.py` | Note-level transcription precision/recall/F1 and timing-error metrics. |
| `src/ala_pianist/pipelines/indirect.py` | Benchmark sequence constants and symbolic-frontend orchestration. |
| `scripts/run_indirect_oracle_vertical_slice.py` | Tiny generated-audio/oracle/predicted frontend smoke command. |
| `tests/test_indirect_pipeline_interfaces.py` | Unit tests for the interface contracts and edge cases. |

## Data Contracts

`TimedNote`

- `pitch`: MIDI pitch integer.
- `onset`: onset time in seconds.
- `offset`: release time in seconds.
- `confidence`: transcriber confidence in `[0, 1]`.
- `source`: source label such as `oracle_midi` or a transcriber name.
- `metadata`: optional source-specific fields.

`TranscriptionOutput`

- `notes`: tuple of canonical timed notes.
- `transcriber_name`: model or adapter identifier.
- `source_audio_path`: audio file used, when applicable.
- `metadata`: transcriber-level metadata.

`ControllerGoalSequence`

- `notes`: normalized canonical timed notes.
- `note_events`: existing `NoteEvent` objects suitable for writing RoboPianist MIDI.
- `midi_min`, `midi_max`: supported local range.
- `timing_preserved`: true when onset/offset times are passed through.
- `monophonic`: true for the initial benchmark.

## Evaluation Design

Report transcription and controller performance separately:

- transcription note precision, recall and F1;
- onset mean absolute error;
- offset mean absolute error;
- duration mean absolute error;
- controller metrics with oracle MIDI;
- controller metrics with predicted MIDI;
- degradation from oracle to predicted symbolic input.

Controller metrics should continue to include pressed-key precision/recall/F1, timestep precision/recall/F1, max/integrated unintended travel, wrong-key counts, strict outcomes and reward-component summaries where available.

## First Benchmark Set

Use the existing 13 five-note anchor/transition sequences:

`[72]`, `[73]`, `[74]`, `[75]`, `[76]`, `[72,73]`, `[73,72]`, `[73,74]`, `[74,73]`, `[74,75]`, `[75,74]`, `[75,76]`, `[76,75]`

The first audio benchmark should render these known MIDI sequences locally so the ground-truth symbolic target is exact. No external dataset is needed yet.

## First Vertical Slice Command

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist \
python scripts/run_indirect_oracle_vertical_slice.py
```

This writes ignored generated files under:

`experiments/indirect_pipeline/oracle_vertical_slice/`

## Implementation Plan

1. Keep the oracle MIDI adapter as the upper-bound symbolic-input condition.
2. Add a renderer/evaluator script that creates all 13 benchmark MIDIs and WAVs.
3. Run the frozen controller twice per sequence: once from oracle MIDI and once from predicted MIDI.
4. Produce an oracle-vs-predicted degradation table.
5. Only after that vertical slice works, substitute a stronger audio-to-MIDI model behind the same `AudioToMidiTranscriber` protocol.

## Five-Note v1 Benchmark Update

The first real rendered-audio benchmark uses Basic Pitch through the same transcriber protocol. In this NumPy 2.2 environment Basic Pitch's TFLite backend is incompatible, so the adapter selects the packaged ONNX model and `onnxruntime`.

The generated benchmark/evaluation artifacts live under:

`experiments/indirect_pipeline/five_note_rendered_benchmark/`

The tracked experiment note is:

`docs/experiments/indirect_pipeline_five_note_v1.md`

## Unresolved Decisions

- Which non-oracle audio-to-MIDI model to evaluate first.
- Whether transcription matching should allow pitch-only matching inside the five-note range or require onset/offset tolerances for every note.
- The exact onset and offset tolerances to use in the MSc evaluation tables.
- Whether low-confidence predictions should be dropped, retained with confidence features, or evaluated at multiple thresholds.
- Whether rendered audio should initially use the current sine/harmonic renderer or a FluidSynth-rendered piano soundfont benchmark.
