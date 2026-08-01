# Indirect Pipeline Five-Note v1

## Decision Context

No further RL/controller refinement is part of this stage. The downstream symbolic controller is frozen:

`artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt`

Pipeline 1 is considered architecturally complete when rendered audio for the full five-note range can pass through a real pretrained audio-to-MIDI model, the canonical timed-note interface, the frozen controller, and RoboPianist action execution with quantitative oracle-vs-predicted evaluation.

## Architecture

`rendered piano audio -> Basic Pitch -> TimedNote -> ControllerGoalSequence -> frozen DroQ controller -> RoboPianist`

The oracle condition uses:

`ground-truth MIDI -> OracleMidiTranscriber -> TimedNote -> ControllerGoalSequence -> frozen DroQ controller -> RoboPianist`

Both conditions converge onto `ControllerGoalSequence` before controller evaluation.

## Benchmark

Range:

- MIDI pitches `72-76`
- RoboPianist key indices `51-55`
- mapping: `key_index = midi_pitch - 21`
- right hand only
- no sustain
- monophonic reference sequences
- aligned timing: note duration `0.28s`, gap `0.12s`

Sequences:

| Name | MIDI pitches |
| --- | --- |
| `anchor_72` | `[72]` |
| `anchor_73` | `[73]` |
| `anchor_74` | `[74]` |
| `anchor_75` | `[75]` |
| `anchor_76` | `[76]` |
| `transition_72_73` | `[72, 73]` |
| `transition_73_72` | `[73, 72]` |
| `transition_73_74` | `[73, 74]` |
| `transition_74_73` | `[74, 73]` |
| `transition_74_75` | `[74, 75]` |
| `transition_75_74` | `[75, 74]` |
| `transition_75_76` | `[75, 76]` |
| `transition_76_75` | `[76, 75]` |

## Rendering

Audio is rendered from exact MIDI using FluidSynth and the existing RoboPianist soundfont:

`third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2`

Output location:

`experiments/indirect_pipeline/five_note_rendered_benchmark/`

Generated WAV, MIDI, CSV and JSON files are ignored experiment artifacts.

## Transcriber

Chosen transcriber:

- package: `basic-pitch==0.4.0`
- backend: ONNX model `nmp.onnx`
- runtime: `onnxruntime==1.23.2`

Basic Pitch's default TFLite backend was not used because `tflite-runtime==2.14.0` is compiled against NumPy 1.x and fails in this environment with NumPy `2.2.6` (`_ARRAY_API not found`). The ONNX backend avoids this failure without downgrading NumPy or touching RoboPianist/PyTorch.

## Timing and Matching

Primary transcription note matching uses:

- exact MIDI pitch match;
- onset tolerance `0.05s`;
- no offset gate for the primary note F1;
- offset and duration errors are still reported for matched notes.

Sensitivity fields also report:

- strict with-offset F1 using offset tolerance `0.10s`;
- loose onset F1 using onset tolerance `0.10s`.

Controller timing is discretised by RoboPianist control steps. The benchmark records the control timestep and a maximum timing-quantisation error for each controller MIDI sequence instead of silently hiding discretisation.

## Results Summary

Pipeline 1 architectural completion:

`true`

Performance assessment:

`usable baseline`

Raw benchmark runtime:

`28.12s` for the no-rerender benchmark run.

Final raw-vs-cleaned comparison runtime:

`43.44s`

Final selected Pipeline 1 v1 path:

`Basic Pitch RAW`

### Transcription Summary

| Metric | Mean |
| --- | ---: |
| onset-matched note precision | 0.590 |
| onset-matched note recall | 1.000 |
| onset-matched note F1 | 0.736 |
| with-offset note F1 | 0.000 |
| onset MAE | 0.0097s |
| offset MAE | 0.3266s |
| duration MAE | 0.3201s |
| predicted notes per sequence | 2.692 |
| reference notes per sequence | 1.615 |
| false positives per sequence | 1.077 |
| false negatives per sequence | 0.000 |

Interpretation: Basic Pitch detected all target pitches with good onset timing, but systematically over-extended note offsets and produced late duplicate notes from the rendered tail. This is a transcription/post-processing limitation, not a controller failure.

### Controller Summary

| Condition | Pressed-key F1 mean | Timestep F1 mean | Max unintended mean |
| --- | ---: | ---: | ---: |
| Oracle MIDI | 0.782 | 0.758 | 0.328 |
| Basic Pitch predicted MIDI | 0.723 | 0.655 | 0.482 |
| Predicted - oracle | -0.059 | -0.103 | +0.154 |

The predicted condition still drives RoboPianist through the frozen controller for every benchmark sequence. Performance degrades because predicted duplicate/long notes create extra or delayed controller goals.

## Per-Sequence Notes

Predicted-controller anchors:

- `[72]`, `[73]`, `[75]`, `[76]` reached pressed-key F1 `1.0`.
- `[74]` reached recall `1.0` but was dirty, pressing keys `52-53`.

Predicted-controller transitions:

- all transitions executed through the full audio-to-action path;
- transition pressed-key F1 ranged from `0.4` to `0.667`;
- most transition failures were dirty or incomplete relative to the reference target set.

Worst predicted transition:

- `transition_74_75`: pressed-key F1 `0.4`, pressed keys `52-53-55`, max unintended `1.0`.

## Error Decomposition

Supported by the data:

1. Transcription quality is onset-useful but offset-weak: recall `1.0`, F1 `0.736`, with-offset F1 `0.0`.
2. The frozen controller's oracle symbolic performance is imperfect but functional: pressed-key F1 mean `0.782`.
3. Predicted symbolic goals reduce pressed-key F1 to `0.723` and timestep F1 to `0.655`.
4. The predicted-vs-oracle degradation is therefore measurable but not catastrophic for architectural validation.

Tentative interpretation:

- duplicate late Basic Pitch notes and long offsets likely drive the extra controller timing degradation;
- cleaner note-off post-processing should be tested later, but not tuned using controller outcomes as a hidden validation loop.

## Final Cleanup Comparison

A final bounded post-processing pass was evaluated before freezing Pipeline 1. The cleanup is deliberately minimal and model-independent:

- keep the existing MIDI range filter `72-76`;
- keep confidence threshold `0.3`;
- discard invalid zero/negative-duration notes;
- suppress later same-pitch duplicate predictions while preserving the earliest onset;
- truncate monophonic overlaps at the next predicted onset;
- do not inspect reference MIDI, controller results, or oracle outcomes.

Raw Basic Pitch failure diagnostics across the 13-sequence benchmark:

- same-pitch duplicate false positives: mean `1.077` per sequence, max `2`;
- raw predicted notes: mean `2.692` vs reference mean `1.615`;
- raw false positives: mean `1.077`;
- raw false negatives: `0`;
- raw overlaps beyond next predicted onset: `0`;
- raw offsets beyond next reference onset: mean `0.615`;
- offset/duration errors remained dominated by long note tails.

Transcription effect:

| Metric | Raw | Cleaned | Delta |
| --- | ---: | ---: | ---: |
| onset-matched precision | 0.590 | 1.000 | +0.410 |
| onset-matched recall | 1.000 | 1.000 | +0.000 |
| onset-matched F1 | 0.736 | 1.000 | +0.264 |
| onset MAE | 0.0097s | 0.0097s | +0.0000s |
| offset MAE | 0.3266s | 0.3266s | +0.0000s |
| duration MAE | 0.3201s | 0.3201s | +0.0000s |
| with-offset F1 @ 0.10s | 0.000 | 0.000 | +0.000 |
| duplicate count | 1.077 | 0.000 | -1.077 |
| false positives | 1.077 | 0.000 | -1.077 |

Controller effect:

| Condition | Pressed-key precision | Pressed-key recall | Pressed-key F1 | Timestep F1 | Max unintended |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle MIDI | 0.962 | 0.692 | 0.782 | 0.758 | 0.328 |
| Raw Basic Pitch | 0.833 | 0.692 | 0.723 | 0.655 | 0.482 |
| Cleaned Basic Pitch | 0.833 | 0.692 | 0.723 | 0.655 | 0.482 |
| Cleaned - raw | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

Per-sequence controller deltas were all zero for pressed-key F1, timestep F1 and max unintended key state. Cleanup improved the symbolic note-count representation but did not improve the target timing/note-off failure mode and did not improve downstream controller behaviour. Under the predefined decision rule, CLEANED is rejected because offset/duration quality did not improve meaningfully. Pipeline 1 v1 is therefore frozen as Basic Pitch RAW plus the existing canonical TimedNote conversion.

## Output Artifacts

Main generated files:

- `experiments/indirect_pipeline/five_note_rendered_benchmark/benchmark_manifest.json`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/transcription_predictions.json`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/transcription_per_sequence.csv`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/transcription_summary.json`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/controller_oracle_per_sequence.csv`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/controller_predicted_per_sequence.csv`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/controller_comparison.csv`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/end_to_end_summary.json`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/pipeline_1_v1_report.md`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/final_cleanup_comparison/raw_vs_cleaned_transcription_per_sequence.csv`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/final_cleanup_comparison/raw_vs_cleaned_transcription_summary.json`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/final_cleanup_comparison/oracle_raw_cleaned_controller_per_sequence.csv`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/final_cleanup_comparison/oracle_raw_cleaned_controller_summary.json`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/final_cleanup_comparison/cleanup_diagnostics.json`
- `experiments/indirect_pipeline/five_note_rendered_benchmark/final_cleanup_comparison/pipeline_1_v1_final_report.md`
- `artifacts/frozen_models/indirect_pipeline_five_note_v1/manifest.json`

## Exact Command

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist \
python scripts/run_indirect_five_note_benchmark.py \
  --transcriber basic_pitch \
  --midi-min 72 \
  --midi-max 76 \
  --evaluate-transcription \
  --evaluate-controller \
  --controller-checkpoint /home/reece_dev/msc-audio-pianist/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt \
  --output-dir /home/reece_dev/msc-audio-pianist/experiments/indirect_pipeline/five_note_rendered_benchmark \
  --confidence-threshold 0.3 \
  --onset-tolerance 0.05 \
  --offset-tolerance 0.10
```

Add `--render-audio` to regenerate the FluidSynth WAV/MIDI benchmark.

Final cleanup-comparison command:

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist \
python scripts/run_indirect_cleanup_comparison.py \
  --benchmark-dir /home/reece_dev/msc-audio-pianist/experiments/indirect_pipeline/five_note_rendered_benchmark \
  --output-dir /home/reece_dev/msc-audio-pianist/experiments/indirect_pipeline/five_note_rendered_benchmark/final_cleanup_comparison \
  --controller-checkpoint /home/reece_dev/msc-audio-pianist/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt \
  --transcriber basic_pitch \
  --confidence-threshold 0.3 \
  --onset-tolerance 0.05 \
  --offset-tolerance 0.10
```

## Limitations Before Freezing Pipeline 1

- The cleanup comparison documented duplicate handling, but RAW remains selected because the cleanup did not improve offset/duration quality or controller behaviour.
- Only one transcriber threshold was used for the reported run.
- The frozen controller remains mechanically imperfect under oracle MIDI, so some end-to-end errors are controller-side rather than transcription-side.
- No external dataset has been used yet, by design.

## Freeze Status

`pipeline_1_architecturally_complete`: `true`

`pipeline_1_frozen_for_comparison`: `true`

This does not mean the pipeline is perfect. It means Pipeline 1 v1 is now the fixed indirect baseline for later comparison with Pipeline 2.
