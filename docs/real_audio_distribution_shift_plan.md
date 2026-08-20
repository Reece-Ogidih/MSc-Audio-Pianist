# Real-Audio Distribution-Shift Plan

This is the final pre-real-audio technical harness. It does not train models and does not change the frozen Pipeline 1 or Pipeline 2 checkpoints.

## Purpose

Use isolated real-piano recordings for MIDI 72-76 to test whether the existing synthetic-audio conclusions survive a controlled acoustic shift. The benchmark timing and MIDI targets remain identical to the synthetic experiments; only the waveform source changes.

## Inputs

Raw recordings go in `data/real_piano/raw/` and are ignored by Git. Required filenames are `midi72_take01.wav` through `midi76_take05.wav`. WAV is preferred; FLAC, M4A, and MP3 are accepted through `ffmpeg`.

## Preparation

`scripts/real_audio/prepare_real_audio_distribution_shift.py` validates hashes, duration, clipping, silence, and missing takes. It converts every take to mono 16 kHz WAV with conservative attack-preserving silence trimming and peak normalization. It then constructs deterministic benchmark clips from isolated notes using the existing benchmark timing and writes target MIDI sidecars for controller/evaluator use.

Primary outputs:

- `recording_manifest.csv`
- `preprocessing_manifest.csv`
- `benchmark_realization_manifest.csv`
- summary JSON files for each step

## Evaluation Matrix

Primary models:

- Pipeline 1 frozen 800k symbolic controller with Basic Pitch and oracle conditions.
- Pipeline 2 seed13 1M direct-audio controller with correct, zero, and mismatched audio where requested.

Primary benchmark groups:

- exact-13 retention;
- complete pairwise;
- clean unseen compositions.

With three realizations, the full prepared evaluation contains 153 real-audio clips before pipeline/condition expansion.

## Metrics

Use the same metrics as the final synthetic evaluation:

- pressed-key precision/recall/F1;
- timestep precision/recall/F1;
- target event hit/order metrics;
- maximum and integrated unintended key travel;
- wrong presses and strict outcome.

Pipeline 1 additionally reports Basic Pitch note precision/recall/F1 and onset/offset timing errors. Pipeline 2 audio interventions compare behaviour under correct, zero, and mismatched audio.

## Interpretation

Synthetic final-evaluation metrics remain the primary controlled benchmark. Real-audio results should be interpreted as a distribution-shift probe: a drop under real recordings identifies acoustic robustness limits, while stable behaviour supports the claim that the learned controller can consume realistic audio features under this constrained benchmark.

## First Vertical Slice

After recording all required takes:

```bash
bash scripts/real_audio/run_real_audio_distribution_shift.sh
```

For a tiny local smoke after preparation:

```bash
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist \
python scripts/real_audio/evaluate_real_audio_distribution_shift.py \
  --prepared-manifest artifacts/real_audio_distribution_shift/prepared_v1/benchmark_realization_manifest.csv \
  --output-dir artifacts/real_audio_distribution_shift/evaluation_smoke \
  --benchmark exact_13_retention \
  --realization 1 \
  --max-clips 2 \
  --device cpu
```
