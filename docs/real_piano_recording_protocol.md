# Real Piano Recording Protocol

Record isolated single notes for the final acoustic distribution-shift evaluation.

## Folder

Place raw recordings here:

`/home/reece_dev/msc-audio-pianist/data/real_piano/raw/`

This folder is ignored by Git. Do not commit recordings.

## Required Notes

Record three clean takes for each MIDI pitch:

- MIDI 72, C5
- MIDI 73, C#5
- MIDI 74, D5
- MIDI 75, D#5
- MIDI 76, E5

MIDI-style filenames are accepted:

```text
midi72_take01.wav
midi72_take02.wav
...
midi76_take03.wav
```

The final recorded dataset may also use note-name filenames:

```text
C take 1.wav
C# take 2.wav
D# take 3.wav
```

WAV is preferred. FLAC, M4A, and MP3 are accepted if `ffmpeg` can read them.

## Recording Setup

- Use one piano, one room, one recording device, one device position.
- Record isolated single strikes, not full melodies.
- Let the note decay naturally for at least about one second.
- Avoid clipping: if the waveform clips, lower the recording level and retake.
- Natural modest velocity variation is fine. Do not cherry-pick only perfect takes.
- Keep background noise low, but do not denoise recordings manually.

## August 23 Command

After placing files in the raw folder, run:

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
bash scripts/real_audio/run_real_audio_distribution_shift.sh
```

The command validates recordings, converts them to mono 16 kHz WAV, records hashes/statistics, and constructs deterministic benchmark audio clips from the isolated notes using the existing benchmark timing.

## Evaluation Command

After preparation succeeds, run the evaluation explicitly. This can be done locally for a small subset or on Hex for the full matrix:

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist \
python scripts/real_audio/evaluate_real_audio_distribution_shift.py \
  --prepared-manifest artifacts/real_audio_distribution_shift/prepared_v1/benchmark_realization_manifest.csv \
  --output-dir artifacts/real_audio_distribution_shift/evaluation_v1 \
  --benchmark exact_13_retention \
  --benchmark complete_pairwise \
  --benchmark clean_unseen_compositions \
  --include-audio-interventions
```

Then aggregate:

```bash
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist \
python scripts/real_audio/analyze_real_audio_distribution_shift.py \
  --evaluation-root artifacts/real_audio_distribution_shift/evaluation_v1 \
  --output-dir artifacts/real_audio_distribution_shift/analysis_v1
```

The default full prepared set contains the exact-13 retention sequences, the complete pairwise benchmark, and clean unseen composition probes, each with three deterministic realizations from the isolated-note takes. The generated clips, manifests, evaluations, and analysis outputs live under `artifacts/real_audio_distribution_shift/` and must not be committed.
