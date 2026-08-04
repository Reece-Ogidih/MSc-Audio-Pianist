# Final Audio-to-Action Experimental Phase

This phase studies Pipeline 2 as an Audio-to-Action learning problem rather than only as a head-to-head replacement for Pipeline 1. The frozen Pipeline 1 five-note indirect controller remains unchanged and serves as a comparison instrument.

## Phase-A Artifact Backup

Hex scratch is not backed up. Preserve the direct-audio Phase-A artifacts before new work.

Recommended local destination:

```bash
LOCAL_ROOT=/home/reece_dev/msc-audio-pianist/artifacts/pipeline2_phase_a_hex
mkdir -p "$LOCAL_ROOT"
```

Seed 13 priority transfer:

```bash
HEX_HOST=<hex-host>
HEX_ROOT=/mnt/fast1/rgkgo20/msc-audio-pianist/experiments/pipeline2_direct_audio
RUN=pipeline2_direct_audio_droq_v1_seed13_1m_retry1

rsync -avz --progress "$HEX_HOST:$HEX_ROOT/$RUN/checkpoints/full_checkpoint_1000000_steps.pt" \
  "$LOCAL_ROOT/$RUN/checkpoints/"

rsync -avz --progress "$HEX_HOST:$HEX_ROOT/$RUN/lightweight_checkpoints/checkpoint_1000000_steps.pt" \
  "$LOCAL_ROOT/$RUN/lightweight_checkpoints/"

rsync -avz --progress "$HEX_HOST:$HEX_ROOT/$RUN/evaluation/" \
  "$LOCAL_ROOT/$RUN/evaluation/"
```

Seed 37 priority transfer:

```bash
HEX_HOST=<hex-host>
HEX_ROOT=/mnt/fast1/rgkgo20/msc-audio-pianist/experiments/pipeline2_direct_audio
RUN=pipeline2_direct_audio_droq_v1_seed37_1m_retry1

rsync -avz --progress "$HEX_HOST:$HEX_ROOT/$RUN/checkpoints/full_checkpoint_1000000_steps.pt" \
  "$LOCAL_ROOT/$RUN/checkpoints/"

rsync -avz --progress "$HEX_HOST:$HEX_ROOT/$RUN/evaluation/" \
  "$LOCAL_ROOT/$RUN/evaluation/"
```

Optional complete lightweight checkpoint transfer:

```bash
rsync -avz --progress "$HEX_HOST:$HEX_ROOT/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/" \
  "$LOCAL_ROOT/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/"
rsync -avz --progress "$HEX_HOST:$HEX_ROOT/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/lightweight_checkpoints/" \
  "$LOCAL_ROOT/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/lightweight_checkpoints/"
```

SHA-256 verification:

```bash
find "$LOCAL_ROOT" -type f \( -name "*.pt" -o -name "*.csv" -o -name "*.json" \) -print0 \
  | sort -z | xargs -0 sha256sum > "$LOCAL_ROOT/SHA256SUMS.local"

ssh "$HEX_HOST" "cd '$HEX_ROOT' && find pipeline2_direct_audio_droq_v1_seed13_1m_retry1 pipeline2_direct_audio_droq_v1_seed37_1m_retry1 -type f \( -name '*.pt' -o -name '*.csv' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum" \
  > "$LOCAL_ROOT/SHA256SUMS.hex"
```

Expected sizes are approximately 1.2-1.8 GB per full 1M run because full checkpoints include indexed replay. Lightweight actor checkpoints and evaluation CSV/JSON outputs should be comparatively small.

## Visual Rollout And Motion Audit

Pipeline 2 rollout rendering uses correct canonical audio only, deterministic actor inference, the same RoboPianist one-hand task and control timestep, and MuJoCo `camera_id=0`, matching the earlier Pipeline 1 smoke renderer where possible.

Default montage sequences:

- `[72]`
- `[76]`
- `[72,73]`
- `[73,72]`
- `[74,75]`
- `[76,75]`

Local command:

```bash
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist \
python scripts/render_pipeline2_direct_audio_rollouts.py \
  --checkpoint /home/reece_dev/msc-audio-pianist/artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt \
  --output-dir /home/reece_dev/msc-audio-pianist/artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/rollout_audit \
  --device cpu
```

Hex command inside the container:

```bash
PYTHONPATH=/app/src:/app/third_party/robopianist \
python scripts/render_pipeline2_direct_audio_rollouts.py \
  --checkpoint /workspace/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt \
  --output-dir /workspace/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/rollout_audit \
  --device cuda
```

Motion metrics include mean absolute action delta, action saturation fraction, fingertip velocity, acceleration, jerk, p95 jerk, and per-action-dimension statistics. They are compatible with existing Pipeline 1 trace-based motion helpers when the trace columns match.

## Compositional Generalisation Benchmark

The zero-shot composition benchmark uses only the trained pitch vocabulary `72..76` and excludes the five anchors and eight two-note transitions used in Phase-A training.

| Sequence name | MIDI sequence | Category |
|---|---:|---|
| adjacent_up_72_73_74 | 72-73-74 | three_note_adjacent_run |
| adjacent_down_74_73_72 | 74-73-72 | three_note_reversed_run |
| adjacent_up_74_75_76 | 74-75-76 | three_note_adjacent_run |
| adjacent_down_76_75_74 | 76-75-74 | three_note_reversed_run |
| jump_return_low | 72-74-73 | three_note_nonadjacent_jump |
| jump_return_mid | 73-75-74 | three_note_nonadjacent_jump |
| jump_return_high | 76-74-75 | three_note_nonadjacent_jump |
| four_note_walk_up | 72-73-74-75 | four_note_sequence |
| four_note_walk_down | 76-75-74-73 | four_note_sequence |
| four_note_mixed_low | 72-73-75-74 | four_note_sequence |
| four_note_mixed_high | 76-74-75-73 | four_note_sequence |
| five_note_span_walk | 72-73-74-75-76 | five_note_sequence |
| five_note_nonmonotonic | 72-76-74-73-75 | five_note_sequence |

Pipeline 2 command:

```bash
PYTHONPATH=/app/src:/app/third_party/robopianist \
python scripts/evaluate_pipeline2_compositional.py \
  --checkpoint /workspace/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt \
  --output-dir /workspace/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/compositional_evaluation \
  --device cuda \
  --include-audio-interventions
```

Pipeline 1 should be evaluated on the same WAV files through Basic Pitch and the frozen controller. Its outputs should remain separate from Pipeline 2 metrics to avoid contaminating the direct-policy evaluator.

## Real-Piano Audio Protocol

Manifest schema:

```json
{
  "schema": "ala_pianist_real_audio_manifest_v1",
  "target_sample_rate": 16000,
  "normalisation_policy": "mono average, deterministic resample to 16 kHz, preserve timing, no pitch extraction for Pipeline 2",
  "recordings": [
    {
      "sequence_name": "anchor_72_take1",
      "midi_sequence": [72],
      "wav_path": "wav/anchor_72_take1.wav",
      "sample_rate": null,
      "take_id": "take1",
      "recording_device": "",
      "notes": ""
    }
  ]
}
```

Audit command:

```bash
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src \
python scripts/audit_real_audio_manifest.py \
  --manifest /home/reece_dev/msc-audio-pianist/data/real_audio/five_note_benchmark/manifest.json \
  --output-dir /home/reece_dev/msc-audio-pianist/data/real_audio/five_note_benchmark/audit
```

Preferred recording protocol:

- Record the 13 Phase-A benchmark sequences plus selected held-out compositions.
- Use MIDI pitches 72-76 only.
- Use the aligned timing target: 0.28 s note duration and 0.12 s gaps where humanly practical.
- Record 2-3 takes per sequence if time allows.
- Keep the same room, instrument, microphone, device gain and distance.
- Avoid post-processing except trimming gross silence if documented.
- Do not apply pitch correction, compression, denoising, tempo correction or artificial sustain.
- Store raw WAVs and a manifest; evaluator handles mono conversion and deterministic 16 kHz resampling.

MAESTRO and MAPS are useful optional supplementary natural-piano tests, but neither is a clean replacement for controlled recordings of the exact 13-sequence benchmark. They include broader repertoire, polyphony, sustain, expressive timing, variable recording conditions and notes outside the trained local range. They are better suited to later stress tests after the controlled benchmark is established.

## Seed Failure Diagnostic

Command:

```bash
PYTHONPATH=/app/src:/app/third_party/robopianist \
python scripts/diagnose_pipeline2_seed_failure.py \
  --seed13-checkpoint /workspace/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt \
  --seed37-checkpoint /workspace/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt \
  --output-dir /workspace/experiments/pipeline2_direct_audio/seed13_vs_seed37_diagnostics \
  --device cuda
```

The diagnostic reports audio latent variance, correct-vs-zero and correct-vs-mismatch latent/action differences, branch weight norms and action saturation. It is descriptive, not causal proof.

## Prepared Next Training Jobs

Fresh seed 61, 0 to 1M:

```bash
cd /homes/rgkgo20/msc-audio-pianist
nohup hare run --rm --gpus "device=0" \
  --name pipeline2_seed61_1m \
  --user "$(id -u):$(id -g)" \
  -v /homes/rgkgo20/msc-audio-pianist:/app \
  -v /mnt/fast1/rgkgo20/msc-audio-pianist:/workspace \
  --workdir /app \
  rgkgo20/msc-audio-pianist:five-note-b5be771-20260723 \
  bash -lc 'PROJECT_ROOT=/app SCRATCH_ROOT=/workspace DEVICE=cuda bash scripts/hex/run_pipeline2_direct_audio_seed61_phase_a.sh' \
  > /mnt/fast1/rgkgo20/msc-audio-pianist/pipeline2_seed61_1m_hare.log 2>&1 &
```

Seed 13 true resume, 1M to 1.5M:

```bash
cd /homes/rgkgo20/msc-audio-pianist
nohup hare run --rm --gpus "device=1" \
  --name pipeline2_seed13_resume_1m_to_1p5m \
  --user "$(id -u):$(id -g)" \
  -v /homes/rgkgo20/msc-audio-pianist:/app \
  -v /mnt/fast1/rgkgo20/msc-audio-pianist:/workspace \
  --workdir /app \
  rgkgo20/msc-audio-pianist:five-note-b5be771-20260723 \
  bash -lc 'PROJECT_ROOT=/app OUTPUT_ROOT=/workspace/experiments/pipeline2_direct_audio DEVICE=cuda bash scripts/hex/resume_pipeline2_seed13_to_1p5m.sh' \
  > /mnt/fast1/rgkgo20/msc-audio-pianist/pipeline2_seed13_resume_1m_to_1p5m_hare.log 2>&1 &
```

The resume job must start from the full 1M checkpoint, not the lightweight actor checkpoint. It restores actor, critics, target critics, optimizers, alpha state, replay buffer and RNG where present in the checkpoint.
