# Hex Next Experiments

This document prepares the next HexCloud experiments after the first serious DroQ Stage 3c run.

Do not launch these jobs until the current commit is pushed and the Hex repository has been updated.

## Experiments

### A. Fresh SAC 1M Fair Control

Purpose: controlled SAC baseline for the same six-sequence benchmark used by DroQ.

- Run name: `sac_fair_six_sequence_seed13_1m`
- Config: `configs/sac_fair_six_sequence_seed13_1m.json`
- Input checkpoint: none
- Timesteps: 1,000,000 from scratch
- Sequences: `[73]`, `[74]`, `[75]`, `[73,75]`, `[75,73]`, `[74,75]`
- Weights: `0.22,0.22,0.22,0.14,0.10,0.10`
- Reward: existing `transition_cleanup`
- Timing: `aligned`
- Action mode/repeat: `direct`, `1`

### B. Fresh DroQ 300k Cleanliness Pilot

Purpose: test whether a reward that penalises unintended-key duration/travel improves mechanical cleanliness without sacrificing target recall.

- Run name: `droq_cleanliness_sensitive_v1_seed13_300k`
- Config: `configs/droq_cleanliness_sensitive_v1_seed13_300k.json`
- Input checkpoint: none
- Timesteps: 300,000 from scratch
- Reward: new `transition_cleanup_sensitive_v1`
- Checkpoints: every 50k
- Early evaluations: 50k, 100k, 200k, 300k

### C. Five-Note DroQ Expansion From 300k

Purpose: prepare a compact five-note controller around C5-E5 without expanding range unnecessarily.

- Run name: `droq_five_note_expansion_from_300k_seed13`
- Config: `configs/droq_five_note_expansion_from_300k_seed13.json`
- Input checkpoint: required
- Source checkpoint: `checkpoint_300000_steps.pt`
- Source SHA-256: `f1bb6a62804b173d07f897318f9e0381294560cbb82d43a9e03a3a0727bfe390`
- Additional timesteps: 700,000
- Reward: existing `transition_cleanup`, because the replay buffer was collected with that reward
- Status: prepared only; review B before launch

## Files Changed

- `src/ala_pianist/rl/general_one_hand_env.py`
- `src/ala_pianist/evaluation/unintended.py`
- `src/ala_pianist/music/staged_curriculum.py`
- `src/ala_pianist/music/__init__.py`
- `scripts/analyse_unintended_key_trajectories.py`
- `scripts/train_general_one_hand_policy.py`
- `scripts/train_droq_general_one_hand_policy.py`
- `scripts/evaluate_general_one_hand_policy.py`
- `scripts/hex/run_sac_fair_1m.sh`
- `scripts/hex/run_droq_cleanliness_300k.sh`
- `scripts/hex/run_droq_five_note_expansion.sh`
- `scripts/hex/evaluate_sac_fair_checkpoints.sh`
- `scripts/hex/evaluate_cleanliness_checkpoints.sh`
- `scripts/hex/evaluate_five_note_checkpoints.sh`
- `scripts/hex/smoke_test_all_hex_jobs.sh`
- `scripts/hex/verify_hex_inputs.sh`
- `scripts/hex/collect_hex_outputs.sh`
- `configs/*.json` for the three experiments
- `tests/*` covering the new reward/classification/config helpers

## Home Versus Scratch

Home directory:

- Git repository
- `third_party/robopianist/`
- soundfont under RoboPianist source tree
- scripts and configs

Scratch:

- transferred input checkpoints
- generated MIDI
- logs
- replay buffers
- model checkpoints
- evaluation CSV/JSON/plots

Expected scratch storage:

- SAC fair 1M: tens of GB if replay buffers are saved with periodic checkpoints.
- DroQ cleanliness 300k: several GB.
- Five-note expansion: several GB to tens of GB depending on checkpoints.

## Smoke Commands

```bash
export HEX_GPU_INDEX=<gpu-index>
export HEX_SCRATCH=<scratch-path>
export HEX_IMAGE_TAG=<image-tag>
scripts/hex/verify_hex_inputs.sh
scripts/hex/smoke_test_all_hex_jobs.sh
```

Tiny local-style smoke commands inside the container should use `--timesteps 1000` or similar, not the production budgets.

## Long-Run Commands

Run A and B independently on separate Hex nodes/GPUs:

```bash
export HEX_GPU_INDEX=<gpu-index>
export HEX_SCRATCH=<scratch-path>
export HEX_IMAGE_TAG=<image-tag>
scripts/hex/run_sac_fair_1m.sh
```

```bash
export HEX_GPU_INDEX=<gpu-index>
export HEX_SCRATCH=<scratch-path>
export HEX_IMAGE_TAG=<image-tag>
scripts/hex/run_droq_cleanliness_300k.sh
```

Prepare C only after review:

```bash
export HEX_GPU_INDEX=<gpu-index>
export HEX_SCRATCH=<scratch-path>
export HEX_IMAGE_TAG=<image-tag>
export HEX_DROQ_300K_CHECKPOINT=<scratch-path>/incoming/checkpoint_300000_steps.pt
scripts/hex/run_droq_five_note_expansion.sh
```

## Monitoring

```bash
hare me
hare usage
hare ps -a
hare logs -n 120 <run-name>
nvidia-smi
df -h "$HEX_SCRATCH"
```

## Safe Stop

```bash
scripts/hex/stop_run.sh --run-name <run-name> --scratch "$HEX_SCRATCH"
scripts/hex/stop_run.sh --run-name <run-name> --scratch "$HEX_SCRATCH" --confirm-stop
```

## Collection

```bash
export HEX_USER=<bath-username>
export HEX_HOST=<hex-host>
export HEX_REMOTE_RUN_DIR=<scratch-run-dir>
export LOCAL_OUTPUT_DIR=artifacts/hex_runs/<run-name>
scripts/hex/collect_hex_outputs.sh
```

## Resume Interrupted Runs

Find the newest checkpoint in scratch, compute remaining additional timesteps, and resume with the corresponding trainer. Do not use reset-replay-buffer or reset-optimizer options unless deliberately starting a less faithful continuation.
