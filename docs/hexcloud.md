# HexCloud DroQ Continuation Workflow

This document prepares the local ALA Pianist DroQ Stage 3c continuation for a later Hex deployment. It is command-oriented but unvalidated on Hex until the deployment prompt. Do not treat any Hare/GPU command here as tested on Hex yet.

## Current Experiment State

- Source checkpoint: `experiments/general_one_hand/droq/checkpoints/droq_stage3c_fair_1m_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_300000_steps.pt`
- Saved global timestep: `300000`
- Production continuation: `700000` additional timesteps
- Final intended global timestep: `1000000`
- Training script: `scripts/train_droq_general_one_hand_policy.py`
- Resume flag: `--resume-checkpoint`
- Additional-step flag: `--additional-timesteps`

`700000` means additional timesteps, not a final timestep. The continuation must preserve global checkpoint numbering and finish at `checkpoint_1000000_steps.pt`.

## Hex Principles From The Local PDFs

- Open the live Hex Usage page at deployment time. The saved `hex_docs/Hex _ Usage.pdf` is only a snapshot.
- Choose only a node whose access includes `Everyone` or `MSc`.
- Select a currently free GPU from the live Usage page and confirm with `nvidia-smi` after login.
- Scratch storage is node-local and usually under `/mnt/fast*` or `/mnt/faster*`.
- Docker/Hare images are node-local.
- The home directory is shared but unsuitable for high-throughput experiment data.
- Scratch must be reserved before Hare mounts it.
- Code and data should be mounted into containers, not embedded into the image.
- Only the selected GPU should be exposed with `--gpus device=<GPU_INDEX>`.
- The original 300k checkpoint must remain preserved.
- Back up valuable checkpoints outside Hex because Hex is not a backup service.

## Local Files Prepared

- Docker definition: `docker/hex/Dockerfile`
- Docker context ignore file: `docker/hex/.dockerignore`
- Dependency pins: `docker/hex/requirements.txt`
- Helper scripts: `scripts/hex/*.sh`

The Docker image contains dependencies only. The repository is mounted at `/app`; scratch/experiment storage is mounted at `/workspace`.

## Step 1: Choose A Node And GPU Later

On the day of deployment:

1. Open the live Hex Usage page.
2. Choose a node whose access includes `Everyone` or `MSc`.
3. Pick one GPU with low utilisation and enough free memory.
4. Pick a scratch drive with enough free space, preferably `/mnt/fast*` or `/mnt/faster*`.

The Usage PDF in this repo is not live and must not be used for final availability decisions.

## Step 2: Test SSH Manually

Unvalidated until Prompt 2:

```bash
ssh <bath-username>@<node>.cs.bath.ac.uk
```

After login, inspect the node:

```bash
hostname
nvidia-smi
free -h
df -h
hare me
quota || true
```

Do not infer the Bath username from the local WSL username.

## Step 3: Prepare Scratch

Unvalidated until Prompt 2:

```bash
SCRATCH=/mnt/fast0/<bath-username>/msc-audio-pianist
mkdir -p "$SCRATCH"
hare reserve "$SCRATCH"
hare me
```

Use `hare extend "$SCRATCH"` before expiry if needed. Use `hare release "$SCRATCH"` only after all valuable outputs are copied elsewhere.

## Step 4: Put The Repository On Hex

Use the home directory for code, not training outputs:

```bash
cd ~
git clone <repo-url> msc-audio-pianist
cd msc-audio-pianist
git checkout 4406ac9
```

If later commits add Hex prep/resume support, check out that later commit instead.

Ensure `third_party/robopianist` exists at the recorded commit and includes `robopianist/soundfonts/TimGM6mb.sf2`. The source install can also be recreated from `third_party/robopianist_commit.txt`, but the soundfont must be present or transferred.

## Step 5: Transfer The 300k Checkpoint

From local WSL, dry-run first:

```bash
scripts/hex/sync_checkpoint_to_hex.sh \
  --hex-user <bath-username> \
  --node <node> \
  --remote-directory /mnt/fast0/<bath-username>/msc-audio-pianist/checkpoints \
  --checkpoint experiments/general_one_hand/droq/checkpoints/droq_stage3c_fair_1m_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_300000_steps.pt
```

Execute only after checking the destination:

```bash
scripts/hex/sync_checkpoint_to_hex.sh ... --execute
```

Compare local and remote SHA-256 checksums. Do not modify or overwrite the original local 300k checkpoint.

## Step 6: Build The Image With Hare

Run later on the selected Hex node, not locally:

```bash
cd ~/msc-audio-pianist
scripts/hex/build_image.sh droq-stage3c
```

This resolves an image tag like:

```text
$USER/msc-audio-pianist:droq-stage3c
```

The helper uses `hare build`; no sudo is required. The image must be validated against the selected node's NVIDIA driver in Prompt 2.

## Step 7: Run The General Smoke Test

Unvalidated until Prompt 2:

```bash
scripts/hex/smoke_test.sh \
  --gpu <gpu-index> \
  --scratch /mnt/fast0/<bath-username>/msc-audio-pianist \
  --image <bath-username>/msc-audio-pianist:droq-stage3c
```

The smoke test checks:

- `torch.cuda.is_available()`;
- a small CUDA tensor operation;
- imports of PyTorch, MuJoCo, dm-control, RoboPianist and project modules;
- `TimGM6mb.sf2` discovery;
- reset and a few headless RoboPianist environment steps;
- write/persist a file in `/workspace`.

It exposes only the selected GPU with `--gpus device=<GPU_INDEX>`.

## Step 8: Run A Short Resume Test From A Copy

Do not test against the only copy of the checkpoint. Copy it under scratch first:

```bash
cp "$SCRATCH/checkpoints/checkpoint_300000_steps.pt" \
   "$SCRATCH/checkpoints/checkpoint_300000_steps.resume-test.pt"
```

Dry-run:

```bash
scripts/hex/launch_resume.sh \
  --gpu <gpu-index> \
  --scratch "$SCRATCH" \
  --image <image-tag> \
  --checkpoint "$SCRATCH/checkpoints/checkpoint_300000_steps.resume-test.pt" \
  --additional-timesteps 1000 \
  --run-name droq_stage3c_resume_test \
  --checkpoint-every 500 \
  --mode foreground
```

Launch only after explicit approval:

```bash
scripts/hex/launch_resume.sh ... --confirm-launch
```

Verify that the script reports `resume_start_step=300000` and saves checkpoints such as `checkpoint_300500_steps.pt` or `checkpoint_301000_steps.pt`.

## Step 9: Dry-Run The Production Continuation

Unvalidated until Prompt 2:

```bash
scripts/hex/launch_resume.sh \
  --gpu <gpu-index> \
  --scratch "$SCRATCH" \
  --image <image-tag> \
  --checkpoint "$SCRATCH/checkpoints/checkpoint_300000_steps.pt" \
  --additional-timesteps 700000 \
  --run-name droq_stage3c_fair_hexcloud_resume \
  --checkpoint-every 100000 \
  --mode detached
```

This prints the fully resolved command and exits. It will not launch without `--confirm-launch`.

## Step 10: Launch Only After Approval

After the dry-run is inspected:

```bash
scripts/hex/launch_resume.sh ... --confirm-launch
```

Outputs should be under:

```text
/workspace/runs/droq_stage3c_fair_hexcloud_resume/
```

The container command uses:

- `/app` for the mounted repository;
- `/workspace` for mounted scratch;
- `--device cuda` so unavailable CUDA fails clearly;
- unbuffered Python output;
- `--output-dir /workspace/runs/<run-name>/droq`.

## Monitoring

Unvalidated until Prompt 2:

```bash
scripts/hex/status.sh droq_stage3c_fair_hexcloud_resume "$SCRATCH"
hare me
hare usage
nvidia-smi
df -h "$SCRATCH"
```

Do not treat the existence of a process alone as proof of progress. Check log timestamps, checkpoint timestamps, latest global step, CPU/RAM/GPU usage and remaining scratch space.

## Disconnecting And Reconnecting

Detached Hare runs should continue after SSH disconnects. Use:

```bash
hare logs -n 80 droq_stage3c_fair_hexcloud_resume
hare logs -f droq_stage3c_fair_hexcloud_resume
```

If launched in tmux mode:

```bash
tmux attach -t droq_stage3c_fair_hexcloud_resume
```

Detach from tmux with `Ctrl+B`, then `D`. For Hare interactive attach, the docs say detach with `Ctrl+P`, then `Ctrl+Q`.

## Graceful Stop

Dry-run first:

```bash
scripts/hex/stop_run.sh \
  --run-name droq_stage3c_fair_hexcloud_resume \
  --scratch "$SCRATCH"
```

Stop only after confirmation:

```bash
scripts/hex/stop_run.sh \
  --run-name droq_stage3c_fair_hexcloud_resume \
  --scratch "$SCRATCH" \
  --confirm-stop
```

The script attempts graceful termination first and never deletes checkpoints.

## Resume After Interruption

Find the latest checkpoint:

```bash
find "$SCRATCH/runs/droq_stage3c_fair_hexcloud_resume" \
  -name 'checkpoint_*_steps.pt' -type f | sort | tail -1
```

Resume from that checkpoint with `--additional-timesteps` equal to the remaining timesteps. For example, if the latest checkpoint is `checkpoint_500000_steps.pt`, use `--additional-timesteps 500000`.

## Copy Results Back To WSL

Dry-run first:

```bash
scripts/hex/sync_results_from_hex.sh \
  --hex-user <bath-username> \
  --node <node> \
  --remote-run-directory "$SCRATCH/runs/droq_stage3c_fair_hexcloud_resume" \
  --local-output-directory experiments/hex_results/droq_stage3c_fair_hexcloud_resume
```

Execute only after inspecting:

```bash
scripts/hex/sync_results_from_hex.sh ... --execute
```

Back up important checkpoints outside Hex. Scratch is node-local and not a backup.

## Untested Until Prompt 2

- SSH to Hex.
- Hare image build.
- Hare GPU container run.
- Scratch reservation.
- CUDA execution on a selected Hex node.
- Resume test from the 300k checkpoint on Hex.
- The 700000-step production continuation.
