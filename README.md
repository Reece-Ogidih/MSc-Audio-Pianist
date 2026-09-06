# MSc Audio Pianist

Dissertation title: **Exploring Audio-to-Action Learning for Robotic Piano Playing**

This repository contains the software artefacts for an MSc project on audio-conditioned dexterous piano playing in simulation. The research problem is whether a robotic piano policy can use audio-derived musical intent to drive continuous hand actions, and how an indirect transcription-based pipeline compares with a direct raw-audio-to-action policy.

The final benchmark is deliberately constrained so that the behaviour can be inspected in detail:

- one right Shadow Hand in RoboPianist;
- local five-note range, MIDI pitches 72-76;
- monophonic note sequences;
- no sustain pedal;
- RoboPianist's one-hand forearm translation controls;
- continuous 22-dimensional exposed action interface.

Two complete audio-to-action systems are implemented:

- **Pipeline 1, indirect:** audio -> Basic Pitch transcription -> symbolic timed-note targets -> learned dexterous controller -> robotic action.
- **Pipeline 2, direct:** raw waveform features plus physical state -> learned audio representation -> continuous robotic action.

These are complete system designs rather than a controlled representation-only ablation: Pipeline 1 uses an explicit symbolic intermediate representation, while Pipeline 2 learns directly from audio-conditioned observations.

## Repository Structure

- `src/ala_pianist/audio/`: audio utilities, synthesis, transcription interfaces, Basic Pitch integration and recorded-piano helpers.
- `src/ala_pianist/music/`: MIDI/timed-note representations, generated curricula, sequence timing and pair/long-horizon curriculum generation.
- `src/ala_pianist/envs/`: wrapper around RoboPianist's one-hand task, including the 22D no-sustain action interface.
- `src/ala_pianist/controllers/`: symbolic action-library, learned-policy and hybrid controller helpers used mainly by Pipeline 1.
- `src/ala_pianist/pipelines/`: Pipeline 1 indirect interfaces and orchestration code.
- `src/ala_pianist/rl/`: RL environments and trainers, including symbolic DroQ/SAC support and the direct-audio Pipeline 2 implementation in `direct_audio_env.py` and `direct_audio_droq.py`.
- `src/ala_pianist/evaluation/`: precision/recall/F1 metrics, trajectory diagnostics, unintended-key metrics, long-horizon evaluation and direct-audio evaluation.
- `configs/`: retained experiment and release configurations.
- `scripts/`: training, evaluation, audit and packaging entry points.
- `scripts/hex/`: University of Bath Hex GPU-cluster launch, status, smoke-test and transfer helpers.
- `docs/dissertation_results/`: compact dissertation evidence tables, including Appendix B evidence and Pipeline 2 Phase-A seed summaries.
- `docs/experiments/`: experiment design notes and selected analysis summaries.
- `artifacts/`: curated retained model artefacts and compact provenance/evaluation evidence.
- `tests/`: unit and smoke tests for the implementation.
- `third_party/`: external dependency provenance; RoboPianist itself is not vendored in Git, but its checked-out commit is recorded in `third_party/robopianist_commit.txt`.

## Environment and Dependencies

The final local development environment used:

- Python 3.10.20;
- RoboPianist 1.0.10, with external source commit `0d9736c64eba5faafdf214ed7d38d648ffbd5c7f`;
- MuJoCo 3.10.0;
- dm-control 1.0.43;
- NumPy 2.2.6;
- PyTorch 2.13.0 locally for retained checkpoints;
- Gymnasium 1.3.0 and Stable-Baselines3 2.9.0 where used;
- Basic Pitch 0.4.0 and ONNX Runtime 1.23.2 for Pipeline 1 transcription;
- FluidSynth for MIDI-to-audio rendering.

RoboPianist is treated as an external dependency. The exact checked-out RoboPianist commit is recorded in `third_party/robopianist_commit.txt`.

Retained dependency references:

- `requirements-pipeline1-v1.txt`;
- `docker/hex/requirements.txt`;
- `docker/hex/Dockerfile`;
- `docker/hex/torch-constraints.txt`.

The Hex Docker files document the GPU environment used for larger runs. The repository intentionally does not vendor Conda environments, virtual environments, CUDA installations or the full RoboPianist source tree.

## Canonical Model Artefacts

### Pipeline 1

The selected symbolic controller is the immutable five-note DroQ-sensitive checkpoint:

- `artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt`
- SHA-256: `927c1050c08769c49568013ead0c69d69d4bd19ff23eb632e89bd89fb735ac4c`

Its retained documentation and provenance are:

- `artifacts/frozen_models/five_note_symbolic_controller_v1/manifest.json`
- `artifacts/frozen_models/five_note_symbolic_controller_v1/model_card.md`
- `artifacts/frozen_models/five_note_symbolic_controller_v1/SHA256SUMS.txt`
- `artifacts/frozen_models/five_note_symbolic_controller_v1/provenance/`

The selected checkpoint was chosen from the seed-13 five-note factorial comparison. The refinement canary did not replace it.

### Pipeline 2

The submission retains compact Phase-A and final Pipeline 2 evidence plus lightweight inference checkpoints where available:

- seed 13, 1M: `artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/`
- seed 37, 1M: `artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/`
- seed 61, 1M: `artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/`
- seed 13 continuation checkpoints, 1.25M and 1.5M: `artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed13_resume_1m_to_1p5m_retry1/`

Seed 13 at 1M is the selected downstream evaluation model for Pipeline 2. The ZIP includes lightweight checkpoints and compact CSV/JSON evaluation summaries. It does not include the large full resumable checkpoints, replay buffers or raw Hex scratch directories.

## Testing

From the repository root, with RoboPianist available on `PYTHONPATH`, run:

```bash
PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:/home/reece_dev/msc-audio-pianist/third_party/robopianist pytest tests -q
```

The final merged project passed:

```text
213 passed in 205.52s
```

## Main Training and Evaluation Entry Points

Useful entry points for inspecting or reproducing the work are:

- symbolic controller training: `scripts/train_droq_general_one_hand_policy.py`
- symbolic controller evaluation: `scripts/evaluate_general_one_hand_policy.py`
- five-note factorial evaluation: `scripts/evaluate_five_note_factorial_checkpoint_sweep.py`
- Pipeline 1 audio-to-action benchmark: `scripts/run_pipeline1_audio_to_action.py`
- Pipeline 1 five-note benchmark: `scripts/run_indirect_five_note_benchmark.py`
- direct-audio Pipeline 2 training: `scripts/train_direct_audio_droq.py`
- direct-audio Pipeline 2 evaluation: `scripts/evaluate_pipeline2_direct_audio.py`
- long-horizon evaluation: `scripts/evaluate_long_horizon_compositional.py`
- pair-complete/long-horizon curriculum preparation: `scripts/prepare_paircomplete_longhorizon_curriculum.py`
- recorded-audio distribution-shift preparation and evaluation: `scripts/real_audio/prepare_real_audio_distribution_shift.py` and `scripts/real_audio/evaluate_real_audio_distribution_shift.py`

Hex launch wrappers are under `scripts/hex/`. They are included for reproducibility and provenance, but are cluster-specific.

## Dissertation Result Evidence

Compact retained evidence is stored in:

- `docs/dissertation_results/appendix_b_evidence.md`;
- `docs/dissertation_results/pipeline2_phase_a/`;
- `docs/experiments/`;
- `figures/p2_seed_learning_curve.pdf`;
- `figures/p2_seed_learning_curve.png`;
- curated files under `artifacts/frozen_models/five_note_symbolic_controller_v1/`;
- compact Pipeline 2 summaries under `artifacts/pipeline2_phase_a_hex/` and `artifacts/pipeline2_final_hex/`.

Large generated rollout directories, raw Hex outputs, replay buffers, full checkpoints, logs and redundant media are excluded from the code ZIP. The submitted package retains the implementation, configurations, canonical lightweight checkpoints and compact result evidence needed to inspect the reported work.

## Recorded Audio

The recorded-audio distribution-shift experiment used 15 self-recorded isolated keyboard-note takes covering the five-note benchmark range. Raw recordings may be excluded from the submission ZIP to keep the software package compact.

Relevant documentation and scripts:

- `docs/real_piano_recording_protocol.md`;
- `docs/real_audio_distribution_shift_plan.md`;
- `scripts/real_audio/prepare_real_audio_distribution_shift.py`;
- `scripts/real_audio/evaluate_real_audio_distribution_shift.py`;
- `scripts/real_audio/analyze_real_audio_distribution_shift.py`.

## Compute

Most development, smoke tests and smaller evaluations were performed locally. Larger RL training runs and factorial/refinement experiments were performed on the University of Bath Hex GPU cluster. The retained Hex scripts document the launch commands, image configuration and runtime assumptions used for those experiments.

## Reproducibility Scope

Pipeline 1 controller selection used seed 13. Pipeline 2 main direct-audio training used seeds 13, 37 and 61. The benchmark is intentionally small and diagnostic: it evaluates one-hand robotic piano control over MIDI 72-76 using monophonic material and no sustain pedal. Broader-range piano performance, polyphony and real-dataset training were outside the final submitted scope.
