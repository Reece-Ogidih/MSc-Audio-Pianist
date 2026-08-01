# Pipeline 2 Direct Audio v1 Design

## Definition

Pipeline 2 removes Pipeline 1's explicit symbolic bottleneck. The direct policy receives:

`raw reference waveform + non-goal physical RoboPianist state -> 22-D continuous action`

At inference the musical instruction is only the waveform. Learned internal convolutional or recurrent features are allowed because they are not separately supervised, interpretable note labels.

Forbidden policy inputs include MIDI, target note IDs, key one-hots, fingering labels, `TimedNote`, `ControllerGoalSequence`, Basic Pitch outputs, onset/offset detections, transcription confidences, and native RoboPianist goal arrays.

Privileged score/MIDI remains allowed only for rendering reference audio, training reward, and evaluation metrics.

## Expert-Trajectory Feasibility Audit

The local RoboPianist tree does not contain a directly usable expert dataset for this five-note one-hand task.

Candidate sources:

| Source | Finding | Compatibility |
| --- | --- | --- |
| RoboPianist tutorial oracle | Reads `goal` and drives piano/key actuators or tutorial task actions | Not independent of symbolic target observations; not suitable |
| Tutorial `twinkle_twinkle_actions.npy` | External downloadable action playback for two-hand Twinkle example | Not local, not five-note task, likely two-hand action shape, not our 22-D one-hand space |
| PIG/RoboPianist MIDI dataset | Dataset docs describe fingering MIDI preprocessing, not state-action expert rollouts | No actions; not a direct demonstration source |
| RP1M/equivalent local trajectory assets | No local RP1M trajectory/checkpoint schema found in the cloned tree | Not available without separate download/integration |
| Frozen Pipeline 1 controller | Can generate our exact task actions | Rejected: it would distil Pipeline 1 and weaken the comparison |

Result: `EXPERT_DATA_COMPATIBLE=false`.

## Selected Training Route

`SELECTED_TRAINING_ROUTE=DIRECT_RL`

Imitation is rejected for v1 because no clean, independent, one-hand, action-compatible expert source is available. Direct DroQ-style RL preserves the clearest comparison: Pipeline 1 and Pipeline 2 both use RL-trained control policies, while the architectural difference is symbolic goal conditioning versus raw-audio conditioning.

## Matched Task

MIDI range: `72-76`

Sequences:

- anchors: `[72]`, `[73]`, `[74]`, `[75]`, `[76]`
- transitions: `[72,73]`, `[73,72]`, `[73,74]`, `[74,73]`, `[74,75]`, `[75,74]`, `[75,76]`, `[76,75]`

Timing:

- note duration: `0.28s`
- gap: `0.12s`
- event period: `0.40s`
- RoboPianist control timestep: `0.05s`
- MuJoCo physics timestep: `0.005s`
- action repeat: `1`

Pipeline 1's native symbolic training used `lookahead=1`, which corresponds to one future RoboPianist goal frame, or `0.05s`. Pipeline 2 uses `0.40s` future raw-audio context so a whole next musical event can be represented without symbolic labels.

## Observation Whitelist

Policy observation is a Gym dictionary:

- `audio`: raw waveform window, shape `(8000,)`
- `physical`: whitelisted physical state, shape `(118,)`

Physical fields:

- `piano/state`: 88
- `piano/sustain_state`: 1
- `rh_shadow_hand/joints_pos`: 26
- `rh_shadow_hand/position`: 3

Removed symbolic/non-physical fields:

- `goal`: 178
- `fingering`: 5
- `phase`: 1

The environment includes runtime assertions that forbidden tokens do not appear in direct policy observation names. `clip_id` and `audio_sample_index` are replay/storage metadata only and are not policy features.

## Audio Reference Bank

`AudioReferenceBank` stores each rendered waveform once and resolves windows by:

- `clip_id`
- `audio_sample_index`

The policy receives only the resolved raw waveform. The replay buffer stores compact indices rather than waveform windows.

Audio defaults:

- sample rate: `16000 Hz`
- past context: `0.10s`
- future context: `0.40s`
- total context: `0.50s`
- samples: `8000`
- deterministic zero padding at boundaries

Training variation initially uses deterministic MIDI velocity and global gain variants only. Final evaluation remains on canonical FluidSynth/TimGM6mb.sf2 audio.

## Model

Minimal direct DroQ architecture:

`waveform[8000] -> Conv1D stack -> GRU -> 128-D audio latent`

The audio latent is concatenated with 118-D physical state and passed through a fusion MLP to produce a tanh-squashed 22-D continuous action distribution.

Critics use the same raw-audio encoder form and receive:

`waveform + physical state + action -> Q`

Temporal ordering is preserved by the convolution sequence followed by a GRU. There is no order-invariant global average over the audio context.

## Replay and Checkpointing

Replay stores:

- physical observation
- next physical observation
- action
- reward
- done
- clip/audio indices
- next clip/audio indices

It does not store waveform windows.

Approximate indexed replay memory:

- per transition: about `1.05 KB`
- 1M transitions: about `1.05 GB`
- 2M transitions: about `2.10 GB`

Naively storing current and next 8000-sample float32 waveform windows would add about `64 KB` per transition:

- 1M transitions: about `64 GB`
- 2M transitions: about `128 GB`

Full resumable checkpoints must include actor, critics, target critics, optimizers, alpha state, indexed replay arrays, RNG state, counters, config, and audio-bank provenance. Expected checkpoint sizes are dominated by replay:

- 1M: roughly `1.1-1.5 GB`
- 1.5M: roughly `1.6-2.3 GB`
- 2M: roughly `2.2-3.0 GB`

The Phase-A launcher saves lightweight policy checkpoints at:

`10k, 25k, 50k, 100k, 250k, 500k, 750k, 1M`

The random step-0 policy can be evaluated directly from a freshly initialised model without storing a duplicate checkpoint.

At `1M` it saves a full resumable checkpoint containing:

- actor;
- critics;
- target critics;
- actor and critic optimizers;
- entropy/alpha value and optimizer;
- indexed replay buffer;
- Python, NumPy, PyTorch CPU and PyTorch CUDA RNG state where available;
- step counters;
- seed and training config;
- audio-bank provenance.

Full resume is distinct from actor-only warm-starting. Pipeline 2 continuation must use:

`--resume-checkpoint FULL_1M.pt --additional-timesteps 500000`

and not actor-only warm start.

Two concurrent Phase-A 1M seeds are expected to require roughly:

- per seed: `1.2-1.8 GB`;
- two seeds: `2.4-3.6 GB`.

If later retaining full checkpoints at `1M`, `1.5M`, and `2M`, budget roughly `5-7 GB` per seed, dominated by replay state.

GPU memory estimate for one run with batch size `64`:

- waveform batch: `64 x 8000`, about `2 MB` before activations;
- actor and two critic audio encoders plus target critics fit comfortably in model memory;
- UTD ratio increases compute time, not simultaneous batch memory.

An RTX 2080 with `8 GB` should be adequate for the configured model and batch size. If CUDA memory pressure appears on Hex, the smallest justified change is reducing batch size before changing model architecture.

## Phase A Plan

Maximum prepared budget: `2,000,000` environment steps.

First formal decision point: `1,000,000` environment steps.

Continue past 1M only if learning curves show anchors/transitions still improving, temporal/control metrics improve, and there is no architectural collapse.

Additional evidence supporting continuation:

- meaningful difference between correct-audio and zero/wrong-audio evaluation;
- increasing transition completion;
- timestep F1 still trending upward.

Evidence against continuation:

- long flat/non-functional learning;
- no measurable audio dependence;
- instability/collapse;
- replay/checkpoint defect.

Do not continue merely because the model has not yet beaten Pipeline 1.

Prepared lightweight evaluation checkpoints:

`0, 10k, 25k, 50k, 100k, 250k, 500k, 750k, 1M, 1.25M, 1.5M, 1.75M, 2M`

Prepared full checkpoints:

`1M, 1.5M, 2M`

## Evaluation

Pipeline comparison:

1. Oracle symbolic: ground-truth MIDI to frozen symbolic controller
2. Pipeline 1 indirect: Basic Pitch RAW to frozen symbolic controller
3. Pipeline 2 direct: raw audio plus physical state to direct policy

Metrics match Pipeline 1:

- pressed-key precision/recall/F1
- timestep precision/recall/F1
- max and integrated unintended activation
- wrong-key crossings
- transition completion
- previous-target release
- second-target completion
- action delta/saturation where available

## Completion Gate

`pipeline_2_architecturally_complete=true` only after a trained direct policy:

- uses raw audio as the only musical instruction;
- receives no symbolic target representation at inference;
- covers MIDI `72-76`;
- evaluates all five anchors and eight transitions;
- produces RoboPianist actions end to end;
- records quantitative metrics.

It does not need to beat Pipeline 1 to be architecturally complete.

## Methodological Limitation

The first Pipeline 2 route is direct RL, not VLA-style imitation. This makes the Pipeline 1 comparison more controlled with respect to training algorithm, but it does not test whether expert-demonstration learning would be superior if a compatible independent expert dataset later becomes available.
