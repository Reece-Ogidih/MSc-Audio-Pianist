# Pipeline 2 Phase-A Dissertation Results

This directory contains lightweight copies of frozen Pipeline 2 Phase-A evaluation outputs used for dissertation analysis. The original run artifacts, checkpoints, replay buffers, logs, rendered media, and generated audio remain excluded from Git.

The seed 13 and seed 37 outputs were synced from Hex into `artifacts/pipeline2_phase_a_hex/`. The seed 61 outputs were synced into `artifacts/pipeline2_final_hex/`.

These files are evidence copies. Values should not be modified during copying; if regenerated or repaired, write a new clearly named result set rather than editing these rows in place.

Included per seed:

- `evaluation_sequence_metrics.csv`
- `evaluation_checkpoint_metrics.csv`
- `learning_curve.csv`
- `evaluation_summary.json`
- `evaluation_manifest.json`
- `audio_dependence.csv`

The `audio_dependence.csv` files are included because they are small lightweight evidence files and directly support the Pipeline 2 direct-audio dependence analysis.
