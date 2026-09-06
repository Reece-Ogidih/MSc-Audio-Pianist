#!/usr/bin/env python3
"""Audit external real-piano recording manifests for final experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ala_pianist.evaluation.final_experiments import (
    audit_real_audio_manifest,
    write_csv,
    write_json,
    write_real_audio_manifest_template,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()

    if args.write_template and not args.manifest.exists():
        write_real_audio_manifest_template(args.manifest)
        print(f"template_written={args.manifest}")
        return

    rows, summary = audit_real_audio_manifest(args.manifest)
    output_dir = args.output_dir or args.manifest.parent / "audit"
    write_csv(output_dir / "real_audio_manifest_audit.csv", rows)
    write_json(output_dir / "real_audio_manifest_audit_summary.json", summary)
    print(f"recording_count={summary['recording_count']}")
    print(f"ok_count={summary['ok_count']}")
    print(f"missing_count={summary['missing_count']}")
    print(f"output_dir={output_dir}")
    print("REAL_AUDIO_MANIFEST_AUDIT_COMPLETE=true")


if __name__ == "__main__":
    main()
