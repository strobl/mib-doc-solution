#!/usr/bin/env python3
"""Run and report bounded one-variable OCR ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.ocr_ablation import (  # noqa: E402
    BASELINE_CONFIG,
    build_report,
    config_sha256,
    registered_variants,
    render_markdown,
    run_variant,
)


def command_list(_args: argparse.Namespace) -> int:
    payload = {
        "baseline": {
            "variant_id": "baseline",
            "config": BASELINE_CONFIG,
            "config_sha256": config_sha256(BASELINE_CONFIG),
        },
        "variants": [variant.to_dict() for variant in registered_variants()],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    observation = run_variant(
        variant_id=args.variant,
        benchmark_id=args.benchmark_id,
        source_revision=args.source_revision,
        repeat_index=args.repeat,
        input_dir=Path(args.input_dir),
        predictions_path=Path(args.predictions),
        observation_path=Path(args.observation),
        max_workers=args.max_workers,
    )
    print(
        f"{observation['variant_id']} repeat={observation['repeat_index']} "
        f"answered={observation['answered']}/{observation['attempted']} "
        f"cpu={observation['cpu_seconds']:.3f}s "
        f"wall={observation['wall_seconds']:.3f}s"
    )
    return 0 if observation["omitted"] == 0 else 2


def command_report(args: argparse.Namespace) -> int:
    report = build_report(
        repo_root=REPO_ROOT,
        truth_path=Path(args.truth),
        observation_paths=tuple(Path(path) for path in args.observation),
    )
    json_path = Path(args.output_json)
    markdown_path = Path(args.output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"measured={sum(v['evidence_status'] != 'not_measured' for v in report['variants'])} "
        f"recommended={len(report['ranked_recommendations'])}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one-variable OCR ablations without exposing truth labels "
            "to the runtime command."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    variants = commands.add_parser("list-variants")
    variants.set_defaults(handler=command_list)

    run = commands.add_parser("run")
    run.add_argument("--variant", required=True)
    run.add_argument("--benchmark-id", required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--repeat", type=int, required=True)
    run.add_argument("--input-dir", required=True)
    run.add_argument("--predictions", required=True)
    run.add_argument("--observation", required=True)
    run.add_argument("--max-workers", type=int, default=4)
    run.set_defaults(handler=command_run)

    report = commands.add_parser("report")
    report.add_argument("--truth", required=True)
    report.add_argument("--observation", action="append", required=True)
    report.add_argument("--output-json", required=True)
    report.add_argument("--output-markdown", required=True)
    report.set_defaults(handler=command_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
