"""One-case entry point for crash-contained production processing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from .production import build_production_processor


def main(argv: Sequence[str] | None = None) -> int:
    arguments = sys.argv if argv is None else argv
    if len(arguments) != 2:
        return 64
    pdf_path = Path(arguments[1])
    if not pdf_path.is_file():
        return 66
    try:
        row = build_production_processor().process_case(pdf_path)
    except Exception:
        return 70
    if row is None:
        return 70
    sys.stdout.write(
        json.dumps(
            row.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
        )
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
