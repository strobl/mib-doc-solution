#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: run.sh <input_pdf_dir> <output_predictions_path>" >&2
  exit 64
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 -B "$SCRIPT_DIR/solution.py" "$1" "$2"
