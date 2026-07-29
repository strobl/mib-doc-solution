"""Length-framed offline RapidOCR worker with pure-Python responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .rapid_recovery import (
    _RAPID_OCR_FRAME,
    _read_exact,
    _rapid_ocr_params,
)


def _materialize_result(result: Any) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None or scores is None:
        return {"status": "ok", "boxes": [], "txts": [], "scores": []}

    materialized_boxes: list[list[list[float]]] = []
    materialized_texts: list[str] = []
    materialized_scores: list[float] = []
    for box, text, score in zip(boxes, texts, scores, strict=False):
        materialized_boxes.append(
            [[float(point[0]), float(point[1])] for point in box]
        )
        materialized_texts.append(str(text))
        materialized_scores.append(float(score))
    return {
        "status": "ok",
        "boxes": materialized_boxes,
        "txts": materialized_texts,
        "scores": materialized_scores,
    }


def _read_request(stream: Any) -> bytes | None:
    try:
        size = _RAPID_OCR_FRAME.unpack(
            _read_exact(stream, _RAPID_OCR_FRAME.size)
        )[0]
    except EOFError:
        return None
    if size == 0:
        return None
    return _read_exact(stream, size)


def _write_response(stream: Any, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    stream.write(_RAPID_OCR_FRAME.pack(len(encoded)))
    stream.write(encoded)
    stream.flush()


def main() -> int:
    try:
        import rapidocr as rapidocr_package
    except ImportError:
        return 70
    package_file = getattr(rapidocr_package, "__file__", None)
    if package_file is None:
        return 70
    model_root = str(Path(package_file).resolve().parent / "models")
    try:
        engine = rapidocr_package.RapidOCR(
            params=_rapid_ocr_params(model_root)
        )
    except Exception:
        return 70

    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    while True:
        image_png = _read_request(input_stream)
        if image_png is None:
            return 0
        try:
            payload = _materialize_result(engine(image_png))
        except Exception:
            payload = {"status": "error"}
        _write_response(output_stream, payload)


if __name__ == "__main__":
    raise SystemExit(main())
