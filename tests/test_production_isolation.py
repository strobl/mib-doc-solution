from __future__ import annotations

import contextlib
import io
import json
import types
from pathlib import Path
from unittest.mock import patch

from mib_pipeline import case_worker
from mib_pipeline.models import PredictionRow
from mib_pipeline.production import IsolatedProductionProcessor


def _row(case_id: str = "MIB-000001") -> PredictionRow:
    return PredictionRow.from_mapping(
        {
            "case_id": case_id,
            "applicant_name": "Zed Zarnax",
            "species_code": "ORION_GRAYS",
            "home_world": "Kepler-186f",
            "visa_class": "XW-2",
            "sponsor_id": "SPN-1042",
            "arrival_date": "2026-04-17",
            "declared_purpose": "research",
            "risk_flags": "none",
            "fee_status": "paid",
            "adjudication": "APPROVED",
            "confidence": 0.91,
        }
    )


def _completed(returncode: int, stdout: bytes = b"") -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=returncode, stdout=stdout)


def test_isolated_processor_retries_native_abort_and_returns_valid_row(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "MIB-000001.pdf"
    pdf_path.write_bytes(b"pdf")
    responses = [
        _completed(134),
        _completed(0, json.dumps(_row().to_dict()).encode("utf-8")),
    ]
    calls = []

    def run_factory(command, **kwargs):
        calls.append((command, kwargs))
        return responses[len(calls) - 1]

    processor = IsolatedProductionProcessor(run_factory=run_factory)

    result = processor.process_case(pdf_path)

    assert result == _row()
    assert len(calls) == 2
    assert calls[0][0][-3:] == [
        "-m",
        "mib_pipeline.case_worker",
        str(pdf_path),
    ]
    assert calls[0][1]["timeout"] == 180


def test_isolated_processor_fails_closed_after_two_bad_children(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "MIB-000001.pdf"
    pdf_path.write_bytes(b"pdf")
    responses = [_completed(134), _completed(70)]

    def run_factory(command, **kwargs):
        del command, kwargs
        return responses.pop(0)

    processor = IsolatedProductionProcessor(run_factory=run_factory)

    assert processor.process_case(pdf_path) is None


def test_case_worker_emits_one_canonical_row(tmp_path: Path) -> None:
    pdf_path = tmp_path / "MIB-000001.pdf"
    pdf_path.write_bytes(b"pdf")
    processor = types.SimpleNamespace(process_case=lambda _path: _row())
    stdout = io.StringIO()

    with (
        patch.object(
            case_worker,
            "build_production_processor",
            return_value=processor,
        ),
        contextlib.redirect_stdout(stdout),
    ):
        exit_code = case_worker.main(["case_worker", str(pdf_path)])

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == _row().to_dict()


def test_case_worker_returns_failure_without_partial_json(tmp_path: Path) -> None:
    pdf_path = tmp_path / "MIB-000001.pdf"
    pdf_path.write_bytes(b"pdf")
    processor = types.SimpleNamespace(process_case=lambda _path: None)
    stdout = io.StringIO()

    with (
        patch.object(
            case_worker,
            "build_production_processor",
            return_value=processor,
        ),
        contextlib.redirect_stdout(stdout),
    ):
        exit_code = case_worker.main(["case_worker", str(pdf_path)])

    assert exit_code == 70
    assert stdout.getvalue() == ""
