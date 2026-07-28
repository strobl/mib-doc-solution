from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import mib_pipeline.score_heads as score_heads
import mib_pipeline.score_confidence as score_confidence
import mib_pipeline.score_finalizer as score_finalizer
from mib_pipeline.batch import BatchRunner
from mib_pipeline.models import PredictionRow
from mib_pipeline.score_finalizer import VisibleScoreFinalizer


def _review_row() -> PredictionRow:
    return PredictionRow.from_mapping(
        {
            "case_id": "MIB-999999",
            "applicant_name": "Ada Visitor",
            "species_code": "HUM",
            "home_world": "Earth",
            "visa_class": "XW-2",
            "sponsor_id": "SPN-1234",
            "arrival_date": "2026-07-01",
            "declared_purpose": "research",
            "risk_flags": "none",
            "fee_status": "paid",
            "adjudication": "NEEDS_REVIEW",
            "confidence": 0.5,
        }
    )


def test_pdfium_fallback_preserves_page_boundaries(monkeypatch, tmp_path: Path) -> None:
    class FakeTextPage:
        def __init__(self, value: str) -> None:
            self.value = value

        def get_text_bounded(self) -> str:
            return self.value

    class FakePage:
        def __init__(self, value: str) -> None:
            self.value = value

        def get_textpage(self) -> FakeTextPage:
            return FakeTextPage(self.value)

    class FakeDocument:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("FORM"), FakePage("RECEIPT"), FakePage("IDENTITY")]

        def __len__(self) -> int:
            return len(self.pages)

        def __getitem__(self, index: int) -> FakePage:
            return self.pages[index]

        def close(self) -> None:
            return None

    monkeypatch.setattr(score_heads.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "pypdfium2",
        type("FakePdfium", (), {"PdfDocument": FakeDocument}),
    )

    text = score_heads._pdf_layout_text(tmp_path / "case.pdf")

    assert text == "FORM\fRECEIPT\fIDENTITY"


def test_finalizer_uses_only_visible_heads_and_calibration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    row = _review_row()
    calls: list[str] = []

    def record(name: str):
        def transform(current: PredictionRow, *args, **kwargs) -> PredictionRow:
            calls.append(name)
            return current

        return transform

    monkeypatch.setattr(score_heads, "apply_visible_field_repairs", record("fields"))
    monkeypatch.setattr(score_heads, "apply_layout_consensus_approval", record("layout"))
    monkeypatch.setattr(
        score_heads,
        "apply_visible_slash_stamp_denial",
        record("slash"),
        raising=False,
    )
    monkeypatch.setattr(
        score_heads,
        "apply_visible_sample_denial",
        record("sample"),
        raising=False,
    )
    monkeypatch.setattr(score_heads, "apply_visible_finding_decision", record("finding"))
    monkeypatch.setattr(score_heads, "apply_damage_weak_review", record("damage"))
    monkeypatch.setattr(score_heads, "apply_approval_safety_demotion", record("safety"))
    monkeypatch.setattr(
        score_heads,
        "apply_denial_to_review_softening",
        record("softening"),
    )
    monkeypatch.setattr(score_finalizer, "apply_confidence_blend", record("blend"))
    monkeypatch.setattr(
        score_finalizer,
        "apply_platt_calibration",
        record("platt"),
        raising=False,
    )

    result = VisibleScoreFinalizer()(tmp_path / "case.pdf", row)

    assert result.case_id == row.case_id
    assert calls == [
        "fields",
        "layout",
        "slash",
        "sample",
        "finding",
        "damage",
        "safety",
        "softening",
        "finding",
        "blend",
        "platt",
    ]


def test_hollow_blue_slash_stamp_pixels_are_detected() -> None:
    image = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(image)
    blue = (60, 100, 220)
    draw.rectangle((110, 110, 189, 189), outline=blue, width=4)
    draw.line((110, 189, 189, 110), fill=blue, width=4)

    assert score_heads.has_hollow_slash_stamp_pixels(np.asarray(image))


def test_solid_blue_square_is_not_a_hollow_slash_stamp() -> None:
    image = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((110, 110, 189, 189), fill=(60, 100, 220))

    assert not score_heads.has_hollow_slash_stamp_pixels(np.asarray(image))


def test_slash_denial_requires_review_paid_and_unproven_fee() -> None:
    row = _review_row()

    denied = score_heads.apply_visible_slash_stamp_denial_from_signals(
        row,
        has_stamp=True,
        fee_paid_proven=False,
    )
    proven_fee = score_heads.apply_visible_slash_stamp_denial_from_signals(
        row,
        has_stamp=True,
        fee_paid_proven=True,
    )
    no_stamp = score_heads.apply_visible_slash_stamp_denial_from_signals(
        row,
        has_stamp=False,
        fee_paid_proven=False,
    )

    assert denied.adjudication == "DENIED"
    assert denied.confidence == 0.95
    assert proven_fee == row
    assert no_stamp == row


def test_red_channel_binary_mask_isolates_red_pixels() -> None:
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    image[1, 1] = (220, 40, 40)
    image[0, 0] = (20, 20, 20)

    mask = score_heads.red_channel_binary_mask(image)

    assert mask is not None
    assert mask[1, 1] == 0
    assert mask[0, 0] == 255
    assert mask[2, 2] == 255


def test_sample_denial_requires_narrow_visible_evidence_gate() -> None:
    row = _review_row()
    dip_row = PredictionRow.from_mapping(
        row.to_dict()
        | {
            "visa_class": "DIP-1",
            "declared_purpose": "diplomatic",
        }
    )

    denied = score_heads.apply_visible_sample_denial_from_signals(
        dip_row,
        has_sample_denial=True,
        fee_signal=False,
        review_signal=False,
    )
    explicit_fee = score_heads.apply_visible_sample_denial_from_signals(
        dip_row,
        has_sample_denial=True,
        fee_signal=True,
        review_signal=False,
    )
    damaged = score_heads.apply_visible_sample_denial_from_signals(
        dip_row,
        has_sample_denial=True,
        fee_signal=False,
        review_signal=True,
    )
    transit = score_heads.apply_visible_sample_denial_from_signals(
        PredictionRow.from_mapping(
            dip_row.to_dict() | {"declared_purpose": "transit"}
        ),
        has_sample_denial=True,
        fee_signal=False,
        review_signal=False,
    )

    assert denied.adjudication == "DENIED"
    assert denied.fee_status == "unpaid"
    assert denied.confidence == 0.98
    assert explicit_fee == dip_row
    assert damaged == dip_row
    assert transit.adjudication == "NEEDS_REVIEW"


def test_platt_calibration_changes_only_confidence() -> None:
    row = _review_row()

    calibrated = score_confidence.apply_platt_calibration(row)

    expected = 1.0 / (1.0 + math.exp(-0.1618425654246005))
    assert calibrated.to_dict() | {"confidence": row.confidence} == row.to_dict()
    assert calibrated.confidence == pytest.approx(expected)


def test_runtime_contains_no_answer_key_module_or_opt_in_switch() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_files = [root / "solution.py", *sorted((root / "mib_pipeline").glob("*.py"))]
    runtime_source = "\n".join(path.read_text() for path in runtime_files)

    assert not (root / "mib_pipeline" / "arjun_answer_key.py").exists()
    assert "MIB_ALLOW_ANSWER_KEY" not in runtime_source
    assert "apply_answer_key_transcription" not in runtime_source
    assert re.search(r"MIB-\d{6}", runtime_source) is None


def test_finalizer_failure_preserves_valid_base_prediction(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "MIB-999999.pdf").touch()
    output_path = tmp_path / "predictions.jsonl"
    base_row = _review_row()

    class BaseProcessor:
        def process_case(self, _pdf_path: Path) -> PredictionRow:
            return base_row

    def broken_finalizer(_pdf_path: Path, _row: PredictionRow) -> PredictionRow:
        raise RuntimeError("layout parser failed")

    report = BatchRunner(
        BaseProcessor(),
        max_workers=1,
        row_finalizer=broken_finalizer,
    ).run(input_dir, output_path)

    assert report.answered == 1
    assert report.omitted == 0
    assert '"case_id":"MIB-999999"' in output_path.read_text()
