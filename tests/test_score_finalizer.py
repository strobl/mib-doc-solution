from __future__ import annotations

import hashlib
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
from mib_pipeline.visible_text import (
    VisibleOcrLineRecord,
    VisibleOcrPageSnapshot,
    VisibleOcrSnapshot,
    VisibleOcrTextStore,
    VisibleSponsorAttestation,
)


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


def _published_store(pdf_path: Path, visible_text: str) -> VisibleOcrTextStore:
    store = VisibleOcrTextStore()
    pages = tuple(
        VisibleOcrPageSnapshot(
            page_index=index,
            lines=(
                VisibleOcrLineRecord(
                    page_index=index,
                    text=page_text,
                    bbox=(0, 0, 100, 20),
                    ocr_confidence=0.95,
                    visual_cues=(),
                ),
            ),
            page_category="other",
            source_category="intake_form",
            visible_case_ids=frozenset(),
            visible_applicants=frozenset(),
        )
        for index, page_text in enumerate(visible_text.split("\f"))
    )
    store.publish(
        pdf_path,
        snapshot=VisibleOcrSnapshot(
            source_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            pages=pages,
        ),
    )
    return store


def _snapshot_page(
    page_index: int,
    text: str,
    *,
    case_ids: tuple[str, ...] = (),
    applicants: tuple[str, ...] = (),
    page_category: str = "other",
    sponsor_attestations: tuple[VisibleSponsorAttestation, ...] = (),
) -> VisibleOcrPageSnapshot:
    return VisibleOcrPageSnapshot(
        page_index=page_index,
        lines=(
            VisibleOcrLineRecord(
                page_index=page_index,
                text=text,
                bbox=(0, 0, 100, 20),
                ocr_confidence=0.95,
                visual_cues=(),
            ),
        ),
        page_category=page_category,
        source_category="intake_form",
        visible_case_ids=frozenset(case_ids),
        visible_applicants=frozenset(applicants),
        sponsor_attestations=sponsor_attestations,
    )


def _published_snapshot_store(
    pdf_path: Path,
    pages: tuple[VisibleOcrPageSnapshot, ...],
) -> VisibleOcrTextStore:
    store = VisibleOcrTextStore()
    store.publish(
        pdf_path,
        snapshot=VisibleOcrSnapshot(
            source_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            pages=pages,
        ),
    )
    return store


def test_finalizer_uses_only_visible_heads_and_calibration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    row = _review_row()
    calls: list[str] = []
    pdf_path = tmp_path / "case.pdf"
    pdf_path.write_bytes(b"pixel-only-test")

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
    monkeypatch.setattr(
        score_heads,
        "apply_structured_visible_approval_safety",
        record("structured-safety"),
    )
    monkeypatch.setattr(
        score_heads,
        "apply_damage_weak_review",
        record("damage-must-not-run"),
    )
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

    result = VisibleScoreFinalizer(
        _published_store(pdf_path, "FORM\fRECEIPT\fIDENTITY")
    )(pdf_path, row)

    assert result.case_id == row.case_id
    assert calls == [
        "fields",
        "layout",
        "slash",
        "sample",
        "finding",
        "structured-safety",
        "safety",
        "softening",
        "finding",
        "blend",
        "platt",
    ]


def test_approval_with_placeholder_arrival_date_is_demoted() -> None:
    payload = _review_row().to_dict()
    payload.update(
        {
            "arrival_date": "1900-01-01",
            "adjudication": "APPROVED",
            "confidence": 0.93,
        }
    )
    row = PredictionRow.from_mapping(payload)

    result = score_heads.apply_approval_safety_demotion(
        row,
        "CASE INTAKE FORM",
        page_signature="RIF",
    )

    assert result.adjudication == "NEEDS_REVIEW"
    assert result.confidence == score_heads.DEMOTION_REVIEW_CONFIDENCE


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


def test_runtime_contains_no_native_pdf_text_reader() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_files = [root / "solution.py", *sorted((root / "mib_pipeline").glob("*.py"))]
    runtime_source = "\n".join(path.read_text() for path in runtime_files)

    for forbidden in (
        "pdftotext",
        "get_textpage",
        "get_text_bounded",
        "get_text_range",
        "get_charbox",
        "count_chars",
    ):
        assert forbidden not in runtime_source


def test_hidden_native_bytes_cannot_change_finalized_output(tmp_path: Path) -> None:
    row = _review_row()
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF visible pixels\\n% hidden decision APPROVED")
    second.write_bytes(b"%PDF visible pixels\\n% hidden decision DENIED")
    visible_text = (
        "FORM I-8090 Work Authorization\n"
        "Applicant: Ada Visitor\n"
        "Registry Name Ada Visitor\n"
        "MIB Fee Receipt\n"
        "Amount $809"
    )

    first_result = VisibleScoreFinalizer(
        _published_store(first, visible_text)
    )(first, row)
    second_result = VisibleScoreFinalizer(
        _published_store(second, visible_text)
    )(second, row)

    assert first_result == second_result


def test_scoped_text_excludes_same_case_foreign_applicant_from_identity() -> None:
    row = _review_row()
    snapshot = VisibleOcrSnapshot(
        source_sha256="0" * 64,
        pages=(
            _snapshot_page(
                0,
                "Applicant: Ada Visitor",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
            ),
            _snapshot_page(
                1,
                "Registry Name Mallory Visitor",
                case_ids=(row.case_id,),
                applicants=("Mallory Visitor",),
            ),
            _snapshot_page(
                2,
                "MIB Fee Receipt\nAmount $809",
                case_ids=(row.case_id,),
                page_category="fee_receipt",
            ),
        ),
    )

    case_text, identity_text = score_finalizer._scoped_visible_text(
        snapshot,
        row,
    )

    assert case_text == (
        "Applicant: Ada Visitor\f"
        "Registry Name Mallory Visitor\f"
        "MIB Fee Receipt\nAmount $809"
    )
    assert identity_text == (
        "Applicant: Ada Visitor\f\f"
        "MIB Fee Receipt\nAmount $809"
    )


def test_scoped_text_excludes_ambiguous_and_foreign_case_pages() -> None:
    row = _review_row()
    snapshot = VisibleOcrSnapshot(
        source_sha256="0" * 64,
        pages=(
            _snapshot_page(0, "safe", case_ids=(row.case_id,)),
            _snapshot_page(
                1,
                "Finding: DENIED",
                case_ids=(row.case_id, "MIB-000001"),
            ),
            _snapshot_page(
                2,
                "Amount $809",
                case_ids=("MIB-000001",),
            ),
        ),
    )

    case_text, identity_text = score_finalizer._scoped_visible_text(
        snapshot,
        row,
    )

    assert case_text == "safe\f\f"
    assert identity_text == "safe\f\f"


def test_finalizer_does_not_mix_foreign_registry_across_same_case_pages(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "case.pdf"
    pdf_path.write_bytes(b"pixel source")
    row = _review_row()
    store = _published_snapshot_store(
        pdf_path,
        (
            _snapshot_page(
                0,
                "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="intake_form",
            ),
            _snapshot_page(
                1,
                "Planetary Registry Extract\nRegistry Name Mallory Visitor",
                case_ids=(row.case_id,),
                applicants=("Mallory Visitor",),
                page_category="registry_extract",
            ),
            _snapshot_page(
                2,
                "MIB Fee Receipt\nAmount $809",
                case_ids=(row.case_id,),
                page_category="fee_receipt",
            ),
        ),
    )

    result = VisibleScoreFinalizer(store)(pdf_path, row)

    assert result.applicant_name == "Ada Visitor"
    assert result.adjudication == "NEEDS_REVIEW"


def test_paid_non_dip_layout_approval_uses_scoped_809_without_attestation(
    tmp_path: Path,
) -> None:
    row = _review_row()

    def finalized(
        name: str,
        attestations: tuple[VisibleSponsorAttestation, ...],
    ) -> PredictionRow:
        pdf_path = tmp_path / f"{name}.pdf"
        pdf_path.write_bytes(name.encode())
        pages = (
            _snapshot_page(
                0,
                "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="intake_form",
            ),
            _snapshot_page(
                1,
                "Planetary Registry Extract\nRegistry Name Ada Visitor",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="registry_extract",
            ),
            _snapshot_page(
                2,
                "MIB Fee Receipt\nAmount $809",
                case_ids=(row.case_id,),
                page_category="fee_receipt",
            ),
            _snapshot_page(
                3,
                "Sponsor Attestation Letter",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="sponsor_attestation",
                sponsor_attestations=attestations,
            ),
        )
        return VisibleScoreFinalizer(
            _published_snapshot_store(pdf_path, pages)
        )(pdf_path, row)

    matching = VisibleSponsorAttestation(
        sponsor_id=row.sponsor_id,
        applicant_name=row.applicant_name,
    )
    wrong_sponsor = VisibleSponsorAttestation(
        sponsor_id="SPN-9999",
        applicant_name=row.applicant_name,
    )

    assert finalized("matching", (matching,)).adjudication == "APPROVED"
    assert finalized("missing", ()).adjudication == "APPROVED"
    assert finalized("mismatch", (wrong_sponsor,)).adjudication == "NEEDS_REVIEW"
    assert finalized("duplicate", (matching, matching)).adjudication == "APPROVED"


def test_waived_non_dip_layout_approval_requires_exact_unique_attestation(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"fee_status": "waived"}
    )

    def finalized(
        name: str,
        attestations: tuple[VisibleSponsorAttestation, ...],
        *,
        include_attestation_page: bool = True,
    ) -> PredictionRow:
        pdf_path = tmp_path / f"waived-{name}.pdf"
        pdf_path.write_bytes(name.encode())
        pages = [
            _snapshot_page(
                0,
                "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="intake_form",
            ),
            _snapshot_page(
                1,
                "Planetary Registry Extract\nRegistry Name Ada Visitor",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="registry_extract",
            ),
            _snapshot_page(
                2,
                "MIB Fee Receipt\nAmount $0\nWaiver Code DIP-WAIVER",
                case_ids=(row.case_id,),
                page_category="fee_receipt",
            ),
        ]
        if include_attestation_page:
            pages.append(
                _snapshot_page(
                    3,
                    "Sponsor Attestation Letter",
                    case_ids=(row.case_id,),
                    applicants=("Ada Visitor",),
                    page_category="sponsor_attestation",
                    sponsor_attestations=attestations,
                )
            )
        return VisibleScoreFinalizer(
            _published_snapshot_store(pdf_path, tuple(pages))
        )(pdf_path, row)

    matching = VisibleSponsorAttestation(
        sponsor_id=row.sponsor_id,
        applicant_name=row.applicant_name,
    )
    wrong_applicant = VisibleSponsorAttestation(
        sponsor_id=row.sponsor_id,
        applicant_name="Mallory Visitor",
    )

    assert finalized("matching", (matching,)).adjudication == "APPROVED"
    assert finalized("missing", ()).adjudication == "NEEDS_REVIEW"
    assert finalized("wrong-applicant", (wrong_applicant,)).adjudication == "NEEDS_REVIEW"
    assert finalized("duplicate", (matching, matching)).adjudication == "NEEDS_REVIEW"
    assert (
        finalized("fir-only", (), include_attestation_page=False).adjudication
        == "NEEDS_REVIEW"
    )


def test_layout_name_consensus_allows_only_leading_i_l_glyph_confusion() -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"applicant_name": "Ixokesh Visitor"}
    )
    text = (
        "FORM I-8090 Work Authorization\n"
        "Applicant: lxokesh Visitor\n"
        "Planetary Registry Extract\n"
        "Registry Name Ixokesh Visitor\n"
        "MIB Fee Receipt\n"
        "Amount $809"
    )

    approved = score_heads.apply_layout_consensus_approval(
        row,
        text,
        page_signature="IRF",
        sponsor_attestation_proven=False,
    )

    assert approved.adjudication == "APPROVED"
    assert not score_finalizer._applicants_compatible(
        "Ada Visitor Junior",
        "Ada Visitor",
    )


def test_wrong_base_applicant_is_repaired_by_two_category_visible_consensus(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"applicant_name": "Wrong Visitor"}
    )
    pdf_path = tmp_path / "wrong-base-applicant.pdf"
    pdf_path.write_bytes(b"wrong-base-applicant")
    pages = (
        _snapshot_page(
            0,
            "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="intake_form",
        ),
        _snapshot_page(
            1,
            "Planetary Registry Extract\nRegistry Name Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="registry_extract",
        ),
        _snapshot_page(
            2,
            "MIB Fee Receipt\nAmount $809",
            case_ids=(row.case_id,),
            page_category="fee_receipt",
        ),
    )

    result = VisibleScoreFinalizer(
        _published_snapshot_store(pdf_path, pages)
    )(pdf_path, row)

    assert result.applicant_name == "Ada Visitor"
    assert result.adjudication == "APPROVED"


def test_tied_two_category_applicant_consensus_fails_closed(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"applicant_name": "Wrong Visitor"}
    )
    pdf_path = tmp_path / "ambiguous-applicant.pdf"
    pdf_path.write_bytes(b"ambiguous-applicant")
    pages = (
        _snapshot_page(
            0,
            "Applicant: Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="intake_form",
        ),
        _snapshot_page(
            1,
            "Registry Name Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="registry_extract",
        ),
        _snapshot_page(
            2,
            "Applicant: Mallory Visitor",
            case_ids=(row.case_id,),
            applicants=("Mallory Visitor",),
            page_category="intake_form",
        ),
        _snapshot_page(
            3,
            "Registry Name Mallory Visitor",
            case_ids=(row.case_id,),
            applicants=("Mallory Visitor",),
            page_category="registry_extract",
        ),
        _snapshot_page(
            4,
            "MIB Fee Receipt\nAmount $809",
            case_ids=(row.case_id,),
            page_category="fee_receipt",
        ),
    )

    scoped = score_finalizer._scoped_visible_evidence(
        VisibleOcrSnapshot(source_sha256="0" * 64, pages=pages),
        row,
    )
    result = VisibleScoreFinalizer(
        _published_snapshot_store(pdf_path, pages)
    )(pdf_path, row)

    assert scoped.identity_text == "\f\f\f\fMIB Fee Receipt\nAmount $809"
    assert result.applicant_name == "Wrong Visitor"
    assert result.adjudication == "NEEDS_REVIEW"


def test_structured_safety_demotes_approval_without_scoped_fee_evidence(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"adjudication": "APPROVED", "confidence": 0.8}
    )
    pdf_path = tmp_path / "missing-fee.pdf"
    pdf_path.write_bytes(b"missing-fee")
    pages = (
        _snapshot_page(
            0,
            "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="intake_form",
        ),
        _snapshot_page(
            1,
            "Planetary Registry Extract\nRegistry Name Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="registry_extract",
        ),
    )

    result = VisibleScoreFinalizer(
        _published_snapshot_store(pdf_path, pages)
    )(pdf_path, row)

    assert result.adjudication == "NEEDS_REVIEW"


def test_structured_safety_preserves_high_confidence_approval_without_fee_retry(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"adjudication": "APPROVED", "confidence": 0.98}
    )
    pdf_path = tmp_path / "high-confidence-missing-fee.pdf"
    pdf_path.write_bytes(b"high-confidence-missing-fee")
    pages = (
        _snapshot_page(
            0,
            "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="intake_form",
        ),
        _snapshot_page(
            1,
            "Planetary Registry Extract\nRegistry Name Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="registry_extract",
        ),
    )

    result = VisibleScoreFinalizer(
        _published_snapshot_store(pdf_path, pages)
    )(pdf_path, row)

    assert result.adjudication == "APPROVED"


def test_structured_safety_demotes_non_intake_sponsor_id_conflict(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"adjudication": "APPROVED", "confidence": 0.9}
    )
    pdf_path = tmp_path / "sponsor-conflict.pdf"
    pdf_path.write_bytes(b"sponsor-conflict")
    pages = (
        _snapshot_page(
            0,
            "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="intake_form",
        ),
        _snapshot_page(
            1,
            "Planetary Registry Extract\n"
            "Registry Name Ada Visitor\n"
            "Sponsor ID: SPN-9999",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="registry_extract",
        ),
        _snapshot_page(
            2,
            "MIB Fee Receipt\nAmount $809",
            case_ids=(row.case_id,),
            page_category="fee_receipt",
        ),
    )

    result = VisibleScoreFinalizer(
        _published_snapshot_store(pdf_path, pages)
    )(pdf_path, row)

    assert result.adjudication == "NEEDS_REVIEW"


def test_structured_safety_uses_exact_unreadable_only_on_intake(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"adjudication": "APPROVED", "confidence": 0.9}
    )

    def finalized(name: str, intake_marker: str, receipt_marker: str) -> PredictionRow:
        pdf_path = tmp_path / f"{name}.pdf"
        pdf_path.write_bytes(name.encode())
        pages = (
            _snapshot_page(
                0,
                f"FORM I-8090 Work Authorization\nApplicant: Ada Visitor\n{intake_marker}",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="intake_form",
            ),
            _snapshot_page(
                1,
                "Planetary Registry Extract\nRegistry Name Ada Visitor",
                case_ids=(row.case_id,),
                applicants=("Ada Visitor",),
                page_category="registry_extract",
            ),
            _snapshot_page(
                2,
                f"MIB Fee Receipt\nAmount $809\n{receipt_marker}",
                case_ids=(row.case_id,),
                page_category="fee_receipt",
            ),
        )
        return VisibleScoreFinalizer(
            _published_snapshot_store(pdf_path, pages)
        )(pdf_path, row)

    assert finalized("unreadable", "UNREADABLE", "").adjudication == "NEEDS_REVIEW"
    assert finalized("redacted", "", "REDACTED").adjudication == "APPROVED"
    assert finalized("unreadable-receipt", "", "UNREADABLE").adjudication == "APPROVED"


def test_foreign_or_ambiguous_pages_neither_establish_nor_poison_safety(
    tmp_path: Path,
) -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict() | {"adjudication": "APPROVED", "confidence": 0.8}
    )

    safe_pdf = tmp_path / "foreign-sponsor.pdf"
    safe_pdf.write_bytes(b"foreign-sponsor")
    safe_pages = (
        _snapshot_page(
            0,
            "FORM I-8090 Work Authorization\nApplicant: Ada Visitor",
            case_ids=(row.case_id,),
            applicants=("Ada Visitor",),
            page_category="intake_form",
        ),
        _snapshot_page(
            1,
            "MIB Fee Receipt\nAmount $809",
            case_ids=(row.case_id,),
            page_category="fee_receipt",
        ),
        _snapshot_page(
            2,
            "Registry Name Mallory Visitor\nSponsor ID: SPN-9999",
            case_ids=(row.case_id,),
            applicants=("Mallory Visitor",),
            page_category="registry_extract",
        ),
    )
    safe = VisibleScoreFinalizer(
        _published_snapshot_store(safe_pdf, safe_pages)
    )(safe_pdf, row)

    missing_pdf = tmp_path / "foreign-fee.pdf"
    missing_pdf.write_bytes(b"foreign-fee")
    missing_pages = (
        safe_pages[0],
        _snapshot_page(
            1,
            "MIB Fee Receipt\nAmount $809",
            case_ids=(row.case_id,),
            applicants=("Mallory Visitor",),
            page_category="fee_receipt",
        ),
    )
    missing = VisibleScoreFinalizer(
        _published_snapshot_store(missing_pdf, missing_pages)
    )(missing_pdf, row)

    assert safe.adjudication == "APPROVED"
    assert missing.adjudication == "NEEDS_REVIEW"


def test_unknown_clean_waiver_denial_softens_for_non_dip_visa() -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict()
        | {
            "applicant_name": "unknown",
            "visa_class": "XW-2",
            "fee_status": "waived",
            "risk_flags": "none",
            "adjudication": "DENIED",
            "confidence": 0.96,
        }
    )

    softened = score_heads.apply_denial_to_review_softening(row)

    assert softened.adjudication == "NEEDS_REVIEW"
    assert softened.confidence == 0.70


def test_unknown_clean_waiver_does_not_soften_transit_denial() -> None:
    row = PredictionRow.from_mapping(
        _review_row().to_dict()
        | {
            "applicant_name": "unknown",
            "visa_class": "TRANSIT-7",
            "fee_status": "waived",
            "risk_flags": "none",
            "adjudication": "DENIED",
            "confidence": 0.96,
        }
    )

    assert score_heads.apply_denial_to_review_softening(row) == row


def test_missing_snapshot_returns_unchanged_base_row(tmp_path: Path) -> None:
    pdf_path = tmp_path / "case.pdf"
    pdf_path.write_bytes(b"source")
    row = _review_row()

    assert VisibleScoreFinalizer(VisibleOcrTextStore())(pdf_path, row) == row


def test_mismatched_snapshot_returns_unchanged_base_row(tmp_path: Path) -> None:
    pdf_path = tmp_path / "case.pdf"
    pdf_path.write_bytes(b"original")
    row = _review_row()
    store = _published_store(pdf_path, "Finding: DENIED")
    pdf_path.write_bytes(b"changed")

    assert VisibleScoreFinalizer(store)(pdf_path, row) == row


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
