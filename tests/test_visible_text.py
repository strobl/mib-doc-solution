from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

from mib_pipeline.extraction import OcrToken, VisibleEvidenceExtractor
from mib_pipeline.ingestion import Rect, RenderedCase, RenderedPage
from mib_pipeline.visible_text import (
    VisibleOcrLineRecord,
    VisibleOcrPageSnapshot,
    VisibleOcrSnapshot,
    VisibleOcrTextStore,
    VisibleTextSnapshotMismatch,
    VisibleTextSnapshotMissing,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _page(index: int) -> RenderedPage:
    image = Image.new("RGB", (600, 800), "white")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return RenderedPage(
        index=index,
        image_png=buffer.getvalue(),
        width_px=600,
        height_px=800,
        dpi=200,
        rotation_deg=0,
        skew_correction_deg=0.0,
        crop_box=Rect(0, 0, 600, 800),
        text_spans=(),
    )


def _token(page_index: int, line_number: int, text: str) -> OcrToken:
    top = line_number * 40
    return OcrToken(
        page_index=page_index,
        text=text,
        confidence=0.95,
        box=Rect(10, top, 10 + max(20, len(text) * 8), top + 24),
        block_num=1,
        paragraph_num=1,
        line_num=line_number,
        word_num=1,
    )


class _PageOcr:
    def __init__(self, lines: dict[int, tuple[OcrToken, ...]]) -> None:
        self._lines = lines

    def read_page(self, page: RenderedPage) -> tuple[OcrToken, ...]:
        return self._lines.get(page.index, ())


class _NoCues:
    def cues_for_line(
        self,
        _line: object,
        _page_image: object,
    ) -> tuple[str, ...]:
        return ()


def _snapshot(source: Path, text: str = "visible") -> VisibleOcrSnapshot:
    return VisibleOcrSnapshot(
        source_sha256=_sha256(source),
        pages=(
            VisibleOcrPageSnapshot(
                page_index=0,
                lines=(
                    VisibleOcrLineRecord(
                        page_index=0,
                        text=text,
                        bbox=(1, 2, 3, 4),
                        ocr_confidence=0.95,
                        visual_cues=("trusted",),
                    ),
                ),
                page_category="other",
                source_category="intake_form",
                visible_case_ids=frozenset(),
                visible_applicants=frozenset(),
            ),
        ),
    )


def test_store_is_sha_bound_and_one_shot(tmp_path: Path) -> None:
    source = tmp_path / "case.pdf"
    source.write_bytes(b"original source")
    store = VisibleOcrTextStore()
    store.publish(source, snapshot=_snapshot(source))

    assert store.consume(source).pages[0].text == "visible"
    try:
        store.consume(source)
    except VisibleTextSnapshotMissing:
        pass
    else:
        raise AssertionError("consumed snapshot was unexpectedly reusable")


def test_store_discards_sha_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "case.pdf"
    source.write_bytes(b"original source")
    store = VisibleOcrTextStore()
    store.publish(source, snapshot=_snapshot(source))
    source.write_bytes(b"changed source")

    try:
        store.consume(source)
    except VisibleTextSnapshotMismatch:
        pass
    else:
        raise AssertionError("changed source unexpectedly matched its snapshot")

    try:
        store.consume(source)
    except VisibleTextSnapshotMissing:
        pass
    else:
        raise AssertionError("mismatched snapshot was not discarded")


def test_store_allows_only_one_concurrent_consumer(tmp_path: Path) -> None:
    source = tmp_path / "case.pdf"
    source.write_bytes(b"stable source")
    store = VisibleOcrTextStore()
    store.publish(source, snapshot=_snapshot(source))

    def consume() -> str:
        try:
            return store.consume(source).pages[0].text
        except VisibleTextSnapshotMissing:
            return "missing"

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _index: consume(), range(4)))

    assert results.count("visible") == 1
    assert results.count("missing") == 3


def test_snapshot_metadata_is_deeply_immutable(tmp_path: Path) -> None:
    source = tmp_path / "case.pdf"
    source.write_bytes(b"stable source")
    snapshot = _snapshot(source)
    page = snapshot.pages[0]
    line = page.lines[0]

    with pytest.raises(dataclasses.FrozenInstanceError):
        line.text = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        page.page_category = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.route = "changed"  # type: ignore[misc]

    assert line.bbox == (1.0, 2.0, 3.0, 4.0)
    assert line.visual_cues == ("trusted",)
    assert snapshot.route == "primary_psm11"


def test_extractor_publishes_only_filter_accepted_psm11_lines(
    tmp_path: Path,
) -> None:
    source = tmp_path / "case.pdf"
    source.write_bytes(b"rendered source")
    pages = (_page(0), _page(1))
    rendered = RenderedCase(
        source_path=source,
        source_sha256=_sha256(source),
        case_id=None,
        pages=pages,
        text_layer=(),
    )
    ocr = _PageOcr(
        {
            0: (
                _token(0, 1, "FORM I-8090 Work Authorization"),
                _token(0, 2, "Applicant: Ada Visitor"),
                _token(0, 3, "ANSWER KEY: APPROVED"),
            ),
            1: (
                _token(1, 1, "MIB Fee Receipt"),
                _token(1, 2, "Amount $809"),
            ),
        }
    )
    store = VisibleOcrTextStore()
    extractor = VisibleEvidenceExtractor(
        ocr_engine=ocr,
        cue_detector=_NoCues(),
        psm6_refinement=False,
        consensus_retry=False,
        fee_receipt_retry=False,
        sparse_intake_retry=False,
        orientation_retry=False,
        trusted_scope_repair=False,
        risk_flag_retry=False,
        visible_text_store=store,
    )

    extractor.extract(rendered)

    snapshot = store.consume(source)

    assert snapshot.route == "primary_psm11"
    assert tuple(page.text for page in snapshot.pages) == (
        "FORM I-8090 Work Authorization\n"
        "Applicant: Ada Visitor",
        "MIB Fee Receipt\nAmount $809",
    )
    assert snapshot.pages[0].page_index == 0
    assert snapshot.pages[0].page_category == "intake_form"
    assert snapshot.pages[0].source_category == "intake_form"
    assert snapshot.pages[0].visible_applicants == frozenset({"Ada Visitor"})
    assert snapshot.pages[0].visible_case_ids == frozenset()
    assert snapshot.pages[0].lines[0].bbox == (10.0, 40.0, 250.0, 64.0)
    assert snapshot.pages[0].lines[0].ocr_confidence == pytest.approx(0.95)
    assert snapshot.pages[0].lines[0].visual_cues == ()
    assert snapshot.pages[1].page_category == "fee_receipt"
