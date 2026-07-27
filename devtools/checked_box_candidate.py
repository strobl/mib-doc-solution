"""Development-only, fail-closed visible checkbox candidate.

The production runtime has no checkbox route.  WO-14 nevertheless requires a
bounded measurement, so this module supplies one additive candidate without
making it importable from ``mib_pipeline``.  It recognizes only a complete
three-option fee group with one pixel-confirmed check mark and two
pixel-confirmed empty boxes.  Every incomplete or ambiguous layout abstains.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable

from mib_pipeline import (
    CandidateEvidence,
    EvidenceType,
    OcrLine,
    OcrToken,
    Rect,
    RenderedCase,
    RenderedPage,
    UntrustedContentFilter,
    VisualCueDetector,
    group_ocr_lines,
)


_FEE_OPTIONS = ("paid", "unpaid", "waived")
_CASE_ID_RE = re.compile(r"\bMIB-[0-9]{6}\b")
_MINIMUM_OCR_CONFIDENCE = 0.80
_MINIMUM_PIXEL_QUALITY = 0.80


@dataclass(frozen=True)
class CheckboxObservation:
    """One unambiguous square immediately left of an OCR option label."""

    state: str
    box: Rect
    quality: float

    def __post_init__(self) -> None:
        if self.state not in {"checked", "empty"}:
            raise ValueError("checkbox state must be checked or empty")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("checkbox quality must be within [0,1]")


class RecordingOcrEngine:
    """Delegate OCR while retaining deterministic page tokens for one wrapper."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._cache: dict[tuple[int, str], tuple[OcrToken, ...]] = {}
        self._inflight: set[tuple[int, str]] = set()
        self._condition = threading.Condition()

    @property
    def provenance_id(self) -> str:
        declared = getattr(self._delegate, "provenance_id", None)
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
        engine_type = type(self._delegate)
        return f"{engine_type.__module__}.{engine_type.__qualname__}"

    @staticmethod
    def _key(page: RenderedPage) -> tuple[int, str]:
        return page.index, hashlib.sha256(page.image_png).hexdigest()

    def read_page(self, page: RenderedPage) -> tuple[OcrToken, ...]:
        key = self._key(page)
        with self._condition:
            while key in self._inflight and key not in self._cache:
                self._condition.wait()
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            self._inflight.add(key)
        try:
            tokens = tuple(self._delegate.read_page(page))
        except BaseException:
            with self._condition:
                self._inflight.discard(key)
                self._condition.notify_all()
            raise
        with self._condition:
            existing = self._cache.setdefault(key, tokens)
            self._inflight.discard(key)
            self._condition.notify_all()
            return existing

    def tokens_for(self, page: RenderedPage) -> tuple[OcrToken, ...]:
        return self.read_page(page)


class CheckboxPixelDetector:
    """Detect a bounded square and classify only clear empty or checked state."""

    def __init__(self, cue_detector: VisualCueDetector | None = None) -> None:
        self._cues = cue_detector or VisualCueDetector()

    def prepare_page(self, grayscale: Any) -> Any:
        return self._cues.prepare_page(grayscale)

    def ordinary_cues(
        self,
        line: OcrLine,
        page_pixels: Any,
    ) -> tuple[str, ...]:
        return self._cues.cues_for_line(line, page_pixels)

    @staticmethod
    def _grayscale_pixels(page_pixels: Any) -> Any:
        try:
            import numpy
        except ImportError as exc:
            raise RuntimeError("numpy is required for checkbox measurement") from exc
        pixels = (
            page_pixels
            if hasattr(page_pixels, "shape")
            else numpy.asarray(page_pixels)
        )
        if len(pixels.shape) == 3:
            pixels = pixels.mean(axis=2)
        return pixels

    def detect(
        self,
        line: OcrLine,
        page_pixels: Any,
    ) -> CheckboxObservation | None:
        pixels = self._grayscale_pixels(page_pixels)
        page_height, page_width = pixels.shape[:2]
        line_top = max(0, min(page_height, int(round(line.box.bottom))))
        line_bottom = max(0, min(page_height, int(round(line.box.top))))
        line_left = max(0, min(page_width, int(round(line.box.left))))
        line_height = max(10, line_bottom - line_top)
        minimum_side = max(8, int(round(line_height * 0.65)))
        maximum_side = min(64, int(round(line_height * 1.35)))
        search_left = max(0, int(round(line_left - line_height * 2.2)))
        search_right = max(search_left, int(round(line_left - line_height * 0.15)))
        search_top = max(0, int(round(line_top - line_height * 0.30)))
        search_bottom = min(
            page_height,
            int(round(line_bottom + line_height * 0.30)),
        )
        if search_right - search_left < minimum_side:
            return None

        ink = pixels < 145
        matches: list[tuple[float, int, int, int, float]] = []
        for side in range(minimum_side, maximum_side + 1):
            border = max(1, min(3, side // 8))
            inset = max(border + 1, int(round(side * 0.20)))
            if side - 2 * inset < 3:
                continue
            for top in range(search_top, max(search_top, search_bottom - side) + 1):
                bottom = top + side
                if bottom > page_height:
                    break
                for left in range(search_left, max(search_left, search_right - side) + 1):
                    right = left + side
                    if right > search_right or right > page_width:
                        break
                    square = ink[top:bottom, left:right]
                    edge_ratios = (
                        float(square[:border, :].mean()),
                        float(square[-border:, :].mean()),
                        float(square[:, :border].mean()),
                        float(square[:, -border:].mean()),
                    )
                    if min(edge_ratios) < 0.60:
                        continue
                    inner = square[inset:-inset, inset:-inset]
                    if not inner.size:
                        continue
                    interior_ratio = float(inner.mean())
                    if interior_ratio <= 0.03:
                        state_score = 1.0 - min(1.0, interior_ratio / 0.03)
                    elif 0.06 <= interior_ratio <= 0.50:
                        ys, xs = inner.nonzero()
                        if not len(xs):
                            continue
                        width_span = (int(xs.max()) - int(xs.min()) + 1) / inner.shape[1]
                        height_span = (int(ys.max()) - int(ys.min()) + 1) / inner.shape[0]
                        if width_span < 0.45 or height_span < 0.35:
                            continue
                        state_score = min(1.0, (width_span + height_span) / 2.0)
                    else:
                        continue
                    border_score = sum(edge_ratios) / len(edge_ratios)
                    matches.append(
                        (
                            border_score + state_score,
                            left,
                            top,
                            side,
                            interior_ratio,
                        )
                    )

        if not matches:
            return None
        matches.sort(reverse=True)
        states = {
            "empty" if match[4] <= 0.03 else "checked"
            for match in matches
        }
        if len(states) != 1:
            return None
        best = matches[0]
        _score, left, top, side, _interior_ratio = best
        # Multiple disjoint square candidates beside one label are ambiguous.
        for other in matches[1:]:
            _other_score, other_left, other_top, other_side, _ratio = other
            intersection_width = max(
                0,
                min(left + side, other_left + other_side)
                - max(left, other_left),
            )
            intersection_height = max(
                0,
                min(top + side, other_top + other_side)
                - max(top, other_top),
            )
            overlap = (
                intersection_width * intersection_height
                / min(side * side, other_side * other_side)
            )
            if overlap < 0.60:
                return None
        state = next(iter(states))
        quality = max(0.0, min(1.0, best[0] / 2.0))
        return CheckboxObservation(
            state=state,
            box=Rect(left, top, left + side, top + side),
            quality=quality,
        )


class CheckedFeeOptionExtractor:
    """Add at most one checked fee option after the ordinary extractor."""

    def __init__(
        self,
        *,
        delegate: Any,
        recording_ocr: RecordingOcrEngine,
        detector: CheckboxPixelDetector | None = None,
        content_filter: UntrustedContentFilter | None = None,
    ) -> None:
        self._delegate = delegate
        self._ocr = recording_ocr
        self._detector = detector or CheckboxPixelDetector()
        self._content_filter = content_filter or UntrustedContentFilter()
        self._activity = {
            "pages_scanned": 0,
            "complete_groups": 0,
            "checked_groups": 0,
            "candidates_added": 0,
            "ambiguous_groups": 0,
        }
        self._lock = threading.Lock()

    @staticmethod
    def _page_image(page: RenderedPage) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for checkbox measurement") from exc
        with Image.open(io.BytesIO(page.image_png)) as image:
            return image.convert("L").copy()

    @staticmethod
    def _normalized_line(text: str) -> str:
        return " ".join(text.casefold().strip().split())

    @staticmethod
    def _page_has_expected_case(
        lines: Iterable[OcrLine],
        expected_case_id: str,
    ) -> bool:
        visible_case_ids = {
            case_id
            for line in lines
            for case_id in _CASE_ID_RE.findall(line.text)
        }
        return bool(expected_case_id) and visible_case_ids == {expected_case_id}

    @staticmethod
    def _coherent_group(
        observations: dict[str, tuple[OcrLine, CheckboxObservation]],
    ) -> bool:
        """Require one compact, aligned vertical option group."""

        pairs = tuple(observations[option] for option in _FEE_OPTIONS)
        line_heights = tuple(line.box.height for line, _observation in pairs)
        box_sides = tuple(observation.box.width for _line, observation in pairs)
        if min(line_heights, default=0.0) <= 0.0 or min(box_sides, default=0.0) <= 0.0:
            return False
        reference_height = sorted(line_heights)[1]
        reference_side = sorted(box_sides)[1]
        line_lefts = tuple(line.box.left for line, _observation in pairs)
        box_centers = tuple(
            (observation.box.left + observation.box.right) / 2.0
            for _line, observation in pairs
        )
        if max(line_lefts) - min(line_lefts) > max(12.0, reference_height * 0.75):
            return False
        if max(box_centers) - min(box_centers) > max(3.0, reference_side * 0.25):
            return False
        if any(
            abs(side - reference_side) > max(2.0, reference_side * 0.25)
            for side in box_sides
        ):
            return False
        for line, observation in pairs:
            gap = line.box.left - observation.box.right
            line_center = (line.box.bottom + line.box.top) / 2.0
            box_center = (observation.box.bottom + observation.box.top) / 2.0
            if not 0.0 <= gap <= max(8.0, reference_height * 1.5):
                return False
            if abs(line_center - box_center) > max(3.0, reference_height * 0.45):
                return False
        vertical_centers = sorted(
            (line.box.bottom + line.box.top) / 2.0
            for line, _observation in pairs
        )
        gaps = tuple(
            vertical_centers[index + 1] - vertical_centers[index]
            for index in range(2)
        )
        return all(
            reference_height * 0.70 <= gap <= reference_height * 4.0
            for gap in gaps
        )

    @staticmethod
    def _anchors_bind_group(
        *,
        lines: tuple[OcrLine, ...],
        normalized: dict[str, tuple[OcrLine, ...]],
        expected_case_id: str,
    ) -> bool:
        title_lines = tuple(
            line
            for line in lines
            if CheckedFeeOptionExtractor._normalized_line(line.text)
            == "mib fee receipt"
        )
        status_lines = tuple(
            line
            for line in lines
            if CheckedFeeOptionExtractor._normalized_line(line.text)
            == "fee status"
        )
        case_lines = tuple(
            line
            for line in lines
            if _CASE_ID_RE.findall(line.text) == [expected_case_id]
        )
        if (
            len(title_lines) != 1
            or len(status_lines) != 1
            or len(case_lines) != 1
            or any(len(normalized[option]) != 1 for option in _FEE_OPTIONS)
        ):
            return False
        required_lines = (
            title_lines[0],
            status_lines[0],
            case_lines[0],
            *(normalized[option][0] for option in _FEE_OPTIONS),
        )
        if any(
            line.confidence < _MINIMUM_OCR_CONFIDENCE
            for line in required_lines
        ):
            return False
        option_lines = tuple(normalized[option][0] for option in _FEE_OPTIONS)
        option_centers = tuple(
            (line.box.bottom + line.box.top) / 2.0 for line in option_lines
        )
        if not (option_centers[0] < option_centers[1] < option_centers[2]):
            return False
        first_option_top = option_lines[0].box.bottom
        if any(
            anchor.box.bottom >= first_option_top
            for anchor in (title_lines[0], status_lines[0], case_lines[0])
        ):
            return False
        reference_height = sorted(line.box.height for line in option_lines)[1]
        first_option_center = option_centers[0]
        return all(
            0.0
            < first_option_center
            - (anchor.box.bottom + anchor.box.top) / 2.0
            <= reference_height * 5.0
            for anchor in (title_lines[0], status_lines[0], case_lines[0])
        )

    def _measure_page(
        self,
        rendered_case: RenderedCase,
        page: RenderedPage,
        base_candidates: tuple[CandidateEvidence, ...],
    ) -> tuple[CandidateEvidence, ...]:
        lines = group_ocr_lines(self._ocr.tokens_for(page))
        normalized = {
            option: tuple(
                line
                for line in lines
                if self._normalized_line(line.text) == option
            )
            for option in _FEE_OPTIONS
        }
        if (
            not self._page_has_expected_case(
                lines,
                rendered_case.case_id or "",
            )
            or not self._anchors_bind_group(
                lines=lines,
                normalized=normalized,
                expected_case_id=rendered_case.case_id or "",
            )
        ):
            return ()
        with self._lock:
            self._activity["complete_groups"] += 1
        if any(
            candidate.field_name == "fee_status"
            and candidate.legible
            and not candidate.superseded
            for candidate in base_candidates
        ):
            return ()

        page_image = self._page_image(page)
        page_pixels = self._detector.prepare_page(page_image)
        for line in lines:
            cues = self._detector.ordinary_cues(line, page_pixels)
            if (
                {"sample_denial_watermark", "strikethrough", "correction"}
                & set(cues)
                or self._content_filter.rejection_reason(line.text, cues)
                is not None
            ):
                return ()
        observations: dict[str, tuple[OcrLine, CheckboxObservation]] = {}
        for option in _FEE_OPTIONS:
            line = normalized[option][0]
            observation = self._detector.detect(line, page_pixels)
            if observation is None:
                with self._lock:
                    self._activity["ambiguous_groups"] += 1
                return ()
            observations[option] = (line, observation)
        if (
            any(
                observation.quality < _MINIMUM_PIXEL_QUALITY
                for _line, observation in observations.values()
            )
            or not self._coherent_group(observations)
        ):
            with self._lock:
                self._activity["ambiguous_groups"] += 1
            return ()
        checked = tuple(
            option
            for option, (_line, observation) in observations.items()
            if observation.state == "checked"
        )
        empty = tuple(
            option
            for option, (_line, observation) in observations.items()
            if observation.state == "empty"
        )
        if len(checked) != 1 or len(empty) != 2:
            with self._lock:
                self._activity["ambiguous_groups"] += 1
            return ()
        with self._lock:
            self._activity["checked_groups"] += 1
        selected = checked[0]
        line, observation = observations[selected]
        candidate_box = line.box.union(observation.box)
        return (
            CandidateEvidence(
                field_name="fee_status",
                value=selected,
                evidence_type=EvidenceType.INTAKE_FORM,
                page_index=page.index,
                box=candidate_box,
                legible=True,
                superseded=False,
                ocr_confidence=min(line.confidence, observation.quality),
                visual_cues=(
                    "checked_checkbox",
                    "complete_fee_option_group",
                ),
                case_id_hint=rendered_case.case_id,
                applicant_hint=None,
            ),
        )

    def extract(self, rendered_case: RenderedCase) -> tuple[CandidateEvidence, ...]:
        base_candidates = tuple(self._delegate.extract(rendered_case))
        recovered: list[CandidateEvidence] = []
        for page in rendered_case.pages:
            with self._lock:
                self._activity["pages_scanned"] += 1
            recovered.extend(
                self._measure_page(
                    rendered_case,
                    page,
                    base_candidates,
                )
            )
        if len(recovered) > 1:
            with self._lock:
                self._activity["ambiguous_groups"] += 1
            return base_candidates
        if len(recovered) == 1:
            with self._lock:
                self._activity["candidates_added"] += 1
        return base_candidates + tuple(recovered)

    def ablation_activity(self) -> dict[str, int]:
        with self._lock:
            return dict(self._activity)
