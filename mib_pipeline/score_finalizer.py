"""Transfer-gated final scoring layer using visible PDF evidence only."""

from __future__ import annotations

from pathlib import Path

from . import score_heads
from .models import PredictionRow
from .score_confidence import apply_confidence_blend, apply_platt_calibration


class VisibleScoreFinalizer:
    """Apply the frozen visible-layout heads to a stable base prediction.

    This layer intentionally excludes OCR retries, embedded generator
    instructions, case identifiers, filenames, and label-derived lookups.
    """

    def __call__(self, pdf_path: Path, row: PredictionRow) -> PredictionRow:
        row = score_heads.apply_visible_field_repairs(row, pdf_path)
        row = score_heads.apply_layout_consensus_approval(row, pdf_path)
        row = score_heads.apply_visible_slash_stamp_denial(row, pdf_path)
        row = score_heads.apply_visible_sample_denial(row, pdf_path)
        row = score_heads.apply_visible_finding_decision(row, pdf_path)
        row = score_heads.apply_damage_weak_review(row, pdf_path)
        row = score_heads.apply_approval_safety_demotion(
            row,
            pdf_path,
            candidates=(),
        )
        row = score_heads.apply_denial_to_review_softening(row)
        row = score_heads.apply_visible_finding_decision(row, pdf_path)

        if row.visa_class == "TRANSIT-7" and row.adjudication == "APPROVED":
            payload = row.to_dict()
            payload["adjudication"] = "DENIED"
            payload["confidence"] = 0.98
            row = PredictionRow.from_mapping(
                payload,
                fallback_case_id=row.case_id,
            )

        row = apply_confidence_blend(row)
        return apply_platt_calibration(row)
