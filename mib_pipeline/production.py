"""Single construction point for the exact submitted production graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adjudication import AdjudicationEngine, GeneralizablePolicyExceptionStore
from .confidence import ConfidenceCalibrator
from .decision_recovery import ReviewDenialRecoveryAdjudicator
from .extraction import VisibleEvidenceExtractor
from .ingestion import DocumentRenderer
from .models import PredictionRow
from .output_confidence import (
    OutputConfidenceRecalibrationProcessor,
    OutputConfidenceRecalibrator,
)
from .rapid_recovery import RapidOutputRecoveryProcessor
from .resolution import CaseLinker, EvidencePrecedenceResolver
from .score_finalizer import VisibleScoreFinalizer
from .visible_text import VisibleOcrTextStore


@dataclass
class _VisibleFinalizingProcessor:
    """Apply the optional visible-score layer with BatchRunner's fail-closed semantics."""

    processor: OutputConfidenceRecalibrationProcessor
    finalizer: VisibleScoreFinalizer

    def process_case(self, pdf_path: Path) -> PredictionRow | None:
        row = self.processor.process_case(pdf_path)
        if row is None:
            return None
        try:
            return self.finalizer(pdf_path, row)
        except Exception:
            # The score lift is optional. Preserve a valid base prediction if
            # the visible-layout layer cannot safely finalize this case.
            return row


def build_production_processor() -> _VisibleFinalizingProcessor:
    """Construct one fresh processor identical to the two-argument runtime."""

    visible_text_store = VisibleOcrTextStore()
    processor = OutputConfidenceRecalibrationProcessor(
        processor=RapidOutputRecoveryProcessor(
            renderer=DocumentRenderer(),
            primary_extractor=VisibleEvidenceExtractor(
                packet_page_type_markers=True,
                visible_text_store=visible_text_store,
            ),
            linker=CaseLinker(),
            resolver=EvidencePrecedenceResolver(),
            adjudicator=ReviewDenialRecoveryAdjudicator(
                AdjudicationEngine(
                    calibrator=ConfidenceCalibrator.from_pinned_artifact(),
                    exceptions=GeneralizablePolicyExceptionStore.from_pinned_artifact(),
                )
            ),
        ),
        recalibrator=OutputConfidenceRecalibrator.from_pinned_artifact(),
    )
    return _VisibleFinalizingProcessor(
        processor=processor,
        finalizer=VisibleScoreFinalizer(visible_text_store),
    )
