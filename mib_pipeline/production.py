"""Single composition root for the submitted production processor.

The offline CLI and development evaluation harness must execute the same
processor graph.  Keeping that graph here prevents evaluation tooling from
silently drifting behind late recovery and confidence stages.
"""

from __future__ import annotations

from .adjudication import AdjudicationEngine, GeneralizablePolicyExceptionStore
from .confidence import ConfidenceCalibrator
from .decision_recovery import ReviewDenialRecoveryAdjudicator
from .extraction import VisibleEvidenceExtractor
from .ingestion import DocumentRenderer
from .output_confidence import (
    OutputConfidenceRecalibrationProcessor,
    OutputConfidenceRecalibrator,
)
from .rapid_recovery import RapidOutputRecoveryProcessor
from .resolution import CaseLinker, EvidencePrecedenceResolver


def build_production_processor() -> OutputConfidenceRecalibrationProcessor:
    """Build the exact identity-free processor used by the submitted runtime."""

    return OutputConfidenceRecalibrationProcessor(
        processor=RapidOutputRecoveryProcessor(
            renderer=DocumentRenderer(),
            primary_extractor=VisibleEvidenceExtractor(
                packet_page_type_markers=True,
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
