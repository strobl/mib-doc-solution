"""Single construction point for the exact submitted production graph."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adjudication import AdjudicationEngine, GeneralizablePolicyExceptionStore
from .confidence import ConfidenceCalibrator
from .decision_recovery import ReviewDenialRecoveryAdjudicator
from .extraction import VisibleEvidenceExtractor
from .ingestion import DocumentRenderer
from .models import PredictionRow, RowValidationError
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


class IsolatedProductionProcessor:
    """Run one exact production graph per disposable case subprocess."""

    def __init__(
        self,
        *,
        attempts: int = 2,
        timeout_seconds: int = 180,
        run_factory: Callable[..., Any] = subprocess.run,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be positive")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self._attempts = attempts
        self._timeout_seconds = timeout_seconds
        self._run_factory = run_factory

    def process_case(self, pdf_path: Path) -> PredictionRow | None:
        path = Path(pdf_path)
        command = [
            sys.executable,
            "-B",
            "-m",
            "mib_pipeline.case_worker",
            str(path),
        ]
        for _attempt in range(self._attempts):
            try:
                completed = self._run_factory(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=self._timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if completed.returncode != 0:
                continue
            try:
                payload = json.loads(completed.stdout.decode("utf-8"))
                if not isinstance(payload, dict):
                    continue
                return PredictionRow.from_mapping(
                    payload,
                    fallback_case_id=path.stem,
                )
            except (
                AttributeError,
                json.JSONDecodeError,
                RowValidationError,
                UnicodeDecodeError,
            ):
                continue
        return None


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


def build_isolated_production_processor() -> IsolatedProductionProcessor:
    """Return the crash-contained processor used by the submitted batch CLI."""

    return IsolatedProductionProcessor()
