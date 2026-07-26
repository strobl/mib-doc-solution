from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devtools.experiment_control import canonical_json, require_aggregate_only
from devtools.grouped_recovery_evidence import load_recovery_audit
from devtools.recovery_audit_run import (
    RecoveryAuditRunError,
    main,
    run_recovery_audit,
)
from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.ingestion import Rect
from mib_pipeline.models import PredictionRow
from mib_pipeline.provenance import make_ocr_provenance
from mib_pipeline.recovery_audit import (
    CandidateValidationPolicy,
    RecoveryAuditOverlay,
    VisibleRecoveryResult,
    recovered_field_audit,
    unchanged_field_audit,
)
from mib_pipeline.resolution import FieldState, ResolvedField


SOURCE_REVISION = "a" * 40
SOURCE_SHA256 = "b" * 64
APPLICANT = "Audit Applicant"
BOX = Rect(10, 20, 90, 44)


def _resolved(
    field_name: str,
    value: str | None,
    *,
    state: FieldState,
) -> ResolvedField:
    return ResolvedField(
        field_name=field_name,
        state=state,
        value=value,
        winning_evidence=None,
        considered=(),
        reason="test",
    )


def _result(case_id: str) -> VisibleRecoveryResult:
    candidate = CandidateEvidence(
        field_name="risk_flags",
        value="none",
        evidence_type=EvidenceType.INTAKE_FORM,
        page_index=0,
        box=BOX,
        legible=True,
        superseded=False,
        ocr_confidence=0.98,
        source="visible_ocr",
        case_id_hint=case_id,
        applicant_hint=APPLICANT,
        ocr_provenance=(
            make_ocr_provenance(
                source_sha256=SOURCE_SHA256,
                page_index=0,
                view_box=BOX,
                applicant_scope=APPLICANT,
                route_id="targeted_rapidocr",
                engine_id="rapidocr:test-models",
                view_id="rendered_page",
            ),
        ),
    )
    recovered, validation = recovered_field_audit(
        primary=_resolved(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        ),
        serialization_before="none",
        serialization_after="none",
        candidate=candidate,
        recovery_source="targeted_rapidocr",
        linked_recovery_scope=APPLICANT,
        validation_policy=CandidateValidationPolicy(
            expected_field_name="risk_flags",
            expected_source_sha256=SOURCE_SHA256,
            expected_page_index=0,
            expected_applicant_scope=APPLICANT,
            minimum_confidence=0.90,
        ),
    )
    assert validation.accepted and recovered is not None
    unchanged = unchanged_field_audit(
        primary=_resolved(
            "fee_status",
            None,
            state=FieldState.UNKNOWN,
        ),
        serialized_value="unknown",
        linked_recovery_scope=APPLICANT,
    )
    row = PredictionRow.from_mapping(
        {
            "case_id": case_id,
            "risk_flags": "none",
            "fee_status": "unknown",
        },
        fallback_case_id=case_id,
    )
    return VisibleRecoveryResult(
        row=row,
        audit=RecoveryAuditOverlay.from_fields(
            case_id=case_id,
            fields=(recovered, unchanged),
        ),
    )


class _AuditedProcessor:
    def process_case_with_audit(self, pdf_path: Path) -> VisibleRecoveryResult:
        return _result(pdf_path.stem)


class _ProductionWrapper:
    def __init__(self, processor: object | None = None) -> None:
        self.processor = processor or _AuditedProcessor()


class RecoveryAuditRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        for case_id in ("MIB-000002", "MIB-000001"):
            (self.input_dir / f"{case_id}.pdf").touch()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_run_emits_only_deterministic_identity_free_aggregate_counters(self):
        first = run_recovery_audit(
            input_dir=self.input_dir,
            source_revision=SOURCE_REVISION.upper(),
            max_workers=2,
            processor_factory=_ProductionWrapper,
        )
        second = run_recovery_audit(
            input_dir=self.input_dir,
            source_revision=SOURCE_REVISION,
            max_workers=1,
            processor_factory=_ProductionWrapper,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first,
            {
                "counts": {
                    "recovered_field_count": 2,
                    "recovered_field_complete_provenance_count": 2,
                    "serialization_default_used_as_evidence_count": 0,
                },
                "source_revision_sha": SOURCE_REVISION,
            },
        )
        require_aggregate_only(first)
        serialized = canonical_json(first)
        for forbidden in (
            "MIB-",
            ".pdf",
            APPLICANT,
            "risk_flags",
            "fee_status",
            "none",
            "unknown",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_cli_writes_canonical_json_accepted_by_grouped_evidence_loader(self):
        output = self.root / "audit.json"
        with patch(
            "devtools.recovery_audit_run.build_production_processor",
            _ProductionWrapper,
        ):
            exit_code = main(
                [
                    "--input-dir",
                    str(self.input_dir),
                    "--output",
                    str(output),
                    "--source-revision",
                    SOURCE_REVISION,
                    "--max-workers",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            output.read_text(encoding="utf-8"),
            canonical_json(parsed) + "\n",
        )
        require_aggregate_only(parsed)
        self.assertEqual(set(parsed["counts"]), {
            "recovered_field_count",
            "recovered_field_complete_provenance_count",
            "serialization_default_used_as_evidence_count",
        })
        loaded = load_recovery_audit(
            output,
            expected_source_revision_sha=SOURCE_REVISION,
        )
        self.assertEqual(loaded.recovered_field_count, 2)
        self.assertEqual(loaded.recovered_field_complete_provenance_count, 2)
        self.assertEqual(
            loaded.serialization_default_used_as_evidence_count,
            0,
        )

    def test_incomplete_recovered_provenance_fails_closed_without_replacing_output(self):
        broken = _result("MIB-000001")
        recovered = broken.audit.field("risk_flags")
        candidate = recovered.winning_evidence
        assert candidate is not None
        object.__setattr__(candidate, "ocr_provenance", ())

        class BrokenProcessor:
            def process_case_with_audit(self, pdf_path: Path) -> VisibleRecoveryResult:
                del pdf_path
                return broken

        output = self.root / "existing.json"
        output.write_text("existing\n", encoding="utf-8")
        with patch(
            "devtools.recovery_audit_run.build_production_processor",
            lambda: _ProductionWrapper(BrokenProcessor()),
        ):
            exit_code = main(
                [
                    "--input-dir",
                    str(self.input_dir),
                    "--output",
                    str(output),
                    "--source-revision",
                    SOURCE_REVISION,
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")

    def test_invalid_composition_source_revision_and_worker_bounds_are_rejected(self):
        with self.assertRaisesRegex(
            RecoveryAuditRunError,
            "composition root",
        ):
            run_recovery_audit(
                input_dir=self.input_dir,
                source_revision=SOURCE_REVISION,
                processor_factory=lambda: object(),
            )
        with self.assertRaisesRegex(RecoveryAuditRunError, "source revision"):
            run_recovery_audit(
                input_dir=self.input_dir,
                source_revision="deadbeef",
                processor_factory=_ProductionWrapper,
            )
        with self.assertRaisesRegex(RecoveryAuditRunError, "max_workers"):
            run_recovery_audit(
                input_dir=self.input_dir,
                source_revision=SOURCE_REVISION,
                max_workers=5,
                processor_factory=_ProductionWrapper,
            )

    def test_empty_input_fails_before_processor_construction(self):
        empty = self.root / "empty"
        empty.mkdir()
        constructed = False

        def factory() -> _ProductionWrapper:
            nonlocal constructed
            constructed = True
            return _ProductionWrapper()

        with self.assertRaisesRegex(RecoveryAuditRunError, "no PDF"):
            run_recovery_audit(
                input_dir=empty,
                source_revision=SOURCE_REVISION,
                processor_factory=factory,
            )
        self.assertFalse(constructed)


if __name__ == "__main__":
    unittest.main()
