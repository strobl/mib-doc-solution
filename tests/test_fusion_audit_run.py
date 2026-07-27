from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from devtools.experiment_control import canonical_json, require_aggregate_only
from devtools.fusion_audit_contract import (
    FUSION_AUDIT_COMPARISON_SCOPE,
    FUSION_AUDIT_INVOCATION_SCOPE,
)
from devtools.fusion_audit_run import (
    REQUIRED_FUSION_COUNTS,
    FusionAuditRunError,
    case_id_set_sha256,
    cohort_tree_sha256,
    main,
    run_fusion_audit,
)
from devtools.grouped_fusion_evidence import load_fusion_audit
from mib_pipeline.models import PredictionRow
from mib_pipeline.resolution import ResolvedCase


SOURCE_REVISION = "a" * 40
BASE_COUNTS = {
    "changed_field_count": 2,
    "changed_field_complete_provenance_count": 2,
    "clean_higher_authority_override_count": 0,
    "binding_authority_override_count": 0,
    "text_layer_winner_count": 0,
    "serialization_default_used_as_evidence_count": 0,
    "correlated_views_collapsed": 3,
    "independent_agreement_resolutions": 1,
    "same_rank_contested_count": 1,
    "cross_applicant_candidates_excluded": 2,
}


def _resolved(counts: object = BASE_COUNTS) -> ResolvedCase:
    return ResolvedCase(
        case_id="internal-test-case",
        active_applicant=None,
        fields=MappingProxyType({}),
        unresolved_linkage=False,
        unresolved_reasons=(),
        fusion_audit_counts=counts,  # type: ignore[arg-type]
    )


class _Resolver:
    def __init__(
        self,
        *,
        counts: object = BASE_COUNTS,
        fusion_enabled: bool = True,
    ) -> None:
        self.counts = counts
        self.fusion_enabled = fusion_enabled

    def resolve(self, linked_case: object) -> ResolvedCase:
        del linked_case
        return _resolved(self.counts)


class _InnerProcessor:
    def __init__(
        self,
        resolver: object | None = None,
        *,
        invocations_per_case: int = 2,
        omit: bool = False,
    ) -> None:
        self._resolver = resolver or _Resolver()
        self._invocations_per_case = invocations_per_case
        self._omit = omit

    def process_case(self, pdf_path: Path) -> PredictionRow | None:
        resolved = None
        for _ in range(self._invocations_per_case):
            resolved = self._resolver.resolve(object())
        if self._omit:
            return None
        observer = getattr(self, "_fusion_audit_observer", None)
        if observer is not None and resolved is not None:
            observer(resolved.fusion_audit_counts)
        return PredictionRow.from_mapping(
            {"case_id": pdf_path.stem},
            fallback_case_id=pdf_path.stem,
        )


class _ProductionWrapper:
    def __init__(self, processor: object | None = None) -> None:
        self.processor = processor or _InnerProcessor()

    def process_case(self, pdf_path: Path) -> PredictionRow | None:
        return self.processor.process_case(pdf_path)  # type: ignore[union-attr]


class FusionAuditRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        for case_id in ("MIB-000002", "MIB-000001", "MIB-000003"):
            (self.input_dir / f"{case_id}.pdf").touch()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _factory(
        self,
        *,
        counts: object = BASE_COUNTS,
        fusion_enabled: bool = True,
        invocations_per_case: int = 2,
        omit: bool = False,
    ):
        return lambda: _ProductionWrapper(
            _InnerProcessor(
                _Resolver(
                    counts=counts,
                    fusion_enabled=fusion_enabled,
                ),
                invocations_per_case=invocations_per_case,
                omit=omit,
            )
        )

    def test_serial_and_parallel_runs_are_deterministic_and_aggregate_only(self):
        serial = run_fusion_audit(
            input_dir=self.input_dir,
            source_revision=SOURCE_REVISION.upper(),
            max_workers=1,
            processor_factory=self._factory(),
        )
        parallel = run_fusion_audit(
            input_dir=self.input_dir,
            source_revision=SOURCE_REVISION,
            max_workers=4,
            processor_factory=self._factory(),
        )

        expected_counts = {
            name: value * 3 for name, value in BASE_COUNTS.items()
        }
        self.assertEqual(serial, parallel)
        self.assertEqual(
            parallel,
            {
                "case_id_set_sha256": case_id_set_sha256(
                    ("MIB-000001", "MIB-000002", "MIB-000003")
                ),
                "comparison_scope": FUSION_AUDIT_COMPARISON_SCOPE,
                "counts": expected_counts,
                "input_pdf_count": 3,
                "input_tree_sha256": cohort_tree_sha256(
                    tuple(sorted(self.input_dir.glob("*.pdf")))
                ),
                "invocation_scope": FUSION_AUDIT_INVOCATION_SCOPE,
                "source_revision_sha": SOURCE_REVISION,
            },
        )
        require_aggregate_only(parallel)
        serialized = canonical_json(parallel)
        self.assertEqual(set(parallel["counts"]), set(REQUIRED_FUSION_COUNTS))
        for forbidden in (
            "MIB-",
            ".pdf",
            "internal-test-case",
            "Audit Applicant",
            "field_value",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_rejected_optional_resolver_path_is_not_counted(self):
        transient_counts = dict(BASE_COUNTS)
        transient_counts["changed_field_count"] = 9
        transient_counts["changed_field_complete_provenance_count"] = 9

        class RejectedOptionalInner:
            def __init__(self) -> None:
                self._resolver = _Resolver()

            def process_case(self, pdf_path: Path) -> PredictionRow:
                accepted = _resolved(BASE_COUNTS)
                _discarded = _resolved(transient_counts)
                observer = getattr(self, "_fusion_audit_observer", None)
                if observer is not None:
                    observer(accepted.fusion_audit_counts)
                return PredictionRow.from_mapping(
                    {"case_id": pdf_path.stem},
                    fallback_case_id=pdf_path.stem,
                )

        payload = run_fusion_audit(
            input_dir=self.input_dir,
            source_revision=SOURCE_REVISION,
            processor_factory=lambda: _ProductionWrapper(
                RejectedOptionalInner()
            ),
        )

        self.assertEqual(
            payload["counts"]["changed_field_count"],
            BASE_COUNTS["changed_field_count"] * 3,
        )
        self.assertNotEqual(
            payload["counts"]["changed_field_count"],
            transient_counts["changed_field_count"] * 3,
        )

    def test_cli_is_atomic_canonical_and_grouped_loader_compatible(self):
        output = self.root / "fusion-audit.json"
        output.write_text("old output\n", encoding="utf-8")
        with patch(
            "devtools.fusion_audit_run.build_production_processor",
            self._factory(),
        ):
            exit_code = main(
                [
                    "--input-dir",
                    str(self.input_dir),
                    "--output",
                    str(output),
                    "--source-revision",
                    SOURCE_REVISION.upper(),
                    "--max-workers",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            output.read_text(encoding="utf-8"),
            canonical_json(parsed) + "\n",
        )
        loaded = load_fusion_audit(
            output,
            expected_source_revision_sha=SOURCE_REVISION,
        )
        self.assertEqual(loaded.changed_field_count, 6)
        self.assertEqual(
            loaded.changed_field_complete_provenance_count,
            6,
        )
        self.assertEqual(loaded.correlated_views_collapsed, 9)
        self.assertEqual(loaded.cross_applicant_candidates_excluded, 6)

    def test_non_fusion_or_missing_resolver_is_rejected(self):
        with self.assertRaisesRegex(FusionAuditRunError, "not fusion-enabled"):
            run_fusion_audit(
                input_dir=self.input_dir,
                source_revision=SOURCE_REVISION,
                processor_factory=self._factory(fusion_enabled=False),
            )

        class MissingResolverInner:
            def process_case(self, pdf_path: Path) -> PredictionRow:
                return PredictionRow.from_mapping(
                    {"case_id": pdf_path.stem},
                    fallback_case_id=pdf_path.stem,
                )

        with self.assertRaisesRegex(FusionAuditRunError, "resolver"):
            run_fusion_audit(
                input_dir=self.input_dir,
                source_revision=SOURCE_REVISION,
                processor_factory=lambda: _ProductionWrapper(
                    MissingResolverInner()
                ),
            )

    def test_missing_negative_boolean_and_overlarge_provenance_counts_fail(self):
        malformed: list[tuple[str, dict[str, object]]] = []
        missing = dict(BASE_COUNTS)
        missing.pop("same_rank_contested_count")
        malformed.append(("missing", missing))
        negative = dict(BASE_COUNTS)
        negative["correlated_views_collapsed"] = -1
        malformed.append(("non-negative", negative))
        boolean = dict(BASE_COUNTS)
        boolean["independent_agreement_resolutions"] = True
        malformed.append(("non-negative", boolean))
        provenance = dict(BASE_COUNTS)
        provenance["changed_field_complete_provenance_count"] = 3
        malformed.append(("cannot exceed", provenance))

        for message, counts in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(FusionAuditRunError, message):
                    run_fusion_audit(
                        input_dir=self.input_dir,
                        source_revision=SOURCE_REVISION,
                        processor_factory=self._factory(counts=counts),
                    )

    def test_unsafe_counts_fail_closed_and_do_not_replace_output(self):
        for unsafe_name in (
            "clean_higher_authority_override_count",
            "binding_authority_override_count",
            "text_layer_winner_count",
            "serialization_default_used_as_evidence_count",
        ):
            with self.subTest(unsafe_name=unsafe_name):
                counts = dict(BASE_COUNTS)
                counts[unsafe_name] = 1
                output = self.root / f"{unsafe_name}.json"
                output.write_text("existing\n", encoding="utf-8")
                with patch(
                    "devtools.fusion_audit_run.build_production_processor",
                    self._factory(counts=counts),
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
                self.assertEqual(
                    output.read_text(encoding="utf-8"),
                    "existing\n",
                )

    def test_incomplete_output_and_missing_invocations_fail_closed(self):
        with self.assertRaisesRegex(FusionAuditRunError, "incomplete"):
            run_fusion_audit(
                input_dir=self.input_dir,
                source_revision=SOURCE_REVISION,
                processor_factory=self._factory(omit=True),
            )
        with self.assertRaisesRegex(FusionAuditRunError, "no accepted final"):
            run_fusion_audit(
                input_dir=self.input_dir,
                source_revision=SOURCE_REVISION,
                processor_factory=self._factory(invocations_per_case=0),
            )

    def test_revision_worker_bounds_and_empty_input_are_rejected(self):
        with self.assertRaisesRegex(FusionAuditRunError, "source revision"):
            run_fusion_audit(
                input_dir=self.input_dir,
                source_revision="deadbeef",
                processor_factory=self._factory(),
            )
        with self.assertRaisesRegex(FusionAuditRunError, "max_workers"):
            run_fusion_audit(
                input_dir=self.input_dir,
                source_revision=SOURCE_REVISION,
                max_workers=5,
                processor_factory=self._factory(),
            )
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(FusionAuditRunError, "no PDF"):
            run_fusion_audit(
                input_dir=empty,
                source_revision=SOURCE_REVISION,
                processor_factory=self._factory(),
            )


if __name__ == "__main__":
    unittest.main()
