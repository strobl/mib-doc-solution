from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devtools.experiment_control import canonical_json
from devtools.policy_revalidation_audit_contract import (
    CONTRACT_AUDIT_COUNTS,
    POLICY_REVALIDATION_AUDIT_SCHEMA,
)
from devtools.policy_revalidation_audit_run import (
    PolicyRevalidationAuditRunError,
    main,
    run_policy_revalidation_audit,
)
from devtools.policy_revalidation_contract_probe import run_contract_probes
from mib_pipeline import BatchRunner
from mib_pipeline.decision_recovery import POLICY_AUDIT_COUNT_NAMES
from mib_pipeline.models import PredictionRow


SOURCE_SHA = "a" * 40
FIXTURE_SHA = "b" * 64
BASE_CORE_COUNTS = {
    name: 0 for name in POLICY_AUDIT_COUNT_NAMES
}
BASE_CORE_COUNTS.update(
    {
        "late_recovery_before_revalidation_count": 1,
        "contradicted_synthetic_reason_removed_count": 1,
        "independent_denial_reason_retained_count": 1,
        "review_confidence_restored_count": 1,
        "normal_policy_rerun_count": 1,
        "signed_late_authority_recovery_count": 1,
        "late_adjudication_evidence_preserved_count": 1,
        "late_biohazard_evidence_preserved_count": 1,
    }
)


def _contract_counts() -> dict[str, int]:
    values = {name: 0 for name in CONTRACT_AUDIT_COUNTS}
    values.update(
        {
            "legacy_synthetic_before_late_recovery_count": 1,
            "candidate_late_recovery_before_revalidation_count": 1,
            "candidate_revalidation_after_late_recovery_count": 1,
            "contradicted_synthetic_reason_before_count": 1,
            "contradicted_synthetic_reason_removed_count": 1,
            "independent_denial_reason_retained_count": 1,
            "review_confidence_restored_count": 1,
            "normal_policy_rerun_count": 1,
            "signed_late_authority_recovery_count": 3,
            "late_adjudication_evidence_preserved_count": 3,
            "late_biohazard_evidence_preserved_count": 1,
            "placeholder_guard_probe_count": 35,
            "sentinel_guard_probe_count": 2,
            "serialization_default_guard_probe_count": 35,
            "stale_threshold_guard_probe_count": 2,
            "forced_approval_guard_probe_count": 35,
            "direct_approval_head_guard_probe_count": 1,
        }
    )
    return values


def _row(case_id: str) -> PredictionRow:
    return PredictionRow.from_mapping(
        {
            "case_id": case_id,
            "applicant_name": "Probe",
            "species_code": "HUM",
            "home_world": "Earth",
            "visa_class": "XW-2",
            "sponsor_id": "SPN-1042",
            "arrival_date": "2026-04-17",
            "declared_purpose": "testing",
            "risk_flags": "none",
            "fee_status": "paid",
            "adjudication": "NEEDS_REVIEW",
            "confidence": 0.5,
        }
    )


class _Inner:
    def __init__(
        self,
        counts: object = BASE_CORE_COUNTS,
        *,
        omit_observer: bool = False,
        changed_output: bool = False,
    ) -> None:
        self.counts = counts
        self.omit_observer = omit_observer
        self.changed_output = changed_output

    def process_case(self, pdf_path: Path) -> PredictionRow:
        observer = getattr(self, "_policy_audit_observer", None)
        if observer is not None and not self.omit_observer:
            observer(self.counts)
        row = _row(pdf_path.stem)
        if self.changed_output:
            return PredictionRow.from_mapping(
                {
                    **row.to_dict(),
                    "home_world": "Mars",
                }
            )
        return row


class _Production:
    def __init__(self, inner: _Inner | None = None) -> None:
        self.processor = inner or _Inner()

    def process_case(self, pdf_path: Path) -> PredictionRow:
        return self.processor.process_case(pdf_path)


class PolicyRevalidationAuditRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        for case_id in ("MIB-000003", "MIB-000001", "MIB-000002"):
            (self.input_dir / f"{case_id}.pdf").touch()
        self.predictions = self.root / "predictions.jsonl"
        BatchRunner(_Production(), max_workers=1).run(
            self.input_dir, self.predictions
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _run(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "input_dir": self.input_dir,
            "predictions_path": self.predictions,
            "source_revision": SOURCE_SHA,
            "repeat_index": 1,
            "max_workers": 4,
            "processor_factory": lambda: _Production(),
            "contract_probe": _contract_counts,
            "fixture_digest_provider": lambda: FIXTURE_SHA,
            "checkout_verifier": lambda _root, _revision: None,
        }
        values.update(overrides)
        return run_policy_revalidation_audit(**values)  # type: ignore[arg-type]

    def test_audit_is_accepted_final_prediction_bound_and_identity_free(self):
        payload = self._run()

        self.assertEqual(
            payload["schema_version"],
            POLICY_REVALIDATION_AUDIT_SCHEMA,
        )
        self.assertEqual(payload["source_revision_sha"], SOURCE_SHA)
        self.assertEqual(payload["input_pdf_count"], 3)
        self.assertEqual(payload["repeat_index"], 1)
        self.assertEqual(
            payload["cohort_counts"][
                "accepted_final_policy_result_count"
            ],
            3,
        )
        self.assertEqual(
            payload["cohort_counts"][
                "late_recovery_before_revalidation_count"
            ],
            3,
        )
        self.assertEqual(
            payload["cohort_counts"][
                "revalidation_after_late_recovery_count"
            ],
            3,
        )
        self.assertEqual(
            payload["cohort_counts"][
                "contradicted_synthetic_reason_before_count"
            ],
            3,
        )
        serialized = canonical_json(payload)
        for forbidden in ("MIB-", ".pdf", "Probe", "testing"):
            self.assertNotIn(forbidden, serialized)

    def test_serial_and_parallel_runs_are_identical_except_repeat_index(self):
        serial = self._run(max_workers=1, repeat_index=1)
        parallel = self._run(max_workers=4, repeat_index=2)

        self.assertEqual(
            {key: value for key, value in serial.items() if key != "repeat_index"},
            {
                key: value
                for key, value in parallel.items()
                if key != "repeat_index"
            },
        )

    def test_real_contract_probes_are_complete_nonvacuous_and_safe(self):
        counts = run_contract_probes()

        self.assertEqual(set(counts), set(CONTRACT_AUDIT_COUNTS))
        for name in (
            "legacy_synthetic_before_late_recovery_count",
            "candidate_late_recovery_before_revalidation_count",
            "candidate_revalidation_after_late_recovery_count",
            "contradicted_synthetic_reason_removed_count",
            "independent_denial_reason_retained_count",
            "review_confidence_restored_count",
            "signed_late_authority_recovery_count",
            "late_adjudication_evidence_preserved_count",
            "late_biohazard_evidence_preserved_count",
            "placeholder_guard_probe_count",
            "sentinel_guard_probe_count",
            "serialization_default_guard_probe_count",
            "stale_threshold_guard_probe_count",
            "forced_approval_guard_probe_count",
            "direct_approval_head_guard_probe_count",
        ):
            self.assertGreater(counts[name], 0)
        for name in (
            "forced_approval_count",
            "sentinel_value_used_as_evidence_count",
            "placeholder_value_used_as_evidence_count",
            "serialization_default_used_as_evidence_count",
            "stale_threshold_mismatch_count",
            "contradicted_synthetic_reason_remaining_count",
        ):
            self.assertEqual(counts[name], 0)

    def test_missing_or_malformed_accepted_final_observer_fails_closed(self):
        with self.assertRaisesRegex(
            PolicyRevalidationAuditRunError, "does not match"
        ):
            self._run(
                processor_factory=lambda: _Production(
                    _Inner(omit_observer=True)
                )
            )

        broken = dict(BASE_CORE_COUNTS)
        del broken["signed_late_authority_recovery_count"]
        with self.assertRaisesRegex(
            PolicyRevalidationAuditRunError, "complete frozen"
        ):
            self._run(
                processor_factory=lambda: _Production(_Inner(broken))
            )

    def test_prediction_byte_mismatch_fails_closed(self):
        with self.assertRaisesRegex(
            PolicyRevalidationAuditRunError, "do not match"
        ):
            self._run(
                processor_factory=lambda: _Production(
                    _Inner(changed_output=True)
                )
            )

    def test_audit_requires_a_clean_checkout_at_the_declared_revision(self):
        calls: list[tuple[Path, str]] = []

        def verifier(root: Path, revision: str) -> None:
            calls.append((root, revision))

        self._run(checkout_verifier=verifier)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], SOURCE_SHA)

        def reject(_root: Path, _revision: str) -> None:
            raise PolicyRevalidationAuditRunError("dirty checkout")

        with self.assertRaisesRegex(
            PolicyRevalidationAuditRunError, "dirty checkout"
        ):
            self._run(checkout_verifier=reject)

    def test_incomplete_contract_probe_or_bad_fixture_digest_fails_closed(self):
        incomplete = _contract_counts()
        del incomplete["signed_late_authority_recovery_count"]
        with self.assertRaisesRegex(
            PolicyRevalidationAuditRunError, "complete counter"
        ):
            self._run(contract_probe=lambda: incomplete)
        with self.assertRaisesRegex(
            PolicyRevalidationAuditRunError, "full SHA-256"
        ):
            self._run(fixture_digest_provider=lambda: "short")

    def test_cli_is_atomic_and_canonical(self):
        output = self.root / "audit.json"
        output.write_text("old\n", encoding="utf-8")
        with (
            patch(
                "devtools.policy_revalidation_audit_run."
                "build_production_processor",
                lambda: _Production(),
            ),
            patch(
                "devtools.policy_revalidation_audit_run."
                "run_contract_probes",
                _contract_counts,
            ),
            patch(
                "devtools.policy_revalidation_audit_run."
                "contract_fixture_sha256",
                lambda: FIXTURE_SHA,
            ),
            patch(
                "devtools.policy_revalidation_audit_run."
                "verify_clean_candidate_checkout",
                lambda _root, _revision: None,
            ),
        ):
            exit_code = main(
                [
                    "--input-dir",
                    str(self.input_dir),
                    "--predictions",
                    str(self.predictions),
                    "--output",
                    str(output),
                    "--source-revision",
                    SOURCE_SHA,
                    "--repeat-index",
                    "2",
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
        self.assertEqual(parsed["repeat_index"], 2)


if __name__ == "__main__":
    unittest.main()
