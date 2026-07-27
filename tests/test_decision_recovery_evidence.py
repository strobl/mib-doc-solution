from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import devtools.wo18_production_capture as production_capture
from devtools.decision_recovery_evidence import (
    APPROACHES,
    CONTRACT_AUDIT_SCHEMA,
    PROTECTED_ROLE_MANIFEST_SCHEMA,
    PROTECTED_ROLE_TAXONOMY_SHA256,
    DecisionRecoveryEvidenceBuildError,
    _feature_schema_hash,
    _runtime_feature_names,
    build_aggregate_evidence,
    render_aggregate_markdown,
)
from devtools.decision_recovery_cv import (
    FEATURE_ROWS_SCHEMA,
    execute_grouped_oof,
    load_frozen_feature_rows,
    write_comparison_artifacts,
)
from devtools.decision_recovery_contract_probe import (
    contract_fixture_payload,
    run_contract_probes,
)
from devtools.decision_recovery_gate import (
    DecisionRecoveryContractAudit,
    PROTECTED_ROLES,
)
from devtools.experiment_control import canonical_json, require_aggregate_only
from devtools.grouped_recovery_evidence import (
    LAYOUT_MANIFEST_SCHEMA,
    load_layout_manifest,
)
from devtools.ocr_ablation import _input_tree_sha256
from devtools.wo18_production_capture import (
    CAPTURE_OBSERVATION_SCHEMA,
    producer_graph_sha256,
)
from scripts import evaluate as official_evaluate


SOURCE_SHA = "a" * 40
FIELDS = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
    "unrecoverable_fields",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


class DecisionRecoveryEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_dir = self.root / "pdfs"
        self.input_dir.mkdir()
        self.case_ids = tuple(
            f"MIB-{index:06d}" for index in range(1, 33)
        )
        for index, case_id in enumerate(self.case_ids, start=1):
            (self.input_dir / f"{case_id}.pdf").write_bytes(
                f"fixture-{index}".encode("ascii")
            )
        self.input_tree_sha, count = _input_tree_sha256(self.input_dir)
        self.assertEqual(count, 32)

        self.layout_path = _write_json(
            self.root / "layout.json",
            {
                "schema": LAYOUT_MANIFEST_SCHEMA,
                "frozen_before_scoring": True,
                "split_seed": "wo18-test-v1",
                "repeats": 3,
                "folds": 5,
                "cases": [
                    {
                        "case_id": case_id,
                        "layout_group": f"layout-{index % 9}",
                    }
                    for index, case_id in enumerate(self.case_ids)
                ],
            },
        )
        self.layout = load_layout_manifest(self.layout_path)
        self.role_by_case = {
            case_id: PROTECTED_ROLES[index % len(PROTECTED_ROLES)]
            for index, case_id in enumerate(self.case_ids)
        }
        self.role_path = _write_json(
            self.root / "roles.json",
            {
                "schema_version": PROTECTED_ROLE_MANIFEST_SCHEMA,
                "frozen_before_scoring": True,
                "layout_manifest_sha256": self.layout.sha256,
                "role_taxonomy_sha256": PROTECTED_ROLE_TAXONOMY_SHA256,
                "cases": [
                    {
                        "case_id": case_id,
                        "protected_roles": [
                            self.role_by_case[case_id]
                        ],
                    }
                    for case_id in self.case_ids
                ],
            },
        )

        self.truth_rows = [
            {
                "case_id": case_id,
                "applicant_name": f"Applicant {index}",
                "species_code": "HUM",
                "home_world": "Earth",
                "visa_class": "DIP-1",
                "sponsor_id": f"SPN-{index:04d}",
                "arrival_date": "2026-07-27",
                "declared_purpose": "Official visit",
                "risk_flags": "none",
                "fee_status": "paid",
                "adjudication": (
                    "DENIED"
                    if self.role_by_case[case_id]
                    in {"visible_disqualifier", "denial_guard"}
                    else (
                        "NEEDS_REVIEW"
                        if self.role_by_case[case_id] == "uncertainty"
                        else "APPROVED"
                    )
                ),
                "confidence": "1.0",
                "unrecoverable_fields": "",
            }
            for index, case_id in enumerate(self.case_ids, start=1)
        ]
        self.truth_path = self.root / "truth.csv"
        with self.truth_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.truth_rows)

        candidate_rows = [
            {
                key: value
                for key, value in row.items()
                if key != "unrecoverable_fields"
            }
            for row in self.truth_rows
        ]
        control_rows = [
            {
                **row,
                "adjudication": "NEEDS_REVIEW",
                "confidence": 0.5,
            }
            for row in candidate_rows
        ]
        self.feature_names = _runtime_feature_names()
        self.feature_schema_sha = _feature_schema_hash(self.feature_names)
        self.evaluator_sha = _sha(Path(official_evaluate.__file__))
        feature_values = {name: 1.0 for name in self.feature_names}
        feature_values.update(
            {
                "baseline_approved": 0.0,
                "baseline_denied": 0.0,
                "baseline_review": 1.0,
                "authoritative_decision": 0.0,
                "resolved_fraction": 1.0,
                "visible_fraction": 1.0,
                "exact_case_scope_fraction": 1.0,
                "exact_subject_scope_fraction": 1.0,
                "clean_fraction": 1.0,
                "provenance_complete_fraction": 1.0,
                "link_confidence": 1.0,
                "unresolved_linkage": 0.0,
                "rescinded_decision": 0.0,
                "packet_conflict": 0.0,
                "packet_watermark": 0.0,
                "binding_approval": 0.0,
                "binding_denial": 0.0,
                "binding_review": 0.0,
                "policy_explicit_violation": 0.0,
                "policy_review_gap": 0.0,
                "policy_review_conflict": 0.0,
                "policy_review_visibility": 0.0,
                "policy_review_waiver": 0.0,
                "policy_review_other": 0.0,
                "policy_strict_clear": 1.0,
                "route_primary": 1.0,
                "route_late_visible": 0.0,
                "route_rapid_visible": 0.0,
            }
        )
        feature_rows = []
        for row in control_rows:
            values = dict(feature_values)
            role = self.role_by_case[str(row["case_id"])]
            if role == "binding_authority":
                values["binding_approval"] = 1.0
            elif role in {"visible_disqualifier", "denial_guard"}:
                values["policy_explicit_violation"] = 1.0
            elif role == "uncertainty":
                values["packet_conflict"] = 1.0
            feature_rows.append(
                {
                    "case_id": row["case_id"],
                    "baseline_prediction": row,
                    "baseline_decision": "NEEDS_REVIEW",
                    "recovery_route": "primary",
                    "features": values,
                }
            )
        self.feature_rows_path = _write_json(
            self.root / "feature-rows.json",
            {
                "schema_version": FEATURE_ROWS_SCHEMA,
                "frozen_before_fit": True,
                "source_revision_sha": SOURCE_SHA,
                "layout_manifest_sha256": self.layout.sha256,
                "input_tree_sha256": self.input_tree_sha,
                "feature_schema_sha256": self.feature_schema_sha,
                "rows": feature_rows,
            },
        )
        self.rerun_feature_rows_path = (
            self.root / "feature-rows-rerun.json"
        )
        self.rerun_feature_rows_path.write_bytes(
            self.feature_rows_path.read_bytes()
        )
        self.capture_observation_path = _write_json(
            self.root / "capture-observation.json",
            {
                "schema_version": CAPTURE_OBSERVATION_SCHEMA,
                "source_revision_sha": SOURCE_SHA,
                "producer_source_sha256": _sha(
                    Path(production_capture.__file__)
                ),
                "producer_graph_sha256": producer_graph_sha256(),
                "layout_manifest_sha256": self.layout.sha256,
                "input_tree_sha256": self.input_tree_sha,
                "feature_schema_sha256": self.feature_schema_sha,
                "record_count": len(self.case_ids),
                "capture_run_count": 2,
                "feature_rows_sha256": _sha(self.feature_rows_path),
                "rerun_feature_rows_sha256": _sha(
                    self.rerun_feature_rows_path
                ),
                "byte_deterministic": True,
                "truth_or_role_input_count": 0,
            },
        )
        frozen_features = load_frozen_feature_rows(
            self.feature_rows_path,
            layout_manifest=self.layout,
            expected_source_revision_sha=SOURCE_SHA,
            expected_input_tree_sha256=self.input_tree_sha,
        )
        truth_mapping = {
            row["case_id"]: row for row in self.truth_rows
        }
        first = execute_grouped_oof(
            layout_manifest=self.layout,
            truth=truth_mapping,
            features=frozen_features,
        )
        second = execute_grouped_oof(
            layout_manifest=self.layout,
            truth=truth_mapping,
            features=frozen_features,
        )
        self.arm_paths = dict(
            write_comparison_artifacts(
                output_dir=self.root / "cv",
                execution=first,
                rerun=second,
                layout_manifest=self.layout,
                feature_rows_sha256=frozen_features.sha256,
                protected_role_manifest_sha256=_sha(self.role_path),
                truth_sha256=_sha(self.truth_path),
                input_tree_sha256=self.input_tree_sha,
                source_revision_sha=SOURCE_SHA,
            )
        )
        self.contract_fixture = _write_json(
            self.root / "contract-fixture.json",
            contract_fixture_payload(),
        )
        self.audit_paths = tuple(
            self._write_audit(repeat) for repeat in (1, 2)
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_audit(self, repeat: int) -> Path:
        counts = run_contract_probes()
        return _write_json(
            self.root / f"audit-{repeat}.json",
            {
                "schema_version": CONTRACT_AUDIT_SCHEMA,
                "repeat_index": repeat,
                "source_revision_sha": SOURCE_SHA,
                "layout_manifest_sha256": self.layout.sha256,
                "protected_role_manifest_sha256": _sha(self.role_path),
                "input_tree_sha256": self.input_tree_sha,
                "feature_schema_sha256": self.feature_schema_sha,
                "feature_names": list(self.feature_names),
                "contract_fixture_sha256": _sha(self.contract_fixture),
                "counts": counts,
                "deterministic": True,
            },
        )

    def _build(self, **changes: object) -> dict[str, object]:
        values: dict[str, object] = {
            "layout_manifest_path": self.layout_path,
            "expected_layout_manifest_sha256": self.layout.sha256,
            "protected_role_manifest_path": self.role_path,
            "expected_protected_role_manifest_sha256": _sha(
                self.role_path
            ),
            "expected_input_tree_sha256": self.input_tree_sha,
            "input_dir": self.input_dir,
            "feature_rows_path": self.feature_rows_path,
            "rerun_feature_rows_path": self.rerun_feature_rows_path,
            "capture_observation_path": self.capture_observation_path,
            "truth_path": self.truth_path,
            "arm_manifest_paths": self.arm_paths,
            "contract_audit_paths": self.audit_paths,
            "contract_fixture_path": self.contract_fixture,
            "source_revision_sha": SOURCE_SHA,
            "checkout_verifier": lambda _root, _sha: None,
        }
        values.update(changes)
        return build_aggregate_evidence(**values)  # type: ignore[arg-type]

    def test_builds_exact_four_arm_aggregate_evidence(self):
        aggregate = self._build()

        self.assertIn(aggregate["status"], {"passed", "blocked"})
        self.assertEqual(aggregate["approach_count"], 4)
        self.assertEqual(aggregate["fold_count"], 15)
        require_aggregate_only(aggregate)
        rendered = canonical_json(aggregate)
        for forbidden in ("MIB-", ".pdf", "Applicant", "Official visit"):
            self.assertNotIn(forbidden, rendered)
        markdown = render_aggregate_markdown(aggregate)
        self.assertIn("Four-arm comparison", markdown)
        self.assertIn("public-exposed", markdown)

    def test_input_tree_is_recomputed_not_trusted(self):
        (self.input_dir / f"{self.case_ids[0]}.pdf").write_bytes(b"changed")
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError, "input directory"
        ):
            self._build()

    def test_production_capture_requires_two_label_free_bound_runs(self):
        self.rerun_feature_rows_path.write_bytes(
            self.rerun_feature_rows_path.read_bytes() + b"\n"
        )
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError,
            "capture reruns",
        ):
            self._build()

        self.rerun_feature_rows_path.write_bytes(
            self.feature_rows_path.read_bytes()
        )
        observation = json.loads(
            self.capture_observation_path.read_text(encoding="utf-8")
        )
        observation["truth_or_role_input_count"] = 1
        changed = _write_json(
            self.root / "label-aware-capture.json",
            observation,
        )
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError,
            "label-aware",
        ):
            self._build(capture_observation_path=changed)

    def test_production_capture_binds_the_full_graph(self):
        observation = json.loads(
            self.capture_observation_path.read_text(encoding="utf-8")
        )
        observation["producer_graph_sha256"] = "0" * 64
        changed = _write_json(
            self.root / "wrong-graph-capture.json",
            observation,
        )
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError,
            "producer graph binding",
        ):
            self._build(capture_observation_path=changed)

    def test_arm_requires_byte_identical_rerun(self):
        manifest = json.loads(
            self.arm_paths["gated_hybrid"].read_text(encoding="utf-8")
        )
        rerun = Path(manifest["runs"][0]["rerun_predictions_path"])
        rerun.write_text(rerun.read_text(encoding="utf-8") + "\n")
        manifest["runs"][0]["rerun_predictions_sha256"] = _sha(rerun)
        changed = _write_json(self.root / "changed-arm.json", manifest)
        arms = dict(self.arm_paths)
        arms["gated_hybrid"] = changed
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError, "byte-deterministic"
        ):
            self._build(arm_manifest_paths=arms)

    def test_oof_fit_attestation_is_derived_from_frozen_split(self):
        manifest = json.loads(
            self.arm_paths["compact_identity_free_model"].read_text(
                encoding="utf-8"
            )
        )
        manifest["runs"][0]["folds"][0][
            "training_groups_sha256"
        ] = "0" * 64
        changed = _write_json(self.root / "bad-fit-arm.json", manifest)
        arms = dict(self.arm_paths)
        arms["compact_identity_free_model"] = changed
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError, "group-exclusive fitting"
        ):
            self._build(arm_manifest_paths=arms)

    def test_roles_must_be_disjoint_supported_and_fixed(self):
        roles = json.loads(self.role_path.read_text(encoding="utf-8"))
        roles["cases"][0]["protected_roles"].append("uncertainty")
        changed = _write_json(self.root / "bad-roles.json", roles)
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError, "unique, recognized"
        ):
            self._build(
                protected_role_manifest_path=changed,
                expected_protected_role_manifest_sha256=_sha(changed),
            )

    def test_contract_fixture_bytes_and_two_reruns_are_bound(self):
        self.contract_fixture.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(
            DecisionRecoveryEvidenceBuildError,
            "fixture bytes|executable WO-18 probes",
        ):
            self._build()


if __name__ == "__main__":
    unittest.main()
