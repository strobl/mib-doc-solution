from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from devtools.grouped_recovery_evidence import FrozenLayoutManifest
from devtools.ocr_ablation import _input_tree_sha256
from devtools.wo18_production_capture import (
    CapturedProductionCase,
    ProductionCaptureError,
    _recovery_route,
    build_feature_payload,
)
from mib_pipeline.adjudication import AdjudicationOutcome, DecisionTrace
from mib_pipeline.model_recovery import (
    FEATURE_NAMES,
    IdentityFreeDecisionFeatures,
)
from mib_pipeline.models import PredictionRow
from mib_pipeline.recovery_audit import SerializationOrigin
from mib_pipeline.resolution import ResolvedCase


def _row(case_id: str) -> PredictionRow:
    return PredictionRow.from_mapping(
        {
            "case_id": case_id,
            "applicant_name": "unknown",
            "species_code": "unknown",
            "home_world": "unknown",
            "visa_class": "unknown",
            "sponsor_id": "SPN-0000",
            "arrival_date": "1900-01-01",
            "declared_purpose": "unknown",
            "risk_flags": "none",
            "fee_status": "unknown",
            "adjudication": "NEEDS_REVIEW",
            "confidence": 0.25,
        }
    )


def _capture(pdf_path: Path) -> CapturedProductionCase:
    case_id = pdf_path.stem
    row = _row(case_id)
    resolved = ResolvedCase(
        case_id=case_id,
        active_applicant=None,
        fields={},
        unresolved_linkage=True,
        unresolved_reasons=("no_visible_subject",),
    )
    outcome = AdjudicationOutcome(
        row=row,
        trace=DecisionTrace(
            decision="NEEDS_REVIEW",
            authoritative_source=False,
            denial_reasons=(),
            review_reasons=("required_output_unknown:applicant_name",),
            approval_facts=(),
            exception_ids=(),
        ),
    )
    features = IdentityFreeDecisionFeatures.from_mapping(
        {
            name: float(
                name
                in {
                    "baseline_review",
                    "unresolved_linkage",
                    "policy_review_gap",
                    "route_primary",
                }
            )
            for name in FEATURE_NAMES
        }
    )
    return CapturedProductionCase(
        row=row,
        resolved_case=resolved,
        outcome=outcome,
        recovery_route="primary",
        features=features,
    )


class ProductionFeatureCaptureTests(unittest.TestCase):
    def test_capture_contract_has_no_truth_or_role_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_ids = tuple(f"MIB-{index:06d}" for index in range(1, 6))
            for case_id in case_ids:
                (root / f"{case_id}.pdf").write_bytes(
                    f"fixture:{case_id}".encode("ascii")
                )
            tree_sha, count = _input_tree_sha256(root)
            self.assertEqual(count, len(case_ids))
            manifest = FrozenLayoutManifest(
                groups={
                    f"layout-{index}": (case_id,)
                    for index, case_id in enumerate(case_ids)
                },
                split_seed="wo18-test",
                sha256="b" * 64,
                frozen_before_scoring=True,
            )

            payload = build_feature_payload(
                input_dir=root,
                layout_manifest=manifest,
                source_revision_sha="a" * 40,
                expected_input_tree_sha256=tree_sha,
                capture_case=_capture,
            )

        self.assertEqual(payload["schema_version"], "mib-wo18-frozen-feature-rows/v1")
        self.assertEqual(len(payload["rows"]), 5)
        self.assertEqual(
            [row["case_id"] for row in payload["rows"]],
            sorted(case_ids),
        )
        rendered_keys = set(payload)
        self.assertFalse(
            rendered_keys.intersection(
                {"truth", "labels", "protected_roles", "role_assignments"}
            )
        )
        for row in payload["rows"]:
            self.assertEqual(set(row["features"]), set(FEATURE_NAMES))
            self.assertEqual(
                row["baseline_prediction"]["case_id"],
                row["case_id"],
            )

    def test_input_tree_binding_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "MIB-000001.pdf").write_bytes(b"fixture")
            manifest = FrozenLayoutManifest(
                groups={"layout": ("MIB-000001",)},
                split_seed="wo18-test",
                sha256="b" * 64,
                frozen_before_scoring=True,
            )
            with self.assertRaisesRegex(
                ProductionCaptureError,
                "input tree",
            ):
                build_feature_payload(
                    input_dir=root,
                    layout_manifest=manifest,
                    source_revision_sha="a" * 40,
                    expected_input_tree_sha256="0" * 64,
                    capture_case=_capture,
                )

    def test_route_comes_from_the_accepted_recovery_source(self):
        def audit(source):
            return SimpleNamespace(
                fields={
                    "field": SimpleNamespace(
                        serialization_after_origin=(
                            SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE
                        ),
                        recovery_source=source,
                    )
                }
            )

        self.assertEqual(
            _recovery_route(audit("primary_visible_ocr"), {}),
            "primary",
        )
        self.assertEqual(
            _recovery_route(audit("targeted_rapidocr"), {}),
            "rapid_visible",
        )
        self.assertEqual(
            _recovery_route(
                SimpleNamespace(fields={}),
                {"late_recovery_before_revalidation_count": 1},
            ),
            "late_visible",
        )
        with self.assertRaisesRegex(
            ProductionCaptureError,
            "no auditable route",
        ):
            _recovery_route(audit(None), {})


if __name__ == "__main__":
    unittest.main()
