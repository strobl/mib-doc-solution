from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from devtools.final_confidence_capture import (
    CAPTURE_SCHEMA,
    CaptureBindings,
    CapturedFinalPrediction,
    FinalConfidenceCaptureError,
    _capture_with_outer,
    _require_external_outputs,
    _validate_context,
    _verify_frozen_inputs_unchanged,
    build_capture_payload,
    context_contract_mapping,
    non_confidence_jsonl,
    prediction_jsonl,
)
from devtools.grouped_recovery_evidence import FrozenLayoutManifest
from devtools.ocr_ablation import _input_tree_sha256
from mib_pipeline.final_confidence import (
    FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION,
    FinalConfidenceContext,
    FinalPredictionWithConfidenceContext,
)
from mib_pipeline.models import FIELD_NAMES, PredictionRow
from mib_pipeline.output_confidence import (
    OutputConfidenceRecalibrationProcessor,
    OutputConfidenceRecalibrator,
)
from mib_pipeline.writer import CanonicalJsonlWriter


def _row(case_id: str, *, confidence: float = 0.4) -> PredictionRow:
    return PredictionRow.from_mapping(
        {
            "case_id": case_id,
            "applicant_name": "Arix Vale",
            "species_code": "ARCTURIAN",
            "home_world": "Mars",
            "visa_class": "XW-1",
            "sponsor_id": "SPN-0001",
            "arrival_date": "2026-01-01",
            "declared_purpose": "research",
            "risk_flags": "none",
            "fee_status": "paid",
            "adjudication": "NEEDS_REVIEW",
            "confidence": confidence,
        }
    )


def _context() -> FinalConfidenceContext:
    return FinalConfidenceContext(
        schema_version=FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION,
        final_class="NEEDS_REVIEW",
        policy_route="deterministic_policy",
        authoritative=False,
        visible_completeness=1.0,
        has_conflict=False,
        ocr_disagreement=0.0,
        recovery_route="primary",
        model_margin=None,
        ensemble_agreement=None,
        resolution_entropy=0.0,
    )


def _captured(pdf_path: Path) -> CapturedFinalPrediction:
    accepted = FinalPredictionWithConfidenceContext(
        row=_row(pdf_path.stem),
        context=_context(),
    )
    final = OutputConfidenceRecalibrator.from_pinned_artifact().recalibrate(
        accepted.row,
        context=accepted.context,
    )
    return CapturedFinalPrediction(accepted=accepted, final_row=final)


def _bindings(
    *,
    layout_sha: str,
    input_tree_sha: str,
) -> CaptureBindings:
    return CaptureBindings(
        source_revision_sha="a" * 40,
        layout_manifest_sha256=layout_sha,
        input_tree_sha256=input_tree_sha,
        producer_graph_sha256="b" * 64,
        producer_source_sha256="c" * 64,
        runtime_confidence_artifact_sha256="d" * 64,
        runtime_confidence_artifact_file_sha256="e" * 64,
        runtime_confidence_artifact_id="test-confidence-artifact",
        context_contract_sha256="f" * 64,
        context_contract_source_sha256="1" * 64,
    )


class _ContextualInner:
    def __init__(self, accepted: FinalPredictionWithConfidenceContext):
        self.accepted = accepted
        self.paths = []

    def process_case_with_confidence_context(self, pdf_path: Path):
        self.paths.append(pdf_path)
        return self.accepted


class FinalConfidenceProductionCaptureTests(unittest.TestCase):
    def test_capture_contract_has_no_truth_or_label_input(self):
        parameters = inspect.signature(build_capture_payload).parameters
        self.assertNotIn("truth", parameters)
        self.assertNotIn("labels", parameters)
        self.assertNotIn("correctness", parameters)

    def test_outer_stage_consumes_inner_context_and_only_changes_confidence(self):
        accepted = FinalPredictionWithConfidenceContext(
            row=_row("MIB-000001"),
            context=_context(),
        )
        inner = _ContextualInner(accepted)
        outer = OutputConfidenceRecalibrationProcessor(
            processor=inner,
            recalibrator=OutputConfidenceRecalibrator.from_pinned_artifact(),
        )

        captured = _capture_with_outer(Path("MIB-000001.pdf"), outer)

        self.assertEqual(inner.paths, [Path("MIB-000001.pdf")])
        self.assertIs(captured.accepted, accepted)
        self.assertNotEqual(
            captured.final_row.confidence,
            accepted.row.confidence,
        )
        self.assertEqual(
            {
                field: captured.final_row.to_dict()[field]
                for field in FIELD_NAMES
                if field != "confidence"
            },
            {
                field: accepted.row.to_dict()[field]
                for field in FIELD_NAMES
                if field != "confidence"
            },
        )

    def test_prediction_jsonl_exactly_matches_production_writer_bytes(self):
        row = dataclasses.replace(
            _row("MIB-000001"),
            applicant_name="Ärix Vale",
        )
        payload = {
            "rows": [
                {
                    "case_id": row.case_id,
                    "layout_group": "layout-a",
                    "prediction": row.to_dict(),
                    "context": _context().to_dict(),
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "production.jsonl"
            CanonicalJsonlWriter().write(output, [row])
            production_bytes = output.read_bytes()

        self.assertEqual(prediction_jsonl(payload), production_bytes)
        expected_projection = {
            field_name: row.to_dict()[field_name]
            for field_name in FIELD_NAMES
            if field_name != "confidence"
        }
        self.assertEqual(
            non_confidence_jsonl(payload),
            (
                json.dumps(
                    expected_projection,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )

    def test_frozen_capture_serializes_context_and_full_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_ids = tuple(f"MIB-{index:06d}" for index in range(1, 6))
            for case_id in case_ids:
                (root / f"{case_id}.PDF").write_bytes(case_id.encode("ascii"))
            input_sha, count = _input_tree_sha256(root)
            self.assertEqual(count, len(case_ids))
            manifest = FrozenLayoutManifest(
                groups={
                    f"layout-{index}": (case_id,)
                    for index, case_id in enumerate(case_ids)
                },
                split_seed="wo19-test",
                sha256="9" * 64,
                frozen_before_scoring=True,
            )
            payload = build_capture_payload(
                input_dir=root,
                layout_manifest=manifest,
                bindings=_bindings(
                    layout_sha=manifest.sha256,
                    input_tree_sha=input_sha,
                ),
                capture_case=_captured,
            )

        self.assertEqual(payload["schema_version"], CAPTURE_SCHEMA)
        self.assertTrue(payload["frozen_before_truth_join"])
        self.assertEqual(
            payload["confidence_topology"],
            {
                "captured_probability": "post_current_outer_recalibrator",
                "candidate_position": "downstream_confidence_only",
                "parallel_batch_runtime_equivalence": (
                    "not_claimed_without_shadow_solution_proof"
                ),
            },
        )
        self.assertEqual(payload["record_count"], len(case_ids))
        self.assertEqual(
            [row["case_id"] for row in payload["rows"]],
            list(case_ids),
        )
        self.assertFalse(
            set(payload).intersection(
                {"truth", "labels", "correct", "correctness"}
            )
        )
        for row in payload["rows"]:
            self.assertEqual(
                tuple(row["prediction"]),
                FIELD_NAMES,
            )
            self.assertEqual(
                row["context"]["final_class"],
                row["prediction"]["adjudication"],
            )
            self.assertTrue(
                0.0 <= row["prediction"]["confidence"] <= 1.0
            )

        full = prediction_jsonl(payload)
        projected = non_confidence_jsonl(payload)
        self.assertEqual(len(full.splitlines()), len(case_ids))
        self.assertEqual(len(projected.splitlines()), len(case_ids))
        for line in projected.splitlines():
            self.assertNotIn("confidence", json.loads(line))

    def test_prediction_serialization_rejects_extra_or_invalid_schema(self):
        row = _row("MIB-000001")
        capture_row = {
            "case_id": row.case_id,
            "layout_group": "layout-a",
            "prediction": row.to_dict(),
            "context": _context().to_dict(),
        }
        extra = dict(capture_row)
        extra_prediction = dict(row.to_dict())
        extra_prediction["unexpected"] = "value"
        extra["prediction"] = extra_prediction
        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "fields are not exact",
        ):
            prediction_jsonl({"rows": [extra]})

        missing = dict(capture_row)
        missing_prediction = dict(row.to_dict())
        del missing_prediction["fee_status"]
        missing["prediction"] = missing_prediction
        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "fields are not exact",
        ):
            prediction_jsonl({"rows": [missing]})

    def test_invalid_final_context_and_non_confidence_mutation_fail_closed(self):
        accepted = FinalPredictionWithConfidenceContext(
            row=_row("MIB-000001"),
            context=_context(),
        )
        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "outside",
        ):
            CapturedFinalPrediction(
                accepted=accepted,
                final_row=dataclasses.replace(
                    accepted.row,
                    confidence=1.5,
                ),
            )
        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "non-confidence",
        ):
            CapturedFinalPrediction(
                accepted=accepted,
                final_row=dataclasses.replace(
                    accepted.row,
                    applicant_name="Changed Name",
                ),
            )
        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "accepted decision",
        ):
            CapturedFinalPrediction(
                accepted=accepted,
                final_row=dataclasses.replace(
                    accepted.row,
                    adjudication="APPROVED",
                ),
            )
        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "typed confidence context",
        ):
            _validate_context(object())  # type: ignore[arg-type]

    def test_duplicate_or_mismatched_case_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case_id in ("MIB-000001", "MIB-000002"):
                (root / f"{case_id}.pdf").write_bytes(case_id.encode("ascii"))
            input_sha, _ = _input_tree_sha256(root)
            manifest = FrozenLayoutManifest(
                groups={
                    "layout-a": ("MIB-000001",),
                    "layout-b": ("MIB-000002",),
                },
                split_seed="wo19-test",
                sha256="9" * 64,
                frozen_before_scoring=True,
            )

            with self.assertRaisesRegex(
                FinalConfidenceCaptureError,
                "does not match",
            ):
                build_capture_payload(
                    input_dir=root,
                    layout_manifest=manifest,
                    bindings=_bindings(
                        layout_sha=manifest.sha256,
                        input_tree_sha=input_sha,
                    ),
                    capture_case=lambda _path: _captured(
                        Path("MIB-000001.pdf")
                    ),
                )

    def test_binding_mismatch_and_post_pass_file_mutation_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "MIB-000001.pdf"
            pdf_path.write_bytes(b"frozen")
            layout_path = root / "layout.json"
            layout_path.write_bytes(b'{"frozen":true}\n')
            input_sha, _ = _input_tree_sha256(root)
            layout_sha = hashlib.sha256(layout_path.read_bytes()).hexdigest()
            manifest = FrozenLayoutManifest(
                groups={"layout-a": ("MIB-000001",)},
                split_seed="wo19-test",
                sha256=layout_sha,
                frozen_before_scoring=True,
            )
            with self.assertRaisesRegex(
                FinalConfidenceCaptureError,
                "layout manifest",
            ):
                build_capture_payload(
                    input_dir=root,
                    layout_manifest=manifest,
                    bindings=_bindings(
                        layout_sha="0" * 64,
                        input_tree_sha=input_sha,
                    ),
                    capture_case=_captured,
                )

            pdf_path.write_bytes(b"changed")
            with self.assertRaisesRegex(
                FinalConfidenceCaptureError,
                "input tree changed",
            ):
                _verify_frozen_inputs_unchanged(
                    input_dir=root,
                    layout_manifest_path=layout_path,
                    layout_manifest=manifest,
                    expected_input_tree_sha256=input_sha,
                )

            changed_sha, _ = _input_tree_sha256(root)
            layout_path.write_bytes(b'{"frozen":false}\n')
            with self.assertRaisesRegex(
                FinalConfidenceCaptureError,
                "layout manifest changed",
            ):
                _verify_frozen_inputs_unchanged(
                    input_dir=root,
                    layout_manifest_path=layout_path,
                    layout_manifest=manifest,
                    expected_input_tree_sha256=changed_sha,
                )

    def test_outputs_inside_repository_are_rejected(self):
        from devtools.final_confidence_capture import REPO_ROOT

        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "outside the repository",
        ):
            _require_external_outputs(
                (
                    REPO_ROOT / "capture.json",
                    Path("/private/tmp/rerun-capture.json"),
                )
            )

    def test_duplicate_output_paths_are_rejected(self):
        duplicate = Path("/private/tmp/wo19-capture.json")
        with self.assertRaisesRegex(
            FinalConfidenceCaptureError,
            "distinct",
        ):
            _require_external_outputs((duplicate, duplicate))

    def test_context_contract_is_closed_and_identity_free(self):
        contract = context_contract_mapping()

        self.assertEqual(
            contract["schema_version"],
            FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION,
        )
        self.assertFalse(contract["free_form_values_allowed"])
        rendered = json.dumps(contract, sort_keys=True)
        self.assertNotIn("case_id", rendered)
        self.assertNotIn("filename", rendered)
        self.assertNotIn("truth", rendered)


if __name__ == "__main__":
    unittest.main()
