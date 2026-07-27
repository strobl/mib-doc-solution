import csv
import json
import tempfile
import unittest
from pathlib import Path

from devtools.ocr_ablation import (
    BASELINE_CONFIG,
    CHECKBOX_ACTIVITY_COUNTERS,
    AblationConfigurationError,
    AblationVariant,
    build_ablation_processor,
    build_report,
    config_sha256,
    registered_variants,
    render_markdown,
    run_variant,
)
from devtools.render_candidates import (
    BoundedContrastRenderer,
    BoundedTemplateRegistrationRenderer,
)
from mib_pipeline import (
    DocumentRenderer,
    PredictionRow,
    VisualCueDetector,
    build_production_processor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
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
)


def truth(case_id, adjudication):
    return {
        "case_id": case_id,
        "applicant_name": "Zed Zarnax",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-1042",
        "arrival_date": "2026-04-17",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": adjudication,
    }


def prediction(row, *, adjudication=None, confidence=0.8, risk_flags=None):
    return PredictionRow.from_mapping(
        {
            **row,
            "risk_flags": risk_flags or row["risk_flags"],
            "adjudication": adjudication or row["adjudication"],
            "confidence": confidence,
        }
    )


class FixedProcessor:
    def __init__(self, rows):
        self._rows = rows

    def process_case(self, path):
        return self._rows[path.stem]


class ActivityFixedProcessor(FixedProcessor):
    def ablation_activity(self):
        return {
            "pages_scanned": 2,
            "complete_groups": 0,
            "checked_groups": 0,
            "candidates_added": 0,
            "ambiguous_groups": 0,
        }


class AblationPlanTests(unittest.TestCase):
    def test_registry_is_stable_and_every_variant_changes_exactly_one_setting(self):
        variants = registered_variants()

        self.assertEqual(len(variants), 13)
        self.assertEqual(
            len({variant.variant_id for variant in variants}),
            len(variants),
        )
        baseline = {
            f"{section}.{name}": value
            for section, values in BASELINE_CONFIG.items()
            for name, value in values.items()
        }
        for variant in variants:
            candidate = {
                f"{section}.{name}": value
                for section, values in variant.config.items()
                for name, value in values.items()
            }
            differences = [
                key for key in baseline if baseline[key] != candidate[key]
            ]
            self.assertEqual(differences, [variant.changed_variable])
            expected_enabled_side = (
                "variant"
                if variant.variant_id.startswith("with_")
                else "baseline"
            )
            self.assertEqual(
                variant.technique_enabled_in,
                expected_enabled_side,
            )

    def test_scope_closure_variants_are_registered_and_buildable(self):
        variant_ids = {variant.variant_id for variant in registered_variants()}

        self.assertIn("without_renderer_deskew", variant_ids)
        self.assertIn("without_visible_cue_interpretation", variant_ids)
        self.assertIn("with_checked_fee_option_recovery", variant_ids)
        self.assertIn("with_bounded_template_registration", variant_ids)
        self.assertIn("with_bounded_contrast", variant_ids)

        deskew_processor = build_ablation_processor("without_renderer_deskew")
        deskew_renderer = deskew_processor.processor._renderer
        self.assertEqual(
            deskew_renderer._estimate_skew(None, None, None),
            0.0,
        )

        cue_processor = build_ablation_processor(
            "without_visible_cue_interpretation"
        )
        cue_detector = cue_processor.processor._primary_extractor._cues
        self.assertEqual(cue_detector.cues_for_line(None, None), ())

        control = build_ablation_processor("without_orientation_retry")
        self.assertIs(type(control.processor._renderer), DocumentRenderer)
        self.assertIsInstance(
            control.processor._primary_extractor._cues,
            VisualCueDetector,
        )

        checkbox = build_ablation_processor(
            "with_checked_fee_option_recovery"
        )
        checkbox_extractor = checkbox.processor._primary_extractor
        self.assertEqual(
            checkbox_extractor.ablation_activity(),
            {name: 0 for name in CHECKBOX_ACTIVITY_COUNTERS},
        )
        self.assertIs(
            checkbox_extractor._ocr,
            checkbox_extractor._delegate._ocr,
        )

        registration = build_ablation_processor(
            "with_bounded_template_registration"
        )
        self.assertIsInstance(
            registration.processor._renderer,
            BoundedTemplateRegistrationRenderer,
        )
        self.assertTrue(
            all(
                value == 0
                for value in registration.processor._renderer
                .ablation_activity()
                .values()
            )
        )

        contrast = build_ablation_processor("with_bounded_contrast")
        self.assertIsInstance(
            contrast.processor._renderer,
            BoundedContrastRenderer,
        )
        self.assertTrue(
            all(
                value == 0
                for value in contrast.processor._renderer
                .ablation_activity()
                .values()
            )
        )

    def test_two_variable_or_unregistered_change_is_rejected(self):
        changed = {
            section: dict(values) for section, values in BASELINE_CONFIG.items()
        }
        changed["primary"]["orientation_retry"] = False
        changed["primary"]["risk_geometry_retry"] = False
        with self.assertRaises(AblationConfigurationError):
            AblationVariant(
                variant_id="invalid",
                family="invalid",
                technique="invalid",
                changed_variable="primary.orientation_retry",
                config=changed,
                target_fields=("risk_flags",),
            )

    def test_baseline_factory_is_the_production_composition_root(self):
        baseline = build_production_processor()

        self.assertEqual(
            type(baseline),
            type(build_production_processor()),
        )


class LabelBlindRunTests(unittest.TestCase):
    def test_run_records_only_aggregate_runtime_evidence(self):
        rows = {
            "MIB-000001": prediction(truth("MIB-000001", "APPROVED")),
            "MIB-000002": prediction(truth("MIB-000002", "DENIED")),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            for case_id in rows:
                (input_dir / f"{case_id}.pdf").write_bytes(b"%PDF-fixture")
            observation = run_variant(
                variant_id="baseline",
                benchmark_id="fixture-v1",
                source_revision="a" * 40,
                repeat_index=1,
                input_dir=input_dir,
                predictions_path=root / "predictions.jsonl",
                observation_path=root / "observation.json",
                max_workers=1,
                processor_factory=lambda _variant: FixedProcessor(rows),
            )
            serialized = json.dumps(observation, sort_keys=True)

        self.assertEqual(observation["answered"], 2)
        self.assertEqual(observation["omitted"], 0)
        self.assertGreater(observation["cpu_seconds"], 0.0)
        self.assertNotIn("MIB-000001", serialized)
        self.assertNotIn("truth", serialized.casefold())

    def test_run_records_only_aggregate_route_activity(self):
        rows = {
            "MIB-000001": prediction(truth("MIB-000001", "APPROVED")),
            "MIB-000002": prediction(truth("MIB-000002", "DENIED")),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            for case_id in rows:
                (input_dir / f"{case_id}.pdf").write_bytes(b"%PDF-fixture")
            observation = run_variant(
                variant_id="with_checked_fee_option_recovery",
                benchmark_id="fixture-v1",
                source_revision="a" * 40,
                repeat_index=1,
                input_dir=input_dir,
                predictions_path=root / "predictions.jsonl",
                observation_path=root / "observation.json",
                max_workers=1,
                processor_factory=lambda _variant: ActivityFixedProcessor(rows),
            )

        self.assertEqual(
            observation["activity_counts"],
            {
                "pages_scanned": 2,
                "complete_groups": 0,
                "checked_groups": 0,
                "candidates_added": 0,
                "ambiguous_groups": 0,
            },
        )


class AblationReportTests(unittest.TestCase):
    def _fixture(self):
        stack = tempfile.TemporaryDirectory()
        root = Path(stack.name)
        truth_path = root / "truth.csv"
        truths = [
            truth("MIB-000001", "APPROVED"),
            truth("MIB-000002", "DENIED"),
        ]
        with truth_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(truths)
        return stack, root, truth_path, truths

    @staticmethod
    def _write_predictions(path, rows):
        path.write_text(
            "".join(
                json.dumps(row.to_dict(), separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _observation(
        path,
        *,
        variant_id,
        repeat,
        predictions_path,
        cpu,
        config,
        activity=None,
    ):
        import hashlib

        prediction_hash = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        value = {
            "schema_version": "mib_ocr_ablation_v1",
            "benchmark_id": "fixture-v1",
            "variant_id": variant_id,
            "repeat_index": repeat,
            "source_revision": "b" * 40,
            "config_sha256": config_sha256(config),
            "input_tree_sha256": "c" * 64,
            "input_pdf_count": 2,
            "max_workers": 1,
            "predictions_path": str(predictions_path),
            "predictions_sha256": prediction_hash,
            "attempted": 2,
            "answered": 2,
            "omitted": 0,
            "cpu_seconds": cpu,
            "wall_seconds": cpu / 2,
            "peak_memory_mib": 64.0,
            "metrics_source": (
                "fresh_process_rusage_self_plus_waited_children_and_monotonic_wall"
            ),
            "activity_counts": activity or {},
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_report_ranks_positive_deterministic_safe_removal_contribution(self):
        stack, root, truth_path, truths = self._fixture()
        self.addCleanup(stack.cleanup)
        baseline_rows = [
            prediction(truths[0]),
            prediction(truths[1]),
        ]
        ablated_rows = [
            prediction(truths[0], risk_flags="active_warrant"),
            prediction(truths[1]),
        ]
        baseline_path = root / "baseline.jsonl"
        variant_path = root / "variant.jsonl"
        self._write_predictions(baseline_path, baseline_rows)
        self._write_predictions(variant_path, ablated_rows)
        variant = next(
            item
            for item in registered_variants()
            if item.variant_id == "without_risk_geometry_retry"
        )
        paths = []
        for repeat in (1, 2):
            paths.append(
                self._observation(
                    root / f"baseline-{repeat}.json",
                    variant_id="baseline",
                    repeat=repeat,
                    predictions_path=baseline_path,
                    cpu=10.0 + repeat,
                    config=BASELINE_CONFIG,
                )
            )
            paths.append(
                self._observation(
                    root / f"variant-{repeat}.json",
                    variant_id=variant.variant_id,
                    repeat=repeat,
                    predictions_path=variant_path,
                    cpu=8.0 + repeat,
                    config=variant.config,
                )
            )

        report = build_report(
            repo_root=REPO_ROOT,
            truth_path=truth_path,
            observation_paths=paths,
        )
        ranked = report["ranked_recommendations"]
        markdown = render_markdown(report)

        self.assertEqual(ranked[0]["variant_id"], variant.variant_id)
        entry = next(
            item
            for item in report["variants"]
            if item["variant_id"] == variant.variant_id
        )
        self.assertGreater(entry["technique_score_gain"], 0.0)
        self.assertGreater(entry["score_gain_per_cpu_second"], 0.0)
        self.assertEqual(
            entry["target_field_raw_point_deltas"]["risk_flags"],
            8.0,
        )
        self.assertTrue(entry["deterministic"])
        self.assertTrue(entry["safety_pass"])
        self.assertIn("not_measured", {item["evidence_status"] for item in report["variants"]})
        self.assertNotIn("MIB-000001", markdown)

    def test_non_deterministic_repetitions_are_not_recommended(self):
        stack, root, truth_path, truths = self._fixture()
        self.addCleanup(stack.cleanup)
        baseline_path = root / "baseline.jsonl"
        first_path = root / "first.jsonl"
        second_path = root / "second.jsonl"
        self._write_predictions(
            baseline_path,
            [prediction(truths[0]), prediction(truths[1])],
        )
        self._write_predictions(
            first_path,
            [
                prediction(truths[0], risk_flags="active_warrant"),
                prediction(truths[1]),
            ],
        )
        self._write_predictions(
            second_path,
            [
                prediction(truths[0], risk_flags="contraband_match"),
                prediction(truths[1]),
            ],
        )
        variant = next(
            item
            for item in registered_variants()
            if item.variant_id == "without_risk_geometry_retry"
        )
        paths = [
            self._observation(
                root / f"baseline-{repeat}.json",
                variant_id="baseline",
                repeat=repeat,
                predictions_path=baseline_path,
                cpu=10.0,
                config=BASELINE_CONFIG,
            )
            for repeat in (1, 2)
        ]
        paths.extend(
            [
                self._observation(
                    root / "variant-1.json",
                    variant_id=variant.variant_id,
                    repeat=1,
                    predictions_path=first_path,
                    cpu=9.0,
                    config=variant.config,
                ),
                self._observation(
                    root / "variant-2.json",
                    variant_id=variant.variant_id,
                    repeat=2,
                    predictions_path=second_path,
                    cpu=9.0,
                    config=variant.config,
                ),
            ]
        )

        report = build_report(
            repo_root=REPO_ROOT,
            truth_path=truth_path,
            observation_paths=paths,
        )
        entry = next(
            item
            for item in report["variants"]
            if item["variant_id"] == variant.variant_id
        )

        self.assertEqual(entry["evidence_status"], "insufficient_evidence")
        self.assertFalse(entry["recommendation_eligible"])

    def test_catastrophic_false_approval_in_enabled_technique_blocks_recommendation(self):
        stack, root, truth_path, truths = self._fixture()
        self.addCleanup(stack.cleanup)
        baseline_path = root / "baseline.jsonl"
        variant_path = root / "variant.jsonl"
        self._write_predictions(
            baseline_path,
            [
                prediction(truths[0]),
                prediction(truths[1], adjudication="APPROVED"),
            ],
        )
        self._write_predictions(
            variant_path,
            [prediction(truths[0]), prediction(truths[1])],
        )
        variant = next(
            item
            for item in registered_variants()
            if item.variant_id == "without_targeted_rapidocr"
        )
        paths = []
        for repeat in (1, 2):
            paths.extend(
                [
                    self._observation(
                        root / f"baseline-{repeat}.json",
                        variant_id="baseline",
                        repeat=repeat,
                        predictions_path=baseline_path,
                        cpu=11.0,
                        config=BASELINE_CONFIG,
                    ),
                    self._observation(
                        root / f"variant-{repeat}.json",
                        variant_id=variant.variant_id,
                        repeat=repeat,
                        predictions_path=variant_path,
                        cpu=9.0,
                        config=variant.config,
                    ),
                ]
            )

        report = build_report(
            repo_root=REPO_ROOT,
            truth_path=truth_path,
            observation_paths=paths,
        )
        entry = next(
            item
            for item in report["variants"]
            if item["variant_id"] == variant.variant_id
        )

        self.assertFalse(entry["safety_pass"])
        self.assertFalse(entry["recommendation_eligible"])

    def test_checkbox_activity_is_rendered_and_part_of_determinism(self):
        stack, root, truth_path, truths = self._fixture()
        self.addCleanup(stack.cleanup)
        predictions_path = root / "predictions.jsonl"
        self._write_predictions(
            predictions_path,
            [prediction(truths[0]), prediction(truths[1])],
        )
        variant = next(
            item
            for item in registered_variants()
            if item.variant_id == "with_checked_fee_option_recovery"
        )
        baseline_paths = [
            self._observation(
                root / f"baseline-{repeat}.json",
                variant_id="baseline",
                repeat=repeat,
                predictions_path=predictions_path,
                cpu=10.0,
                config=BASELINE_CONFIG,
            )
            for repeat in (1, 2)
        ]
        activity = {
            "pages_scanned": 2,
            "complete_groups": 0,
            "checked_groups": 0,
            "candidates_added": 0,
            "ambiguous_groups": 0,
        }
        variant_paths = [
            self._observation(
                root / f"variant-{repeat}.json",
                variant_id=variant.variant_id,
                repeat=repeat,
                predictions_path=predictions_path,
                cpu=10.5,
                config=variant.config,
                activity=activity,
            )
            for repeat in (1, 2)
        ]
        report = build_report(
            repo_root=REPO_ROOT,
            truth_path=truth_path,
            observation_paths=baseline_paths + variant_paths,
        )
        markdown = render_markdown(report)

        entry = next(
            item
            for item in report["variants"]
            if item["variant_id"] == variant.variant_id
        )
        self.assertTrue(entry["deterministic"])
        self.assertEqual(entry["enabled_activity_counts"], activity)
        self.assertIn("`complete_groups=0`", markdown)
        self.assertNotIn("MIB-000001", markdown)

        changed_activity = dict(activity)
        changed_activity["pages_scanned"] = 3
        changed_path = self._observation(
            root / "variant-2-changed.json",
            variant_id=variant.variant_id,
            repeat=2,
            predictions_path=predictions_path,
            cpu=10.5,
            config=variant.config,
            activity=changed_activity,
        )
        changed_report = build_report(
            repo_root=REPO_ROOT,
            truth_path=truth_path,
            observation_paths=baseline_paths + [variant_paths[0], changed_path],
        )
        changed_entry = next(
            item
            for item in changed_report["variants"]
            if item["variant_id"] == variant.variant_id
        )
        self.assertFalse(changed_entry["deterministic"])
        self.assertEqual(
            changed_entry["evidence_status"],
            "insufficient_evidence",
        )


if __name__ == "__main__":
    unittest.main()
