from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devtools.wo20_parallel_compare import (
    CAPTURE_SCHEMA,
    IMAGE_MODEL_SCAN_SCOPE,
    OUTPUT_SCHEMA,
    WO20ParallelCompareError,
    compare_capture_evidence,
    main,
)
from scripts import run_docker_submission as docker_runner


REVISION = "a" * 40
PDF_COUNT = 5000
OUTPUT_HASH = "b" * 64


def _coverage() -> dict[str, int]:
    return {
        "attempted": PDF_COUNT,
        "answered": PDF_COUNT,
        "omitted": 0,
        "invalid": 0,
        "rows_emitted": PDF_COUNT,
    }


def _capture(
    *,
    image_id_character: str,
    elapsed_seconds: float = 18_000.0,
) -> dict[str, object]:
    peak_rss = 1024 * 1024 * 1024
    peak_container = 1536 * 1024 * 1024
    seconds_per_pdf = elapsed_seconds / PDF_COUNT
    coverage = _coverage()
    artifacts = [
        {"extension": ".onnx", "bytes": 20 * 1024 * 1024},
        {"extension": ".traineddata", "bytes": 10 * 1024 * 1024},
        {"extension": ".json", "bytes": 4096},
    ]
    return {
        "schema_version": CAPTURE_SCHEMA,
        "mode": "single_run_capture",
        "status": "CAPTURE_COMPLETE_WITH_WARNINGS",
        "blocking_reasons": [],
        "warnings": ["per_case_deadline_not_enforced"],
        "environment": {
            "docker_available": True,
            "docker_status": "available",
            "docker_server_version": "28.0.4",
        },
        "limits": {
            "cpus": "4",
            "memory": "8g",
            "network": "none",
            "read_only_root": True,
            "read_only_input": True,
            "tmpfs": "/tmp:rw,nosuid,nodev,size=2g",
            "pids_limit": 512,
            "max_image_bytes": 4 * 1024**3,
            "max_model_artifact_bytes": 250 * 1024**2,
            "max_total_model_bytes": 1024 * 1024**2,
            "max_output_bytes": 25 * 1024**2,
            "timeout_seconds": 19_500,
            "max_average_seconds_per_pdf": 6.0,
        },
        "requested_repeat_count": 1,
        "deadline_controls": {
            "whole_run_timeout_seconds": 19_500,
            "per_case_deadline_enforced": False,
        },
        "resilience": {
            "output_commit_strategy": "batch_end_atomic",
            "partial_progress_recovery": False,
        },
        "source_binding": {
            "git_revision": REVISION,
            "dockerfile_sha256": "c" * 64,
            "requirements_lock_sha256": "d" * 64,
            "run_sh_sha256": "e" * 64,
            "solution_sha256": "f" * 64,
            "harness_sha256": "1" * 64,
            "producer_graph_sha256": "2" * 64,
            "clean_worktree": True,
        },
        "input_binding": {
            "pdf_count": PDF_COUNT,
            "input_tree_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "manifest_matches_pdf_inventory": True,
        },
        "image": {
            "size_bytes": 768 * 1024**2,
            "within_limit": True,
            "build_mode": "reused",
            "image_id": "sha256:" + image_id_character * 64,
            "operating_system": "linux",
            "architecture": "amd64",
            "source_binding_labels_match": True,
        },
        "installed_model_artifacts": {
            "scan_scope": IMAGE_MODEL_SCAN_SCOPE,
            "extensions": [".json", ".onnx", ".traineddata"],
            "artifact_count": 3,
            "extension_counts": {
                ".json": 1,
                ".onnx": 1,
                ".traineddata": 1,
            },
            "total_bytes": sum(item["bytes"] for item in artifacts),
            "maximum_artifact_bytes": max(
                item["bytes"] for item in artifacts
            ),
            "artifacts": artifacts,
        },
        "runs": [
            {
                "repeat_index": 1,
                "elapsed_seconds": elapsed_seconds,
                "peak_process_tree_rss_bytes": peak_rss,
                "peak_process_tree_rss_mib": peak_rss / 1024**2,
                "peak_process_tree_rss_source": (
                    "in_container_procfs_summed_process_tree_vmrss"
                ),
                "peak_container_memory_bytes": peak_container,
                "peak_container_memory_mib": peak_container / 1024**2,
                "peak_container_memory_source": (
                    "max_available_in_container_cgroup_and_docker_stats"
                ),
                "peak_container_memory_components": {
                    "combination": "maximum_of_available_sources",
                    "in_container_cgroup_bytes": peak_container,
                    "in_container_cgroup_source": (
                        "cgroup_v2_memory_peak"
                    ),
                    "docker_stats_peak_bytes": peak_rss,
                    "docker_stats_sample_count": 1,
                },
                "output_sha256": OUTPUT_HASH,
                "output_bytes": 1_000_000,
                "coverage": coverage,
            }
        ],
        "coverage": coverage,
        "determinism": {
            "evaluated": False,
            "reason": (
                "single capture requires an external comparison against an "
                "independently executed capture"
            ),
        },
        "bindings_reverified_after_runs": True,
        "runtime": {
            "measurement_basis": (
                "full_container_elapsed_divided_by_attempted_pdf_count; "
                "percentiles_across_repeat_normalized_runs_not_individual_"
                "pdf_latency"
            ),
            "repeat_sample_count": 1,
            "total_elapsed_seconds": elapsed_seconds,
            "mean_run_elapsed_seconds": elapsed_seconds,
            "per_pdf_seconds": {
                "average": seconds_per_pdf,
                "p50": seconds_per_pdf,
                "p90": seconds_per_pdf,
                "p95": seconds_per_pdf,
                "max": seconds_per_pdf,
            },
            "all_repeats_within_average_limit": True,
            "peak_process_tree_rss_bytes": peak_rss,
            "peak_process_tree_rss_mib": peak_rss / 1024**2,
            "peak_process_tree_rss_source": (
                "in_container_procfs_summed_process_tree_vmrss"
            ),
            "peak_container_memory_bytes": peak_container,
            "peak_container_memory_mib": peak_container / 1024**2,
            "peak_container_memory_source": (
                "max_available_in_container_cgroup_and_docker_stats"
            ),
        },
    }


class WO20ParallelCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.left = self.root / "left.json"
        self.right = self.root / "right.json"
        self.left_payload = _capture(image_id_character="5")
        self.right_payload = _capture(
            image_id_character="6",
            elapsed_seconds=18_250.0,
        )
        self._write()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self) -> None:
        self.left.write_text(
            json.dumps(self.left_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.right.write_text(
            json.dumps(self.right_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _compare(self):
        return compare_capture_evidence(
            (self.left, self.right),
            expected_source_revision=REVISION,
            expected_pdf_count=PDF_COUNT,
        )

    def test_matching_captures_emit_aggregate_only_determinism(self) -> None:
        result = self._compare()

        self.assertEqual(result["schema_version"], OUTPUT_SCHEMA)
        self.assertEqual(result["status"], "PASS_WITH_WARNINGS")
        self.assertTrue(result["aggregate_only"])
        self.assertTrue(result["determinism"]["evaluated"])
        self.assertTrue(result["determinism"]["byte_identical"])
        self.assertEqual(
            result["determinism"]["output_sha256"],
            OUTPUT_HASH,
        )
        self.assertEqual(
            result["images"]["image_ids"],
            ["sha256:" + "5" * 64, "sha256:" + "6" * 64],
        )
        self.assertEqual(
            len(result["runtime"]["peak_container_memory_components"]),
            2,
        )
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("case_id", rendered.casefold())
        self.assertNotIn("predictions", rendered.casefold())
        self.assertNotIn(str(self.root), rendered)

    def test_capture_fixture_matches_runtime_model_inventory_scope(self) -> None:
        self.assertEqual(
            IMAGE_MODEL_SCAN_SCOPE,
            docker_runner.IMAGE_MODEL_SCAN_SCOPE,
        )
        self.assertEqual(
            self.left_payload["installed_model_artifacts"]["scan_scope"],
            docker_runner.IMAGE_MODEL_SCAN_SCOPE,
        )
        self.assertNotIn(
            "scan_roots",
            self.left_payload["installed_model_artifacts"],
        )

    def test_real_scanner_inventory_shape_is_accepted_by_comparator(self) -> None:
        scanner_payload = {
            "scan_roots": [
                "/app",
                "/opt",
                "/usr/local/lib/python3.12/site-packages",
                "/usr/share/tesseract-ocr",
            ],
            "artifacts": [
                {"path": "/app/ocr.onnx", "bytes": 1024},
                {
                    "path": (
                        "/app/mib_pipeline/artifacts/"
                        "confidence_calibration.json"
                    ),
                    "bytes": 2048,
                },
                {
                    "path": "/usr/share/tesseract-ocr/eng.traineddata",
                    "bytes": 4096,
                },
            ],
        }
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(scanner_payload),
            stderr="",
        )
        with mock.patch.object(
            docker_runner.subprocess,
            "run",
            return_value=completed,
        ):
            inventory = docker_runner.scan_image_model_artifacts(
                "sha256:" + "8" * 64,
                cpus="4",
                memory="8g",
                max_model_bytes=250 * 1024**2,
                max_total_bytes=1024 * 1024**2,
            )

        self.left_payload["installed_model_artifacts"] = inventory
        self.right_payload["installed_model_artifacts"] = copy.deepcopy(
            inventory
        )
        self._write()

        result = self._compare()

        self.assertEqual(result["status"], "PASS_WITH_WARNINGS")
        self.assertEqual(
            result["installed_model_artifacts"]["extension_counts"],
            {".json": 1, ".onnx": 1, ".traineddata": 1},
        )

    def test_output_hash_mismatch_fails_closed(self) -> None:
        self.right_payload["runs"][0]["output_sha256"] = "7" * 64
        self._write()

        with self.assertRaisesRegex(
            WO20ParallelCompareError,
            "SHA-256 differs",
        ) as caught:
            self._compare()

        self.assertEqual(caught.exception.code, "output_hash_mismatch")

    def test_source_and_input_bindings_must_match_exactly(self) -> None:
        self.right_payload["input_binding"]["input_tree_sha256"] = "8" * 64
        self._write()

        with self.assertRaises(WO20ParallelCompareError) as caught:
            self._compare()

        self.assertEqual(caught.exception.code, "input_binding_mismatch")

    def test_single_capture_cannot_self_certify_determinism(self) -> None:
        self.left_payload["determinism"] = {
            "evaluated": True,
            "reason": "already deterministic",
        }
        self._write()

        with self.assertRaises(WO20ParallelCompareError) as caught:
            self._compare()

        self.assertEqual(
            caught.exception.code,
            "capture_self_certified_determinism",
        )

    def test_runtime_and_container_memory_limits_are_hard_gates(self) -> None:
        self.right_payload["runs"][0]["elapsed_seconds"] = 30_001.0
        self.right_payload["runtime"]["total_elapsed_seconds"] = 30_001.0
        self.right_payload["runtime"]["mean_run_elapsed_seconds"] = 30_001.0
        for key in ("average", "p50", "p90", "p95", "max"):
            self.right_payload["runtime"]["per_pdf_seconds"][key] = (
                30_001.0 / PDF_COUNT
            )
        self._write()

        with self.assertRaises(WO20ParallelCompareError) as caught:
            self._compare()

        self.assertEqual(caught.exception.code, "runtime_limit_exceeded")

        self.right_payload = _capture(image_id_character="6")
        self.right_payload["runs"][0]["peak_container_memory_bytes"] = (
            8 * 1024**3 + 1
        )
        self.right_payload["runs"][0]["peak_container_memory_mib"] = (
            (8 * 1024**3 + 1) / 1024**2
        )
        self.right_payload["runtime"]["peak_container_memory_bytes"] = (
            8 * 1024**3 + 1
        )
        self.right_payload["runtime"]["peak_container_memory_mib"] = (
            (8 * 1024**3 + 1) / 1024**2
        )
        self._write()

        with self.assertRaises(WO20ParallelCompareError) as caught:
            self._compare()

        self.assertEqual(
            caught.exception.code,
            "container_memory_limit_exceeded",
        )

    def test_container_memory_components_are_bound_to_reported_peak(self) -> None:
        self.right_payload["runs"][0][
            "peak_container_memory_components"
        ]["in_container_cgroup_bytes"] -= 1
        self._write()

        with self.assertRaises(WO20ParallelCompareError) as caught:
            self._compare()

        self.assertEqual(caught.exception.code, "run_evidence_invalid")

    def test_exactly_two_distinct_capture_files_are_required(self) -> None:
        with self.assertRaises(WO20ParallelCompareError) as caught:
            compare_capture_evidence(
                (self.left,),
                expected_source_revision=REVISION,
            )
        self.assertEqual(caught.exception.code, "capture_count_invalid")

        with self.assertRaises(WO20ParallelCompareError) as caught:
            compare_capture_evidence(
                (self.left, self.left),
                expected_source_revision=REVISION,
            )
        self.assertEqual(
            caught.exception.code,
            "capture_files_not_independent",
        )

    def test_cli_writes_blocked_aggregate_on_comparison_failure(self) -> None:
        self.right_payload["runs"][0]["output_sha256"] = "9" * 64
        self._write()
        output = self.root / "aggregate.json"
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            return_code = main(
                (
                    "--capture",
                    str(self.left),
                    "--capture",
                    str(self.right),
                    "--expected-source-revision",
                    REVISION,
                    "--expected-pdf-count",
                    str(PDF_COUNT),
                    "--output",
                    str(output),
                )
            )

        self.assertEqual(return_code, 2)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertEqual(
            payload["blocking_reasons"],
            ["output_hash_mismatch"],
        )
        self.assertFalse(payload["determinism"]["evaluated"])
        self.assertNotIn(str(self.root), output.read_text(encoding="utf-8"))

    def test_duplicate_json_keys_are_rejected(self) -> None:
        raw = self.left.read_text(encoding="utf-8")
        self.left.write_text(
            raw.replace(
                '"mode": "single_run_capture",',
                (
                    '"mode": "single_run_capture", '
                    '"mode": "single_run_capture",'
                ),
                1,
            ),
            encoding="utf-8",
        )

        with self.assertRaises(WO20ParallelCompareError) as caught:
            self._compare()

        self.assertEqual(caught.exception.code, "duplicate_json_key")


if __name__ == "__main__":
    unittest.main()
