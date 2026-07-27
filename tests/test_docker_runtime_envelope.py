import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_docker_submission as docker_runner


def valid_row(case_id):
    return {
        "case_id": case_id,
        "applicant_name": "Test Applicant",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-1000",
        "arrival_date": "2026-01-01",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "APPROVED",
        "confidence": 0.9,
    }


def run_metric(index, elapsed, output_hash="a" * 64):
    return docker_runner.ContainerRun(
        repeat_index=index,
        elapsed_seconds=elapsed,
        peak_process_tree_rss_bytes=index * 1024 * 1024,
        peak_container_memory_bytes=(index + 1) * 1024 * 1024,
        output_sha256=output_hash,
        output_bytes=100,
        output=docker_runner.OutputSummary(
            attempted=2,
            answered=2,
            omitted=0,
            invalid=0,
            rows_emitted=2,
        ),
    )


class DockerRuntimeEnvelopeTests(unittest.TestCase):
    def test_container_cleanup_is_time_bounded_and_best_effort(self):
        with mock.patch.object(
            docker_runner.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["docker", "rm"], 15),
        ) as run_mock:
            docker_runner._best_effort_remove_container("mib-test")

        self.assertEqual(run_mock.call_args.kwargs["timeout"], 15.0)

    def test_parses_docker_memory_units(self):
        self.assertEqual(docker_runner.parse_memory_bytes("0B / 8GiB"), 0)
        self.assertEqual(
            docker_runner.parse_memory_bytes("12.5MiB / 8GiB"),
            int(12.5 * 1024 * 1024),
        )
        self.assertEqual(
            docker_runner.parse_memory_bytes("1.25 GB / 8 GB"),
            int(1.25 * 1000**3),
        )
        with self.assertRaises(docker_runner.RuntimeEnvelopeError):
            docker_runner.parse_memory_bytes("unknown")

    def test_container_command_enforces_the_complete_resource_envelope(self):
        command = docker_runner.build_container_command(
            image_tag="submission:test",
            container_name="mib-test",
            input_dir=Path("/input-host"),
            output_dir=Path("/output-host"),
            output_name="predictions.jsonl",
            metrics_name="runtime-metrics.json",
            cpus="4",
            memory="8g",
        )

        rendered = " ".join(command)
        for required in (
            "--network none",
            "--cpus 4",
            "--memory 8g",
            "--memory-swap 8g",
            "--read-only",
            "--cap-drop ALL",
            "no-new-privileges",
            "/tmp:rw,nosuid,nodev,size=2g",
            "dst=/input,readonly",
            "--entrypoint python3",
            "/app/run.sh /input /output/predictions.jsonl",
        ):
            self.assertIn(required, rendered)
        compile(
            docker_runner._RUNTIME_RSS_WRAPPER_SCRIPT,
            "<runtime-rss-wrapper>",
            "exec",
        )
        self.assertIn(
            "os.chmod(metrics_path, 0o644)",
            docker_runner._RUNTIME_RSS_WRAPPER_SCRIPT,
        )
        self.assertIn(
            "os.chmod(output_path, 0o644)",
            docker_runner._RUNTIME_RSS_WRAPPER_SCRIPT,
        )

    def test_latency_percentiles_are_explicitly_repeat_normalized(self):
        summary = docker_runner.latency_summary(
            (run_metric(1, 10.0), run_metric(2, 14.0))
        )

        self.assertIn(
            "not_individual_pdf_latency",
            summary["measurement_basis"],
        )
        self.assertEqual(summary["total_elapsed_seconds"], 24.0)
        self.assertEqual(summary["per_pdf_seconds"]["average"], 6.0)
        self.assertEqual(summary["per_pdf_seconds"]["p50"], 6.0)
        self.assertAlmostEqual(summary["per_pdf_seconds"]["p90"], 6.8)
        self.assertAlmostEqual(summary["per_pdf_seconds"]["p95"], 6.9)
        self.assertEqual(summary["per_pdf_seconds"]["max"], 7.0)

    def test_determinism_requires_two_byte_identical_outputs(self):
        self.assertTrue(
            docker_runner.determinism_summary(
                (run_metric(1, 1.0), run_metric(2, 1.1))
            )["byte_identical"]
        )
        with self.assertRaisesRegex(
            docker_runner.RuntimeEnvelopeError,
            "exactly identical",
        ):
            docker_runner.determinism_summary(
                (
                    run_metric(1, 1.0),
                    run_metric(2, 1.1, output_hash="b" * 64),
                )
            )
        with self.assertRaisesRegex(
            docker_runner.RuntimeEnvelopeError,
            "at least two",
        ):
            docker_runner.determinism_summary((run_metric(1, 1.0),))

    def test_single_run_capture_never_self_certifies_determinism(self):
        evidence = docker_runner.determinism_evidence(
            (run_metric(1, 1.0),),
            single_run_capture=True,
        )

        self.assertFalse(evidence["evaluated"])
        self.assertIn("external comparison", evidence["reason"])
        self.assertEqual(
            docker_runner.completed_status(
                single_run_capture=True,
                warnings=("per_case_deadline_not_enforced",),
            ),
            "CAPTURE_COMPLETE_WITH_WARNINGS",
        )
        with self.assertRaisesRegex(
            docker_runner.RuntimeEnvelopeError,
            "exactly one",
        ):
            docker_runner.determinism_evidence(
                (run_metric(1, 1.0), run_metric(2, 1.1)),
                single_run_capture=True,
            )

    def test_every_repeat_must_meet_the_per_pdf_runtime_limit(self):
        passing = (run_metric(1, 10.0), run_metric(2, 12.0))
        docker_runner.require_each_repeat_within_average_limit(
            passing,
            max_seconds_per_pdf=6.0,
        )

        with self.assertRaisesRegex(
            docker_runner.RuntimeEnvelopeError,
            "at least one repeat",
        ):
            docker_runner.require_each_repeat_within_average_limit(
                (run_metric(1, 10.0), run_metric(2, 12.1)),
                max_seconds_per_pdf=6.0,
            )

    def test_output_summary_reports_coverage_without_case_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            output.write_text(
                json.dumps(valid_row("MIB-000001")) + "\n",
                encoding="utf-8",
            )

            summary = docker_runner.summarize_output(
                output,
                expected_ids=("MIB-000001", "MIB-000002"),
            )

        self.assertEqual(
            summary.to_dict(),
            {
                "attempted": 2,
                "answered": 1,
                "omitted": 1,
                "invalid": 0,
                "rows_emitted": 1,
            },
        )
        self.assertNotIn("MIB-", json.dumps(summary.to_dict()))

    def test_certification_requires_exact_canonical_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.jsonl"
            canonical.write_text(
                json.dumps(
                    valid_row("MIB-000001"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            docker_runner.require_canonical_jsonl(canonical)

            pretty = root / "pretty.json"
            pretty.write_text(
                json.dumps([valid_row("MIB-000001")], indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                docker_runner.RuntimeEnvelopeError,
                "canonical",
            ):
                docker_runner.require_canonical_jsonl(pretty)

            unsorted = root / "unsorted.jsonl"
            unsorted.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":")) + "\n"
                    for row in (
                        valid_row("MIB-000002"),
                        valid_row("MIB-000001"),
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                docker_runner.RuntimeEnvelopeError,
                "ordered",
            ):
                docker_runner.require_canonical_jsonl(unsorted)

    def test_image_inventory_counts_onnx_and_traineddata_and_fails_limits(self):
        payload = {
            "scan_roots": [
                "/app",
                "/opt",
                "/usr/local/lib/python3.12/site-packages",
                "/usr/share/tesseract-ocr",
            ],
            "artifacts": [
                {"path": "/app/model.onnx", "bytes": 10},
                {
                    "path": "/usr/share/tesseract-ocr/eng.traineddata",
                    "bytes": 20,
                },
                {
                    "path": (
                        "/app/mib_pipeline/artifacts/"
                        "confidence_calibration.json"
                    ),
                    "bytes": 5,
                },
            ],
        }
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        with mock.patch.object(
            docker_runner.subprocess,
            "run",
            return_value=completed,
        ):
            inventory = docker_runner.scan_image_model_artifacts(
                "submission:test",
                cpus="4",
                memory="8g",
                max_model_bytes=25,
                max_total_bytes=45,
            )
            self.assertEqual(
                inventory["scan_scope"],
                "fixed_runtime_roots",
            )
            self.assertNotIn("scan_roots", inventory)
            self.assertIn(".json", inventory["extensions"])
            self.assertEqual(inventory["artifact_count"], 3)
            self.assertEqual(inventory["total_bytes"], 35)
            self.assertEqual(
                inventory["extension_counts"],
                {".json": 1, ".onnx": 1, ".traineddata": 1},
            )
            self.assertNotIn("/app/", json.dumps(inventory))
            with self.assertRaisesRegex(
                docker_runner.RuntimeEnvelopeError,
                "per-file",
            ):
                docker_runner.scan_image_model_artifacts(
                    "submission:test",
                    cpus="4",
                    memory="8g",
                    max_model_bytes=15,
                    max_total_bytes=45,
                )

    def test_image_identity_requires_exact_source_binding_labels(self):
        valid_labels = json.dumps(
            {
                "mib.wo20.source_revision": "a" * 40,
                "mib.wo20.producer_graph_sha256": "b" * 64,
            }
        )
        with mock.patch.object(
            docker_runner,
            "docker_output",
            side_effect=(
                "sha256:" + "c" * 64,
                "linux/amd64",
                valid_labels,
            ),
        ):
            identity = docker_runner.image_identity(
                "submission:test",
                expected_source_revision="a" * 40,
                expected_producer_graph_sha256="b" * 64,
            )
        self.assertTrue(identity["source_binding_labels_match"])

        with mock.patch.object(
            docker_runner,
            "docker_output",
            side_effect=(
                "sha256:" + "c" * 64,
                "linux/amd64",
                "{}",
            ),
        ):
            with self.assertRaisesRegex(
                docker_runner.RuntimeEnvelopeError,
                "do not match",
            ):
                docker_runner.image_identity(
                    "submission:test",
                    expected_source_revision="a" * 40,
                    expected_producer_graph_sha256="b" * 64,
                )

    def test_docker_unavailable_writes_honest_identity_free_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            input_dir = root / "input"
            repo.mkdir()
            input_dir.mkdir()
            (repo / "Dockerfile").write_text("FROM scratch\n")
            (input_dir / "MIB-000001.pdf").write_bytes(b"%PDF")
            output = root / "predictions.jsonl"
            evidence = root / "runtime-evidence.json"

            with mock.patch.object(
                docker_runner,
                "docker_status",
                return_value=(False, "docker_cli_missing"),
            ):
                result = docker_runner.main(
                    [
                        "--repo",
                        str(repo),
                        "--input-dir",
                        str(input_dir),
                        "--output",
                        str(output),
                        "--evidence-json",
                        str(evidence),
                    ]
                )

            payload = json.loads(evidence.read_text(encoding="utf-8"))

        self.assertEqual(result, 3)
        self.assertEqual(payload["status"], "DOCKER_UNAVAILABLE")
        self.assertEqual(
            payload["blocking_reasons"],
            ["docker_cli_missing"],
        )
        self.assertEqual(payload["limits"]["max_output_bytes"], 25 * 1024 * 1024)
        self.assertNotIn("MIB-000001", json.dumps(payload))

    def test_successful_main_writes_repeat_metrics_and_exact_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            input_dir = root / "input"
            repo.mkdir()
            input_dir.mkdir()
            (repo / "Dockerfile").write_text("FROM scratch\n")
            for case_id in ("MIB-000001", "MIB-000002"):
                (input_dir / f"{case_id}.pdf").write_bytes(b"%PDF")
            output = root / "predictions.jsonl"
            evidence = root / "runtime-evidence.json"
            calls = []
            output_directory_modes = []

            def fake_execute(command, **_kwargs):
                calls.append(command)
                output_mount = next(
                    item
                    for item in command
                    if item.startswith("type=bind,src=")
                    and item.endswith(",dst=/output")
                )
                host_dir = Path(
                    output_mount[
                        len("type=bind,src=") : -len(",dst=/output")
                    ]
                )
                output_directory_modes.append(
                    host_dir.stat().st_mode & 0o7777
                )
                output_name = Path(command[-2]).name
                rendered = "".join(
                    json.dumps(valid_row(case_id), separators=(",", ":"))
                    + "\n"
                    for case_id in ("MIB-000001", "MIB-000002")
                )
                (host_dir / output_name).write_text(
                    rendered,
                    encoding="utf-8",
                )
                return (
                    2.0 + len(calls) / 10,
                    80 * 1024 * 1024,
                    100 * 1024 * 1024,
                )

            with (
                mock.patch.object(
                    docker_runner,
                    "docker_status",
                    return_value=(True, "test-version"),
                ),
                mock.patch.object(docker_runner, "run"),
                mock.patch.object(
                    docker_runner,
                    "source_bindings",
                    return_value={
                        "git_revision": "a" * 40,
                        "producer_graph_sha256": "b" * 64,
                        "clean_worktree": True,
                    },
                ),
                mock.patch.object(
                    docker_runner,
                    "input_tree_sha256",
                    return_value="c" * 64,
                ),
                mock.patch.object(
                    docker_runner,
                    "image_size_bytes",
                    return_value=500 * 1024 * 1024,
                ),
                mock.patch.object(
                    docker_runner,
                    "image_identity",
                    return_value={
                        "image_id": "sha256:" + "d" * 64,
                        "operating_system": "linux",
                        "architecture": "amd64",
                        "source_binding_labels_match": True,
                    },
                ),
                mock.patch.object(
                    docker_runner,
                    "scan_image_model_artifacts",
                    return_value={
                        "scan_scope": "fixed_runtime_roots",
                        "extensions": [".onnx", ".traineddata"],
                        "artifact_count": 2,
                        "extension_counts": {
                            ".onnx": 1,
                            ".traineddata": 1,
                        },
                        "total_bytes": 30,
                        "maximum_artifact_bytes": 20,
                        "artifacts": [
                            {"extension": ".onnx", "bytes": 10},
                            {"extension": ".traineddata", "bytes": 20},
                        ],
                    },
                ),
                mock.patch.object(
                    docker_runner,
                    "execute_container",
                    side_effect=fake_execute,
                ),
            ):
                result = docker_runner.main(
                    [
                        "--repo",
                        str(repo),
                        "--input-dir",
                        str(input_dir),
                        "--output",
                        str(output),
                        "--evidence-json",
                        str(evidence),
                    ]
                )

            payload = json.loads(evidence.read_text(encoding="utf-8"))
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            all("sha256:" + "d" * 64 in command for command in calls)
        )
        self.assertEqual(output_directory_modes, [0o1777, 0o1777])
        self.assertEqual(payload["status"], "PASS_WITH_WARNINGS")
        self.assertEqual(payload["image"]["build_mode"], "fresh")
        self.assertTrue(payload["determinism"]["byte_identical"])
        self.assertEqual(payload["coverage"]["attempted"], 2)
        self.assertEqual(payload["coverage"]["answered"], 2)
        self.assertTrue(payload["bindings_reverified_after_runs"])
        self.assertEqual(
            payload["runtime"]["peak_process_tree_rss_mib"],
            80.0,
        )
        self.assertEqual(
            payload["runtime"]["peak_container_memory_mib"],
            100.0,
        )
        self.assertIn(
            "per_case_deadline_not_enforced",
            payload["warnings"],
        )
        self.assertEqual(output_text.count("\n"), 2)
        self.assertNotIn("MIB-000001", json.dumps(payload))

    def test_pdf_inventory_matches_runtime_case_insensitive_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            (input_dir / "MIB-000001.PDF").write_bytes(b"%PDF")
            (input_dir / "MIB-000002.pdf").write_bytes(b"%PDF")
            (input_dir / "notes.txt").write_text("not a case")

            expected = docker_runner._expected_ids(input_dir, None)

        self.assertEqual(expected, ("MIB-000001", "MIB-000002"))

    def test_manifest_must_exactly_match_actual_pdf_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "MIB-000001.pdf").write_bytes(b"%PDF")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "case_id\nMIB-000002\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                docker_runner.RuntimeEnvelopeError,
                "exactly match",
            ):
                docker_runner._expected_ids(input_dir, manifest)


if __name__ == "__main__":
    unittest.main()
