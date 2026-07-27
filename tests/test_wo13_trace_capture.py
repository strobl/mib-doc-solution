import json
import inspect
import hashlib
import os
import subprocess
import tempfile
import threading
import time
import unittest
import warnings
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mib_pipeline import CanonicalJsonlWriter, PredictionRow

from devtools.wo13_trace_capture import (
    DelegatingAdjudicatorObserver,
    DelegatingExtractorObserver,
    DelegatingLinkerObserver,
    DelegatingResolverObserver,
    CaptureAuthority,
    TraceCaptureError,
    TraceCollector,
    TraceSignals,
    TracingCaseProcessor,
    _instrument_rapid_processor,
    _parser,
    _run_trace_capture_core,
    authoritative_git_binding,
    classify_conflict,
    classify_linking,
    classify_ocr_path,
    classify_policy,
    classify_provenance,
    compute_input_tree_sha256,
    private_capture_snapshot,
    runtime_identity,
    source_snapshot_sha256,
    verify_dataset_archive_authority,
    current_trace_signals,
    load_frozen_baseline_authority,
    load_capture_authority,
    load_runtime_contract_authority,
    run_authoritative_trace_capture,
    run_non_authoritative_test_capture,
)
from devtools.wo13_trace_contract import (
    TRACE_SCHEMA_VERSION,
    TraceContractError,
    canonical_json_bytes,
    sha256_path,
    validate_trace_capture,
)


def prediction(case_id):
    return PredictionRow(
        case_id=case_id,
        applicant_name="Test Applicant",
        species_code="ORION_GRAYS",
        home_world="Kepler-186f",
        visa_class="XW-2",
        sponsor_id="SPN-1234",
        arrival_date="2026-04-17",
        declared_purpose="research",
        risk_flags="none",
        fee_status="paid",
        adjudication="APPROVED",
        confidence=0.75,
    )


def trace_payload(rows):
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "capture_mode": "test",
        "source_revision_sha": "a" * 40,
        "checkout_revision_sha": "a" * 40,
        "capture_source_revision_sha": "a" * 40,
        "input_tree_sha256": "b" * 64,
        "processing_snapshot_input_tree_sha256": "b" * 64,
        "layout_manifest_sha256": "c" * 64,
        "dataset_archive_sha256": "d" * 64,
        "runtime_contract_sha256": "2" * 64,
        "frozen_baseline_manifest_sha256": "3" * 64,
        "baseline_predictions_sha256": "1" * 64,
        "runtime_graph_sha256": "e" * 64,
        "trace_tool_sha256": "f" * 64,
        "container_graph_sha256": "4" * 64,
        "source_snapshot_sha256": "5" * 64,
        "authority_manifest_sha256": "6" * 64,
        "runtime_identity_sha256": "7" * 64,
        "dependency_identity_sha256": "8" * 64,
        "python_executable_sha256": "9" * 64,
        "predictions_sha256": "1" * 64,
        "production_tree_verified": False,
        "runtime_contract_verified": False,
        "runtime_environment_verified": False,
        "runtime_interface_verified": False,
        "container_limits_verified": False,
        "processing_snapshot_verified": False,
        "stability_checks": {
            "input_tree_unchanged": False,
            "layout_manifest_unchanged": False,
            "dataset_archive_unchanged": False,
            "runtime_contract_unchanged": False,
            "frozen_baseline_manifest_unchanged": False,
            "baseline_predictions_unchanged": False,
            "runtime_graph_unchanged": False,
            "trace_tool_unchanged": False,
            "container_graph_unchanged": False,
            "source_snapshot_unchanged": False,
            "authority_manifest_unchanged": False,
            "runtime_identity_unchanged": False,
            "checkout_revision_unchanged": False,
            "capture_source_revision_unchanged": False,
            "production_tree_unchanged": False,
        },
        "case_count": len(rows),
        "attempted": len(rows),
        "answered": len(rows),
        "omitted": 0,
        "max_workers": 4,
        "retry_missing_attempts": 1,
        "retry_passes_used": 0,
        "batch_wall_seconds": 1.25,
        "rows": rows,
    }


def trace_row(case_id):
    return {
        "case_id": case_id,
        "provenance_route": "visible_ocr",
        "applicant_linking_state": "linked_unique",
        "evidence_conflict": "none",
        "ocr_recovery_path": "primary",
        "policy_trace": "deterministic_policy",
        "runtime_seconds": 0.25,
    }


def runtime_contract_payload(*, max_workers=4):
    return {
        "schema_version": "mib-wo17-runtime-contract/v1",
        "capture": {
            "arm_repeat_count": 2,
            "execution": "sequential",
            "max_workers": max_workers,
            "metrics_source": (
                "fresh_process_rusage_self_plus_waited_children_"
                "and_monotonic_wall"
            ),
            "required_byte_determinism": True,
        },
        "container_limits": {
            "image_bytes": 4_294_967_296,
            "max_model_artifact_bytes": 262_144_000,
            "model_bytes": 1_073_741_824,
            "network": "none",
            "output_bytes": 26_214_400,
            "peak_memory_bytes": 8_589_934_592,
            "per_record_runtime_seconds": 6,
            "runtime_seconds": 14_400,
            "tmp_bytes": 2_147_483_648,
        },
        "environment": {
            "BLIS_NUM_THREADS": str(max_workers),
            "HOME": "/tmp",
            "MALLOC_ARENA_MAX": str(max_workers),
            "MIB_MAX_WORKERS": str(max_workers),
            "MKL_NUM_THREADS": str(max_workers),
            "NUMEXPR_NUM_THREADS": str(max_workers),
            "OC_DISABLE_DOT_ACCESS_WARNING": "1",
            "OMP_NUM_THREADS": str(max_workers),
            "OPENBLAS_NUM_THREADS": str(max_workers),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": "/tmp",
            "TOKENIZERS_PARALLELISM": "false",
            "VECLIB_MAXIMUM_THREADS": str(max_workers),
        },
        "evaluation": {
            "evaluator_sha256": "1" * 64,
            "evidence_label": "public_grouped_robustness_not_unseen",
            "expected_record_count": 1,
            "input_tree_sha256": "2" * 64,
            "layout_manifest_sha256": "3" * 64,
            "truth_sha256": "4" * 64,
        },
        "interface": {
            "entrypoint": "solution.py",
            "input": "directory containing canonical PDF cases",
            "output": "canonical twelve-field JSONL",
            "runner": "run.sh",
        },
    }


def write_dataset_zip(path, members):
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for name, content in members:
            archive.writestr(name, content)


class FakeProcessor:
    def __init__(self, *, fail_once=None, delay_seconds=0.0):
        self._fail_once = set(fail_once or ())
        self._failed = set()
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()

    def process_case(self, pdf_path):
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        case_id = pdf_path.stem
        with self._lock:
            should_fail = (
                case_id in self._fail_once and case_id not in self._failed
            )
            if should_fail:
                self._failed.add(case_id)
        if should_fail:
            return None

        signals = current_trace_signals()
        if signals is not None:
            signals.primary_candidates = (case_id,)
            signals.linked_cases.append(
                SimpleNamespace(
                    active_applicant=f"Applicant {case_id}",
                    unresolved=False,
                    unresolved_reasons=(),
                )
            )
            signals.resolved_cases.append(
                SimpleNamespace(contested_fields=())
            )
            signals.primary_outcome = SimpleNamespace(
                trace=SimpleNamespace(
                    authoritative_source=False,
                    denial_reasons=(),
                    review_reasons=(),
                    approval_facts=(),
                )
            )
        return prediction(case_id)


class Wo13TraceCaptureTests(unittest.TestCase):
    def test_contract_is_exact_sorted_and_allowlisted(self):
        rows = [trace_row("MIB-000001"), trace_row("MIB-000002")]
        normalized = validate_trace_capture(trace_payload(rows))
        self.assertEqual(normalized["rows"], rows)

        extra = trace_payload(rows)
        extra["unexpected"] = True
        with self.assertRaises(TraceContractError):
            validate_trace_capture(extra)

        unsorted = trace_payload(list(reversed(rows)))
        with self.assertRaises(TraceContractError):
            validate_trace_capture(unsorted)

        bad_category = trace_payload(rows)
        bad_category["rows"][0]["policy_trace"] = "case_secret"
        with self.assertRaises(TraceContractError):
            validate_trace_capture(bad_category)

    def test_delegating_observers_preserve_return_identity_and_call_once(self):
        sentinel = object()

        class Extractor:
            def __init__(self):
                self.calls = 0

            def extract(self, rendered):
                self.calls += 1
                return sentinel

        extractor = Extractor()
        self.assertIs(
            DelegatingExtractorObserver(extractor, role="rapid").extract(
                object()
            ),
            sentinel,
        )
        self.assertEqual(extractor.calls, 1)

        for observer_type, method_name, arguments in (
            (DelegatingLinkerObserver, "link", ("MIB-000001", ())),
            (DelegatingResolverObserver, "resolve", (object(),)),
            (
                DelegatingAdjudicatorObserver,
                "adjudicate_case",
                (object(),),
            ),
        ):
            class Delegate:
                calls = 0

            def delegated(*args):
                Delegate.calls += 1
                return sentinel

            setattr(Delegate, method_name, staticmethod(delegated))
            delegate = Delegate()
            observed = observer_type(delegate)
            self.assertIs(getattr(observed, method_name)(*arguments), sentinel)
            self.assertEqual(Delegate.calls, 1)

    def test_capture_is_thread_isolated_and_prediction_bound(self):
        case_ids = [f"MIB-{index:06d}" for index in range(1, 17)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            for case_id in case_ids:
                (input_dir / f"{case_id}.pdf").write_bytes(b"%PDF-test")
            layout = root / "layout.json"
            layout.write_text("{}", encoding="utf-8")
            expected = root / "expected.jsonl"
            CanonicalJsonlWriter().write(
                expected,
                (prediction(case_id) for case_id in case_ids),
            )
            expected_hash = sha256_path(expected)
            predictions = root / "predictions.jsonl"
            trace = root / "trace.json"
            result = run_non_authoritative_test_capture(
                input_dir=input_dir,
                predictions_output=predictions,
                trace_output=trace,
                source_revision_sha="a" * 40,
                input_tree_sha256=compute_input_tree_sha256(input_dir),
                dataset_archive_sha256="c" * 64,
                layout_manifest_path=layout,
                expected_predictions_sha256=expected_hash,
                max_workers=4,
                processor=FakeProcessor(),
                runtime_sha256="d" * 64,
                trace_tool_sha256="e" * 64,
            )
            payload = json.loads(trace.read_text(encoding="utf-8"))

        self.assertEqual(result.predictions_sha256, expected_hash)
        self.assertEqual(
            payload["capture_mode"],
            "test",
        )
        self.assertFalse(payload["production_tree_verified"])
        self.assertEqual(payload["case_count"], len(case_ids))
        self.assertEqual(
            [row["case_id"] for row in payload["rows"]],
            case_ids,
        )
        self.assertTrue(
            all(
                row["provenance_route"] == "visible_ocr"
                and row["applicant_linking_state"] == "linked_unique"
                for row in payload["rows"]
            )
        )

    def test_prediction_parity_failure_never_finalizes_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "MIB-000001.pdf").write_bytes(b"%PDF-test")
            layout = root / "layout.json"
            layout.write_text("{}", encoding="utf-8")
            trace = root / "trace.json"
            with self.assertRaisesRegex(
                TraceCaptureError,
                "prediction-byte parity failed",
            ):
                run_non_authoritative_test_capture(
                    input_dir=input_dir,
                    predictions_output=root / "predictions.jsonl",
                    trace_output=trace,
                    source_revision_sha="a" * 40,
                    input_tree_sha256=compute_input_tree_sha256(input_dir),
                    dataset_archive_sha256="c" * 64,
                    layout_manifest_path=layout,
                    expected_predictions_sha256="9" * 64,
                    processor=FakeProcessor(),
                    runtime_sha256="d" * 64,
                    trace_tool_sha256="e" * 64,
                )
            self.assertFalse(trace.exists())

    def test_input_tree_digest_is_computed_from_actual_pdf_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "MIB-000001.pdf").write_bytes(b"%PDF-test")
            layout = root / "layout.json"
            layout.write_text("{}", encoding="utf-8")
            trace = root / "trace.json"
            with self.assertRaisesRegex(
                TraceCaptureError,
                "input tree does not match",
            ):
                run_non_authoritative_test_capture(
                    input_dir=input_dir,
                    predictions_output=root / "predictions.jsonl",
                    trace_output=trace,
                    source_revision_sha="a" * 40,
                    input_tree_sha256="0" * 64,
                    dataset_archive_sha256="c" * 64,
                    layout_manifest_path=layout,
                    expected_predictions_sha256="9" * 64,
                    processor=FakeProcessor(),
                    runtime_sha256="d" * 64,
                    trace_tool_sha256="e" * 64,
                )
            self.assertFalse(trace.exists())

    def test_only_missing_cases_are_retried_before_parity_gate(self):
        case_ids = ["MIB-000001", "MIB-000002"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            for case_id in case_ids:
                (input_dir / f"{case_id}.pdf").write_bytes(b"%PDF-test")
            layout = root / "layout.json"
            layout.write_text("{}", encoding="utf-8")
            expected = root / "expected.jsonl"
            CanonicalJsonlWriter().write(
                expected,
                (prediction(case_id) for case_id in case_ids),
            )
            trace = root / "trace.json"
            result = run_non_authoritative_test_capture(
                input_dir=input_dir,
                predictions_output=root / "predictions.jsonl",
                trace_output=trace,
                source_revision_sha="a" * 40,
                input_tree_sha256=compute_input_tree_sha256(input_dir),
                dataset_archive_sha256="c" * 64,
                layout_manifest_path=layout,
                expected_predictions_sha256=sha256_path(expected),
                processor=FakeProcessor(
                    fail_once={"MIB-000002"},
                    delay_seconds=0.01,
                ),
                runtime_sha256="d" * 64,
                trace_tool_sha256="e" * 64,
                retry_missing_attempts=1,
            )
            payload = json.loads(trace.read_text(encoding="utf-8"))

        self.assertEqual(result.report.answered, 2)
        self.assertEqual(result.report.omitted, 0)
        self.assertEqual(result.retry_passes_used, 1)
        self.assertEqual(payload["retry_passes_used"], 1)
        self.assertEqual(len(payload["rows"]), 2)
        retried_row = next(
            row
            for row in payload["rows"]
            if row["case_id"] == "MIB-000002"
        )
        self.assertGreaterEqual(retried_row["runtime_seconds"], 0.018)

    def test_injected_processor_cannot_emit_authoritative_capture(self):
        signature = inspect.signature(run_authoritative_trace_capture)
        self.assertNotIn("processor", signature.parameters)

        payload = trace_payload([trace_row("MIB-000001")])
        payload["capture_mode"] = "authoritative_production"
        payload["production_tree_verified"] = False
        with self.assertRaisesRegex(
            TraceContractError,
            "verification gate",
        ):
            validate_trace_capture(payload)

    def test_internal_core_fake_processor_and_probe_are_forced_to_test_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "MIB-000001.pdf").write_bytes(b"%PDF-test")
            layout = root / "layout.json"
            layout.write_text("{}", encoding="utf-8")
            expected = root / "expected.jsonl"
            CanonicalJsonlWriter().write(
                expected,
                [prediction("MIB-000001")],
            )
            binding = {
                "capture_mode": "authoritative_production",
                "source_revision_sha": "a" * 40,
                "checkout_revision_sha": "b" * 40,
                "capture_source_revision_sha": "b" * 40,
                "input_tree_sha256": compute_input_tree_sha256(input_dir),
                "layout_manifest_sha256": sha256_path(layout),
                "dataset_archive_sha256": "3" * 64,
                "runtime_contract_sha256": "4" * 64,
                "frozen_baseline_manifest_sha256": "5" * 64,
                "baseline_predictions_sha256": sha256_path(expected),
                "runtime_graph_sha256": "6" * 64,
                "trace_tool_sha256": "7" * 64,
                "container_graph_sha256": "8" * 64,
                "source_snapshot_sha256": "9" * 64,
                "authority_manifest_sha256": "1" * 64,
                "runtime_identity_sha256": "2" * 64,
                "dependency_identity_sha256": "a" * 64,
                "python_executable_sha256": "b" * 64,
                "production_tree_verified": True,
                "runtime_contract_verified": True,
                "runtime_environment_verified": True,
                "runtime_interface_verified": True,
                "container_limits_verified": True,
                "processing_snapshot_verified": True,
                "expected_record_count": 1,
            }
            trace = root / "trace.json"
            _run_trace_capture_core(
                input_dir=input_dir,
                predictions_output=root / "predictions.jsonl",
                trace_output=trace,
                layout_manifest_path=layout,
                binding=binding,
                processor=FakeProcessor(),
                stability_probe=lambda: dict(binding),
                _authority_capability=object(),
            )
            payload = json.loads(trace.read_text(encoding="utf-8"))

        self.assertEqual(payload["capture_mode"], "test")
        self.assertFalse(payload["production_tree_verified"])
        self.assertFalse(payload["runtime_contract_verified"])
        self.assertFalse(payload["processing_snapshot_verified"])

    def test_authoritative_contract_requires_every_stability_gate(self):
        payload = trace_payload([trace_row("MIB-000001")])
        payload["capture_mode"] = "authoritative_production"
        for key in (
            "production_tree_verified",
            "runtime_contract_verified",
            "runtime_environment_verified",
            "runtime_interface_verified",
            "container_limits_verified",
            "processing_snapshot_verified",
        ):
            payload[key] = True
        payload["stability_checks"] = {
            key: True for key in payload["stability_checks"]
        }
        validate_trace_capture(payload)
        payload["stability_checks"]["trace_tool_unchanged"] = False
        with self.assertRaisesRegex(
            TraceContractError,
            "every stability check",
        ):
            validate_trace_capture(payload)

    def test_frozen_baseline_rejects_declared_hash_without_matching_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "predictions.jsonl"
            baseline.write_bytes(b"fake baseline bytes\n")
            manifest = root / "frozen.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "mib-frozen-baseline/v1",
                        "metadata": {
                            "baseline_commit_sha": "a" * 40,
                            "count": 1,
                        },
                        "artifacts": [
                            {
                                "path": (
                                    "external/full1000_predictions.jsonl"
                                ),
                                "sha256": (
                                    "d6e23641a4e4c7a5517c2b6917911461"
                                    "77665f5c297667adae17565f6918a42d"
                                ),
                                "size_bytes": baseline.stat().st_size,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "does not match pinned",
            ):
                load_frozen_baseline_authority(manifest, baseline)

    def test_git_binding_rejects_arbitrary_revision(self):
        with self.assertRaises(TraceCaptureError):
            authoritative_git_binding("0" * 40)

    def test_runtime_contract_rejects_one_worker_and_wrong_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_contract = root / "bad-runtime.json"
            bad_contract.write_text(
                json.dumps(runtime_contract_payload(max_workers=1)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "capture contract",
            ):
                load_runtime_contract_authority(bad_contract)

            oversized_payload = runtime_contract_payload()
            oversized_payload["container_limits"][
                "runtime_seconds"
            ] = 30_000
            oversized_contract = root / "oversized-runtime.json"
            oversized_contract.write_text(
                json.dumps(oversized_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "four-hour contract",
            ):
                load_runtime_contract_authority(oversized_contract)

            wrong_interface_payload = runtime_contract_payload()
            wrong_interface_payload["interface"]["runner"] = "other.sh"
            wrong_interface_contract = root / "wrong-interface.json"
            wrong_interface_contract.write_text(
                json.dumps(wrong_interface_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "interface contract",
            ):
                load_runtime_contract_authority(
                    wrong_interface_contract
                )

            contract = root / "runtime.json"
            contract.write_text(
                json.dumps(runtime_contract_payload()),
                encoding="utf-8",
            )
            loaded = load_runtime_contract_authority(contract)
            with patch.dict(
                os.environ,
                {
                    **loaded["environment"],
                    "MIB_MAX_WORKERS": "1",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(
                    TraceCaptureError,
                    "process environment",
                ):
                    runtime_identity(loaded)

    def test_git_authority_rejects_dirty_and_untracked_trace_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required_files = {
                "Dockerfile": "FROM scratch\n",
                "run.sh": "#!/bin/sh\n",
                "solution.py": "pass\n",
                "requirements.lock": "example==1\n",
                "mib_pipeline/__init__.py": "",
                "third_party_licenses/NOTICE": "fixture\n",
                "devtools/__init__.py": "",
                "devtools/wo13_trace_capture.py": "CAPTURE = 1\n",
                "devtools/wo13_trace_contract.py": "CONTRACT = 1\n",
                "devtools/grouped_split_evidence.py": "GROUPS = 1\n",
                "devtools/layout_manifest_freezer.py": "LAYOUT = 1\n",
                "devtools/experiment_control.py": "CONTROL = 1\n",
                "scripts/score_loss_atlas.py": "ATLAS = 1\n",
            }
            for relative, content in required_files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            subprocess.run(
                ["/usr/bin/git", "init", "-q"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["/usr/bin/git", "add", "."],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    "/usr/bin/git",
                    "-c",
                    "user.name=WO13 Test",
                    "-c",
                    "user.email=wo13@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )
            revision = subprocess.run(
                ["/usr/bin/git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            graph_binding = {
                "devtools.wo13_trace_capture.runtime_graph_sha256": (
                    "1" * 64
                ),
                "devtools.wo13_trace_capture.trace_tool_graph_sha256": (
                    "2" * 64
                ),
                "devtools.wo13_trace_capture.container_graph_sha256": (
                    "3" * 64
                ),
            }
            with ExitStack() as stack:
                for target, value in graph_binding.items():
                    stack.enter_context(patch(target, return_value=value))
                clean = authoritative_git_binding(
                    revision,
                    capture_source_revision_sha=revision,
                    repository_root=root,
                )
            self.assertTrue(clean["capture_tree_verified"])

            tool = root / "devtools/wo13_trace_capture.py"
            tool.write_text("CAPTURE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(
                TraceCaptureError,
                "modified or untracked",
            ):
                authoritative_git_binding(
                    revision,
                    capture_source_revision_sha=revision,
                    repository_root=root,
                )
            tool.write_text("CAPTURE = 1\n", encoding="utf-8")
            (root / "devtools/untracked_probe.py").write_text(
                "PROBE = 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "modified or untracked",
            ):
                authoritative_git_binding(
                    revision,
                    capture_source_revision_sha=revision,
                    repository_root=root,
                )

    def test_capture_rejects_archive_authority_change_before_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "MIB-000001.pdf").write_bytes(b"%PDF-test")
            layout = root / "layout.json"
            layout.write_text("{}", encoding="utf-8")
            expected = root / "expected.jsonl"
            CanonicalJsonlWriter().write(
                expected,
                [prediction("MIB-000001")],
            )
            input_hash = compute_input_tree_sha256(input_dir)
            binding = {
                "capture_mode": "authoritative_production",
                "source_revision_sha": "a" * 40,
                "checkout_revision_sha": "a" * 40,
                "input_tree_sha256": input_hash,
                "layout_manifest_sha256": sha256_path(layout),
                "dataset_archive_sha256": "3" * 64,
                "runtime_contract_sha256": "4" * 64,
                "frozen_baseline_manifest_sha256": "5" * 64,
                "baseline_predictions_sha256": sha256_path(expected),
                "runtime_graph_sha256": "6" * 64,
                "trace_tool_sha256": "7" * 64,
                "production_tree_verified": True,
                "expected_record_count": 1,
            }
            changed = dict(binding)
            changed["dataset_archive_sha256"] = "8" * 64
            with self.assertRaisesRegex(
                TraceCaptureError,
                "dataset_archive_unchanged",
            ):
                _run_trace_capture_core(
                    input_dir=input_dir,
                    predictions_output=root / "predictions.jsonl",
                    trace_output=root / "trace.json",
                    layout_manifest_path=layout,
                    binding=binding,
                    processor=FakeProcessor(),
                    stability_probe=lambda: changed,
                )
            self.assertFalse((root / "trace.json").exists())

    def test_preregistered_authority_rejects_substituted_archive_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            pdf = input_dir / "MIB-000001.pdf"
            pdf.write_bytes(b"%PDF-fixture")
            paths = {
                "input_dir": input_dir.resolve(),
                "layout_manifest": (root / "layout.json").resolve(),
                "dataset_archive": (root / "dataset.zip").resolve(),
                "runtime_contract": (root / "runtime.json").resolve(),
                "frozen_baseline_manifest": (
                    root / "frozen.json"
                ).resolve(),
                "baseline_predictions": (
                    root / "baseline.jsonl"
                ).resolve(),
            }
            for name, path in paths.items():
                if name not in {"input_dir", "dataset_archive"}:
                    path.write_bytes(name.encode("utf-8"))
            write_dataset_zip(
                paths["dataset_archive"],
                [("MIB-000001.pdf", pdf.read_bytes())],
            )
            runtime_payload = runtime_contract_payload()
            runtime_loaded = {
                "file_sha256": "4" * 64,
                "input_tree_sha256": "1" * 64,
                "layout_manifest_sha256": sha256_path(
                    paths["layout_manifest"]
                ),
                "expected_record_count": 1,
                "max_workers": 4,
                "environment": runtime_payload["environment"],
                "interface": runtime_payload["interface"],
                "container_limits": runtime_payload["container_limits"],
            }
            unsigned_identity = {
                "python_version": "3.12 fixture",
                "python_implementation": "cpython",
                "python_isolated": True,
                "python_dont_write_bytecode": True,
                "python_executable_sha256": "6" * 64,
                "dependency_versions": {"fixture": "1.0"},
                "dependency_identity_sha256": "7" * 64,
                "environment": runtime_payload["environment"],
                "max_workers": 4,
                "interface": runtime_payload["interface"],
                "container_limits": runtime_payload["container_limits"],
            }
            identity = dict(unsigned_identity)
            identity["runtime_identity_sha256"] = hashlib.sha256(
                canonical_json_bytes(unsigned_identity)
            ).hexdigest()
            archive_hash = sha256_path(paths["dataset_archive"])
            authority_payload = {
                "schema_version": "mib-wo13-capture-authority/v1",
                "source_revision_sha": "a" * 40,
                "approved_paths": {
                    name: path.as_posix()
                    for name, path in sorted(paths.items())
                },
                "expected_hashes": {
                    "input_tree_sha256": "1" * 64,
                    "layout_manifest_sha256": runtime_loaded[
                        "layout_manifest_sha256"
                    ],
                    "dataset_archive_sha256": archive_hash,
                    "runtime_contract_sha256": "4" * 64,
                    "frozen_baseline_manifest_sha256": "5" * 64,
                    "baseline_predictions_sha256": "b" * 64,
                    "runtime_graph_sha256": "8" * 64,
                    "trace_tool_sha256": "9" * 64,
                    "container_graph_sha256": "c" * 64,
                    "source_snapshot_sha256": "d" * 64,
                },
                "expected_record_count": 1,
                "retry_missing_attempts": 1,
                "runtime_identity": identity,
            }
            authority_path = root / "authority.json"
            authority_path.write_bytes(
                canonical_json_bytes(authority_payload)
            )
            paths["dataset_archive"].write_bytes(b"substituted archive")
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "devtools.wo13_trace_capture."
                        "load_runtime_contract_authority",
                        return_value=runtime_loaded,
                    )
                )
                stack.enter_context(
                    patch(
                        "devtools.wo13_trace_capture."
                        "load_frozen_baseline_authority",
                        return_value={
                            "manifest_sha256": "5" * 64,
                            "predictions_sha256": "b" * 64,
                            "source_revision_sha": "e" * 40,
                            "expected_record_count": 1,
                        },
                    )
                )
                stack.enter_context(
                    patch(
                        "devtools.wo13_trace_capture."
                        "authoritative_git_binding",
                        return_value={
                            "runtime_graph_sha256": "8" * 64,
                            "trace_tool_sha256": "9" * 64,
                            "container_graph_sha256": "c" * 64,
                        },
                    )
                )
                stack.enter_context(
                    patch(
                        "devtools.wo13_trace_capture.runtime_identity",
                        return_value=identity,
                    )
                )
                stack.enter_context(
                    patch(
                        "devtools.wo13_trace_capture."
                        "source_snapshot_sha256",
                        return_value="d" * 64,
                    )
                )
                with self.assertRaisesRegex(
                    TraceCaptureError,
                    "valid canonical ZIP",
                ):
                    load_capture_authority(authority_path)

    def test_dataset_archive_is_exactly_bound_to_input_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            members = [
                ("MIB-000001.pdf", b"%PDF-one"),
                ("MIB-000002.pdf", b"%PDF-two"),
            ]
            for name, content in members:
                (input_dir / name).write_bytes(content)
            expected_tree = compute_input_tree_sha256(input_dir)
            valid = root / "valid.zip"
            write_dataset_zip(valid, members)
            binding = verify_dataset_archive_authority(
                valid,
                expected_record_count=2,
                expected_input_tree_sha256=expected_tree,
            )
            self.assertEqual(binding["record_count"], 2)
            self.assertEqual(
                binding["input_tree_sha256"],
                expected_tree,
            )
            deflated = root / "deflated.zip"
            with zipfile.ZipFile(
                deflated,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                for name, content in members:
                    archive.writestr(name, content)
            self.assertEqual(
                verify_dataset_archive_authority(
                    deflated,
                    expected_record_count=2,
                    expected_input_tree_sha256=expected_tree,
                )["input_tree_sha256"],
                expected_tree,
            )

            arbitrary = root / "arbitrary.zip"
            arbitrary.write_bytes(b"not a zip archive")
            with self.assertRaisesRegex(
                TraceCaptureError,
                "valid canonical ZIP",
            ):
                verify_dataset_archive_authority(
                    arbitrary,
                    expected_record_count=2,
                    expected_input_tree_sha256=expected_tree,
                )

            wrong_byte = root / "wrong-byte.zip"
            write_dataset_zip(
                wrong_byte,
                [
                    ("MIB-000001.pdf", b"%PDF-engineered"),
                    members[1],
                ],
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "differs from input authority",
            ):
                verify_dataset_archive_authority(
                    wrong_byte,
                    expected_record_count=2,
                    expected_input_tree_sha256=expected_tree,
                )

            wrong_name = root / "wrong-name.zip"
            write_dataset_zip(
                wrong_name,
                [
                    ("MIB-999999.pdf", members[0][1]),
                    members[1],
                ],
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "differs from input authority",
            ):
                verify_dataset_archive_authority(
                    wrong_name,
                    expected_record_count=2,
                    expected_input_tree_sha256=expected_tree,
                )

    def test_dataset_archive_rejects_duplicate_traversal_and_extras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "MIB-000001.pdf").write_bytes(b"%PDF-one")
            (input_dir / "MIB-000002.pdf").write_bytes(b"%PDF-two")
            expected_tree = compute_input_tree_sha256(input_dir)

            duplicate = root / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                write_dataset_zip(
                    duplicate,
                    [
                        ("MIB-000001.pdf", b"%PDF-one"),
                        ("MIB-000001.pdf", b"%PDF-two"),
                    ],
                )
            traversal = root / "traversal.zip"
            write_dataset_zip(
                traversal,
                [
                    ("../MIB-000001.pdf", b"%PDF-one"),
                    ("MIB-000002.pdf", b"%PDF-two"),
                ],
            )
            extra = root / "extra.zip"
            write_dataset_zip(
                extra,
                [
                    ("MIB-000001.pdf", b"%PDF-one"),
                    ("MIB-000002.pdf", b"%PDF-two"),
                    ("MIB-000003.pdf", b"%PDF-extra"),
                ],
            )
            cases = (
                (duplicate, "duplicate members"),
                (traversal, "root MIB PDFs"),
                (extra, "member count"),
            )
            for archive_path, message in cases:
                with self.subTest(archive=archive_path.name):
                    with self.assertRaisesRegex(
                        TraceCaptureError,
                        message,
                    ):
                        verify_dataset_archive_authority(
                            archive_path,
                            expected_record_count=2,
                            expected_input_tree_sha256=expected_tree,
                        )

    def test_engineered_valid_archive_cannot_replace_approved_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            original = [
                ("MIB-000001.pdf", b"%PDF-visible-one"),
                ("MIB-000002.pdf", b"%PDF-visible-two"),
            ]
            for name, content in original:
                (input_dir / name).write_bytes(content)
            expected_tree = compute_input_tree_sha256(input_dir)
            engineered = root / "engineered.zip"
            write_dataset_zip(
                engineered,
                [
                    ("MIB-000001.pdf", b"%PDF-label-engineered-one"),
                    ("MIB-000002.pdf", b"%PDF-label-engineered-two"),
                ],
            )
            with self.assertRaisesRegex(
                TraceCaptureError,
                "differs from input authority",
            ):
                verify_dataset_archive_authority(
                    engineered,
                    expected_record_count=2,
                    expected_input_tree_sha256=expected_tree,
                )

    def test_private_snapshot_is_immune_to_transient_origin_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            origin_pdf = input_dir / "MIB-000001.pdf"
            origin_pdf.write_bytes(b"%PDF-original")
            source_file = root / "solution.py"
            source_file.write_text("ORIGINAL = True\n", encoding="utf-8")
            layout = root / "layout.json"
            layout.write_text("{}", encoding="utf-8")
            archive_path = root / "dataset.zip"
            write_dataset_zip(
                archive_path,
                [("MIB-000001.pdf", origin_pdf.read_bytes())],
            )
            input_hash = compute_input_tree_sha256(input_dir)
            with patch(
                "devtools.wo13_trace_capture._capture_source_paths",
                return_value=(Path("solution.py"),),
            ):
                source_hash = source_snapshot_sha256(root)
                authority = CaptureAuthority(
                    manifest_path=root / "authority.json",
                    manifest_sha256="a" * 64,
                    payload={
                        "source_revision_sha": "b" * 40,
                        "expected_record_count": 1,
                        "retry_missing_attempts": 1,
                    },
                    approved_paths={
                        "input_dir": input_dir,
                        "layout_manifest": layout,
                        "dataset_archive": archive_path,
                    },
                    expected_hashes={
                        "input_tree_sha256": input_hash,
                        "layout_manifest_sha256": sha256_path(layout),
                        "dataset_archive_sha256": sha256_path(
                            archive_path
                        ),
                        "source_snapshot_sha256": source_hash,
                    },
                    runtime_identity={"max_workers": 4},
                )
                with patch(
                    "devtools.wo13_trace_capture."
                    "_verify_layout_and_input",
                    return_value=input_hash,
                ):
                    with private_capture_snapshot(
                        authority,
                        repository_root=root,
                    ) as snapshot:
                        origin_pdf.write_bytes(b"%PDF-transient")
                        source_file.write_text(
                            "ORIGINAL = False\n",
                            encoding="utf-8",
                        )
                        self.assertEqual(
                            (
                                snapshot.input_dir
                                / "MIB-000001.pdf"
                            ).read_bytes(),
                            b"%PDF-original",
                        )
                        self.assertEqual(
                            (
                                snapshot.source_root / "solution.py"
                            ).read_text(encoding="utf-8"),
                            "ORIGINAL = True\n",
                        )
                        origin_pdf.write_bytes(b"%PDF-original")
                        source_file.write_text(
                            "ORIGINAL = True\n",
                            encoding="utf-8",
                        )

    def test_failed_rapid_recovery_rolls_back_provisional_authority(self):
        class FailingRapid:
            def _recover(self, *, primary_row):
                self._authoritative_rapid_decision()
                self._repair_source_priority_fields()
                raise RuntimeError("recover failed")

            def _authoritative_rapid_decision(self):
                return "DENIED"

            def _semantic_denial_rules(self):
                return False

            def _repair_biometric_applicant(self):
                return None, False

            def _repair_source_priority_fields(self):
                return None, True

            def _apply_review_approval_heads(self, *, final_row):
                return final_row

            def process_case(self, pdf_path):
                row = prediction(pdf_path.stem)
                signals = current_trace_signals()
                signals.linked_cases.append(
                    SimpleNamespace(
                        active_applicant="Applicant",
                        unresolved=False,
                        unresolved_reasons=(),
                    )
                )
                signals.resolved_cases.append(
                    SimpleNamespace(contested_fields=())
                )
                signals.primary_outcome = SimpleNamespace(
                    trace=SimpleNamespace(
                        authoritative_source=False,
                        denial_reasons=(),
                        review_reasons=(),
                        approval_facts=(),
                    )
                )
                try:
                    self._recover(primary_row=row)
                except RuntimeError:
                    pass
                return row

        collector = TraceCollector()
        observed = TracingCaseProcessor(
            _instrument_rapid_processor(FailingRapid()),
            collector,
        )
        observed.process_case(Path("MIB-000001.pdf"))
        self.assertEqual(
            collector.rows()[0]["policy_trace"],
            "deterministic_policy",
        )
        self.assertEqual(
            collector.rows()[0]["provenance_route"],
            "no_accepted_provenance",
        )

    def test_classifiers_cover_recovery_conflict_and_authority_routes(self):
        authority = TraceSignals(
            primary_candidates=(object(),),
            primary_outcome=SimpleNamespace(
                trace=SimpleNamespace(
                    authoritative_source=True,
                    denial_reasons=(),
                    review_reasons=(),
                    approval_facts=(),
                )
            ),
        )
        authority.linked_cases.append(
            SimpleNamespace(
                active_applicant="A",
                unresolved=False,
                unresolved_reasons=(),
            )
        )
        self.assertEqual(classify_provenance(authority), "authoritative_source")
        self.assertEqual(classify_linking(authority), "authoritative_scope")
        self.assertEqual(
            classify_policy(authority, prediction("MIB-000001")),
            "binding_authority",
        )

        recovered = TraceSignals(
            primary_candidates=(object(),),
            rapid_candidates=(object(),),
            ocr_paths={"orientation_retry", "targeted_rapidocr"},
            semantic_denial_applied=True,
            rapid_output_changed=True,
            primary_outcome=SimpleNamespace(
                trace=SimpleNamespace(
                    authoritative_source=False,
                    denial_reasons=(),
                    review_reasons=("review_flag:identity_conflict",),
                    approval_facts=(),
                )
            ),
        )
        recovered.linked_cases.append(
            SimpleNamespace(
                active_applicant="A",
                unresolved=True,
                unresolved_reasons=("identity conflict",),
            )
        )
        recovered.resolved_cases.append(
            SimpleNamespace(
                contested_fields=("applicant_name", "fee_status"),
                fields={
                    "fee_status": SimpleNamespace(
                        winning_evidence=object()
                    )
                },
            )
        )
        self.assertEqual(
            classify_provenance(recovered),
            "mixed_visible_sources",
        )
        self.assertEqual(classify_linking(recovered), "linked_ambiguous")
        self.assertEqual(classify_conflict(recovered), "multiple_conflicts")
        self.assertEqual(
            classify_ocr_path(recovered),
            "multiple_recovery_paths",
        )
        self.assertEqual(
            classify_policy(recovered, prediction("MIB-000001")),
            "recovery_denial",
        )

    def test_cli_has_no_label_or_score_input(self):
        parser = _parser()
        subcommands = next(
            action.choices
            for action in parser._actions
            if getattr(action, "choices", None)
        )
        help_text = "\n".join(
            [
                parser.format_help(),
                *(child.format_help() for child in subcommands.values()),
            ]
        )
        self.assertNotIn("--truth", help_text)
        self.assertNotIn("--case-scores", help_text)
        self.assertNotIn("--evaluation", help_text)
        self.assertNotIn("--source-revision-sha", help_text)
        self.assertNotIn("--input-tree-sha256", help_text)
        self.assertNotIn("--dataset-archive-sha256", help_text)
        self.assertNotIn("--runtime-graph-sha256", help_text)
        self.assertNotIn("--trace-tool-sha256", help_text)
        self.assertIn("--runtime-contract", help_text)
        self.assertIn("--baseline-predictions", help_text)
        self.assertIn("--authority-manifest", help_text)


if __name__ == "__main__":
    unittest.main()
