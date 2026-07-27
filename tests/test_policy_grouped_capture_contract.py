from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devtools.governed_result_cli import (
    GovernedResultCLIError,
    _GROUPED_TRUTH_ADJACENT_CONTROL_PATHS,
    _grouped_graph_sha256_exact,
    _validate_grouped_control_modules_unchanged,
)
from devtools.grouped_policy_revalidation_evidence import (
    _early_require_control_blobs_unchanged,
)
from devtools.policy_grouped_capture_contract import (
    FIELD_NAMES,
    validate_prediction_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = "devtools/policy_grouped_capture_contract.py"


def _canonical_row() -> tuple[dict[str, object], bytes]:
    row: dict[str, object] = {
        "case_id": "MIB-000001",
        "applicant_name": "Test Applicant",
        "species_code": "HUM",
        "home_world": "Earth",
        "visa_class": "A1",
        "sponsor_id": "SPN-0001",
        "arrival_date": "2026-01-01",
        "declared_purpose": "Testing",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "APPROVED",
        "confidence": 0.9,
    }
    raw = (
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return row, raw


class TruthSafeCaptureContractTests(unittest.TestCase):
    def test_evidence_has_no_candidate_runtime_import_edge(self):
        source = (
            REPO_ROOT
            / "devtools"
            / "grouped_policy_revalidation_evidence.py"
        ).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(
            any(name.startswith("mib_pipeline") for name in imported)
        )
        self.assertNotIn("devtools.policy_grouped_capture", imported)

    def test_truth_argument_cannot_reach_candidate_import_hook(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            truth = temporary / "truth.csv"
            marker = temporary / "exfiltrated"
            truth.write_bytes(b"identity-bearing-secret\n")
            program = """
import builtins
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
truth = pathlib.Path(sys.argv[2])
marker = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(repo))
sys.argv = [
    "grouped_policy_revalidation_evidence.py",
    "--truth",
    str(truth),
]
original = builtins.__import__

def guarded(name, *args, **kwargs):
    if (
        name == "devtools.policy_grouped_capture"
        or name.startswith("mib_pipeline")
    ):
        marker.write_bytes(truth.read_bytes())
    return original(name, *args, **kwargs)

builtins.__import__ = guarded
import devtools.grouped_policy_revalidation_evidence
"""
            completed = subprocess.run(
                (
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    program,
                    str(REPO_ROOT),
                    str(truth),
                    str(marker),
                ),
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr,
            )
            self.assertFalse(marker.exists())

    def test_prediction_validation_is_independent_of_candidate_model(self):
        row, raw = _canonical_row()
        with mock.patch(
            "mib_pipeline.models.PredictionRow.from_mapping",
            side_effect=AssertionError("candidate model must not execute"),
        ):
            result = validate_prediction_bytes(raw, expected_count=1)
        self.assertEqual(tuple(result[0]), FIELD_NAMES)
        self.assertEqual(result[0], row)

    def test_early_bootstrap_rejects_candidate_control_substitution(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            repository = Path(temporary_name)
            path = repository / CONTRACT_PATH
            path.parent.mkdir(parents=True)
            path.write_text("SAFE = True\n", encoding="utf-8")
            self._git(repository, "init", "-q")
            self._git(repository, "add", CONTRACT_PATH)
            self._git(repository, "commit", "-q", "-m", "control")
            control = self._git(repository, "rev-parse", "HEAD").strip()
            path.write_text(
                "from pathlib import Path\n"
                "Path('/tmp/exfil').write_text('truth')\n",
                encoding="utf-8",
            )
            self._git(repository, "add", CONTRACT_PATH)
            self._git(repository, "commit", "-q", "-m", "candidate")
            candidate = self._git(
                repository, "rev-parse", "HEAD"
            ).strip()
            with self.assertRaisesRegex(
                RuntimeError,
                "replaced truth-adjacent control module",
            ):
                _early_require_control_blobs_unchanged(
                    repository,
                    candidate_revision=candidate,
                    control_revision=control,
                    paths=(CONTRACT_PATH,),
                )

    def test_final_provenance_rejects_candidate_control_substitution(self):
        def blob(_root: Path, revision: str, path: str) -> bytes:
            if revision == "c" * 40 and path == CONTRACT_PATH:
                return b"malicious import-time code"
            return b"frozen-control"

        with mock.patch(
            "devtools.governed_result_cli._git_blob",
            side_effect=blob,
        ), self.assertRaisesRegex(
            GovernedResultCLIError,
            CONTRACT_PATH,
        ):
            _validate_grouped_control_modules_unchanged(
                REPO_ROOT,
                control_revision="p" * 40,
                candidate_revision="c" * 40,
            )

    def test_grouped_producer_graph_binds_truth_safe_contract(self):
        paths = {
            "Dockerfile",
            "requirements.lock",
            "run.sh",
            CONTRACT_PATH,
            "mib_pipeline/core.py",
        }

        def blob(_root: Path, revision: str, path: str) -> bytes:
            if path == CONTRACT_PATH:
                return f"contract-{revision}".encode("ascii")
            return path.encode("utf-8")

        with mock.patch(
            "devtools.governed_result_cli._revision_paths",
            return_value=tuple(paths),
        ), mock.patch(
            "devtools.governed_result_cli._git_blob",
            side_effect=blob,
        ):
            control = _grouped_graph_sha256_exact(REPO_ROOT, "p" * 40)
            candidate = _grouped_graph_sha256_exact(REPO_ROOT, "c" * 40)
        self.assertNotEqual(control, candidate)

    def test_all_truth_adjacent_controls_include_safe_contract(self):
        self.assertIn(
            CONTRACT_PATH,
            _GROUPED_TRUTH_ADJACENT_CONTROL_PATHS,
        )

    @staticmethod
    def _git(repository: Path, *arguments: str) -> str:
        completed = subprocess.run(
            (
                "/usr/bin/git",
                "-c",
                "user.name=WO17 Test",
                "-c",
                "user.email=wo17@example.invalid",
                *arguments,
            ),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout


if __name__ == "__main__":
    unittest.main()
