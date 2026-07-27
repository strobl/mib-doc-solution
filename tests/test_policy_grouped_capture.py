from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import devtools.policy_grouped_capture as capture_module
from devtools.policy_grouped_capture import (
    _ARCHIVE_DIGEST_ENV,
    _ARCHIVE_CHILD_FLAG,
    _ARCHIVE_CONTRACT_FD_ENV,
    _ARCHIVE_REVISION_ENV,
    _ARCHIVE_ROOT_ENV,
    _CONTRACT_CHECK_NAMES,
    _ExactMatcherObserver,
    _OBSERVATION_ROOT_KEYS,
    _SANDBOX_BACKEND,
    _SANDBOX_PATH,
    _bootstrap_archive_child,
    _early_tree_sha256,
    main,
    _parser,
    _sandbox_profile,
    _sandbox_runtime_environment,
    _write_output_set,
    matcher_contract_checks,
    verify_clean_source_revision,
    PolicyGroupedCaptureError,
)


class PolicyGroupedCaptureTests(unittest.TestCase):
    def test_exact_contract_probes_cover_all_required_vetoes(self):
        checks = matcher_contract_checks()
        self.assertEqual(set(checks), set(_CONTRACT_CHECK_NAMES))
        self.assertFalse(checks["authoritative_review_veto"])
        self.assertTrue(
            all(
                checks[name]
                for name in checks
                if name != "authoritative_review_veto"
            )
        )
        self.assertIn("deterministic", _OBSERVATION_ROOT_KEYS)
        self.assertIn("max_worker_count", _OBSERVATION_ROOT_KEYS)
        self.assertNotIn(
            "truth",
            {action.dest for action in _parser()._actions},
        )

    def test_exact_matcher_observer_separates_guarded_unguarded_and_late(self):
        review = SimpleNamespace(
            row=SimpleNamespace(adjudication="NEEDS_REVIEW"),
            trace=SimpleNamespace(decision="NEEDS_REVIEW"),
        )
        approval = SimpleNamespace(
            row=SimpleNamespace(adjudication="APPROVED"),
            trace=SimpleNamespace(decision="APPROVED"),
        )
        observer = _ExactMatcherObserver()
        observer.observe(
            allow_approval_recovery=True,
            matching_rules=("guard",),
            baseline=review,
            result=approval,
        )
        observer.observe(
            allow_approval_recovery=True,
            matching_rules=(),
            baseline=review,
            result=approval,
        )
        observer.observe(
            allow_approval_recovery=False,
            matching_rules=("guard",),
            baseline=review,
            result=approval,
        )
        self.assertEqual(
            observer.snapshot(),
            {
                "eligible_guarded_initial_count": 1,
                "guarded_initial_approval_count": 1,
                "unguarded_initial_approval_count": 1,
                "late_revalidation_approval_count": 1,
            },
        )

    def test_create_once_output_set_rejects_existing_and_symlink_targets(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            root = Path(name).resolve()
            first = root / "first.json"
            second = root / "second.json"
            _write_output_set({first: b"one", second: b"two"})
            self.assertEqual(first.read_bytes(), b"one")
            with self.assertRaises(PolicyGroupedCaptureError):
                _write_output_set({first: b"changed"})

            target = root / "target"
            target.write_bytes(b"x")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(PolicyGroupedCaptureError):
                _write_output_set({link: b"unsafe"})

    def test_archive_binding_rejects_post_extraction_source_mutation(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            temporary = Path(name).resolve()
            origin = temporary / "origin"
            root = temporary / "source"
            origin.mkdir()
            root.mkdir()

            def git(*arguments: str, text: bool = True):
                return subprocess.run(
                    ("/usr/bin/git", *arguments),
                    cwd=origin,
                    check=True,
                    capture_output=True,
                    text=text,
                ).stdout

            git("init")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            tracked = origin / "mib_pipeline.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            git("add", "mib_pipeline.py")
            git("commit", "-m", "base")
            revision = git("rev-parse", "HEAD").strip()
            archive = git(
                "archive", "--format=tar", revision, text=False
            )
            source = root / "mib_pipeline.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            digest = _early_tree_sha256(root)
            contract = {
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "bootstrap_pid": os.getpid(),
                "denied_sensitive_paths": [str(tracked)],
                "origin_root": str(origin),
                "revision": revision,
                "sandbox_backend": _SANDBOX_BACKEND,
                "sandbox_canary_path": str(tracked),
                "sandbox_profile_sha256": "f" * 64,
                "schema": "mib-wo17-archive-child/v2",
                "source_root": str(root),
                "tree_sha256": digest,
            }
            synthetic_file = root / "devtools" / "policy_grouped_capture.py"
            with mock.patch.object(
                capture_module, "_ARCHIVE_CONTRACT", contract
            ), mock.patch.object(
                capture_module, "__file__", str(synthetic_file)
            ), mock.patch.object(
                capture_module, "_verify_enforced_sandbox"
            ):
                verify_clean_source_revision(revision, repo_root=root)
                source.write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    PolicyGroupedCaptureError, "archive source binding"
                ):
                    verify_clean_source_revision(
                        revision, repo_root=root
                    )

    def test_caller_supplied_legacy_archive_environment_is_rejected(self):
        environment = {
            _ARCHIVE_ROOT_ENV: "/private/tmp/forged-source",
            _ARCHIVE_DIGEST_ENV: "a" * 64,
            _ARCHIVE_REVISION_ENV: "b" * 40,
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            self.assertEqual(_bootstrap_archive_child(()), 1)

    def test_forged_child_contract_cannot_create_evidence_outside_sandbox(self):
        """A reproducible pipe/flag contract is insufficient without Seatbelt."""

        source_root = Path(__file__).resolve().parents[1]
        truth = (source_root / "data" / "train_labels.csv").resolve(strict=True)
        revision = subprocess.run(
            ("/usr/bin/git", "rev-parse", "HEAD"),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            root = Path(name).resolve()
            input_root = root / "inputs"
            input_root.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            outputs = tuple(root / f"output-{index}" for index in range(5))
            read_fd, write_fd = os.pipe()
            contract = {
                "archive_sha256": "a" * 64,
                "bootstrap_pid": os.getpid(),
                "denied_sensitive_paths": [str(truth)],
                "origin_root": str(root),
                "revision": revision,
                "sandbox_backend": _SANDBOX_BACKEND,
                "sandbox_canary_path": str(truth),
                "sandbox_profile_sha256": "b" * 64,
                "schema": "mib-wo17-archive-child/v2",
                "source_root": str(source_root),
                "tree_sha256": "c" * 64,
            }
            raw = (
                json.dumps(
                    contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            try:
                os.set_inheritable(read_fd, True)
                os.write(write_fd, raw)
                os.close(write_fd)
                write_fd = -1
                environment = {
                    "LC_ALL": "C",
                    "MIB_MAX_WORKERS": "4",
                    "MKL_NUM_THREADS": "4",
                    "NUMEXPR_NUM_THREADS": "4",
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "PATH": _SANDBOX_PATH,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    _ARCHIVE_CONTRACT_FD_ENV: str(read_fd),
                }
                completed = subprocess.run(
                    (
                        sys.executable,
                        "-I",
                        "-B",
                        str(
                            source_root
                            / "devtools"
                            / "policy_grouped_capture.py"
                        ),
                        _ARCHIVE_CHILD_FLAG,
                        "--arm",
                        "candidate",
                        "--source-revision-sha",
                        revision,
                        "--input-dir",
                        str(input_root),
                        "--layout-manifest",
                        str(manifest),
                        "--expected-layout-manifest-sha256",
                        "d" * 64,
                        "--expected-input-tree-sha256",
                        "e" * 64,
                        "--denied-sensitive-path",
                        str(truth),
                        "--first-predictions",
                        str(outputs[0]),
                        "--second-predictions",
                        str(outputs[1]),
                        "--first-audit",
                        str(outputs[2]),
                        "--second-audit",
                        str(outputs[3]),
                        "--observation",
                        str(outputs[4]),
                    ),
                    cwd=source_root,
                    env=environment,
                    pass_fds=(read_fd,),
                    check=False,
                    capture_output=True,
                    text=True,
                )
            finally:
                os.close(read_fd)
                if write_fd >= 0:
                    os.close(write_fd)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "sandbox did not deny canonical truth/repository access",
                completed.stderr,
            )
            self.assertTrue(all(not path.exists() for path in outputs))

    def test_cli_threads_sensitive_paths_into_capture_contract_check(self):
        observation = {
            "status": "candidate",
            "record_count": 1000,
            "deterministic": True,
        }
        with mock.patch.object(
            capture_module,
            "run_grouped_capture",
            return_value=observation,
        ) as runner:
            result = main(
                (
                    "--arm",
                    "candidate",
                    "--source-revision-sha",
                    "a" * 40,
                    "--input-dir",
                    "/private/tmp/input",
                    "--layout-manifest",
                    "/private/tmp/manifest.json",
                    "--expected-layout-manifest-sha256",
                    "b" * 64,
                    "--expected-input-tree-sha256",
                    "c" * 64,
                    "--denied-sensitive-path",
                    "/private/tmp/truth.csv",
                    "--first-predictions",
                    "/private/tmp/prediction-1.jsonl",
                    "--second-predictions",
                    "/private/tmp/prediction-2.jsonl",
                    "--first-audit",
                    "/private/tmp/audit-1.json",
                    "--second-audit",
                    "/private/tmp/audit-2.json",
                    "--observation",
                    "/private/tmp/observation.json",
                )
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            runner.call_args.kwargs["denied_sensitive_paths"],
            [Path("/private/tmp/truth.csv")],
        )

    def test_sandbox_profile_denies_each_canonical_sensitive_path(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            root = Path(name).resolve()
            origin = root / "origin"
            source = root / "source"
            inputs = root / "inputs"
            for path in (origin, source, inputs):
                path.mkdir()
            truth = root / "truth.csv"
            second = root / "second-secret.csv"
            truth.write_text("truth\n", encoding="utf-8")
            second.write_text("secret\n", encoding="utf-8")
            profile = _sandbox_profile(
                origin_root=origin,
                source_root=source,
                input_root=inputs,
                denied_sensitive_paths=(truth, second),
            )
        self.assertIn(f'(literal "{truth}")', profile)
        self.assertIn(f'(literal "{second}")', profile)
        self.assertIn("(deny network*)", profile)

    def test_sandbox_environment_has_exact_trusted_macos_tool_path(self):
        environment = _sandbox_runtime_environment()
        self.assertEqual(
            environment,
            {
                "LC_ALL": "C",
                "MIB_MAX_WORKERS": "4",
                "MKL_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "PATH": (
                    "/opt/homebrew/bin:/usr/local/bin:"
                    "/usr/bin:/bin:/usr/sbin:/sbin"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(environment["PATH"], _SANDBOX_PATH)

    def test_clean_source_rejects_assume_unchanged_index_flag(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            root = Path(name).resolve()

            def git(*arguments: str) -> str:
                completed = subprocess.run(
                    ("/usr/bin/git", *arguments),
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return completed.stdout.strip()

            git("init")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            tracked = root / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            git("add", "tracked.py")
            git("commit", "-m", "base")
            revision = git("rev-parse", "HEAD")
            verify_clean_source_revision(revision, repo_root=root)
            git("update-index", "--assume-unchanged", "tracked.py")
            with self.assertRaisesRegex(
                PolicyGroupedCaptureError, "index flags"
            ):
                verify_clean_source_revision(revision, repo_root=root)

    def test_archive_child_invocation_is_isolated_in_source(self):
        source = Path(
            __file__
        ).resolve().parents[1] / "devtools" / "policy_grouped_capture.py"
        text = source.read_text(encoding="utf-8")
        bootstrap = text.index("_bootstrap_archive_child(sys.argv[1:])")
        production_import = text.index(
            "from mib_pipeline import BatchRunner"
        )
        self.assertLess(bootstrap, production_import)
        self.assertIn('"-I"', text)
        self.assertIn('"(deny network*)"', text)
        self.assertIn('"PYTHONDONTWRITEBYTECODE": "1"', text)
        self.assertIn("_verify_enforced_sandbox(expected_root)", text)


if __name__ == "__main__":
    unittest.main()
