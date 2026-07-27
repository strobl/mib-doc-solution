from __future__ import annotations

import ast
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "wo17-grouped-revalidation.yml"
)
ACTION_SHA_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _step(text: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    start = text.index(marker)
    end = text.find("\n      - name: ", start + len(marker))
    return text[start:] if end < 0 else text[start:end]


def _python_heredoc(step: str) -> str:
    start = step.index("<<'PY'\n") + len("<<'PY'\n")
    end = step.index("\n          PY", start)
    return step[start:end]


def _run_script(step: str) -> str:
    start = step.index("        run: |\n") + len("        run: |\n")
    lines = step[start:].splitlines()
    return "\n".join(
        line[10:] if line.startswith(" " * 10) else line
        for line in lines
    )


def _evidence_preimport_control_paths() -> tuple[str, ...]:
    source = (
        ROOT / "devtools" / "grouped_policy_revalidation_evidence.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "_EVIDENCE_PREIMPORT_CONTROL_PATHS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return tuple(value)
    raise AssertionError("evidence pre-import control set is missing")


class WO17GroupedRevalidationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_duplicate_prone_job_and_output_keys_are_unique(self) -> None:
        control_start = self.text.index("  control:\n")
        candidate_start = self.text.index("  candidate:\n")
        control_header = self.text[
            control_start : self.text.index(
                "\n    steps:\n", control_start
            )
        ]
        candidate_header = self.text[
            candidate_start : self.text.index(
                "\n    steps:\n", candidate_start
            )
        ]
        for header in (control_header, candidate_header):
            self.assertEqual(header.count("\n    if:"), 1)
            self.assertEqual(header.count("\n    runs-on:"), 1)
            self.assertEqual(header.count("\n    timeout-minutes:"), 1)
        control_plan = _step(
            self.text, "Reconstruct and verify the preregistered plan"
        )
        candidate_plan = _step(
            self.text, "Reconstruct plan and prove full governed topology"
        )
        self.assertEqual(control_plan.count('"sha256="'), 1)
        self.assertEqual(candidate_plan.count('"sha256="'), 1)

    def test_workflow_has_exact_trigger_runner_and_owner_guards(self) -> None:
        self.assertIn(
            "name: WO17 grouped policy revalidation\n", self.text
        )
        self.assertIn("  push:\n", self.text)
        self.assertNotIn("workflow_dispatch:", self.text)
        self.assertNotIn("pull_request:", self.text)
        self.assertEqual(self.text.count("runs-on: macos-14"), 2)
        self.assertNotIn("runs-on: ubuntu", self.text)
        for guard in (
            "github.event_name == 'push'",
            "github.repository == 'strobl/mib-doc-solution'",
            "github.actor == github.repository_owner",
            "github.triggering_actor == github.repository_owner",
            "github.ref == 'refs/heads/improve/score-148'",
        ):
            self.assertEqual(self.text.count(guard), 2)
        self.assertIn("[wo17-control]", self.text)
        self.assertIn("[wo17-candidate]", self.text)
        self.assertEqual(
            self.text.count(
                "!contains(github.event.head_commit.message, "
                "'[wo17-candidate]')"
            ),
            1,
        )
        self.assertEqual(
            self.text.count(
                "!contains(github.event.head_commit.message, "
                "'[wo17-control]')"
            ),
            1,
        )

    def test_every_action_reference_is_an_immutable_expected_sha(self) -> None:
        uses = tuple(
            line.strip()[len("uses:") :].split("#", 1)[0].strip()
            for line in self.text.splitlines()
            if line.strip().startswith("uses:")
        )
        self.assertTrue(uses)
        self.assertTrue(all(ACTION_SHA_RE.fullmatch(value) for value in uses))
        self.assertEqual(
            set(uses),
            {
                (
                    "actions/checkout@"
                    "11bd71901bbe5b1630ceea73d27597364c9af683"
                ),
                (
                    "actions/cache/restore@"
                    "5a3ec84eff668545956fd18022155c47e93e2684"
                ),
                (
                    "actions/cache/save@"
                    "5a3ec84eff668545956fd18022155c47e93e2684"
                ),
                (
                    "actions/upload-artifact@"
                    "ea165f8d65b6e75b540449e92b4886f43607fa02"
                ),
            },
        )
        self.assertEqual(
            self.text.count("persist-credentials: false"), 2
        )
        self.assertEqual(self.text.count("fetch-depth: 0"), 2)

    def test_public_population_bindings_are_exact(self) -> None:
        expected = (
            "a9bb8c1bbf51346ebf49c2e3e1acdb7a5d6cd0760162767b0d133c7b7200f3c4",
            "21e821aa3089b841683375da59cf961e679e10f7009e5332ea9e8582f00f4c8e",
            "9c6210df4a600c9520435cf7d79d61d7113795dbf94b0e7ab3e39d237388bc8a",
            "d7aac395c2d42dc42128ba3b4ce15fef6c42c37a6e247a066c267fba8a514b7c",
            "20515df0d93d6ac73f1f989dd230e4bb1ff6e295e118302114ddd5b4b753c2cf",
            "8fb622cfc9046a1cde3b2fcf499959a16a9f97e3dfffcdbf75137e5cdb6429b8",
            "1c5d199d0919337b57f03ae817cd77cc9bdb9a94c36738017f887a4361e08d71",
            "f92cba7c466118bd913e7865fc13d7e1e24ac701a503e5798caebd9547486f55",
        )
        for digest in expected:
            self.assertIn(digest, self.text)
        self.assertIn(
            "mib-doc-challenge-public-data-v2026-07-07.zip?download=true",
            self.text,
        )
        self.assertEqual(
            self.text.count("--split-seed wo12-full-public-layout-v2-20260727"),
            1,
        )
        self.assertIn("len(pdfs) != 1000", self.text)

    def test_topology_and_preregistered_scope_are_proven(self) -> None:
        self.assertIn("Bind direct A to P topology", self.text)
        self.assertIn("Bind exact A to P to C topology and markers", self.text)
        self.assertIn(
            "verify_governed_experiment_topology(", self.text
        )
        for path in (
            '"mib_pipeline/decision_recovery.py"',
            '"tests/test_decision_recovery.py"',
        ):
            self.assertEqual(self.text.count(path), 2)
        self.assertIn(
            '"evaluation/program/experiment_ledger.jsonl"', self.text
        )
        self.assertIn(
            '"evaluation/program/current_checkpoint.json"', self.text
        )
        self.assertIn(
            "P is not the exact governance-only transition", self.text
        )

    def test_candidate_preflight_precedes_data_and_mirrors_exact_set(
        self,
    ) -> None:
        candidate = self.text[self.text.index("  candidate:\n") :]
        preflight_name = "Prove immutable truth-adjacent control blobs"
        preflight_offset = candidate.index(
            f"      - name: {preflight_name}\n"
        )
        for later_step in (
            "Create isolated pinned Python runtime",
            "Reconstruct plan and prove full governed topology",
            "Restore immutable public dataset cache",
            "Download and verify immutable public dataset",
            "Extract exact public evaluation population",
            "Capture truth-blind candidate C twice",
            "Build and gate exact aggregate-only evidence",
        ):
            self.assertLess(
                preflight_offset,
                candidate.index(f"      - name: {later_step}\n"),
            )
        self.assertLess(
            preflight_offset,
            candidate.index("train_labels.csv"),
        )

        preflight = _step(self.text, preflight_name)
        match = re.search(
            r"(?ms)^\s*control_paths=\(\n(.*?)^\s*\)\n",
            preflight,
        )
        self.assertIsNotNone(match)
        workflow_paths = tuple(
            re.findall(r'(?m)^\s*"([^"]+)"\s*$', match.group(1))
        )
        evidence_paths = _evidence_preimport_control_paths()
        self.assertEqual(workflow_paths, evidence_paths)
        self.assertEqual(len(workflow_paths), len(set(workflow_paths)))
        self.assertIn(
            "devtools/policy_grouped_capture_contract.py",
            workflow_paths,
        )
        self.assertIn(
            'test "$control" = "$CONTROL_SHA"',
            preflight,
        )
        self.assertIn(
            'test "$candidate_entry" = "$control_entry"',
            preflight,
        )
        self.assertIn("/usr/bin/cmp -s", preflight)
        self.assertIn("GIT_NO_REPLACE_OBJECTS=1", preflight)
        self.assertIn("/usr/bin/git --no-replace-objects", preflight)

    def test_candidate_preflight_rejects_malicious_control_substitution(
        self,
    ) -> None:
        preflight = _step(
            self.text, "Prove immutable truth-adjacent control blobs"
        )
        script = _run_script(preflight)
        paths = _evidence_preimport_control_paths()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*arguments: str) -> str:
                completed = subprocess.run(
                    ("/usr/bin/git", *arguments),
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return completed.stdout.strip()

            git("init", "-q")
            git("config", "user.name", "WO17 Workflow Test")
            git("config", "user.email", "wo17@example.invalid")
            for path_text in paths:
                path = root / path_text
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"immutable control fixture: {path_text}\n",
                    encoding="utf-8",
                )
            git("add", "--", *paths)
            git("commit", "-q", "-m", "control")
            control = git("rev-parse", "HEAD")

            allowed = root / "mib_pipeline" / "decision_recovery.py"
            allowed.parent.mkdir(parents=True, exist_ok=True)
            allowed.write_text("candidate variable\n", encoding="utf-8")
            git("add", "--", str(allowed.relative_to(root)))
            git("commit", "-q", "-m", "candidate")
            candidate = git("rev-parse", "HEAD")

            environment = os.environ.copy()
            environment.update(
                {
                    "CANDIDATE_SHA": candidate,
                    "CONTROL_SHA": control,
                    "GITHUB_WORKSPACE": str(root),
                }
            )
            accepted = subprocess.run(
                (
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-euo",
                    "pipefail",
                    "-c",
                    script,
                ),
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

            git("checkout", "-q", "--detach", control)
            attacked = (
                root / "devtools" / "policy_grouped_capture_contract.py"
            )
            attacked.write_text(
                "malicious truth-adjacent substitution\n",
                encoding="utf-8",
            )
            git("add", "--", str(attacked.relative_to(root)))
            git("commit", "-q", "-m", "malicious candidate")
            malicious = git("rev-parse", "HEAD")
            environment["CANDIDATE_SHA"] = malicious
            rejected = subprocess.run(
                (
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-euo",
                    "pipefail",
                    "-c",
                    script,
                ),
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_both_captures_are_truth_blind_and_os_sandboxed(self) -> None:
        control = _step(self.text, "Capture truth-blind control P twice")
        candidate = _step(
            self.text, "Capture truth-blind candidate C twice"
        )
        for block, arm in ((control, "baseline"), (candidate, "candidate")):
            self.assertIn("devtools/policy_grouped_capture.py", block)
            self.assertIn(f"--arm {arm}", block)
            self.assertIn("--denied-sensitive-path", block)
            self.assertIn("--max-workers 4", block)
            self.assertNotRegex(block, r"(?m)^\s*--truth(?:\s|$)")
        self.assertEqual(self.text.count("test -x /usr/bin/sandbox-exec"), 2)
        self.assertEqual(self.text.count("MIB_MAX_WORKERS: \"4\""), 1)
        self.assertEqual(self.text.count("--max-workers 4"), 2)
        archive_delete = 'rm "$RUNNER_TEMP/$DATA_ARCHIVE"'
        archive_absent = 'test ! -e "$RUNNER_TEMP/$DATA_ARCHIVE"'
        self.assertEqual(self.text.count(archive_delete), 2)
        self.assertEqual(self.text.count(archive_absent), 2)
        for capture_name in (
            "Capture truth-blind control P twice",
            "Capture truth-blind candidate C twice",
        ):
            capture_offset = self.text.index(
                f"      - name: {capture_name}\n"
            )
            self.assertGreater(
                self.text.rfind(archive_delete, 0, capture_offset),
                0,
            )
            self.assertGreater(
                self.text.rfind(archive_absent, 0, capture_offset),
                0,
            )

        builder = _step(
            self.text, "Build and gate exact aggregate-only evidence"
        )
        self.assertIn(
            "devtools/grouped_policy_revalidation_evidence.py", builder
        )
        self.assertRegex(builder, r"(?m)^\s*--truth\s")
        self.assertIn("--control-revision-sha", builder)
        self.assertIn("--candidate-revision-sha", builder)
        self.assertIn("set +e", builder)
        self.assertIn("builder_status=$?", builder)
        self.assertRegex(builder, r"(?m)^\s*0\|2\)")
        self.assertIn("exit \"$builder_status\"", builder)
        self.assertIn('test -f "$json_output"', builder)
        self.assertIn('test -f "$markdown_output"', builder)

    def test_control_artifact_is_exact_short_lived_and_attempt_bound(
        self,
    ) -> None:
        stage = _step(
            self.text, "Stage exact short-retention control artifact"
        )
        expected_members = {
            "control-audit-1.json",
            "control-audit-2.json",
            "control-observation.json",
            "control-prediction-1.jsonl",
            "control-prediction-2.jsonl",
            "environment-fingerprint.json",
            "hypothesis.txt",
            "layout-manifest.json",
            "primary-variable.txt",
            "runtime-contract.json",
        }
        for member in expected_members:
            self.assertIn(f'"{member}"', stage)
        upload = _step(
            self.text, "Upload run-attempt-bound control P artifact"
        )
        self.assertIn(
            (
                "name: wo17-control-"
                "${{ steps.topology.outputs.control }}-"
                "${{ github.run_attempt }}"
            ),
            upload,
        )
        self.assertIn("retention-days: 1", upload)
        self.assertIn("if-no-files-found: error", upload)

    def test_control_and_candidate_runtime_fingerprints_are_identical(
        self,
    ) -> None:
        control = _step(
            self.text, "Fingerprint the exact control runtime"
        )
        candidate = _step(
            self.text, "Fingerprint the exact candidate runtime"
        )
        self.assertEqual(
            _python_heredoc(control), _python_heredoc(candidate)
        )
        for required in (
            "/usr/bin/sw_vers\", \"-productVersion",
            "/usr/bin/sw_vers\", \"-buildVersion",
            "platform.machine()",
            "sys.version",
            "importlib.metadata.distributions()",
            '"content_sha256"',
            '"executable_sha256"',
            'shutil.which("tesseract", path=sandbox_path)',
            '"mib-wo17-environment-fingerprint/v1"',
            '"/opt/homebrew/bin:/usr/local/bin:"',
            '"/usr/bin:/bin:/usr/sbin:/sbin"',
        ):
            self.assertIn(required, control)
        verify = _step(
            self.text, "Verify downloaded plan-bound public inputs"
        )
        self.assertIn("cmp -s", verify)
        self.assertEqual(
            verify.count("environment-fingerprint.json"), 2
        )

    def test_candidate_authenticates_unique_current_attempt_artifact(
        self,
    ) -> None:
        download = _step(
            self.text,
            "Discover and download only the authenticated P artifact",
        )
        for binding in (
            '"actor"',
            '"conclusion"',
            '"event"',
            '"head_branch"',
            '"head_commit_message"',
            '"head_repository"',
            '"head_sha"',
            '"path"',
            '"repository"',
            '"run_attempt"',
            '"triggering_actor"',
            '"workflow_id"',
            '"workflow_url"',
        ):
            self.assertIn(binding, download)
        self.assertIn(
            "wo17-grouped-revalidation.yml", download
        )
        self.assertIn(
            'workflow_path + "@improve/score-148"', download
        )
        self.assertIn('run_list.get("total_count") != 1', download)
        self.assertIn("len(matches) != 1", download)
        self.assertIn(
            "f\"wo17-control-{control}-{bound_run['run_attempt']}\"",
            download,
        )
        self.assertIn(
            'or "[wo17-candidate]"\n'
            '              in str(bound_run["head_commit_message"])',
            download,
        )
        self.assertIn(r"sha256:[0-9a-f]{64}", download)
        self.assertIn("size != bound_artifact", download)
        self.assertEqual(
            download.count("actions/artifacts/{artifact_id}"), 2
        )
        self.assertGreaterEqual(
            download.count("actions/runs/{run_id}"), 2
        )
        for zip_guard in (
            "len(names) != len(set(names))",
            "set(names) != expected_files",
            "info.flag_bits & 1",
            "stat.S_ISLNK(mode)",
            '"/" in info.filename',
            '"\\\\" in info.filename',
            "os.O_EXCL",
            'getattr(os, "O_NOFOLLOW", 0)',
        ):
            self.assertIn(zip_guard, download)

    def test_final_artifact_has_only_root_json_and_attempt_bound_name(
        self,
    ) -> None:
        builder = _step(
            self.text, "Build and gate exact aggregate-only evidence"
        )
        self.assertIn("len(entries) != 1", builder)
        self.assertIn(
            'entries[0].name != "wo17-grouped-aggregate.json"',
            builder,
        )
        self.assertIn('rm "$json_output" "$markdown_output"', builder)
        upload = _step(
            self.text, "Upload sole run-attempt-bound aggregate"
        )
        self.assertIn(
            (
                "name: wo17-grouped-aggregate-"
                "${{ steps.topology.outputs.candidate }}-"
                "${{ github.run_attempt }}"
            ),
            upload,
        )
        self.assertIn(
            (
                "path: ${{ runner.temp }}/wo17-final/"
                "wo17-grouped-aggregate.json"
            ),
            upload,
        )
        self.assertNotIn("*.json", upload)
        self.assertNotIn(".md", upload)

    def test_builder_status_switch_accepts_pass_and_block_only(self) -> None:
        builder = _step(
            self.text, "Build and gate exact aggregate-only evidence"
        )
        match = re.search(
            r'(?ms)^\s*case "\$builder_status" in\n.*?^\s*esac$',
            builder,
        )
        self.assertIsNotNone(match)
        case_statement = "\n".join(
            line[10:] if line.startswith(" " * 10) else line
            for line in match.group(0).splitlines()
        )
        for status in (0, 2):
            completed = subprocess.run(
                (
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-euo",
                    "pipefail",
                    "-c",
                    f"builder_status={status}\n{case_statement}\n",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        for status in (1, 3, 127):
            completed = subprocess.run(
                (
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-euo",
                    "pipefail",
                    "-c",
                    f"builder_status={status}\n{case_statement}\n",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, status)


if __name__ == "__main__":
    unittest.main()
