from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from devtools.experiment_control import (
    CanonicalHashChainStore,
    ExperimentLedger,
    IntegrityError,
    canonical_json,
)
from devtools.governed_experiment_cli import (
    CHECKPOINT_DIRECTORY_RELATIVE_PATH,
    CUTOVER_COMMIT_ACTOR,
    CUTOVER_COMMIT_MESSAGE,
    CUTOVER_AUTHORITY_RELATIVE_PATH,
    POINTER_RELATIVE_PATH,
    GovernedExperimentCLIError,
    GitHubCheckpointAuthority,
    _atomic_replace_exact,
    _canonical_bytes,
    _gh_json,
    _git_index_path,
    _program_paths,
    _repository_state,
    create_cutover,
    main,
    publish_prepared_cutover,
    preregister,
    verify_authority,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def write_canonical(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


class FakeGitHub:
    def __init__(self, head: str) -> None:
        self.head = head
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, object]] = []

    def publish_local_governance(self, repository_root: Path) -> None:
        pointer_path = repository_root.joinpath(
            *POINTER_RELATIVE_PATH.parts
        )
        pointer = json.loads(pointer_path.read_bytes())
        self.files[POINTER_RELATIVE_PATH.as_posix()] = (
            pointer_path.read_bytes()
        )
        checkpoint = repository_root / pointer["checkpoint_path"]
        self.files[pointer["checkpoint_path"]] = checkpoint.read_bytes()
        authority = repository_root.joinpath(
            *CUTOVER_AUTHORITY_RELATIVE_PATH.parts
        )
        self.files[CUTOVER_AUTHORITY_RELATIVE_PATH.as_posix()] = (
            authority.read_bytes()
        )

    def __call__(self, _root, endpoint, *, fields=None):
        self.calls.append((endpoint, fields))
        if "/branches/" in endpoint:
            branch = endpoint.split("/branches/", 1)[1]
            return {
                "commit": {"sha": self.head},
                "name": branch.replace("%2F", "/"),
            }
        marker = "/contents/"
        if marker not in endpoint:
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        path = endpoint.split(marker, 1)[1]
        if path not in self.files:
            raise GovernedExperimentCLIError(
                f"remote file is absent: {path}"
            )
        raw = self.files[path]
        encoded = base64.b64encode(raw).decode("ascii")
        encoded = "\n".join(
            encoded[index : index + 60]
            for index in range(0, len(encoded), 60)
        )
        return {
            "content": encoded,
            "encoding": "base64",
            "path": path,
            "type": "file",
        }


class FakeGitDatabase:
    def __init__(self, repository: Path, head: str, branch: str) -> None:
        self.repository = repository
        self.head = head
        self.branch = branch
        self.calls: list[dict[str, object]] = []
        self.messages: dict[str, str] = {}
        self.repository_id = "R_fake_governed_repository"
        self.blob_posts = 0
        self.fail_blob_post: int | None = None
        self.fail_before_patch = False
        self.race_on_patch = False
        self.race_to_ancestor_on_patch = False
        self.lose_patch_and_ref_response = False
        self.lose_patch_response = False
        self.ref_read_failures = 0
        self.tamper_contents_path: str | None = None

    def git_bytes(
        self,
        *arguments: str,
        input_bytes: bytes | None = None,
        environment: dict[str, str] | None = None,
    ) -> bytes:
        env = os.environ.copy()
        env.update(environment or {})
        completed = subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=self.repository,
            check=True,
            capture_output=True,
            input=input_bytes,
            env=env,
        )
        return completed.stdout

    def git(self, *arguments: str, **kwargs) -> str:
        return self.git_bytes(*arguments, **kwargs).decode().strip()

    def commit_response(self, revision: str):
        lineage = self.git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            revision,
        ).split()
        tree = self.git("rev-parse", f"{revision}^{{tree}}")
        return {
            "message": self.messages.get(revision, "existing commit"),
            "parents": [{"sha": parent} for parent in lineage[1:]],
            "sha": revision,
            "tree": {"sha": tree},
        }

    def tree_response(self, tree_sha: str):
        raw = self.git_bytes("ls-tree", "-r", "-t", "-z", tree_sha)
        entries = []
        for record in raw.split(b"\0"):
            if not record:
                continue
            metadata, path = record.split(b"\t", 1)
            mode, kind, sha = metadata.decode().split()
            entries.append(
                {
                    "mode": mode,
                    "path": path.decode(),
                    "sha": sha,
                    "type": kind,
                }
            )
        return {
            "sha": tree_sha,
            "tree": entries,
            "truncated": False,
        }

    def create_tree(self, payload):
        descriptor, name = tempfile.mkstemp(
            prefix="fake-github-index.",
            dir=self.repository.parent,
        )
        os.close(descriptor)
        index = Path(name)
        index.unlink()
        environment = {"GIT_INDEX_FILE": str(index)}
        try:
            self.git(
                "read-tree",
                payload["base_tree"],
                environment=environment,
            )
            for entry in payload["tree"]:
                self.git(
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    entry["mode"],
                    entry["sha"],
                    entry["path"],
                    environment=environment,
                )
            return self.git("write-tree", environment=environment)
        finally:
            if index.exists():
                index.unlink()

    def create_commit(self, payload):
        environment = {
            "GIT_AUTHOR_NAME": "GitHub",
            "GIT_AUTHOR_EMAIL": "noreply@github.invalid",
            "GIT_AUTHOR_DATE": "2001-01-01T00:00:00+00:00",
            "GIT_COMMITTER_NAME": "GitHub",
            "GIT_COMMITTER_EMAIL": "noreply@github.invalid",
            "GIT_COMMITTER_DATE": "2001-01-01T00:00:00+00:00",
        }
        arguments = ["commit-tree", payload["tree"]]
        for parent in payload["parents"]:
            arguments.extend(("-p", parent))
        revision = self.git(
            *arguments,
            input_bytes=(payload["message"] + "\n").encode(),
            environment=environment,
        )
        self.messages[revision] = payload["message"]
        return self.commit_response(revision)

    def __call__(
        self,
        _root,
        endpoint,
        *,
        fields=None,
        method="GET",
        payload=None,
    ):
        self.calls.append(
            {
                "endpoint": endpoint,
                "fields": fields,
                "method": method,
                "payload": payload,
            }
        )
        if endpoint == "graphql":
            if method != "POST":
                raise AssertionError("GraphQL operations must use POST")
            query = payload.get("query")
            variables = payload.get("variables")
            if "query CutoverRepositoryId" in query:
                return {
                    "data": {
                        "repository": {
                            "id": self.repository_id,
                            "nameWithOwner": "example/mib-doc-solution",
                        },
                    }
                }
            if "mutation CutoverRefCAS" not in query:
                raise AssertionError("unexpected GraphQL operation")
            mutation_input = variables["input"]
            if mutation_input["repositoryId"] != self.repository_id:
                raise AssertionError("CAS names another repository")
            updates = mutation_input["refUpdates"]
            if len(updates) != 1:
                raise AssertionError("CAS must update exactly one ref")
            update = updates[0]
            if update["force"] is not False:
                raise AssertionError("cutover ref update may not force")
            if update["name"] != f"refs/heads/{self.branch}":
                raise AssertionError("cutover CAS names another branch")
            if self.fail_before_patch:
                raise GovernedExperimentCLIError(
                    "simulated pre-CAS transport failure"
                )
            if self.race_to_ancestor_on_patch:
                self.head = self.git("rev-parse", f"{self.head}^")
            if self.race_on_patch:
                self.head = "f" * 40
            if self.head != update["beforeOid"]:
                raise GovernedExperimentCLIError(
                    "simulated GraphQL beforeOid CAS rejection"
                )
            self.head = update["afterOid"]
            response = {
                "data": {
                    "updateRefs": {
                        "clientMutationId": mutation_input[
                            "clientMutationId"
                        ],
                    }
                }
            }
            if self.lose_patch_and_ref_response:
                self.lose_patch_and_ref_response = False
                self.ref_read_failures = 1
                raise GovernedExperimentCLIError(
                    "simulated ambiguous CAS response"
                )
            if self.lose_patch_response:
                self.lose_patch_response = False
                raise GovernedExperimentCLIError(
                    "simulated lost CAS response"
                )
            return response
        if "/branches/" in endpoint:
            return {
                "commit": {"sha": self.head},
                "name": self.branch,
            }
        if "/git/ref/heads/" in endpoint:
            if self.ref_read_failures:
                self.ref_read_failures -= 1
                raise GovernedExperimentCLIError(
                    "simulated ambiguous ref read"
                )
            return {
                "object": {"sha": self.head, "type": "commit"},
                "ref": f"refs/heads/{self.branch}",
            }
        if endpoint.endswith("/git/blobs"):
            if method != "POST":
                raise AssertionError("blob creation must use POST")
            self.blob_posts += 1
            if self.fail_blob_post == self.blob_posts:
                raise GovernedExperimentCLIError(
                    "simulated partial blob publication"
                )
            raw = base64.b64decode(payload["content"])
            sha = self.git(
                "hash-object",
                "-w",
                "--stdin",
                input_bytes=raw,
            )
            return {"sha": sha}
        if endpoint.endswith("/git/trees"):
            if method != "POST":
                raise AssertionError("tree creation must use POST")
            return {"sha": self.create_tree(payload)}
        if endpoint.endswith("/git/commits"):
            if method != "POST":
                raise AssertionError("commit creation must use POST")
            return self.create_commit(payload)
        marker = "/git/commits/"
        if marker in endpoint:
            return self.commit_response(endpoint.split(marker, 1)[1])
        marker = "/git/trees/"
        if marker in endpoint:
            tree_sha = endpoint.split(marker, 1)[1].split("?", 1)[0]
            return self.tree_response(tree_sha)
        marker = "/contents/"
        if marker in endpoint:
            path = endpoint.split(marker, 1)[1]
            revision = fields["ref"]
            raw = self.git_bytes("show", f"{revision}:{path}")
            if self.tamper_contents_path == path:
                raw += b" "
            sha = self.git("rev-parse", f"{revision}:{path}")
            return {
                "content": base64.b64encode(raw).decode(),
                "encoding": "base64",
                "path": path,
                "sha": sha,
                "type": "file",
            }
        raise AssertionError(f"unexpected endpoint: {endpoint}")


class GovernedExperimentCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repo"
        self.repository.mkdir()
        self.external = self.root / "external"
        self.external.mkdir()
        self.github_repository = "example/mib-doc-solution"
        self.branch = "improve/score-148"

        self.git("init", "-b", self.branch)
        self.git("config", "user.name", "Governance Test")
        self.git("config", "user.email", "governance@example.invalid")
        self.git(
            "remote",
            "add",
            "solution",
            f"https://github.com/{self.github_repository}.git",
        )
        self.git(
            "config",
            f"branch.{self.branch}.remote",
            "solution",
        )
        self.git(
            "config",
            f"branch.{self.branch}.merge",
            f"refs/heads/{self.branch}",
        )
        (self.repository / ".gitignore").write_text(
            "evaluation/program/.program-integrity.lock\n",
            encoding="utf-8",
        )
        (self.repository / "solution.py").write_text(
            "def main():\n    return 0\n",
            encoding="utf-8",
        )
        pipeline = self.repository / "mib_pipeline"
        pipeline.mkdir()
        (pipeline / "__init__.py").write_text("", encoding="utf-8")
        program = self.repository / "evaluation/program"
        program.mkdir(parents=True)
        write_canonical(
            program / "frozen_baseline_manifest.json",
            {"schema": "test-baseline"},
        )
        for filename in (
            "candidate_state_ledger.jsonl",
            "experiment_ledger.jsonl",
            "protected_access_ledger.jsonl",
            "taint_registry.jsonl",
        ):
            (program / filename).write_bytes(b"")
        self.git("add", ".")
        self.git("commit", "-m", "Initial governed baseline")
        self.backend = FakeGitHub(self.head)

        self.population_path = self.external / "population.json"
        write_canonical(
            self.population_path,
            {
                "evaluator_sha256": SHA_A,
                "expected_record_count": 1000,
                "input_tree_sha256": SHA_B,
                "runtime_contract_sha256": SHA_C,
                "split_manifest_sha256": SHA_A,
                "truth_sha256": SHA_B,
            },
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def git(self, *arguments: str) -> str:
        completed = subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    @property
    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def cutover(self):
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ):
            return create_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                promotion_population_path=self.population_path,
            )

    def publish_cutover(self) -> None:
        self.git("add", "evaluation/program")
        self.git("commit", "-m", "Publish checkpoint cutover")
        self.backend.head = self.head
        self.backend.publish_local_governance(self.repository)

    def checkpoint_files(self):
        program = self.repository.joinpath(
            *CHECKPOINT_DIRECTORY_RELATIVE_PATH.parts
        )
        return tuple(
            path
            for path in program.glob("*.json")
            if len(path.stem) == 64
            and all(character in "0123456789abcdef" for character in path.stem)
        )

    def plan(self, **overrides):
        value = {
            "changed_files": ["mib_pipeline/decision_recovery.py"],
            "evidence_label": "public_grouped_robustness_not_unseen",
            "evaluator_sha256": SHA_A,
            "expected_record_count": 1000,
            "hypothesis_sha256": SHA_A,
            "input_tree_sha256": SHA_B,
            "parent_commit_sha": self.head,
            "primary_variable_sha256": SHA_C,
            "runtime_contract_sha256": SHA_C,
            "split_manifest_sha256": SHA_A,
            "truth_sha256": SHA_B,
        }
        value.update(overrides)
        return value

    def test_cutover_is_canonical_create_once_and_does_not_mutate_ledgers(self):
        _program, pointer_path, _directory, stores = _program_paths(
            self.repository
        )
        before = {
            name: store.path.read_bytes()
            for name, store in stores.items()
        }

        result = self.cutover()

        self.assertEqual(
            result["status"],
            "cutover_created_not_published",
        )
        pointer_raw = pointer_path.read_bytes()
        pointer = json.loads(pointer_raw)
        self.assertEqual(pointer_raw, _canonical_bytes(pointer))
        checkpoint_path = self.repository / pointer["checkpoint_path"]
        checkpoint_raw = checkpoint_path.read_bytes()
        checkpoint = json.loads(checkpoint_raw)
        self.assertEqual(checkpoint_raw, _canonical_bytes(checkpoint))
        self.assertIn("promotion_population", checkpoint)
        self.assertTrue(
            all(
                "path" in anchor
                for anchor in checkpoint["stores"].values()
            )
        )
        self.assertEqual(
            {
                name: store.path.read_bytes()
                for name, store in stores.items()
            },
            before,
        )
        self.assertEqual(
            set(self.git("status", "--porcelain=v1").splitlines()),
            {
                f"?? {result['checkpoint_path']}",
                "?? evaluation/program/cutover_authority.json",
                "?? evaluation/program/current_checkpoint.json",
            },
        )
        self.assertIn(
            "compare-and-swap",
            result["next_required_action"],
        )

    def test_cutover_rejects_dirty_tree_remote_drift_and_noncanonical_input(self):
        dirty = self.repository / "untracked.txt"
        dirty.write_text("dirty", encoding="utf-8")
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "worktree must be clean",
        ):
            create_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                promotion_population_path=self.population_path,
            )
        dirty.unlink()

        self.backend.head = "f" * 40
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "does not match authenticated",
        ):
            create_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                promotion_population_path=self.population_path,
            )
        self.backend.head = self.head

        self.population_path.write_text(
            json.dumps(json.loads(self.population_path.read_text()), indent=2),
            encoding="utf-8",
        )
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "must be canonical JSON",
        ):
            create_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                promotion_population_path=self.population_path,
            )
        self.assertFalse(
            self.repository.joinpath(
                *POINTER_RELATIVE_PATH.parts
            ).exists()
        )

    def test_cutover_rejects_external_symlink(self):
        symlink = self.external / "population-link.json"
        symlink.symlink_to(self.population_path)
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "may not be supplied through a symlink",
        ):
            create_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                promotion_population_path=symlink,
            )

    def test_repository_state_rejects_assume_unchanged_index_flag(self):
        self.git(
            "update-index",
            "--assume-unchanged",
            "solution.py",
        )
        (self.repository / "solution.py").write_text(
            "def main():\n    return 99\n",
            encoding="utf-8",
        )
        self.assertEqual(self.git("status", "--porcelain=v1"), "")

        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "exceptional index flags",
        ):
            _repository_state(self.repository, branch=self.branch)

    def test_repository_state_rejects_skip_worktree_index_flag(self):
        self.git(
            "update-index",
            "--skip-worktree",
            "solution.py",
        )
        (self.repository / "solution.py").write_text(
            "def main():\n    return 99\n",
            encoding="utf-8",
        )
        self.assertEqual(self.git("status", "--porcelain=v1"), "")

        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "exceptional index flags",
        ):
            _repository_state(self.repository, branch=self.branch)

    def test_cutover_binds_authority_to_exact_configured_upstream(self):
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "does not match the branch upstream",
        ):
            create_cutover(
                repository_root=self.repository,
                github_repository="attacker/mib-doc-solution",
                branch=self.branch,
                promotion_population_path=self.population_path,
            )
        self.assertEqual(self.backend.calls, [])

    def test_cutover_rejects_remote_head_move_during_operation(self):
        commit_calls = 0

        def moving_backend(root, endpoint, *, fields=None):
            nonlocal commit_calls
            if "/branches/" in endpoint:
                commit_calls += 1
                return {
                    "commit": {
                        "sha": (
                            self.head
                            if commit_calls == 1
                            else "f" * 40
                        )
                    },
                    "name": self.branch,
                }
            return self.backend(root, endpoint, fields=fields)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=moving_backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "branch moved during cutover",
        ):
            create_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                promotion_population_path=self.population_path,
            )
        self.assertFalse(
            self.repository.joinpath(
                *POINTER_RELATIVE_PATH.parts
            ).exists()
        )

    def test_cutover_rolls_back_if_local_head_moves_after_start(self):
        module = __import__(
            "devtools.governed_experiment_cli",
            fromlist=["_create_once"],
        )
        original = module._create_once

        def move_head(path, raw, *, label):
            created = original(path, raw, label=label)
            if label == "current checkpoint pointer":
                self.git("commit", "--allow-empty", "-m", "Concurrent commit")
            return created

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch(
            "devtools.governed_experiment_cli._create_once",
            side_effect=move_head,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "local Git HEAD moved during cutover",
        ):
            create_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                promotion_population_path=self.population_path,
            )
        self.assertFalse(
            self.repository.joinpath(*POINTER_RELATIVE_PATH.parts).exists()
        )

    def test_cutover_publisher_uses_exact_three_blob_github_cas(self):
        prepared = self.cutover()
        database = FakeGitDatabase(
            self.repository,
            self.head,
            self.branch,
        )

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            result = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(
            result["status"],
            "authenticated_cutover_published",
            result,
        )
        self.assertTrue(result["authority_verified"])
        self.assertEqual(result["parent_commit_sha"], prepared["local_head"])
        self.assertEqual(self.head, database.head)
        self.assertEqual(self.git("status", "--porcelain=v1"), "")
        blob_calls = [
            call
            for call in database.calls
            if call["method"] == "POST"
            and str(call["endpoint"]).endswith("/git/blobs")
        ]
        self.assertEqual(len(blob_calls), 3)
        tree_calls = [
            call
            for call in database.calls
            if call["method"] == "POST"
            and str(call["endpoint"]).endswith("/git/trees")
        ]
        self.assertEqual(len(tree_calls), 1)
        self.assertEqual(
            {
                entry["path"]
                for entry in tree_calls[0]["payload"]["tree"]
            },
            {
                prepared["authority_path"],
                prepared["checkpoint_path"],
                prepared["pointer_path"],
            },
        )
        self.assertEqual(
            set(tree_calls[0]["payload"]),
            {"base_tree", "tree"},
        )
        commit_calls = [
            call
            for call in database.calls
            if call["method"] == "POST"
            and str(call["endpoint"]).endswith("/git/commits")
        ]
        self.assertEqual(len(commit_calls), 1)
        self.assertEqual(
            commit_calls[0]["payload"]["parents"],
            [prepared["local_head"]],
        )
        self.assertEqual(
            commit_calls[0]["payload"]["message"],
            CUTOVER_COMMIT_MESSAGE,
        )
        self.assertEqual(
            commit_calls[0]["payload"]["author"],
            CUTOVER_COMMIT_ACTOR,
        )
        self.assertEqual(
            commit_calls[0]["payload"]["committer"],
            CUTOVER_COMMIT_ACTOR,
        )
        cas_calls = [
            call
            for call in database.calls
            if call["endpoint"] == "graphql"
            and "mutation CutoverRefCAS"
            in call["payload"]["query"]
        ]
        self.assertEqual(len(cas_calls), 1)
        mutation_input = cas_calls[0]["payload"]["variables"]["input"]
        self.assertEqual(
            mutation_input["repositoryId"],
            database.repository_id,
        )
        self.assertEqual(
            mutation_input["refUpdates"],
            [
                {
                    "afterOid": result["published_commit_sha"],
                    "beforeOid": prepared["local_head"],
                    "force": False,
                    "name": f"refs/heads/{self.branch}",
                }
            ],
        )

    def test_cutover_publisher_rejects_branch_race_at_patch(self):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        database.race_on_patch = True

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "compare-and-swap rejected.*branch moved",
        ):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(self.head, parent)
        self.assertEqual(database.blob_posts, 3)
        self.assertFalse(
            Path(self.git("rev-parse", "--git-dir"))
            .joinpath("governed-experiment-recovery")
            .joinpath(
                "cutover-" + database.head + ".json"
            )
            .exists()
        )

    def test_graphql_before_oid_rejects_race_to_parent_ancestor(self):
        self.git("commit", "--allow-empty", "-m", "Second baseline")
        self.backend.head = self.head
        self.cutover()
        parent = self.head
        ancestor = self.git("rev-parse", f"{parent}^")
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        database.race_to_ancestor_on_patch = True

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "compare-and-swap rejected.*branch moved",
        ):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(database.head, ancestor)
        self.assertEqual(self.head, parent)
        mutation_calls = [
            call
            for call in database.calls
            if call["endpoint"] == "graphql"
            and "mutation CutoverRefCAS"
            in call["payload"]["query"]
        ]
        self.assertEqual(len(mutation_calls), 1)
        update = mutation_calls[0]["payload"]["variables"]["input"][
            "refUpdates"
        ][0]
        self.assertEqual(update["beforeOid"], parent)
        self.assertFalse(update["force"])

    def test_pre_patch_failure_reuses_immutable_recorded_commit(self):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        database.fail_before_patch = True
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "compare-and-swap rejected",
        ):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(database.head, parent)
        commit_posts = [
            call
            for call in database.calls
            if call["method"] == "POST"
            and str(call["endpoint"]).endswith("/git/commits")
        ]
        self.assertEqual(len(commit_posts), 1)
        recovery_directory = (
            self.repository
            / ".git/governed-experiment-recovery"
        )
        transaction_receipts = tuple(
            recovery_directory.glob("cutover-transaction-*.json")
        )
        self.assertEqual(len(transaction_receipts), 1)
        transaction_before = transaction_receipts[0].read_bytes()
        recorded = json.loads(transaction_before)

        database.fail_before_patch = False
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            result = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            result["status"],
            "authenticated_cutover_published",
            result,
        )
        commit_posts = [
            call
            for call in database.calls
            if call["method"] == "POST"
            and str(call["endpoint"]).endswith("/git/commits")
        ]
        self.assertEqual(len(commit_posts), 1)
        self.assertEqual(
            result["published_commit_sha"],
            recorded["published_commit_sha"],
        )
        self.assertEqual(
            transaction_receipts[0].read_bytes(),
            transaction_before,
        )

    def test_retry_rejects_equivalent_commit_not_named_by_transaction(self):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        database.fail_before_patch = True
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaises(GovernedExperimentCLIError):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        transaction_path = next(
            (
                self.repository
                / ".git/governed-experiment-recovery"
            ).glob("cutover-transaction-*.json")
        )
        transaction = json.loads(transaction_path.read_bytes())
        recorded_commit = transaction["published_commit_sha"]
        equivalent_commit = database.git(
            "commit-tree",
            transaction["tree_sha"],
            "-p",
            parent,
            input_bytes=(CUTOVER_COMMIT_MESSAGE + "\n").encode(),
            environment={
                "GIT_AUTHOR_NAME": "Competing Publisher",
                "GIT_AUTHOR_EMAIL": "other@example.invalid",
                "GIT_AUTHOR_DATE": "2002-01-01T00:00:00+00:00",
                "GIT_COMMITTER_NAME": "Competing Publisher",
                "GIT_COMMITTER_EMAIL": "other@example.invalid",
                "GIT_COMMITTER_DATE": "2002-01-01T00:00:00+00:00",
            },
        )
        self.assertNotEqual(equivalent_commit, recorded_commit)
        database.messages[equivalent_commit] = CUTOVER_COMMIT_MESSAGE
        database.head = equivalent_commit
        database.fail_before_patch = False
        mutation_count = len(
            [
                call
                for call in database.calls
                if call["endpoint"] == "graphql"
                and "mutation CutoverRefCAS"
                in call["payload"]["query"]
            ]
        )

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "immutable transaction parent and its recorded commit",
        ):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(database.head, equivalent_commit)
        self.assertEqual(
            len(
                [
                    call
                    for call in database.calls
                    if call["endpoint"] == "graphql"
                    and "mutation CutoverRefCAS"
                    in call["payload"]["query"]
                ]
            ),
            mutation_count,
        )

    def test_cutover_publisher_rejects_tampered_prepared_bytes(self):
        self.cutover()
        pointer = self.repository.joinpath(*POINTER_RELATIVE_PATH.parts)
        pointer.write_bytes(pointer.read_bytes() + b" ")
        database = FakeGitDatabase(
            self.repository,
            self.head,
            self.branch,
        )

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "must be canonical JSON",
        ):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(database.calls, [])

    def test_partial_remote_objects_do_not_publish_and_retry_is_safe(self):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        database.fail_blob_post = 2
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "partial blob publication",
        ):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(database.head, parent)
        self.assertEqual(self.head, parent)
        self.assertEqual(database.blob_posts, 2)

        database.fail_blob_post = None
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            result = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            result["status"],
            "authenticated_cutover_published",
            result,
        )
        self.assertEqual(self.head, database.head)

    def test_patch_success_local_sync_failure_emits_immutable_receipt_and_retry(
        self,
    ):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), mock.patch(
            "devtools.governed_experiment_cli._synchronize_local_cutover",
            side_effect=OSError("simulated local sync failure"),
        ):
            failed = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(
            failed["status"],
            "remote_published_local_unsynced",
        )
        self.assertFalse(failed["authority_verified"])
        self.assertNotEqual(database.head, parent)
        self.assertEqual(self.head, parent)
        receipt = Path(failed["recovery_receipt_path"])
        receipt_before = receipt.read_bytes()
        receipt_value = json.loads(receipt_before)
        self.assertEqual(
            receipt_value["published_commit_sha"],
            database.head,
        )
        self.assertEqual(
            receipt_value["status"],
            "remote_published_local_unsynced",
        )
        blob_posts_before_retry = database.blob_posts

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            recovered = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(
            recovered["status"],
            "authenticated_cutover_already_published",
            recovered,
        )
        self.assertTrue(recovered["authority_verified"])
        self.assertEqual(database.blob_posts, blob_posts_before_retry)
        self.assertEqual(self.head, database.head)
        self.assertEqual(self.git("status", "--porcelain=v1"), "")
        self.assertEqual(receipt.read_bytes(), receipt_before)

    def test_existing_git_index_lock_is_never_overwritten(self):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        index_lock = self.repository / ".git/index.lock"
        lock_bytes = b"concurrent git writer owns this lock"
        index_lock.write_bytes(lock_bytes)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            failed = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            failed["status"],
            "remote_published_local_unsynced",
        )
        self.assertIn("index lock already exists", failed["error"])
        self.assertEqual(index_lock.read_bytes(), lock_bytes)
        self.assertEqual(self.head, parent)
        commit_posts_before = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )

        index_lock.unlink()
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            recovered = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            recovered["status"],
            "authenticated_cutover_already_published",
            recovered,
        )
        commit_posts_after = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )
        self.assertEqual(commit_posts_after, commit_posts_before)
        self.assertEqual(self.head, database.head)

    def test_git_index_symlink_cannot_escape_or_overwrite_target(self):
        index = self.repository / ".git/index"
        external_index = self.external / "external-index"
        external_index.write_bytes(index.read_bytes())
        before = external_index.read_bytes()
        index.unlink()
        index.symlink_to(external_index)

        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "owned regular, non-symlink file",
        ):
            _git_index_path(self.repository)

        self.assertTrue(index.is_symlink())
        self.assertEqual(external_index.read_bytes(), before)
        self.assertFalse(
            external_index.with_name("external-index.lock").exists()
        )

    def test_git_index_hardlink_is_rejected(self):
        index = self.repository / ".git/index"
        external_index = self.external / "hardlinked-index"
        external_index.write_bytes(index.read_bytes())
        index.unlink()
        os.link(external_index, index)

        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "owned regular, non-symlink file",
        ):
            _git_index_path(self.repository)

    def test_index_lock_hardlink_race_is_detached_before_failure(self):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        index = self.repository / ".git/index"
        external_alias = self.external / "raced-index-alias"
        real_replace = os.replace
        raced = False

        def hardlink_during_install(source, destination):
            nonlocal raced
            if (
                not raced
                and Path(source).name == "index.lock"
                and Path(destination).name == "index"
            ):
                raced = True
                os.link(source, external_alias)
            return real_replace(source, destination)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), mock.patch(
            "devtools.governed_experiment_cli.os.replace",
            side_effect=hardlink_during_install,
        ):
            failed = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertTrue(raced)
        self.assertEqual(
            failed["status"],
            "remote_published_local_unsynced",
        )
        self.assertIn(
            "external hard-link alias",
            failed["error"],
        )
        self.assertEqual(self.head, database.head)
        self.assertTrue(external_alias.exists())
        live_before = index.read_bytes()
        live_metadata = index.lstat()
        alias_metadata = external_alias.lstat()
        self.assertEqual(live_metadata.st_nlink, 1)
        self.assertNotEqual(
            (live_metadata.st_dev, live_metadata.st_ino),
            (alias_metadata.st_dev, alias_metadata.st_ino),
        )
        external_alias.write_bytes(b"tampered alias")
        self.assertEqual(index.read_bytes(), live_before)
        self.assertEqual(_git_index_path(self.repository), index.resolve())
        self.assertFalse((self.repository / ".git/index.lock").exists())

        commit_posts_before_retry = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            recovered = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            recovered["status"],
            "authenticated_cutover_already_published",
            recovered,
        )
        commit_posts_after_retry = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )
        self.assertEqual(
            commit_posts_after_retry,
            commit_posts_before_retry,
        )

    def test_nonregular_git_index_is_rejected(self):
        index = self.repository / ".git/index"
        index.unlink()
        index.mkdir()

        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "owned regular, non-symlink file",
        ):
            _git_index_path(self.repository)

    def test_retry_repairs_head_updated_before_index_lock_rename(self):
        self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        real_replace = os.replace
        failed_once = False

        def fail_index_install(source, destination):
            nonlocal failed_once
            if (
                not failed_once
                and Path(source).name == "index.lock"
                and Path(destination).name == "index"
            ):
                failed_once = True
                raise OSError("simulated index-lock rename failure")
            return real_replace(source, destination)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), mock.patch(
            "devtools.governed_experiment_cli.os.replace",
            side_effect=fail_index_install,
        ):
            failed = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            failed["status"],
            "remote_published_local_unsynced",
        )
        self.assertEqual(self.head, database.head)
        self.assertNotEqual(self.head, parent)
        self.assertFalse((self.repository / ".git/index.lock").exists())
        commit_posts_before = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            recovered = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            recovered["status"],
            "authenticated_cutover_already_published",
            recovered,
        )
        self.assertEqual(self.git("status", "--porcelain=v1"), "")
        commit_posts_after = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )
        self.assertEqual(commit_posts_after, commit_posts_before)

    def test_lost_patch_response_detects_exact_p_without_second_commit(self):
        self.cutover()
        database = FakeGitDatabase(
            self.repository,
            self.head,
            self.branch,
        )
        database.lose_patch_response = True

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            result = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(
            result["status"],
            "authenticated_cutover_published",
            result,
        )
        commit_posts = [
            call
            for call in database.calls
            if call["method"] == "POST"
            and str(call["endpoint"]).endswith("/git/commits")
        ]
        self.assertEqual(len(commit_posts), 1)

    def test_ambiguous_patch_and_ref_response_records_intent_and_retries_p(
        self,
    ):
        self.cutover()
        database = FakeGitDatabase(
            self.repository,
            self.head,
            self.branch,
        )
        database.lose_patch_and_ref_response = True

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            ambiguous = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            ambiguous["status"],
            "branch_cas_outcome_ambiguous_local_unsynced",
        )
        self.assertFalse(ambiguous["authority_verified"])
        receipt = json.loads(
            Path(ambiguous["recovery_receipt_path"]).read_bytes()
        )
        self.assertEqual(
            receipt["status"],
            "branch_cas_outcome_ambiguous_local_unsynced",
        )
        commit_posts_before = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ):
            recovered = publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            recovered["status"],
            "authenticated_cutover_already_published",
            recovered,
        )
        commit_posts_after = len(
            [
                call
                for call in database.calls
                if call["method"] == "POST"
                and str(call["endpoint"]).endswith("/git/commits")
            ]
        )
        self.assertEqual(commit_posts_after, commit_posts_before)
        self.assertEqual(
            recovered["published_commit_sha"],
            ambiguous["published_commit_sha"],
        )

    def test_new_remote_commit_is_fully_verified_before_branch_cas(self):
        prepared = self.cutover()
        parent = self.head
        database = FakeGitDatabase(
            self.repository,
            parent,
            self.branch,
        )
        database.tamper_contents_path = prepared["pointer_path"]

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=database,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "published cutover contents differ",
        ):
            publish_prepared_cutover(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.assertEqual(database.head, parent)
        self.assertEqual(self.head, prepared["local_head"])
        mutation_calls = [
            call
            for call in database.calls
            if call["endpoint"] == "graphql"
            and "mutation CutoverRefCAS"
            in call["payload"]["query"]
        ]
        self.assertEqual(mutation_calls, [])

    def test_authority_requires_remote_local_byte_identity(self):
        self.cutover()
        self.publish_cutover()

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ):
            verified = verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertEqual(
            verified["status"],
            "authenticated_current_checkpoint_verified",
        )
        self.assertEqual(
            verified["publication"]["transition"],
            "cutover",
        )
        self.assertEqual(
            set(verified["publication"]["changed_paths"]),
            {
                verified["checkpoint_path"],
                CUTOVER_AUTHORITY_RELATIVE_PATH.as_posix(),
                POINTER_RELATIVE_PATH.as_posix(),
            },
        )

        self.backend.files[
            POINTER_RELATIVE_PATH.as_posix()
        ] += b" "
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "pointers differ",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

    def test_authority_rejects_non_file_github_contents_response(self):
        self.cutover()
        self.publish_cutover()

        def symlink_response(root, endpoint, *, fields=None):
            response = self.backend(root, endpoint, fields=fields)
            if "/contents/" in endpoint:
                response["type"] = "symlink"
            return response

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=symlink_response,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "is not a regular file",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.backend.publish_local_governance(self.repository)
        pointer = json.loads(
            self.repository.joinpath(
                *POINTER_RELATIVE_PATH.parts
            ).read_text()
        )
        self.backend.files[pointer["checkpoint_path"]] += b" "
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "checkpoints differ",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

    def test_verify_authority_rejects_non_governance_publication_changes(self):
        result = self.cutover()
        (self.repository / "unrelated.txt").write_text(
            "not governance\n",
            encoding="utf-8",
        )
        self.git("add", "evaluation/program", "unrelated.txt")
        self.git("commit", "-m", "Mixed governance publication")
        self.backend.head = self.head
        self.backend.publish_local_governance(self.repository)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "not the exact governance-only transition",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )
        self.assertTrue(
            (self.repository / result["checkpoint_path"]).exists()
        )

    def test_verify_rejects_commit_inserted_after_cutover_return(self):
        self.cutover()
        self.git("commit", "--allow-empty", "-m", "Inserted commit")
        self.git("add", "evaluation/program")
        self.git("commit", "-m", "Publish stale cutover")
        self.backend.head = self.head
        self.backend.publish_local_governance(self.repository)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "parent does not match.*authority attestation",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

    def test_verify_authority_rejects_branch_move_during_topology_check(self):
        self.cutover()
        self.publish_cutover()
        branch_calls = 0

        def moving_backend(root, endpoint, *, fields=None):
            nonlocal branch_calls
            if "/branches/" in endpoint:
                branch_calls += 1
                return {
                    "commit": {
                        "sha": (
                            self.backend.head
                            if branch_calls == 1
                            else "f" * 40
                        )
                    },
                    "name": self.branch,
                }
            return self.backend(root, endpoint, fields=fields)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=moving_backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "moved during authority verification",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

    def test_repository_state_rejects_replace_graft_and_shallow_history(self):
        self.cutover()
        self.publish_cutover()
        self.git("replace", "HEAD", "HEAD^")
        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "replacement objects",
        ):
            _repository_state(self.repository, branch=self.branch)
        self.git("replace", "-d", "HEAD")

        git_directory = Path(self.git("rev-parse", "--git-dir"))
        if not git_directory.is_absolute():
            git_directory = self.repository / git_directory
        grafts = git_directory / "info/grafts"
        grafts.write_text(f"{self.head} {self.head}^\n", encoding="utf-8")
        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "graft history",
        ):
            _repository_state(self.repository, branch=self.branch)
        grafts.unlink()

        shallow = git_directory / "shallow"
        shallow.write_text(f"{self.head}\n", encoding="utf-8")
        with self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "shallow Git history",
        ):
            _repository_state(self.repository, branch=self.branch)

    def test_authority_rejects_pointer_escape_and_checkpoint_symlink(self):
        self.cutover()
        self.publish_cutover()
        pointer_path = self.repository.joinpath(
            *POINTER_RELATIVE_PATH.parts
        )
        pointer = json.loads(pointer_path.read_text())
        pointer["checkpoint_path"] = "evaluation/program/../escape.json"
        write_canonical(pointer_path, pointer)
        self.git("add", pointer_path.relative_to(self.repository).as_posix())
        self.git("commit", "-m", "Malicious pointer")
        self.backend.head = self.head
        self.backend.files[
            POINTER_RELATIVE_PATH.as_posix()
        ] = pointer_path.read_bytes()
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "digest-addressed path",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

        self.git("reset", "--hard", "HEAD^")
        self.backend.head = self.head
        self.backend.publish_local_governance(self.repository)
        pointer = json.loads(pointer_path.read_text())
        checkpoint_path = self.repository / pointer["checkpoint_path"]
        replacement = self.repository / "checkpoint-target"
        replacement.write_bytes(checkpoint_path.read_bytes())
        checkpoint_path.unlink()
        checkpoint_path.symlink_to(replacement)
        self.git("add", "-A")
        self.git("commit", "-m", "Symlink checkpoint")
        self.backend.head = self.head
        self.backend.files[pointer["checkpoint_path"]] = (
            replacement.read_bytes()
        )
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "escapes the trusted program root|non-symlink file",
        ):
            verify_authority(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
            )

    def test_preregister_appends_exactly_one_and_writes_one_successor(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        source_before = (
            self.repository / "mib_pipeline/__init__.py"
        ).read_bytes()

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ):
            result = preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-guarded-review-approval-v1",
                plan_path=plan_path,
            )

        ledger = ExperimentLedger(
            self.repository
            / "evaluation/program/experiment_ledger.jsonl"
        )
        self.assertEqual(len(ledger.plans()), 1)
        self.assertEqual(len(ledger.results()), 0)
        self.assertEqual(
            result["plan_record_hash"],
            ledger.store.head,
        )
        pointer = json.loads(
            self.repository.joinpath(
                *POINTER_RELATIVE_PATH.parts
            ).read_text()
        )
        self.assertEqual(
            pointer["checkpoint_sha256"],
            result["checkpoint_sha256"],
        )
        self.assertEqual(
            pointer["previous_checkpoint_sha256"],
            result["previous_checkpoint_sha256"],
        )
        checkpoints = self.checkpoint_files()
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(
            (
                self.repository / "mib_pipeline/__init__.py"
            ).read_bytes(),
            source_before,
        )
        status = self.git("status", "--porcelain=v1")
        self.assertIn("experiment_ledger.jsonl", status)
        self.assertIn("current_checkpoint.json", status)
        self.assertIn(result["checkpoint_path"], status)
        self.assertIn("commit and push", result["next_required_action"])

    def test_preregister_rejects_noncanonical_plan_and_parent_drift(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        plan_path.write_text(
            json.dumps(self.plan(), indent=2),
            encoding="utf-8",
        )
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "must be canonical JSON",
        ):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-v1",
                plan_path=plan_path,
            )
        self.assertEqual(
            CanonicalHashChainStore(
                self.repository
                / "evaluation/program/experiment_ledger.jsonl"
            ).length,
            0,
        )

        write_canonical(
            plan_path,
            self.plan(parent_commit_sha="f" * 40),
        )
        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "parent_commit_sha",
        ):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-v1",
                plan_path=plan_path,
            )

    def test_preregister_rolls_back_partial_pointer_update(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        pointer_path = self.repository.joinpath(
            *POINTER_RELATIVE_PATH.parts
        )
        pointer_before = pointer_path.read_bytes()
        experiment_path = (
            self.repository
            / "evaluation/program/experiment_ledger.jsonl"
        )
        experiment_before = experiment_path.read_bytes()
        original = __import__(
            "devtools.governed_experiment_cli",
            fromlist=["_atomic_replace_exact"],
        )._atomic_replace_exact
        calls = 0

        def fail_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated pointer failure")
            return original(*args, **kwargs)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch(
            "devtools.governed_experiment_cli._atomic_replace_exact",
            side_effect=fail_first,
        ), self.assertRaisesRegex(OSError, "simulated pointer"):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-v1",
                plan_path=plan_path,
            )
        self.assertEqual(pointer_path.read_bytes(), pointer_before)
        self.assertEqual(experiment_path.read_bytes(), experiment_before)
        self.assertEqual(
            len(self.checkpoint_files()),
            1,
        )
        self.assertEqual(self.git("status", "--porcelain=v1"), "")

    def test_preregister_rolls_back_if_remote_moves_after_local_transition(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        pointer_path = self.repository.joinpath(
            *POINTER_RELATIVE_PATH.parts
        )
        pointer_before = pointer_path.read_bytes()
        experiment_path = (
            self.repository
            / "evaluation/program/experiment_ledger.jsonl"
        )
        experiment_before = experiment_path.read_bytes()
        branch_calls = 0

        def moving_after_mutation(root, endpoint, *, fields=None):
            nonlocal branch_calls
            if "/branches/" in endpoint:
                branch_calls += 1
                return {
                    "commit": {
                        "sha": (
                            self.backend.head
                            if branch_calls < 3
                            else "f" * 40
                        )
                    },
                    "name": self.branch,
                }
            return self.backend(root, endpoint, fields=fields)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=moving_after_mutation,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "moved during preregistration",
        ):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-remote-race-v1",
                plan_path=plan_path,
            )

        self.assertEqual(pointer_path.read_bytes(), pointer_before)
        self.assertEqual(experiment_path.read_bytes(), experiment_before)
        self.assertEqual(len(self.checkpoint_files()), 1)
        self.assertEqual(self.git("status", "--porcelain=v1"), "")

    def test_atomic_pointer_replace_restores_original_after_directory_fsync_error(
        self,
    ):
        target = self.external / "pointer.json"
        original_bytes = b'{"old":true}\n'
        replacement_bytes = b'{"new":true}\n'
        target.write_bytes(original_bytes)
        real_fsync = os.fsync
        calls = 0

        def fail_directory_fsync(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated directory fsync failure")
            return real_fsync(descriptor)

        with mock.patch(
            "devtools.governed_experiment_cli.os.fsync",
            side_effect=fail_directory_fsync,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "original bytes were restored",
        ):
            _atomic_replace_exact(
                target,
                expected=original_bytes,
                replacement=replacement_bytes,
                label="test pointer",
            )
        self.assertEqual(target.read_bytes(), original_bytes)

    def test_concurrent_preregistration_loser_cannot_rollback_winner(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        barrier = threading.Barrier(2)
        real_resolve = GitHubCheckpointAuthority.resolve
        first_resolution = threading.local()

        def synchronized_resolve(authority, *, stores):
            result = real_resolve(authority, stores=stores)
            if not getattr(first_resolution, "done", False):
                first_resolution.done = True
                barrier.wait(timeout=10)
            return result

        def attempt():
            try:
                result = preregister(
                    repository_root=self.repository,
                    github_repository=self.github_repository,
                    branch=self.branch,
                    experiment_id="wo17-concurrent-v1",
                    plan_path=plan_path,
                )
                return ("accepted", result)
            except (GovernedExperimentCLIError, IntegrityError) as exc:
                return ("blocked", str(exc))

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch.object(
            GitHubCheckpointAuthority,
            "resolve",
            synchronized_resolve,
        ):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2
            ) as executor:
                outcomes = list(executor.map(lambda _index: attempt(), range(2)))

        self.assertEqual(
            [status for status, _value in outcomes].count("accepted"),
            1,
        )
        self.assertEqual(
            [status for status, _value in outcomes].count("blocked"),
            1,
        )
        ledger = ExperimentLedger(
            self.repository
            / "evaluation/program/experiment_ledger.jsonl"
        )
        self.assertEqual(len(ledger.plans()), 1)
        accepted = next(
            value
            for status, value in outcomes
            if status == "accepted"
        )
        self.assertEqual(ledger.store.head, accepted["plan_record_hash"])
        pointer = json.loads(
            self.repository.joinpath(
                *POINTER_RELATIVE_PATH.parts
            ).read_text()
        )
        self.assertEqual(
            pointer["checkpoint_sha256"],
            accepted["checkpoint_sha256"],
        )

    def test_preregister_rejects_concurrent_candidate_source_change(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        experiment_path = (
            self.repository
            / "evaluation/program/experiment_ledger.jsonl"
        )
        experiment_before = experiment_path.read_bytes()
        pointer_path = self.repository.joinpath(
            *POINTER_RELATIVE_PATH.parts
        )
        pointer_before = pointer_path.read_bytes()
        module = __import__(
            "devtools.governed_experiment_cli",
            fromlist=["_create_once"],
        )
        original = module._create_once

        def change_candidate(path, raw, *, label):
            created = original(path, raw, label=label)
            if label == "successor checkpoint":
                (
                    self.repository / "mib_pipeline/__init__.py"
                ).write_text("# concurrent candidate edit\n", encoding="utf-8")
            return created

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch(
            "devtools.governed_experiment_cli._create_once",
            side_effect=change_candidate,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "outside the exact governance publication set",
        ):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-v1",
                plan_path=plan_path,
            )
        self.assertEqual(experiment_path.read_bytes(), experiment_before)
        self.assertEqual(pointer_path.read_bytes(), pointer_before)
        self.assertEqual(len(self.checkpoint_files()), 1)
        self.assertEqual(
            self.git("status", "--porcelain=v1"),
            "M mib_pipeline/__init__.py",
        )

    def test_preregister_rejects_hidden_post_start_candidate_edit(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        pointer_path = self.repository.joinpath(
            *POINTER_RELATIVE_PATH.parts
        )
        pointer_before = pointer_path.read_bytes()
        experiment_path = (
            self.repository / "evaluation/program/experiment_ledger.jsonl"
        )
        experiment_before = experiment_path.read_bytes()
        module = __import__(
            "devtools.governed_experiment_cli",
            fromlist=["_create_once"],
        )
        original = module._create_once

        def hide_candidate_edit(path, raw, *, label):
            created = original(path, raw, label=label)
            if label == "successor checkpoint":
                (self.repository / "solution.py").write_text(
                    "def main():\n    return 99\n",
                    encoding="utf-8",
                )
                self.git(
                    "update-index",
                    "--assume-unchanged",
                    "solution.py",
                )
            return created

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch(
            "devtools.governed_experiment_cli._create_once",
            side_effect=hide_candidate_edit,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "exceptional Git index flag",
        ):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-hidden-edit-v1",
                plan_path=plan_path,
            )
        self.assertEqual(pointer_path.read_bytes(), pointer_before)
        self.assertEqual(experiment_path.read_bytes(), experiment_before)
        self.assertEqual(len(self.checkpoint_files()), 1)

    def test_preregister_rolls_back_if_local_head_moves_after_start(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        pointer_path = self.repository.joinpath(
            *POINTER_RELATIVE_PATH.parts
        )
        pointer_before = pointer_path.read_bytes()
        experiment_path = (
            self.repository / "evaluation/program/experiment_ledger.jsonl"
        )
        experiment_before = experiment_path.read_bytes()
        module = __import__(
            "devtools.governed_experiment_cli",
            fromlist=["_create_once"],
        )
        original = module._create_once

        def move_head(path, raw, *, label):
            created = original(path, raw, label=label)
            if label == "successor checkpoint":
                self.git("commit", "--allow-empty", "-m", "Concurrent commit")
            return created

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch(
            "devtools.governed_experiment_cli._create_once",
            side_effect=move_head,
        ), self.assertRaisesRegex(
            GovernedExperimentCLIError,
            "local Git HEAD moved during preregistration",
        ):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-local-head-race-v1",
                plan_path=plan_path,
            )
        self.assertEqual(pointer_path.read_bytes(), pointer_before)
        self.assertEqual(experiment_path.read_bytes(), experiment_before)
        self.assertEqual(len(self.checkpoint_files()), 1)

    def test_preregister_rejects_rewritten_successor_population(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        real = ExperimentLedger.preregister

        def rewrite_population(ledger, *args, **kwargs):
            receipt = real(ledger, *args, **kwargs)
            value = json.loads(receipt.next_checkpoint_bytes)
            value["promotion_population"]["expected_record_count"] += 1
            raw = _canonical_bytes(value)
            integrity = replace(
                receipt.integrity,
                checkpoint_bytes=raw,
                checkpoint_sha256=__import__("hashlib").sha256(raw).hexdigest(),
            )
            return replace(receipt, integrity=integrity)

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch.object(
            ExperimentLedger,
            "preregister",
            rewrite_population,
        ), mock.patch.object(
            __import__(
                "devtools.governed_experiment_cli",
                fromlist=["CheckpointAuthorityResolver"],
            ).CheckpointAuthorityResolver,
            "validate_successor",
        ), self.assertRaisesRegex(
            IntegrityError,
            "rewrote immutable checkpoint authority",
        ):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-rewritten-population-v1",
                plan_path=plan_path,
            )
        self.assertEqual(self.git("status", "--porcelain=v1"), "")

    def test_extra_ledger_append_is_rejected_and_rolled_back(self):
        self.cutover()
        self.publish_cutover()
        plan_path = self.external / "plan.json"
        write_canonical(plan_path, self.plan())
        taint_path = (
            self.repository / "evaluation/program/taint_registry.jsonl"
        )
        before = taint_path.read_bytes()
        real = ExperimentLedger.preregister

        def append_extra(ledger, *args, **kwargs):
            receipt = real(ledger, *args, **kwargs)
            CanonicalHashChainStore(taint_path).append(
                {"event": "unexpected"}
            )
            return receipt

        with mock.patch(
            "devtools.governed_experiment_cli._gh_json",
            side_effect=self.backend,
        ), mock.patch.object(
            ExperimentLedger,
            "preregister",
            append_extra,
        ), self.assertRaises(IntegrityError):
            preregister(
                repository_root=self.repository,
                github_repository=self.github_repository,
                branch=self.branch,
                experiment_id="wo17-v1",
                plan_path=plan_path,
            )
        self.assertEqual(taint_path.read_bytes(), before)
        self.assertEqual(self.git("status", "--porcelain=v1"), "")

    def test_gh_json_uses_gh_api_subprocess_without_shell(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"sha":"' + ("a" * 40) + '"}\n',
            stderr="",
        )
        with mock.patch(
            "devtools.governed_experiment_cli.subprocess.run",
            return_value=completed,
        ) as run:
            value = _gh_json(
                self.repository,
                "repos/example/repo/branches/main",
                fields={"ref": "a" * 40},
            )
        self.assertEqual(value["sha"], "a" * 40)
        arguments = run.call_args.args[0]
        self.assertTrue(Path(arguments[0]).is_absolute())
        self.assertEqual(
            arguments[1:6],
            ["api", "--hostname", "github.com", "--method", "GET"],
        )
        self.assertNotIn("shell", run.call_args.kwargs)
        environment = run.call_args.kwargs["env"]
        self.assertFalse(
            any(
                name.startswith(("GIT_", "GH_", "GITHUB_"))
                for name in environment
            )
        )

    def test_gh_json_write_uses_typed_canonical_stdin_not_string_fields(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"ref":"refs/heads/main"}\n',
            stderr="",
        )
        with mock.patch(
            "devtools.governed_experiment_cli.subprocess.run",
            return_value=completed,
        ) as run:
            _gh_json(
                self.repository,
                "repos/example/repo/git/refs/heads/main",
                method="PATCH",
                payload={"force": False, "sha": "a" * 40},
            )
        arguments = run.call_args.args[0]
        self.assertEqual(
            arguments[1:6],
            ["api", "--hostname", "github.com", "--method", "PATCH"],
        )
        self.assertEqual(arguments[-2:], ["--input", "-"])
        self.assertNotIn("-f", arguments)
        self.assertEqual(
            run.call_args.kwargs["input"],
            canonical_json({"force": False, "sha": "a" * 40}),
        )
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_cutover_cli_returns_nonzero_for_emitted_recovery_receipt(self):
        with mock.patch(
            "devtools.governed_experiment_cli.execute_cutover",
            return_value={
                "authority_verified": False,
                "recovery_receipt_sha256": SHA_A,
                "status": "remote_published_local_unsynced",
            },
        ), mock.patch("builtins.print"):
            exit_code = main(
                [
                    "--repository-root",
                    str(self.repository),
                    "--github-repository",
                    self.github_repository,
                    "--branch",
                    self.branch,
                    "cutover",
                    "--promotion-population",
                    str(self.population_path),
                ]
            )
        self.assertEqual(exit_code, 3)


if __name__ == "__main__":
    unittest.main()
