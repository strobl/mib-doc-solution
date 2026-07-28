from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import subprocess
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from devtools import grouped_split_evidence as evidence
from devtools.experiment_control import (
    CanonicalHashChainStore,
    RepeatedGroupedSplitManager,
    canonical_json,
    require_aggregate_only,
)


SOURCE_REVISION = "a" * 40


def _minimal_pdf_bytes(*, width: int, height: int) -> bytes:
    """Return one structurally valid blank-page PDF without test dependencies."""

    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
            + str(width).encode("ascii")
            + b" "
            + str(height).encode("ascii")
            + b"] /Resources << >> /Contents 4 0 R >>"
        ),
        b"<< /Length 4 >>\nstream\nq\nQ\nendstream",
    )
    content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode("ascii"))
        content.extend(body)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(b"xref\n0 5\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n"
    )
    content.extend(str(xref_offset).encode("ascii"))
    content.extend(b"\n%%EOF\n")
    return bytes(content)


class GroupedSplitEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_dir = self.root / "public-pdfs"
        self.input_dir.mkdir()
        self.case_ids = tuple(
            f"MIB-{index:06d}" for index in range(1, 21)
        )
        self.group_ids = tuple(
            f"page-count-{index:02d}__ink-bucket-{index + 10:02d}"
            for index in range(1, 11)
        )
        for index, case_id in enumerate(self.case_ids):
            (self.input_dir / f"{case_id}.pdf").write_bytes(
                _minimal_pdf_bytes(
                    width=600 + index,
                    height=800 + index,
                )
            )
        self.manifest = self.root / "layout-manifest.json"
        with self._rendering_patches():
            self.manifest_payload, _ = (
                evidence.manifest_freezer.build_layout_manifest(
                    self.input_dir,
                    expected_count=len(self.case_ids),
                    split_seed="wo12-real-demo-test-v1",
                )
            )
        self._write_manifest(self.manifest_payload)

        self.taint_registry = self.root / "taint-registry.jsonl"
        CanonicalHashChainStore(self.taint_registry).append(
            {
                "event": "taint",
                "group_id": "public-labeled-cohort",
                "reason": "public labels were evaluated",
                "source": "test fixture",
            }
        )
        self.expected_input_sha = evidence._input_tree_sha256(
            tuple(sorted(self.input_dir.iterdir()))
        )
        self.tool_sha = hashlib.sha256(
            Path(evidence.__file__).read_bytes()
        ).hexdigest()
        self.checkout_calls: list[tuple[Path, str]] = []

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_manifest(
        self,
        payload: dict[str, object],
        *,
        canonical: bool = True,
    ) -> None:
        if canonical:
            content = canonical_json(payload) + "\n"
        else:
            content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self.manifest.write_text(content, encoding="utf-8")

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _checkout_verifier(self, root: Path, revision: str) -> None:
        self.checkout_calls.append((root, revision))

    @staticmethod
    def _layout_signature(pdf_path: Path) -> tuple[int, int]:
        case_number = int(pdf_path.stem[len("MIB-") :])
        group_number = (case_number - 1) // 2 + 1
        return group_number, group_number + 10

    @staticmethod
    def _versions() -> dict[str, str]:
        return {
            "pillow_version": "test-pillow",
            "pdfium_version": "test-pdfium",
            "pypdfium2_version": "test-pypdfium2",
        }

    @contextlib.contextmanager
    def _rendering_patches(self):
        with mock.patch.object(
            evidence.manifest_freezer,
            "layout_signature",
            side_effect=self._layout_signature,
        ), mock.patch.object(
            evidence.manifest_freezer,
            "_rendering_version_metadata",
            side_effect=self._versions,
        ):
            yield

    def _arguments(self) -> dict[str, object]:
        return {
            "layout_manifest_path": self.manifest,
            "expected_layout_manifest_sha256": self._sha(self.manifest),
            "input_dir": self.input_dir,
            "expected_input_tree_sha256": self.expected_input_sha,
            "taint_registry_path": self.taint_registry,
            "expected_taint_registry_sha256": self._sha(
                self.taint_registry
            ),
            "source_revision_sha": SOURCE_REVISION,
            "expected_tool_source_sha256": self.tool_sha,
        }

    def _build(self, **overrides: object) -> dict[str, object]:
        arguments = self._arguments()
        arguments.update(overrides)
        with mock.patch.object(
            evidence,
            "verify_clean_source_checkout",
            side_effect=self._checkout_verifier,
        ), self._rendering_patches():
            return evidence.build_aggregate_evidence(  # type: ignore[arg-type]
                **arguments
            )

    def test_builds_exact_identity_free_3x5_aggregate(self) -> None:
        aggregate = self._build()

        self.assertEqual(
            self.checkout_calls,
            [(evidence.REPO_ROOT, SOURCE_REVISION)],
        )
        self.assertEqual(aggregate["status"], "passed")
        self.assertEqual(
            aggregate["evidence_label"],
            "public_grouped_robustness_not_unseen",
        )
        self.assertEqual(aggregate["repeat_count"], 3)
        self.assertEqual(aggregate["fold_count"], 5)
        self.assertEqual(aggregate["split_count"], 15)
        self.assertEqual(aggregate["expected_record_count"], 20)
        self.assertEqual(aggregate["layout_group_count"], 10)
        self.assertEqual(len(aggregate["fold_metrics"]), 15)
        for metrics in aggregate["fold_metrics"].values():
            self.assertEqual(
                metrics,
                {
                    "tuning_group_count": 8,
                    "tuning_record_count": 16,
                    "validation_group_count": 2,
                    "validation_record_count": 4,
                },
            )
        self.assertTrue(all(aggregate["checks"].values()))
        self.assertEqual(
            aggregate["counts"]["validation_record_assignment_count"],
            60,
        )
        self.assertEqual(
            aggregate["counts"]["validation_group_assignment_count"],
            30,
        )
        self.assertEqual(
            aggregate["counts"][
                "whole_public_cohort_taint_token_event_count"
            ],
            1,
        )
        self.assertEqual(
            aggregate["counts"]["matching_layout_taint_group_count"],
            0,
        )
        require_aggregate_only(aggregate)

        serialized = canonical_json(aggregate)
        self.assertNotIn("MIB-", serialized)
        self.assertNotIn("page-count-", serialized)
        self.assertNotIn(".pdf", serialized.casefold())
        self.assertNotIn(str(self.root), serialized)

    def test_markdown_is_deterministic_aggregate_only_and_not_unseen(self) -> None:
        aggregate = self._build()

        first = evidence.render_aggregate_markdown(aggregate)
        second = evidence.render_aggregate_markdown(dict(aggregate))

        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        self.assertEqual(first.count("\n| 1 |"), 5)
        self.assertEqual(first.count("\n| 2 |"), 5)
        self.assertEqual(first.count("\n| 3 |"), 5)
        self.assertIn("public_grouped_robustness_not_unseen", first)
        self.assertIn("evaluated PDFs and labels are public", first)
        self.assertIn(
            "no fold is represented as protected, private, pristine, or unseen",
            first,
        )
        self.assertIn("Generic public-cohort taint token present: YES", first)
        self.assertIn("not bound to the layout-manifest", first)
        self.assertIn("does not attest", first)
        self.assertIn("Synthetic exclusion-mechanics probe: PASS", first)
        self.assertIn(
            "Canonical manifest declares label-blind construction: PASS",
            first,
        )
        self.assertIn("makes no claim that the manifest was fixed", first)
        self.assertIn("manifest hash and seed to be preregistered", first)
        self.assertNotIn("frozen before scoring", first)
        self.assertNotIn("MIB-", first)
        self.assertNotIn("page-count-", first)
        self.assertNotIn(".pdf", first.casefold())
        self.assertNotIn(str(self.root), first)

    def test_rejects_bad_or_mismatched_explicit_bindings(self) -> None:
        cases = (
            (
                {"expected_layout_manifest_sha256": "0" * 64},
                "layout manifest does not match",
            ),
            (
                {"expected_input_tree_sha256": "0" * 64},
                "input tree does not match",
            ),
            (
                {"expected_taint_registry_sha256": "0" * 64},
                "taint registry does not match",
            ),
            (
                {"expected_tool_source_sha256": "0" * 64},
                "tool source does not match",
            ),
            (
                {"source_revision_sha": "short"},
                "full Git commit SHA",
            ),
            (
                {"expected_input_tree_sha256": "short"},
                "full SHA-256 digest",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    evidence.GroupedSplitEvidenceBuildError,
                    message,
                ):
                    self._build(**overrides)

    def test_tool_digest_is_checked_before_checkout_verifier(self) -> None:
        verifier = mock.Mock()
        arguments = self._arguments()
        arguments["expected_tool_source_sha256"] = "0" * 64
        with mock.patch.object(
            evidence,
            "verify_clean_source_checkout",
            verifier,
        ), self._rendering_patches(), self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "tool source does not match",
        ):
            evidence.build_aggregate_evidence(**arguments)  # type: ignore[arg-type]
        verifier.assert_not_called()

    def test_manifest_must_be_exact_canonical_freezer_output(self) -> None:
        mutations = []

        extra_root = json.loads(json.dumps(self.manifest_payload))
        extra_root["unexpected"] = True
        mutations.append(("schema", extra_root, True))

        bad_signature = json.loads(json.dumps(self.manifest_payload))
        bad_signature["layout_signature"]["inputs"] = ["case_id"]
        mutations.append(("signature metadata", bad_signature, True))

        bad_group = json.loads(json.dumps(self.manifest_payload))
        bad_group["cases"][0]["layout_group"] = "identity-derived-group"
        mutations.append(("case rows", bad_group, True))

        unsorted = json.loads(json.dumps(self.manifest_payload))
        unsorted["cases"] = list(reversed(unsorted["cases"]))
        mutations.append(("case rows", unsorted, True))

        noncanonical = json.loads(json.dumps(self.manifest_payload))
        mutations.append(("not canonical", noncanonical, False))

        for message, payload, canonical in mutations:
            with self.subTest(message=message):
                self._write_manifest(payload, canonical=canonical)
                with self.assertRaisesRegex(
                    evidence.GroupedSplitEvidenceBuildError,
                    message,
                ):
                    self._build(
                        expected_layout_manifest_sha256=self._sha(
                            self.manifest
                        )
                    )
                self._write_manifest(self.manifest_payload)

    def test_manifest_requires_exact_label_blind_three_by_five_contract(
        self,
    ) -> None:
        for key, value in (
            ("repeats", 2),
            ("folds", 4),
            ("label_blind_construction", False),
        ):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(self.manifest_payload))
                changed[key] = value
                self._write_manifest(changed)
                with self.assertRaisesRegex(
                    evidence.GroupedSplitEvidenceBuildError,
                    "invalid WO-12 split contract",
                ):
                    self._build(
                        expected_layout_manifest_sha256=self._sha(
                            self.manifest
                        )
                    )
                self._write_manifest(self.manifest_payload)

    def test_manifest_groups_are_recomputed_from_exact_pdf_bytes(self) -> None:
        changed = json.loads(json.dumps(self.manifest_payload))
        changed["cases"][0]["layout_group"] = (
            "page-count-99__ink-bucket-99"
        )
        self._write_manifest(changed)

        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "does not match recomputed label-blind bindings",
        ) as captured:
            self._build(
                expected_layout_manifest_sha256=self._sha(
                    self.manifest
                )
            )
        message = str(captured.exception)
        self.assertNotIn(self.case_ids[0], message)
        self.assertNotIn(str(self.manifest), message)

    def test_manifest_renderer_versions_are_exactly_recomputed(self) -> None:
        changed = json.loads(json.dumps(self.manifest_payload))
        changed["layout_signature"]["pillow_version"] = (
            "caller-invented-version"
        )
        self._write_manifest(changed)

        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "does not match recomputed label-blind bindings",
        ):
            self._build(
                expected_layout_manifest_sha256=self._sha(
                    self.manifest
                )
            )

    def test_input_population_must_match_even_with_matching_tree_digest(self) -> None:
        original = self.input_dir / f"{self.case_ids[-1]}.pdf"
        replacement = self.input_dir / "MIB-999999.pdf"
        original.rename(replacement)
        changed_input_sha = evidence._input_tree_sha256(
            tuple(sorted(self.input_dir.iterdir()))
        )

        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "input population does not match",
        ):
            self._build(expected_input_tree_sha256=changed_input_sha)

    def test_input_directory_rejects_non_pdf_entries(self) -> None:
        (self.input_dir / "labels.csv").write_text(
            "identity,label\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "PDF files only",
        ):
            self._build()

    def test_taint_registry_requires_generic_public_cohort_token(self) -> None:
        other = self.root / "other-taint.jsonl"
        CanonicalHashChainStore(other).append(
            {
                "event": "taint",
                "group_id": self.group_ids[0],
                "reason": "local exposure",
                "source": "test",
            }
        )
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "lacks the required generic public-cohort token",
        ):
            self._build(
                taint_registry_path=other,
                expected_taint_registry_sha256=self._sha(other),
            )

    def test_generic_taint_token_never_claims_exact_population(self) -> None:
        generic = self.root / "generic-taint.jsonl"
        CanonicalHashChainStore(generic).append(
            {
                "event": "taint",
                "group_id": "public-labeled-cohort",
                "reason": "arbitrary generic marker without a digest",
                "source": "unbound test fixture",
            }
        )

        aggregate = self._build(
            taint_registry_path=generic,
            expected_taint_registry_sha256=self._sha(generic),
        )
        serialized = canonical_json(aggregate)
        markdown = evidence.render_aggregate_markdown(aggregate)

        self.assertTrue(
            aggregate["checks"][
                "whole_public_cohort_taint_token_present"
            ]
        )
        self.assertNotIn("whole_cohort_tainted", serialized)
        self.assertNotIn("taint_attested", serialized)
        self.assertNotIn("entire public labeled cohort is already tainted", markdown)
        self.assertIn("not bound to the layout-manifest", markdown)
        self.assertIn("does not attest that it identifies", markdown)

    def test_taint_registry_rejects_malformed_payload_types_and_shape(
        self,
    ) -> None:
        malformed_payloads = (
            {
                "event": "taint",
                "group_id": "public-labeled-cohort",
                "reason": 7,
                "source": "test",
            },
            {
                "event": "taint",
                "group_id": "public-labeled-cohort",
                "reason": "test",
                "source": True,
            },
            {
                "event": "taint",
                "group_id": "public-labeled-cohort",
                "reason": "test",
                "source": "test",
                "extra": "not allowed",
            },
        )
        for index, payload in enumerate(malformed_payloads):
            with self.subTest(index=index):
                malformed = self.root / f"malformed-taint-{index}.jsonl"
                CanonicalHashChainStore(malformed).append(payload)
                with self.assertRaisesRegex(
                    evidence.GroupedSplitEvidenceBuildError,
                    "payload schema is invalid",
                ):
                    self._build(
                        taint_registry_path=malformed,
                        expected_taint_registry_sha256=self._sha(
                            malformed
                        ),
                    )

    def test_taint_registry_hash_chain_must_be_canonical(self) -> None:
        record = self.taint_registry.read_text(encoding="utf-8").strip()
        parsed = json.loads(record)
        self.taint_registry.write_text(
            json.dumps(parsed, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "canonical hash-chain verification",
        ):
            self._build(
                expected_taint_registry_sha256=self._sha(
                    self.taint_registry
                )
            )

    def test_matching_layout_group_taints_are_really_excluded(self) -> None:
        CanonicalHashChainStore(self.taint_registry).append(
            {
                "event": "taint",
                "group_id": self.group_ids[0],
                "reason": "declared layout exposure",
                "source": "test fixture",
            }
        )

        aggregate = self._build(
            expected_taint_registry_sha256=self._sha(
                self.taint_registry
            )
        )

        self.assertEqual(
            aggregate["counts"]["matching_layout_taint_group_count"],
            1,
        )
        self.assertEqual(
            aggregate["counts"]["matching_layout_taint_record_count"],
            2,
        )
        self.assertNotIn(
            self.group_ids[0],
            canonical_json(aggregate),
        )

    def test_split_contract_detects_leakage_and_incomplete_coordinates(self) -> None:
        manifest = evidence._strict_freezer_manifest(
            self.manifest
        ).manifest
        self.assertFalse(manifest.frozen_before_scoring)
        manager = RepeatedGroupedSplitManager(
            seed=manifest.split_seed,
            repeats=3,
            folds=5,
        )
        splits = manager.split_groups(manifest.groups)
        evidence._verify_split_contract(manifest.groups, splits)

        first = splits[0]
        leaking = replace(
            first,
            tuning_groups=first.tuning_groups
            + (first.validation_groups[0],),
        )
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "exclusivity",
        ):
            evidence._verify_split_contract(
                manifest.groups,
                (leaking,) + splits[1:],
            )
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "exactly three repeats of five folds",
        ):
            evidence._verify_split_contract(
                manifest.groups,
                splits[:-1],
            )

    def test_split_contract_detects_population_coverage_failure(self) -> None:
        manifest = evidence._strict_freezer_manifest(
            self.manifest
        ).manifest
        manager = RepeatedGroupedSplitManager(
            seed=manifest.split_seed,
            repeats=3,
            folds=5,
        )
        splits = list(manager.split_groups(manifest.groups))
        first = splits[0]
        moved_group = first.validation_groups[0]
        moved_records = tuple(manifest.groups[moved_group])
        splits[0] = replace(
            first,
            tuning_groups=tuple(
                sorted(first.tuning_groups + (moved_group,))
            ),
            validation_groups=tuple(
                group
                for group in first.validation_groups
                if group != moved_group
            ),
            tuning_case_ids=tuple(
                sorted(first.tuning_case_ids + moved_records)
            ),
            validation_case_ids=tuple(
                case_id
                for case_id in first.validation_case_ids
                if case_id not in moved_records
            ),
        )

        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "covered exactly once",
        ):
            evidence._verify_split_contract(
                manifest.groups,
                tuple(splits),
            )

    def test_synthetic_exclusion_probe_detects_ignored_taint_argument(self) -> None:
        real_manager = evidence.RepeatedGroupedSplitManager

        class IgnoringTaintManager:
            def __init__(self, **kwargs: object) -> None:
                self.delegate = real_manager(**kwargs)

            def split_groups(
                self,
                groups: dict[str, tuple[str, ...]],
                *,
                tainted_groups: tuple[str, ...] = (),
            ):
                del tainted_groups
                return self.delegate.split_groups(groups)

        with mock.patch.object(
            evidence,
            "RepeatedGroupedSplitManager",
            IgnoringTaintManager,
        ):
            with self.assertRaisesRegex(
                evidence.GroupedSplitEvidenceBuildError,
                "exclusivity",
            ):
                self._build()

    def test_nondeterministic_manager_is_rejected(self) -> None:
        real_manager = evidence.RepeatedGroupedSplitManager

        class NondeterministicManager:
            def __init__(self, **kwargs: object) -> None:
                self.delegate = real_manager(**kwargs)
                self.calls = 0

            def split_groups(self, *args: object, **kwargs: object):
                self.calls += 1
                result = self.delegate.split_groups(*args, **kwargs)
                return result if self.calls == 1 else tuple(reversed(result))

        with mock.patch.object(
            evidence,
            "RepeatedGroupedSplitManager",
            NondeterministicManager,
        ):
            with self.assertRaisesRegex(
                evidence.GroupedSplitEvidenceBuildError,
                "not deterministic",
            ):
                self._build()

    def test_checkout_verifier_requires_matching_clean_tracked_revision(self) -> None:
        def result(
            args: list[str],
            *,
            stdout: str | bytes,
            returncode: int = 0,
        ) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=args,
                returncode=returncode,
                stdout=stdout,
                stderr="" if isinstance(stdout, str) else b"",
            )

        repository = self.root / "checkout"
        producer_bytes = {}
        imported_modules = {}
        for producer_path in evidence._PRODUCER_PATHS:
            path = repository / producer_path
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"tracked:{producer_path}\n".encode("utf-8")
            path.write_bytes(content)
            producer_bytes[producer_path] = content
            if producer_path != "devtools/grouped_split_evidence.py":
                imported_modules[producer_path] = types.SimpleNamespace(
                    __file__=str(path)
                )
        clean_results = [
            result([], stdout=SOURCE_REVISION + "\n"),
            result([], stdout=""),
            *(
                result([], stdout=producer_bytes[path])
                for path in evidence._PRODUCER_PATHS
            ),
        ]
        tool_path = repository / "devtools/grouped_split_evidence.py"
        with mock.patch.object(
            evidence,
            "__file__",
            str(tool_path),
        ), mock.patch.object(
            evidence,
            "_IMPORTED_PRODUCER_MODULES",
            imported_modules,
        ), mock.patch.object(
            evidence.subprocess,
            "run",
            side_effect=clean_results,
        ) as runner, mock.patch.dict(
            evidence.os.environ,
            {
                "GIT_DIR": "/attacker",
                "GIT_WORK_TREE": "/attacker",
                "PATH": "/attacker",
            },
        ):
            evidence.verify_clean_source_checkout(
                repository,
                SOURCE_REVISION,
            )
        self.assertEqual(runner.call_count, 6)
        for call in runner.call_args_list:
            self.assertEqual(
                call.args[0][0],
                str(evidence.GIT_EXECUTABLE),
            )
            environment = call.kwargs["env"]
            self.assertFalse(
                any(key.startswith("GIT_") for key in environment)
            )
        self.assertEqual(
            set(evidence._PRODUCER_PATHS),
            {
                "devtools/grouped_split_evidence.py",
                "devtools/__init__.py",
                "devtools/layout_manifest_freezer.py",
                "devtools/experiment_control.py",
            },
        )

        failures = (
            (
                [
                    result([], stdout="b" * 40 + "\n"),
                    clean_results[1],
                ],
                "does not match checkout HEAD",
            ),
            (
                [
                    clean_results[0],
                    result([], stdout=" M hidden-change.py\n"),
                ],
                "working tree modifications",
            ),
            (
                [
                    clean_results[0],
                    clean_results[1],
                    result([], stdout=b"different bytes"),
                ],
                "producer source bytes do not match",
            ),
        )
        for results, message in failures:
            with self.subTest(message=message), mock.patch.object(
                evidence,
                "__file__",
                str(tool_path),
            ), mock.patch.object(
                evidence,
                "_IMPORTED_PRODUCER_MODULES",
                imported_modules,
            ), mock.patch.object(
                evidence.subprocess,
                "run",
                side_effect=results,
            ):
                with self.assertRaisesRegex(
                    evidence.GroupedSplitEvidenceBuildError,
                    message,
                ):
                    evidence.verify_clean_source_checkout(
                        repository,
                        SOURCE_REVISION,
                    )

    def test_builder_has_no_caller_checkout_verifier_bypass(self) -> None:
        self.assertNotIn(
            "checkout_verifier",
            inspect.signature(
                evidence.build_aggregate_evidence
            ).parameters,
        )

    def test_cli_writes_canonical_json_and_identity_free_markdown(self) -> None:
        output_json = self.root / "aggregate.json"
        output_markdown = self.root / "aggregate.md"
        argv = [
            "--layout-manifest",
            str(self.manifest),
            "--expected-layout-manifest-sha256",
            self._sha(self.manifest),
            "--input-dir",
            str(self.input_dir),
            "--expected-input-tree-sha256",
            self.expected_input_sha,
            "--taint-registry",
            str(self.taint_registry),
            "--expected-taint-registry-sha256",
            self._sha(self.taint_registry),
            "--source-revision-sha",
            SOURCE_REVISION,
            "--tool-source-sha256",
            self.tool_sha,
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            evidence,
            "verify_clean_source_checkout",
            return_value=None,
        ), self._rendering_patches(), contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(
            stderr
        ):
            status = evidence.main(argv)

        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("passed (3x5)", stdout.getvalue())
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(
            output_json.read_text(encoding="utf-8"),
            canonical_json(payload) + "\n",
        )
        markdown = output_markdown.read_text(encoding="utf-8")
        for content in (
            output_json.read_text(encoding="utf-8"),
            markdown,
        ):
            self.assertNotIn("MIB-", content)
            self.assertNotIn("page-count-", content)
            self.assertNotIn(".pdf", content.casefold())
            self.assertNotIn(str(self.root), content)

    def test_cli_failure_does_not_write_outputs(self) -> None:
        output_json = self.root / "aggregate.json"
        output_markdown = self.root / "aggregate.md"
        argv = [
            "--layout-manifest",
            str(self.manifest),
            "--expected-layout-manifest-sha256",
            "0" * 64,
            "--input-dir",
            str(self.input_dir),
            "--expected-input-tree-sha256",
            self.expected_input_sha,
            "--taint-registry",
            str(self.taint_registry),
            "--expected-taint-registry-sha256",
            self._sha(self.taint_registry),
            "--source-revision-sha",
            SOURCE_REVISION,
            "--expected-tool-source-sha256",
            self.tool_sha,
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
        with mock.patch.object(
            evidence,
            "verify_clean_source_checkout",
            return_value=None,
        ), self._rendering_patches(), contextlib.redirect_stdout(
            io.StringIO()
        ), (
            contextlib.redirect_stderr(io.StringIO())
        ):
            status = evidence.main(argv)

        self.assertEqual(status, 1)
        self.assertFalse(output_json.exists())
        self.assertFalse(output_markdown.exists())

    def test_cli_error_never_echoes_identity_or_paths(self) -> None:
        loop = self.root / "MIB-999999-manifest.json"
        try:
            loop.symlink_to(loop)
        except OSError:
            self.skipTest("symlinks unavailable")
        output_json = self.root / "aggregate.json"
        output_markdown = self.root / "aggregate.md"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = evidence.main(
                [
                    "--layout-manifest",
                    str(loop),
                    "--expected-layout-manifest-sha256",
                    "0" * 64,
                    "--input-dir",
                    str(self.input_dir),
                    "--expected-input-tree-sha256",
                    self.expected_input_sha,
                    "--taint-registry",
                    str(self.taint_registry),
                    "--expected-taint-registry-sha256",
                    self._sha(self.taint_registry),
                    "--source-revision-sha",
                    SOURCE_REVISION,
                    "--expected-tool-source-sha256",
                    self.tool_sha,
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_markdown),
                ]
            )

        self.assertEqual(status, 1)
        self.assertNotIn("MIB-", stderr.getvalue())
        self.assertNotIn(str(self.root), stderr.getvalue())

    def test_outputs_cannot_overwrite_inputs_or_enter_input_tree(self) -> None:
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "must not overlap",
        ):
            evidence._validate_output_paths(
                output_json=self.root / "same",
                output_markdown=self.root / "same",
                layout_manifest=self.manifest,
                input_dir=self.input_dir,
                taint_registry=self.taint_registry,
            )
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "must not overwrite",
        ):
            evidence._validate_output_paths(
                output_json=self.manifest,
                output_markdown=self.root / "report.md",
                layout_manifest=self.manifest,
                input_dir=self.input_dir,
                taint_registry=self.taint_registry,
            )
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "outside the PDF input directory",
        ):
            evidence._validate_output_paths(
                output_json=self.input_dir / "report.json",
                output_markdown=self.root / "report.md",
                layout_manifest=self.manifest,
                input_dir=self.input_dir,
                taint_registry=self.taint_registry,
            )

    def test_output_pair_is_create_once_idempotent_and_recovers_partial(self) -> None:
        output_json = self.root / "pair.json"
        output_markdown = self.root / "pair.md"
        json_content = '{"status":"passed"}\n'
        markdown_content = "# Passed\n"

        evidence._write_output_pair(
            output_json,
            json_content,
            output_markdown,
            markdown_content,
        )
        json_stat = output_json.stat()
        markdown_stat = output_markdown.stat()
        evidence._write_output_pair(
            output_json,
            json_content,
            output_markdown,
            markdown_content,
        )
        self.assertEqual(output_json.stat().st_ino, json_stat.st_ino)
        self.assertEqual(output_json.stat().st_mtime_ns, json_stat.st_mtime_ns)
        self.assertEqual(
            output_markdown.stat().st_ino,
            markdown_stat.st_ino,
        )

        output_markdown.unlink()
        evidence._write_output_pair(
            output_json,
            json_content,
            output_markdown,
            markdown_content,
        )
        self.assertEqual(output_json.stat().st_ino, json_stat.st_ino)
        self.assertEqual(
            output_markdown.read_text(encoding="utf-8"),
            markdown_content,
        )

    def test_output_pair_never_overwrites_a_conflict(self) -> None:
        output_json = self.root / "pair.json"
        output_markdown = self.root / "pair.md"
        output_json.write_text("conflict\n", encoding="utf-8")
        original_stat = output_json.stat()

        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "different bytes",
        ):
            evidence._write_output_pair(
                output_json,
                '{"status":"passed"}\n',
                output_markdown,
                "# Passed\n",
            )

        self.assertEqual(
            output_json.read_text(encoding="utf-8"),
            "conflict\n",
        )
        self.assertEqual(output_json.stat().st_ino, original_stat.st_ino)
        self.assertFalse(output_markdown.exists())

    def test_output_pair_rolls_back_link_after_parent_is_moved(self) -> None:
        output_parent = self.root / "safe-output"
        output_parent.mkdir()
        output_json = output_parent / "pair.json"
        output_markdown = output_parent / "pair.md"
        relocated = self.input_dir / "relocated-output"
        real_link = evidence.manifest_freezer.os.link

        def link_then_move(*args, **kwargs):
            result = real_link(*args, **kwargs)
            output_parent.rename(relocated)
            return result

        with mock.patch.object(
            evidence.manifest_freezer.os,
            "link",
            side_effect=link_then_move,
        ), self.assertRaises(
            evidence.GroupedSplitEvidenceBuildError
        ):
            evidence._write_output_pair(
                output_json,
                '{"status":"passed"}\n',
                output_markdown,
                "# Passed\n",
            )

        self.assertTrue(relocated.is_dir())
        self.assertFalse((relocated / output_json.name).exists())
        self.assertFalse((relocated / output_markdown.name).exists())
        self.assertEqual(tuple(relocated.iterdir()), ())

    def test_output_paths_reject_ancestor_and_symlink_aliases(self) -> None:
        ancestor = self.root / "ancestor"
        ancestor.mkdir()
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "must not overlap",
        ):
            evidence._validate_output_paths(
                output_json=ancestor,
                output_markdown=ancestor / "report.md",
                layout_manifest=self.manifest,
                input_dir=self.input_dir,
                taint_registry=self.taint_registry,
            )

        real_parent = self.root / "real-output"
        real_parent.mkdir()
        alias_parent = self.root / "output-alias"
        try:
            alias_parent.symlink_to(real_parent, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(
            evidence.GroupedSplitEvidenceBuildError,
            "must not contain symlinks",
        ):
            evidence._validate_output_paths(
                output_json=alias_parent / "report.json",
                output_markdown=real_parent / "report.md",
                layout_manifest=self.manifest,
                input_dir=self.input_dir,
                taint_registry=self.taint_registry,
            )

    def test_cli_uses_canonical_outputs_after_ancestor_alias_swap(
        self,
    ) -> None:
        aggregate = self._build()
        safe_nested = self.root / "safe-output" / "nested"
        safe_nested.mkdir(parents=True)
        input_nested = self.input_dir / "nested"
        input_nested.mkdir()
        alias = self.root / "output-alias"
        try:
            alias.symlink_to(
                safe_nested.parent,
                target_is_directory=True,
            )
        except OSError:
            self.skipTest("symlinks unavailable")
        requested_json = alias / "nested" / "report.json"
        requested_markdown = alias / "nested" / "report.md"
        safe_json = safe_nested / "report.json"
        safe_markdown = safe_nested / "report.md"
        original_validate = evidence._validate_output_paths
        validation_count = 0

        def validate_and_swap(**kwargs: Path):
            nonlocal validation_count
            resolved = original_validate(**kwargs)
            validation_count += 1
            if validation_count == 2:
                alias.unlink()
                alias.symlink_to(
                    self.input_dir,
                    target_is_directory=True,
                )
            return resolved

        argv = [
            "--layout-manifest",
            str(self.manifest),
            "--expected-layout-manifest-sha256",
            self._sha(self.manifest),
            "--input-dir",
            str(self.input_dir),
            "--expected-input-tree-sha256",
            self.expected_input_sha,
            "--taint-registry",
            str(self.taint_registry),
            "--expected-taint-registry-sha256",
            self._sha(self.taint_registry),
            "--source-revision-sha",
            SOURCE_REVISION,
            "--expected-tool-source-sha256",
            self.tool_sha,
            "--output-json",
            str(requested_json),
            "--output-markdown",
            str(requested_markdown),
        ]
        with mock.patch.object(
            evidence,
            "build_aggregate_evidence",
            return_value=aggregate,
        ), mock.patch.object(
            evidence,
            "_validate_output_paths",
            side_effect=validate_and_swap,
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ):
            status = evidence.main(argv)

        self.assertEqual(status, 0)
        self.assertEqual(validation_count, 2)
        self.assertTrue(safe_json.is_file())
        self.assertTrue(safe_markdown.is_file())
        self.assertFalse((input_nested / "report.json").exists())
        self.assertFalse((input_nested / "report.md").exists())

    def test_markdown_rejects_identity_bearing_mutation(self) -> None:
        aggregate = self._build()
        aggregate["fold_metrics"]["MIB-000001"] = {
            "record_count": 1,
        }
        with self.assertRaises(
            evidence.GroupedSplitEvidenceBuildError
        ):
            evidence.render_aggregate_markdown(aggregate)


if __name__ == "__main__":
    unittest.main()
