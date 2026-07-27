#!/usr/bin/env python3
"""Build identity-free WO-12 evidence for grouped-split mechanics.

This development-only tool reads the external identity-bearing layout manifest
created by :mod:`devtools.layout_manifest_freezer`.  It proves deterministic
3x5 assignment, whole-group separation, exact per-repeat population coverage,
and the split manager's taint-exclusion mechanism.  It does not score records
and it does not describe the public labeled cohort as protected or unseen.

Only aggregate counts and cryptographic bindings are written.  Case identities,
layout-group identities, and input filenames remain in external inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import devtools as devtools_package  # noqa: E402
from devtools import layout_manifest_freezer as manifest_freezer  # noqa: E402
from devtools import experiment_control as experiment_control_module  # noqa: E402
from devtools.experiment_control import (  # noqa: E402
    CanonicalHashChainStore,
    ExperimentControlError,
    GroupedFold,
    RepeatedGroupedSplitManager,
    canonical_json,
    require_aggregate_only,
)


EVIDENCE_CLASS = "public_grouped_robustness_not_unseen"
GENERIC_PUBLIC_COHORT_TAINT_TOKEN = "public-labeled-cohort"
REPEATS = 3
FOLDS = 5
GIT_EXECUTABLE = Path("/usr/bin/git")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_LAYOUT_GROUP_RE = re.compile(
    r"page-count-[0-9]{2,}__ink-bucket-[0-9]{2,}"
)
_TAINT_PAYLOAD_KEYS = frozenset(
    {"event", "group_id", "reason", "source"}
)
_MANIFEST_ROOT_KEYS = frozenset(
    {
        "cases",
        "folds",
        "label_blind_construction",
        "layout_signature",
        "repeats",
        "schema",
        "split_seed",
    }
)
_LAYOUT_SIGNATURE_KEYS = frozenset(
    {
        "first_page_grayscale_ink_bucket_width",
        "first_page_grayscale_ink_pixel_threshold_exclusive",
        "first_page_render_height",
        "first_page_render_width",
        "inputs",
        "pillow_version",
        "pdfium_version",
        "pypdfium2_version",
        "version",
    }
)
_AGGREGATE_ROOT_KEYS = frozenset(
    {
        "checks",
        "comparison_scope",
        "counts",
        "evaluation_mode",
        "evidence_label",
        "expected_record_count",
        "fold_count",
        "fold_metrics",
        "input_pdf_count",
        "input_tree_sha256",
        "layout_group_count",
        "layout_manifest_sha256",
        "repeat_count",
        "source_revision_sha",
        "split_count",
        "status",
        "taint_registry_head_sha256",
        "taint_registry_sha256",
        "tool_source_sha256",
    }
)
_AGGREGATE_CHECK_KEYS = frozenset(
    {
        "group_exclusive",
        "input_population_matches_manifest",
        "input_tree_digest_verified",
        "manifest_canonical_freezer_output",
        "manifest_declares_label_blind_construction",
        "manifest_digest_verified",
        "no_unseen_or_protected_claim",
        "population_coverage_once_per_repeat",
        "public_robustness_not_unseen",
        "source_revision_verified",
        "split_deterministic",
        "synthetic_exclusion_mechanics_verified",
        "tool_source_verified",
        "whole_public_cohort_taint_token_present",
    }
)
_AGGREGATE_COUNT_KEYS = frozenset(
    {
        "matching_layout_taint_group_count",
        "matching_layout_taint_record_count",
        "synthetic_exclusion_group_count",
        "synthetic_exclusion_record_count",
        "taint_event_count",
        "validation_group_assignment_count",
        "validation_record_assignment_count",
        "whole_public_cohort_taint_token_event_count",
    }
)
_FOLD_METRIC_KEYS = frozenset(
    {
        "tuning_group_count",
        "tuning_record_count",
        "validation_group_count",
        "validation_record_count",
    }
)
_PRODUCER_PATHS = (
    "devtools/grouped_split_evidence.py",
    "devtools/__init__.py",
    "devtools/layout_manifest_freezer.py",
    "devtools/experiment_control.py",
)
_IMPORTED_PRODUCER_MODULES = {
    "devtools/__init__.py": devtools_package,
    "devtools/layout_manifest_freezer.py": manifest_freezer,
    "devtools/experiment_control.py": experiment_control_module,
}


class GroupedSplitEvidenceBuildError(ExperimentControlError):
    """The requested WO-12 split demonstration could not be proven."""


@dataclass(frozen=True)
class TaintSnapshot:
    """Aggregate facts from one exact, verified taint-registry snapshot."""

    sha256: str
    head_sha256: str
    event_count: int
    whole_public_cohort_taint_token_event_count: int
    tainted_group_ids: frozenset[str]


@dataclass(frozen=True)
class FrozenLayoutManifest:
    """Private WO-12 groups with an explicitly unproved temporal status."""

    groups: Mapping[str, tuple[str, ...]]
    split_seed: str
    sha256: str
    frozen_before_scoring: bool

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                case_id
                for case_ids in self.groups.values()
                for case_id in case_ids
            )
        )


@dataclass(frozen=True)
class ManifestSnapshot:
    """One exact manifest byte snapshot plus its validated private mapping."""

    manifest: FrozenLayoutManifest
    raw: Mapping[str, Any]


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GroupedSplitEvidenceBuildError(
            "a required evidence input could not be hashed"
        ) from exc
    return digest.hexdigest()


def _read_bound_regular_file(path: Path | str, *, label: str) -> bytes:
    """Read one no-follow file descriptor and verify its path binding."""

    requested = Path(path)
    descriptor = -1
    try:
        resolved = requested.resolve(strict=True)
        if requested.is_symlink():
            raise GroupedSplitEvidenceBuildError(
                f"{label} path must not contain symlinks"
            )
        descriptor = os.open(
            str(resolved),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        entry = os.stat(resolved, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (entry.st_dev, entry.st_ino)
        ):
            raise GroupedSplitEvidenceBuildError(
                f"{label} must be a regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final = os.fstat(descriptor)
        final_entry = os.stat(resolved, follow_symlinks=False)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (
            final_entry.st_dev,
            final_entry.st_ino,
        ):
            raise GroupedSplitEvidenceBuildError(
                f"{label} changed while it was being read"
            )
        return b"".join(chunks)
    except GroupedSplitEvidenceBuildError:
        raise
    except (OSError, RuntimeError) as exc:
        raise GroupedSplitEvidenceBuildError(
            f"{label} could not be read safely"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_sha256(label: str, value: object) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise GroupedSplitEvidenceBuildError(
            f"{label} must be a full SHA-256 digest"
        )
    return normalized


def _require_revision(value: object) -> str:
    normalized = str(value).strip().lower()
    if not _GIT_COMMIT_RE.fullmatch(normalized):
        raise GroupedSplitEvidenceBuildError(
            "source_revision_sha must be a full Git commit SHA"
        )
    return normalized


def verify_clean_source_checkout(
    repo_root: Path, expected_source_revision_sha: str
) -> None:
    """Require clean HEAD and exact HEAD bytes for every producer source."""

    expected = _require_revision(expected_source_revision_sha)
    sanitized_environment = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("GIT_")
    }
    try:
        head = subprocess.run(
            [
                str(GIT_EXECUTABLE),
                "-C",
                str(repo_root),
                "rev-parse",
                "HEAD",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_environment,
        )
        status_result = subprocess.run(
            [
                str(GIT_EXECUTABLE),
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_environment,
        )
    except OSError as exc:
        raise GroupedSplitEvidenceBuildError(
            "source checkout could not be verified"
        ) from exc
    if head.returncode != 0 or status_result.returncode != 0:
        raise GroupedSplitEvidenceBuildError(
            "source checkout could not be verified"
        )
    if head.stdout.strip().lower() != expected:
        raise GroupedSplitEvidenceBuildError(
            "source revision does not match checkout HEAD"
        )
    if status_result.stdout.strip():
        raise GroupedSplitEvidenceBuildError(
            "source checkout has working tree modifications"
        )
    for producer_path in _PRODUCER_PATHS:
        worktree_path = repo_root / PurePosixPath(producer_path)
        try:
            if producer_path == "devtools/grouped_split_evidence.py":
                imported_path = Path(__file__).resolve()
            else:
                module_file = getattr(
                    _IMPORTED_PRODUCER_MODULES[producer_path],
                    "__file__",
                    None,
                )
                if not module_file:
                    raise GroupedSplitEvidenceBuildError(
                        "an imported producer module has no source path"
                    )
                imported_path = Path(module_file).resolve()
            if imported_path != worktree_path.resolve():
                raise GroupedSplitEvidenceBuildError(
                    "an imported producer module is outside the verified "
                    "repository"
                )
            if worktree_path.is_symlink() or not worktree_path.is_file():
                raise GroupedSplitEvidenceBuildError(
                    "a producer source is missing or not a regular file"
                )
            worktree_bytes = worktree_path.read_bytes()
            committed = subprocess.run(
                [
                    str(GIT_EXECUTABLE),
                    "-C",
                    str(repo_root),
                    "show",
                    f"HEAD:{producer_path}",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=sanitized_environment,
            )
        except (OSError, RuntimeError) as exc:
            raise GroupedSplitEvidenceBuildError(
                "producer source bytes could not be verified"
            ) from exc
        if (
            committed.returncode != 0
            or committed.stdout != worktree_bytes
        ):
            raise GroupedSplitEvidenceBuildError(
                "producer source bytes do not match the named revision"
            )


def _strict_freezer_manifest(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> ManifestSnapshot:
    """Parse and validate exactly one immutable manifest byte snapshot."""

    content = _read_bound_regular_file(
        path,
        label="layout manifest",
    )
    digest = hashlib.sha256(content).hexdigest()
    if (
        expected_sha256 is not None
        and digest
        != _require_sha256(
            "expected_layout_manifest_sha256", expected_sha256
        )
    ):
        raise GroupedSplitEvidenceBuildError(
            "layout manifest does not match the expected SHA-256 digest"
        )
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroupedSplitEvidenceBuildError(
            "layout manifest must be canonical UTF-8 JSON"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_ROOT_KEYS:
        raise GroupedSplitEvidenceBuildError(
            "layout manifest does not match the freezer schema"
        )
    if content != (canonical_json(raw) + "\n").encode("utf-8"):
        raise GroupedSplitEvidenceBuildError(
            "layout manifest bytes are not canonical freezer output"
        )
    if (
        raw["schema"] != manifest_freezer.SCHEMA
        or raw["repeats"] != REPEATS
        or raw["folds"] != FOLDS
        or raw["label_blind_construction"] is not True
        or not isinstance(raw["split_seed"], str)
        or not raw["split_seed"]
        or raw["split_seed"] != raw["split_seed"].strip()
    ):
        raise GroupedSplitEvidenceBuildError(
            "layout manifest has an invalid WO-12 split contract"
        )

    signature = raw["layout_signature"]
    expected_signature_values = {
        "first_page_grayscale_ink_bucket_width": (
            manifest_freezer.INK_BUCKET_WIDTH
        ),
        "first_page_grayscale_ink_pixel_threshold_exclusive": (
            manifest_freezer.INK_THRESHOLD
        ),
        "first_page_render_height": manifest_freezer.RENDER_HEIGHT,
        "first_page_render_width": manifest_freezer.RENDER_WIDTH,
        "inputs": ["pdf_page_count", "first_page_rendered_pixels"],
        "version": manifest_freezer.SIGNATURE_VERSION,
    }
    if (
        not isinstance(signature, dict)
        or set(signature) != _LAYOUT_SIGNATURE_KEYS
        or any(
            signature.get(key) != value
            for key, value in expected_signature_values.items()
        )
        or any(
            not isinstance(signature.get(key), str)
            or not signature[key].strip()
            for key in (
                "pillow_version",
                "pdfium_version",
                "pypdfium2_version",
            )
        )
    ):
        raise GroupedSplitEvidenceBuildError(
            "layout manifest has invalid label-blind signature metadata"
        )

    rows = raw["cases"]
    if (
        not isinstance(rows, list)
        or any(
            not isinstance(row, dict)
            or set(row) != {"case_id", "layout_group"}
            or not isinstance(row["case_id"], str)
            or manifest_freezer.CASE_ID_PATTERN.fullmatch(row["case_id"]) is None
            or not isinstance(row["layout_group"], str)
            or _LAYOUT_GROUP_RE.fullmatch(row["layout_group"]) is None
            for row in rows
        )
        or [row["case_id"] for row in rows]
        != sorted(row["case_id"] for row in rows)
    ):
        raise GroupedSplitEvidenceBuildError(
            "layout manifest case rows are not canonical freezer output"
        )
    groups: dict[str, list[str]] = {}
    seen_case_ids: set[str] = set()
    for row in rows:
        case_id = row["case_id"]
        group_id = row["layout_group"]
        if case_id in seen_case_ids:
            raise GroupedSplitEvidenceBuildError(
                "layout manifest case identities must occur exactly once"
            )
        seen_case_ids.add(case_id)
        groups.setdefault(group_id, []).append(case_id)
    if len(groups) < FOLDS:
        raise GroupedSplitEvidenceBuildError(
            "layout manifest must contain at least five layout groups"
        )
    manifest = FrozenLayoutManifest(
        groups={
            group_id: tuple(sorted(case_ids))
            for group_id, case_ids in sorted(groups.items())
        },
        split_seed=raw["split_seed"],
        sha256=digest,
        frozen_before_scoring=False,
    )
    return ManifestSnapshot(manifest=manifest, raw=raw)


def _input_tree_sha256(pdf_paths: Sequence[Path]) -> str:
    """Stream the project-standard name/content tree digest."""

    digest = hashlib.sha256()
    for path in sorted(
        pdf_paths,
        key=lambda item: (item.name.casefold(), item.name),
    ):
        name_bytes = path.name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _verify_input_tree_and_recomputed_manifest(
    input_dir: Path | str,
    snapshot: ManifestSnapshot,
    *,
    expected_sha256: str,
) -> str:
    """Stream, render, and bind one stable PDF snapshot at a time."""

    expected = _require_sha256(
        "expected_input_tree_sha256", expected_sha256
    )
    manifest = snapshot.manifest
    try:
        pdfs = manifest_freezer.discover_pdfs(
            Path(input_dir),
            expected_count=len(manifest.case_ids),
        )
    except ExperimentControlError as exc:
        raise GroupedSplitEvidenceBuildError(str(exc)) from exc
    except manifest_freezer.LayoutManifestFreezeError as exc:
        raise GroupedSplitEvidenceBuildError(str(exc)) from exc
    except OSError as exc:
        raise GroupedSplitEvidenceBuildError(
            "input tree could not be enumerated"
        ) from exc
    if tuple(path.stem for path in pdfs) != manifest.case_ids:
        raise GroupedSplitEvidenceBuildError(
            "input population does not match the frozen manifest"
        )
    with tempfile.TemporaryDirectory(
        prefix="mib-wo12-input-snapshot-",
        dir=str(manifest_freezer._external_temporary_root()),
    ) as temporary_name:
        snapshot_root = Path(temporary_name)
        tree_digest = hashlib.sha256()
        cases: list[dict[str, str]] = []
        for index, source_path in enumerate(pdfs):
            snapshot_path = snapshot_root / source_path.name
            try:
                manifest_freezer._copy_pdf_snapshot(
                    source_path,
                    snapshot_path,
                )
                name_bytes = source_path.name.encode("utf-8")
                tree_digest.update(
                    len(name_bytes).to_bytes(4, "big")
                )
                tree_digest.update(name_bytes)
                tree_digest.update(
                    bytes.fromhex(_sha256_file(snapshot_path))
                )
                page_count, ink_bucket = (
                    manifest_freezer.layout_signature(snapshot_path)
                )
            except GroupedSplitEvidenceBuildError:
                raise
            except Exception as exc:
                raise GroupedSplitEvidenceBuildError(
                    "input layout could not be recomputed at a canonical "
                    f"position {index + 1}"
                ) from exc
            finally:
                try:
                    snapshot_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise GroupedSplitEvidenceBuildError(
                        "temporary input snapshot could not be removed"
                    ) from exc
            cases.append(
                {
                    "case_id": source_path.stem,
                    "layout_group": (
                        f"page-count-{page_count:02d}__"
                        f"ink-bucket-{ink_bucket:02d}"
                    ),
                }
            )
    actual = tree_digest.hexdigest()
    if actual != expected:
        raise GroupedSplitEvidenceBuildError(
            "input tree does not match the expected SHA-256 digest"
        )
    try:
        versions = manifest_freezer._rendering_version_metadata()
    except Exception as exc:
        raise GroupedSplitEvidenceBuildError(
            "renderer version metadata could not be verified"
        ) from exc
    rebuilt = {
        "cases": cases,
        "folds": FOLDS,
        "label_blind_construction": True,
        "layout_signature": {
            "first_page_grayscale_ink_bucket_width": (
                manifest_freezer.INK_BUCKET_WIDTH
            ),
            "first_page_grayscale_ink_pixel_threshold_exclusive": (
                manifest_freezer.INK_THRESHOLD
            ),
            "first_page_render_height": manifest_freezer.RENDER_HEIGHT,
            "first_page_render_width": manifest_freezer.RENDER_WIDTH,
            "inputs": ["pdf_page_count", "first_page_rendered_pixels"],
            "pillow_version": versions["pillow_version"],
            "pdfium_version": versions["pdfium_version"],
            "pypdfium2_version": versions["pypdfium2_version"],
            "version": manifest_freezer.SIGNATURE_VERSION,
        },
        "repeats": REPEATS,
        "schema": manifest_freezer.SCHEMA,
        "split_seed": manifest.split_seed,
    }
    if canonical_json(rebuilt) != canonical_json(snapshot.raw):
        raise GroupedSplitEvidenceBuildError(
            "layout manifest does not match recomputed label-blind bindings"
        )
    return actual


def _load_taint_snapshot(
    path: Path | str, *, expected_sha256: str
) -> TaintSnapshot:
    expected = _require_sha256(
        "expected_taint_registry_sha256", expected_sha256
    )
    content = _read_bound_regular_file(
        path,
        label="taint registry",
    )
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise GroupedSplitEvidenceBuildError(
            "taint registry does not match the expected SHA-256 digest"
        )
    try:
        records = CanonicalHashChainStore._parse(
            content.decode("utf-8")
        )
    except (ExperimentControlError, UnicodeError) as exc:
        raise GroupedSplitEvidenceBuildError(
            "taint registry failed canonical hash-chain verification"
        ) from exc
    events: list[Mapping[str, str]] = []
    for record in records:
        payload = record["payload"]
        if (
            not isinstance(payload, dict)
            or set(payload) != _TAINT_PAYLOAD_KEYS
            or payload.get("event") != "taint"
            or any(
                type(payload.get(key)) is not str
                for key in ("event", "group_id", "reason", "source")
            )
            or any(
                not payload[key]
                or payload[key] != payload[key].strip()
                for key in ("group_id", "reason", "source")
            )
        ):
            raise GroupedSplitEvidenceBuildError(
                "taint registry payload schema is invalid"
            )
        events.append(payload)
    head = str(records[-1]["record_hash"]) if records else "0" * 64
    token_events = tuple(
        event
        for event in events
        if event["group_id"] == GENERIC_PUBLIC_COHORT_TAINT_TOKEN
    )
    if not token_events:
        raise GroupedSplitEvidenceBuildError(
            "taint registry lacks the required generic public-cohort token"
        )
    return TaintSnapshot(
        sha256=actual,
        head_sha256=head,
        event_count=len(events),
        whole_public_cohort_taint_token_event_count=len(token_events),
        tainted_group_ids=frozenset(
            event["group_id"] for event in events
        ),
    )


def _ordered_groups(
    groups: Mapping[str, Sequence[str]],
    *,
    reverse: bool = False,
) -> dict[str, tuple[str, ...]]:
    items = sorted(groups.items(), reverse=reverse)
    return {
        group_id: tuple(sorted(case_ids, reverse=reverse))
        for group_id, case_ids in items
    }


def _verify_split_contract(
    groups: Mapping[str, Sequence[str]],
    splits: Sequence[GroupedFold],
) -> dict[str, dict[str, int]]:
    """Verify the full split invariant set and return aggregate fold counts."""

    group_ids = frozenset(groups)
    case_owner: dict[str, str] = {}
    normalized: dict[str, frozenset[str]] = {}
    for group_id, raw_case_ids in groups.items():
        case_ids = frozenset(raw_case_ids)
        if not group_id or not case_ids:
            raise GroupedSplitEvidenceBuildError(
                "split verification received an empty group"
            )
        for case_id in case_ids:
            if case_id in case_owner:
                raise GroupedSplitEvidenceBuildError(
                    "split verification found a multiply owned record"
                )
            case_owner[case_id] = group_id
        normalized[group_id] = case_ids
    population = frozenset(case_owner)
    expected_coordinates = {
        (repeat, fold)
        for repeat in range(REPEATS)
        for fold in range(FOLDS)
    }
    by_coordinate: dict[tuple[int, int], GroupedFold] = {}
    for split in splits:
        coordinate = (split.repeat, split.fold)
        if coordinate in by_coordinate:
            raise GroupedSplitEvidenceBuildError(
                "split assignment contains a duplicate repeat/fold"
            )
        by_coordinate[coordinate] = split
    if set(by_coordinate) != expected_coordinates:
        raise GroupedSplitEvidenceBuildError(
            "split assignment is not exactly three repeats of five folds"
        )

    validation_group_frequency: Counter[tuple[int, str]] = Counter()
    validation_record_frequency: Counter[tuple[int, str]] = Counter()
    tuning_group_frequency: Counter[tuple[int, str]] = Counter()
    tuning_record_frequency: Counter[tuple[int, str]] = Counter()
    fold_metrics: dict[str, dict[str, int]] = {}
    for repeat, fold in sorted(expected_coordinates):
        split = by_coordinate[(repeat, fold)]
        tuning_groups = frozenset(split.tuning_groups)
        validation_groups = frozenset(split.validation_groups)
        tuning_records = frozenset(split.tuning_case_ids)
        validation_records = frozenset(split.validation_case_ids)
        if (
            len(tuning_groups) != len(split.tuning_groups)
            or len(validation_groups) != len(split.validation_groups)
            or len(tuning_records) != len(split.tuning_case_ids)
            or len(validation_records) != len(split.validation_case_ids)
            or tuning_groups & validation_groups
            or tuning_records & validation_records
            or tuning_groups | validation_groups != group_ids
            or tuning_records | validation_records != population
        ):
            raise GroupedSplitEvidenceBuildError(
                "split assignment violates group or record exclusivity"
            )
        expected_tuning_records = frozenset(
            case_id
            for group_id in tuning_groups
            for case_id in normalized[group_id]
        )
        expected_validation_records = frozenset(
            case_id
            for group_id in validation_groups
            for case_id in normalized[group_id]
        )
        if (
            tuning_records != expected_tuning_records
            or validation_records != expected_validation_records
        ):
            raise GroupedSplitEvidenceBuildError(
                "split assignment separates a record from its group"
            )
        for group_id in validation_groups:
            validation_group_frequency[(repeat, group_id)] += 1
        for case_id in validation_records:
            validation_record_frequency[(repeat, case_id)] += 1
        for group_id in tuning_groups:
            tuning_group_frequency[(repeat, group_id)] += 1
        for case_id in tuning_records:
            tuning_record_frequency[(repeat, case_id)] += 1
        fold_metrics[f"repeat_{repeat + 1}_fold_{fold + 1}"] = {
            "tuning_group_count": len(tuning_groups),
            "tuning_record_count": len(tuning_records),
            "validation_group_count": len(validation_groups),
            "validation_record_count": len(validation_records),
        }

    if any(
        validation_group_frequency[(repeat, group_id)] != 1
        or tuning_group_frequency[(repeat, group_id)] != FOLDS - 1
        for repeat in range(REPEATS)
        for group_id in group_ids
    ) or any(
        validation_record_frequency[(repeat, case_id)] != 1
        or tuning_record_frequency[(repeat, case_id)] != FOLDS - 1
        for repeat in range(REPEATS)
        for case_id in population
    ):
        raise GroupedSplitEvidenceBuildError(
            "split population is not covered exactly once per repeat"
        )
    return fold_metrics


def _synthetic_exclusion_probe(
    manager: RepeatedGroupedSplitManager,
    groups: Mapping[str, Sequence[str]],
    expected_splits: Sequence[GroupedFold],
) -> None:
    """Prove taint exclusion without implying a real public group is untainted."""

    synthetic_group = "__wo12_synthetic_exclusion_group__"
    while synthetic_group in groups:
        synthetic_group += "_"
    all_case_ids = {
        case_id for case_ids in groups.values() for case_id in case_ids
    }
    synthetic_case = "__wo12_synthetic_exclusion_record__"
    while synthetic_case in all_case_ids:
        synthetic_case += "_"
    probe_groups = _ordered_groups(groups)
    probe_groups[synthetic_group] = (synthetic_case,)
    probe = manager.split_groups(
        probe_groups,
        tainted_groups=(synthetic_group,),
    )
    _verify_split_contract(groups, probe)
    if tuple(probe) != tuple(expected_splits) or any(
        synthetic_group in split.tuning_groups
        or synthetic_group in split.validation_groups
        or synthetic_case in split.tuning_case_ids
        or synthetic_case in split.validation_case_ids
        for split in probe
    ):
        raise GroupedSplitEvidenceBuildError(
            "split manager failed the synthetic taint-exclusion probe"
        )


def _declared_layout_taint_probe(
    manager: RepeatedGroupedSplitManager,
    groups: Mapping[str, Sequence[str]],
    tainted_group_ids: frozenset[str],
) -> tuple[int, int]:
    """Exclude any registry taints that use the manifest's group namespace."""

    matching = frozenset(groups) & tainted_group_ids
    if not matching:
        return 0, 0
    eligible = {
        group_id: tuple(case_ids)
        for group_id, case_ids in groups.items()
        if group_id not in matching
    }
    if len(eligible) < FOLDS:
        raise GroupedSplitEvidenceBuildError(
            "declared layout-group taints leave too few groups for five folds"
        )
    try:
        splits = manager.split_groups(
            groups,
            tainted_groups=matching,
        )
    except ExperimentControlError as exc:
        raise GroupedSplitEvidenceBuildError(
            "split manager rejected declared layout-group taints"
        ) from exc
    _verify_split_contract(eligible, splits)
    tainted_records = frozenset(
        case_id
        for group_id in matching
        for case_id in groups[group_id]
    )
    if any(
        matching
        & (
            frozenset(split.tuning_groups)
            | frozenset(split.validation_groups)
        )
        or tainted_records
        & (
            frozenset(split.tuning_case_ids)
            | frozenset(split.validation_case_ids)
        )
        for split in splits
    ):
        raise GroupedSplitEvidenceBuildError(
            "split manager retained a declared layout-group taint"
        )
    return len(matching), len(tainted_records)


def _validate_aggregate_shape(aggregate: Mapping[str, Any]) -> None:
    """Reject additions or omissions before JSON or Markdown serialization."""

    require_aggregate_only(aggregate)
    expected_fold_keys = {
        f"repeat_{repeat}_fold_{fold}"
        for repeat in range(1, REPEATS + 1)
        for fold in range(1, FOLDS + 1)
    }
    checks = aggregate.get("checks")
    counts = aggregate.get("counts")
    fold_metrics = aggregate.get("fold_metrics")
    if (
        set(aggregate) != _AGGREGATE_ROOT_KEYS
        or not isinstance(checks, Mapping)
        or set(checks) != _AGGREGATE_CHECK_KEYS
        or any(value is not True for value in checks.values())
        or not isinstance(counts, Mapping)
        or set(counts) != _AGGREGATE_COUNT_KEYS
        or not isinstance(fold_metrics, Mapping)
        or set(fold_metrics) != expected_fold_keys
        or any(
            not isinstance(metrics, Mapping)
            or set(metrics) != _FOLD_METRIC_KEYS
            for metrics in fold_metrics.values()
        )
        or aggregate.get("comparison_scope") != EVIDENCE_CLASS
        or aggregate.get("evidence_label") != EVIDENCE_CLASS
        or aggregate.get("evaluation_mode") != "aggregate_only"
        or aggregate.get("status") != "passed"
        or aggregate.get("repeat_count") != REPEATS
        or aggregate.get("fold_count") != FOLDS
        or aggregate.get("split_count") != REPEATS * FOLDS
    ):
        raise GroupedSplitEvidenceBuildError(
            "aggregate evidence does not match the canonical WO-12 shape"
        )


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise GroupedSplitEvidenceBuildError(
            "an evidence path could not be resolved safely"
        ) from exc


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        _safe_resolve(path).relative_to(_safe_resolve(root))
    except ValueError:
        return False
    return True


def _require_external_identity_inputs(
    *,
    layout_manifest_path: Path | str,
    input_dir: Path | str,
) -> tuple[Path, Path]:
    manifest_path = Path(layout_manifest_path)
    pdf_root = Path(input_dir)
    try:
        manifest_resolved = manifest_path.resolve(strict=True)
        input_resolved = pdf_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GroupedSplitEvidenceBuildError(
            "identity-bearing inputs could not be resolved safely"
        ) from exc
    if (
        manifest_path.is_symlink()
        or pdf_root.is_symlink()
    ):
        raise GroupedSplitEvidenceBuildError(
            "identity-bearing input paths must not contain symlinks"
        )
    if _path_is_within(manifest_path, REPO_ROOT):
        raise GroupedSplitEvidenceBuildError(
            "identity-bearing layout manifest must be outside the repository"
        )
    return manifest_resolved, input_resolved


def build_aggregate_evidence(
    *,
    layout_manifest_path: Path | str,
    expected_layout_manifest_sha256: str,
    input_dir: Path | str,
    expected_input_tree_sha256: str,
    taint_registry_path: Path | str,
    expected_taint_registry_sha256: str,
    source_revision_sha: str,
    expected_tool_source_sha256: str,
) -> dict[str, Any]:
    """Build canonical, identity-free proof of the WO-12 split mechanics."""

    source_revision = _require_revision(source_revision_sha)
    expected_tool_sha = _require_sha256(
        "expected_tool_source_sha256", expected_tool_source_sha256
    )
    tool_sha_before = _sha256_file(Path(__file__))
    if tool_sha_before != expected_tool_sha:
        raise GroupedSplitEvidenceBuildError(
            "evidence tool source does not match the expected SHA-256 digest"
        )
    verify_clean_source_checkout(REPO_ROOT, source_revision)
    if _sha256_file(Path(__file__)) != expected_tool_sha:
        raise GroupedSplitEvidenceBuildError(
            "evidence tool source changed during checkout verification"
        )

    resolved_manifest_path, resolved_input_dir = (
        _require_external_identity_inputs(
            layout_manifest_path=layout_manifest_path,
            input_dir=input_dir,
        )
    )
    snapshot = _strict_freezer_manifest(
        resolved_manifest_path,
        expected_sha256=expected_layout_manifest_sha256,
    )
    manifest = snapshot.manifest
    input_tree_sha = _verify_input_tree_and_recomputed_manifest(
        resolved_input_dir,
        snapshot,
        expected_sha256=expected_input_tree_sha256,
    )
    if _safe_resolve(Path(input_dir)) != resolved_input_dir:
        raise GroupedSplitEvidenceBuildError(
            "PDF input directory binding changed during verification"
        )
    taint = _load_taint_snapshot(
        taint_registry_path,
        expected_sha256=expected_taint_registry_sha256,
    )

    manager = RepeatedGroupedSplitManager(
        seed=manifest.split_seed,
        repeats=REPEATS,
        folds=FOLDS,
    )
    try:
        first = manager.split_groups(_ordered_groups(manifest.groups))
        second = manager.split_groups(
            _ordered_groups(manifest.groups, reverse=True)
        )
    except ExperimentControlError as exc:
        raise GroupedSplitEvidenceBuildError(
            "split manager rejected the recomputed manifest"
        ) from exc
    if first != second:
        raise GroupedSplitEvidenceBuildError(
            "split assignment is not deterministic"
        )
    fold_metrics = _verify_split_contract(manifest.groups, first)
    _synthetic_exclusion_probe(manager, manifest.groups, first)
    matching_taint_groups, matching_taint_records = (
        _declared_layout_taint_probe(
            manager,
            manifest.groups,
            taint.tainted_group_ids,
        )
    )

    record_count = len(manifest.case_ids)
    group_count = len(manifest.groups)
    aggregate: dict[str, Any] = {
        "checks": {
            "group_exclusive": True,
            "input_population_matches_manifest": True,
            "input_tree_digest_verified": True,
            "manifest_canonical_freezer_output": True,
            "manifest_declares_label_blind_construction": True,
            "manifest_digest_verified": True,
            "no_unseen_or_protected_claim": True,
            "population_coverage_once_per_repeat": True,
            "public_robustness_not_unseen": True,
            "source_revision_verified": True,
            "split_deterministic": True,
            "synthetic_exclusion_mechanics_verified": True,
            "tool_source_verified": True,
            "whole_public_cohort_taint_token_present": True,
        },
        "comparison_scope": EVIDENCE_CLASS,
        "counts": {
            "matching_layout_taint_group_count": (
                matching_taint_groups
            ),
            "matching_layout_taint_record_count": (
                matching_taint_records
            ),
            "synthetic_exclusion_group_count": 1,
            "synthetic_exclusion_record_count": 1,
            "taint_event_count": taint.event_count,
            "validation_group_assignment_count": (
                group_count * REPEATS
            ),
            "validation_record_assignment_count": (
                record_count * REPEATS
            ),
            "whole_public_cohort_taint_token_event_count": (
                taint.whole_public_cohort_taint_token_event_count
            ),
        },
        "evaluation_mode": "aggregate_only",
        "evidence_label": EVIDENCE_CLASS,
        "expected_record_count": record_count,
        "fold_count": FOLDS,
        "fold_metrics": fold_metrics,
        "input_pdf_count": record_count,
        "input_tree_sha256": input_tree_sha,
        "layout_group_count": group_count,
        "layout_manifest_sha256": manifest.sha256,
        "repeat_count": REPEATS,
        "source_revision_sha": source_revision,
        "split_count": REPEATS * FOLDS,
        "status": "passed",
        "taint_registry_head_sha256": taint.head_sha256,
        "taint_registry_sha256": taint.sha256,
        "tool_source_sha256": expected_tool_sha,
    }
    _validate_aggregate_shape(aggregate)
    return aggregate


def render_aggregate_markdown(aggregate: Mapping[str, Any]) -> str:
    """Render deterministic aggregate-only Markdown."""

    _validate_aggregate_shape(aggregate)
    checks = aggregate["checks"]
    counts = aggregate["counts"]
    fold_metrics = aggregate["fold_metrics"]
    label_blind_status = (
        "PASS"
        if checks["manifest_declares_label_blind_construction"]
        else "FAIL"
    )
    public_token_status = (
        "YES"
        if checks["whole_public_cohort_taint_token_present"]
        else "NO"
    )
    public_token_count = counts[
        "whole_public_cohort_taint_token_event_count"
    ]
    lines = [
        "# WO-12 repeated grouped-split demonstration",
        "",
        f"- Evidence class: `{aggregate['evidence_label']}`",
        f"- Status: **{str(aggregate['status']).upper()}**",
        f"- Source revision: `{aggregate['source_revision_sha']}`",
        f"- Evidence-tool source: `{aggregate['tool_source_sha256']}`",
        f"- Canonical layout-manifest bytes: "
        f"`{aggregate['layout_manifest_sha256']}`",
        f"- Input tree: `{aggregate['input_tree_sha256']}`",
        f"- Taint registry / head: "
        f"`{aggregate['taint_registry_sha256']}` / "
        f"`{aggregate['taint_registry_head_sha256']}`",
        f"- Records / layout groups: "
        f"{aggregate['expected_record_count']} / "
        f"{aggregate['layout_group_count']}",
        f"- Repeats / folds per repeat / total splits: "
        f"{aggregate['repeat_count']} / {aggregate['fold_count']} / "
        f"{aggregate['split_count']}",
        "",
        "## Aggregate split counts",
        "",
        "| Repeat | Fold | Tuning records | Tuning groups | "
        "Validation records | Validation groups |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for repeat in range(1, REPEATS + 1):
        for fold in range(1, FOLDS + 1):
            metrics = fold_metrics[f"repeat_{repeat}_fold_{fold}"]
            lines.append(
                f"| {repeat} | {fold} | "
                f"{metrics['tuning_record_count']} | "
                f"{metrics['tuning_group_count']} | "
                f"{metrics['validation_record_count']} | "
                f"{metrics['validation_group_count']} |"
            )
    lines.extend(
        [
            "",
            "## Verified mechanics and evidence boundary",
            "",
            f"- Deterministic assignment: "
            f"{'PASS' if checks['split_deterministic'] else 'FAIL'}",
            f"- Whole-group tuning/validation exclusivity: "
            f"{'PASS' if checks['group_exclusive'] else 'FAIL'}",
            f"- Every manifest record and group appears in validation exactly "
            f"once per repeat: "
            f"{'PASS' if checks['population_coverage_once_per_repeat'] else 'FAIL'}",
            f"- Synthetic exclusion-mechanics probe: "
            f"{'PASS' if checks['synthetic_exclusion_mechanics_verified'] else 'FAIL'} "
            f"({counts['synthetic_exclusion_record_count']} record in "
            f"{counts['synthetic_exclusion_group_count']} group; identities "
            "not emitted)",
            f"- Registry entries matching the layout-group namespace: "
            f"{counts['matching_layout_taint_group_count']} groups / "
            f"{counts['matching_layout_taint_record_count']} records",
            f"- Generic public-cohort taint token present: "
            f"{public_token_status} ({public_token_count} matching event)",
            f"- Canonical manifest declares label-blind construction: "
            f"{label_blind_status}",
            "",
            "> This demonstrates public grouped-robustness split mechanics "
            "only. The evaluated PDFs and labels are public, so no fold is "
            "represented as protected, private, pristine, or unseen.",
            "",
            "> The generic public-cohort taint token is not bound to the "
            "layout-manifest or input-tree digest and does not attest that it "
            "identifies this exact population.",
            "",
            "> This report makes no claim that the manifest was fixed before "
            "scoring. A future temporal claim requires the manifest hash and "
            "seed to be preregistered in a governed plan.",
            "",
        ]
    )
    return "\n".join(lines)


def _output_state(path: Path, expected: str) -> bool:
    """Return whether a securely bound output exists with exact bytes."""

    parent_descriptor = -1
    try:
        parent_descriptor, resolved_parent, name = (
            manifest_freezer._open_output_parent(path)
        )
        manifest_freezer._verify_parent_binding(
            parent_descriptor,
            resolved_parent,
        )
        try:
            metadata = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise GroupedSplitEvidenceBuildError(
                "existing evidence output must be a regular file"
            )
        actual = manifest_freezer._read_output_at(
            parent_descriptor,
            name,
        )
        manifest_freezer._verify_parent_binding(
            parent_descriptor,
            resolved_parent,
        )
    except GroupedSplitEvidenceBuildError:
        raise
    except Exception as exc:
        raise GroupedSplitEvidenceBuildError(
            "existing evidence output could not be verified"
        ) from exc
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    if actual != expected.encode("utf-8"):
        raise GroupedSplitEvidenceBuildError(
            "evidence output already exists with different bytes"
        )
    return True


def _atomic_write(path: Path, content: str) -> bool:
    """Create one output without replacement; return whether it was created."""

    try:
        return manifest_freezer._atomic_write(
            path,
            content.encode("utf-8"),
        )
    except Exception as exc:
        raise GroupedSplitEvidenceBuildError(
            "evidence output could not be created safely"
        ) from exc


def _remove_created_output(path: Path, expected: str) -> None:
    """Best-effort rollback only for bytes created by this invocation."""

    parent_descriptor = -1
    try:
        parent_descriptor, resolved_parent, name = (
            manifest_freezer._open_output_parent(path)
        )
        actual = manifest_freezer._read_output_at(
            parent_descriptor,
            name,
        )
        if actual == expected.encode("utf-8"):
            manifest_freezer._verify_parent_binding(
                parent_descriptor,
                resolved_parent,
            )
            os.unlink(name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
    except Exception:
        # A remaining partial pair is detected and rejected on the next run.
        pass
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _write_output_pair(
    output_json: Path,
    json_content: str,
    output_markdown: Path,
    markdown_content: str,
) -> None:
    """Create both evidence files idempotently and fail closed on partials."""

    pairs = (
        (output_json, json_content),
        (output_markdown, markdown_content),
    )
    exists = tuple(
        _output_state(path, content) for path, content in pairs
    )
    if all(exists):
        return

    created: list[tuple[Path, str]] = []
    try:
        for path, content in pairs:
            if not _output_state(path, content) and _atomic_write(
                path,
                content,
            ):
                created.append((path, content))
        for path, content in pairs:
            if not _output_state(path, content):
                raise GroupedSplitEvidenceBuildError(
                    "evidence output pair is incomplete"
                )
    except BaseException:
        for path, content in reversed(created):
            _remove_created_output(path, content)
        raise


def _validate_output_paths(
    *,
    output_json: Path,
    output_markdown: Path,
    layout_manifest: Path,
    input_dir: Path,
    taint_registry: Path,
) -> tuple[Path, Path]:
    for output_path in (output_json, output_markdown):
        try:
            resolved_parent = output_path.parent.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GroupedSplitEvidenceBuildError(
                "evidence output directory must already exist safely"
            ) from exc
        if (
            output_path.parent.is_symlink()
            or output_path.is_symlink()
            or not resolved_parent.is_dir()
        ):
            raise GroupedSplitEvidenceBuildError(
                "evidence output paths must not contain symlinks"
            )
    json_path = _safe_resolve(output_json)
    markdown_path = _safe_resolve(output_markdown)
    if _paths_overlap(json_path, markdown_path):
        raise GroupedSplitEvidenceBuildError(
            "JSON and Markdown outputs must not overlap"
        )
    protected = {
        _safe_resolve(layout_manifest),
        _safe_resolve(taint_registry),
        _safe_resolve(Path(__file__)),
    }
    if any(
        _paths_overlap(output_path, protected_path)
        for output_path in (json_path, markdown_path)
        for protected_path in protected
    ):
        raise GroupedSplitEvidenceBuildError(
            "evidence output must not overwrite an evidence input"
        )
    input_root = _safe_resolve(input_dir)
    for output_path in (json_path, markdown_path):
        if _paths_overlap(output_path, input_root):
            raise GroupedSplitEvidenceBuildError(
                "evidence outputs must be outside the PDF input directory"
            )
    return json_path, markdown_path


def _paths_overlap(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build aggregate-only WO-12 evidence for deterministic 3x5 "
            "grouped-split mechanics."
        )
    )
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-layout-manifest-sha256",
        required=True,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-input-tree-sha256", required=True)
    parser.add_argument("--taint-registry", type=Path, required=True)
    parser.add_argument("--expected-taint-registry-sha256", required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument(
        "--expected-tool-source-sha256",
        "--tool-source-sha256",
        dest="expected_tool_source_sha256",
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        _validate_output_paths(
            output_json=arguments.output_json,
            output_markdown=arguments.output_markdown,
            layout_manifest=arguments.layout_manifest,
            input_dir=arguments.input_dir,
            taint_registry=arguments.taint_registry,
        )
        aggregate = build_aggregate_evidence(
            layout_manifest_path=arguments.layout_manifest,
            expected_layout_manifest_sha256=(
                arguments.expected_layout_manifest_sha256
            ),
            input_dir=arguments.input_dir,
            expected_input_tree_sha256=(
                arguments.expected_input_tree_sha256
            ),
            taint_registry_path=arguments.taint_registry,
            expected_taint_registry_sha256=(
                arguments.expected_taint_registry_sha256
            ),
            source_revision_sha=arguments.source_revision_sha,
            expected_tool_source_sha256=(
                arguments.expected_tool_source_sha256
            ),
        )
        json_content = canonical_json(aggregate) + "\n"
        markdown_content = render_aggregate_markdown(aggregate)
        output_json, output_markdown = _validate_output_paths(
            output_json=arguments.output_json,
            output_markdown=arguments.output_markdown,
            layout_manifest=arguments.layout_manifest,
            input_dir=arguments.input_dir,
            taint_registry=arguments.taint_registry,
        )
        _write_output_pair(
            output_json,
            json_content,
            output_markdown,
            markdown_content,
        )
    except (ExperimentControlError, OSError, UnicodeError):
        print(
            "grouped split evidence error: verification or output failed",
            file=sys.stderr,
        )
        return 1
    print(
        "WO-12 grouped split evidence: "
        f"{aggregate['status']} "
        f"({aggregate['repeat_count']}x{aggregate['fold_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
