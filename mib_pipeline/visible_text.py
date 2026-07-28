"""Thread-safe handoff of structured, pixel-derived OCR evidence."""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class VisibleTextSnapshotError(RuntimeError):
    """A visible OCR snapshot cannot safely be used for final scoring."""


class VisibleTextSnapshotMissing(VisibleTextSnapshotError):
    """No unconsumed snapshot exists for the requested source."""


class VisibleTextSnapshotMismatch(VisibleTextSnapshotError):
    """The source bytes no longer match the snapshot's bound digest."""


@dataclass(frozen=True)
class VisibleOcrLineRecord:
    """One filter-accepted primary PSM 11 OCR line."""

    page_index: int
    text: str
    bbox: tuple[float, float, float, float]
    ocr_confidence: float
    visual_cues: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain four coordinates")
        if not 0.0 <= self.ocr_confidence <= 1.0:
            raise ValueError("ocr_confidence must be between 0 and 1")
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(
            self,
            "bbox",
            tuple(float(value) for value in self.bbox),
        )
        object.__setattr__(
            self,
            "visual_cues",
            tuple(sorted({str(cue) for cue in self.visual_cues})),
        )


@dataclass(frozen=True)
class VisibleSponsorAttestation:
    """One structured sponsor/applicant pair read from an attestation page."""

    sponsor_id: str
    applicant_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sponsor_id", str(self.sponsor_id).strip())
        object.__setattr__(
            self,
            "applicant_name",
            " ".join(str(self.applicant_name).split()),
        )


@dataclass(frozen=True)
class VisibleOcrPageSnapshot:
    """Accepted OCR records and routing metadata for one rendered page."""

    page_index: int
    lines: tuple[VisibleOcrLineRecord, ...]
    page_category: str
    source_category: str
    visible_case_ids: frozenset[str]
    visible_applicants: frozenset[str]
    sponsor_attestations: tuple[VisibleSponsorAttestation, ...] = ()

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        lines = tuple(self.lines)
        if any(line.page_index != self.page_index for line in lines):
            raise ValueError("all OCR lines must belong to their page snapshot")
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "page_category", str(self.page_category))
        object.__setattr__(self, "source_category", str(self.source_category))
        object.__setattr__(
            self,
            "visible_case_ids",
            frozenset(str(value) for value in self.visible_case_ids),
        )
        object.__setattr__(
            self,
            "visible_applicants",
            frozenset(str(value) for value in self.visible_applicants),
        )
        object.__setattr__(
            self,
            "sponsor_attestations",
            tuple(self.sponsor_attestations),
        )

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(frozen=True)
class VisibleOcrSnapshot:
    """Immutable OCR snapshot bound to one exact source digest."""

    source_sha256: str
    pages: tuple[VisibleOcrPageSnapshot, ...]
    route: str = "primary_psm11"

    def __post_init__(self) -> None:
        normalized_sha256 = str(self.source_sha256).casefold()
        if _SHA256_PATTERN.fullmatch(normalized_sha256) is None:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        pages = tuple(self.pages)
        page_indexes = tuple(page.page_index for page in pages)
        if page_indexes != tuple(sorted(page_indexes)) or len(set(page_indexes)) != len(
            page_indexes
        ):
            raise ValueError("snapshot pages must have unique ascending indexes")
        if self.route != "primary_psm11":
            raise ValueError("visible OCR route must be primary_psm11")
        object.__setattr__(self, "source_sha256", normalized_sha256)
        object.__setattr__(self, "pages", pages)


class VisibleOcrTextStore:
    """Keep one SHA-bound, one-shot OCR snapshot per source path.

    The extractor publishes only text reconstructed from accepted pixel OCR
    lines. The finalizer consumes the entry exactly once after independently
    hashing the source bytes. A missing or changed source therefore cannot
    fall back to a PDF text layer or reuse stale visible evidence.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[Path, VisibleOcrSnapshot] = {}

    @staticmethod
    def _key(source_path: Path) -> Path:
        return Path(source_path).resolve(strict=False)

    @staticmethod
    def _sha256(source_path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with Path(source_path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise VisibleTextSnapshotMismatch(
                "visible OCR source is unavailable"
            ) from exc
        return digest.hexdigest()

    def publish(
        self,
        source_path: Path,
        *,
        snapshot: VisibleOcrSnapshot,
    ) -> None:
        """Publish or replace the pending snapshot for one exact source."""

        if not isinstance(snapshot, VisibleOcrSnapshot):
            raise TypeError("snapshot must be a VisibleOcrSnapshot")
        with self._lock:
            self._snapshots[self._key(source_path)] = snapshot

    def consume(self, source_path: Path) -> VisibleOcrSnapshot:
        """Return one matching snapshot and make it unavailable thereafter."""

        key = self._key(source_path)
        actual_sha256 = self._sha256(source_path)
        with self._lock:
            snapshot = self._snapshots.pop(key, None)
        if snapshot is None:
            raise VisibleTextSnapshotMissing("visible OCR snapshot is missing")
        if snapshot.source_sha256 != actual_sha256:
            raise VisibleTextSnapshotMismatch(
                "visible OCR snapshot does not match source SHA-256"
            )
        return snapshot


def page_text_projection(
    pages: Iterable[VisibleOcrPageSnapshot],
    *,
    include_page_indexes: frozenset[int],
) -> str:
    """Project selected pages while retaining every original page boundary."""

    if not include_page_indexes:
        return ""
    return "\x0c".join(
        page.text if page.page_index in include_page_indexes else ""
        for page in pages
    )


def person_name_consensus_key(value: str | None) -> str:
    """Normalize a name with only the narrow leading ``I``/``l`` OCR repair.

    Person-name scoping is intentionally not fuzzy.  Tokens must otherwise be
    byte-for-byte equal after case and whitespace normalization.  The one
    tolerated visual ambiguity is a leading capital-I/lowercase-l glyph, which
    Tesseract commonly swaps in names such as ``Ixokesh``/``lxokesh``.
    """

    words = " ".join(str(value or "").casefold().split()).split()
    normalized: list[str] = []
    for word in words:
        if word and word[0] in {"i", "l"}:
            normalized.append(f"\N{SECTION SIGN}{word[1:]}")
        else:
            normalized.append(word)
    return " ".join(normalized)


def person_names_compatible(left: str | None, right: str | None) -> bool:
    """Return exact name equality after the bounded OCR-glyph normalization."""

    left_key = person_name_consensus_key(left)
    right_key = person_name_consensus_key(right)
    return bool(left_key and right_key and left_key == right_key)
