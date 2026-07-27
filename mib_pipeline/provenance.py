"""Deterministic provenance for visible OCR observations.

The public output contains normalized values, but adjudication must retain the
visible pixels that produced each value.  These records deliberately contain
no process IDs, object identities, timestamps, filenames, or label-derived
lookups: equal source pixels and equal OCR routes serialize identically.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable

from .ingestion import Rect


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _rect_values(rect: Rect) -> tuple[float, float, float, float]:
    return tuple(
        round(_finite(value, "box coordinate"), 6)
        for value in (
            rect.left,
            rect.bottom,
            rect.right,
            rect.top,
        )
    )


def _stable_identifier(value: str, name: str) -> str:
    identifier = value.strip()
    if not identifier or _CONTROL_CHARACTER_RE.search(identifier):
        raise ValueError(f"{name} must be a stable non-empty identifier")
    return identifier


@dataclass(frozen=True)
class CoordinateTransform:
    """Affine map from one OCR view into physical rendered-page pixels."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 1.0
    f: float = 0.0
    view_space: str = "ocr_view_pixels"
    physical_space: str = "rendered_page_pixels"

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d", "e", "f"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "view_space",
            _stable_identifier(self.view_space, "view_space"),
        )
        object.__setattr__(
            self,
            "physical_space",
            _stable_identifier(self.physical_space, "physical_space"),
        )
        determinant = self.a * self.e - self.b * self.d
        if abs(determinant) < 1e-12:
            raise ValueError("coordinate transform must be invertible")

    @classmethod
    def identity(cls) -> "CoordinateTransform":
        return cls()

    @classmethod
    def crop_translation(
        cls,
        *,
        left: float,
        upper: float,
    ) -> "CoordinateTransform":
        return cls(c=_finite(left, "left"), f=_finite(upper, "upper"))

    @classmethod
    def inverse_quarter_turn(
        cls,
        *,
        angle_degrees: int,
        source_width: float,
        source_height: float,
    ) -> "CoordinateTransform":
        """Map a Pillow-rotated, expanded view back to its source page."""

        angle = int(angle_degrees) % 360
        width = _finite(source_width, "source_width")
        height = _finite(source_height, "source_height")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("source dimensions must be positive")
        if angle == 0:
            return cls.identity()
        if angle == 90:
            return cls(a=0.0, b=-1.0, c=width, d=1.0, e=0.0, f=0.0)
        if angle == 180:
            return cls(a=-1.0, b=0.0, c=width, d=0.0, e=-1.0, f=height)
        if angle == 270:
            return cls(a=0.0, b=1.0, c=0.0, d=-1.0, e=0.0, f=height)
        raise ValueError("only quarter-turn rotations are supported")

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        x_value = _finite(x, "x")
        y_value = _finite(y, "y")
        return (
            self.a * x_value + self.b * y_value + self.c,
            self.d * x_value + self.e * y_value + self.f,
        )

    def apply_box(self, box: Rect) -> Rect:
        corners = (
            self.apply_point(box.left, box.bottom),
            self.apply_point(box.left, box.top),
            self.apply_point(box.right, box.bottom),
            self.apply_point(box.right, box.top),
        )
        xs = tuple(point[0] for point in corners)
        ys = tuple(point[1] for point in corners)
        return Rect(min(xs), min(ys), max(xs), max(ys))

    def to_dict(self) -> dict[str, object]:
        return {
            "from": self.view_space,
            "to": self.physical_space,
            "matrix_2x3": [
                round(self.a, 12),
                round(self.b, 12),
                round(self.c, 12),
                round(self.d, 12),
                round(self.e, 12),
                round(self.f, 12),
            ],
        }


@dataclass(frozen=True)
class PhysicalObservation:
    """One visible region on one immutable source-document page."""

    source_sha256: str
    page_index: int
    box: Rect
    applicant_scope: str | None = None

    def __post_init__(self) -> None:
        digest = self.source_sha256.strip().casefold()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("source_sha256 must be a 64-character lowercase hex digest")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        _rect_values(self.box)
        if self.box.width <= 0.0 or self.box.height <= 0.0:
            raise ValueError("physical observation box must have positive area")
        applicant_scope = self.applicant_scope
        if applicant_scope is not None:
            applicant_scope = _stable_identifier(
                applicant_scope,
                "applicant_scope",
            )
        object.__setattr__(self, "source_sha256", digest)
        object.__setattr__(self, "applicant_scope", applicant_scope)

    @property
    def observation_id(self) -> str:
        """Content-address the physical region, independent of OCR route."""

        payload = {
            "source_sha256": self.source_sha256,
            "page_index": self.page_index,
            "box": _rect_values(self.box),
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "source_sha256": self.source_sha256,
            "page_index": self.page_index,
            "box": list(_rect_values(self.box)),
            "applicant_scope": self.applicant_scope,
        }


@dataclass(frozen=True)
class OcrProvenance:
    """How an OCR view produced a candidate from a physical observation."""

    route_id: str
    engine_id: str
    view_id: str
    view_box: Rect
    transform: CoordinateTransform
    observation: PhysicalObservation

    def __post_init__(self) -> None:
        for name in ("route_id", "engine_id", "view_id"):
            object.__setattr__(
                self,
                name,
                _stable_identifier(getattr(self, name), name),
            )
        _rect_values(self.view_box)
        if self.view_box.width <= 0.0 or self.view_box.height <= 0.0:
            raise ValueError("view_box must have positive area")
        expected = self.transform.apply_box(self.view_box)
        if any(
            abs(actual - transformed) > 1e-6
            for actual, transformed in zip(
                _rect_values(self.observation.box),
                _rect_values(expected),
            )
        ):
            raise ValueError("physical observation box must match the coordinate transform")

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "engine_id": self.engine_id,
            "view_id": self.view_id,
            "view_box": list(_rect_values(self.view_box)),
            "coordinate_transform": self.transform.to_dict(),
            "physical_observation": self.observation.to_dict(),
        }


def make_ocr_provenance(
    *,
    source_sha256: str,
    page_index: int,
    view_box: Rect,
    applicant_scope: str | None,
    route_id: str,
    engine_id: str,
    view_id: str,
    transform: CoordinateTransform | None = None,
) -> OcrProvenance:
    coordinate_transform = transform or CoordinateTransform.identity()
    return OcrProvenance(
        route_id=route_id,
        engine_id=engine_id,
        view_id=view_id,
        view_box=view_box,
        transform=coordinate_transform,
        observation=PhysicalObservation(
            source_sha256=source_sha256,
            page_index=page_index,
            box=coordinate_transform.apply_box(view_box),
            applicant_scope=applicant_scope,
        ),
    )


def merge_ocr_provenance(
    *groups: Iterable[OcrProvenance],
) -> tuple[OcrProvenance, ...]:
    """Deduplicate and sort route evidence without depending on input order."""

    unique: dict[str, OcrProvenance] = {}
    for group in groups:
        for item in group:
            unique[item.fingerprint] = item
    return tuple(unique[key] for key in sorted(unique))
