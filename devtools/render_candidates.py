"""Development-only bounded rendering candidates for WO-14.

The submitted runtime never imports this module.  Both wrappers are
label-blind, preserve page dimensions, apply to at most two pages per case,
and expose aggregate counters for deterministic ablation evidence.
"""

from __future__ import annotations

import io
import threading
from dataclasses import replace
from typing import Any

from mib_pipeline import Rect, RenderedCase, RenderedPage


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Pillow and numpy are required for rendering ablations"
        ) from exc
    return Image, ImageOps, numpy


def _page_image(page: RenderedPage) -> Any:
    Image, _ImageOps, _numpy = _dependencies()
    with Image.open(io.BytesIO(page.image_png)) as image:
        return image.convert("L").copy()


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def _translate_box(box: Rect, dx: int, dy: int) -> Rect:
    return Rect(
        box.left + dx,
        box.bottom + dy,
        box.right + dx,
        box.top + dy,
    )


class BoundedTemplateRegistrationRenderer:
    """Translate a robust content frame toward canonical page coordinates."""

    _MAX_REGISTERED_PAGES_PER_CASE = 2
    _MAX_SHIFT_FRACTION = 0.03

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._activity = {
            "pages_scanned": 0,
            "eligible_frames": 0,
            "pages_registered": 0,
            "pages_unchanged": 0,
            "pages_abstained": 0,
        }
        self._lock = threading.Lock()

    def _increment(self, name: str, count: int = 1) -> None:
        with self._lock:
            self._activity[name] += count

    @classmethod
    def _registration_shift(cls, image: Any) -> tuple[int, int] | None:
        _Image, _ImageOps, numpy = _dependencies()
        pixels = numpy.asarray(image)
        height, width = pixels.shape[:2]
        if min(width, height) < 100:
            return None
        ink = pixels < 210
        border_x = max(1, int(round(width * 0.01)))
        border_y = max(1, int(round(height * 0.01)))
        ink[:border_y, :] = False
        ink[-border_y:, :] = False
        ink[:, :border_x] = False
        ink[:, -border_x:] = False
        ink_fraction = float(ink.mean())
        if not 0.005 <= ink_fraction <= 0.35:
            return None
        ys, xs = ink.nonzero()
        if len(xs) < 100:
            return None
        left, right = (
            float(numpy.percentile(xs, percentile))
            for percentile in (5, 95)
        )
        top, bottom = (
            float(numpy.percentile(ys, percentile))
            for percentile in (5, 95)
        )
        frame_width = right - left
        frame_height = bottom - top
        if frame_width < width * 0.35 or frame_height < height * 0.35:
            return None
        content_center_x = (left + right) / 2.0
        content_center_y = (top + bottom) / 2.0
        target_center_x = (width - 1) / 2.0
        target_center_y = (height - 1) / 2.0
        dx = int(round(target_center_x - content_center_x))
        dy = int(round(target_center_y - content_center_y))
        max_dx = max(2, int(round(width * cls._MAX_SHIFT_FRACTION)))
        max_dy = max(2, int(round(height * cls._MAX_SHIFT_FRACTION)))
        if abs(dx) > max_dx or abs(dy) > max_dy:
            return None
        return dx, dy

    @staticmethod
    def _translate_image(image: Any, dx: int, dy: int) -> Any:
        Image, _ImageOps, _numpy = _dependencies()
        width, height = image.size
        translated = Image.new("L", image.size, color=255)
        source_left = max(0, -dx)
        source_top = max(0, -dy)
        source_right = min(width, width - dx)
        source_bottom = min(height, height - dy)
        if source_right <= source_left or source_bottom <= source_top:
            return image.copy()
        region = image.crop(
            (source_left, source_top, source_right, source_bottom)
        )
        translated.paste(
            region,
            (max(0, dx), max(0, dy)),
        )
        return translated

    def render(self, pdf_path: Any) -> RenderedCase:
        rendered_case = self._delegate.render(pdf_path)
        registered_count = 0
        shifts: dict[int, tuple[int, int]] = {}
        pages: list[RenderedPage] = []
        for page in rendered_case.pages:
            self._increment("pages_scanned")
            if registered_count >= self._MAX_REGISTERED_PAGES_PER_CASE:
                self._increment("pages_abstained")
                pages.append(page)
                continue
            image = _page_image(page)
            shift = self._registration_shift(image)
            if shift is None:
                self._increment("pages_abstained")
                pages.append(page)
                continue
            self._increment("eligible_frames")
            dx, dy = shift
            if abs(dx) <= 1 and abs(dy) <= 1:
                self._increment("pages_unchanged")
                pages.append(page)
                continue
            translated = self._translate_image(image, dx, dy)
            translated_spans = tuple(
                replace(span, box=_translate_box(span.box, dx, dy))
                for span in page.text_spans
            )
            pages.append(
                replace(
                    page,
                    image_png=_png_bytes(translated),
                    text_spans=translated_spans,
                )
            )
            shifts[page.index] = (dx, dy)
            registered_count += 1
            self._increment("pages_registered")
        case_spans = tuple(
            replace(
                span,
                box=_translate_box(
                    span.box,
                    shifts[span.page_index][0],
                    shifts[span.page_index][1],
                ),
            )
            if span.page_index in shifts
            else span
            for span in rendered_case.text_layer
        )
        return replace(
            rendered_case,
            pages=tuple(pages),
            text_layer=case_spans,
        )

    def ablation_activity(self) -> dict[str, int]:
        with self._lock:
            return dict(self._activity)


class BoundedContrastRenderer:
    """Autocontrast only low-contrast pages behind a visible-pixel gate."""

    _MAX_ENHANCED_PAGES_PER_CASE = 2

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._activity = {
            "pages_scanned": 0,
            "low_contrast_pages": 0,
            "pages_enhanced": 0,
            "pages_unchanged": 0,
            "pages_abstained": 0,
        }
        self._lock = threading.Lock()

    def _increment(self, name: str, count: int = 1) -> None:
        with self._lock:
            self._activity[name] += count

    @staticmethod
    def _is_low_contrast(image: Any) -> bool:
        _Image, _ImageOps, numpy = _dependencies()
        pixels = numpy.asarray(image)
        foreground = pixels[pixels < 245]
        foreground_fraction = foreground.size / max(1, pixels.size)
        if not 0.003 <= foreground_fraction <= 0.40:
            return False
        if float(numpy.percentile(pixels, 95)) < 245.0:
            return False
        median_foreground = float(numpy.median(foreground))
        darkest_foreground = float(numpy.percentile(foreground, 5))
        return (
            median_foreground >= 96.0
            and darkest_foreground >= 48.0
            and 255.0 - darkest_foreground >= 15.0
        )

    def render(self, pdf_path: Any) -> RenderedCase:
        rendered_case = self._delegate.render(pdf_path)
        enhanced_count = 0
        pages: list[RenderedPage] = []
        for page in rendered_case.pages:
            self._increment("pages_scanned")
            if enhanced_count >= self._MAX_ENHANCED_PAGES_PER_CASE:
                self._increment("pages_abstained")
                pages.append(page)
                continue
            image = _page_image(page)
            if not self._is_low_contrast(image):
                self._increment("pages_abstained")
                pages.append(page)
                continue
            self._increment("low_contrast_pages")
            _Image, ImageOps, _numpy = _dependencies()
            enhanced = ImageOps.autocontrast(image, cutoff=(1, 1))
            if enhanced.tobytes() == image.tobytes():
                self._increment("pages_unchanged")
                pages.append(page)
                continue
            enhanced_png = _png_bytes(enhanced)
            pages.append(replace(page, image_png=enhanced_png))
            enhanced_count += 1
            self._increment("pages_enhanced")
        return replace(rendered_case, pages=tuple(pages))

    def ablation_activity(self) -> dict[str, int]:
        with self._lock:
            return dict(self._activity)
