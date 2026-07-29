import io
import tempfile
import unittest
from pathlib import Path

from mib_pipeline import (
    DocumentRenderer,
    RecoverableRenderError,
    Rect,
    RenderFirstFallbackProcessor,
    SafeFallbackProcessor,
    TextLayerReader,
)


try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None


class FakePage:
    def get_textpage(self):
        raise AssertionError("native PDF text must never be opened")


class TextLayerTests(unittest.TestCase):
    def test_compatibility_reader_is_a_native_text_noop(self):
        spans, truncated = TextLayerReader().read_page(
            FakePage(),
            page_index=0,
            crop_box=Rect(0, 0, 100, 100),
        )

        self.assertFalse(truncated)
        self.assertEqual(spans, ())

    def test_compatibility_reader_keeps_positive_limit_validation(self):
        with self.assertRaises(ValueError):
            TextLayerReader(max_chars_per_page=0)


class RenderFirstCompositionTests(unittest.TestCase):
    def test_fallback_is_emitted_only_after_renderer_runs(self):
        class RecordingRenderer:
            def __init__(self):
                self.paths = []

            def render(self, path):
                self.paths.append(path)

        renderer = RecordingRenderer()
        processor = RenderFirstFallbackProcessor(renderer, SafeFallbackProcessor())
        path = Path("MIB-000001.pdf")

        row = processor.process_case(path)

        self.assertEqual(renderer.paths, [path])
        self.assertEqual(row["case_id"], "MIB-000001")


@unittest.skipIf(Image is None, "Pillow rendering dependency is not installed")
class DocumentRendererTests(unittest.TestCase):
    def make_image_pdf(self, path):
        image = Image.new("RGB", (600, 800), "white")
        draw = ImageDraw.Draw(image)
        for y in range(100, 700, 60):
            draw.line((80, y, 520, y), fill="black", width=4)
        image.save(path, "PDF", resolution=100)

    def test_every_page_is_rasterized_to_png(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "MIB-000123.pdf"
            self.make_image_pdf(pdf_path)

            rendered = DocumentRenderer(target_dpi=120).render(pdf_path)

            self.assertEqual(rendered.case_id, "MIB-000123")
            self.assertEqual(len(rendered.pages), 1)
            page = rendered.pages[0]
            self.assertTrue(page.image_png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(page.width_px, 0)
            self.assertGreater(page.height_px, 0)
            self.assertGreater(page.crop_box.width, 0)
            self.assertEqual(rendered.text_layer, ())
            with Image.open(io.BytesIO(page.image_png)) as normalized:
                self.assertEqual(normalized.mode, "RGB")

    def test_corrupt_pdf_is_a_recoverable_case_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            pdf_path = Path(temporary_dir) / "MIB-000124.pdf"
            pdf_path.write_bytes(b"not a PDF")

            with self.assertRaises(RecoverableRenderError):
                DocumentRenderer().render(pdf_path)

    def test_render_resolution_is_bounded_by_pixel_budget(self):
        renderer = DocumentRenderer(target_dpi=300, max_render_pixels=1_000_000)
        scale = renderer._render_scale(1000, 1000)
        self.assertLessEqual(1000 * 1000 * scale**2, 1_000_001)

    def test_skew_estimation_finds_a_correction_for_visible_lines(self):
        image = Image.new("L", (800, 500), "white")
        draw = ImageDraw.Draw(image)
        for y in range(100, 420, 50):
            draw.line((80, y, 720, y), fill="black", width=5)
        skewed = image.rotate(2.0, resample=Image.Resampling.BICUBIC, fillcolor=255)
        _, image_module, numpy_module = DocumentRenderer._dependencies()

        correction = DocumentRenderer._estimate_skew(
            skewed, image_module, numpy_module
        )

        self.assertAlmostEqual(correction, -2.0, delta=0.75)


if __name__ == "__main__":
    unittest.main()
