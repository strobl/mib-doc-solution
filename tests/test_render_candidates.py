import io
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from devtools.render_candidates import (
    BoundedContrastRenderer,
    BoundedTemplateRegistrationRenderer,
)
from mib_pipeline import Rect, RenderedCase, RenderedPage, TextSpan


class FixedRenderer:
    def __init__(self, rendered_case):
        self._rendered_case = rendered_case

    def render(self, pdf_path):
        del pdf_path
        return self._rendered_case


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _case(image, *, span_box=Rect(100, 100, 160, 120)):
    page = RenderedPage(
        index=0,
        image_png=_png(image),
        width_px=image.width,
        height_px=image.height,
        dpi=200,
        rotation_deg=0,
        skew_correction_deg=0.0,
        crop_box=Rect(0, 0, image.width, image.height),
        text_spans=(
            TextSpan(
                page_index=0,
                text="visible fixture",
                box=span_box,
            ),
        ),
    )
    return RenderedCase(
        source_path=Path("/tmp/MIB-123456.pdf"),
        source_sha256="a" * 64,
        case_id="MIB-123456",
        pages=(page,),
        text_layer=page.text_spans,
    )


def _content_center(image_png):
    import numpy

    with Image.open(io.BytesIO(image_png)) as image:
        pixels = numpy.asarray(image.convert("L"))
    ys, xs = (pixels < 210).nonzero()
    return (
        (float(xs.min()) + float(xs.max())) / 2.0,
        (float(ys.min()) + float(ys.max())) / 2.0,
    )


class TemplateRegistrationRendererTests(unittest.TestCase):
    def test_bounded_registration_recenters_frame_and_translates_spans(self):
        image = Image.new("L", (400, 300), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((63, 49, 353, 259), outline=0, width=6)
        rendered_case = _case(image)
        wrapper = BoundedTemplateRegistrationRenderer(
            FixedRenderer(rendered_case)
        )

        result = wrapper.render(Path("/tmp/input.pdf"))
        activity = wrapper.ablation_activity()
        before_center = _content_center(rendered_case.pages[0].image_png)
        after_center = _content_center(result.pages[0].image_png)

        self.assertEqual(activity["pages_scanned"], 1)
        self.assertEqual(activity["eligible_frames"], 1)
        self.assertEqual(activity["pages_registered"], 1)
        self.assertLess(
            abs(after_center[0] - 199.5),
            abs(before_center[0] - 199.5),
        )
        self.assertLess(
            abs(after_center[1] - 149.5),
            abs(before_center[1] - 149.5),
        )
        dx = result.text_layer[0].box.left - rendered_case.text_layer[0].box.left
        dy = result.text_layer[0].box.bottom - rendered_case.text_layer[0].box.bottom
        self.assertEqual((dx, dy), (-8, -4))

    def test_centered_or_excessively_shifted_frames_do_not_change_pixels(self):
        scenarios = (
            (55, 45, 345, 255),
            (90, 45, 380, 255),
        )
        for box in scenarios:
            with self.subTest(box=box):
                image = Image.new("L", (400, 300), color=255)
                ImageDraw.Draw(image).rectangle(box, outline=0, width=6)
                rendered_case = _case(image)
                wrapper = BoundedTemplateRegistrationRenderer(
                    FixedRenderer(rendered_case)
                )

                result = wrapper.render(Path("/tmp/input.pdf"))

                self.assertEqual(
                    result.pages[0].image_png,
                    rendered_case.pages[0].image_png,
                )
                self.assertEqual(
                    wrapper.ablation_activity()["pages_registered"],
                    0,
                )

    def test_registration_is_byte_deterministic(self):
        image = Image.new("L", (400, 300), color=255)
        ImageDraw.Draw(image).rectangle(
            (63, 49, 353, 259),
            outline=0,
            width=6,
        )
        rendered_case = _case(image)

        first = BoundedTemplateRegistrationRenderer(
            FixedRenderer(rendered_case)
        ).render(Path("/tmp/input.pdf"))
        second = BoundedTemplateRegistrationRenderer(
            FixedRenderer(rendered_case)
        ).render(Path("/tmp/input.pdf"))

        self.assertEqual(first, second)


class ContrastRendererTests(unittest.TestCase):
    def test_low_contrast_page_is_enhanced_behind_gate(self):
        image = Image.new("L", (400, 300), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((55, 45, 345, 255), outline=150, width=6)
        for top in range(80, 240, 30):
            draw.line((80, top, 320, top), fill=150, width=4)
        rendered_case = _case(image)
        wrapper = BoundedContrastRenderer(FixedRenderer(rendered_case))

        result = wrapper.render(Path("/tmp/input.pdf"))
        activity = wrapper.ablation_activity()

        self.assertNotEqual(
            result.pages[0].image_png,
            rendered_case.pages[0].image_png,
        )
        self.assertEqual(activity["low_contrast_pages"], 1)
        self.assertEqual(activity["pages_enhanced"], 1)
        with Image.open(io.BytesIO(result.pages[0].image_png)) as enhanced:
            self.assertEqual(min(enhanced.tobytes()), 0)

    def test_high_contrast_or_sparse_page_abstains(self):
        high_contrast = Image.new("L", (400, 300), color=255)
        ImageDraw.Draw(high_contrast).rectangle(
            (55, 45, 345, 255),
            outline=0,
            width=6,
        )
        sparse = Image.new("L", (400, 300), color=255)
        ImageDraw.Draw(sparse).point((200, 150), fill=150)
        for image in (high_contrast, sparse):
            with self.subTest(image=image):
                rendered_case = _case(image)
                wrapper = BoundedContrastRenderer(
                    FixedRenderer(rendered_case)
                )

                result = wrapper.render(Path("/tmp/input.pdf"))

                self.assertEqual(
                    result.pages[0].image_png,
                    rendered_case.pages[0].image_png,
                )
                self.assertEqual(
                    wrapper.ablation_activity()["pages_enhanced"],
                    0,
                )

    def test_contrast_is_byte_deterministic(self):
        image = Image.new("L", (400, 300), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((55, 45, 345, 255), outline=150, width=6)
        for top in range(80, 240, 30):
            draw.line((80, top, 320, top), fill=150, width=4)
        rendered_case = _case(image)

        first = BoundedContrastRenderer(
            FixedRenderer(rendered_case)
        ).render(Path("/tmp/input.pdf"))
        second = BoundedContrastRenderer(
            FixedRenderer(rendered_case)
        ).render(Path("/tmp/input.pdf"))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
