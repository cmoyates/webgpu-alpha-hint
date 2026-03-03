"""GPU shader tests — keying, blur, and morphology on synthetic pixels."""

import numpy as np

from conftest import (
    make_key_params,
    run_blur_pass,
    run_key_pass,
    run_morph_pass,
    solid_rgba,
)


class TestKeyingShader:
    """Verify normalized-chroma distance keyer on known pixel colors."""

    def test_pure_green_is_background(self, device):
        """Pure green => alpha near 0 (background)."""
        img = solid_rgba(16, 16, 0, 255, 0)
        matte = run_key_pass(device, img, make_key_params())
        assert matte.mean() < 0.05, f"Pure green should be ~0, got {matte.mean():.3f}"

    def test_pure_red_is_foreground(self, device):
        """Pure red => alpha near 1 (foreground)."""
        img = solid_rgba(16, 16, 255, 0, 0)
        matte = run_key_pass(device, img, make_key_params())
        assert matte.mean() > 0.95, f"Pure red should be ~1, got {matte.mean():.3f}"

    def test_pure_blue_is_foreground(self, device):
        img = solid_rgba(16, 16, 0, 0, 255)
        matte = run_key_pass(device, img, make_key_params())
        assert matte.mean() > 0.95, f"Pure blue should be ~1, got {matte.mean():.3f}"

    def test_skin_tone_is_foreground(self, device):
        """Typical skin tone should be kept as foreground."""
        img = solid_rgba(16, 16, 200, 150, 120)
        matte = run_key_pass(device, img, make_key_params())
        assert matte.mean() > 0.9, f"Skin tone should be ~1, got {matte.mean():.3f}"

    def test_grey_preserved_by_sat_gate(self, device):
        """Mid-grey has no saturation => sat_gate should protect it (alpha ~1)."""
        img = solid_rgba(16, 16, 128, 128, 128)
        matte = run_key_pass(device, img, make_key_params(sat_gate=0.1))
        assert matte.mean() > 0.9, f"Grey should be protected, got {matte.mean():.3f}"

    def test_white_preserved_by_sat_gate(self, device):
        """White pixel should not be keyed."""
        img = solid_rgba(16, 16, 255, 255, 255)
        matte = run_key_pass(device, img, make_key_params(sat_gate=0.1))
        assert matte.mean() > 0.9, f"White should be protected, got {matte.mean():.3f}"

    def test_black_preserved_by_sat_gate(self, device):
        """Black pixel — low sat => kept."""
        img = solid_rgba(16, 16, 0, 0, 0)
        matte = run_key_pass(device, img, make_key_params(sat_gate=0.1))
        assert matte.mean() > 0.9, f"Black should be protected, got {matte.mean():.3f}"

    def test_greenish_intermediate_is_partial(self, device):
        """A greenish but not pure-green pixel should produce intermediate alpha."""
        img = solid_rgba(16, 16, 50, 200, 30)
        matte = run_key_pass(device, img, make_key_params(softness=0.3))
        mean = matte.mean()
        assert mean < 0.8, f"Greenish should not be fully foreground, got {mean:.3f}"

    def test_blue_screen_keying(self, device):
        """Using blue key color, blue pixels should be background."""
        img = solid_rgba(16, 16, 0, 0, 255)
        params = make_key_params(key_r=0.0, key_g=0.0, key_b=1.0)
        matte = run_key_pass(device, img, params)
        assert matte.mean() < 0.05, f"Blue on blue-key should be ~0, got {matte.mean():.3f}"

    def test_gamma_contracts_edges(self, device):
        """gamma > 1 should darken partial-alpha pixels."""
        img = solid_rgba(16, 16, 50, 200, 30)
        matte_g1 = run_key_pass(device, img, make_key_params(gamma=1.0))
        matte_g3 = run_key_pass(device, img, make_key_params(gamma=3.0))
        assert matte_g3.mean() < matte_g1.mean(), (
            f"gamma=3 ({matte_g3.mean():.3f}) should be darker than gamma=1 ({matte_g1.mean():.3f})"
        )

    def test_softness_zero_is_hard(self, device):
        """softness=0 should produce near-binary output."""
        img = np.zeros((16, 32, 4), dtype=np.uint8)
        img[:, :16] = [0, 255, 0, 255]
        img[:, 16:] = [255, 0, 0, 255]
        matte = run_key_pass(device, img, make_key_params(softness=0.0))
        assert matte[:, :16].mean() < 0.05
        assert matte[:, 16:].mean() > 0.95


class TestBlurShader:
    def test_uniform_image_unchanged(self, device):
        """Blurring a uniform image should produce the same value."""
        img = solid_rgba(16, 16, 128, 128, 128)
        result = run_blur_pass(device, img, radius=3)
        expected = 128.0 / 255.0
        assert abs(result.mean() - expected) < 0.02, (
            f"Expected ~{expected:.3f}, got {result.mean():.3f}"
        )

    def test_blur_reduces_contrast(self, device):
        """Blur should reduce contrast of a checkerboard pattern."""
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        for y in range(16):
            for x in range(16):
                v = 255 if (x + y) % 2 == 0 else 0
                img[y, x] = [v, v, v, 255]
        result = run_blur_pass(device, img, radius=2)
        original_std = 0.5
        assert result.std() < original_std, (
            f"Blur should reduce std from {original_std:.3f}, got {result.std():.3f}"
        )

    def test_radius_zero_is_identity(self, device):
        """Radius 0 should not change the image."""
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        img[:8, :] = [255, 255, 255, 255]
        img[8:, :] = [0, 0, 0, 255]
        result = run_blur_pass(device, img, radius=0)
        expected = np.zeros((16, 16), dtype=np.float32)
        expected[:8, :] = 1.0
        assert np.allclose(result, expected, atol=0.02)


class TestMorphologyShader:
    def _dot_image(self):
        """16x16 black image with a single white 3x3 block in center."""
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        img[7:10, 7:10] = [255, 255, 255, 255]
        return img

    def test_erode_shrinks_dot(self, device):
        img = self._dot_image()
        result = run_morph_pass(device, img, "erode")
        original_white = (img[:, :, 0] > 128).sum()
        eroded_white = (result > 0.5).sum()
        assert eroded_white < original_white, (
            f"Erode should shrink: {original_white} -> {eroded_white}"
        )

    def test_dilate_grows_dot(self, device):
        img = self._dot_image()
        result = run_morph_pass(device, img, "dilate")
        original_white = (img[:, :, 0] > 128).sum()
        dilated_white = (result > 0.5).sum()
        assert dilated_white > original_white, (
            f"Dilate should grow: {original_white} -> {dilated_white}"
        )

    def test_erode_uniform_white_unchanged(self, device):
        img = solid_rgba(16, 16, 255, 255, 255)
        result = run_morph_pass(device, img, "erode")
        assert result.min() > 0.95

    def test_dilate_uniform_black_unchanged(self, device):
        img = solid_rgba(16, 16, 0, 0, 0)
        result = run_morph_pass(device, img, "dilate")
        assert result.max() < 0.05
