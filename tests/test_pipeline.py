"""Pipeline combo tests — compose shader helpers sequentially."""

import numpy as np
from conftest import (
    make_key_params,
    matte_to_rgba,
    run_blur_pass,
    run_key_pass,
    run_morph_pass,
)


class TestPipelineCombinations:
    """Test that various stage combinations produce valid output."""

    def _green_with_red_block(self):
        """32x32 green image with a 16x16 red block in center."""
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[:, :] = [0, 255, 0, 255]
        rgba[8:24, 8:24] = [255, 0, 0, 255]
        return rgba

    def test_key_only(self, device):
        matte = run_key_pass(device, self._green_with_red_block(), make_key_params())
        assert matte[0, 0] < 0.05, "Green corner should be ~0"
        assert matte[16, 16] > 0.95, "Red center should be ~1"

    def test_key_then_blur(self, device):
        matte = run_key_pass(device, self._green_with_red_block(), make_key_params())
        blurred = run_blur_pass(device, matte_to_rgba(matte), radius=2)
        assert blurred[16, 16] > 0.7, "Center should still be mostly white after blur"

    def test_key_then_erode(self, device):
        matte = run_key_pass(device, self._green_with_red_block(), make_key_params())
        eroded = run_morph_pass(device, matte_to_rgba(matte), "erode")
        assert eroded[16, 16] > 0.95, "Interior should survive 1 erode"
        assert eroded[0, 0] < 0.05, "Background should stay black"

    def test_key_then_dilate(self, device):
        matte = run_key_pass(device, self._green_with_red_block(), make_key_params())
        dilated = run_morph_pass(device, matte_to_rgba(matte), "dilate")
        assert dilated[16, 16] > 0.95, "Subject center should be white"

    def test_key_blur_erode_dilate(self, device):
        img = self._green_with_red_block()
        matte = run_key_pass(device, img, make_key_params())
        blurred = run_blur_pass(device, matte_to_rgba(matte), radius=1)
        eroded = run_morph_pass(device, matte_to_rgba(blurred), "erode")
        dilated = run_morph_pass(device, matte_to_rgba(eroded), "dilate")
        assert dilated[16, 16] > 0.0, "Center should have some white"
        assert dilated.shape == (32, 32), "Output shape should match input"

    def test_output_always_valid_range(self, device):
        rgba = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        matte = run_key_pass(device, rgba, make_key_params())
        blurred = run_blur_pass(device, matte_to_rgba(matte), radius=1)
        eroded = run_morph_pass(device, matte_to_rgba(blurred), "erode")
        dilated = run_morph_pass(device, matte_to_rgba(eroded), "dilate")
        assert dilated.min() >= 0.0
        assert dilated.max() <= 1.0
