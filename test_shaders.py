"""GPU shader tests — run keying, blur, and morphology on synthetic pixels."""

import numpy as np
import pytest
import wgpu

WORKGROUP_SIZE = 16


def load_wgsl(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("No WebGPU adapter")
    return adapter.request_device_sync()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BYTES_PER_PIXEL = 4


def make_rgba_texture(device, width, height, *, readable=False):
    usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
    if readable:
        usage |= wgpu.TextureUsage.COPY_SRC
    return device.create_texture(
        size=(width, height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=usage,
    )


def make_storage_texture(device, width, height, *, readable=False):
    usage = wgpu.TextureUsage.STORAGE_BINDING
    if readable:
        usage |= wgpu.TextureUsage.COPY_SRC
    return device.create_texture(
        size=(width, height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=usage,
    )


def upload_rgba(device, tex, data: np.ndarray):
    """Upload uint8 RGBA array (H, W, 4) to texture."""
    h, w, _ = data.shape
    bpr = w * BYTES_PER_PIXEL
    device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        data.tobytes(),
        {"bytes_per_row": bpr, "rows_per_image": h},
        (w, h, 1),
    )


def readback_r_channel(device, tex, width, height) -> np.ndarray:
    """Read back R channel as float array (H, W) in [0, 1]."""
    bpr = width * BYTES_PER_PIXEL
    padded_bpr = ((bpr + 255) // 256) * 256
    buf = device.create_buffer(
        size=padded_bpr * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    enc = device.create_command_encoder()
    enc.copy_texture_to_buffer(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf, "offset": 0, "bytes_per_row": padded_bpr, "rows_per_image": height},
        (width, height, 1),
    )
    device.queue.submit([enc.finish()])
    buf.map_sync(mode=wgpu.MapMode.READ)
    raw = np.frombuffer(buf.read_mapped(), dtype=np.uint8).reshape((height, padded_bpr))
    rgba = raw[:, :bpr].reshape((height, width, 4)).copy()
    buf.unmap()
    return rgba[:, :, 0].astype(np.float32) / 255.0


def normalize_key(r, g, b):
    s = r + g + b + 1e-4
    return r / s, g / s, b / s


def make_key_params(key_r=0.0, key_g=1.0, key_b=0.0, softness=0.3, gamma=1.0, sat_gate=0.1):
    knr, kng, knb = normalize_key(key_r, key_g, key_b)
    return np.array([knr, kng, knb, softness, gamma, sat_gate, 0.0, 0.0], dtype=np.float32)


def run_key_pass(device, input_rgba: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Run the keying shader on input_rgba (H,W,4 uint8), return R channel as float (H,W)."""
    h, w = input_rgba.shape[:2]

    in_tex = make_rgba_texture(device, w, h)
    out_tex = make_storage_texture(device, w, h, readable=True)
    upload_rgba(device, in_tex, input_rgba)

    params_buf = device.create_buffer_with_data(
        data=params.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader = device.create_shader_module(code=load_wgsl("alpha_hint.wgsl"))

    bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.float}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba8unorm,
                                 "view_dimension": wgpu.TextureViewDimension.d2}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]
    )
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
        compute={"module": shader, "entry_point": "main"},
    )
    bg = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": in_tex.create_view()},
            {"binding": 1, "resource": out_tex.create_view()},
            {"binding": 2, "resource": {"buffer": params_buf, "offset": 0, "size": 32}},
        ],
    )

    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg, [], 0, 999999)
    cp.dispatch_workgroups((w + 15) // 16, (h + 15) // 16, 1)
    cp.end()
    device.queue.submit([enc.finish()])

    return readback_r_channel(device, out_tex, w, h)


def run_blur_pass(device, input_rgba: np.ndarray, radius: int) -> np.ndarray:
    """Run horizontal + vertical blur on input (H,W,4 uint8), return R channel float."""
    h, w = input_rgba.shape[:2]

    in_tex = make_rgba_texture(device, w, h)
    temp_tex = make_storage_texture(device, w, h)
    # temp also needs TEXTURE_BINDING for blur_v to read it
    temp_tex = device.create_texture(
        size=(w, h, 1), dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    out_tex = device.create_texture(
        size=(w, h, 1), dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
    )
    upload_rgba(device, in_tex, input_rgba)

    blur_params = np.array([radius, 0, 0, 0], dtype=np.int32)
    blur_buf = device.create_buffer_with_data(
        data=blur_params.tobytes(), usage=wgpu.BufferUsage.UNIFORM,
    )
    shader = device.create_shader_module(code=load_wgsl("blur.wgsl"))
    bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.float}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba8unorm,
                                 "view_dimension": wgpu.TextureViewDimension.d2}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    h_pipe = device.create_compute_pipeline(
        layout=layout, compute={"module": shader, "entry_point": "blur_h"})
    v_pipe = device.create_compute_pipeline(
        layout=layout, compute={"module": shader, "entry_point": "blur_v"})

    bg_h = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": in_tex.create_view()},
        {"binding": 1, "resource": temp_tex.create_view()},
        {"binding": 2, "resource": {"buffer": blur_buf, "offset": 0, "size": 16}},
    ])
    bg_v = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": temp_tex.create_view()},
        {"binding": 1, "resource": out_tex.create_view()},
        {"binding": 2, "resource": {"buffer": blur_buf, "offset": 0, "size": 16}},
    ])

    gx, gy = (w + 15) // 16, (h + 15) // 16
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(h_pipe)
    cp.set_bind_group(0, bg_h, [], 0, 999999)
    cp.dispatch_workgroups(gx, gy, 1)
    cp.end()
    cp2 = enc.begin_compute_pass()
    cp2.set_pipeline(v_pipe)
    cp2.set_bind_group(0, bg_v, [], 0, 999999)
    cp2.dispatch_workgroups(gx, gy, 1)
    cp2.end()
    device.queue.submit([enc.finish()])

    return readback_r_channel(device, out_tex, w, h)


def run_morph_pass(device, input_rgba: np.ndarray, entry_point: str) -> np.ndarray:
    """Run one morphology pass (erode or dilate) on input, return R channel float."""
    h, w = input_rgba.shape[:2]

    in_tex = make_rgba_texture(device, w, h)
    out_tex = device.create_texture(
        size=(w, h, 1), dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
    )
    upload_rgba(device, in_tex, input_rgba)

    shader = device.create_shader_module(code=load_wgsl("morphology.wgsl"))
    bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "texture": {"sample_type": wgpu.TextureSampleType.float}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
         "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                             "format": wgpu.TextureFormat.rgba8unorm,
                             "view_dimension": wgpu.TextureViewDimension.d2}},
    ])
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
        compute={"module": shader, "entry_point": entry_point},
    )
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": in_tex.create_view()},
        {"binding": 1, "resource": out_tex.create_view()},
    ])

    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg, [], 0, 999999)
    cp.dispatch_workgroups((w + 15) // 16, (h + 15) // 16, 1)
    cp.end()
    device.queue.submit([enc.finish()])

    return readback_r_channel(device, out_tex, w, h)


def solid_rgba(width, height, r, g, b, a=255):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :] = [r, g, b, a]
    return img


# ---------------------------------------------------------------------------
# Keying shader tests
# ---------------------------------------------------------------------------


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
        # Not fully background, not fully foreground
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
        # Make a half-green, half-red image
        img = np.zeros((16, 32, 4), dtype=np.uint8)
        img[:, :16] = [0, 255, 0, 255]  # green left half
        img[:, 16:] = [255, 0, 0, 255]  # red right half
        matte = run_key_pass(device, img, make_key_params(softness=0.0))
        green_vals = matte[:, :16]
        red_vals = matte[:, 16:]
        assert green_vals.mean() < 0.05
        assert red_vals.mean() > 0.95


# ---------------------------------------------------------------------------
# Blur shader tests
# ---------------------------------------------------------------------------


class TestBlurShader:
    def test_uniform_image_unchanged(self, device):
        """Blurring a uniform image should produce the same value."""
        img = solid_rgba(16, 16, 128, 128, 128)
        result = run_blur_pass(device, img, radius=3)
        expected = 128.0 / 255.0
        assert abs(result.mean() - expected) < 0.02, f"Expected ~{expected:.3f}, got {result.mean():.3f}"

    def test_blur_reduces_contrast(self, device):
        """Blur should reduce contrast of a checkerboard pattern."""
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        for y in range(16):
            for x in range(16):
                v = 255 if (x + y) % 2 == 0 else 0
                img[y, x] = [v, v, v, 255]
        result = run_blur_pass(device, img, radius=2)
        # Std dev should be lower than original (which is 0.5)
        original_std = 0.5
        assert result.std() < original_std, f"Blur should reduce std from {original_std:.3f}, got {result.std():.3f}"

    def test_radius_zero_is_identity(self, device):
        """Radius 0 should not change the image."""
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        img[:8, :] = [255, 255, 255, 255]
        img[8:, :] = [0, 0, 0, 255]
        result = run_blur_pass(device, img, radius=0)
        expected = np.zeros((16, 16), dtype=np.float32)
        expected[:8, :] = 1.0
        assert np.allclose(result, expected, atol=0.02)


# ---------------------------------------------------------------------------
# Morphology shader tests
# ---------------------------------------------------------------------------


class TestMorphologyShader:
    def _dot_image(self):
        """16x16 black image with a single white 3x3 block in center."""
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        img[7:10, 7:10] = [255, 255, 255, 255]
        return img

    def test_erode_shrinks_dot(self, device):
        """Erode should shrink the white region."""
        img = self._dot_image()
        result = run_morph_pass(device, img, "erode")
        original_white = (img[:, :, 0] > 128).sum()
        eroded_white = (result > 0.5).sum()
        assert eroded_white < original_white, (
            f"Erode should shrink: {original_white} -> {eroded_white}"
        )

    def test_dilate_grows_dot(self, device):
        """Dilate should expand the white region."""
        img = self._dot_image()
        result = run_morph_pass(device, img, "dilate")
        original_white = (img[:, :, 0] > 128).sum()
        dilated_white = (result > 0.5).sum()
        assert dilated_white > original_white, (
            f"Dilate should grow: {original_white} -> {dilated_white}"
        )

    def test_erode_uniform_white_unchanged(self, device):
        """Eroding an all-white image should stay all-white."""
        img = solid_rgba(16, 16, 255, 255, 255)
        result = run_morph_pass(device, img, "erode")
        assert result.min() > 0.95

    def test_dilate_uniform_black_unchanged(self, device):
        """Dilating an all-black image should stay all-black."""
        img = solid_rgba(16, 16, 0, 0, 0)
        result = run_morph_pass(device, img, "dilate")
        assert result.max() < 0.05
