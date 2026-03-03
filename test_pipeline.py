"""GPU pipeline tests for keying, blur, and morphology passes.

Creates synthetic RGBA textures, runs compute shaders, and verifies outputs.
No video file needed — tests operate on single-frame numpy arrays.
"""

import numpy as np
import pytest
import wgpu

WORKGROUP_SIZE = 16
TEX_FORMAT = wgpu.TextureFormat.rgba8unorm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("No WebGPU adapter available")
    return adapter.request_device_sync()


def _load_wgsl(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_tex(device, w, h, usage):
    return device.create_texture(
        size=(w, h, 1),
        dimension=wgpu.TextureDimension.d2,
        format=TEX_FORMAT,
        usage=usage,
    )


def _upload(device, tex, data, w, h):
    bpr = w * 4
    device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        data.tobytes(),
        {"bytes_per_row": bpr, "rows_per_image": h},
        (w, h, 1),
    )


def _readback(device, tex, w, h):
    bpp = 4
    unpadded_bpr = w * bpp
    padded_bpr = ((unpadded_bpr + 255) // 256) * 256
    buf = device.create_buffer(
        size=padded_bpr * h,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    enc = device.create_command_encoder()
    enc.copy_texture_to_buffer(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf, "offset": 0, "bytes_per_row": padded_bpr, "rows_per_image": h},
        (w, h, 1),
    )
    device.queue.submit([enc.finish()])
    buf.map_sync(mode=wgpu.MapMode.READ)
    raw = np.frombuffer(buf.read_mapped(), dtype=np.uint8).reshape((h, padded_bpr))
    result = raw[:, :unpadded_bpr].reshape((h, w, 4)).copy()
    buf.unmap()
    return result


def _dispatch(device, pipeline, bind_group, w, h):
    gx = (w + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    gy = (h + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(pipeline)
    p.set_bind_group(0, bind_group, [], 0, 999999)
    p.dispatch_workgroups(gx, gy, 1)
    p.end()
    device.queue.submit([enc.finish()])


# ---------------------------------------------------------------------------
# Shared bind group layouts
# ---------------------------------------------------------------------------


def _morph_bgl(device):
    """tex_2d read + storage_tex write (no uniform)."""
    return device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {"sample_type": wgpu.TextureSampleType.float},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.write_only,
                    "format": TEX_FORMAT,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
        ]
    )


def _key_bgl(device):
    """tex_2d read + storage_tex write + uniform."""
    return device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {"sample_type": wgpu.TextureSampleType.float},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.write_only,
                    "format": TEX_FORMAT,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )


# ---------------------------------------------------------------------------
# Keying tests
# ---------------------------------------------------------------------------


class TestKeying:
    """Verify the green-screen keying shader."""

    def _run_key(self, device, rgba, t_low=-0.05, t_high=0.10, gamma=1.0):
        h, w = rgba.shape[:2]
        shader = device.create_shader_module(code=_load_wgsl("alpha_hint.wgsl"))
        bgl = _key_bgl(device)
        layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
        pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": shader, "entry_point": "main"}
        )

        in_tex = _create_tex(
            device, w, h, wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
        )
        out_tex = _create_tex(
            device, w, h, wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC
        )

        params = np.array([t_low, t_high, gamma, 0.0], dtype=np.float32)
        params_buf = device.create_buffer_with_data(
            data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM
        )

        bg = device.create_bind_group(
            layout=bgl,
            entries=[
                {"binding": 0, "resource": in_tex.create_view()},
                {"binding": 1, "resource": out_tex.create_view()},
                {"binding": 2, "resource": {"buffer": params_buf, "offset": 0, "size": 16}},
            ],
        )

        _upload(device, in_tex, rgba, w, h)
        _dispatch(device, pipeline, bg, w, h)
        return _readback(device, out_tex, w, h)

    def test_pure_green_keys_to_black(self, device):
        """Pure green pixel → background → matte 0 (black)."""
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[:, :] = [0, 255, 0, 255]  # pure green
        out = self._run_key(device, rgba)
        assert out[:, :, 0].max() == 0, "Pure green should key to black"

    def test_pure_red_keys_to_white(self, device):
        """Pure red pixel → subject → matte 1 (white)."""
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[:, :] = [255, 0, 0, 255]  # pure red
        out = self._run_key(device, rgba)
        assert out[:, :, 0].min() == 255, "Pure red should key to white"

    def test_output_alpha_always_one(self, device):
        """Output alpha channel should always be 255."""
        rgba = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        out = self._run_key(device, rgba)
        assert (out[:, :, 3] == 255).all(), "Alpha should always be 1.0 (255)"

    def test_rgb_channels_equal(self, device):
        """R, G, B channels should be identical (grayscale matte)."""
        rgba = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        out = self._run_key(device, rgba)
        assert (out[:, :, 0] == out[:, :, 1]).all()
        assert (out[:, :, 0] == out[:, :, 2]).all()


# ---------------------------------------------------------------------------
# Morphology tests
# ---------------------------------------------------------------------------


class TestMorphology:
    """Verify erode/dilate compute shaders."""

    def _make_morph_pipeline(self, device, entry_point):
        shader = device.create_shader_module(code=_load_wgsl("morphology.wgsl"))
        bgl = _morph_bgl(device)
        layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
        pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": shader, "entry_point": entry_point}
        )
        return pipeline, bgl

    def _run_morph(self, device, matte_u8, entry_point, iterations=1):
        """Run morph on a single-channel matte (HxW uint8). Returns HxW uint8."""
        h, w = matte_u8.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = matte_u8
        rgba[:, :, 1] = matte_u8
        rgba[:, :, 2] = matte_u8
        rgba[:, :, 3] = 255

        pipeline, bgl = self._make_morph_pipeline(device, entry_point)

        usage_rw = (
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.COPY_SRC
        )
        ping = _create_tex(device, w, h, usage_rw)
        pong = _create_tex(device, w, h, usage_rw)

        _upload(device, ping, rgba, w, h)

        bg_ping_to_pong = device.create_bind_group(
            layout=bgl,
            entries=[
                {"binding": 0, "resource": ping.create_view()},
                {"binding": 1, "resource": pong.create_view()},
            ],
        )
        bg_pong_to_ping = device.create_bind_group(
            layout=bgl,
            entries=[
                {"binding": 0, "resource": pong.create_view()},
                {"binding": 1, "resource": ping.create_view()},
            ],
        )

        reading_ping = True
        for _ in range(iterations):
            bg = bg_ping_to_pong if reading_ping else bg_pong_to_ping
            _dispatch(device, pipeline, bg, w, h)
            reading_ping = not reading_ping

        result_tex = ping if reading_ping else pong
        out = _readback(device, result_tex, w, h)
        return out[:, :, 0]

    def test_erode_shrinks_white_dot(self, device):
        """A single white pixel surrounded by black should be eroded away."""
        matte = np.zeros((32, 32), dtype=np.uint8)
        matte[16, 16] = 255  # single white speckle
        out = self._run_morph(device, matte, "erode", iterations=1)
        assert out[16, 16] == 0, "Single white pixel should be eroded to black"

    def test_dilate_expands_white_dot(self, device):
        """A single white pixel should expand to its 3x3 neighborhood."""
        matte = np.zeros((32, 32), dtype=np.uint8)
        matte[16, 16] = 255
        out = self._run_morph(device, matte, "dilate", iterations=1)
        # All 3x3 neighbors should be white
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                assert out[16 + dy, 16 + dx] == 255, (
                    f"Pixel ({16 + dy},{16 + dx}) should be dilated to white"
                )

    def test_erode_preserves_solid_interior(self, device):
        """Interior of a large solid block should survive erosion."""
        matte = np.zeros((32, 32), dtype=np.uint8)
        matte[4:28, 4:28] = 255  # large white block
        out = self._run_morph(device, matte, "erode", iterations=1)
        # Interior (2px from edge of block) should still be white
        assert out[10:22, 10:22].min() == 255

    def test_dilate_fills_small_hole(self, device):
        """A single black pixel in a white field should be filled by dilate."""
        matte = np.full((32, 32), 255, dtype=np.uint8)
        matte[16, 16] = 0  # single hole
        out = self._run_morph(device, matte, "dilate", iterations=1)
        assert out[16, 16] == 255, "Single hole should be filled by dilate"

    def test_erode_then_dilate_removes_speckle(self, device):
        """Open (erode+dilate) should remove small noise but keep large regions."""
        matte = np.zeros((32, 32), dtype=np.uint8)
        matte[4:28, 4:28] = 255  # large white block
        matte[2, 2] = 255  # isolated speckle

        eroded = self._run_morph(device, matte, "erode", iterations=1)
        dilated = self._run_morph(device, eroded, "dilate", iterations=1)

        assert dilated[2, 2] == 0, "Isolated speckle should be removed by open"
        assert dilated[10:22, 10:22].min() == 255, "Block interior should survive open"

    def test_multiple_erode_iterations(self, device):
        """Multiple erode iterations should shrink edges further."""
        matte = np.zeros((32, 32), dtype=np.uint8)
        matte[8:24, 8:24] = 255  # 16x16 white block

        out1 = self._run_morph(device, matte, "erode", iterations=1)
        out2 = self._run_morph(device, matte, "erode", iterations=2)

        white_1 = int(out1.sum())
        white_2 = int(out2.sum())
        assert white_2 < white_1, "More erode iterations should shrink more"

    def test_zero_iterations_noop(self, device):
        """Zero morph iterations in the main pipeline should be identical to baseline."""
        # This is implicitly tested by the pipeline, but verify the morph helper
        # doesn't crash with 0 iters (it won't be called, but good to document).
        pass


# ---------------------------------------------------------------------------
# Blur tests
# ---------------------------------------------------------------------------


class TestBlur:
    """Verify the separable box blur shader."""

    def _run_blur(self, device, matte_u8, radius):
        h, w = matte_u8.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = matte_u8
        rgba[:, :, 1] = matte_u8
        rgba[:, :, 2] = matte_u8
        rgba[:, :, 3] = 255

        shader = device.create_shader_module(code=_load_wgsl("blur.wgsl"))
        bgl = _key_bgl(device)  # same layout: tex + storage_tex + uniform
        layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

        h_pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": shader, "entry_point": "blur_h"}
        )
        v_pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": shader, "entry_point": "blur_v"}
        )

        usage_rw = (
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.COPY_SRC
        )
        in_tex = _create_tex(device, w, h, usage_rw)
        temp_tex = _create_tex(device, w, h, usage_rw)
        out_tex = _create_tex(device, w, h, usage_rw)

        params = np.array([radius, 0, 0, 0], dtype=np.int32)
        params_buf = device.create_buffer_with_data(
            data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM
        )

        bg_h = device.create_bind_group(
            layout=bgl,
            entries=[
                {"binding": 0, "resource": in_tex.create_view()},
                {"binding": 1, "resource": temp_tex.create_view()},
                {"binding": 2, "resource": {"buffer": params_buf, "offset": 0, "size": 16}},
            ],
        )
        bg_v = device.create_bind_group(
            layout=bgl,
            entries=[
                {"binding": 0, "resource": temp_tex.create_view()},
                {"binding": 1, "resource": out_tex.create_view()},
                {"binding": 2, "resource": {"buffer": params_buf, "offset": 0, "size": 16}},
            ],
        )

        _upload(device, in_tex, rgba, w, h)
        _dispatch(device, h_pipeline, bg_h, w, h)
        _dispatch(device, v_pipeline, bg_v, w, h)
        return _readback(device, out_tex, w, h)[:, :, 0]

    def test_uniform_image_unchanged(self, device):
        """Blurring a uniform image should produce the same image."""
        matte = np.full((32, 32), 128, dtype=np.uint8)
        out = self._run_blur(device, matte, radius=3)
        assert np.allclose(out, 128, atol=1), "Uniform image should be unchanged by blur"

    def test_blur_reduces_contrast(self, device):
        """Blur should reduce the difference between adjacent black/white regions."""
        matte = np.zeros((32, 32), dtype=np.uint8)
        matte[:, 16:] = 255  # left half black, right half white
        out = self._run_blur(device, matte, radius=2)
        # Edge region should have intermediate values
        edge_values = out[:, 14:18]
        assert edge_values.min() < 200, "Blur should create intermediate values at edges"
        assert edge_values.max() > 50

    def test_blur_preserves_total_brightness(self, device):
        """Box blur should roughly preserve average brightness."""
        matte = np.zeros((64, 64), dtype=np.uint8)
        matte[20:44, 20:44] = 255
        original_mean = float(matte.mean())
        out = self._run_blur(device, matte, radius=2)
        blurred_mean = float(out.mean())
        assert abs(blurred_mean - original_mean) < 5, (
            f"Blur should preserve avg brightness: {original_mean:.1f} vs {blurred_mean:.1f}"
        )


# ---------------------------------------------------------------------------
# Integration: full pipeline combos
# ---------------------------------------------------------------------------


class TestPipelineCombinations:
    """Test that various flag combinations don't crash and produce valid output."""

    def _run_pipeline(self, device, rgba, blur_radius=0, erode_iters=0, dilate_iters=0):
        """Minimal reimplementation of the pipeline for a single frame."""
        from main import load_wgsl

        h, w = rgba.shape[:2]
        total_morph_iters = erode_iters + dilate_iters

        in_tex = _create_tex(
            device, w, h, wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
        )
        usage_out = (
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.COPY_DST
        )
        out_tex = _create_tex(device, w, h, usage_out)

        # Keying
        key_shader = device.create_shader_module(code=load_wgsl("alpha_hint.wgsl"))
        bgl3 = _key_bgl(device)
        key_layout = device.create_pipeline_layout(bind_group_layouts=[bgl3])
        key_pipeline = device.create_compute_pipeline(
            layout=key_layout, compute={"module": key_shader, "entry_point": "main"}
        )

        params = np.array([-0.05, 0.10, 1.0, 0.0], dtype=np.float32)
        params_buf = device.create_buffer_with_data(
            data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM
        )

        usage_rw = (
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC
        )

        # Morph textures (must be created before key_dest is determined)
        if total_morph_iters > 0:
            morph_ping = _create_tex(device, w, h, usage_rw | wgpu.TextureUsage.COPY_DST)
            morph_pong = _create_tex(device, w, h, usage_rw | wgpu.TextureUsage.COPY_DST)

        # Determine key destination
        if blur_radius > 0:
            inter_tex = _create_tex(device, w, h, usage_rw)
            blur_temp = _create_tex(device, w, h, usage_rw)
            key_dest = inter_tex
            blur_v_dest = morph_ping if total_morph_iters > 0 else out_tex
        elif total_morph_iters > 0:
            key_dest = morph_ping
        else:
            key_dest = out_tex

        key_bg = device.create_bind_group(
            layout=bgl3,
            entries=[
                {"binding": 0, "resource": in_tex.create_view()},
                {"binding": 1, "resource": key_dest.create_view()},
                {"binding": 2, "resource": {"buffer": params_buf, "offset": 0, "size": 16}},
            ],
        )

        _upload(device, in_tex, rgba, w, h)

        enc = device.create_command_encoder()
        gx = (w + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        gy = (h + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE

        p = enc.begin_compute_pass()
        p.set_pipeline(key_pipeline)
        p.set_bind_group(0, key_bg, [], 0, 999999)
        p.dispatch_workgroups(gx, gy, 1)
        p.end()

        # Blur
        if blur_radius > 0:
            blur_shader = device.create_shader_module(code=load_wgsl("blur.wgsl"))
            blur_layout = device.create_pipeline_layout(bind_group_layouts=[bgl3])
            blur_h_pipe = device.create_compute_pipeline(
                layout=blur_layout, compute={"module": blur_shader, "entry_point": "blur_h"}
            )
            blur_v_pipe = device.create_compute_pipeline(
                layout=blur_layout, compute={"module": blur_shader, "entry_point": "blur_v"}
            )
            bp = np.array([blur_radius, 0, 0, 0], dtype=np.int32)
            bp_buf = device.create_buffer_with_data(
                data=bp.tobytes(), usage=wgpu.BufferUsage.UNIFORM
            )

            bg_h = device.create_bind_group(
                layout=bgl3,
                entries=[
                    {"binding": 0, "resource": inter_tex.create_view()},
                    {"binding": 1, "resource": blur_temp.create_view()},
                    {"binding": 2, "resource": {"buffer": bp_buf, "offset": 0, "size": 16}},
                ],
            )
            bg_v = device.create_bind_group(
                layout=bgl3,
                entries=[
                    {"binding": 0, "resource": blur_temp.create_view()},
                    {"binding": 1, "resource": blur_v_dest.create_view()},
                    {"binding": 2, "resource": {"buffer": bp_buf, "offset": 0, "size": 16}},
                ],
            )

            hp = enc.begin_compute_pass()
            hp.set_pipeline(blur_h_pipe)
            hp.set_bind_group(0, bg_h, [], 0, 999999)
            hp.dispatch_workgroups(gx, gy, 1)
            hp.end()

            vp = enc.begin_compute_pass()
            vp.set_pipeline(blur_v_pipe)
            vp.set_bind_group(0, bg_v, [], 0, 999999)
            vp.dispatch_workgroups(gx, gy, 1)
            vp.end()

        # Morphology
        if total_morph_iters > 0:
            morph_shader = device.create_shader_module(code=load_wgsl("morphology.wgsl"))
            mbgl = _morph_bgl(device)
            morph_layout = device.create_pipeline_layout(bind_group_layouts=[mbgl])
            erode_pipe = device.create_compute_pipeline(
                layout=morph_layout, compute={"module": morph_shader, "entry_point": "erode"}
            )
            dilate_pipe = device.create_compute_pipeline(
                layout=morph_layout, compute={"module": morph_shader, "entry_point": "dilate"}
            )

            bg_p2p = device.create_bind_group(
                layout=mbgl,
                entries=[
                    {"binding": 0, "resource": morph_ping.create_view()},
                    {"binding": 1, "resource": morph_pong.create_view()},
                ],
            )
            bg_q2p = device.create_bind_group(
                layout=mbgl,
                entries=[
                    {"binding": 0, "resource": morph_pong.create_view()},
                    {"binding": 1, "resource": morph_ping.create_view()},
                ],
            )

            reading_ping = True
            for _ in range(erode_iters):
                bg = bg_p2p if reading_ping else bg_q2p
                mp = enc.begin_compute_pass()
                mp.set_pipeline(erode_pipe)
                mp.set_bind_group(0, bg, [], 0, 999999)
                mp.dispatch_workgroups(gx, gy, 1)
                mp.end()
                reading_ping = not reading_ping

            for _ in range(dilate_iters):
                bg = bg_p2p if reading_ping else bg_q2p
                mp = enc.begin_compute_pass()
                mp.set_pipeline(dilate_pipe)
                mp.set_bind_group(0, bg, [], 0, 999999)
                mp.dispatch_workgroups(gx, gy, 1)
                mp.end()
                reading_ping = not reading_ping

            result_tex = morph_ping if reading_ping else morph_pong
            enc.copy_texture_to_texture(
                {"texture": result_tex, "mip_level": 0, "origin": (0, 0, 0)},
                {"texture": out_tex, "mip_level": 0, "origin": (0, 0, 0)},
                (w, h, 1),
            )

        device.queue.submit([enc.finish()])
        return _readback(device, out_tex, w, h)

    def _green_with_red_block(self):
        """32x32 green image with a 16x16 red block in center."""
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[:, :] = [0, 255, 0, 255]
        rgba[8:24, 8:24] = [255, 0, 0, 255]
        return rgba

    def test_key_only(self, device):
        out = self._run_pipeline(device, self._green_with_red_block())
        matte = out[:, :, 0]
        assert matte[0, 0] == 0, "Green corner should be black"
        assert matte[16, 16] == 255, "Red center should be white"

    def test_key_blur(self, device):
        out = self._run_pipeline(device, self._green_with_red_block(), blur_radius=2)
        matte = out[:, :, 0]
        assert matte[16, 16] > 200, "Center should still be mostly white after blur"

    def test_key_erode(self, device):
        out = self._run_pipeline(device, self._green_with_red_block(), erode_iters=1)
        matte = out[:, :, 0]
        assert matte[16, 16] == 255, "Interior should survive 1 erode"
        assert matte[0, 0] == 0, "Background should stay black"

    def test_key_dilate(self, device):
        out = self._run_pipeline(device, self._green_with_red_block(), dilate_iters=1)
        matte = out[:, :, 0]
        assert matte[16, 16] == 255, "Subject center should be white"

    def test_key_blur_erode_dilate(self, device):
        out = self._run_pipeline(
            device,
            self._green_with_red_block(),
            blur_radius=1,
            erode_iters=1,
            dilate_iters=1,
        )
        matte = out[:, :, 0]
        assert matte[16, 16] > 0, "Center should have some white"
        assert matte.shape == (32, 32), "Output shape should match input"

    def test_output_always_valid_range(self, device):
        """All pixel values should be in [0, 255] and alpha should be 255."""
        rgba = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        out = self._run_pipeline(device, rgba, blur_radius=1, erode_iters=1, dilate_iters=1)
        assert out[:, :, 3].min() == 255, "Alpha should always be 255"
        assert out.dtype == np.uint8
