"""Shared GPU test fixtures and shader helpers."""

import numpy as np
import pytest
import wgpu

WORKGROUP_SIZE = 16
BYTES_PER_PIXEL = 4
TEX_FORMAT = wgpu.TextureFormat.rgba8unorm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("No WebGPU adapter")
    return adapter.request_device_sync()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def load_wgsl(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def create_texture(device, w, h, usage):
    return device.create_texture(
        size=(w, h, 1),
        dimension=wgpu.TextureDimension.d2,
        format=TEX_FORMAT,
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


def readback_r_channel(device, tex, w, h) -> np.ndarray:
    """Read back R channel as float array (H, W) in [0, 1]."""
    bpr = w * BYTES_PER_PIXEL
    padded_bpr = ((bpr + 255) // 256) * 256
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
    rgba = raw[:, :bpr].reshape((h, w, 4)).copy()
    buf.unmap()
    return rgba[:, :, 0].astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Image factories
# ---------------------------------------------------------------------------


def solid_rgba(w, h, r, g, b, a=255):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :] = [r, g, b, a]
    return img


def matte_to_rgba(matte_float: np.ndarray) -> np.ndarray:
    """Convert (H,W) float [0,1] matte back to uint8 RGBA for piping between stages."""
    v = np.clip(matte_float * 255.0, 0, 255).astype(np.uint8)
    h, w = v.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = v
    rgba[:, :, 1] = v
    rgba[:, :, 2] = v
    rgba[:, :, 3] = 255
    return rgba


# ---------------------------------------------------------------------------
# Keying helpers
# ---------------------------------------------------------------------------


def normalize_key(r, g, b):
    s = r + g + b + 1e-4
    return r / s, g / s, b / s


def make_key_params(key_r=0.0, key_g=1.0, key_b=0.0, softness=0.3, gamma=1.0, sat_gate=0.1):
    knr, kng, knb = normalize_key(key_r, key_g, key_b)
    return np.array([knr, kng, knb, softness, gamma, sat_gate, 0.0, 0.0], dtype=np.float32)


def run_key_pass(device, input_rgba: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Run keying shader on input_rgba (H,W,4 uint8), return R channel as float (H,W)."""
    h, w = input_rgba.shape[:2]

    in_tex = create_texture(
        device, w, h, wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
    )
    out_tex = create_texture(
        device, w, h, wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC
    )
    upload_rgba(device, in_tex, input_rgba)

    params_buf = device.create_buffer_with_data(
        data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM
    )

    shader = device.create_shader_module(code=load_wgsl("alpha_hint.wgsl"))
    bgl = device.create_bind_group_layout(
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


# ---------------------------------------------------------------------------
# Blur helper
# ---------------------------------------------------------------------------


def run_blur_pass(device, input_rgba: np.ndarray, radius: int) -> np.ndarray:
    """Run horizontal + vertical blur on input (H,W,4 uint8), return R channel float."""
    h, w = input_rgba.shape[:2]

    in_tex = create_texture(
        device, w, h, wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
    )
    temp_tex = create_texture(
        device, w, h, wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
    )
    out_tex = create_texture(
        device, w, h, wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC
    )
    upload_rgba(device, in_tex, input_rgba)

    blur_params = np.array([radius, 0, 0, 0], dtype=np.int32)
    blur_buf = device.create_buffer_with_data(
        data=blur_params.tobytes(), usage=wgpu.BufferUsage.UNIFORM
    )
    shader = device.create_shader_module(code=load_wgsl("blur.wgsl"))
    bgl = device.create_bind_group_layout(
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
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    h_pipe = device.create_compute_pipeline(
        layout=layout, compute={"module": shader, "entry_point": "blur_h"}
    )
    v_pipe = device.create_compute_pipeline(
        layout=layout, compute={"module": shader, "entry_point": "blur_v"}
    )

    bg_h = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": in_tex.create_view()},
            {"binding": 1, "resource": temp_tex.create_view()},
            {"binding": 2, "resource": {"buffer": blur_buf, "offset": 0, "size": 16}},
        ],
    )
    bg_v = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": temp_tex.create_view()},
            {"binding": 1, "resource": out_tex.create_view()},
            {"binding": 2, "resource": {"buffer": blur_buf, "offset": 0, "size": 16}},
        ],
    )

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


# ---------------------------------------------------------------------------
# Morphology helper
# ---------------------------------------------------------------------------


def run_morph_pass(device, input_rgba: np.ndarray, entry_point: str) -> np.ndarray:
    """Run one morphology pass (erode or dilate) on input, return R channel float."""
    h, w = input_rgba.shape[:2]

    in_tex = create_texture(
        device, w, h, wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
    )
    out_tex = create_texture(
        device, w, h, wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC
    )
    upload_rgba(device, in_tex, input_rgba)

    shader = device.create_shader_module(code=load_wgsl("morphology.wgsl"))
    bgl = device.create_bind_group_layout(
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
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
        compute={"module": shader, "entry_point": entry_point},
    )
    bg = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": in_tex.create_view()},
            {"binding": 1, "resource": out_tex.create_view()},
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
