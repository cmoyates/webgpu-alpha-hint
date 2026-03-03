"""Shared GPU test fixtures and shader helpers."""

import numpy as np
import pytest
import wgpu

from webgpu_alpha_hint.gpu import TEX_FORMAT, create_texture, readback_r_channel, upload_rgba
from webgpu_alpha_hint.shader_utils import WORKGROUP_SIZE, load_wgsl

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
# Image factories
# ---------------------------------------------------------------------------


def solid_rgba(width, height, red, green, blue, alpha=255):
    """Create a uniform-color uint8 RGBA image."""
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :] = [red, green, blue, alpha]
    return image


def matte_to_rgba(matte_float: np.ndarray) -> np.ndarray:
    """Convert (H,W) float [0,1] matte back to uint8 RGBA for piping between stages."""
    matte_uint8 = np.clip(matte_float * 255.0, 0, 255).astype(np.uint8)
    frame_height, frame_width = matte_uint8.shape
    rgba = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
    rgba[:, :, 0] = matte_uint8
    rgba[:, :, 1] = matte_uint8
    rgba[:, :, 2] = matte_uint8
    rgba[:, :, 3] = 255
    return rgba


# ---------------------------------------------------------------------------
# Keying helpers
# ---------------------------------------------------------------------------


def normalize_key(red, green, blue):
    """Project key color into chromaticity space to match shader convention."""
    channel_sum = red + green + blue + 1e-4
    return red / channel_sum, green / channel_sum, blue / channel_sum


def make_key_params(key_r=0.0, key_g=1.0, key_b=0.0, softness=0.3, gamma=1.0, sat_gate=0.1):
    """Build the keying uniform buffer as a float32 array matching the shader's Params struct."""
    key_norm_r, key_norm_g, key_norm_b = normalize_key(key_r, key_g, key_b)
    return np.array(
        [key_norm_r, key_norm_g, key_norm_b, softness, gamma, sat_gate, 0.0, 0.0],
        dtype=np.float32,
    )


def run_key_pass(device, input_rgba: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Run keying shader on input_rgba (H,W,4 uint8), return R channel as float (H,W)."""
    frame_height, frame_width = input_rgba.shape[:2]

    input_texture = create_texture(
        device,
        frame_width,
        frame_height,
        wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    output_texture = create_texture(
        device,
        frame_width,
        frame_height,
        wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
    )
    upload_rgba(device, input_texture, input_rgba)

    params_buffer = device.create_buffer_with_data(data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)

    shader = device.create_shader_module(code=load_wgsl("alpha_hint"))
    bind_group_layout = device.create_bind_group_layout(
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
        layout=device.create_pipeline_layout(bind_group_layouts=[bind_group_layout]),
        compute={"module": shader, "entry_point": "main"},
    )
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": input_texture.create_view()},
            {"binding": 1, "resource": output_texture.create_view()},
            {
                "binding": 2,
                "resource": {"buffer": params_buffer, "offset": 0, "size": 32},
            },
        ],
    )

    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch_workgroups(
        (frame_width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
        (frame_height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
        1,
    )
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    return readback_r_channel(device, output_texture, frame_width, frame_height)


# ---------------------------------------------------------------------------
# Blur helper
# ---------------------------------------------------------------------------


def run_blur_pass(device, input_rgba: np.ndarray, radius: int) -> np.ndarray:
    """Run horizontal + vertical blur on input (H,W,4 uint8), return R channel float."""
    frame_height, frame_width = input_rgba.shape[:2]

    input_texture = create_texture(
        device,
        frame_width,
        frame_height,
        wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    intermediate_texture = create_texture(
        device,
        frame_width,
        frame_height,
        wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    output_texture = create_texture(
        device,
        frame_width,
        frame_height,
        wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
    )
    upload_rgba(device, input_texture, input_rgba)

    blur_params = np.array([radius, 0, 0, 0], dtype=np.int32)
    blur_params_buffer = device.create_buffer_with_data(data=blur_params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    shader = device.create_shader_module(code=load_wgsl("blur"))
    bind_group_layout = device.create_bind_group_layout(
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
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    horizontal_blur_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "blur_h"},
    )
    vertical_blur_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "blur_v"},
    )

    horizontal_bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": input_texture.create_view()},
            {"binding": 1, "resource": intermediate_texture.create_view()},
            {
                "binding": 2,
                "resource": {
                    "buffer": blur_params_buffer,
                    "offset": 0,
                    "size": 16,
                },
            },
        ],
    )
    vertical_bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": intermediate_texture.create_view()},
            {"binding": 1, "resource": output_texture.create_view()},
            {
                "binding": 2,
                "resource": {
                    "buffer": blur_params_buffer,
                    "offset": 0,
                    "size": 16,
                },
            },
        ],
    )

    workgroup_count_x = (frame_width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    workgroup_count_y = (frame_height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    command_encoder = device.create_command_encoder()
    horizontal_pass = command_encoder.begin_compute_pass()
    horizontal_pass.set_pipeline(horizontal_blur_pipeline)
    horizontal_pass.set_bind_group(0, horizontal_bind_group, [], 0, 999999)
    horizontal_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1)
    horizontal_pass.end()
    vertical_pass = command_encoder.begin_compute_pass()
    vertical_pass.set_pipeline(vertical_blur_pipeline)
    vertical_pass.set_bind_group(0, vertical_bind_group, [], 0, 999999)
    vertical_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1)
    vertical_pass.end()
    device.queue.submit([command_encoder.finish()])

    return readback_r_channel(device, output_texture, frame_width, frame_height)


# ---------------------------------------------------------------------------
# Morphology helper
# ---------------------------------------------------------------------------


def run_morph_pass(device, input_rgba: np.ndarray, entry_point: str) -> np.ndarray:
    """Run one morphology pass (erode or dilate) on input, return R channel float."""
    frame_height, frame_width = input_rgba.shape[:2]

    input_texture = create_texture(
        device,
        frame_width,
        frame_height,
        wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    output_texture = create_texture(
        device,
        frame_width,
        frame_height,
        wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
    )
    upload_rgba(device, input_texture, input_rgba)

    shader = device.create_shader_module(code=load_wgsl("morphology"))
    bind_group_layout = device.create_bind_group_layout(
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
        layout=device.create_pipeline_layout(bind_group_layouts=[bind_group_layout]),
        compute={"module": shader, "entry_point": entry_point},
    )
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": input_texture.create_view()},
            {"binding": 1, "resource": output_texture.create_view()},
        ],
    )

    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch_workgroups(
        (frame_width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
        (frame_height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
        1,
    )
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    return readback_r_channel(device, output_texture, frame_width, frame_height)
