"""GPU texture helpers: create, upload, readback."""

import numpy as np
import wgpu

from .shader_utils import BYTES_PER_PIXEL

TEX_FORMAT = wgpu.TextureFormat.rgba8unorm


def create_texture(device, width: int, height: int, usage):
    """Create a 2D rgba8unorm texture with the given usage flags."""
    return device.create_texture(
        size=(width, height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=TEX_FORMAT,
        usage=usage,
    )


def upload_rgba(device, texture, data: np.ndarray):
    """Upload uint8 RGBA array (H, W, 4) to texture."""
    frame_height, frame_width, _ = data.shape
    bytes_per_row = frame_width * BYTES_PER_PIXEL
    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        data.tobytes(),
        {"bytes_per_row": bytes_per_row, "rows_per_image": frame_height},
        (frame_width, frame_height, 1),
    )


def readback_r_channel(device, texture, width: int, height: int) -> np.ndarray:
    """Read back R channel as float array (H, W) in [0, 1]."""
    bytes_per_row = width * BYTES_PER_PIXEL
    # WebGPU spec requires bytes_per_row aligned to 256
    padded_bytes_per_row = ((bytes_per_row + 255) // 256) * 256
    readback_buffer = device.create_buffer(
        size=padded_bytes_per_row * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {
            "buffer": readback_buffer,
            "offset": 0,
            "bytes_per_row": padded_bytes_per_row,
            "rows_per_image": height,
        },
        (width, height, 1),
    )
    device.queue.submit([command_encoder.finish()])
    readback_buffer.map_sync(mode=wgpu.MapMode.READ)
    raw = np.frombuffer(readback_buffer.read_mapped(), dtype=np.uint8).reshape((height, padded_bytes_per_row))
    rgba = raw[:, :bytes_per_row].reshape((height, width, 4)).copy()
    readback_buffer.unmap()
    return rgba[:, :, 0].astype(np.float32) / 255.0
