"""webgpu-alpha-hint: GPU-accelerated green-screen keying via WebGPU compute shaders."""

__version__ = "0.1.0"

from .gpu import create_texture, readback_r_channel, upload_rgba
from .pipeline import process_video
from .shader_utils import load_wgsl

__all__ = [
    "create_texture",
    "load_wgsl",
    "process_video",
    "readback_r_channel",
    "upload_rgba",
]
