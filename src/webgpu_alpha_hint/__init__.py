"""webgpu-alpha-hint: GPU-accelerated green-screen keying via WebGPU compute shaders."""

__version__ = "0.1.0"

from .console import console, log
from .pipeline import process_video
from .shader_utils import load_wgsl

__all__ = ["console", "load_wgsl", "log", "process_video"]
