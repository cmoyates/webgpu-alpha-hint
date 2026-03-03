"""Shader loading and shared constants."""

from importlib.resources import files

WORKGROUP_SIZE = 16
BYTES_PER_PIXEL = 4
BYTES_PER_ROW_ALIGNMENT = 256
MAX_BLUR_RADIUS = 8


def load_wgsl(name: str) -> str:
    """Load a WGSL shader from the package's shaders/ subpackage by name (without extension)."""
    resource = files("webgpu_alpha_hint") / "shaders" / f"{name}.wgsl"
    return resource.read_text(encoding="utf-8")
