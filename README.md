# WebGPU Alpha Hint

GPU-accelerated green-screen keying via WebGPU compute shaders.

## Install

```bash
pip install -e .
```

For development (includes pytest, ruff, ty):

```bash
pip install -e . --group dev
```

## Usage

### CLI

```bash
# Basic usage — outputs masks to output/ by default
webgpu-alpha-hint input.mp4

# Custom output directory
webgpu-alpha-hint input.mp4 --out masks/

# Green screen with blur + morphology cleanup
webgpu-alpha-hint input.mp4 --softness 0.3 --blur_radius 3 --erode_iters 1 --dilate_iters 1

# Blue screen keying, first 100 frames only
webgpu-alpha-hint input.mp4 --key_r 0 --key_g 0 --key_b 1 --max_frames 100

# View all options
webgpu-alpha-hint --help
```

### Python

```python
from webgpu_alpha_hint import process_video

# Minimal — uses default green key and saves to output/
process_video("input.mp4")

# Full control
process_video(
    "input.mp4",
    out_dir="masks/",
    softness=0.3,
    blur_radius=3,
    erode_iters=1,
    dilate_iters=1,
)
```

Individual utilities are also importable:

```python
from webgpu_alpha_hint.shader_utils import load_wgsl
from webgpu_alpha_hint.gpu import create_texture, upload_rgba, readback_r_channel
from webgpu_alpha_hint.console import console, log
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `output` | Output directory for mask PNGs |
| `--key_r` | `0.0` | Key color red channel (0..1) |
| `--key_g` | `1.0` | Key color green channel (0..1) |
| `--key_b` | `0.0` | Key color blue channel (0..1) |
| `--softness` | `0.3` | Chroma-distance transition width (0=hard, higher=softer edges) |
| `--gamma` | `1.0` | Gamma bias on matte edges (<1 expands edges, >1 contracts) |
| `--sat_gate` | `0.1` | Saturation below which keying is suppressed (protects greys/whites) |
| `--max_frames` | all | Stop after N frames |
| `--blur_radius` | `0` | Separable box blur radius (0=off, max 8) |
| `--erode_iters` | `0` | Erode iterations (3x3 min, removes speckle) |
| `--dilate_iters` | `0` | Dilate iterations (3x3 max, fills small holes) |

## Key metric

Uses **normalized-chroma distance**: each pixel's RGB is projected into chromaticity space
(`rgb / (r+g+b)`) and compared to the key color. Distance from key determines the matte value
via a smoothstep transition controlled by `--softness`.

A **saturation gate** protects low-saturation pixels (grey, white, black) from being keyed,
even if their chromaticity happens to land near the key color.

## Architecture

`src/webgpu_alpha_hint/` package with three WGSL compute shaders (`shaders/alpha_hint.wgsl`, `shaders/blur.wgsl`, `shaders/morphology.wgsl`). All textures are `rgba8unorm`.

### GPU data flow

Which texture each stage writes to changes depending on which passes are enabled:

- **Key only**: key -> `output_tex`
- **Key + blur**: key -> `intermediate_tex` -> blur_h -> `blur_temp_tex` -> blur_v -> `output_tex`
- **Key + morph**: key -> `morph_ping` -> ping-pong erode/dilate -> copy final -> `output_tex`
- **Key + blur + morph**: blur_v writes to `morph_ping` instead of `output_tex`, then morph ping-pongs, then copies to `output_tex`

Morphology ping-pongs between `morph_ping` and `morph_pong`. `reading_ping` bool tracks which texture holds current data.

Key color is pre-normalized on CPU before upload (avoids per-pixel division in shader). Matte stored in R channel only; readback extracts `rgba_out[:, :, 0]`.

## Tuning guide

| Scenario | Suggested flags |
|----------|----------------|
| Standard green screen | `--softness 0.3` (default) |
| Hair flyaways / motion blur | `--softness 0.4 --gamma 0.8 --blur_radius 2` |
| Uneven / dark green screen | `--softness 0.5 --sat_gate 0.05` |
| Blue screen | `--key_r 0 --key_g 0 --key_b 1` |
| Tight / clean matte | `--softness 0.15 --erode_iters 1` |
