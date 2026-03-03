# WebGPU Alpha Hint

GPU-accelerated green-screen keying via WebGPU compute shaders.

## Install

```bash
uv add webgpu-alpha-hint
```

With CLI support (adds `rich-argparse`):

```bash
uv add "webgpu-alpha-hint[cli]"
```

## Usage

### Python

```python
from webgpu_alpha_hint import process_video

# Minimal — default green key, saves masks to output/
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

### CLI

```bash
webgpu-alpha-hint input.mp4
webgpu-alpha-hint input.mp4 --out masks/ --softness 0.3 --blur_radius 3
webgpu-alpha-hint input.mp4 --key_r 0 --key_g 0 --key_b 1  # blue screen
webgpu-alpha-hint --help
```

## Public API

```python
# Primary
from webgpu_alpha_hint import process_video

# GPU utilities (advanced)
from webgpu_alpha_hint import create_texture, upload_rgba, readback_r_channel

# Shader loading
from webgpu_alpha_hint import load_wgsl
```

| Symbol | Description |
|--------|-------------|
| `process_video()` | Run full keying pipeline on a video file, save per-frame masks |
| `load_wgsl(name)` | Load a bundled WGSL shader by name (`alpha_hint`, `blur`, `morphology`) |
| `create_texture()` | Create a 2D rgba8unorm GPU texture |
| `upload_rgba()` | Upload uint8 RGBA array to a GPU texture |
| `readback_r_channel()` | Read back R channel from a GPU texture as float32 array |

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `output` | Output directory for mask PNGs |
| `--key_r` | `0.0` | Key color red channel (0..1) |
| `--key_g` | `1.0` | Key color green channel (0..1) |
| `--key_b` | `0.0` | Key color blue channel (0..1) |
| `--softness` | `0.3` | Chroma-distance transition width (0=hard, higher=softer edges) |
| `--gamma` | `1.0` | Gamma bias on matte edges (<1 expands, >1 contracts) |
| `--sat_gate` | `0.1` | Saturation below which keying is suppressed |
| `--max_frames` | all | Stop after N frames |
| `--blur_radius` | `0` | Separable box blur radius (0=off, max 8) |
| `--erode_iters` | `0` | Erode iterations (3x3 min) |
| `--dilate_iters` | `0` | Dilate iterations (3x3 max) |

## Optional extras

| Extra | Dependencies | Purpose |
|-------|-------------|---------|
| `cli` | `rich-argparse` | Pretty CLI help formatting |

## Consuming from another project

### Local path

```bash
uv add /path/to/webgpu-alpha-hint
```

### Editable local install

```bash
uv add --editable /path/to/webgpu-alpha-hint
```

### Git URL

```bash
uv add "webgpu-alpha-hint @ git+https://github.com/OWNER/webgpu-alpha-hint.git"
```

## Development

```bash
git clone <repo-url> && cd webgpu-alpha-hint
uv sync --group dev
uv run pytest -v          # requires GPU adapter
uv run ruff check . && uv run ruff format .
uv run ty
uv build                  # produces wheel in dist/
```

## Architecture

Three WGSL compute shaders in `src/webgpu_alpha_hint/shaders/`:

- `alpha_hint.wgsl` — normalized-chroma distance keyer
- `blur.wgsl` — separable box blur (horizontal + vertical)
- `morphology.wgsl` — 3x3 erode/dilate

Pipeline: **key → blur → erode → dilate**. Each stage optional.

Key color is pre-normalized on CPU. Matte stored in R channel only.
