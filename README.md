# webgpu-alpha-hint

GPU-accelerated green-screen keying via WebGPU compute shaders.

## Usage

```bash
python main.py input.mp4 --out masks/ --softness 0.3 --blur_radius 3 --erode_iters 1 --dilate_iters 1 --max_frames 10
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `alpha_hint_frames` | Output directory for mask PNGs |
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

## Tuning guide

| Scenario | Suggested flags |
|----------|----------------|
| Standard green screen | `--softness 0.3` (default) |
| Hair flyaways / motion blur | `--softness 0.4 --gamma 0.8 --blur_radius 2` |
| Uneven / dark green screen | `--softness 0.5 --sat_gate 0.05` |
| Blue screen | `--key_r 0 --key_g 0 --key_b 1` |
| Tight / clean matte | `--softness 0.15 --erode_iters 1` |
