# webgpu-alpha-hint

GPU-accelerated green-screen keying via WebGPU compute shaders.

## Usage

```bash
python main.py input.mp4 --out masks/ --blur_radius 3 --erode_iters 1 --dilate_iters 1 --max_frames 10
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `alpha_hint_frames` | Output directory for mask PNGs |
| `--t_low` | `-0.05` | Lower threshold for excess-green keying |
| `--t_high` | `0.10` | Upper threshold for excess-green keying |
| `--gamma` | `1.0` | Gamma bias on matte edges |
| `--max_frames` | all | Stop after N frames |
| `--blur_radius` | `0` | Separable box blur radius (0=off, max 8) |
| `--erode_iters` | `0` | Erode iterations (3x3 min, removes speckle) |
| `--dilate_iters` | `0` | Dilate iterations (3x3 max, fills small holes) |
