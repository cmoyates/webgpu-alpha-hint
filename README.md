# webgpu-alpha-hint

GPU-accelerated green-screen keying via WebGPU compute shaders.

## Usage

```bash
python main.py input.mp4 --out masks/ --blur_radius 3 --max_frames 10
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
