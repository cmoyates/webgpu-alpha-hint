"""CLI entry point for webgpu-alpha-hint."""

import argparse

from rich_argparse import RichHelpFormatter

from .pipeline import process_video


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated green-screen keying via WebGPU compute shaders.",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("input_video", help="Input video file path")
    parser.add_argument("--out", default="output", help="Output directory for masks")
    parser.add_argument("--key_r", type=float, default=0.0, help="Key color red (0..1)")
    parser.add_argument("--key_g", type=float, default=1.0, help="Key color green (0..1)")
    parser.add_argument("--key_b", type=float, default=0.0, help="Key color blue (0..1)")
    parser.add_argument(
        "--softness",
        type=float,
        default=0.3,
        help="Chroma-distance transition width (0=hard, ~0.3=typical green screen)",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma bias on matte edges")
    parser.add_argument(
        "--sat_gate",
        type=float,
        default=0.1,
        help="Saturation below which keying is suppressed (protects greys/whites)",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--blur_radius",
        type=int,
        default=0,
        help="Separable box blur radius (0=off, max 8)",
    )
    parser.add_argument(
        "--erode_iters",
        type=int,
        default=0,
        help="Erode iterations (3x3 min, removes speckle)",
    )
    parser.add_argument(
        "--dilate_iters",
        type=int,
        default=0,
        help="Dilate iterations (3x3 max, fills small holes)",
    )
    args = parser.parse_args()

    process_video(
        args.input_video,
        out_dir=args.out,
        key_r=args.key_r,
        key_g=args.key_g,
        key_b=args.key_b,
        softness=args.softness,
        gamma=args.gamma,
        sat_gate=args.sat_gate,
        max_frames=args.max_frames,
        blur_radius=args.blur_radius,
        erode_iters=args.erode_iters,
        dilate_iters=args.dilate_iters,
    )
