from pathlib import Path

import av
import numpy as np
import wgpu
import wgpu.utils
from PIL import Image


def load_wgsl(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(
    input_video: str,
    out_dir: str = "alpha_hint_frames",
    t_low: float = -0.05,
    t_high: float = 0.10,
    gamma: float = 1.0,
    max_frames: int | None = None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Decode first frame to get size ---
    container = av.open(input_video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    first_frame = next(container.decode(stream))
    width, height = first_frame.width, first_frame.height

    # Re-open so we start from frame 0
    container.close()
    container = av.open(input_video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # --- WebGPU setup ---
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No WebGPU adapter found.")
    device = adapter.request_device_sync()

    shader_code = load_wgsl("alpha_hint.wgsl")
    shader = device.create_shader_module(code=shader_code)

    # Input texture: rgba8unorm (we upload uint8 RGBA)
    input_tex = device.create_texture(
        size=(width, height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    input_view = input_tex.create_view()

    # Output texture: rgba8unorm (shader writes grayscale into RGB)
    output_tex = device.create_texture(
        size=(width, height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
    )
    output_view = output_tex.create_view()

    # Uniform params buffer (std140-ish layout handled by wgpu)
    # We'll pack 4 floats to match Params struct.
    params_data = np.array([t_low, t_high, gamma, 0.0], dtype=np.float32)
    params_buf = device.create_buffer_with_data(
        data=params_data.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {"sample_type": wgpu.TextureSampleType.float},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.write_only,
                    "format": wgpu.TextureFormat.rgba8unorm,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "main"},
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": input_view},
            {"binding": 1, "resource": output_view},
            {"binding": 2, "resource": {"buffer": params_buf, "offset": 0, "size": 16}},
        ],
    )

    # Readback buffer (RGBA8)
    bytes_per_pixel = 4
    unpadded_bpr = width * bytes_per_pixel

    # WebGPU requires bytes_per_row to be multiple of 256.
    padded_bpr = ((unpadded_bpr + 255) // 256) * 256
    readback_size = padded_bpr * height

    readback_buf = device.create_buffer(
        size=readback_size,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )

    # --- Process frames ---
    frame_index = 0
    for frame in container.decode(stream):
        if max_frames is not None and frame_index >= max_frames:
            break

        # Convert to RGBA on CPU
        rgba = frame.to_ndarray(format="rgba")  # uint8, shape (H, W, 4)

        # Upload frame to GPU texture
        device.queue.write_texture(
            {"texture": input_tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba.tobytes(),
            {"bytes_per_row": unpadded_bpr, "rows_per_image": height},
            (width, height, 1),
        )

        # Encode compute pass + copy output texture -> readback buffer
        command_encoder = device.create_command_encoder()

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
        gx = (width + 15) // 16
        gy = (height + 15) // 16
        compute_pass.dispatch_workgroups(gx, gy, 1)
        compute_pass.end()

        command_encoder.copy_texture_to_buffer(
            {"texture": output_tex, "mip_level": 0, "origin": (0, 0, 0)},
            {
                "buffer": readback_buf,
                "offset": 0,
                "bytes_per_row": padded_bpr,
                "rows_per_image": height,
            },
            (width, height, 1),
        )

        device.queue.submit([command_encoder.finish()])

        # Map buffer and extract the unpadded RGBA rows
        readback_buf.map_sync(mode=wgpu.MapMode.READ)
        data = readback_buf.read_mapped()
        raw = np.frombuffer(data, dtype=np.uint8).reshape((height, padded_bpr))
        rgba_out = raw[:, :unpadded_bpr].reshape((height, width, 4)).copy()
        readback_buf.unmap()

        # Take grayscale from R channel (same as G,B)
        matte = rgba_out[:, :, 0]

        # Save PNG
        img = Image.fromarray(matte, mode="L")
        img.save(out_path / f"mask_{frame_index:06d}.png")

        frame_index += 1
        if frame_index % 30 == 0:
            print(f"Saved {frame_index} frames...")

    container.close()
    print(f"Done. Saved {frame_index} PNG masks to {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("input_video")
    p.add_argument("--out", default="alpha_hint_frames")
    p.add_argument("--t_low", type=float, default=-0.05)
    p.add_argument("--t_high", type=float, default=0.10)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=None)
    args = p.parse_args()

    main(
        args.input_video,
        out_dir=args.out,
        t_low=args.t_low,
        t_high=args.t_high,
        gamma=args.gamma,
        max_frames=args.max_frames,
    )
