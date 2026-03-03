from pathlib import Path

import av
import numpy as np
import wgpu
import wgpu.utils
from PIL import Image


def load_wgsl(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


MAX_BLUR_RADIUS = 8


def main(
    input_video: str,
    out_dir: str = "alpha_hint_frames",
    t_low: float = -0.05,
    t_high: float = 0.10,
    gamma: float = 1.0,
    max_frames: int | None = None,
    blur_radius: int = 0,
    erode_iters: int = 0,
    dilate_iters: int = 0,
):
    blur_radius = min(max(blur_radius, 0), MAX_BLUR_RADIUS)
    erode_iters = max(erode_iters, 0)
    dilate_iters = max(dilate_iters, 0)
    total_morph_iters = erode_iters + dilate_iters

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

    key_shader = device.create_shader_module(code=load_wgsl("alpha_hint.wgsl"))

    # Input texture: rgba8unorm (we upload uint8 RGBA)
    input_tex = device.create_texture(
        size=(width, height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    input_view = input_tex.create_view()

    # Output texture: final result, always needed for readback
    output_tex = device.create_texture(
        size=(width, height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.STORAGE_BINDING
        | wgpu.TextureUsage.COPY_SRC
        | wgpu.TextureUsage.COPY_DST,
    )
    output_view = output_tex.create_view()

    # Keying params uniform
    params_data = np.array([t_low, t_high, gamma, 0.0], dtype=np.float32)
    params_buf = device.create_buffer_with_data(
        data=params_data.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # --- Key pass bind group layout (shared pattern: tex_2d + storage_tex + uniform) ---
    key_bgl = device.create_bind_group_layout(
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

    key_pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[key_bgl])

    # Determine where keying and blur write their result.
    # If morphology follows, the pre-morph stage writes to morph_ping.
    # Otherwise, if blur follows keying, key writes to intermediate.
    # Otherwise, key writes directly to output.

    if blur_radius > 0:
        # Intermediate texture: keying result, read by blur_h
        intermediate_tex = device.create_texture(
            size=(width, height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        intermediate_view = intermediate_tex.create_view()

        # Temp texture for between blur_h and blur_v
        blur_temp_tex = device.create_texture(
            size=(width, height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        blur_temp_view = blur_temp_tex.create_view()

        key_dest_view = intermediate_view
    else:
        key_dest_view = output_view

    # --- Morphology ping-pong textures ---
    if total_morph_iters > 0:
        morph_tex_usage = (
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC
        )
        morph_ping_tex = device.create_texture(
            size=(width, height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=morph_tex_usage,
        )
        morph_ping_view = morph_ping_tex.create_view()

        morph_pong_tex = device.create_texture(
            size=(width, height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=morph_tex_usage,
        )
        morph_pong_view = morph_pong_tex.create_view()

        # Redirect the last pre-morph stage to write to morph_ping
        if blur_radius > 0:
            blur_v_dest_view = morph_ping_view
        else:
            key_dest_view = morph_ping_view
    else:
        blur_v_dest_view = output_view if blur_radius > 0 else None

    key_pipeline = device.create_compute_pipeline(
        layout=key_pipeline_layout,
        compute={"module": key_shader, "entry_point": "main"},
    )

    key_bind_group = device.create_bind_group(
        layout=key_bgl,
        entries=[
            {"binding": 0, "resource": input_view},
            {"binding": 1, "resource": key_dest_view},
            {"binding": 2, "resource": {"buffer": params_buf, "offset": 0, "size": 16}},
        ],
    )

    # --- Blur passes (only if blur_radius > 0) ---
    if blur_radius > 0:
        blur_shader = device.create_shader_module(code=load_wgsl("blur.wgsl"))

        blur_params_data = np.array(
            [blur_radius, 0, 0, 0], dtype=np.int32
        )
        blur_params_buf = device.create_buffer_with_data(
            data=blur_params_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
        )

        blur_bgl = device.create_bind_group_layout(
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

        blur_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[blur_bgl]
        )

        blur_h_pipeline = device.create_compute_pipeline(
            layout=blur_pipeline_layout,
            compute={"module": blur_shader, "entry_point": "blur_h"},
        )
        blur_v_pipeline = device.create_compute_pipeline(
            layout=blur_pipeline_layout,
            compute={"module": blur_shader, "entry_point": "blur_v"},
        )

        blur_h_bind_group = device.create_bind_group(
            layout=blur_bgl,
            entries=[
                {"binding": 0, "resource": intermediate_view},
                {"binding": 1, "resource": blur_temp_view},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": blur_params_buf,
                        "offset": 0,
                        "size": 16,
                    },
                },
            ],
        )

        # blur_v writes to morph_ping (if morph follows) or output
        blur_v_bind_group = device.create_bind_group(
            layout=blur_bgl,
            entries=[
                {"binding": 0, "resource": blur_temp_view},
                {"binding": 1, "resource": blur_v_dest_view},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": blur_params_buf,
                        "offset": 0,
                        "size": 16,
                    },
                },
            ],
        )

    # --- Morphology pipelines and bind groups ---
    if total_morph_iters > 0:
        morph_shader = device.create_shader_module(code=load_wgsl("morphology.wgsl"))

        # Morphology uses same 2-binding layout: tex_2d read + storage_tex write
        morph_bgl = device.create_bind_group_layout(
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
            ]
        )
        morph_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[morph_bgl]
        )

        erode_pipeline = device.create_compute_pipeline(
            layout=morph_pipeline_layout,
            compute={"module": morph_shader, "entry_point": "erode"},
        )
        dilate_pipeline = device.create_compute_pipeline(
            layout=morph_pipeline_layout,
            compute={"module": morph_shader, "entry_point": "dilate"},
        )

        # Pre-build bind groups for ping→pong and pong→ping
        morph_bg_ping_to_pong = device.create_bind_group(
            layout=morph_bgl,
            entries=[
                {"binding": 0, "resource": morph_ping_view},
                {"binding": 1, "resource": morph_pong_view},
            ],
        )
        morph_bg_pong_to_ping = device.create_bind_group(
            layout=morph_bgl,
            entries=[
                {"binding": 0, "resource": morph_pong_view},
                {"binding": 1, "resource": morph_ping_view},
            ],
        )

        # Build iteration schedule: list of (pipeline, bind_group) tuples
        # Data starts in morph_ping. Iterations alternate ping→pong, pong→ping.
        morph_schedule = []
        reading_ping = True
        for _ in range(erode_iters):
            bg = morph_bg_ping_to_pong if reading_ping else morph_bg_pong_to_ping
            morph_schedule.append((erode_pipeline, bg))
            reading_ping = not reading_ping
        for _ in range(dilate_iters):
            bg = morph_bg_ping_to_pong if reading_ping else morph_bg_pong_to_ping
            morph_schedule.append((dilate_pipeline, bg))
            reading_ping = not reading_ping

        # reading_ping tracks the *next* read source; the last write went to the other.
        morph_result_tex = morph_ping_tex if reading_ping else morph_pong_tex

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

        command_encoder = device.create_command_encoder()
        gx = (width + 15) // 16
        gy = (height + 15) // 16

        # Pass 1: keying
        key_pass = command_encoder.begin_compute_pass()
        key_pass.set_pipeline(key_pipeline)
        key_pass.set_bind_group(0, key_bind_group, [], 0, 999999)
        key_pass.dispatch_workgroups(gx, gy, 1)
        key_pass.end()

        # Pass 2+3: separable blur (if enabled)
        if blur_radius > 0:
            h_pass = command_encoder.begin_compute_pass()
            h_pass.set_pipeline(blur_h_pipeline)
            h_pass.set_bind_group(0, blur_h_bind_group, [], 0, 999999)
            h_pass.dispatch_workgroups(gx, gy, 1)
            h_pass.end()

            v_pass = command_encoder.begin_compute_pass()
            v_pass.set_pipeline(blur_v_pipeline)
            v_pass.set_bind_group(0, blur_v_bind_group, [], 0, 999999)
            v_pass.dispatch_workgroups(gx, gy, 1)
            v_pass.end()

        # Pass 4+: morphology iterations (if enabled)
        if total_morph_iters > 0:
            for pipeline, bind_group in morph_schedule:
                m_pass = command_encoder.begin_compute_pass()
                m_pass.set_pipeline(pipeline)
                m_pass.set_bind_group(0, bind_group, [], 0, 999999)
                m_pass.dispatch_workgroups(gx, gy, 1)
                m_pass.end()

            # Copy morph result → output_tex for readback
            command_encoder.copy_texture_to_texture(
                {"texture": morph_result_tex, "mip_level": 0, "origin": (0, 0, 0)},
                {"texture": output_tex, "mip_level": 0, "origin": (0, 0, 0)},
                (width, height, 1),
            )

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

        readback_buf.map_sync(mode=wgpu.MapMode.READ)
        data = readback_buf.read_mapped()
        raw = np.frombuffer(data, dtype=np.uint8).reshape((height, padded_bpr))
        rgba_out = raw[:, :unpadded_bpr].reshape((height, width, 4)).copy()
        readback_buf.unmap()

        matte = rgba_out[:, :, 0]

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
    p.add_argument(
        "--blur_radius", type=int, default=0,
        help="Separable box blur radius (0=off, max 8)",
    )
    p.add_argument(
        "--erode_iters", type=int, default=0,
        help="Erode iterations (3x3 min, removes speckle)",
    )
    p.add_argument(
        "--dilate_iters", type=int, default=0,
        help="Dilate iterations (3x3 max, fills small holes)",
    )
    args = p.parse_args()

    main(
        args.input_video,
        out_dir=args.out,
        t_low=args.t_low,
        t_high=args.t_high,
        gamma=args.gamma,
        max_frames=args.max_frames,
        blur_radius=args.blur_radius,
        erode_iters=args.erode_iters,
        dilate_iters=args.dilate_iters,
    )
