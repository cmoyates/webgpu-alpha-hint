"""Main GPU keying pipeline: video in, per-frame alpha masks out."""

from pathlib import Path

import av
import numpy as np
import wgpu
import wgpu.utils
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from .console import console, log
from .shader_utils import BYTES_PER_PIXEL, BYTES_PER_ROW_ALIGNMENT, MAX_BLUR_RADIUS, load_wgsl


def process_video(
    input_video: str,
    out_dir: str = "output",
    key_r: float = 0.0,
    key_g: float = 1.0,
    key_b: float = 0.0,
    softness: float = 0.3,
    gamma: float = 1.0,
    sat_gate: float = 0.1,
    max_frames: int | None = None,
    blur_radius: int = 0,
    erode_iters: int = 0,
    dilate_iters: int = 0,
):
    """Run the GPU keying pipeline on a video, saving per-frame alpha masks as PNGs.

    Stages run in order: chroma key -> blur -> erode -> dilate.
    Each stage is optional; blur/morph are skipped when their params are 0.
    """
    blur_radius = min(max(blur_radius, 0), MAX_BLUR_RADIUS)
    erode_iters = max(erode_iters, 0)
    dilate_iters = max(dilate_iters, 0)
    total_morph_iters = erode_iters + dilate_iters

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    container = av.open(input_video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    first_frame = next(container.decode(stream))
    frame_width, frame_height = first_frame.width, first_frame.height

    # PyAV doesn't support seeking back to frame 0 after decode, so re-open
    container.close()
    container = av.open(input_video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No WebGPU adapter found.")
    device = adapter.request_device_sync()

    keying_shader = device.create_shader_module(code=load_wgsl("alpha_hint"))

    input_texture = device.create_texture(
        size=(frame_width, frame_height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    input_texture_view = input_texture.create_view()

    output_texture = device.create_texture(
        size=(frame_width, frame_height, 1),
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST,
    )
    output_texture_view = output_texture.create_view()

    # Pre-normalize on CPU so the shader avoids per-pixel division
    key_channel_sum = key_r + key_g + key_b + 1e-4
    key_norm_r = key_r / key_channel_sum
    key_norm_g = key_g / key_channel_sum
    key_norm_b = key_b / key_channel_sum

    # Padded to 32 bytes (8 x f32) to satisfy uniform buffer alignment
    keying_params_data = np.array(
        [key_norm_r, key_norm_g, key_norm_b, softness, gamma, sat_gate, 0.0, 0.0],
        dtype=np.float32,
    )
    keying_params_buffer = device.create_buffer_with_data(
        data=keying_params_data.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    keying_bind_group_layout = device.create_bind_group_layout(
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

    keying_pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[keying_bind_group_layout])

    # Texture routing: each stage writes to the next stage's input.
    # The last active stage writes to output_texture for readback.
    if blur_radius > 0:
        keying_output_texture = device.create_texture(
            size=(frame_width, frame_height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        keying_output_view = keying_output_texture.create_view()

        blur_intermediate_texture = device.create_texture(
            size=(frame_width, frame_height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        blur_intermediate_view = blur_intermediate_texture.create_view()

        keying_dest_view = keying_output_view
    else:
        keying_dest_view = output_texture_view

    # Morphology needs two textures to alternate reads/writes between iterations
    if total_morph_iters > 0:
        morphology_texture_usage = (
            wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
        )
        morph_ping_texture = device.create_texture(
            size=(frame_width, frame_height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=morphology_texture_usage,
        )
        morph_ping_view = morph_ping_texture.create_view()

        morph_pong_texture = device.create_texture(
            size=(frame_width, frame_height, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=morphology_texture_usage,
        )
        morph_pong_view = morph_pong_texture.create_view()

        if blur_radius > 0:
            vertical_blur_dest_view = morph_ping_view
        else:
            keying_dest_view = morph_ping_view
    else:
        vertical_blur_dest_view = output_texture_view if blur_radius > 0 else None

    keying_pipeline = device.create_compute_pipeline(
        layout=keying_pipeline_layout,
        compute={"module": keying_shader, "entry_point": "main"},
    )

    keying_bind_group = device.create_bind_group(
        layout=keying_bind_group_layout,
        entries=[
            {"binding": 0, "resource": input_texture_view},
            {"binding": 1, "resource": keying_dest_view},
            {
                "binding": 2,
                "resource": {
                    "buffer": keying_params_buffer,
                    "offset": 0,
                    "size": 32,
                },
            },
        ],
    )

    if blur_radius > 0:
        blur_shader = device.create_shader_module(code=load_wgsl("blur"))

        blur_params_data = np.array([blur_radius, 0, 0, 0], dtype=np.int32)
        blur_params_buffer = device.create_buffer_with_data(
            data=blur_params_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM,
        )

        blur_bind_group_layout = device.create_bind_group_layout(
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

        blur_pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[blur_bind_group_layout])

        horizontal_blur_pipeline = device.create_compute_pipeline(
            layout=blur_pipeline_layout,
            compute={"module": blur_shader, "entry_point": "blur_h"},
        )
        vertical_blur_pipeline = device.create_compute_pipeline(
            layout=blur_pipeline_layout,
            compute={"module": blur_shader, "entry_point": "blur_v"},
        )

        horizontal_blur_bind_group = device.create_bind_group(
            layout=blur_bind_group_layout,
            entries=[
                {"binding": 0, "resource": keying_output_view},
                {"binding": 1, "resource": blur_intermediate_view},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": blur_params_buffer,
                        "offset": 0,
                        "size": 16,
                    },
                },
            ],
        )

        vertical_blur_bind_group = device.create_bind_group(
            layout=blur_bind_group_layout,
            entries=[
                {"binding": 0, "resource": blur_intermediate_view},
                {"binding": 1, "resource": vertical_blur_dest_view},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": blur_params_buffer,
                        "offset": 0,
                        "size": 16,
                    },
                },
            ],
        )

    if total_morph_iters > 0:
        morphology_shader = device.create_shader_module(code=load_wgsl("morphology"))

        morphology_bind_group_layout = device.create_bind_group_layout(
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
        morphology_pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[morphology_bind_group_layout])

        erode_pipeline = device.create_compute_pipeline(
            layout=morphology_pipeline_layout,
            compute={"module": morphology_shader, "entry_point": "erode"},
        )
        dilate_pipeline = device.create_compute_pipeline(
            layout=morphology_pipeline_layout,
            compute={"module": morphology_shader, "entry_point": "dilate"},
        )

        morph_bind_group_ping_to_pong = device.create_bind_group(
            layout=morphology_bind_group_layout,
            entries=[
                {"binding": 0, "resource": morph_ping_view},
                {"binding": 1, "resource": morph_pong_view},
            ],
        )
        morph_bind_group_pong_to_ping = device.create_bind_group(
            layout=morphology_bind_group_layout,
            entries=[
                {"binding": 0, "resource": morph_pong_view},
                {"binding": 1, "resource": morph_ping_view},
            ],
        )

        # Pre-compute the full erode/dilate schedule so the per-frame loop
        # just iterates a flat list without branching
        morphology_schedule = []
        reading_ping = True
        for _ in range(erode_iters):
            bind_group = morph_bind_group_ping_to_pong if reading_ping else morph_bind_group_pong_to_ping
            morphology_schedule.append((erode_pipeline, bind_group))
            reading_ping = not reading_ping
        for _ in range(dilate_iters):
            bind_group = morph_bind_group_ping_to_pong if reading_ping else morph_bind_group_pong_to_ping
            morphology_schedule.append((dilate_pipeline, bind_group))
            reading_ping = not reading_ping

        # reading_ping tracks the next read source; the last write went to the other
        morphology_result_texture = morph_ping_texture if reading_ping else morph_pong_texture

    unpadded_bytes_per_row = frame_width * BYTES_PER_PIXEL

    # WebGPU spec requires bytes_per_row aligned to 256
    padded_bytes_per_row = (
        (unpadded_bytes_per_row + BYTES_PER_ROW_ALIGNMENT - 1) // BYTES_PER_ROW_ALIGNMENT * BYTES_PER_ROW_ALIGNMENT
    )
    readback_size = padded_bytes_per_row * frame_height

    readback_buffer = device.create_buffer(
        size=readback_size,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )

    # Total frame count for progress bar (None if stream lacks duration metadata)
    estimated_frames = max_frames or stream.frames or None

    frame_index = 0
    with Progress(
        TextColumn("[bold blue]Processing"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("frames"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("frames", total=estimated_frames)

        for frame in container.decode(stream):
            if max_frames is not None and frame_index >= max_frames:
                break

            rgba = frame.to_ndarray(format="rgba")

            device.queue.write_texture(
                {"texture": input_texture, "mip_level": 0, "origin": (0, 0, 0)},
                rgba.tobytes(),
                {"bytes_per_row": unpadded_bytes_per_row, "rows_per_image": frame_height},
                (frame_width, frame_height, 1),
            )

            command_encoder = device.create_command_encoder()
            workgroup_count_x = (frame_width + 15) // 16
            workgroup_count_y = (frame_height + 15) // 16

            keying_pass = command_encoder.begin_compute_pass()
            keying_pass.set_pipeline(keying_pipeline)
            keying_pass.set_bind_group(0, keying_bind_group, [], 0, 999999)
            keying_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1)
            keying_pass.end()

            if blur_radius > 0:
                horizontal_pass = command_encoder.begin_compute_pass()
                horizontal_pass.set_pipeline(horizontal_blur_pipeline)
                horizontal_pass.set_bind_group(0, horizontal_blur_bind_group, [], 0, 999999)
                horizontal_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1)
                horizontal_pass.end()

                vertical_pass = command_encoder.begin_compute_pass()
                vertical_pass.set_pipeline(vertical_blur_pipeline)
                vertical_pass.set_bind_group(0, vertical_blur_bind_group, [], 0, 999999)
                vertical_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1)
                vertical_pass.end()

            if total_morph_iters > 0:
                for pipeline, bind_group in morphology_schedule:
                    morphology_pass = command_encoder.begin_compute_pass()
                    morphology_pass.set_pipeline(pipeline)
                    morphology_pass.set_bind_group(0, bind_group, [], 0, 999999)
                    morphology_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1)
                    morphology_pass.end()

                # Morph result lands in whichever ping/pong texture was last written;
                # copy it to output_texture so readback always reads the same buffer
                command_encoder.copy_texture_to_texture(
                    {
                        "texture": morphology_result_texture,
                        "mip_level": 0,
                        "origin": (0, 0, 0),
                    },
                    {"texture": output_texture, "mip_level": 0, "origin": (0, 0, 0)},
                    (frame_width, frame_height, 1),
                )

            command_encoder.copy_texture_to_buffer(
                {"texture": output_texture, "mip_level": 0, "origin": (0, 0, 0)},
                {
                    "buffer": readback_buffer,
                    "offset": 0,
                    "bytes_per_row": padded_bytes_per_row,
                    "rows_per_image": frame_height,
                },
                (frame_width, frame_height, 1),
            )

            device.queue.submit([command_encoder.finish()])

            readback_buffer.map_sync(mode=wgpu.MapMode.READ)
            data = readback_buffer.read_mapped()
            raw = np.frombuffer(data, dtype=np.uint8).reshape((frame_height, padded_bytes_per_row))
            rgba_out = raw[:, :unpadded_bytes_per_row].reshape((frame_height, frame_width, 4)).copy()
            readback_buffer.unmap()

            matte = rgba_out[:, :, 0]

            image = Image.fromarray(matte, mode="L")
            image.save(output_path / f"mask_{frame_index:06d}.png")

            frame_index += 1
            progress.advance(task)

    container.close()
    log.info("Saved %d PNG masks to %s", frame_index, output_path)
