// morphology.wgsl — 3x3 erode (min) and dilate (max) on R channel
@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var outputTex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn erode(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(inputTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let max_x = i32(size.x) - 1;
  let max_y = i32(size.y) - 1;
  let cx = i32(gid.x);
  let cy = i32(gid.y);

  var min_val = 1.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let sx = clamp(cx + dx, 0, max_x);
      let sy = clamp(cy + dy, 0, max_y);
      min_val = min(min_val, textureLoad(inputTex, vec2<i32>(sx, sy), 0).r);
    }
  }

  textureStore(outputTex, vec2<i32>(cx, cy), vec4<f32>(min_val, min_val, min_val, 1.0));
}

@compute @workgroup_size(16, 16)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(inputTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let max_x = i32(size.x) - 1;
  let max_y = i32(size.y) - 1;
  let cx = i32(gid.x);
  let cy = i32(gid.y);

  var max_val = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let sx = clamp(cx + dx, 0, max_x);
      let sy = clamp(cy + dy, 0, max_y);
      max_val = max(max_val, textureLoad(inputTex, vec2<i32>(sx, sy), 0).r);
    }
  }

  textureStore(outputTex, vec2<i32>(cx, cy), vec4<f32>(max_val, max_val, max_val, 1.0));
}
