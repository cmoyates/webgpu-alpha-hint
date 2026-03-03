// blur.wgsl — separable box blur (horizontal + vertical entry points)
@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var outputTex : texture_storage_2d<rgba8unorm, write>;

struct BlurParams {
  radius: i32,
  pad0: i32,
  pad1: i32,
  pad2: i32,
}
@group(0) @binding(2) var<uniform> blur_params : BlurParams;

@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(inputTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let r = blur_params.radius;
  let y = i32(gid.y);
  let max_x = i32(size.x) - 1;
  var sum = vec3<f32>(0.0);
  var count = 0.0;

  for (var dx = -r; dx <= r; dx = dx + 1) {
    let sx = clamp(i32(gid.x) + dx, 0, max_x);
    sum += textureLoad(inputTex, vec2<i32>(sx, y), 0).rgb;
    count += 1.0;
  }

  let avg = sum / count;
  textureStore(outputTex, vec2<i32>(i32(gid.x), y), vec4<f32>(avg, 1.0));
}

@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(inputTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let r = blur_params.radius;
  let x = i32(gid.x);
  let max_y = i32(size.y) - 1;
  var sum = vec3<f32>(0.0);
  var count = 0.0;

  for (var dy = -r; dy <= r; dy = dy + 1) {
    let sy = clamp(i32(gid.y) + dy, 0, max_y);
    sum += textureLoad(inputTex, vec2<i32>(x, sy), 0).rgb;
    count += 1.0;
  }

  let avg = sum / count;
  textureStore(outputTex, vec2<i32>(x, i32(gid.y)), vec4<f32>(avg, 1.0));
}
