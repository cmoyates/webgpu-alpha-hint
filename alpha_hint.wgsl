// alpha_hint.wgsl
@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var outputTex : texture_storage_2d<rgba8unorm, write>;

struct Params {
  t_low: f32;
  t_high: f32;
  gamma: f32;
  pad: f32;
};
@group(0) @binding(2) var<uniform> params : Params;

fn smooth01(x: f32, a: f32, b: f32) -> f32 {
  // smoothstep(a, b, x) but explicit
  let t = clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(inputTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let uv = vec2<i32>(i32(gid.x), i32(gid.y));
  let rgb = textureLoad(inputTex, uv, 0).rgb;

  // "Excess green" key metric (bigger => more green => more background)
  let excessG = rgb.g - max(rgb.r, rgb.b);

  // Map to matte: background => 0, subject => 1
  // We invert by using -excessG as the "keep" signal.
  var a = smooth01(-excessG, params.t_low, params.t_high);

  // Bias matte edges
  a = pow(a, params.gamma);

  // Write grayscale in RGB (and alpha=1)
  textureStore(outputTex, uv, vec4<f32>(a, a, a, 1.0));
}