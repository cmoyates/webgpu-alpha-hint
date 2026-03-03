// alpha_hint.wgsl — normalized-chroma distance keyer
@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var outputTex : texture_storage_2d<rgba8unorm, write>;

struct Params {
  // Pre-normalized key color: key / (key.r + key.g + key.b + eps)
  key_norm_r: f32,
  key_norm_g: f32,
  key_norm_b: f32,
  softness: f32,   // chroma-distance transition width (0 = hard, larger = softer)
  gamma: f32,       // power curve on matte edges
  sat_gate: f32,    // saturation below which keying is suppressed (protects greys)
  pad0: f32,
  pad1: f32,
}
@group(0) @binding(2) var<uniform> params : Params;

const CHROMA_EPS: f32 = 0.0001;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(inputTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }

  let pixel_coords = vec2<i32>(i32(gid.x), i32(gid.y));
  let rgb = textureLoad(inputTex, pixel_coords, 0).rgb;

  // Normalized chroma: project pixel into chromaticity space
  let pixel_sum = rgb.r + rgb.g + rgb.b + CHROMA_EPS;
  let pixel_norm = rgb / pixel_sum;

  let key_norm = vec3<f32>(params.key_norm_r, params.key_norm_g, params.key_norm_b);

  // Euclidean distance in normalized-chroma space
  let chroma_dist = length(pixel_norm - key_norm);

  // Map distance to alpha: close to key => 0 (background), far => 1 (foreground)
  // softness controls the transition band width
  let half_softness = params.softness * 0.5;
  let distance_lower = max(half_softness, CHROMA_EPS);
  let distance_upper = distance_lower + params.softness;
  let smoothstep_input = clamp((chroma_dist - distance_lower) / (distance_upper - distance_lower + CHROMA_EPS), 0.0, 1.0);
  var alpha = smoothstep_input * smoothstep_input * (3.0 - 2.0 * smoothstep_input); // smoothstep

  // Saturation gate: protect low-saturation pixels (grey, white, black)
  let saturation = max(rgb.r, max(rgb.g, rgb.b)) - min(rgb.r, min(rgb.g, rgb.b));
  if (saturation < params.sat_gate) {
    // Lerp alpha toward 1.0 as saturation drops below gate
    let gate_factor = saturation / max(params.sat_gate, CHROMA_EPS);
    alpha = mix(1.0, alpha, gate_factor);
  }

  // Gamma bias on edges
  alpha = pow(alpha, params.gamma);

  // Write grayscale matte (no hard threshold — preserve soft edges)
  textureStore(outputTex, pixel_coords, vec4<f32>(alpha, alpha, alpha, 1.0));
}
