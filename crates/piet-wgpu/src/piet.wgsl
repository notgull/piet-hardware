// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the
// terms of either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
//   version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
// * The Patron License (https://github.com/notgull/piet-hardware/blob/main/LICENSE-PATRON.md)
//   for sponsors and contributors, who can ignore the copyleft provisions of the above licenses
//   for this project.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License or the Mozilla Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/>.

struct Uniforms {
    // 3x3 matrix for transforming vertices.
    transform: mat3x3<f32>,

    // Viewport size.
    viewport_size: vec2<f32>,
};

struct VertexShaderOutput {
    @location(0) tex_coords: vec2<f32>,
    @location(1) mask_coords: vec2<f32>,
    @location(2) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var texColor: texture_2d<f32>;
@group(1) @binding(1) var texSampler: sampler;
@group(2) @binding(0) var maskColor: texture_2d<f32>;
@group(2) @binding(1) var maskSampler: sampler;

fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(color & 255u),
        f32((color >> 8u) & 255u),
        f32((color >> 16u) & 255u),
        f32((color >> 24u) & 255u),
    ) / 255.0;
}

fn unpack_position(posn: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        (2.0 * posn.x / uniforms.viewport_size.x) - 1.0,
        1.0 - (2.0 * posn.y / uniforms.viewport_size.y),
        0.0,
        1.0,
    );
}

struct InVertex {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color: u32,
}

@vertex
fn vertex_main(vert: InVertex) -> VertexShaderOutput {
    var out: VertexShaderOutput;

    // Transform the vertex position.
    var pos: vec3<f32> = uniforms.transform * vec3<f32>(vert.position, 1.0);
    pos = pos / pos.z;

    out.position = unpack_position(pos.xy);
    out.tex_coords = vert.tex_coords;
    out.mask_coords = vec2<f32>(
        vert.tex_coords.x / uniforms.viewport_size.x,
        vert.tex_coords.y / uniforms.viewport_size.y,
    );
    out.color = unpack_color(vert.color);

    return out;
}

@fragment
fn fragment_main(in: VertexShaderOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(texColor, texSampler, in.tex_coords);
    let mask_color = textureSample(maskColor, maskSampler, in.mask_coords);

    let main_color = in.color * tex_color;
    return main_color * mask_color;
}

