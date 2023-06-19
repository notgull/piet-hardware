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

const VERTS = array<vec4<f32>, 6>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>(1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0, 1.0, 0.0, 1.0),

    vec4<f32>(1.0, -1.0, 0.0, 1.0),
    vec4<f32>(1.0, 1.0, 0.0, 1.0),
    vec4<f32>(-1.0, 1.0, 0.0, 1.0),
);

@group(0) @binding(0) var<uniform> clearColor: u32;

fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((color >> 0u) & 255u),
        f32((color >> 8u) & 255u),
        f32((color >> 16u) & 255u),
        f32((color >> 24u) & 255u),
    ) / 255.0;
}

@vertex
fn vertex_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    if (in_vertex_index == 0u) {
        return VERTS[0];
    } else if (in_vertex_index == 1u) {
        return VERTS[1];
    } else if (in_vertex_index == 2u) {
        return VERTS[2];
    } else if (in_vertex_index == 3u) {
        return VERTS[3];
    } else if (in_vertex_index == 4u) {
        return VERTS[4];
    } else if (in_vertex_index == 5u) {
        return VERTS[5];
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return unpack_color(clearColor);
}
