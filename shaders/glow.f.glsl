// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-gpu`.
//
// `piet-gpu` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-gpu` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-gpu`. If not, see <https://www.gnu.org/licenses/> or
// <https://www.mozilla.org/en-US/MPL/2.0/>.

// Fragment shader for OpenGL.
// Assume that the appropriate version of OpenGL is already set.

#ifdef GL_ES
precision mediump float;
#endif

// Vertex shader gives us the color, the texture coordinates, and the mask coordinates.
in vec4 fRgbaColor;
in vec2 fTexCoord;
in vec2 fMaskCoord;

// We also take samplers (textures) for the image and the mask.
uniform sampler2D uImage;
uniform sampler2D uMask;

void main() {
    vec4 textureColor = texture2D(uImage, fTexCoord);
    vec4 mainColor = fRgbaColor * textureColor;

    vec4 maskColor = texture2D(uMask, fMaskCoord);
    vec4 finalColor = mainColor * maskColor;

    gl_FragColor = finalColor;
}
