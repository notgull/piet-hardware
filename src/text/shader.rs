// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-glow`.
// 
// `piet-glow` is free software: you can redistribute it and/or modify it under the terms of
// either:
// 
// * GNU Lesser General Public License as published by the Free Software Foundation, either 
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2. 
// 
// `piet-glow` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License and the Mozilla 
// Public License along with `piet-glow`. If not, see <https://www.gnu.org/licenses/>.

//! Shader for drawing text.

use euclid::default::{Point, Rect, Size};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct GlyphPoint {
    /// The position of the point.
    pub(super) position: Point<f32>,
}
