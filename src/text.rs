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

//! Uses a text atlas for text rendering.

use crate::resources::Texture;

use cosmic_text::CacheKey as GlyphKey;
use etagere::{Allocation, AtlasAllocator, Size};
use glow::HasContext;

use piet::Error;

use std::cell::RefCell;
use std::collections::hash_map::HashMap;

use std::rc::Rc;

/// The text atlas for text rendering.
struct Atlas<H: HasContext + ?Sized> {
    /// Reference to the context.
    context: Rc<H>,

    /// The list of textures in our atlas.
    textures: RefCell<Vec<(AtlasAllocator, Texture<H>)>>,

    /// Max size of a texture.
    max_size: Size,

    /// Map keeping track of allocations.
    allocations: RefCell<HashMap<GlyphKey, GlyphInfo>>,
}

impl<H: HasContext + ?Sized> Atlas<H> {
    /// Create a new text atlas.
    fn new(context: &Rc<H>) -> Result<Self, Error> {
        // Get the maximum texture size.
        let max_size = unsafe {
            let max_dim = context.get_parameter_i32(glow::MAX_TEXTURE_SIZE);
            Size::new(max_dim as _, max_dim as _)
        };

        Ok(Self {
            context: context.clone(),
            textures: RefCell::new(Vec::with_capacity(1)),
            max_size,
            allocations: RefCell::new(HashMap::new()),
        })
    }
}

struct GlyphInfo {
    /// The key for this glyph.
    key: GlyphKey,

    /// The index into the `textures` vector.
    texture_index: usize,

    /// The allocation in the texture.
    allocation: Allocation,
}
