// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/> or
// <https://www.mozilla.org/en-US/MPL/2.0/>.

//! The text atlas, which is used to cache glyphs.

use super::gpu_backend::{GpuContext, RepeatStrategy};
use super::resources::Texture;
use super::ResultExt;

use ahash::RandomState;
use cosmic_text::{CacheKey, LayoutGlyph};
use etagere::{Allocation, AtlasAllocator};
use hashbrown::hash_map::{Entry, HashMap};

use piet::kurbo::{Point, Rect};
use piet::{Error as Pierror, InterpolationMode};

use std::rc::Rc;

/// The atlas, combining all of the glyphs into a single texture.
pub(crate) struct Atlas<C: GpuContext + ?Sized> {
    /// The texture atlas.
    texture: Rc<Texture<C>>,

    /// The size of the texture atlas.
    size: (u32, u32),

    /// The allocator for the texture atlas.
    allocator: AtlasAllocator,

    /// The hash map between the glyphs used and the texture allocation.
    glyphs: HashMap<CacheKey, Position, RandomState>,
}

/// The positioning of a glyph in the atlas.
struct Position {
    /// The allocation of the glyph in the atlas.
    allocation: Allocation,

    /// The offset at which to draw the glyph.
    offset: Point,
}

impl<C: GpuContext + ?Sized> Atlas<C> {
    /// Create a new, empty texture atlas.
    pub(crate) fn new(context: &Rc<C>) -> Result<Self, Pierror> {
        let (max_width, max_height) = context.max_texture_size();
        let texture = Texture::new(
            context,
            InterpolationMode::Bilinear,
            RepeatStrategy::Color(piet::Color::TRANSPARENT),
        )
        .piet_err()?;

        // Initialize the texture to be transparent.
        texture.write_texture((max_width, max_height), piet::ImageFormat::RgbaPremul, None);

        Ok(Atlas {
            texture: Rc::new(texture),
            size: (max_width, max_height),
            allocator: AtlasAllocator::new([max_width as i32, max_height as i32].into()),
            glyphs: HashMap::with_hasher(RandomState::new()),
        })
    }

    /// Get a reference to the inner texture.
    pub(crate) fn texture(&self) -> &Rc<Texture<C>> {
        &self.texture
    }

    /// Get the UV rectangle for the given glyph.
    ///
    /// This function rasterizes the glyph if it isn't already cached.
    pub(crate) fn uv_rect(
        &mut self,
        glyph: &LayoutGlyph,
        font_data: &cosmic_text::Font<'_>,
    ) -> Result<(Rect, Point), Pierror> {
        let alloc_to_rect = {
            let (width, height) = self.size;
            move |posn: &Position| {
                let alloc = &posn.allocation;

                (
                    Rect::new(
                        alloc.rectangle.min.x as f64 / width as f64,
                        alloc.rectangle.min.y as f64 / height as f64,
                        alloc.rectangle.max.x as f64 / width as f64,
                        alloc.rectangle.max.y as f64 / height as f64,
                    ),
                    posn.offset,
                )
            }
        };

        let key = glyph.cache_key;

        match self.glyphs.entry(key) {
            Entry::Occupied(o) => {
                let alloc = o.get();
                Ok(alloc_to_rect(alloc))
            }

            Entry::Vacant(v) => {
                use ab_glyph::Font as _;

                // Q: Why are we using ab_glyph instead of swash, which cosmic-text uses?
                // A: ab_glyph already exists in the winit dep tree, which this crate is intended
                //    to be used with.
                let font_ref = ab_glyph::FontRef::try_from_slice(font_data.data).piet_err()?;
                let glyph_id = ab_glyph::GlyphId(glyph.cache_key.glyph_id)
                    .with_scale(glyph.cache_key.font_size as f32);
                let outline = font_ref.outline_glyph(glyph_id).ok_or_else(|| {
                    Pierror::BackendError({
                        format!("Failed to outline glyph {}", glyph.cache_key.glyph_id).into()
                    })
                })?;

                let bounds = outline.px_bounds();
                let (width, height) = (bounds.width() as i32, bounds.height() as i32);
                let mut buffer = vec![0u32; (width * height) as usize];

                // Draw the glyph.
                outline.draw(|x, y, c| {
                    let pixel = {
                        let pixel_offset = (x + y * width as u32) as usize;

                        match buffer.get_mut(pixel_offset) {
                            Some(pixel) => pixel,
                            None => return,
                        }
                    };

                    // Convert the color to a u32.
                    let color = {
                        let cbyte = (255.0 * c) as u8;
                        u32::from_ne_bytes([cbyte, cbyte, cbyte, cbyte])
                    };

                    // Set the pixel.
                    *pixel = color;
                });

                // Find a place for it in the texture.
                let alloc = self
                    .allocator
                    .allocate([width, height].into())
                    .ok_or_else(|| {
                        Pierror::BackendError("Failed to allocate glyph in texture atlas.".into())
                    })?;

                // Insert the glyph into the texture.
                self.texture.write_subtexture(
                    (alloc.rectangle.min.x as u32, alloc.rectangle.min.y as u32),
                    (width as u32, height as u32),
                    piet::ImageFormat::RgbaPremul,
                    bytemuck::cast_slice::<_, u8>(&buffer),
                );

                // Insert the allocation into the map.
                let alloc = v.insert(Position {
                    allocation: alloc,
                    offset: Point::new(
                        bounds.min.x.into(),
                        (glyph.cache_key.font_size as f32 + bounds.min.y).into(),
                    ),
                });

                // Return the UV rectangle.
                Ok(alloc_to_rect(alloc))
            }
        }
    }
}
