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

//! The text atlas, which is used to cache glyphs.

use super::gpu_backend::{GpuContext, RepeatStrategy};
use super::resources::Texture;
use super::ResultExt;

use ahash::RandomState;
use cosmic_text::{CacheKey, FontSystem, LayoutGlyph, Placement, SwashCache, SwashContent};
use etagere::{Allocation, AtlasAllocator};
use hashbrown::hash_map::{Entry, HashMap};

use piet::kurbo::{Point, Rect, Size};
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

    /// The cache for the swash layout.
    swash_cache: SwashCache,
}

/// The data needed for rendering a glyph.
pub(crate) struct GlyphData {
    /// The UV rectangle for the glyph.
    pub(crate) uv_rect: Rect,

    /// The size of the glyph.
    pub(crate) size: Size,

    /// The offset at which to draw the glyph.
    pub(crate) offset: Point,
}

/// The positioning of a glyph in the atlas.
struct Position {
    /// The allocation of the glyph in the atlas.
    allocation: Allocation,

    /// Placement of the glyph.
    placement: Placement,
}

impl<C: GpuContext + ?Sized> Atlas<C> {
    /// Create a new, empty texture atlas.
    pub(crate) fn new(
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
    ) -> Result<Self, Pierror> {
        let (max_width, max_height) = context.max_texture_size(device);
        let texture = Texture::new(
            context,
            device,
            InterpolationMode::Bilinear,
            RepeatStrategy::Color(piet::Color::TRANSPARENT),
        )
        .piet_err()?;

        // Initialize the texture to be transparent.
        texture.write_texture(
            context,
            device,
            queue,
            (max_width, max_height),
            piet::ImageFormat::RgbaPremul,
            None,
        );

        Ok(Atlas {
            texture: Rc::new(texture),
            size: (max_width, max_height),
            allocator: AtlasAllocator::new([max_width as i32, max_height as i32].into()),
            glyphs: HashMap::with_hasher(RandomState::new()),
            swash_cache: SwashCache::new(),
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
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        glyph: &LayoutGlyph,
        font_system: &mut FontSystem,
    ) -> Result<GlyphData, Pierror> {
        let alloc_to_rect = {
            let (width, height) = self.size;
            move |posn: &Position| {
                let alloc = &posn.allocation;

                let max_x = alloc.rectangle.min.x + posn.placement.width as i32;
                let max_y = alloc.rectangle.min.y + posn.placement.height as i32;

                let uv_rect = Rect::new(
                    alloc.rectangle.min.x as f64 / width as f64,
                    alloc.rectangle.min.y as f64 / height as f64,
                    max_x as f64 / width as f64,
                    max_y as f64 / height as f64,
                );
                let offset = (posn.placement.left as f64, posn.placement.top as f64);
                let size = (posn.placement.width as f64, posn.placement.height as f64);

                GlyphData {
                    uv_rect,
                    size: size.into(),
                    offset: offset.into(),
                }
            }
        };

        // TODO: Scaling.
        let physical = glyph.physical((0.0, 0.0), 1.0);
        let key = physical.cache_key;

        match self.glyphs.entry(key) {
            Entry::Occupied(o) => {
                let alloc = o.get();
                Ok(alloc_to_rect(alloc))
            }

            Entry::Vacant(v) => {
                // Get the swash image.
                let sw_image = self
                    .swash_cache
                    .get_image_uncached(font_system, key)
                    .ok_or_else(|| {
                        Pierror::BackendError({
                            format!("Failed to outline glyph {}", glyph.glyph_id).into()
                        })
                    })?;

                // Render it to a buffer.
                let mut buffer = vec![
                    0u32;
                    sw_image.placement.width as usize
                        * sw_image.placement.height as usize
                ];
                match sw_image.content {
                    SwashContent::Color => {
                        // Copy the color to the buffer.
                        buffer
                            .iter_mut()
                            .zip(sw_image.data.chunks(4))
                            .for_each(|(buf, input)| {
                                let color =
                                    u32::from_ne_bytes([input[0], input[1], input[2], input[3]]);
                                *buf = color;
                            });
                    }
                    SwashContent::Mask => {
                        // Copy the mask to the buffer.
                        buffer
                            .iter_mut()
                            .zip(sw_image.data.iter())
                            .for_each(|(buf, input)| {
                                let color = u32::from_ne_bytes([255, 255, 255, *input]);
                                *buf = color;
                            });
                    }
                    content => {
                        tracing::warn!("Unsupported swash content: {:?}", content);
                        return Err(Pierror::NotSupported);
                    }
                }

                let (width, height) = (sw_image.placement.width, sw_image.placement.height);

                // Find a place for it in the texture.
                let alloc = self
                    .allocator
                    .allocate([width as i32, height as i32].into())
                    .ok_or_else(|| {
                        Pierror::BackendError("Failed to allocate glyph in texture atlas.".into())
                    })?;

                // Insert the glyph into the texture.
                self.texture.write_subtexture(
                    context,
                    device,
                    queue,
                    (alloc.rectangle.min.x as u32, alloc.rectangle.min.y as u32),
                    (width, height),
                    piet::ImageFormat::RgbaPremul,
                    bytemuck::cast_slice::<_, u8>(&buffer),
                );

                // Insert the allocation into the map.
                let alloc = v.insert(Position {
                    allocation: alloc,
                    placement: sw_image.placement,
                });

                // Return the UV rectangle.
                Ok(alloc_to_rect(alloc))
            }
        }
    }
}
