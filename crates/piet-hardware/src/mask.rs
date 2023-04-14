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

//! The mask used for clipping.

use super::gpu_backend::{GpuContext, RepeatStrategy};
use super::resources::Texture;
use super::ResultExt;

use piet::kurbo::{Affine, PathEl, Shape};
use piet::{Error as Pierror, InterpolationMode};

use std::mem;
use std::rc::Rc;

use tiny_skia::{ClipMask, FillRule, PathBuilder, Pixmap};

/// A wrapper around an `Option<Mask>` that supports being easily drawn into.
pub(crate) struct MaskSlot<C: GpuContext + ?Sized> {
    /// The slot containing the mask.
    slot: MaskSlotState<C>,

    /// A cached path builder for drawing into the mask.
    path_builder: PathBuilder,
}

impl<C: GpuContext + ?Sized> Default for MaskSlot<C> {
    fn default() -> Self {
        Self {
            slot: MaskSlotState::Empty(None),
            path_builder: PathBuilder::new(),
        }
    }
}

enum MaskSlotState<C: GpuContext + ?Sized> {
    /// The mask slot is empty.
    ///
    /// We keep the texture around so that we can reuse it.
    Empty(Option<Texture<C>>),

    /// The mask slot is being drawn into.
    Mask(Mask<C>),
}

impl<C: GpuContext + ?Sized> MaskSlot<C> {
    /// Create a new mask slot.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Is this mask empty?
    pub(crate) fn is_empty(&self) -> bool {
        match &self.slot {
            MaskSlotState::Empty(_) => true,
            MaskSlotState::Mask(_) => false,
        }
    }

    /// Draw a shape into the mask.
    pub(crate) fn clip(
        &mut self,
        context: &Rc<C>,
        shape: impl Shape,
        tolerance: f64,
        transform: Affine,
        (width, height): (u32, u32),
    ) -> Result<(), Pierror> {
        // TODO: There has to be a better way of doing this.
        let path = {
            let path = shape.into_path(tolerance);
            let transformed = transform * path;

            let mut builder = mem::take(&mut self.path_builder);
            shape_to_skia_path(&mut builder, transformed, tolerance);
            builder.finish().expect("path builder failed")
        };

        match self.slot {
            MaskSlotState::Mask(ref mut mask) => {
                // Intersect the new path with the existing mask.
                mask.mask.intersect_path(&path, FillRule::EvenOdd, false);
                mask.dirty = true;
            }

            MaskSlotState::Empty(ref mut texture) => {
                // Create a mask if there isn't already one.
                let texture = match texture.take() {
                    Some(texture) => texture,
                    None => Texture::new(
                        context,
                        InterpolationMode::Bilinear,
                        RepeatStrategy::Color(piet::Color::TRANSPARENT),
                    )
                    .piet_err()?,
                };

                let mut mask = Mask {
                    texture,
                    pixmap: Pixmap::new(width, height).unwrap(),
                    mask: ClipMask::new(),
                    dirty: true,
                };

                mask.mask
                    .set_path(width, height, &path, FillRule::EvenOdd, false)
                    .ok_or_else(|| Pierror::BackendError("Failed to set clipping path".into()))?;

                self.slot = MaskSlotState::Mask(mask);
            }
        }

        self.path_builder = PathBuilder::new();
        Ok(())
    }

    /// Get the texture for this mask.
    pub(crate) fn texture(&mut self) -> Result<Option<&Texture<C>>, Pierror> {
        match self.slot {
            MaskSlotState::Mask(ref mut mask) => mask.upload().map(Some),

            MaskSlotState::Empty(_) => Ok(None),
        }
    }
}

struct Mask<C: GpuContext + ?Sized> {
    /// The texture that is used as the mask.
    texture: Texture<C>,

    /// The pixmap we use as scratch space for drawing.
    pixmap: tiny_skia::Pixmap,

    /// The clipping mask we use to calculate the mask.
    mask: tiny_skia::ClipMask,

    /// Whether the mask contains data that needs to be uploaded to the texture.
    dirty: bool,
}

impl<C: GpuContext + ?Sized> Mask<C> {
    /// Upload the mask to the texture.
    fn upload(&mut self) -> Result<&Texture<C>, Pierror> {
        if self.dirty {
            // First, clear the pixmap.
            self.pixmap.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0));

            // Now, composite the mask onto the pixmap.
            let paint = tiny_skia::Paint {
                shader: tiny_skia::Shader::SolidColor(tiny_skia::Color::from_rgba8(
                    0xFF, 0xFF, 0xFF, 0xFF,
                )),
                ..Default::default()
            };
            let rect = tiny_skia::Rect::from_xywh(
                0.0,
                0.0,
                self.pixmap.width() as f32,
                self.pixmap.height() as f32,
            )
            .unwrap();
            self.pixmap.fill_rect(
                rect,
                &paint,
                tiny_skia::Transform::identity(),
                Some(&self.mask),
            );

            // Finally, upload the pixmap to the texture.
            let data = self.pixmap.data();
            self.texture.write_texture(
                (self.pixmap.width(), self.pixmap.height()),
                piet::ImageFormat::RgbaSeparate,
                Some(data),
            );

            self.dirty = false;
        }

        Ok(&self.texture)
    }
}

fn shape_to_skia_path(builder: &mut PathBuilder, shape: impl Shape, tolerance: f64) {
    shape.path_elements(tolerance).for_each(|el| match el {
        PathEl::MoveTo(pt) => builder.move_to(pt.x as f32, pt.y as f32),
        PathEl::LineTo(pt) => builder.line_to(pt.x as f32, pt.y as f32),
        PathEl::QuadTo(p1, p2) => {
            builder.quad_to(p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32)
        }
        PathEl::CurveTo(p1, p2, p3) => builder.cubic_to(
            p1.x as f32,
            p1.y as f32,
            p2.x as f32,
            p2.y as f32,
            p3.x as f32,
            p3.y as f32,
        ),
        PathEl::ClosePath => builder.close(),
    })
}
