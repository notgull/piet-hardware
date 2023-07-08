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
use super::{shape_to_skia_path, ResultExt};

use piet::kurbo::{Affine, Shape};
use piet::{Error as Pierror, InterpolationMode};

use std::{fmt, mem};

use tiny_skia::{FillRule, Mask as ClipMask, PathBuilder, Pixmap};

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

impl<C: GpuContext + ?Sized> fmt::Debug for MaskSlot<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Ellipses;
        impl fmt::Debug for Ellipses {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("...")
            }
        }

        match self.slot {
            MaskSlotState::Empty(_) => f.debug_struct("MaskSlot").finish_non_exhaustive(),
            MaskSlotState::Mask(_) => f.debug_struct("MaskSlot").field("mask", &Ellipses).finish(),
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
        context: &mut C,
        device: &C::Device,
        shape: impl Shape,
        tolerance: f64,
        transform: Affine,
        (width, height): (u32, u32),
    ) -> Result<(), Pierror> {
        let path = {
            let mut builder = mem::take(&mut self.path_builder);
            shape_to_skia_path(&mut builder, shape, tolerance);
            builder.finish().expect("path builder failed")
        };

        match self.slot {
            MaskSlotState::Mask(ref mut mask) => {
                // Intersect the new path with the existing mask.
                mask.mask.intersect_path(
                    &path,
                    FillRule::EvenOdd,
                    false,
                    tiny_skia::Transform::identity(),
                );
                mask.dirty = true;
            }

            MaskSlotState::Empty(ref mut texture) => {
                // Create a mask if there isn't already one.
                let texture = match texture.take() {
                    Some(texture) => texture,
                    None => Texture::new(
                        context,
                        device,
                        InterpolationMode::Bilinear,
                        RepeatStrategy::Color(piet::Color::TRANSPARENT),
                    )
                    .piet_err()?,
                };

                let mut mask = Mask {
                    texture,
                    pixmap: Pixmap::new(width, height).unwrap(),
                    mask: ClipMask::new(width, height).unwrap(),
                    dirty: true,
                };

                mask.mask
                    .fill_path(&path, FillRule::EvenOdd, false, cvt_transform(transform));

                self.slot = MaskSlotState::Mask(mask);
            }
        }

        self.path_builder = PathBuilder::new();
        Ok(())
    }

    /// Get the texture for this mask.
    pub(crate) fn texture(
        &mut self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
    ) -> Result<Option<&Texture<C>>, Pierror> {
        match self.slot {
            MaskSlotState::Mask(ref mut mask) => mask.upload(context, device, queue).map(Some),

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
    mask: ClipMask,

    /// Whether the mask contains data that needs to be uploaded to the texture.
    dirty: bool,
}

impl<C: GpuContext + ?Sized> Mask<C> {
    /// Upload the mask to the texture.
    fn upload(
        &mut self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
    ) -> Result<&Texture<C>, Pierror> {
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
                context,
                device,
                queue,
                (self.pixmap.width(), self.pixmap.height()),
                piet::ImageFormat::RgbaSeparate,
                Some(data),
            );

            self.dirty = false;
        }

        Ok(&self.texture)
    }
}

fn cvt_transform(p: kurbo::Affine) -> tiny_skia::Transform {
    tiny_skia::Transform::from_row(
        p.as_coeffs()[0] as f32,
        p.as_coeffs()[1] as f32,
        p.as_coeffs()[2] as f32,
        p.as_coeffs()[3] as f32,
        p.as_coeffs()[4] as f32,
        p.as_coeffs()[5] as f32,
    )
}
