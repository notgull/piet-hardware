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
use super::shape_to_skia_path;

use piet::kurbo::Shape;
use piet::InterpolationMode;

use std::{fmt, mem};

use tiny_skia as ts;
use ts::{FillRule, Mask as ClipMask, PathBuilder, PixmapMut};

/// The context for creating and modifying masks.
pub(crate) struct MaskContext<C: GpuContext + ?Sized> {
    /// A scratch buffer for rendering masks into.
    mask_render_buffer: Vec<u32>,

    /// List of GPU textures to re-use.
    gpu_textures: Vec<Texture<C>>,

    /// Cached path builder for drawing into the mask.
    path_builder: PathBuilder,
}

/// A mask that can be clipped into.
pub(crate) struct Mask<C: GpuContext + ?Sized> {
    /// The underlying tiny-skia mask.
    mask: ClipMask,

    /// The current state of the mask.
    state: MaskState<C>,
}

enum MaskState<C: GpuContext + ?Sized> {
    /// The mask is empty.
    Empty,

    /// The mask has data and the texture has not been created yet.
    DirtyWithNoTexture,

    /// The mask has data but the texture is out of date.
    DirtyWithTexture(Texture<C>),

    /// The mask has data and the texture is up to date.
    Clean(Texture<C>),
}

impl<C: GpuContext + ?Sized> MaskState<C> {
    /// Move into the dirty state.
    fn dirty(&mut self) {
        let new = match mem::replace(self, Self::Empty) {
            Self::Empty => Self::DirtyWithNoTexture,
            Self::Clean(tex) => Self::DirtyWithTexture(tex),
            rem => rem,
        };

        *self = new;
    }

    /// Is this mask dirty?
    fn is_dirty(&self) -> bool {
        matches!(self, Self::DirtyWithNoTexture | Self::DirtyWithTexture(_))
    }

    /// Get a reference to the texture, if there is one.
    fn texture(&self) -> Option<&Texture<C>> {
        match self {
            Self::Clean(tex) | Self::DirtyWithTexture(tex) => Some(tex),
            _ => None,
        }
    }

    /// Take the texture out.
    fn take_texture(&mut self) -> Option<Texture<C>> {
        let (new, tex) = match mem::replace(self, Self::Empty) {
            Self::Clean(tex) => (Self::DirtyWithNoTexture, Some(tex)),
            Self::DirtyWithTexture(tex) => (Self::DirtyWithNoTexture, Some(tex)),
            rem => (rem, None),
        };

        *self = new;
        tex
    }

    /// Tell whether this is `Empty`.
    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

impl<C: GpuContext + ?Sized> MaskContext<C> {
    /// Create a new, empty mask context.
    pub(crate) fn new() -> Self {
        Self {
            mask_render_buffer: Vec::new(),
            gpu_textures: Vec::new(),
            path_builder: PathBuilder::new(),
        }
    }

    /// Add a new path to a mask.
    pub(crate) fn add_path(&mut self, mask: &mut Mask<C>, shape: impl Shape, tolerance: f64) {
        // Convert the shape to a tiny-skia path.
        let path = {
            let mut builder = mem::take(&mut self.path_builder);
            shape_to_skia_path(&mut builder, shape, tolerance);
            builder.finish().expect("path builder failed")
        };

        if mask.state.is_empty() {
            // This is the first stroke, so fill the mask with the path.
            mask.mask
                .fill_path(&path, FillRule::EvenOdd, false, ts::Transform::identity());
        } else {
            // This is an intersection, so intersect the path with the mask.
            mask.mask
                .intersect_path(&path, FillRule::EvenOdd, false, ts::Transform::identity());
        }

        mask.state.dirty();
        self.path_builder = path.clear();
    }

    /// Get the texture for a mask.
    pub(crate) fn texture<'a>(
        &mut self,
        mask: &'a mut Mask<C>,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
    ) -> &'a Texture<C> {
        self.upload_mask(mask, context, device, queue);
        mask.state.texture().expect("mask texture")
    }

    /// Upload a mask into a texture.
    fn upload_mask(
        &mut self,
        mask: &mut Mask<C>,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
    ) {
        if mask.state.is_empty() {
            unreachable!("uploading empty mask");
        }

        if !mask.state.is_dirty() {
            // No need to change anything.
            return;
        }

        // Create a pixmap to render into, using our scratch space.
        // TODO: It would be nice to go right from the clip mask to the texture without using the
        // pixmap as an intermediary.
        let width = mask.mask.width();
        let height = mask.mask.height();
        self.mask_render_buffer
            .resize(width as usize * height as usize, 0);
        let mut pixmap = PixmapMut::from_bytes(
            bytemuck::cast_slice_mut(&mut self.mask_render_buffer),
            width,
            height,
        )
        .expect("If the width/height is valid for the mask, it should work for the pixmap as well");

        // Clear the pixmap with a black color.
        pixmap.fill(ts::Color::TRANSPARENT);

        // Render the mask into the pixmap.
        pixmap.fill_rect(
            ts::Rect::from_xywh(0., 0., width as f32, height as f32).expect("valid rect"),
            &ts::Paint {
                shader: ts::Shader::SolidColor(ts::Color::WHITE),
                ..Default::default()
            },
            ts::Transform::identity(),
            Some(&mask.mask),
        );

        // Either create a new GPU texture or re-use an older one.
        let texture = mask
            .state
            .take_texture()
            .or_else(|| self.gpu_textures.pop())
            .unwrap_or_else(|| {
                Texture::new(
                    context,
                    device,
                    InterpolationMode::Bilinear,
                    RepeatStrategy::Color(piet::Color::TRANSPARENT),
                )
                .expect("failed to create texture")
            });

        // Upload the pixmap to the texture.
        texture.write_texture(
            context,
            device,
            queue,
            (width, height),
            piet::ImageFormat::RgbaSeparate,
            Some(pixmap.data_mut()),
        );

        // Put the texture back.
        //
        // This also marks the texture as non-dirty.
        mask.state = MaskState::Clean(texture);
    }

    /// Reclaim the textures of a set of masks.
    pub(crate) fn reclaim(&mut self, masks: impl Iterator<Item = Mask<C>>) {
        // Take out all of the GPU textures.
        self.gpu_textures
            .extend(masks.flat_map(|mut mask| mask.state.take_texture()));
    }
}

impl<C: GpuContext + ?Sized> Mask<C> {
    /// Create a new mask with the given size.
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self {
            mask: ClipMask::new(width, height).expect("failed to create mask"),
            state: MaskState::Empty,
        }
    }
}

impl<C: GpuContext + ?Sized> Clone for Mask<C> {
    /// Makes a new copy without the cached texture.
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
            state: if self.state.is_empty() {
                MaskState::Empty
            } else {
                MaskState::DirtyWithNoTexture
            },
        }
    }
}

impl<C: GpuContext + ?Sized> fmt::Debug for Mask<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mask")
            .field("width", &self.mask.width())
            .field("height", &self.mask.height())
            .field("state", &self.state)
            .finish()
    }
}

impl<C: GpuContext + ?Sized> fmt::Debug for MaskState<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("Empty"),
            Self::DirtyWithNoTexture => f.write_str("DirtyWithNoTexture"),
            Self::DirtyWithTexture(_) => f.write_str("DirtyWithTexture"),
            Self::Clean(_) => f.write_str("Clean"),
        }
    }
}
