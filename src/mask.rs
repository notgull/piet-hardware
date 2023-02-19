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

//! Handles the mask.
//!
//! There is probably a better way of handling masking in OpenGL, but if it exists I
//! don't know about it. Right now, the best way is to fall back to software rendering
//! using the `tiny-skia` crate. In the future, we may want to migrate to a
//! hardware-accelerated solution.
//!
//! TODO: Find a way to reduce the amount of rendering done here. On Debug mode, this
//! causes a very noticable slowdown.

use crate::resources::{BoundFramebuffer, Framebuffer, Texture};
use crate::{Error, RenderContext};

use glow::HasContext;
use piet::kurbo::{Affine, PathEl, Shape};
use std::mem;
use std::rc::Rc;
use tiny_skia::{ClipMask, Path, PathBuilder, Pixmap};

pub(super) struct Mask<H: HasContext + ?Sized> {
    /// The frame buffer for drawing to the mask.
    framebuffer: Framebuffer<H>,

    /// The texture for this mask.
    texture: Texture<H>,

    /// The pixmap containing the clip for this mask.
    pixmap: Option<Pixmap>,

    /// The current clip for this mask.
    clip: ClipMask,

    /// Reusable `PathBuilder` for drawing shapes.
    path_builder: PathBuilder,

    /// Do the contents of `pixmap` and `texture` match?
    dirty: bool,

    /// Whether the `clip` is empty.
    empty: bool,

    /// The transform for this mask.
    transform: Affine,

    /// The width and height of this mask.
    size: (u32, u32),
}

impl<H: HasContext + ?Sized> Mask<H> {
    /// Create a new, empty mask.
    pub(crate) fn new(
        context: &Rc<H>,
        width: u32,
        height: u32,
        transform: Affine,
    ) -> Result<Self, Error> {
        // Create a texture and fill it with nothingness.
        let texture = Texture::new(context)?;

        {
            let mut bound = texture.bind(None);
            bound.fill_with_nothing(width as _, height as _);
            bound.filtering_nearest();
        }

        // Create a framebuffer.
        let framebuffer = Framebuffer::new(context)?;

        // Create a pixmap.
        let mut pixmap = Pixmap::new(width, height).unwrap();
        pixmap
            .as_mut()
            .fill(tiny_skia::Color::from_rgba8(0xFF, 0xFF, 0xFF, 0xFF));

        // Create a clipping mask.
        let clip_mask = ClipMask::new();

        Ok(Self {
            texture,
            framebuffer,
            pixmap: Some(pixmap),
            clip: clip_mask,
            path_builder: PathBuilder::new(),
            dirty: true,
            empty: true,
            transform,
            size: (width, height),
        })
    }

    /// Set the transform of this mask.
    pub(crate) fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    fn pixmap(&mut self) -> &mut Pixmap {
        self.pixmap.as_mut().unwrap()
    }

    /// Clear out the mask.
    pub(crate) fn clear(&mut self) {
        self.pixmap()
            .as_mut()
            .fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0));
        self.clip.clear();

        self.empty = true;
        self.dirty = true;
    }

    /// Resize this mask, clearing it in the process.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.size = (width, height);
        self.pixmap = Some(Pixmap::new(width, height).unwrap());

        self.clear();
    }

    /// Get the size of this mask.
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Add a path clip to this mask.
    pub(crate) fn add_path(&mut self, shape: impl Shape, tolerance: f64) {
        // Build to a path.
        let path = {
            let mut builder = mem::replace(&mut self.path_builder, PathBuilder::new());
            convert_kurbo_path_to_ts(&mut builder, shape, tolerance);
            match builder.finish() {
                Some(path) => path,
                None => return,
            }
        };

        // Add the path to the clip mask.
        if self.empty {
            self.clip
                .set_path(
                    self.size.0,
                    self.size.1,
                    &path,
                    tiny_skia::FillRule::EvenOdd,
                    false,
                )
                .unwrap();
            self.empty = false;
        } else {
            self.clip
                .intersect_path(&path, tiny_skia::FillRule::EvenOdd, false)
                .unwrap();
        }

        // Mark the mask as dirty.
        self.dirty = true;

        // Restore the path builder.
        self.path_builder = path.clear();
    }

    /// Update the texture.
    pub(crate) fn update_texture(&mut self) {
        if self.dirty {
            // Fill the pixmap with alpha.
            self.pixmap()
                .as_mut()
                .fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0));

            // Clip the pixmap.
            let mut paint = tiny_skia::Paint::default();
            paint.set_color_rgba8(0xFF, 0xFF, 0xFF, 0xFF);
            let (width, height) = self.size;

            self.pixmap
                .as_mut()
                .unwrap()
                .as_mut()
                .fill_rect(
                    tiny_skia::Rect::from_xywh(0.0, 0.0, width as _, height as _).unwrap(),
                    &paint,
                    tiny_skia::Transform::identity(),
                    Some(&self.clip),
                )
                .unwrap();

            // Update the texture.
            let mut bound = self.texture.bind(None);
            bound
                .fill_with_image(
                    self.size.0 as _,
                    self.size.1 as _,
                    piet::ImageFormat::RgbaSeparate,
                    self.pixmap.as_ref().unwrap().as_ref().data(),
                )
                .unwrap();

            self.dirty = false;
        }
    }

    /// Get the transform of this mask.
    pub(crate) fn transform(&self) -> &Affine {
        &self.transform
    }

    /// Get the texture.
    pub(crate) fn texture(&self) -> &Texture<H> {
        &self.texture
    }

    pub(crate) fn as_brush_mask(&self) -> crate::brush::Mask<'_, H> {
        crate::brush::Mask {
            texture: &self.texture,
            transform: &self.transform,
        }
    }
}

fn convert_kurbo_path_to_ts(builder: &mut PathBuilder, shape: impl Shape, tolerance: f64) {
    shape.path_elements(tolerance).for_each(|el| match el {
        PathEl::MoveTo(pt) => {
            builder.move_to(pt.x as f32, pt.y as f32);
        }

        PathEl::LineTo(pt) => {
            builder.line_to(pt.x as f32, pt.y as f32);
        }

        PathEl::QuadTo(ctrl1, end) => {
            builder.quad_to(ctrl1.x as f32, ctrl1.y as f32, end.x as f32, end.y as f32);
        }

        PathEl::CurveTo(ctrl1, ctrl2, end) => {
            builder.cubic_to(
                ctrl1.x as f32,
                ctrl1.y as f32,
                ctrl2.x as f32,
                ctrl2.y as f32,
                end.x as f32,
                end.y as f32,
            );
        }

        PathEl::ClosePath => {
            builder.close();
        }
    })
}

fn convert_kurbo_affine_to_ts(affine: &Affine) -> tiny_skia::Transform {
    let [a, b, c, d, tx, ty] = affine.as_coeffs();
    tiny_skia::Transform::from_row(a as f32, b as f32, c as f32, d as f32, tx as f32, ty as f32)
}

struct CallOnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for CallOnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}
