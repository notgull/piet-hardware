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

use crate::resources::{BoundFramebuffer, Framebuffer, Texture};
use crate::{Error, RenderContext};

use glow::HasContext;
use piet::kurbo::{Affine, Shape};
use std::rc::Rc;

pub(super) struct Mask<H: HasContext + ?Sized> {
    /// The frame buffer for drawing to the mask.
    framebuffer: Framebuffer<H>,

    /// The texture for this mask.
    texture: Texture<H>,

    /// The transform for this mask.
    transform: Affine,

    /// The width and height of this mask.
    size: (u32, u32),

    /// Is this mask empty?
    empty: bool,
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

        Ok(Self {
            texture,
            framebuffer,
            transform,
            size: (width, height),
            empty: true,
        })
    }

    /// Set the transform of this mask.
    pub(crate) fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Clear out the mask.
    pub(crate) fn clear(&mut self) {
        let mut bound = self.texture.bind(None);
        bound.fill_with_nothing(self.size.0 as _, self.size.1 as _);
        self.empty = true;
    }

    /// Resize this mask, clearing it in the process.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.size = (width, height);
        self.clear();
    }

    /// Get the size of this mask.
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
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

impl<H: HasContext + ?Sized> RenderContext<'_, H> {
    /// Draw to a mask.
    pub(super) fn draw_to_mask(&mut self, mask: &mut Mask<H>, shape: impl Shape) {
        // Bind the framebuffer so we can draw to it.
        let context = self.gl.context.clone();
        let _guard = match bind_framebuffer(&*context, &mask.framebuffer, &mask.texture, self.size)
        {
            Ok(guard) => guard,
            Err(e) => {
                self.last_error = Err(e);
                return;
            }
        };

        unsafe {
            self.gl
                .context
                .viewport(0, 0, mask.size.0 as _, mask.size.1 as _)
        }

        // Draw to the mask.
        let brush_mask = if mask.empty {
            None
        } else {
            Some(mask.as_brush_mask())
        };

        self.fill_impl(
            shape,
            None,
            lyon_tessellation::FillRule::NonZero,
            brush_mask.as_ref(),
        );

        mask.empty = false;
    }
}

struct BoundMask<'a, H: HasContext + ?Sized> {
    context: &'a H,
    bound: Option<BoundFramebuffer<'a, H>>,
    old_size: (u32, u32),
}

impl<H: HasContext + ?Sized> Drop for BoundMask<'_, H> {
    fn drop(&mut self) {
        // Unbind the framebuffer.
        drop(self.bound.take());

        // Restore the viewport.
        unsafe {
            self.context
                .viewport(0, 0, self.old_size.0 as _, self.old_size.1 as _)
        }
    }
}

fn bind_framebuffer<'a, H: HasContext + ?Sized>(
    context: &'a H,
    framebuffer: &'a Framebuffer<H>,
    texture: &Texture<H>,
    old_size: (u32, u32),
) -> Result<BoundMask<'a, H>, Error> {
    let mut bound = framebuffer.bind();

    // Bind the texture as the first attachment.
    bound.bind_color0(texture);

    // Check for errors.
    bound.check_error()?;

    // Keep it bound and unbind it when we're done.
    Ok(BoundMask {
        bound: Some(bound),
        old_size,
        context,
    })
}

struct CallOnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for CallOnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}
