// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the
// terms of either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
//   version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License or the Mozilla Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/>.

//! The image type for the GPU renderer.

use super::gpu_backend::GpuContext;
use super::resources::Texture;

use piet::kurbo::Size;

use std::rc::Rc;

/// The image type used by the GPU renderer.
#[derive(Debug)]
pub struct Image<C: GpuContext + ?Sized> {
    /// The texture.
    texture: Rc<Texture<C>>,

    /// The size of the image.
    size: Size,
}

impl<C: GpuContext + ?Sized> Image<C> {
    /// Create a new image from a texture.
    pub(crate) fn new(texture: Texture<C>, size: Size) -> Self {
        Self {
            texture: Rc::new(texture),
            size,
        }
    }

    /// Get the texture.
    pub(crate) fn texture(&self) -> &Texture<C> {
        &self.texture
    }
}

impl<C: GpuContext + ?Sized> Clone for Image<C> {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            size: self.size,
        }
    }
}

impl<C: GpuContext + ?Sized> piet::Image for Image<C> {
    fn size(&self) -> Size {
        self.size
    }
}
