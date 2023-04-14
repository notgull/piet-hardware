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

//! The brush types used by `piet-hardware`.

use super::gpu_backend::{GpuContext, RepeatStrategy, Vertex};
use super::image::Image;
use super::resources::Texture;
use super::{RenderContext, ResultExt, UV_WHITE};

use piet::kurbo::{Affine, Circle, Point, Rect, Shape};
use piet::{Error as Pierror, FixedLinearGradient, FixedRadialGradient, Image as _};

use std::borrow::Cow;
use std::rc::Rc;

/// The brush type used by the GPU renderer.
pub struct Brush<C: GpuContext + ?Sized>(BrushInner<C>);

impl<C: GpuContext + ?Sized> Clone for Brush<C> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

enum BrushInner<C: GpuContext + ?Sized> {
    /// A solid color.
    Solid(piet::Color),

    /// A texture to apply.
    Texture {
        /// The image to apply.
        image: Image<C>,

        /// The position to offset the gradient rectangle by.
        offset: Point,
    },
}

impl<C: GpuContext + ?Sized> piet::IntoBrush<RenderContext<'_, C>> for Brush<C> {
    fn make_brush<'a>(
        &'a self,
        _piet: &mut RenderContext<'_, C>,
        _bbox: impl FnOnce() -> Rect,
    ) -> Cow<'a, <RenderContext<'_, C> as piet::RenderContext>::Brush> {
        Cow::Borrowed(self)
    }
}

impl<C: GpuContext + ?Sized> Brush<C> {
    /// Create a new solid brush.
    pub(crate) fn solid(color: piet::Color) -> Self {
        Self(BrushInner::Solid(color))
    }

    /// Create a new brush from a linear gradient.
    pub(crate) fn linear_gradient(
        context: &Rc<C>,
        gradient: FixedLinearGradient,
    ) -> Result<Self, Pierror> {
        let texture = Texture::new(
            context,
            piet::InterpolationMode::Bilinear,
            RepeatStrategy::Clamp,
        )
        .piet_err()?;

        let bounds = Rect::from_points(gradient.start, gradient.end);
        let offset = -bounds.origin().to_vec2();

        texture.write_linear_gradient(&gradient, bounds.size(), offset)?;
        Ok(Self::textured(texture, bounds))
    }

    /// Create a new brush from a radial gradient.
    pub(crate) fn radial_gradient(
        context: &Rc<C>,
        gradient: FixedRadialGradient,
    ) -> Result<Self, Pierror> {
        let texture = Texture::new(
            context,
            piet::InterpolationMode::Bilinear,
            RepeatStrategy::Clamp,
        )
        .piet_err()?;

        let bounds = Circle::new(gradient.center, gradient.radius).bounding_box();
        let offset = -bounds.origin().to_vec2();

        texture.write_radial_gradient(&gradient, bounds.size(), offset)?;
        Ok(Self::textured(texture, bounds))
    }

    /// Create a new brush from a texture.
    fn textured(texture: Texture<C>, bounds: Rect) -> Self {
        // Create a new image.
        let image = Image::new(texture, bounds.size());

        Self(BrushInner::Texture {
            image,
            offset: bounds.origin(),
        })
    }

    /// Get the texture associated with this brush.
    pub(crate) fn texture(&self, _size: (u32, u32)) -> Option<&Image<C>> {
        match self.0 {
            BrushInner::Solid(_) => None,
            BrushInner::Texture { ref image, .. } => Some(image),
        }
    }

    /// Transform a two-dimensional point into a vertex using this brush.
    pub(crate) fn make_vertex(&self, point: [f32; 2]) -> Vertex {
        match self.0 {
            BrushInner::Solid(color) => Vertex {
                pos: point,
                uv: UV_WHITE,
                color: {
                    let (r, g, b, a) = color.as_rgba8();
                    [r, g, b, a]
                },
            },

            BrushInner::Texture { ref image, offset } => {
                // Create a transform to convert from image coordinates to
                // UV coordinates.
                let uv_transform =
                    Affine::scale_non_uniform(1.0 / image.size().width, 1.0 / image.size().height)
                        * Affine::translate(-offset.to_vec2());
                let uv = uv_transform * Point::new(point[0] as f64, point[1] as f64);
                Vertex {
                    pos: point,
                    uv: [uv.x as f32, uv.y as f32],
                    color: [0xFF, 0xFF, 0xFF, 0xFF],
                }
            }
        }
    }
}

impl<C: GpuContext + ?Sized> Clone for BrushInner<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Solid(color) => Self::Solid(*color),
            Self::Texture { image, offset } => Self::Texture {
                image: image.clone(),
                offset: *offset,
            },
        }
    }
}
