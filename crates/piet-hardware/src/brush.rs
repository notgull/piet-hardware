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

//! The brush types used by `piet-hardware`.

use super::gpu_backend::{GpuContext, RepeatStrategy, Vertex};
use super::image::Image;
use super::resources::Texture;
use super::{RenderContext, ResultExt, UV_WHITE};

use piet::kurbo::{Affine, Circle, Point, Rect, Shape};
use piet::{Error as Pierror, FixedLinearGradient, FixedRadialGradient};

use std::borrow::Cow;

/// The brush type used by the GPU renderer.
#[derive(Debug)]
pub struct Brush<C: GpuContext + ?Sized>(BrushInner<C>);

impl<C: GpuContext + ?Sized> Clone for Brush<C> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(Debug)]
enum BrushInner<C: GpuContext + ?Sized> {
    /// A solid color.
    Solid(piet::Color),

    /// A texture to apply.
    Texture {
        /// The image to apply.
        image: Image<C>,

        /// The transformation to translate UV texture points by.
        transform: Affine,
    },
}

impl<C: GpuContext + ?Sized> piet::IntoBrush<RenderContext<'_, '_, '_, C>> for Brush<C> {
    fn make_brush<'a>(
        &'a self,
        _piet: &mut RenderContext<'_, '_, '_, C>,
        _bbox: impl FnOnce() -> Rect,
    ) -> Cow<'a, <RenderContext<'_, '_, '_, C> as piet::RenderContext>::Brush> {
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
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        gradient: FixedLinearGradient,
    ) -> Result<Self, Pierror> {
        let texture = Texture::new(
            context,
            device,
            piet::InterpolationMode::Bilinear,
            RepeatStrategy::Clamp,
        )
        .piet_err()?;

        let (gradient, transform) = straighten_gradient(gradient);
        let bounds = Rect::from_points(gradient.start, gradient.end);
        let offset = -bounds.origin().to_vec2();
        texture.write_linear_gradient(context, device, queue, &gradient, bounds.size(), offset)?;
        Ok(Self::textured(texture, bounds.size(), transform))
    }

    /// Create a new brush from a radial gradient.
    pub(crate) fn radial_gradient(
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        gradient: FixedRadialGradient,
    ) -> Result<Self, Pierror> {
        let texture = Texture::new(
            context,
            device,
            piet::InterpolationMode::Bilinear,
            RepeatStrategy::Clamp,
        )
        .piet_err()?;

        let bounds = Circle::new(gradient.center, gradient.radius).bounding_box();
        let offset = -bounds.origin().to_vec2();
        let transform = scale_and_offset(bounds.size(), bounds.origin());

        texture.write_radial_gradient(context, device, queue, &gradient, bounds.size(), offset)?;
        Ok(Self::textured(texture, bounds.size(), transform))
    }

    /// Create a new brush from a texture.
    fn textured(texture: Texture<C>, size: kurbo::Size, transform: Affine) -> Self {
        // Create a new image.
        let image = Image::new(texture, size);

        Self(BrushInner::Texture { image, transform })
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

            BrushInner::Texture { transform, .. } => {
                let uv = transform * Point::new(point[0] as f64, point[1] as f64);
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
            Self::Texture { image, transform } => Self::Texture {
                image: image.clone(),
                transform: *transform,
            },
        }
    }
}

/// Convert a gradient into either a horizontal or vertical gradient as well as
/// a rotation that rotates the start/end points to their former positions.
fn straighten_gradient(gradient: FixedLinearGradient) -> (FixedLinearGradient, Affine) {
    // If the gradient is already almost straight, then no need to do anything.
    if (gradient.start.x - gradient.end.x).abs() < 1.0
        || (gradient.start.y - gradient.end.y).abs() < 1.0
    {
        let mut bounds = Rect::from_points(gradient.start, gradient.end);
        if (bounds.width() as isize) < 0 {
            bounds.x1 += 1.0;
        }
        if (bounds.height() as isize) < 0 {
            bounds.y1 += 1.0;
        }

        return (gradient, scale_and_offset(bounds.size(), bounds.origin()));
    }

    // Get the angle and length between the start and end points.
    let (angle, length) = {
        let delta = gradient.end - gradient.start;
        (delta.angle(), delta.hypot())
    };

    // Create a horizontal line starting at the start point with the same length
    // as the original gradient.
    let horizontal_end = gradient.start + kurbo::Vec2::new(length, 0.0);

    // Use that to create a new gradient.
    let new_gradient = FixedLinearGradient {
        start: gradient.start,
        end: horizontal_end,
        stops: gradient.stops,
    };

    // A transform that maps UV coordinates into this plane.
    let offset = gradient.start.to_vec2();
    let new_bounds = Rect::from_points(new_gradient.start, new_gradient.end);
    let transform = scale_and_offset(new_bounds.size(), new_bounds.origin())
        * Affine::translate(offset)
        * Affine::rotate(-angle)
        * Affine::translate(-offset);
    (new_gradient, transform)
}

fn scale_and_offset(size: kurbo::Size, offset: kurbo::Point) -> Affine {
    Affine::scale_non_uniform(1.0 / size.width, 1.0 / size.height)
        * Affine::translate(-offset.to_vec2())
}
