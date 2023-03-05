// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-gpu`.
//
// `piet-gpu` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-gpu` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-gpu`. If not, see <https://www.gnu.org/licenses/> or
// <https://www.mozilla.org/en-US/MPL/2.0/>.

//! The brush types used by `piet-gpu`.

use super::gpu_backend::{GpuContext, RepeatStrategy, Vertex};
use super::image::Image;
use super::resources::Texture;
use super::{RenderContext, ResultExt, UV_WHITE};

use piet::kurbo::{Affine, Point, Rect, Size};
use piet::{Error as Pierror, FixedLinearGradient, FixedRadialGradient, Image as _};

use std::borrow::Cow;
use std::cell::{RefCell, RefMut};
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
        image: RefCell<Image<C>>,

        /// The source of the image.
        ///
        /// Needed for updates when the window size changes.
        source: ImageSource,
    },
}

#[derive(Clone, Debug)]
enum ImageSource {
    /// The source of the image is a linear gradient.
    LinearGradient(FixedLinearGradient),

    /// The source of the image is a radial gradient.
    RadialGradient(FixedRadialGradient),
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
        size: (u32, u32),
    ) -> Result<Self, Pierror> {
        let texture = Texture::new(
            context,
            piet::InterpolationMode::Bilinear,
            RepeatStrategy::Repeat,
        )
        .piet_err()?;

        texture.write_linear_gradient(&gradient, size)?;
        Ok(Self::textured(
            texture,
            ImageSource::LinearGradient(gradient),
            size,
        ))
    }

    /// Create a new brush from a radial gradient.
    pub(crate) fn radial_gradient(
        context: &Rc<C>,
        gradient: FixedRadialGradient,
        size: (u32, u32),
    ) -> Result<Self, Pierror> {
        let texture = Texture::new(
            context,
            piet::InterpolationMode::Bilinear,
            RepeatStrategy::Repeat,
        )
        .piet_err()?;

        texture.write_radial_gradient(&gradient, size)?;
        Ok(Self::textured(
            texture,
            ImageSource::RadialGradient(gradient),
            size,
        ))
    }

    /// Create a new brush from a texture.
    fn textured(texture: Texture<C>, source: ImageSource, (width, height): (u32, u32)) -> Self {
        let image = Image::new(texture, Size::new(width as f64, height as f64));

        Self(BrushInner::Texture {
            image: RefCell::new(image),
            source,
        })
    }

    /// Get the texture associated with this brush.
    pub(crate) fn texture(
        &self,
        size: (u32, u32),
    ) -> Result<Option<RefMut<'_, Image<C>>>, Pierror> {
        match self.0 {
            BrushInner::Solid(_) => Ok(None),
            BrushInner::Texture {
                ref image,
                ref source,
            } => {
                let mut image = image.borrow_mut();

                // Update the image if necessary.
                let image_size = (image.size().width as u32, image.size().height as u32);
                if image_size != size {
                    match source {
                        ImageSource::LinearGradient(ref linear) => {
                            image.texture().write_linear_gradient(linear, size)?;
                        }

                        ImageSource::RadialGradient(ref radial) => {
                            image.texture().write_radial_gradient(radial, size)?;
                        }
                    }

                    image.set_size(Size::new(size.0 as f64, size.1 as f64));
                }

                Ok(Some(image))
            }
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

            BrushInner::Texture { ref image, .. } => {
                // Create a transform to convert from image coordinates to
                // UV coordinates.
                let image = image.borrow();
                let uv_transform =
                    Affine::scale_non_uniform(1.0 / image.size().width, 1.0 / image.size().height);
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
            Self::Texture { image, source } => Self::Texture {
                image: RefCell::new(image.borrow().clone()),
                source: source.clone(),
            },
        }
    }
}
