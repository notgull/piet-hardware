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

//! Defines useful resource wrappers.

use super::gpu_backend::{GpuContext, RepeatStrategy, Vertex};

use std::fmt;

use piet::kurbo::{Size, Vec2};
use piet::{
    Error as Pierror, FixedLinearGradient, FixedRadialGradient, GradientStop, InterpolationMode,
};
use tiny_skia::{Paint, Pixmap, Shader};

macro_rules! define_resource_wrappers {
    ($($name:ident($res:ident)),* $(,)?) => {
        $(
            pub(crate) struct $name<C: GpuContext + ?Sized> {
                resource: C::$res,
            }

            impl<C: GpuContext + ?Sized> fmt::Debug for $name<C> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.debug_struct(stringify!($name))
                        .finish_non_exhaustive()
                }
            }

            impl<C: GpuContext + ?Sized> $name<C> {
                pub(crate) fn from_raw(resource: C::$res) -> Self {
                    Self { resource }
                }

                pub(crate) fn resource(&self) -> &C::$res {
                    &self.resource
                }
            }
        )*
    };
}

define_resource_wrappers! {
    Texture(Texture),
    VertexBuffer(VertexBuffer),
}

impl<C: GpuContext + ?Sized> Texture<C> {
    pub(crate) fn new(
        context: &mut C,
        device: &C::Device,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Result<Self, C::Error> {
        let resource = context.create_texture(device, interpolation, repeat)?;

        Ok(Self::from_raw(resource))
    }

    pub(crate) fn write_linear_gradient(
        &self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        gradient: &FixedLinearGradient,
        size: Size,
        offset: Vec2,
    ) -> Result<(), Pierror> {
        let shader = tiny_skia::LinearGradient::new(
            convert_to_ts_point(gradient.start),
            convert_to_ts_point(gradient.end),
            gradient
                .stops
                .iter()
                .map(convert_to_ts_gradient_stop)
                .collect(),
            tiny_skia::SpreadMode::Pad,
            tiny_skia::Transform::from_translate(offset.x as f32, offset.y as f32),
        )
        .ok_or_else(|| Pierror::BackendError("Invalid error".into()))?;

        self.write_shader(context, device, queue, shader, size);

        Ok(())
    }

    pub(crate) fn write_radial_gradient(
        &self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        gradient: &FixedRadialGradient,
        size: Size,
        offset: Vec2,
    ) -> Result<(), Pierror> {
        let shader = tiny_skia::RadialGradient::new(
            convert_to_ts_point(gradient.center + gradient.origin_offset),
            convert_to_ts_point(gradient.center),
            gradient.radius as f32,
            gradient
                .stops
                .iter()
                .map(convert_to_ts_gradient_stop)
                .collect(),
            tiny_skia::SpreadMode::Pad,
            tiny_skia::Transform::from_translate(offset.x as f32, offset.y as f32),
        )
        .ok_or_else(|| Pierror::BackendError("Invalid error".into()))?;

        self.write_shader(context, device, queue, shader, size);

        Ok(())
    }

    pub(crate) fn write_shader(
        &self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        shader: Shader<'_>,
        mut size: Size,
    ) {
        // Pad the size out to at least one.
        if (size.width as isize) < 1 {
            size.width = 1.0;
        }
        if (size.height as isize) < 1 {
            size.height = 1.0;
        }

        // Create a pixmap to render the shader into.
        let mut pixmap =
            Pixmap::new(size.width as _, size.height as _).expect("failed to create pixmap");

        // Render the shader into the pixmap.
        let paint = Paint {
            shader,
            ..Default::default()
        };
        pixmap.fill_rect(
            tiny_skia::Rect::from_xywh(0.0, 0.0, size.width as _, size.height as _).unwrap(),
            &paint,
            tiny_skia::Transform::identity(),
            None,
        );

        // Write the pixmap into the texture.
        let data = pixmap.take();
        self.write_texture(
            context,
            device,
            queue,
            (size.width as _, size.height as _),
            piet::ImageFormat::RgbaPremul,
            Some(&data),
        );
        self.set_interpolation(context, device, InterpolationMode::Bilinear);
    }

    pub(crate) fn write_texture(
        &self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        size: (u32, u32),
        format: piet::ImageFormat,
        data: Option<&[u8]>,
    ) {
        context.write_texture(crate::gpu_backend::TextureWrite {
            device,
            queue,
            size,
            format,
            data,
            texture: &self.resource,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_subtexture(
        &self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet::ImageFormat,
        data: &[u8],
    ) {
        context.write_subtexture(crate::gpu_backend::SubtextureWrite {
            device,
            queue,
            offset,
            size,
            format,
            data,
            texture: &self.resource,
        });
    }

    pub(crate) fn set_interpolation(
        &self,
        context: &mut C,
        device: &C::Device,
        interpolation: InterpolationMode,
    ) {
        context.set_texture_interpolation(device, self.resource(), interpolation);
    }
}

impl<C: GpuContext + ?Sized> VertexBuffer<C> {
    pub(crate) fn new(context: &mut C, device: &C::Device) -> Result<Self, C::Error> {
        let resource = context.create_vertex_buffer(device)?;
        Ok(Self::from_raw(resource))
    }

    pub(crate) fn upload(
        &self,
        context: &mut C,
        device: &C::Device,
        queue: &C::Queue,
        data: &[Vertex],
        indices: &[u32],
    ) {
        context.write_vertices(device, queue, self.resource(), data, indices)
    }
}

pub(crate) fn convert_to_ts_point(point: piet::kurbo::Point) -> tiny_skia::Point {
    tiny_skia::Point {
        x: point.x as f32,
        y: point.y as f32,
    }
}

pub(crate) fn convert_to_ts_color(color: piet::Color) -> tiny_skia::Color {
    let (r, g, b, a) = color.as_rgba();

    tiny_skia::Color::from_rgba(r as f32, g as f32, b as f32, a as f32).expect("Invalid color")
}

pub(crate) fn convert_to_ts_gradient_stop(grad_stop: &GradientStop) -> tiny_skia::GradientStop {
    tiny_skia::GradientStop::new(grad_stop.pos, convert_to_ts_color(grad_stop.color))
}
