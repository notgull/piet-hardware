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

//! Defines useful resource wrappers.

use super::gpu_backend::{GpuContext, RepeatStrategy, Vertex};

use piet::kurbo::{Size, Vec2};
use piet::{
    Error as Pierror, FixedLinearGradient, FixedRadialGradient, GradientStop, InterpolationMode,
};
use tiny_skia::{Paint, Pixmap, Shader};

use std::rc::Rc;

macro_rules! define_resource_wrappers {
    ($($name:ident($res:ident => $delete:ident)),* $(,)?) => {
        $(
            pub(crate) struct $name<C: GpuContext + ?Sized> {
                context: Rc<C>,
                resource: Option<C::$res>,
            }

            impl<C: GpuContext + ?Sized> Drop for $name<C> {
                fn drop(&mut self) {
                    if let Some(resource) = self.resource.take() {
                        self.context.$delete(resource);
                    }
                }
            }

            impl<C: GpuContext + ?Sized> $name<C> {
                pub(crate) fn from_raw(
                    context: &Rc<C>,
                    resource: C::$res,
                ) -> Self {
                    Self {
                        context: context.clone(),
                        resource: Some(resource),
                    }
                }

                pub(crate) fn resource(&self) -> &C::$res {
                    self.resource.as_ref().unwrap()
                }
            }
        )*
    };
}

define_resource_wrappers! {
    Texture(Texture => delete_texture),
    VertexBuffer(VertexBuffer => delete_vertex_buffer),
}

impl<C: GpuContext + ?Sized> Texture<C> {
    pub(crate) fn new(
        context: &Rc<C>,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Result<Self, C::Error> {
        let resource = context.create_texture(interpolation, repeat)?;

        Ok(Self::from_raw(context, resource))
    }

    pub(crate) fn write_linear_gradient(
        &self,
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

        self.write_shader(shader, size);

        Ok(())
    }

    pub(crate) fn write_radial_gradient(
        &self,
        gradient: &FixedRadialGradient,
        size: Size,
        offset: Vec2,
    ) -> Result<(), Pierror> {
        let shader = tiny_skia::RadialGradient::new(
            convert_to_ts_point(gradient.center),
            convert_to_ts_point(gradient.center - gradient.origin_offset),
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

        self.write_shader(shader, size);

        Ok(())
    }

    pub(crate) fn write_shader(&self, shader: Shader<'_>, size: Size) {
        // Create a pixmap to render the shader into.
        if approx_eq(size.width as f32, 0.0) || approx_eq(size.height as f32, 0.0) {
            return;
        }

        let mut pixmap =
            Pixmap::new(size.width as _, size.height as _).expect("failed to create pixmap");

        // Render the shader into the pixmap.
        let paint = Paint {
            shader,
            ..Default::default()
        };
        pixmap
            .fill_rect(
                tiny_skia::Rect::from_xywh(0.0, 0.0, size.width as _, size.height as _).unwrap(),
                &paint,
                tiny_skia::Transform::identity(),
                None,
            )
            .expect("failed to render shader");

        // Write the pixmap into the texture.
        let data = pixmap.take();
        self.write_texture(
            (size.width as _, size.height as _),
            piet::ImageFormat::RgbaPremul,
            Some(&data),
        );
        self.set_interpolation(InterpolationMode::Bilinear);
    }

    pub(crate) fn write_texture(
        &self,
        size: (u32, u32),
        format: piet::ImageFormat,
        data: Option<&[u8]>,
    ) {
        self.context
            .write_texture(self.resource(), size, format, data);
    }

    pub(crate) fn write_subtexture(
        &self,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet::ImageFormat,
        data: &[u8],
    ) {
        self.context
            .write_subtexture(self.resource(), offset, size, format, data);
    }

    pub(crate) fn set_interpolation(&self, interpolation: InterpolationMode) {
        self.context
            .set_texture_interpolation(self.resource(), interpolation);
    }
}

impl<C: GpuContext + ?Sized> VertexBuffer<C> {
    pub(crate) fn new(context: &Rc<C>) -> Result<Self, C::Error> {
        let resource = context.create_vertex_buffer()?;
        Ok(Self::from_raw(context, resource))
    }

    pub(crate) fn upload(&self, data: &[Vertex], indices: &[u32]) {
        self.context.write_vertices(self.resource(), data, indices)
    }
}

fn convert_to_ts_point(point: piet::kurbo::Point) -> tiny_skia::Point {
    tiny_skia::Point {
        x: point.x as f32,
        y: point.y as f32,
    }
}

fn convert_to_ts_color(color: piet::Color) -> tiny_skia::Color {
    let (r, g, b, a) = color.as_rgba();

    tiny_skia::Color::from_rgba(r as f32, g as f32, b as f32, a as f32).expect("Invalid color")
}

fn convert_to_ts_gradient_stop(grad_stop: &GradientStop) -> tiny_skia::GradientStop {
    tiny_skia::GradientStop::new(grad_stop.pos, convert_to_ts_color(grad_stop.color))
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.0001
}
