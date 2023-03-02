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

//! A GPU-accelerated backend for piet that uses the [`glow`] crate.
//!
//! [`glow`]: https://crates.io/crates/glow

use glow::HasContext;

use piet_gpu::piet::{self, kurbo};
use piet_gpu::GpuContext;

use std::fmt;

macro_rules! c {
    ($e:expr) => {{
        ($e) as f32
    }};
}

/// A wrapper around a `glow` context.
struct GlContext<H: HasContext + ?Sized> {
    /// The underlying context.
    context: H,
}

/// A wrapper around a `glow` texture.
struct GlTexture<H: HasContext + ?Sized>(H::Texture);

/// A wrapper around a `glow` vertex buffer.
struct GlVertexBuffer<H: HasContext + ?Sized> {
    /// The underlying vertex buffer.
    vbo: H::Buffer,

    /// The index buffer.
    ebo: H::Buffer,

    /// The vertex array object.
    vao: H::VertexArray,

    /// The number of indices.
    num_indices: usize,
}

#[derive(Debug)]
struct GlError(String);

impl fmt::Display for GlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gl error: {}", self.0)
    }
}

impl std::error::Error for GlError {}

impl<H: HasContext + ?Sized> GpuContext for GlContext<H> {
    type Texture = GlTexture<H>;

    type VertexBuffer = GlVertexBuffer<H>;

    type Error = GlError;

    fn clear(&self, color: piet_gpu::piet::Color) {
        let (r, g, b, a) = color.as_rgba();

        unsafe {
            self.context.clear_color(c!(r), c!(g), c!(b), c!(a));
            self.context.clear(glow::COLOR_BUFFER_BIT);
        }
    }

    fn flush(&self) -> Result<(), Self::Error> {
        unsafe {
            self.context.flush();
        }

        Ok(())
    }

    fn create_texture(
        &self,
        interpolation: piet_gpu::piet::InterpolationMode,
        repeat: piet_gpu::RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        todo!()
    }

    fn delete_texture(&self, texture: Self::Texture) {
        todo!()
    }

    fn write_texture<T: bytemuck::Pod>(
        &self,
        texture: &Self::Texture,
        size: (u32, u32),
        format: piet_gpu::piet::ImageFormat,
        data: Option<&[T]>,
    ) {
        todo!()
    }

    fn write_subtexture<T: bytemuck::Pod>(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet_gpu::piet::ImageFormat,
        data: &[T],
    ) {
        todo!()
    }

    fn set_texture_interpolation(
        &self,
        texture: &Self::Texture,
        interpolation: piet_gpu::piet::InterpolationMode,
    ) {
        todo!()
    }

    fn max_texture_size(&self) -> (u32, u32) {
        todo!()
    }

    fn create_vertex_buffer(&self) -> Result<Self::VertexBuffer, Self::Error> {
        todo!()
    }

    fn delete_vertex_buffer(&self, buffer: Self::VertexBuffer) {
        todo!()
    }

    unsafe fn write_vertices(
        &self,
        buffer: &Self::VertexBuffer,
        vertices: &[piet_gpu::Vertex],
        indices: &[u32],
    ) {
        todo!()
    }

    fn push_buffers(
        &self,
        vertex_buffer: &Self::VertexBuffer,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &piet_gpu::piet::kurbo::Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error> {
        todo!()
    }
}
