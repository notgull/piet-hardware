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

//! Defines the GPU backend for piet-hardware.

use piet::kurbo::Affine;
use piet::InterpolationMode;

use std::error::Error;

/// The backend for the GPU renderer.
pub trait GpuContext {
    /// A "device" that can be used to render.
    ///
    /// This corresponds to [`Device`] in [`wgpu`] and nothing in particular in [`glow`].
    ///
    /// [`Device`]: wgpu::Device
    /// [`wgpu`]: https://crates.io/crates/wgpu
    /// [`glow`]: https://crates.io/crates/glow
    type Device;

    /// A "queue" that can be used to render.
    ///
    /// This corresponds to [`Queue`] in [`wgpu`] and nothing in particular in [`glow`].
    ///
    /// [`Queue`]: wgpu::Queue
    /// [`wgpu`]: https://crates.io/crates/wgpu
    /// [`glow`]: https://crates.io/crates/glow
    type Queue;

    /// The type associated with a GPU texture.
    type Texture;

    /// The type associated with a GPU vertex buffer.
    ///
    /// Contains vertices, indices and any layout data.
    type VertexBuffer;

    /// The error type associated with this GPU context.
    type Error: Error + 'static;

    /// Clear the screen with the given color.
    fn clear(&mut self, device: &Self::Device, queue: &Self::Queue, color: piet::Color);

    /// Flush the GPU commands.
    fn flush(&mut self) -> Result<(), Self::Error>;

    /// Create a new texture.
    fn create_texture(
        &mut self,
        device: &Self::Device,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error>;

    /// Write an image to a texture.
    fn write_texture(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        texture: &Self::Texture,
        size: (u32, u32),
        format: piet::ImageFormat,
        data: Option<&[u8]>,
    );

    /// Write a sub-image to a texture.
    #[allow(clippy::too_many_arguments)]
    fn write_subtexture(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet::ImageFormat,
        data: &[u8],
    );

    /// Set the interpolation mode for a texture.
    fn set_texture_interpolation(
        &mut self,
        device: &Self::Device,
        texture: &Self::Texture,
        interpolation: InterpolationMode,
    );

    /// Get the maximum texture size.
    fn max_texture_size(&mut self, device: &Self::Device) -> (u32, u32);

    /// Create a new vertex buffer.
    fn create_vertex_buffer(
        &mut self,
        device: &Self::Device,
    ) -> Result<Self::VertexBuffer, Self::Error>;

    /// Write vertices to a vertex buffer.
    ///
    /// The indices must be valid for the vertices set; however, it is up to the GPU implementation
    /// to actually check this.
    fn write_vertices(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        buffer: &Self::VertexBuffer,
        vertices: &[Vertex],
        indices: &[u32],
    );

    /// Capture an area from the screen and put it into a texture.
    fn capture_area(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
    ) -> Result<(), Self::Error>;

    /// Push buffer data to the GPU.
    #[allow(clippy::too_many_arguments)]
    fn push_buffers(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        vertex_buffer: &Self::VertexBuffer,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error>;
}

impl<C: GpuContext + ?Sized> GpuContext for &mut C {
    type Device = C::Device;
    type Queue = C::Queue;
    type Texture = C::Texture;
    type VertexBuffer = C::VertexBuffer;
    type Error = C::Error;

    fn capture_area(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
    ) -> Result<(), Self::Error> {
        (**self).capture_area(device, queue, texture, offset, size)
    }

    fn clear(&mut self, device: &Self::Device, queue: &Self::Queue, color: piet::Color) {
        (**self).clear(device, queue, color)
    }

    fn create_texture(
        &mut self,
        device: &Self::Device,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        (**self).create_texture(device, interpolation, repeat)
    }

    fn create_vertex_buffer(
        &mut self,
        device: &Self::Device,
    ) -> Result<Self::VertexBuffer, Self::Error> {
        (**self).create_vertex_buffer(device)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        (**self).flush()
    }

    fn max_texture_size(&mut self, device: &Self::Device) -> (u32, u32) {
        (**self).max_texture_size(device)
    }

    fn push_buffers(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        vertex_buffer: &Self::VertexBuffer,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error> {
        (**self).push_buffers(
            device,
            queue,
            vertex_buffer,
            current_texture,
            mask_texture,
            transform,
            size,
        )
    }

    fn set_texture_interpolation(
        &mut self,
        device: &Self::Device,
        texture: &Self::Texture,
        interpolation: InterpolationMode,
    ) {
        (**self).set_texture_interpolation(device, texture, interpolation)
    }

    fn write_subtexture(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet::ImageFormat,
        data: &[u8],
    ) {
        (**self).write_subtexture(device, queue, texture, offset, size, format, data)
    }

    fn write_texture(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        texture: &Self::Texture,
        size: (u32, u32),
        format: piet::ImageFormat,
        data: Option<&[u8]>,
    ) {
        (**self).write_texture(device, queue, texture, size, format, data)
    }

    fn write_vertices(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        buffer: &Self::VertexBuffer,
        vertices: &[Vertex],
        indices: &[u32],
    ) {
        (**self).write_vertices(device, queue, buffer, vertices, indices)
    }
}

/// The strategy to use for repeating.
#[derive(Debug, Copy, Clone, PartialEq)]
#[non_exhaustive]
pub enum RepeatStrategy {
    /// Repeat the image.
    Repeat,

    /// Clamp to the edge of the image.
    Clamp,

    /// Don't repeat and instead use this color.
    Color(piet::Color),
}

/// The vertex type used by the GPU renderer.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vertex {
    /// The position of the vertex.
    pub pos: [f32; 2],

    /// The coordinate of the vertex in the texture.
    pub uv: [f32; 2],

    /// The color of the vertex, in four SRGB channels.
    pub color: [u8; 4],
}

/// The type of the buffer to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferType {
    /// The buffer is used for vertices.
    Vertex,

    /// The buffer is used for indices.
    Index,
}
