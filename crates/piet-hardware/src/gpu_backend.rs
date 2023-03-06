// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/> or
// <https://www.mozilla.org/en-US/MPL/2.0/>.

//! Defines the GPU backend for piet-hardware.

use piet::kurbo::Affine;
use piet::InterpolationMode;

use std::error::Error;

/// The backend for the GPU renderer.
pub trait GpuContext {
    /// The type associated with a GPU texture.
    type Texture;

    /// The type associated with a GPU vertex buffer.
    ///
    /// Contains vertices, indices and any layout data.
    type VertexBuffer;

    /// The error type associated with this GPU context.
    type Error: Error + 'static;

    /// Clear the screen with the given color.
    fn clear(&self, color: piet::Color);

    /// Flush the GPU commands.
    fn flush(&self) -> Result<(), Self::Error>;

    /// Create a new texture.
    fn create_texture(
        &self,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error>;

    /// Delete a texture.
    fn delete_texture(&self, texture: Self::Texture);

    /// Write an image to a texture.
    fn write_texture(
        &self,
        texture: &Self::Texture,
        size: (u32, u32),
        format: piet::ImageFormat,
        data: Option<&[u8]>,
    );

    /// Write a sub-image to a texture.
    fn write_subtexture(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet::ImageFormat,
        data: &[u8],
    );

    /// Set the interpolation mode for a texture.
    fn set_texture_interpolation(&self, texture: &Self::Texture, interpolation: InterpolationMode);

    /// Get the maximum texture size.
    fn max_texture_size(&self) -> (u32, u32);

    /// Create a new vertex buffer.
    fn create_vertex_buffer(&self) -> Result<Self::VertexBuffer, Self::Error>;

    /// Delete a vertex buffer.
    fn delete_vertex_buffer(&self, buffer: Self::VertexBuffer);

    /// Write vertices to a vertex buffer.
    ///
    /// # Safety
    ///
    /// The indices must be valid for the given vertices.
    unsafe fn write_vertices(
        &self,
        buffer: &Self::VertexBuffer,
        vertices: &[Vertex],
        indices: &[u32],
    );

    /// Push buffer data to the GPU.
    fn push_buffers(
        &self,
        vertex_buffer: &Self::VertexBuffer,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error>;
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

/// The format to be provided to the vertex array.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub struct VertexFormat {
    /// The data type associated with the position.
    pub data_type: DataType,

    /// The data format associated with the position.
    pub format: DataFormat,

    /// The number of components in the position.
    pub num_components: u32,

    /// The offset of the position in the vertex.
    pub offset: u32,

    /// The stride of the vertex.
    pub stride: u32,
}

/// The data format associated with a vertex array.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum DataFormat {
    /// Uses floats.
    Float,

    /// Uses unsigned bytes.
    UnsignedByte,
}

/// The type of the data component.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum DataType {
    /// This represents the location of the component, in screen space.
    Position,

    /// This represents the location of the component, in texture space (0..1).
    Texture,

    /// This represents the color of the component.
    Color,
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
