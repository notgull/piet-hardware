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

//! A GPU-accelerated 2D graphics backend for [`piet`] that uses the [`wgpu`] crate.
//!
//! [`piet`]: https://crates.io/crates/piet
//! [`wgpu`]: https://crates.io/crates/wgpu

use std::convert::Infallible;
use std::mem;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::Arc;

use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("piet.wgsl");

/// A reference to a [`wgpu`] [`Device`] and [`Queue`].
///
/// [`wgpu`]: https://crates.io/crates/wgpu
/// [`Device`]: https://docs.rs/wgpu/latest/wgpu/struct.Device.html
/// [`Queue`]: https://docs.rs/wgpu/latest/wgpu/struct.Queue.html
pub trait DeviceAndQueue {
    /// Returns a reference to the [`Device`].
    fn device(&self) -> &wgpu::Device;

    /// Returns a reference to the [`Queue`].
    fn queue(&self) -> &wgpu::Queue;
}

impl<A: AsRef<wgpu::Device>, B: AsRef<wgpu::Queue>> DeviceAndQueue for (A, B) {
    fn device(&self) -> &wgpu::Device {
        self.0.as_ref()
    }

    fn queue(&self) -> &wgpu::Queue {
        self.1.as_ref()
    }
}

/// A wrapper around a `wgpu` context.
struct GpuContext<DaQ: ?Sized> {
    /// The rendering pipeline.
    pipeline: wgpu::RenderPipeline,

    /// The uniform buffer.
    uniform_buffer: wgpu::Buffer,

    /// The `wgpu` device and queue.
    device_and_queue: DaQ,
}

struct WgpuTexture {}

struct WgpuVertexBuffer {}

/// Type of the data stored in the uniform buffer.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct Uniforms {
    /// 3x3 transformation matrix.
    transform: [[f32; 3]; 3],

    /// Viewport size.
    viewport_size: [f32; 2],
}

impl<DaQ: DeviceAndQueue + ?Sized> GpuContext<DaQ> {
    pub fn new(
        device_and_queue: DaQ,
        output_color_format: wgpu::TextureFormat,
        output_depth_format: Option<wgpu::TextureFormat>,
    ) -> Self
    where
        DaQ: Sized,
    {
        // Create the shader module.
        let device = device_and_queue.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("piet-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Create a buffer for the uniforms.
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("piet-wgpu uniform buffer"),
            contents: bytemuck::cast_slice(&[Uniforms {
                transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                viewport_size: [0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_buffer_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("piet-wgpu uniform buffer layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(mem::size_of::<Uniforms>() as u64),
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("piet-wgpu uniform bind group"),
            layout: &uniform_buffer_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // Add texture bindings for the texture and the mask.
        let texture_buffer_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("piet-wgpu texture buffer layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Use these two to create the pipline layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("piet-wgpu pipeline layout"),
            bind_group_layouts: &[&uniform_buffer_layout, &texture_buffer_layout],
            push_constant_ranges: &[],
        });

        let depth_stencil = output_depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Create the pipeline.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("piet-wgpu pipeline"),
            layout: Some(&pipeline_layout),
            ..todo!()
        });

        todo!()
    }
}

impl<DaQ: DeviceAndQueue + ?Sized> piet_hardware::GpuContext for GpuContext<DaQ> {
    type Texture = WgpuTexture;
    type VertexBuffer = WgpuVertexBuffer;
    type Error = Infallible;

    fn clear(&self, color: piet_hardware::piet::Color) {
        todo!()
    }

    fn flush(&self) -> Result<(), Self::Error> {
        todo!()
    }

    fn create_texture(
        &self,
        interpolation: piet_hardware::piet::InterpolationMode,
        repeat: piet_hardware::RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        todo!()
    }

    fn delete_texture(&self, texture: Self::Texture) {
        todo!()
    }

    fn write_texture(
        &self,
        texture: &Self::Texture,
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: Option<&[u8]>,
    ) {
        todo!()
    }

    fn write_subtexture(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: &[u8],
    ) {
        todo!()
    }

    fn set_texture_interpolation(
        &self,
        texture: &Self::Texture,
        interpolation: piet_hardware::piet::InterpolationMode,
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
        vertices: &[piet_hardware::Vertex],
        indices: &[u32],
    ) {
        todo!()
    }

    fn push_buffers(
        &self,
        vertex_buffer: &Self::VertexBuffer,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &piet_hardware::piet::kurbo::Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error> {
        todo!()
    }
}
