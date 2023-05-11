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

//! The underlying GPU context.

use super::buffer::{Buffer, BufferSlice, WgpuVertexBuffer};
use super::texture::{BorrowedTexture, WgpuTexture};
use super::DeviceAndQueue;

use std::cell::{Cell, Ref, RefCell};
use std::collections::hash_map::{Entry, HashMap};
use std::convert::Infallible;
use std::mem;
use std::num::NonZeroU64;
use std::rc::Rc;

use piet_hardware::piet::kurbo::Affine;
use piet_hardware::piet::{Color, InterpolationMode};
use piet_hardware::Vertex;

use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("piet.wgsl");

/// A wrapper around a `wgpu` context.
pub(crate) struct GpuContext<DaQ: ?Sized> {
    /// The rendering pipeline.
    pipeline: wgpu::RenderPipeline,

    /// The bind group layout for uniforms.
    uniform_bind_layout: wgpu::BindGroupLayout,

    /// Bind group for textures.
    texture_bind_layout: wgpu::BindGroupLayout,

    /// The existing uniform buffers.
    uniform_buffers: RefCell<HashMap<UniformBytes, (wgpu::Buffer, Rc<wgpu::BindGroup>)>>,

    /// The clearing color.
    clear_color: Cell<Option<Color>>,

    /// The view of the texture.
    texture_view: RefCell<Option<wgpu::TextureView>>,

    /// Latest buffer pushes.
    pushed_buffers: RefCell<Vec<PushedBuffer>>,

    /// Unique IDs for textures and buffers.
    next_id: Cell<usize>,

    /// The `wgpu` device and queue.
    device_and_queue: DaQ,
}

/// Represents a pushed buffer call.
struct PushedBuffer {
    /// The vertex and index buffers.
    buffers: WgpuVertexBuffer,

    /// The slice into the vertex buffer.
    vertex: BufferSlice,

    /// The slice into the index buffer.
    index: BufferSlice,

    /// The color texture to use.
    color_texture: WgpuTexture,

    /// The mask texture to use.
    mask_texture: WgpuTexture,

    /// The viewport size.
    viewport_size: [f32; 2],

    /// The bind group for uniforms.
    uniform_bind_group: Rc<wgpu::BindGroup>,
}

/// A borrowed pushed buffer.
struct BorrowedPush<'a> {
    /// The original pushed buffer.
    source: &'a PushedBuffer,

    /// Borrowed vertex buffer.
    vb: Ref<'a, Buffer>,

    /// Borrowed index buffer.
    ib: Ref<'a, Buffer>,

    /// Borrowed color texture.
    color_texture: BorrowedTexture<'a>,

    /// Borrowed mask texture.
    mask_texture: BorrowedTexture<'a>,

    /// The uniform buffer.
    uniform_bind_group: Rc<wgpu::BindGroup>,
}

impl PushedBuffer {
    fn borrow(&self) -> BorrowedPush<'_> {
        BorrowedPush {
            source: self,
            vb: self.buffers.borrow_vertex_buffer(),
            ib: self.buffers.borrow_index_buffer(),
            color_texture: self.color_texture.borrow(),
            mask_texture: self.mask_texture.borrow(),
            uniform_bind_group: self.uniform_bind_group.clone(),
        }
    }
}

/// Type of the data stored in the uniform buffer.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct Uniforms {
    /// Viewport size.
    viewport_size: [f32; 2],

    /// Padding.
    pad: [u32; 2],

    /// 3x3 transformation matrix.
    transform: [[f32; 4]; 3],
}

type UniformBytes = [u8; mem::size_of::<Uniforms>()];

impl<DaQ: DeviceAndQueue + ?Sized> GpuContext<DaQ> {
    /// Create a new GPU context.
    pub(crate) fn new(
        device_and_queue: DaQ,
        output_color_format: wgpu::TextureFormat,
        output_depth_format: Option<wgpu::TextureFormat>,
        samples: u32,
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

        // Create a buffer layout for the uniforms.
        let uniform_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("piet-wgpu uniform buffer layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            mem::size_of::<Uniforms>().next_power_of_two() as u64,
                        ),
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
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
                ],
            });

        // Use these two to create the pipline layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("piet-wgpu pipeline layout"),
            bind_group_layouts: &[
                &uniform_bind_layout,
                &texture_buffer_layout,
                &texture_buffer_layout,
            ],
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
            vertex: wgpu::VertexState {
                entry_point: "vertex_main",
                module: &shader,
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        // pos: [f32; 2]
                        0 => Float32x2,
                        // uv: [f32; 2]
                        1 => Float32x2,
                        // color: [u8; 4]
                        2 => Uint32,
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                unclipped_depth: false,
                conservative: false,
                cull_mode: None,
                front_face: wgpu::FrontFace::default(),
                polygon_mode: wgpu::PolygonMode::default(),
                strip_index_format: None,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                alpha_to_coverage_enabled: false,
                count: samples,
                mask: !0,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fragment_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::DstAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self {
            device_and_queue,
            pipeline,
            uniform_bind_layout,
            texture_bind_layout: texture_buffer_layout,
            uniform_buffers: RefCell::new(HashMap::new()),
            clear_color: Cell::new(None),
            texture_view: RefCell::new(None),
            pushed_buffers: RefCell::new(Vec::new()),
            next_id: Cell::new(0),
        }
    }

    /// Get the device and queue.
    pub(crate) fn device_and_queue(&self) -> &DaQ {
        &self.device_and_queue
    }

    /// Set the texture view that this GPU context renders to.
    pub(crate) fn set_texture_view(&self, view: wgpu::TextureView) {
        *self.texture_view.borrow_mut() = Some(view);
    }

    pub(crate) fn texture_bind_layout(&self) -> &wgpu::BindGroupLayout {
        &self.texture_bind_layout
    }

    pub(crate) fn next_id(&self) -> usize {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        id
    }
}

impl<DaQ: DeviceAndQueue + ?Sized> piet_hardware::GpuContext for GpuContext<DaQ> {
    type Texture = WgpuTexture;
    type VertexBuffer = WgpuVertexBuffer;
    type Error = Infallible;

    fn clear(&self, color: piet_hardware::piet::Color) {
        // Set the inner clear color.
        self.clear_color.set(Some(color));

        // This clear will remove all of the currently pushed buffers, delete them if they exist.
        for PushedBuffer { buffers, .. } in self.pushed_buffers.borrow_mut().drain(..) {
            buffers
                .borrow_vertex_buffer_mut()
                .clear(self.device_and_queue().device());
            buffers
                .borrow_index_buffer_mut()
                .clear(self.device_and_queue().device());
        }
    }

    fn flush(&self) -> Result<(), Self::Error> {
        let mut encoder = self.device_and_queue.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("piet-wgpu command encoder"),
            },
        );

        let buffer_pushes = mem::take(&mut *self.pushed_buffers.borrow_mut());
        let pushes = buffer_pushes.iter().map(|x| x.borrow()).collect::<Vec<_>>();
        let mut buffers_to_clear = Vec::with_capacity(1);
        let texture_view = self.texture_view.borrow();

        // Create a render pass.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("piet-wgpu render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view.as_ref().expect("no texture view"),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: match self.clear_color.take() {
                        None => wgpu::LoadOp::Load,
                        Some(clr) => wgpu::LoadOp::Clear({
                            let (r, g, b, a) = clr.as_rgba();
                            wgpu::Color { r, g, b, a }
                        }),
                    },
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        // Set the pipeline.
        pass.set_pipeline(&self.pipeline);

        // Iterate over the pushed buffers.
        for BorrowedPush {
            source:
                PushedBuffer {
                    buffers,
                    vertex: vertex_slice,
                    index: index_slice,
                    viewport_size: [width, height],
                    ..
                },
            vb,
            ib,
            color_texture,
            mask_texture,
            uniform_bind_group,
        } in &pushes
        {
            // Set a viewport.
            pass.set_viewport(0.0, 0.0, *width, *height, 0.0, 1.0);

            // Set the uniforms.
            pass.set_bind_group(0, uniform_bind_group, &[]);

            // Bind textures.
            pass.set_bind_group(1, color_texture.bind_group(), &[]);
            pass.set_bind_group(2, mask_texture.bind_group(), &[]);

            // Get the bufer slices to pass to the shader.
            let num_indices = index_slice.len() / mem::size_of::<u32>();
            let vertex_slice = vb.slice(*vertex_slice).unwrap();
            let index_slice = ib.slice(*index_slice).unwrap();

            // Bind the slices into the shader.
            pass.set_index_buffer(index_slice, wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(0, vertex_slice);

            // Draw the triangles.
            pass.draw_indexed(0..num_indices as u32, 0, 0..1);

            // Push the buffer to the clear list.
            if buffers_to_clear.iter().all(|(id, _)| *id != buffers.id()) {
                buffers_to_clear.push((buffers.id(), buffers.clone()));
            }
        }

        // Encode to a buffer and push to the queue.
        drop(pass);
        self.device_and_queue.queue().submit(Some(encoder.finish()));

        // Clear the buffers.
        drop(pushes);
        for (_, buffers) in buffers_to_clear {
            buffers
                .borrow_vertex_buffer_mut()
                .clear(self.device_and_queue.device());
            buffers
                .borrow_index_buffer_mut()
                .clear(self.device_and_queue.device());
        }

        Ok(())
    }

    fn create_texture(
        &self,
        interpolation: InterpolationMode,
        repeat: piet_hardware::RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        Ok(WgpuTexture::create_texture(self, interpolation, repeat))
    }

    fn delete_texture(&self, texture: Self::Texture) {
        // Drop the texture.
        drop(texture);
    }

    fn write_texture(
        &self,
        tex: &Self::Texture,
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: Option<&[u8]>,
    ) {
        tex.borrow_mut().write_texture(self, size, format, data)
    }

    fn write_subtexture(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: &[u8],
    ) {
        texture
            .borrow_mut()
            .write_subtexture(self, offset, size, format, data)
    }

    fn set_texture_interpolation(&self, texture: &Self::Texture, interpolation: InterpolationMode) {
        texture
            .borrow_mut()
            .set_texture_interpolation(self, interpolation)
    }

    fn max_texture_size(&self) -> (u32, u32) {
        let max_size = self
            .device_and_queue
            .device()
            .limits()
            .max_texture_dimension_2d;
        (max_size, max_size)
    }

    fn create_vertex_buffer(&self) -> Result<Self::VertexBuffer, Self::Error> {
        Ok(WgpuVertexBuffer::new(self))
    }

    fn delete_vertex_buffer(&self, buffer: Self::VertexBuffer) {
        // Drop the buffer.
        drop(buffer);
    }

    fn write_vertices(
        &self,
        buffer: &Self::VertexBuffer,
        vertices: &[piet_hardware::Vertex],
        indices: &[u32],
    ) {
        buffer
            .borrow_vertex_buffer_mut()
            .write_buffer(self, bytemuck::cast_slice::<Vertex, u8>(vertices));
        buffer
            .borrow_index_buffer_mut()
            .write_buffer(self, bytemuck::cast_slice::<u32, u8>(indices));
    }

    fn push_buffers(
        &self,
        vertex_buffer: &Self::VertexBuffer,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &Affine,
        (viewport_width, viewport_height): (u32, u32),
    ) -> Result<(), Self::Error> {
        // Pop off slices.
        let vb_slice = vertex_buffer.borrow_vertex_buffer_mut().pop_slice();
        let ib_slice = vertex_buffer.borrow_index_buffer_mut().pop_slice();

        // See if we have an existing bind group for this buffer.
        let uniforms = Uniforms {
            transform: affine_to_column_major(transform),
            pad: [0xFFFFFFFF; 2],
            viewport_size: [viewport_width as f32, viewport_height as f32],
        };
        let bytes: UniformBytes = bytemuck::cast(uniforms);

        let bind_group = match self.uniform_buffers.borrow_mut().entry(bytes) {
            Entry::Occupied(o) => o.get().1.clone(),
            Entry::Vacant(entry) => {
                // Create a new buffer.
                let buffer = self.device_and_queue.device().create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &bytes,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                );

                // Create a new bind group.
                let bind_group =
                    self.device_and_queue
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &self.uniform_bind_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buffer.as_entire_binding(),
                            }],
                        });

                // Insert it into the set.
                let (_, bind_group) = entry.insert((buffer, Rc::new(bind_group)));

                // Return the bind group.
                bind_group.clone()
            }
        };

        self.pushed_buffers.borrow_mut().push(PushedBuffer {
            buffers: vertex_buffer.clone(),
            vertex: vb_slice,
            index: ib_slice,
            color_texture: current_texture.clone(),
            mask_texture: mask_texture.clone(),
            uniform_bind_group: bind_group,
            viewport_size: [viewport_width as f32, viewport_height as f32],
        });

        Ok(())
    }
}

fn affine_to_column_major(affine: &Affine) -> [[f32; 4]; 3] {
    let [a, b, c, d, e, f] = affine.as_coeffs();

    // Column major
    [
        [a as f32, b as f32, 0.0, 0.0],
        [c as f32, d as f32, 0.0, 0.0],
        [e as f32, f as f32, 1.0, 0.0],
    ]
}
