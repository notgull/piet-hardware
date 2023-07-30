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

//! The underlying GPU context.

use super::buffer::WgpuVertexBuffer;
use super::texture::WgpuTexture;

use std::cell::RefCell;
use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::fmt;
use std::mem;
use std::num::NonZeroU64;
use std::ops::Range;
use std::rc::Rc;

use piet_hardware::piet::kurbo::Affine;
use piet_hardware::piet::{Color, InterpolationMode};
use piet_hardware::Vertex;

use wgpu::util::DeviceExt;

const CLEAR_SHADER_SOURCE: &str = include_str!("shaders/clear.wgsl");
const GEOM_SHADER_SOURCE: &str = include_str!("shaders/geom.wgsl");

/// Common state between drawing operations.
#[derive(Debug)]
pub(crate) struct GpuContext {
    /// The rendering pipeline for geometry.
    geometry_pipeline: wgpu::RenderPipeline,

    /// The rendering pipeline for clearing the screen.
    clear_pipeline: wgpu::RenderPipeline,

    /// The bind group layout for uniforms.
    uniform_bind_layout: wgpu::BindGroupLayout,

    /// Bind group for textures.
    texture_bind_layout: wgpu::BindGroupLayout,

    /// The buffer of drawing operations.
    pushed_buffers: Vec<DrawOp>,

    /// The hash map of uniform buffers.
    uniform_buffers: HashMap<UniformBytes, BufferGroup>,

    /// Map between colors and buffers containing those colors.
    color_buffers: HashMap<Color, BufferGroup>,

    /// Bind group layout for the color bffers.
    color_bind_layout: wgpu::BindGroupLayout,

    /// Unique IDs for textures and buffers.
    next_id: usize,

    /// List of vertex buffers that need to be cleared.
    buffers_to_clear: RefCell<HashSet<WgpuVertexBuffer>>,

    /// A singular white pixel.
    white_pixel: Rc<wgpu::BindGroup>,
}

/// An operation that can be performed by the context.
#[derive(Debug)]
pub(crate) enum DrawOp {
    /// Clear the screen with the provided color.
    Clear(Rc<wgpu::BindGroup>),

    /// Push a buffer to the screen.
    ///
    /// This corresponds to a single draw call.
    PushedBuffer(PushedBuffer),
}

impl DrawOp {
    /// Draw the operation.
    ///
    /// Returns vertex buffers to clear, if any.
    fn process<'this>(
        &'this self,
        pass: &mut wgpu::RenderPass<'this>,
        geom_pipeline: &'this wgpu::RenderPipeline,
        clear_pipeline: &'this wgpu::RenderPipeline,
    ) -> Option<WgpuVertexBuffer> {
        // Figure out the operation to draw.
        match self {
            DrawOp::PushedBuffer(buffer) => {
                let PushedBuffer {
                    buffers,
                    vertex_buffer,
                    index_buffer,
                    vertex,
                    index,
                    color_texture,
                    mask_texture,
                    viewport_size,
                    uniform_bind_group,
                } = buffer;

                // Set the pipeline.
                pass.set_pipeline(geom_pipeline);

                // Set the viewport and the bind group.
                pass.set_viewport(0.0, 0.0, viewport_size[0], viewport_size[1], 0.0, 1.0);
                pass.set_bind_group(0, uniform_bind_group, &[]);

                // Bind textures.
                pass.set_bind_group(1, color_texture, &[]);
                pass.set_bind_group(2, mask_texture, &[]);

                // Get the buffer slices and draw.
                let num_indices = index.clone().count() / mem::size_of::<u32>();
                let vertex_slice = vertex_buffer.slice(vertex.clone());
                let index_slice = index_buffer.slice(index.clone());

                // Bind the slices into the shaders.
                pass.set_vertex_buffer(0, vertex_slice);
                pass.set_index_buffer(index_slice, wgpu::IndexFormat::Uint32);

                // Draw triangles.
                pass.draw_indexed(0..num_indices as u32, 0, 0..1);

                // Queue a buffer clear.
                Some(buffers.clone())
            }

            DrawOp::Clear(color) => {
                // Set the pipeline.
                pass.set_pipeline(clear_pipeline);

                // Set the bind group.
                pass.set_bind_group(0, color, &[]);

                // Draw the quad.
                pass.draw(0..6, 0..1);

                None
            }
        }
    }
}

/// Represents a pushed buffer call.
#[derive(Debug)]
pub(crate) struct PushedBuffer {
    /// The buffers that were pushed to the screen.
    buffers: WgpuVertexBuffer,

    /// The vertex buffer to draw with.
    vertex_buffer: Rc<wgpu::Buffer>,

    /// The index buffer to draw with.
    index_buffer: Rc<wgpu::Buffer>,

    /// The slice into the vertex buffer.
    vertex: Range<u64>,

    /// The slice into the index buffer.
    index: Range<u64>,

    /// The color texture to use.
    color_texture: Rc<wgpu::BindGroup>,

    /// The mask texture to use.
    mask_texture: Rc<wgpu::BindGroup>,

    /// The viewport size.
    viewport_size: [f32; 2],

    /// The bind group for uniforms.
    uniform_bind_group: Rc<wgpu::BindGroup>,
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

#[derive(Debug)]
pub(crate) struct NotYetSupported;

impl fmt::Display for NotYetSupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not yet supported")
    }
}

impl std::error::Error for NotYetSupported {}

impl GpuContext {
    /// Create a new GPU context.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_color_format: wgpu::TextureFormat,
        output_depth_format: Option<wgpu::TextureFormat>,
        samples: u32,
    ) -> Self {
        // Create the shader module.
        let geom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("piet-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(GEOM_SHADER_SOURCE.into()),
        });
        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("piet-wgpu clear shader"),
            source: wgpu::ShaderSource::Wgsl(CLEAR_SHADER_SOURCE.into()),
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

        // Create a buffer layout for the colors.
        let color_buffer_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("piet-wgpu color buffer layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            mem::size_of::<Color>().next_power_of_two() as u64,
                        ),
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            });

        // Use these two to create the pipline layout.
        let geom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("piet-wgpu pipeline layout"),
            bind_group_layouts: &[
                &uniform_bind_layout,
                &texture_buffer_layout,
                &texture_buffer_layout,
            ],
            push_constant_ranges: &[],
        });
        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("piet-wgpu clear pipeline layout"),
                bind_group_layouts: &[&color_buffer_layout],
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
        let geometry_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("piet-wgpu geometry pipeline"),
            layout: Some(&geom_pipeline_layout),
            vertex: wgpu::VertexState {
                entry_point: "vertex_main",
                module: &geom_shader,
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
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState {
                alpha_to_coverage_enabled: false,
                count: samples,
                mask: !0,
            },
            fragment: Some(wgpu::FragmentState {
                module: &geom_shader,
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

        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("piet-wgpu clear pipeline"),
            layout: Some(&clear_pipeline_layout),
            vertex: wgpu::VertexState {
                entry_point: "vertex_main",
                module: &clear_shader,
                buffers: &[],
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
                module: &clear_shader,
                entry_point: "fragment_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        // Create a texture, repeating, with a singular white pixel.
        let white_pixel = {
            let texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("piet-wgpu white pixel texture"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: output_color_format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[output_color_format],
                },
                &[0xFF, 0xFF, 0xFF, 0xFF],
            );

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("piet-wgpu white pixel sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("piet-wgpu white pixel"),
                layout: &texture_buffer_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        };

        Self {
            geometry_pipeline,
            clear_pipeline,
            uniform_bind_layout,
            texture_bind_layout: texture_buffer_layout,
            pushed_buffers: Vec::new(),
            uniform_buffers: HashMap::new(),
            color_bind_layout: color_buffer_layout,
            color_buffers: HashMap::new(),
            next_id: 0,
            buffers_to_clear: RefCell::new(HashSet::new()),
            white_pixel: Rc::new(white_pixel),
        }
    }

    pub(crate) fn next_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Render to a render pass.
    pub(crate) fn render<'this>(&'this self, pass: &mut wgpu::RenderPass<'this>) {
        let mut buffers_to_clear = self.buffers_to_clear.borrow_mut();

        for draw_op in &self.pushed_buffers {
            if let Some(buffer) =
                draw_op.process(pass, &self.geometry_pipeline, &self.clear_pipeline)
            {
                buffers_to_clear.insert(buffer);
            }
        }
    }

    /// Run this once the queue has been flushed.
    pub(crate) fn gpu_flushed(&mut self, device: &wgpu::Device) {
        let mut buffers_to_clear = self.buffers_to_clear.borrow_mut();

        for buffer in buffers_to_clear.drain() {
            buffer.borrow_vertex_buffer_mut().clear(device);
            buffer.borrow_index_buffer_mut().clear(device);
        }

        self.pushed_buffers.clear();
    }

    /// Draw a texture encompassing the entire screen.
    fn push_texture_draw_op(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &WgpuTexture,
        [viewport_width, viewport_height]: [f32; 2],
    ) {
        macro_rules! make_vertex {
            ($px:expr,$py:expr,$ux:expr,$uy:expr) => {{
                Vertex {
                    pos: [$px, $py],
                    uv: [$ux, $uy],
                    color: [0xFF, 0xFF, 0xFF, 0xFF],
                }
            }};
        }

        const RECTANGLE: &[Vertex] = &[
            make_vertex!(-1.0, -1.0, 0.0, 0.0),
            make_vertex!(1.0, -1.0, 1.0, 0.0),
            make_vertex!(-1.0, 1.0, 0.0, 1.0),
            make_vertex!(1.0, -1.0, 1.0, 0.0),
            make_vertex!(1.0, 1.0, 1.0, 1.0),
            make_vertex!(-1.0, 1.0, 0.0, 1.0),
        ];

        // Add a new draw operation that uses the target texture to draw a rectangle on the
        // screen.
        let buffers = WgpuVertexBuffer::new([self.next_id(), self.next_id()], device);
        let vertex = {
            let mut vb = buffers.borrow_vertex_buffer_mut();
            vb.write_buffer(device, queue, bytemuck::cast_slice(RECTANGLE));
            vb.pop_slice()
        };
        let index = {
            const INDEXES: &[usize] = &[0, 1, 2, 3, 4, 5, 6];

            let mut ib = buffers.borrow_index_buffer_mut();
            ib.write_buffer(device, queue, bytemuck::cast_slice(INDEXES));
            ib.pop_slice()
        };

        // Create a uniform buffer for the viewport size.
        let uniforms = {
            let uniforms = Uniforms {
                viewport_size: [viewport_width, viewport_height],
                pad: [0xFFFFFFFF; 2],
                transform: affine_to_column_major(&Affine::IDENTITY),
            };

            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("piet-wgpu capture_area texture uniform buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("piet-wgpu capture_area texture uniform bind group"),
                layout: &self.uniform_bind_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };

        self.pushed_buffers.push(DrawOp::PushedBuffer(PushedBuffer {
            vertex: vertex.range(),
            index: index.range(),
            color_texture: texture.bind_group(),
            mask_texture: self.white_pixel.clone(),
            vertex_buffer: buffers.borrow_vertex_buffer().get(vertex).unwrap().clone(),
            index_buffer: buffers.borrow_index_buffer().get(index).unwrap().clone(),
            buffers: buffers.clone(),
            viewport_size: [viewport_width, viewport_height],
            uniform_bind_group: Rc::new(uniforms),
        }));
    }
}

impl piet_hardware::GpuContext for GpuContext {
    type Device = wgpu::Device;
    type Queue = wgpu::Queue;
    type Texture = WgpuTexture;
    type VertexBuffer = WgpuVertexBuffer;
    type Error = NotYetSupported;

    fn clear(&mut self, device: &wgpu::Device, _: &wgpu::Queue, color: piet_hardware::piet::Color) {
        // This clear will remove all of the currently pushed buffers, delete them if they exist.
        for draw_op in self.pushed_buffers.drain(..) {
            if let DrawOp::PushedBuffer(PushedBuffer { buffers, .. }) = draw_op {
                buffers.borrow_vertex_buffer_mut().clear(device);
                buffers.borrow_index_buffer_mut().clear(device);
            }
        }

        // Get the color binding to use.
        let bind_group = match self.color_buffers.entry(color) {
            Entry::Occupied(o) => o.get().bind_group.clone(),
            Entry::Vacant(v) => {
                // Create a new buffer.
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: {
                        let (r, g, b, a) = color.as_rgba8();
                        &[r, g, b, a]
                    },
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                // Create a new bind group.
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.color_bind_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });

                // Insert it into the set.
                let BufferGroup { bind_group, .. } = v.insert(BufferGroup {
                    bind_group: Rc::new(bind_group),
                    _buffer: buffer,
                });

                // Return the bind group.
                bind_group.clone()
            }
        };

        // Push the clear operation.
        self.pushed_buffers.push(DrawOp::Clear(bind_group));
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        // Flushing is handled up the chain.
        Ok(())
    }

    fn create_texture(
        &mut self,
        device: &wgpu::Device,
        interpolation: InterpolationMode,
        repeat: piet_hardware::RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        Ok(WgpuTexture::create_texture(
            self.next_id(),
            device,
            interpolation,
            repeat,
            false,
        ))
    }

    fn write_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex: &Self::Texture,
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: Option<&[u8]>,
    ) {
        tex.borrow_mut()
            .write_texture(device, queue, &self.texture_bind_layout, size, format, data)
    }

    fn write_subtexture(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: &[u8],
    ) {
        texture
            .borrow_mut()
            .write_subtexture(queue, offset, size, format, data)
    }

    fn set_texture_interpolation(
        &mut self,
        device: &wgpu::Device,
        texture: &Self::Texture,
        interpolation: InterpolationMode,
    ) {
        texture.borrow_mut().set_texture_interpolation(
            device,
            &self.texture_bind_layout,
            interpolation,
        )
    }

    fn capture_area(
        &mut self,
        device: &Self::Device,
        queue: &Self::Queue,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        bitmap_scale: f64,
    ) -> Result<(), Self::Error> {
        tracing::warn!("capture_area is not performant on the wgpu backend, consider using a software rasterizer instead");

        let (x, y) = offset;
        let (width, height) = size;

        let (x, y, width, height) = (
            (x as f64 * bitmap_scale) as u32,
            (y as f64 * bitmap_scale) as u32,
            (width as f64 * bitmap_scale) as u32,
            (height as f64 * bitmap_scale) as u32,
        );

        // Figure out the size of our texture.
        let [vwidth, vheight] = self
            .pushed_buffers
            .iter()
            .fold([0.0, 0.0], |mut size, buffer| match buffer {
                DrawOp::PushedBuffer(buf) => {
                    if buf.viewport_size[0] > size[0] {
                        size[0] = buf.viewport_size[0];
                    }

                    if buf.viewport_size[1] > size[1] {
                        size[1] = buf.viewport_size[1];
                    }

                    size
                }
                _ => size,
            });

        // Create the texture to render our current operations into.
        let target_texture_block: WgpuTexture = WgpuTexture::create_texture(
            self.next_id(),
            device,
            crate::InterpolationMode::Bilinear,
            piet_hardware::RepeatStrategy::Clamp,
            true,
        );
        let mut target_texture = target_texture_block.borrow_mut();
        target_texture.write_texture(
            device,
            queue,
            &self.texture_bind_layout,
            (vwidth as u32, vheight as u32),
            crate::ImageFormat::RgbaSeparate,
            None,
        );
        let target_view = target_texture
            .texture()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create a render pass for rendering into this texture.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("piet-wgpu capture_area encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("piet-wgpu capture_area render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            // Render ourselves into this pass.
            self.render(&mut pass);
        }

        // Schedule an operation on the queue to copy the texture.
        {
            let mut texture = texture.borrow_mut();
            texture.write_texture(
                device,
                queue,
                &self.texture_bind_layout,
                (width, height),
                crate::ImageFormat::RgbaSeparate,
                None,
            );
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: target_texture.texture().unwrap(),
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    mip_level: 0,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: texture.texture().unwrap(),
                    origin: wgpu::Origin3d::ZERO,
                    mip_level: 0,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Submit the queue using this buffer.
        let index = queue.submit(Some(encoder.finish()));

        // Wait for the submission to finish.
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));

        // The queue has now been submitted.
        self.gpu_flushed(device);

        // Write the texture to the screen afterwards to avoid losing data.
        drop(target_texture);
        self.push_texture_draw_op(device, queue, &target_texture_block, [vwidth, vheight]);
        Ok(())
    }

    fn max_texture_size(&mut self, device: &wgpu::Device) -> (u32, u32) {
        let max_size = device.limits().max_texture_dimension_2d;
        (max_size, max_size)
    }

    fn create_vertex_buffer(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Self::VertexBuffer, Self::Error> {
        Ok(WgpuVertexBuffer::new(
            [self.next_id(), self.next_id()],
            device,
        ))
    }

    fn write_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &Self::VertexBuffer,
        vertices: &[piet_hardware::Vertex],
        indices: &[u32],
    ) {
        buffer.borrow_vertex_buffer_mut().write_buffer(
            device,
            queue,
            bytemuck::cast_slice::<Vertex, u8>(vertices),
        );
        buffer.borrow_index_buffer_mut().write_buffer(
            device,
            queue,
            bytemuck::cast_slice::<u32, u8>(indices),
        );
    }

    fn push_buffers(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
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

        let bind_group = match self.uniform_buffers.entry(bytes) {
            Entry::Occupied(o) => o.get().bind_group.clone(),
            Entry::Vacant(entry) => {
                // Create a new buffer.
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &bytes,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                // Create a new bind group.
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.uniform_bind_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });

                // Insert it into the set.
                let BufferGroup { bind_group, .. } = entry.insert(BufferGroup {
                    bind_group: Rc::new(bind_group),
                    _buffer: buffer,
                });

                // Return the bind group.
                bind_group.clone()
            }
        };

        self.pushed_buffers.push(DrawOp::PushedBuffer(PushedBuffer {
            buffers: vertex_buffer.clone(),
            vertex_buffer: vertex_buffer
                .borrow_vertex_buffer()
                .get(vb_slice)
                .unwrap()
                .clone(),
            index_buffer: vertex_buffer
                .borrow_index_buffer()
                .get(ib_slice)
                .unwrap()
                .clone(),
            vertex: vb_slice.range(),
            index: ib_slice.range(),
            color_texture: current_texture.bind_group(),
            mask_texture: mask_texture.bind_group(),
            uniform_bind_group: bind_group,
            viewport_size: [viewport_width as f32, viewport_height as f32],
        }));

        Ok(())
    }
}

#[derive(Debug)]
struct BufferGroup {
    _buffer: wgpu::Buffer,
    bind_group: Rc<wgpu::BindGroup>,
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
