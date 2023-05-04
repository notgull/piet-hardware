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

use std::borrow;
use std::cell::{Cell, Ref, RefCell};
use std::convert::Infallible;
use std::mem;
use std::num::{NonZeroU32, NonZeroU64};
use std::rc::Rc;

use piet_hardware::piet::kurbo::Affine;
use piet_hardware::piet::{self, Color, Error as Pierror, ImageFormat, InterpolationMode};
use piet_hardware::{RepeatStrategy, Vertex};

use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("piet.wgsl");

/// A reference to a [`wgpu`] [`Device`], and [`Queue`].
///
/// This is used by the GPU context to create resources. It can take the form of a (Device, Queue)
/// tuple, or a tuple of references to them.
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

impl<A: borrow::Borrow<wgpu::Device>, B: borrow::Borrow<wgpu::Queue>> DeviceAndQueue for (A, B) {
    fn device(&self) -> &wgpu::Device {
        borrow::Borrow::borrow(&self.0)
    }

    fn queue(&self) -> &wgpu::Queue {
        borrow::Borrow::borrow(&self.1)
    }
}

/// A wrapper around a `wgpu` context.
struct GpuContext<DaQ: ?Sized> {
    /// The rendering pipeline.
    pipeline: wgpu::RenderPipeline,

    /// The uniform buffer.
    uniform_buffer: wgpu::Buffer,

    /// Bind group for uniforms.
    uniform_bind_group: wgpu::BindGroup,

    /// Bind group for textures.
    texture_bind_layout: wgpu::BindGroupLayout,

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
    buffers: Rc<BufferInner>,

    /// The slice into the vertex buffer.
    vertex: BufferSlice,

    /// The slice into the index buffer.
    index: BufferSlice,

    /// The color texture to use.
    color_texture: Rc<RefCell<TextureInner>>,

    /// The mask texture to use.
    mask_texture: Rc<RefCell<TextureInner>>,

    /// The transform to apply.
    transform: Affine,

    /// The viewport size.
    viewport_size: (f32, f32),
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
    color_texture: Ref<'a, TextureInner>,

    /// Borrowed mask texture.
    mask_texture: Ref<'a, TextureInner>,
}

impl PushedBuffer {
    fn borrow(&self) -> BorrowedPush<'_> {
        BorrowedPush {
            source: self,
            vb: self.buffers.vertex_buffer.borrow(),
            ib: self.buffers.index_buffer.borrow(),
            color_texture: self.color_texture.borrow(),
            mask_texture: self.mask_texture.borrow(),
        }
    }
}

/// The resource representing a WGPU texture.
struct WgpuTexture(Rc<RefCell<TextureInner>>);

/// Inner data for a texture.
struct TextureInner {
    /// The texture ID.
    id: usize,

    /// The texture.
    texture: Option<wgpu::Texture>,

    /// The sampler to use.
    sampler: wgpu::Sampler,

    /// The image format we used to render.
    format: ImageFormat,

    /// The interpolation mode.
    interpolation: InterpolationMode,

    /// The address mode.
    address_mode: wgpu::AddressMode,

    /// The border color.
    border_color: Option<wgpu::SamplerBorderColor>,

    /// The bind group to use to bind to the pipeline.
    bind_group: Option<wgpu::BindGroup>,
}

impl TextureInner {
    /// Re-create the `BindGroup` from the current data.
    fn recompute_bind_group<DaQ: DeviceAndQueue + ?Sized>(&mut self, base: &GpuContext<DaQ>) {
        let texture = match self.texture.as_ref() {
            Some(texture) => texture,
            None => {
                self.bind_group = None;
                return;
            }
        };

        let new_bind_group =
            base.device_and_queue
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("piet-wgpu texture bind group {}", self.id)),
                    layout: &base.texture_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

        self.bind_group = Some(new_bind_group);
    }
}

/// The resource representing a WGPU buffer.
struct WgpuVertexBuffer(Rc<BufferInner>);

/// Inner data for a buffer.
struct BufferInner {
    /// Unique ID.
    id: usize,

    /// The buffer for vertices.
    vertex_buffer: RefCell<Buffer>,

    /// The buffer for indices.
    index_buffer: RefCell<Buffer>,
}

/// Describes the data for a buffer.
struct Buffer {
    /// The index of the inner WGPU buffer.
    id: usize,

    /// The capacity of the buffer.
    ///
    /// This is the total number of bytes that can be held by `buffer`.
    capacity: usize,

    /// The capacity of the last buffer.
    ///
    /// This is used to determine when to allocate a new buffer.
    last_capacity: usize,

    /// The starting cursor for the buffer.
    ///
    /// This is the start of the current slice and where new writes will begin. It is into the last
    /// buffer.
    start_cursor: usize,

    /// The ending cursor for the buffer.
    ///
    /// This determines the end of the current slice and where new writes will end. It is into the
    /// last buffer.
    end_cursor: usize,

    /// The buffer usages.
    usage: wgpu::BufferUsages,

    /// The identifier for the buffer.
    buffer_id: &'static str,

    /// The inner WGPU buffer.
    buffer: BufferCollection,
}

/// Either a single buffer or a list of them.
///
/// This is used to dynamically reallocate new buffers during rendering.
enum BufferCollection {
    /// A single buffer.
    Single(wgpu::Buffer),

    /// A list of buffers.
    ///
    /// This is only used when the single buffer overflows.
    List(Vec<wgpu::Buffer>),

    /// Empty hole.
    Hole,
}

impl BufferCollection {
    /// Get the buffer at the given index.
    fn get(&self, i: usize) -> Option<&wgpu::Buffer> {
        match (self, i) {
            (BufferCollection::Single(buffer), 0) => Some(buffer),
            (BufferCollection::List(buffers), i) => buffers.get(i),
            _ => None,
        }
    }

    /// Get the last buffer.
    fn last(&self) -> Option<&wgpu::Buffer> {
        match self {
            BufferCollection::Single(buffer) => Some(buffer),
            BufferCollection::List(buffers) => buffers.last(),
            _ => None,
        }
    }

    /// Get the last buffer, mutably.
    fn last_mut(&mut self) -> Option<&mut wgpu::Buffer> {
        match self {
            BufferCollection::Single(buffer) => Some(buffer),
            BufferCollection::List(buffers) => buffers.last_mut(),
            _ => None,
        }
    }

    /// Push a new buffer.
    fn push(&mut self, buffer: wgpu::Buffer) {
        match mem::replace(self, Self::Hole) {
            Self::Hole => *self = Self::Single(buffer),
            Self::Single(old_buffer) => {
                tracing::debug!("using list-based buffering strategy");
                *self = Self::List(vec![old_buffer, buffer])
            }
            Self::List(mut buffers) => {
                buffers.push(buffer);
                *self = Self::List(buffers);
            }
        }
    }

    /// Get the length of the buffer.
    fn len(&self) -> usize {
        match self {
            BufferCollection::Single(_) => 1,
            BufferCollection::List(buffers) => buffers.len(),
            _ => 0,
        }
    }

    /// Get a slice of the buffer.
    fn slice(&self, slice: BufferSlice, granularity: u64) -> Option<wgpu::BufferSlice<'_>> {
        let map_end = |end| end / granularity;
        let new_range = map_end(slice.range.0)..map_end(slice.range.1);

        self.get(slice.buffer_index).map(|buf| buf.slice(new_range))
    }
}

/// A slice out of the `BufferCollection`.
#[derive(Debug, Clone, Copy)]
struct BufferSlice {
    /// The index of the buffer.
    buffer_index: usize,

    /// The range of the slice.
    range: (u64, u64),
}

impl Buffer {
    /// Create a new buffer.
    fn create_buffer(&self, dev: &wgpu::Device, len: usize) -> wgpu::Buffer {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("piet-wgpu {} buffer {}", self.buffer_id, self.id)),
            usage: self.usage,
            size: len.try_into().expect("buffer too large"),
            mapped_at_creation: false,
        })
    }

    /// Write this data into the buffer.
    fn write_buffer<DaQ: DeviceAndQueue + ?Sized>(&mut self, base: &GpuContext<DaQ>, data: &[u8]) {
        // See if we need to allocate a new buffer.
        let remaining_capacity = self.last_capacity - self.end_cursor;
        if remaining_capacity < data.len() {
            // Round the desired length up to the nearest multiple of 2 to prevent frequent reallocs.
            let new_capacity = data
                .len()
                .checked_add(remaining_capacity)
                .map(|len| len.next_power_of_two())
                .expect("buffer too large");
            let new_buffer = self.create_buffer(base.device_and_queue.device(), new_capacity);

            // If we haven't sliced out this buffer yet, just reallocate in place.
            if self.start_cursor == 0 {
                *self.buffer.last_mut().unwrap() = new_buffer;
                self.capacity -= self.last_capacity;
            } else {
                // Push the buffer to the end.
                self.buffer.push(new_buffer);
                self.start_cursor = 0;
                self.end_cursor = 0;
            }

            self.last_capacity = new_capacity;
            self.capacity += new_capacity;
        }

        // Queue the write to the buffer.
        base.device_and_queue.queue().write_buffer(
            self.buffer.last().unwrap(),
            self.start_cursor.try_into().expect("buffer too large"),
            data,
        );

        // Update the cursor.
        self.end_cursor = self.start_cursor + data.len();
        tracing::debug!(
            "Wrote to {} buffer from {} to {}",
            self.buffer_id,
            self.start_cursor,
            self.end_cursor
        );
    }

    /// Pop off a slice of the buffer.
    fn pop_slice(&mut self) -> BufferSlice {
        let slice = BufferSlice {
            buffer_index: self.buffer.len() - 1,
            range: (self.start_cursor as u64, self.end_cursor as u64),
        };

        tracing::debug!(slice=?slice, "Popped {} buffer slice", self.buffer_id);

        // Update the cursor.
        self.start_cursor = self.end_cursor;

        slice
    }

    /// Empty out the buffer.
    fn clear(&mut self, device: &wgpu::Device) {
        // Reset the cursor.
        self.start_cursor = 0;
        self.end_cursor = 0;

        // If we are using multiple buffers, combine them all into one.
        if matches!(self.buffer, BufferCollection::List(..)) {
            let desired_capacity = self.capacity.next_power_of_two();
            tracing::debug!("Resizing {} buffer to {}", self.buffer_id, desired_capacity);
            let new_buffer = self.create_buffer(device, desired_capacity);
            self.buffer = BufferCollection::Single(new_buffer);
            self.capacity = desired_capacity;
            self.last_capacity = desired_capacity;
        }
    }

    /// Create a new buffer.
    fn new<DaQ: DeviceAndQueue + ?Sized>(
        base: &GpuContext<DaQ>,
        usage: wgpu::BufferUsages,
        starting_size: usize,
        buffer_id: &'static str,
    ) -> Self {
        let starting_size = starting_size.next_power_of_two();
        let mut this = Self {
            id: base.next_id(),
            capacity: starting_size,
            last_capacity: starting_size,
            start_cursor: 0,
            end_cursor: 0,
            buffer_id,
            usage,
            buffer: BufferCollection::Hole,
        };

        this.buffer = BufferCollection::Single(
            this.create_buffer(base.device_and_queue.device(), starting_size),
        );
        this
    }
}

/// Type of the data stored in the uniform buffer.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct Uniforms {
    /// Viewport size.
    viewport_size: [f32; 2],

    /// 3x3 transformation matrix.
    transform: [[f32; 3]; 3],
}

impl<DaQ: DeviceAndQueue + ?Sized> GpuContext<DaQ> {
    /// Create a new GPU context.
    fn new(
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

        let uniform_data = {
            let mut uniform_data = bytemuck::bytes_of(&Uniforms {
                transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                viewport_size: [0.0, 0.0],
            })
            .to_vec();

            // Extend to the next power of two.
            let new_len = uniform_data.len().next_power_of_two();
            uniform_data.resize(new_len, 0);
            uniform_data
        };

        // Create a buffer for the uniforms.
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("piet-wgpu uniform buffer"),
            contents: &uniform_data,
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
                        min_binding_size: NonZeroU64::new(
                            mem::size_of::<Uniforms>().next_power_of_two() as u64,
                        ),
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
                ],
            });

        // Use these two to create the pipline layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("piet-wgpu pipeline layout"),
            bind_group_layouts: &[
                &uniform_buffer_layout,
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
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
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
            uniform_buffer,
            uniform_bind_group,
            texture_bind_layout: texture_buffer_layout,
            clear_color: Cell::new(None),
            texture_view: RefCell::new(None),
            pushed_buffers: RefCell::new(Vec::new()),
            next_id: Cell::new(0),
        }
    }

    fn next_id(&self) -> usize {
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
                    transform,
                    viewport_size: (width, height),
                    vertex: vertex_slice,
                    index: index_slice,
                    ..
                },
            vb,
            ib,
            color_texture,
            mask_texture,
        } in &pushes
        {
            // Set a viewport.
            pass.set_viewport(0.0, 0.0, *width, *height, 0.0, 1.0);

            // Set the uniforms.
            let uniforms = Uniforms {
                transform: affine_to_column_major(transform),
                viewport_size: [*width, *height],
            };
            self.device_and_queue.queue().write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniforms]),
            );
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            // Bind textures.
            pass.set_bind_group(1, color_texture.bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, mask_texture.bind_group.as_ref().unwrap(), &[]);

            // Get the bufer slices to pass to the shader.
            let num_indices =
                (index_slice.range.1 - index_slice.range.0) as usize / mem::size_of::<u32>();
            let vertex_slice = vb
                .buffer
                .slice(*vertex_slice, 1)
                .unwrap();
            let index_slice = ib.buffer.slice(*index_slice, 1).unwrap();

            // Bind the slices into the shader.
            pass.set_index_buffer(index_slice, wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(0, vertex_slice);

            // Draw the triangles.
            pass.draw_indexed(0..num_indices as u32, 0, 0..1);

            // Push the buffer to the clear list.
            if buffers_to_clear.iter().all(|(id, _)| *id != buffers.id) {
                buffers_to_clear.push((buffers.id, Rc::clone(buffers)));
            }
        }

        // Encode to a buffer and push to the queue.
        drop(pass);
        self.device_and_queue.queue().submit(Some(encoder.finish()));

        // Clear the buffers.
        drop(pushes);
        for (_, buffers) in buffers_to_clear {
            buffers
                .vertex_buffer
                .borrow_mut()
                .clear(self.device_and_queue.device());
            buffers
                .index_buffer
                .borrow_mut()
                .clear(self.device_and_queue.device());
        }

        Ok(())
    }

    fn create_texture(
        &self,
        interpolation: InterpolationMode,
        repeat: piet_hardware::RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        let id = self.next_id();
        let filter_mode = match interpolation {
            InterpolationMode::Bilinear => wgpu::FilterMode::Linear,
            InterpolationMode::NearestNeighbor => wgpu::FilterMode::Nearest,
        };

        let mut border_color = None;
        let address_mode = match repeat {
            RepeatStrategy::Clamp => wgpu::AddressMode::ClampToEdge,
            RepeatStrategy::Repeat => wgpu::AddressMode::Repeat,
            RepeatStrategy::Color(color) => {
                border_color = Some({
                    if color == Color::TRANSPARENT {
                        wgpu::SamplerBorderColor::TransparentBlack
                    } else if color == Color::BLACK {
                        wgpu::SamplerBorderColor::OpaqueBlack
                    } else if color == Color::WHITE {
                        wgpu::SamplerBorderColor::OpaqueWhite
                    } else {
                        tracing::warn!("Invalid border color for sampler: {:?}", color);
                        wgpu::SamplerBorderColor::OpaqueWhite
                    }
                });

                wgpu::AddressMode::ClampToBorder
            }
            _ => panic!("unknown repeat strategy"),
        };

        let sampler = self
            .device_and_queue
            .device()
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some(&format!("piet-wgpu sampler {id}")),
                compare: None,
                mag_filter: filter_mode,
                min_filter: filter_mode,
                address_mode_u: address_mode,
                address_mode_v: address_mode,
                border_color,
                ..Default::default()
            });

        Ok(WgpuTexture(Rc::new(RefCell::new(TextureInner {
            id,
            texture: None,
            format: ImageFormat::Grayscale,
            sampler,
            interpolation,
            border_color,
            address_mode,
            bind_group: None,
        }))))
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
        let bytes_per_pixel = bytes_per_pixel(format);

        let size = wgpu::Extent3d {
            width: size.0,
            height: size.0,
            depth_or_array_layers: 1,
        };

        // Get the texture to write to.
        let mut guard = tex.0.borrow_mut();
        let texture = if guard.texture.is_none() || guard.format != format {
            let texture = self
                .device_and_queue
                .device()
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("piet-wgpu texture {}", guard.id)),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: match format {
                        ImageFormat::Grayscale => wgpu::TextureFormat::R8Unorm,
                        ImageFormat::Rgb => panic!("Unsupported"),
                        ImageFormat::RgbaPremul => wgpu::TextureFormat::Rgba8UnormSrgb,
                        ImageFormat::RgbaSeparate => wgpu::TextureFormat::Rgba8UnormSrgb,
                        _ => panic!("Unsupported"),
                    },
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
                });

            guard.format = format;
            guard.texture = Some(texture);

            // Reset the bind group.
            guard.recompute_bind_group(self);

            guard.texture.as_ref().unwrap()
        } else {
            guard.texture.as_ref().unwrap()
        };

        let zeroes;
        let data = match data {
            Some(data) => data,
            None => {
                zeroes =
                    vec![0; size.width as usize * size.height as usize * bytes_per_pixel as usize];
                &zeroes
            }
        };

        // Queue a data write to the texture.
        self.device_and_queue.queue().write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(size.width * bytes_per_pixel),
                rows_per_image: NonZeroU32::new(size.height),
            },
            size,
        );
    }

    fn write_subtexture(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: &[u8],
    ) {
        let guard = texture.0.borrow_mut();
        if guard.format != format {
            panic!("write_subtexture format mismatch");
        }

        let bytes_per_pixel = bytes_per_pixel(format);

        // Queue a data write to the texture.
        self.device_and_queue.queue().write_texture(
            wgpu::ImageCopyTexture {
                texture: guard.texture.as_ref().expect("texture"),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset.0,
                    y: offset.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(size.0 * bytes_per_pixel),
                rows_per_image: NonZeroU32::new(size.1),
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn set_texture_interpolation(&self, texture: &Self::Texture, interpolation: InterpolationMode) {
        let mut guard = texture.0.borrow_mut();
        if guard.interpolation != interpolation {
            let interp_mode = match interpolation {
                InterpolationMode::NearestNeighbor => wgpu::FilterMode::Nearest,
                InterpolationMode::Bilinear => wgpu::FilterMode::Linear,
            };

            guard.interpolation = interpolation;
            guard.sampler =
                self.device_and_queue
                    .device()
                    .create_sampler(&wgpu::SamplerDescriptor {
                        label: Some(&format!("piet-wgpu sampler {}", guard.id)),
                        compare: None,
                        mag_filter: interp_mode,
                        min_filter: interp_mode,
                        address_mode_u: guard.address_mode,
                        address_mode_v: guard.address_mode,
                        border_color: guard.border_color,
                        ..Default::default()
                    });
            guard.recompute_bind_group(self);
        }
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
        const INITIAL_VERTEX_BUFFER_SIZE: usize = 1024 * mem::size_of::<Vertex>();
        const INITIAL_INDEX_BUFFER_SIZE: usize = 1024 * mem::size_of::<u32>();

        let vertex_buffer = Buffer::new(
            self,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            INITIAL_VERTEX_BUFFER_SIZE,
            "vertex",
        );
        let index_buffer = Buffer::new(
            self,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            INITIAL_INDEX_BUFFER_SIZE,
            "index",
        );

        Ok(WgpuVertexBuffer(Rc::new(BufferInner {
            id: self.next_id(),
            vertex_buffer: RefCell::new(vertex_buffer),
            index_buffer: RefCell::new(index_buffer),
        })))
    }

    fn delete_vertex_buffer(&self, buffer: Self::VertexBuffer) {
        // Drop the buffer.
        drop(buffer);
    }

    unsafe fn write_vertices(
        &self,
        buffer: &Self::VertexBuffer,
        vertices: &[piet_hardware::Vertex],
        indices: &[u32],
    ) {
        buffer
            .0
            .vertex_buffer
            .borrow_mut()
            .write_buffer(self, bytemuck::cast_slice::<Vertex, u8>(vertices));
        buffer
            .0
            .index_buffer
            .borrow_mut()
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
        let vb_slice = vertex_buffer.0.vertex_buffer.borrow_mut().pop_slice();
        let ib_slice = vertex_buffer.0.index_buffer.borrow_mut().pop_slice();

        self.pushed_buffers.borrow_mut().push(PushedBuffer {
            buffers: vertex_buffer.0.clone(),
            vertex: vb_slice,
            index: ib_slice,
            color_texture: current_texture.0.clone(),
            mask_texture: mask_texture.0.clone(),
            transform: *transform,
            viewport_size: (viewport_width as f32, viewport_height as f32),
        });

        Ok(())
    }
}

/// A wrapper around a [`wgpu`] [`Device`] and [`Queue`] with cached information.
pub struct WgpuContext<D: DeviceAndQueue + ?Sized> {
    source: piet_hardware::Source<GpuContext<D>>,
    text: Text,
}

impl<D: DeviceAndQueue + ?Sized> WgpuContext<D> {
    /// Create a new [`WgpuContext`] from a [`Device`] and [`Queue`].
    pub fn new(
        device_and_queue: D,
        output_format: wgpu::TextureFormat,
        samples: u32,
    ) -> Result<Self, Pierror>
    where
        D: Sized,
    {
        let source = piet_hardware::Source::new(GpuContext::new(
            device_and_queue,
            output_format,
            None,
            samples,
        ))?;
        let text = source.text().clone();

        Ok(Self {
            source,
            text: Text(text),
        })
    }

    /// Get a reference to the underlying [`Device`] and [`Queue`].
    pub fn device_and_queue(&self) -> &D {
        &self.source.context().device_and_queue
    }

    /// Get the render context.
    pub fn render_context(
        &mut self,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> RenderContext<'_, D> {
        self.source
            .context()
            .texture_view
            .borrow_mut()
            .replace(view);

        RenderContext {
            text: &mut self.text,
            context: self.source.render_context(width, height),
        }
    }
}

/// The whole point.
pub struct RenderContext<'a, D: DeviceAndQueue + ?Sized> {
    context: piet_hardware::RenderContext<'a, GpuContext<D>>,
    text: &'a mut Text,
}

impl<D: DeviceAndQueue + ?Sized> piet::RenderContext for RenderContext<'_, D> {
    type Brush = Brush<D>;
    type Image = Image<D>;
    type Text = Text;
    type TextLayout = TextLayout;

    fn blurred_rect(
        &mut self,
        rect: piet::kurbo::Rect,
        blur_radius: f64,
        brush: &impl piet::IntoBrush<Self>,
    ) {
        let brush = brush.make_brush(self, || rect);
        self.context
            .blurred_rect(rect, blur_radius, &brush.as_ref().0)
    }

    fn capture_image_area(
        &mut self,
        src_rect: impl Into<piet::kurbo::Rect>,
    ) -> Result<Self::Image, Pierror> {
        self.context.capture_image_area(src_rect).map(Image)
    }

    fn clear(&mut self, region: impl Into<Option<piet::kurbo::Rect>>, color: Color) {
        self.context.clear(region, color)
    }

    fn clip(&mut self, shape: impl piet::kurbo::Shape) {
        self.context.clip(shape)
    }

    fn current_transform(&self) -> Affine {
        self.context.current_transform()
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<piet::kurbo::Rect>,
        interp: InterpolationMode,
    ) {
        self.context.draw_image(&image.0, dst_rect, interp)
    }

    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<piet::kurbo::Rect>,
        dst_rect: impl Into<piet::kurbo::Rect>,
        interp: InterpolationMode,
    ) {
        self.context
            .draw_image_area(&image.0, src_rect, dst_rect, interp)
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<piet::kurbo::Point>) {
        self.context.draw_text(&layout.0, pos)
    }

    fn fill(&mut self, shape: impl piet::kurbo::Shape, brush: &impl piet::IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context.fill(shape, &brush.as_ref().0)
    }

    fn fill_even_odd(
        &mut self,
        shape: impl piet::kurbo::Shape,
        brush: &impl piet::IntoBrush<Self>,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context.fill_even_odd(shape, &brush.as_ref().0)
    }

    fn finish(&mut self) -> Result<(), Pierror> {
        self.context.finish()
    }

    fn gradient(
        &mut self,
        gradient: impl Into<piet::FixedGradient>,
    ) -> Result<Self::Brush, Pierror> {
        self.context.gradient(gradient).map(Brush)
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image, Pierror> {
        self.context
            .make_image(width, height, buf, format)
            .map(Image)
    }

    fn restore(&mut self) -> Result<(), Pierror> {
        self.context.restore()
    }

    fn save(&mut self) -> Result<(), Pierror> {
        self.context.save()
    }

    fn solid_brush(&mut self, color: Color) -> Self::Brush {
        Brush(self.context.solid_brush(color))
    }

    fn status(&mut self) -> Result<(), Pierror> {
        self.context.status()
    }

    fn stroke(
        &mut self,
        shape: impl piet::kurbo::Shape,
        brush: &impl piet::IntoBrush<Self>,
        width: f64,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context.stroke(shape, &brush.as_ref().0, width)
    }

    fn stroke_styled(
        &mut self,
        shape: impl piet::kurbo::Shape,
        brush: &impl piet::IntoBrush<Self>,
        width: f64,
        style: &piet::StrokeStyle,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context
            .stroke_styled(shape, &brush.as_ref().0, width, style)
    }

    fn text(&mut self) -> &mut Self::Text {
        self.text
    }

    fn transform(&mut self, transform: Affine) {
        self.context.transform(transform)
    }
}

/// The brush type.
pub struct Brush<D: DeviceAndQueue + ?Sized>(piet_hardware::Brush<GpuContext<D>>);

impl<D: DeviceAndQueue + ?Sized> Clone for Brush<D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<D: DeviceAndQueue + ?Sized> piet::IntoBrush<RenderContext<'_, D>> for Brush<D> {
    fn make_brush<'a>(
        &'a self,
        _piet: &mut RenderContext<'_, D>,
        _bbox: impl FnOnce() -> piet::kurbo::Rect,
    ) -> std::borrow::Cow<'a, Brush<D>> {
        std::borrow::Cow::Borrowed(self)
    }
}

/// The image type.
pub struct Image<D: DeviceAndQueue + ?Sized>(piet_hardware::Image<GpuContext<D>>);

impl<D: DeviceAndQueue + ?Sized> Clone for Image<D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<D: DeviceAndQueue + ?Sized> piet::Image for Image<D> {
    fn size(&self) -> piet::kurbo::Size {
        self.0.size()
    }
}

/// The text layout type.
#[derive(Clone)]
pub struct TextLayout(piet_hardware::TextLayout);

impl piet::TextLayout for TextLayout {
    fn size(&self) -> piet::kurbo::Size {
        self.0.size()
    }

    fn line_text(&self, line_number: usize) -> Option<&str> {
        self.0.line_text(line_number)
    }

    fn line_metric(&self, line_number: usize) -> Option<piet::LineMetric> {
        self.0.line_metric(line_number)
    }

    fn line_count(&self) -> usize {
        self.0.line_count()
    }

    fn hit_test_point(&self, point: piet::kurbo::Point) -> piet::HitTestPoint {
        self.0.hit_test_point(point)
    }

    fn trailing_whitespace_width(&self) -> f64 {
        self.0.trailing_whitespace_width()
    }

    fn image_bounds(&self) -> piet::kurbo::Rect {
        self.0.image_bounds()
    }

    fn text(&self) -> &str {
        self.0.text()
    }

    fn hit_test_text_position(&self, idx: usize) -> piet::HitTestPosition {
        self.0.hit_test_text_position(idx)
    }
}

/// The text layout builder type.
pub struct TextLayoutBuilder(piet_hardware::TextLayoutBuilder);

impl piet::TextLayoutBuilder for TextLayoutBuilder {
    type Out = TextLayout;

    fn max_width(self, width: f64) -> Self {
        Self(self.0.max_width(width))
    }

    fn alignment(self, alignment: piet::TextAlignment) -> Self {
        Self(self.0.alignment(alignment))
    }

    fn default_attribute(self, attribute: impl Into<piet::TextAttribute>) -> Self {
        Self(self.0.default_attribute(attribute))
    }

    fn range_attribute(
        self,
        range: impl std::ops::RangeBounds<usize>,
        attribute: impl Into<piet::TextAttribute>,
    ) -> Self {
        Self(self.0.range_attribute(range, attribute))
    }

    fn build(self) -> Result<Self::Out, Pierror> {
        Ok(TextLayout(self.0.build()?))
    }
}

/// The text engine type.
#[derive(Clone)]
pub struct Text(piet_hardware::Text);

impl piet::Text for Text {
    type TextLayoutBuilder = TextLayoutBuilder;
    type TextLayout = TextLayout;

    fn font_family(&mut self, family_name: &str) -> Option<piet::FontFamily> {
        self.0.font_family(family_name)
    }

    fn load_font(&mut self, data: &[u8]) -> Result<piet::FontFamily, Pierror> {
        self.0.load_font(data)
    }

    fn new_text_layout(&mut self, text: impl piet::TextStorage) -> Self::TextLayoutBuilder {
        TextLayoutBuilder(self.0.new_text_layout(text))
    }
}

fn affine_to_column_major(affine: &Affine) -> [[f32; 3]; 3] {
    let [a, b, c, d, e, f] = affine.as_coeffs();

    // Column major
    [
        [a as f32, c as f32, 0.0],
        [b as f32, d as f32, 0.0],
        [e as f32, f as f32, 1.0],
    ]
}

fn bytes_per_pixel(format: ImageFormat) -> u32 {
    match format {
        ImageFormat::Grayscale => 1u32,
        ImageFormat::Rgb => 3,
        ImageFormat::RgbaPremul => 4,
        ImageFormat::RgbaSeparate => 4,
        _ => panic!("Unsupported"),
    }
}
