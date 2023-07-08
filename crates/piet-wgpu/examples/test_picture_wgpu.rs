// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-glow`.
//
// `piet-glow` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-glow` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-glow`. If not, see <https://www.gnu.org/licenses/>.

//! Test harness for the comparison generator.

use piet::samples;
use std::cell::RefCell;
use std::io::prelude::*;
use std::sync::mpsc;

async fn entry() -> ! {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

    // Get the adaptor, device and queue.
    let adaptor = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find an appropriate adapter");
    let (device, queue) = adaptor
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let samples = 16;

    let context = piet_wgpu::WgpuContext::new(&device, &queue, format, None, samples);

    // Sigh...
    struct WgpuState {
        context: piet_wgpu::WgpuContext,
        device: wgpu::Device,
        queue: wgpu::Queue,
        samples: u32,
        format: wgpu::TextureFormat,
    }

    std::thread_local! {
        static DEVICE_AND_QUEUE: RefCell<Option<WgpuState>>
            = RefCell::new(None);
    }
    DEVICE_AND_QUEUE.with(move |slot| {
        *slot.borrow_mut() = Some(WgpuState {
            context,
            device,
            queue,
            samples,
            format,
        })
    });

    // Call the samples main.
    samples::samples_main(
        |number, scale, path| {
            DEVICE_AND_QUEUE.with(|daq| {
                let mut guard = daq.borrow_mut();
                let state = guard.as_mut().unwrap();

                let context = &mut state.context;
                let device = &state.device;
                let queue = &state.queue;
                let samples = state.samples;
                let format = state.format;

                if number == 12 || number == 16 {
                    return Ok(());
                }

                // Get the picture.
                let picture = samples::get(number)?;
                let size = picture.size();

                let scaled_width = (size.width * scale) as u32;
                let scaled_height = (size.height * scale) as u32;

                // Create a texture to render into.
                let dims = BufferDimensions::new(scaled_width as usize, scaled_height as usize);
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("sample output for #{number}")),
                    size: wgpu::Extent3d {
                        width: dims.width as _,
                        height: dims.height as _,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    view_formats: &[format],
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                });
                let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Create a texture to use for MSAA.
                let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("msaa output for #{number}")),
                    size: wgpu::Extent3d {
                        width: dims.width as _,
                        height: dims.height as _,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    view_formats: &[format],
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                });
                let msaa_texture_view =
                    msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Create a buffer to copy the texture into.
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("buffer output for #{number}")),
                    size: (dims.padded_bytes_per_row * dims.height) as _,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                // Create the render context.
                let mut rc = context.prepare(device, queue, dims.width as u32, dims.height as u32);
                piet::RenderContext::text(&mut rc).set_dpi(72.0);
                piet::RenderContext::transform(&mut rc, piet::kurbo::Affine::scale(scale));

                // Draw the picture.
                picture.draw(&mut rc)?;

                // Begin encoding commands.
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("encoder for sample {number}")),
                });

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &msaa_texture_view,
                            resolve_target: Some(&texture_view),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });

                    // Render to the texture.
                    context.render(&mut render_pass);
                }

                // Copy the texture into the buffer.
                encoder.copy_texture_to_buffer(
                    texture.as_image_copy(),
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(dims.padded_bytes_per_row as u32),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: dims.width as _,
                        height: dims.height as _,
                        depth_or_array_layers: 1,
                    },
                );

                // Map the buffer and read the data.
                let index = queue.submit(std::iter::once(encoder.finish()));
                let buffer_slice = buffer.slice(..);
                let (send, recv) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
                    res.unwrap();
                    send.send(()).unwrap();
                });
                device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));
                recv.recv().unwrap();

                // Get the mapped range and read the data.
                let range = buffer_slice.get_mapped_range();
                let mut png_encoder = png::Encoder::new(
                    std::fs::File::create(path).unwrap(),
                    dims.width as u32,
                    dims.height as u32,
                );
                png_encoder.set_depth(png::BitDepth::Eight);
                png_encoder.set_color(png::ColorType::Rgba);
                let mut png_writer = png_encoder
                    .write_header()
                    .unwrap()
                    .into_stream_writer_with_size(dims.unpadded_bytes_per_row)
                    .unwrap();

                for chunk in range.chunks(dims.padded_bytes_per_row) {
                    png_writer
                        .write_all(&chunk[..dims.unpadded_bytes_per_row])
                        .unwrap();
                }
                png_writer.finish().unwrap();

                drop(range);
                buffer.unmap();

                Ok(())
            })
        },
        "piet-wgpu",
        None,
    )
}

fn main() -> ! {
    futures_lite::future::block_on(entry())
}

#[derive(Debug)]
struct BufferDimensions {
    width: usize,
    height: usize,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl BufferDimensions {
    fn new(width: usize, height: usize) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>();
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}
