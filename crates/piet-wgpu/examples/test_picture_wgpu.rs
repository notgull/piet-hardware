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
                features: wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER,
                ..Default::default()
            },
            None,
        )
        .await
        .expect("Failed to create device");
    let context = piet_wgpu::WgpuContext::new((device, queue), wgpu::TextureFormat::Rgba8Unorm, 1)
        .expect("Failed to create piet context");

    // Sigh...
    std::thread_local! {
        static DEVICE_AND_QUEUE: RefCell<Option<piet_wgpu::WgpuContext<(wgpu::Device, wgpu::Queue)>>>
            = RefCell::new(None);
    }
    DEVICE_AND_QUEUE.with(move |slot| *slot.borrow_mut() = Some(context));

    // Call the samples main.
    samples::samples_main(
        |number, _scale, path| {
            DEVICE_AND_QUEUE.with(|daq| {
                let mut guard = daq.borrow_mut();
                let context = guard.as_mut().unwrap();

                if number == 12 || number == 16 {
                    return Ok(());
                }

                // Get the picture.
                let picture = samples::get(number)?;
                let size = picture.size();

                // Create a texture to render into.
                let aligned_width = (size.width as u32 + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
                    & !(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1);
                let texture =
                    context
                        .device_and_queue()
                        .0
                        .create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            size: wgpu::Extent3d {
                                width: aligned_width,
                                height: size.height as u32,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::COPY_SRC,
                        });
                let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Create a buffer to copy the texture into.
                let buffer = context
                    .device_and_queue()
                    .0
                    .create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: (aligned_width * size.height as u32 * 4) as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });

                // Create the render context.
                let mut rc =
                    context.render_context(texture_view, aligned_width, size.height as u32);

                // Draw the picture.
                picture.draw(&mut rc)?;

                // Copy the texture into the buffer.
                let mut encoder = context
                    .device_and_queue()
                    .0
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(aligned_width * 4),
                            rows_per_image: Some(size.height as u32),
                        },
                    },
                    wgpu::Extent3d {
                        width: size.width as u32,
                        height: size.height as u32,
                        depth_or_array_layers: 1,
                    },
                );

                // Map the buffer and read the data.
                context
                    .device_and_queue()
                    .1
                    .submit(std::iter::once(encoder.finish()));
                let buffer_slice = buffer.slice(..);
                let (send, recv) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
                    res.unwrap();
                    send.send(()).unwrap();
                });
                context.device_and_queue().0.poll(wgpu::Maintain::Wait);
                recv.recv().unwrap();

                // Get the mapped range and read the data.
                let range = buffer_slice.get_mapped_range();
                let img = image::RgbaImage::from_raw(
                    size.width as u32,
                    size.height as u32,
                    range.to_vec(),
                )
                .unwrap();

                // Save the image.
                img.save(path).unwrap();

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
