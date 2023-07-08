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

//! Set up a window to render into with `piet-wgpu`.

use futures_lite::future;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use piet_wgpu::{RenderContext, WgpuContext};

pub(super) fn run(
    mut draw: impl FnMut(&mut RenderContext<'_, '_, '_>, u32, u32) + 'static,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: Default::default(),
    });

    let mut state = None;
    let format = wgpu::TextureFormat::Bgra8Unorm;
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: 0,
        height: 0,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![format],
    };
    let mut window_size = (640, 400);

    event_loop.run(move |ev, elwt, control_flow| {
        control_flow.set_poll();

        match ev {
            Event::Resumed => {
                let window = WindowBuilder::new()
                    .with_title("piet-wgpu basics")
                    .with_inner_size(winit::dpi::PhysicalSize::<u32>::from(window_size))
                    .build(elwt)
                    .expect("Failed to create window");

                let size = window.inner_size();

                let surface =
                    unsafe { instance.create_surface(&window) }.expect("Failed to create surface");

                let adaptor =
                    future::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        compatible_surface: Some(&surface),
                        ..Default::default()
                    }))
                    .expect("Failed to find an appropriate adapter");

                // Create the logical device and command queue
                let (device, queue) = future::block_on(adaptor.request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Device descriptor"),
                        features: wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER,
                        limits: wgpu::Limits::default(),
                    },
                    None,
                ))
                .expect("Failed to create device");

                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);

                let context = WgpuContext::new(&device, &queue, config.format, None, 1);

                state = Some((window, surface, context, device, queue));
            }

            Event::Suspended => {
                // Dump the state.
                state = None;
            }

            Event::RedrawEventsCleared => {
                if let Some((_, surface, context, device, queue)) = &mut state {
                    let frame = surface
                        .get_current_texture()
                        .expect("Failed to get texture view");

                    // Create the command encoder.
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.0,
                                            g: 0.0,
                                            b: 0.0,
                                            a: 1.0,
                                        }),
                                        store: true,
                                    },
                                })],
                                depth_stencil_attachment: None,
                            });

                        let mut piet = context.prepare(device, queue, window_size.0, window_size.1);
                        draw(&mut piet, window_size.0, window_size.1);

                        // Finish the render pass.
                        context.render(&mut render_pass);
                    }

                    // Submit the command buffer to the queue.
                    queue.submit(Some(encoder.finish()));

                    // Present the frame.
                    frame.present();
                }
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::Resized(size) => {
                    window_size = (size.width, size.height);
                    if let Some((_, surface, _, device, _)) = &state {
                        config.width = size.width;
                        config.height = size.height;
                        surface.configure(device, &config);
                    }
                }
                _ => {}
            },

            _ => {}
        }
    })
}
