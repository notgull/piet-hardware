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

//! The `piet-glow` basics.rs example, but with piet-wgpu.

use futures_lite::future;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use piet_hardware::piet::{self, RenderContext as _};
use piet_wgpu::{RenderContext, WgpuContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: Default::default(),
    });

    let mut draw = |rc: &mut RenderContext<'_, _>| {
        rc.clear(None, piet::Color::RED);
        rc.finish().unwrap();
    };

    let mut state = None;
    let mut window_size = (0, 0);

    event_loop.run(move |ev, elwt, control_flow| {
        control_flow.set_poll();

        match ev {
            Event::Resumed => {
                let window = WindowBuilder::new()
                    .with_title("piet-wgpu basics")
                    .build(elwt)
                    .expect("Failed to create window");

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

                let context =
                    WgpuContext::new((device, queue), wgpu::TextureFormat::Rgba8Unorm, 100)
                        .expect("Failed to create WgpuContext");

                state = Some((window, surface, context));
            }

            Event::Suspended => {
                // Dump the state.
                state = None;
            }

            Event::RedrawEventsCleared => {
                if let Some((_, surface, context)) = &mut state {
                    let frame = surface
                        .get_current_texture()
                        .expect("Failed to get texture view");

                    // Draw using piet.
                    draw(
                        &mut context.render_context(
                            frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                            window_size.0,
                            window_size.1,
                        ),
                    );

                    // Present the frame.
                    frame.present();
                }
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::Resized(size) => {
                    window_size = (size.width, size.height);
                }
                _ => {}
            },

            _ => {}
        }
    });

    Ok(())
}
