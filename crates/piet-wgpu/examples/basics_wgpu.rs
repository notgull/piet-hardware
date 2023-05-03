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
use std::rc::Rc;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use piet_hardware::piet::kurbo::{Affine, BezPath, Point};
use piet_hardware::piet::{self, RenderContext as _};
use piet_wgpu::{RenderContext, WgpuContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let event_loop = EventLoop::new();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: Default::default(),
    });

    // A path representing a star.
    let star = generate_five_pointed_star(Point::new(0.0, 0.0), 75.0, 150.0);
    let mut tick = 0;

    // Drawing function.
    let mut solid_red = None;
    //let mut outline = None;
    let mut draw = move |rc: &mut RenderContext<'_, _>| {
        rc.clear(None, piet::Color::rgb8(0x87, 0xce, 0xeb));

        let red_star = {
            let rot = (tick % 360) as f64 / 180.0 * std::f64::consts::PI;
            let transform = Affine::translate((200.0, 200.0)) * Affine::rotate(rot);
            transform * (&star)
        };

        // Draw a solid red using the path.
        let solid_red = solid_red.get_or_insert_with(|| rc.solid_brush(piet::Color::OLIVE));
        rc.fill(&red_star, solid_red);

        // Draw a black outline using the path.
        //let outline = outline.get_or_insert_with(|| rc.solid_brush(piet::Color::BLACK));
        //rc.stroke(&red_star, outline, 5.0);

        // Draw a solid red star with a black outline.
        tick += 1;
        rc.finish().unwrap();
    };

    let mut state = None;
    let format = wgpu::TextureFormat::Bgra8Unorm;
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
        format,
        width: 0,
        height: 0,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![format],
    };
    let mut window_size = (0, 0);

    event_loop.run(move |ev, elwt, control_flow| {
        control_flow.set_poll();

        match ev {
            Event::Resumed => {
                let window = WindowBuilder::new()
                    .with_title("piet-wgpu basics")
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
                let device = Rc::new(device);

                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);

                let context = WgpuContext::new((device.clone(), queue), format, 1)
                    .expect("Failed to create WgpuContext");

                state = Some((window, surface, context, device));
            }

            Event::Suspended => {
                // Dump the state.
                state = None;
            }

            Event::RedrawEventsCleared => {
                if let Some((_, surface, context, _)) = &mut state {
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
                    if let Some((_, surface, _, device)) = &state {
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

fn generate_five_pointed_star(center: Point, inner_radius: f64, outer_radius: f64) -> BezPath {
    let point_from_polar = |radius: f64, angle: f64| {
        let x = center.x + radius * angle.cos();
        let y = center.y + radius * angle.sin();
        Point::new(x, y)
    };

    let one_fifth_circle = std::f64::consts::PI * 2.0 / 5.0;

    let outer_points = (0..5).map(|i| point_from_polar(outer_radius, one_fifth_circle * i as f64));
    let inner_points = (0..5).map(|i| {
        point_from_polar(
            inner_radius,
            one_fifth_circle * i as f64 + one_fifth_circle / 2.0,
        )
    });
    let mut points = outer_points.zip(inner_points).flat_map(|(a, b)| [a, b]);

    // Set up the path.
    let mut path = BezPath::new();
    path.move_to(points.next().unwrap());

    // Add the points to the path.
    for point in points {
        path.line_to(point);
    }

    // Close the path.
    path.close_path();
    path
}
