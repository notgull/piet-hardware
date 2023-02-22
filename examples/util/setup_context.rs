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

// Easy module for setting up a context for the examples.
// Uses glutin on desktop platforms and WebGL on the web.

mod util {
    use piet_glow::RenderContext;

    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    pub(crate) fn with_renderer(
        mut f: impl FnMut(&mut RenderContext<'_, glow::Context>, u32, u32) + 'static,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use glutin::config::ConfigTemplateBuilder;
        use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
        use glutin::display::GetGlDisplay;
        use glutin::prelude::*;

        use glutin::surface::SwapInterval;
        use glutin_winit::{DisplayBuilder, GlWindow};

        use raw_window_handle::HasRawWindowHandle;

        use std::num::NonZeroU32;
        use std::time::{Duration, Instant};

        use winit::event::{Event, WindowEvent};
        use winit::event_loop::EventLoop;
        use winit::window::WindowBuilder;

        // Build an event loop.
        let event_loop = EventLoop::new();

        let make_window_builder = || {
            WindowBuilder::new()
                .with_title("piet-glow example")
                .with_transparent(true)
        };

        // Start building a window.
        let window = if cfg!(windows) {
            Some(make_window_builder())
        } else {
            None
        };

        // Use the window builder to start building a display.
        let display = DisplayBuilder::new().with_window_builder(window);

        // Look for a config that supports transparency and has a good sample count.
        let (mut window, gl_config) = display.build(
            &event_loop,
            ConfigTemplateBuilder::new()
                .with_alpha_size(8)
                .with_transparency(cfg!(target_vendor = "apple")),
            |configs| {
                configs
                    .max_by_key(|config| {
                        // Get the sample count.
                        let mut score = config.num_samples() as u32;

                        // Overwhelmingly prefer a config with transparency.
                        if config.supports_transparency().unwrap_or(false) {
                            score += 1_000;
                        }

                        score
                    })
                    .unwrap()
            },
        )?;

        // Try to build a several different contexts.
        let window_handle = window.as_ref().map(|w| w.raw_window_handle());
        let contexts = [
            ContextAttributesBuilder::new().build(window_handle),
            ContextAttributesBuilder::new()
                .with_context_api(ContextApi::Gles(None))
                .build(window_handle),
            ContextAttributesBuilder::new()
                .with_context_api(ContextApi::Gles(Some(Version::new(2, 0))))
                .build(window_handle),
        ];

        let display = gl_config.display();
        let gl_handler = (|| {
            // Try to build a context for each config.
            for context in &contexts {
                if let Ok(gl_context) = unsafe { display.create_context(&gl_config, context) } {
                    return Ok(gl_context);
                }
            }

            // If we couldn't build a context, return an error.
            Err(Box::<dyn std::error::Error>::from(
                "Could not create a context",
            ))
        })()?;

        // Run the event loop.
        let mut state = None;
        let mut renderer = None;
        let mut not_current_gl_context = Some(gl_handler);
        let mut current_size = None;
        let mut next_render = Instant::now() + Duration::from_millis(16);

        event_loop.run(move |event, window_target, control_flow| {
            control_flow.set_wait_until(next_render);
            match event {
                Event::Resumed => {
                    #[cfg(target_os = "android")]
                    println!("Android window available");

                    let window = window.take().unwrap_or_else(|| {
                        let window_builder = make_window_builder();
                        glutin_winit::finalize_window(window_target, window_builder, &gl_config)
                            .unwrap()
                    });

                    let attrs = window.build_surface_attributes(<_>::default());
                    let gl_surface = unsafe {
                        gl_config
                            .display()
                            .create_window_surface(&gl_config, &attrs)
                            .unwrap()
                    };

                    // Make it current.
                    let gl_context = not_current_gl_context
                        .take()
                        .unwrap()
                        .make_current(&gl_surface)
                        .unwrap();

                    // Set up the Glow context.
                    renderer.get_or_insert_with(|| {
                        let glow_context = unsafe {
                            glow::Context::from_loader_function_cstr(|s| {
                                display.get_proc_address(s) as *const _
                            })
                        };

                        // Wrap it up in a piet-glow context.
                        // # SAFETY: gl_context is current.
                        unsafe { piet_glow::GlContext::new(glow_context).unwrap() }
                    });

                    // Try setting vsync.
                    if let Err(res) = gl_surface.set_swap_interval(
                        &gl_context,
                        SwapInterval::Wait(NonZeroU32::new(1).unwrap()),
                    ) {
                        eprintln!("Error setting vsync: {res:?}");
                    }

                    assert!(state.replace((gl_context, gl_surface, window)).is_none());
                }
                Event::Suspended => {
                    // This event is only raised on Android, where the backing NativeWindow for a GL
                    // Surface can appear and disappear at any moment.
                    println!("Android window removed");

                    // Destroy the GL Surface and un-current the GL Context before ndk-glue releases
                    // the window back to the system.
                    let (gl_context, ..) = state.take().unwrap();
                    assert!(not_current_gl_context
                        .replace(gl_context.make_not_current().unwrap())
                        .is_none());
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(size) => {
                        if size.width != 0 && size.height != 0 {
                            // Some platforms like EGL require resizing GL surface to update the size
                            // Notable platforms here are Wayland and macOS, other don't require it
                            // and the function is no-op, but it's wise to resize it for portability
                            // reasons.
                            if let Some((gl_context, gl_surface, _)) = &state {
                                gl_surface.resize(
                                    gl_context,
                                    NonZeroU32::new(size.width).unwrap(),
                                    NonZeroU32::new(size.height).unwrap(),
                                );
                                current_size = Some(size);
                            }
                        }
                    }
                    WindowEvent::CloseRequested => {
                        control_flow.set_exit();
                    }
                    _ => (),
                },
                Event::RedrawEventsCleared => {
                    if let Some((gl_context, gl_surface, window)) = &state {
                        let renderer = renderer.as_mut().unwrap();

                        // Run the renderer
                        // SAFETY: Context is current
                        let size = current_size.unwrap_or_else(|| window.inner_size());
                        let mut context =
                            unsafe { RenderContext::new(renderer, size.width, size.height) };
                        f(&mut context, size.width, size.height);

                        window.request_redraw();

                        gl_surface.swap_buffers(gl_context).unwrap();
                        next_render += Duration::from_millis(17);
                    }
                }
                _ => (),
            }
        })
    }
}
