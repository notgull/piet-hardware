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
        tracing_subscriber::fmt::init();

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
                .with_alpha_size(8),
            |configs| {
                configs
                    .reduce(|accum, config| {
                    let transparency_check = config.supports_transparency().unwrap_or(false)
                        & !accum.supports_transparency().unwrap_or(false);

                    if transparency_check || config.num_samples() > accum.num_samples() {
                        config
                    } else {
                        accum
                    }
                }).unwrap()
            },
        )?;

        println!("Config: {:?}", &gl_config);
        println!("Color Buffer Type: {:?}", gl_config.color_buffer_type());
        println!("Float Pixels: {:?}", gl_config.float_pixels());
        println!("Alpha Size: {:?}", gl_config.alpha_size());
        println!("Depth Size: {:?}", gl_config.depth_size());
        println!("Stencil Size: {:?}", gl_config.stencil_size());
        println!("Samples: {:?}", gl_config.num_samples());
        println!("SRGB Capable: {:?}", gl_config.srgb_capable());
        println!("Supports Transparency: {:?}", gl_config.supports_transparency());
        println!("Hardware Accelerated: {:?}", gl_config.hardware_accelerated());
        println!("Config Surface Types: {:?}", gl_config.config_surface_types());
        println!("Api: {:?}", gl_config.api());

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

                        unsafe {
                            use glow::HasContext;

                            glow_context.enable(glow::DEBUG_OUTPUT);
                            glow_context.debug_message_callback(
                                debug_message_callback 
                            );
                        }

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
                            unsafe { renderer.render_context(size.width, size.height) };
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

    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    fn debug_message_callback(
        source: u32,
        ty: u32,
        id: u32,
        severity: u32,
        message: &str
    ) {
        let source = match source {
            glow::DEBUG_SOURCE_API => "API",
            glow::DEBUG_SOURCE_WINDOW_SYSTEM => "Window System",
            glow::DEBUG_SOURCE_SHADER_COMPILER => "Shader Compiler",
            glow::DEBUG_SOURCE_THIRD_PARTY => "Third Party",
            glow::DEBUG_SOURCE_APPLICATION => "Application",
            glow::DEBUG_SOURCE_OTHER => "Other",
            _ => "Unknown",
        };

        let ty = match ty {
            glow::DEBUG_TYPE_ERROR => "Error",
            glow::DEBUG_TYPE_DEPRECATED_BEHAVIOR => "Deprecated Behavior",
            glow::DEBUG_TYPE_UNDEFINED_BEHAVIOR => "Undefined Behavior",
            glow::DEBUG_TYPE_PORTABILITY => "Portability",
            glow::DEBUG_TYPE_PERFORMANCE => "Performance",
            glow::DEBUG_TYPE_MARKER => "Marker",
            glow::DEBUG_TYPE_OTHER => "Other",
            _ => "Unknown",
        };

        match severity {
            glow::DEBUG_SEVERITY_HIGH => {
                tracing::error!("{ty}-{id} ({source}): {message}");
            }
            glow::DEBUG_SEVERITY_MEDIUM => {
                tracing::warn!("{ty}-{id} ({source}): {message}");
            },
            glow::DEBUG_SEVERITY_LOW => {
                tracing::info!("{ty}-{id} ({source}): {message}");
            },
            glow::DEBUG_SEVERITY_NOTIFICATION => {
                tracing::debug!("{ty}-{id} ({source}): {message}");
            },
            _ => (),
        };
    }

    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    pub(crate) fn with_renderer(
        mut f: impl FnMut(&mut RenderContext<'_, glow::Context>, u32, u32) + 'static,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen::closure::Closure;

        use std::rc::Rc;
        use std::cell::RefCell;

        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .create_element("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        let webgl2_context = canvas
            .get_context("webgl2")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::WebGl2RenderingContext>()
            .unwrap();
        let gl = glow::Context::from_webgl2_context(webgl2_context);

        // Add the canvas to the DOM.
        web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .body()
            .unwrap()
            .append_child(&canvas)
            .unwrap();

        let mut renderer = unsafe {
            piet_glow::GlContext::new(gl).unwrap()
        };

        let timeout_cb = Rc::new(RefCell::new(None));
        let animate_cb: Rc<RefCell<Option<Closure<_>>>> = Rc::new(RefCell::new(None));

        // Run the callback every 1/60th of a second.
        let request_anim_frame = {
            let animate_cb = animate_cb.clone();
            move || {
                web_sys::window().unwrap()
                    .request_animation_frame(
                        animate_cb
                            .borrow()
                            .as_ref()
                            .unwrap()
                            .as_ref()
                            .unchecked_ref()
                    )
                    .unwrap()
            }
        };
        *timeout_cb.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            request_anim_frame();
        }) as Box<dyn FnMut()>));

        let draw_frame = {
            let timeout_cb = timeout_cb.clone();
            move || {
                let size = (canvas.width(), canvas.height());
                let mut context = unsafe {
                    renderer.render_context(size.0, size.1)
                };
                f(&mut context, size.0, size.1);
                web_sys::window()
                    .unwrap()
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        timeout_cb.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
                        1000
                    )
                    .unwrap();
            }
        };
        let draw_frame = Rc::new(RefCell::new(draw_frame));
        *animate_cb.borrow_mut() = Some(Closure::wrap(Box::new({
            let draw_frame = draw_frame.clone();
            move || {
                (*draw_frame.borrow_mut())();
            }
        }) as Box<dyn FnMut()>));

        // Start the animation.
        (draw_frame.borrow_mut())();

        // Throw an exception to stop the program.
        wasm_bindgen::throw_str("Program exited.")
    }
}
