// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-gpu`.
//
// `piet-gpu` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-gpu` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-gpu`. If not, see <https://www.gnu.org/licenses/> or
// <https://www.mozilla.org/en-US/MPL/2.0/>.

//! An example that uses the `gl` crate to render to a `winit` window.
//!
//! This uses `glutin` crate to set up a GL context, `winit` to create a window, and the `gl`
//! crate to make GL calls.
//!
//! This example exists mostly to give an example of how a `GpuContext` can be implemented.
//! If you actually want to use `piet` with OpenGL, consider the `piet-glow` crate.

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;

use glutin_winit::{DisplayBuilder, GlWindow};

use piet::kurbo::{Affine, BezPath, Point, Rect};
use piet::RenderContext as _;

use raw_window_handle::HasRawWindowHandle;

use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use std::cell::Cell;
use std::ffi::CString;
use std::fmt;
use std::num::NonZeroU32;
use std::time::{Duration, Instant};

const TEST_IMAGE: &[u8] = include_bytes!("test-image.png");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Create the winit event loop.
    let event_loop = EventLoop::new();

    let mut size = PhysicalSize::new(800, 600);
    let make_window_builder = move || {
        WindowBuilder::new()
            .with_title("piet-gpu example")
            .with_transparent(true)
            .with_inner_size(size)
    };

    // If we're on Windows, start with the window.
    let window = if cfg!(windows) {
        Some(make_window_builder())
    } else {
        None
    };

    // Start building an OpenGL display.
    let display = DisplayBuilder::new().with_window_builder(window);

    // Look for a config that supports transparency and has a good sample count.
    let (mut window, gl_config) = display.build(
        &event_loop,
        ConfigTemplateBuilder::new().with_alpha_size(8),
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

    // Set up data for the window.
    let framerate = Duration::from_millis({
        let framerate = 1.0 / 60.0;
        (framerate * 1000.0) as u64
    });
    let mut next_frame = Instant::now() + framerate;
    let mut state = None;
    let mut renderer = None;
    let mut not_current_gl_context = Some(gl_handler);

    // Load the image.
    let (image_width, image_height, image_data) = {
        let image = image::load_from_memory(TEST_IMAGE).unwrap();
        let image = image.to_rgba8();

        let (width, height) = image.dimensions();
        let data = image.into_raw();
        (width, height, data)
    };

    // Drawing data.
    let star = generate_five_pointed_star((0.0, 0.0).into(), 75.0, 150.0);
    let mut solid_red = None;
    let mut outline = None;
    let mut image = None;
    let mut tick = 0;

    // Draw the window.
    let mut draw = move |ctx: &mut piet_gpu::RenderContext<'_, GlContext>| {
        ctx.clear(None, piet::Color::AQUA);

        let outline = outline.get_or_insert_with(|| ctx.solid_brush(piet::Color::BLACK));

        // Draw a rotating star.
        ctx.with_save(|ctx| {
            ctx.transform({
                let rotation = Affine::rotate((tick as f64) * 0.02);
                let translation = Affine::translate((200.0, 200.0));

                translation * rotation
            });

            let solid_red = solid_red
                .get_or_insert_with(|| ctx.solid_brush(piet::Color::rgb8(0x39, 0xe5, 0x8a)));

            ctx.fill(&star, solid_red);
            ctx.stroke(&star, outline, 5.0);

            Ok(())
        })
        .unwrap();

        // Draw a moving image.
        {
            let cos_curve = |x: f64, amp: f64, freq: f64| {
                let x = x * std::f64::consts::PI * freq;
                x.cos() * amp
            };
            let sin_curve = |x: f64, amp: f64, freq: f64| {
                let x = x * std::f64::consts::PI * freq;
                x.sin() * amp
            };

            let posn_shift_x = cos_curve(tick as f64, 50.0, 0.01);
            let posn_shift_y = sin_curve(tick as f64, 50.0, 0.01);
            let posn_x = 450.0 + posn_shift_x;
            let posn_y = 150.0 + posn_shift_y;

            let size_shift_x = cos_curve(tick as f64, 25.0, 0.02);
            let size_shift_y = sin_curve(tick as f64, 25.0, 0.02);
            let size_x = 100.0 + size_shift_x;
            let size_y = 100.0 + size_shift_y;

            let target_rect = Rect::new(posn_x, posn_y, posn_x + size_x, posn_y + size_y);

            let image_handle = image.get_or_insert_with(|| {
                ctx.make_image(
                    image_width as usize,
                    image_height as usize,
                    &image_data,
                    piet::ImageFormat::RgbaSeparate,
                )
                .unwrap()
            });

            ctx.draw_image(image_handle, target_rect, piet::InterpolationMode::Bilinear);

            // Also draw a subset of the image.
            let source_rect = Rect::new(
                25.0 + posn_shift_x,
                25.0 + posn_shift_y,
                100.0 + posn_shift_x,
                100.0 + posn_shift_y,
            );

            let target_rect = Rect::from_origin_size((625.0, 50.0), (100.0, 100.0));

            ctx.draw_image_area(
                image_handle,
                source_rect,
                target_rect,
                piet::InterpolationMode::Bilinear,
            );
            ctx.stroke(target_rect, outline, 5.0);
        }

        tick += 1;
        ctx.finish().unwrap();
        ctx.status()
    };

    event_loop.run(move |event, target, control_flow| {
        control_flow.set_wait_until(next_frame);

        match event {
            Event::Resumed => {
                // We can now create windows.
                let window = window.take().unwrap_or_else(|| {
                    let window_builder = make_window_builder();
                    glutin_winit::finalize_window(target, window_builder, &gl_config).unwrap()
                });

                let attrs = window.build_surface_attributes(Default::default());
                let surface = unsafe {
                    gl_config
                        .display()
                        .create_window_surface(&gl_config, &attrs)
                        .unwrap()
                };

                // Make the context current.
                let gl_context = not_current_gl_context
                    .take()
                    .unwrap()
                    .make_current(&surface)
                    .unwrap();

                unsafe {
                    renderer
                        .get_or_insert_with(|| {
                            // Register the GL pointers if we can.
                            {
                                gl::load_with(|symbol| {
                                    let symbol_cstr = CString::new(symbol).unwrap();
                                    gl_config.display().get_proc_address(symbol_cstr.as_c_str())
                                });

                                piet_gpu::Source::new(GlContext::new()).unwrap()
                            }
                        })
                        .context()
                        .set_context();
                }

                state = Some((surface, window, gl_context));
            }

            Event::Suspended => {
                // Destroy the window.
                if let Some((.., context)) = state.take() {
                    not_current_gl_context = Some(context.make_not_current().unwrap());
                }

                if let Some(renderer) = &renderer {
                    renderer.context().unset_context();
                }
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::Resized(new_size) => {
                    size = new_size;

                    if let Some((surface, _, context)) = &state {
                        surface.resize(
                            context,
                            NonZeroU32::new(size.width).unwrap(),
                            NonZeroU32::new(size.height).unwrap(),
                        );
                    }
                }
                _ => {}
            },

            Event::RedrawEventsCleared => {
                if let (Some((surface, _, context)), Some(renderer)) = (&state, &mut renderer) {
                    // Create the render context.
                    let mut render_context = renderer.render_context(size.width, size.height);

                    // Perform drawing.
                    draw(&mut render_context).unwrap();

                    // Swap buffers.
                    surface.swap_buffers(context).unwrap();
                }

                // Schedule the next frame.
                next_frame += framerate;
            }

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

/// The global OpenGL context.
struct GlContext {
    /// Whether we have a context installed.
    has_context: Cell<bool>,

    /// A program for rendering.
    render_program: gl::types::GLuint,

    // Uniform locations.
    u_transform: gl::types::GLint,
    viewport_size: gl::types::GLint,
    tex: gl::types::GLint,
    mask: gl::types::GLint,
}

#[derive(Clone)]
struct GlVertexBuffer {
    vbo: gl::types::GLuint,
    ebo: gl::types::GLuint,
    vao: gl::types::GLuint,
    num_indices: Cell<usize>,
}

impl GlContext {
    fn assert_context(&self) {
        if !self.has_context.get() {
            panic!("No GL context installed");
        }
    }

    // SAFETY: Context must be current.
    unsafe fn new() -> Self {
        // Create the program.
        let program = unsafe {
            let vertex_shader = Self::compile_shader(gl::VERTEX_SHADER, VERTEX_SHADER).unwrap();

            let fragment_shader =
                Self::compile_shader(gl::FRAGMENT_SHADER, FRAGMENT_SHADER).unwrap();

            let program = gl::CreateProgram();
            gl::AttachShader(program, vertex_shader);
            gl::AttachShader(program, fragment_shader);
            gl::LinkProgram(program);

            let mut success = gl::FALSE as gl::types::GLint;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);

            if success == gl::FALSE as gl::types::GLint {
                let mut len = 0;
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);

                let mut buf = Vec::with_capacity(len as usize);
                gl::GetProgramInfoLog(program, len, std::ptr::null_mut(), buf.as_mut_ptr() as _);
                buf.set_len((len as usize) - 1);
                panic!(
                    "Could not link program: {}",
                    std::str::from_utf8(&buf).unwrap()
                );
            }

            gl::DetachShader(program, vertex_shader);
            gl::DetachShader(program, fragment_shader);
            gl::DeleteShader(vertex_shader);
            gl::DeleteShader(fragment_shader);

            program
        };

        // Enable wireframe mode.
        //unsafe {
        //    gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
        //}

        unsafe {
            extern "system" fn debug_callback(
                source: u32,
                ty: u32,
                id: u32,
                severity: u32,
                msg_len: i32,
                msg: *const i8,
                _user_param: *mut std::ffi::c_void,
            ) {
                let source = match source {
                    gl::DEBUG_SOURCE_API => "API",
                    gl::DEBUG_SOURCE_WINDOW_SYSTEM => "Window System",
                    gl::DEBUG_SOURCE_SHADER_COMPILER => "Shader Compiler",
                    gl::DEBUG_SOURCE_THIRD_PARTY => "Third Party",
                    gl::DEBUG_SOURCE_APPLICATION => "Application",
                    gl::DEBUG_SOURCE_OTHER => "Other",
                    _ => "Unknown",
                };

                let ty = match ty {
                    gl::DEBUG_TYPE_ERROR => "Error",
                    gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR => "Deprecated Behavior",
                    gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR => "Undefined Behavior",
                    gl::DEBUG_TYPE_PORTABILITY => "Portability",
                    gl::DEBUG_TYPE_PERFORMANCE => "Performance",
                    gl::DEBUG_TYPE_MARKER => "Marker",
                    gl::DEBUG_TYPE_OTHER => "Other",
                    _ => "Unknown",
                };

                let message = {
                    let slice =
                        unsafe { std::slice::from_raw_parts(msg as *const u8, msg_len as usize) };
                    std::str::from_utf8(slice).unwrap()
                };

                match severity {
                    gl::DEBUG_SEVERITY_HIGH => {
                        log::error!("{ty}-{id} ({source}): {message}");
                    }
                    gl::DEBUG_SEVERITY_MEDIUM => {
                        log::warn!("{ty}-{id} ({source}): {message}");
                    }
                    gl::DEBUG_SEVERITY_LOW => {
                        log::info!("{ty}-{id} ({source}): {message}");
                    }
                    gl::DEBUG_SEVERITY_NOTIFICATION => {
                        log::debug!("{ty}-{id} ({source}): {message}");
                    }
                    _ => (),
                };
            }

            // Set up a debug callback.
            gl::Enable(gl::DEBUG_OUTPUT);

            gl::DebugMessageCallback(Some(debug_callback), std::ptr::null());
        }

        // Get the uniform locations.
        let u_transform = unsafe {
            let name = CString::new("transform").unwrap();
            gl::GetUniformLocation(program, name.as_ptr())
        };

        let viewport_size = unsafe {
            let name = CString::new("viewportSize").unwrap();
            gl::GetUniformLocation(program, name.as_ptr())
        };

        let tex = unsafe {
            let name = CString::new("tex").unwrap();
            gl::GetUniformLocation(program, name.as_ptr())
        };

        let mask = unsafe {
            let name = CString::new("mask").unwrap();
            gl::GetUniformLocation(program, name.as_ptr())
        };

        gl_error();

        Self {
            has_context: Cell::new(true),
            render_program: program,
            u_transform,
            viewport_size,
            tex,
            mask,
        }
    }

    fn unset_context(&self) {
        self.has_context.set(false);
    }

    unsafe fn set_context(&self) {
        self.has_context.set(true);
    }

    unsafe fn compile_shader(
        shader_type: gl::types::GLenum,
        source: &str,
    ) -> Result<gl::types::GLuint, GlError> {
        let shader = gl::CreateShader(shader_type);
        let source = CString::new(source).unwrap();
        gl::ShaderSource(shader, 1, &source.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);

        let mut success = gl::FALSE as gl::types::GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);

        if success == gl::FALSE as gl::types::GLint {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);

            let mut buf = Vec::with_capacity(len as usize);
            gl::GetShaderInfoLog(
                shader,
                len,
                std::ptr::null_mut(),
                buf.as_mut_ptr() as *mut gl::types::GLchar,
            );
            buf.set_len((len as usize) - 1);

            return Err(GlError(format!(
                "Shader compilation failed: {}",
                std::str::from_utf8(&buf).unwrap()
            )));
        }

        Ok(shader)
    }
}

#[derive(Debug)]
struct GlError(String);

impl fmt::Display for GlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GL error: {}", self.0)
    }
}

impl std::error::Error for GlError {}

impl piet_gpu::GpuContext for GlContext {
    type Error = GlError;
    type Texture = gl::types::GLuint;
    type VertexBuffer = GlVertexBuffer;

    fn clear(&self, color: piet::Color) {
        self.assert_context();
        let (r, g, b, a) = color.as_rgba();

        unsafe {
            gl::ClearColor(r as f32, g as f32, b as f32, a as f32);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl_error();
        }
    }

    fn flush(&self) -> Result<(), Self::Error> {
        self.assert_context();

        unsafe {
            gl::Flush();
            gl_error();
            Ok(())
        }
    }

    fn create_texture(
        &self,
        interpolation: piet::InterpolationMode,
        repeat: piet_gpu::RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        self.assert_context();

        unsafe {
            let mut texture = 0;
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);

            let (min_filter, mag_filter) = match interpolation {
                piet::InterpolationMode::NearestNeighbor => (gl::NEAREST, gl::NEAREST),
                piet::InterpolationMode::Bilinear => (gl::LINEAR, gl::LINEAR),
            };

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, min_filter as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, mag_filter as _);

            let (wrap_s, wrap_t) = match repeat {
                piet_gpu::RepeatStrategy::Color(clr) => {
                    let (r, g, b, a) = clr.as_rgba();
                    gl::TexParameterfv(
                        gl::TEXTURE_2D,
                        gl::TEXTURE_BORDER_COLOR,
                        [r as f32, g as f32, b as f32, a as f32].as_ptr(),
                    );

                    (gl::CLAMP_TO_EDGE, gl::CLAMP_TO_EDGE)
                }
                piet_gpu::RepeatStrategy::Repeat => (gl::REPEAT, gl::REPEAT),
                _ => panic!("unsupported repeat strategy"),
            };

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, wrap_s as _);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, wrap_t as _);

            Ok(texture as _)
        }
    }

    fn delete_texture(&self, texture: Self::Texture) {
        self.assert_context();

        unsafe {
            gl::DeleteTextures(1, &texture);
        }
    }

    fn write_texture(
        &self,
        texture: &Self::Texture,
        size: (u32, u32),
        format: piet::ImageFormat,
        data: Option<&[u8]>,
    ) {
        self.assert_context();

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, *texture);

            let (internal_format, format, ty) = match format {
                piet::ImageFormat::RgbaSeparate => (gl::RGBA8, gl::RGBA, gl::UNSIGNED_BYTE),
                _ => panic!("unsupported image format"),
            };

            let (width, height) = size;
            let data_ptr = data
                .map(|data| data.as_ptr() as *const _)
                .unwrap_or(std::ptr::null());

            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                internal_format as _,
                width as _,
                height as _,
                0,
                format,
                ty,
                data_ptr,
            );
        }
    }

    fn write_subtexture(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet::ImageFormat,
        data: &[u8],
    ) {
        self.assert_context();

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, *texture);

            let (format, ty) = match format {
                piet::ImageFormat::RgbaSeparate => (gl::RGBA, gl::UNSIGNED_BYTE),
                _ => panic!("unsupported image format"),
            };

            let (width, height) = size;
            let (x, y) = offset;

            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                x as _,
                y as _,
                width as _,
                height as _,
                format,
                ty,
                data.as_ptr() as *const _,
            );
        }
    }

    fn set_texture_interpolation(
        &self,
        texture: &Self::Texture,
        interpolation: piet::InterpolationMode,
    ) {
        self.assert_context();

        let mode = match interpolation {
            piet::InterpolationMode::Bilinear => gl::LINEAR,
            piet::InterpolationMode::NearestNeighbor => gl::NEAREST,
        };

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, *texture);
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_MAG_FILTER,
                mode as gl::types::GLint,
            );
            gl::TexParameteri(
                gl::TEXTURE_2D,
                gl::TEXTURE_MIN_FILTER,
                mode as gl::types::GLint,
            );
            //gl::BindTexture(gl::TEXTURE_2D, 0);
        }
    }

    fn max_texture_size(&self) -> (u32, u32) {
        self.assert_context();

        unsafe {
            let mut side = 0;
            gl::GetIntegerv(gl::MAX_TEXTURE_SIZE, &mut side);
            (side as u32, side as u32)
        }
    }

    fn create_vertex_buffer(&self) -> Result<Self::VertexBuffer, Self::Error> {
        self.assert_context();

        unsafe {
            let mut buffers = [0; 2];
            gl::GenBuffers(2, buffers.as_mut_ptr());
            let [vbo, ebo] = buffers;

            // Set up the vertex array object.
            let mut vao = 0;
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
            gl::UseProgram(self.render_program);

            let stride = std::mem::size_of::<piet_gpu::Vertex>() as _;

            // Set up the layout:
            // - vec2 of floats for aPos
            // - vec2 of floats for aTexCoord
            // - vec4 of unsigned bytes for aColor
            let apos_name = CString::new("aPos").unwrap();
            let apos_coord = gl::GetAttribLocation(self.render_program, apos_name.as_ptr());
            gl::EnableVertexAttribArray(apos_coord as _);
            gl::VertexAttribPointer(
                apos_coord as _,
                2,
                gl::FLOAT,
                gl::FALSE,
                stride,
                bytemuck::offset_of!(piet_gpu::Vertex, pos) as *const _,
            );

            let atex_name = CString::new("aTexCoord").unwrap();
            let atex_coord = gl::GetAttribLocation(self.render_program, atex_name.as_ptr() as _);
            gl::EnableVertexAttribArray(atex_coord as _);
            gl::VertexAttribPointer(
                atex_coord as _,
                2,
                gl::FLOAT,
                gl::FALSE,
                stride,
                bytemuck::offset_of!(piet_gpu::Vertex, uv) as *const _,
            );

            let acolor_name = CString::new("aColor").unwrap();
            let acolor_coord = gl::GetAttribLocation(self.render_program, acolor_name.as_ptr());
            gl::EnableVertexAttribArray(acolor_coord as _);
            gl::VertexAttribPointer(
                acolor_coord as _,
                4,
                gl::UNSIGNED_BYTE,
                gl::FALSE,
                stride,
                bytemuck::offset_of!(piet_gpu::Vertex, color) as *const _,
            );

            // Unbind the vertex array object.
            //gl::BindVertexArray(0);
            //gl::BindBuffer(gl::ARRAY_BUFFER, 0);

            Ok(GlVertexBuffer {
                vao,
                vbo,
                ebo,
                num_indices: Cell::new(0),
            })
        }
    }

    unsafe fn write_vertices(
        &self,
        buffer: &Self::VertexBuffer,
        vertices: &[piet_gpu::Vertex],
        indices: &[u32],
    ) {
        self.assert_context();

        unsafe {
            //gl::BindBuffer(gl::ARRAY_BUFFER, buffer.vbo);
            //gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, buffer.ebo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * std::mem::size_of::<piet_gpu::Vertex>()) as _,
                vertices.as_ptr() as *const _,
                gl::DYNAMIC_DRAW,
            );
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (indices.len() * std::mem::size_of::<u32>()) as _,
                indices.as_ptr() as *const _,
                gl::DYNAMIC_DRAW,
            );
            gl_error();
            buffer.num_indices.set(indices.len() as _);
        }
    }

    fn delete_vertex_buffer(&self, buffer: Self::VertexBuffer) {
        self.assert_context();

        unsafe {
            let buffers = [buffer.vbo, buffer.ebo];
            gl::DeleteBuffers(2, buffers.as_ptr());
        }
    }

    fn push_buffers(
        &self,
        vertex_buffer: &Self::VertexBuffer,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error> {
        unsafe {
            // Use our program.
            gl::UseProgram(self.render_program);

            // Set the viewport size.
            let (width, height) = size;
            gl::Viewport(0, 0, width as i32, height as i32);
            gl::Uniform2f(self.viewport_size, width as f32, height as f32);

            // Set the transform.
            let [a, b, c, d, e, f] = transform.as_coeffs();
            let transform = [
                a as f32, b as f32, 0.0, c as f32, d as f32, 0.0, e as f32, f as f32, 1.0,
            ];
            gl::UniformMatrix3fv(self.u_transform, 1, gl::FALSE, transform.as_ptr());

            // Set the texture.
            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, *current_texture);
            gl::Uniform1i(self.tex, 1);

            // Set the mask texture.
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, *mask_texture);
            gl::Uniform1i(self.mask, 0);

            // Set the blend mode.
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            // Set vertex attributes.
            gl::BindVertexArray(vertex_buffer.vao);

            // Set buffers.
            gl::BindBuffer(gl::ARRAY_BUFFER, vertex_buffer.vbo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, vertex_buffer.ebo);

            // Draw.
            gl::DrawElements(
                gl::TRIANGLES,
                vertex_buffer.num_indices.get() as i32,
                gl::UNSIGNED_INT,
                std::ptr::null(),
            );

            // Unbind everything.
            //gl::BindVertexArray(0);
            //gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            //gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
            //gl::BindTexture(gl::TEXTURE_2D, 0);
            //gl::UseProgram(0);
        }

        Ok(())
    }
}

fn gl_error() {
    let err = unsafe { gl::GetError() };

    if err != gl::NO_ERROR {
        let error_str = match err {
            gl::INVALID_ENUM => "GL_INVALID_ENUM",
            gl::INVALID_VALUE => "GL_INVALID_VALUE",
            gl::INVALID_OPERATION => "GL_INVALID_OPERATION",
            gl::STACK_OVERFLOW => "GL_STACK_OVERFLOW",
            gl::STACK_UNDERFLOW => "GL_STACK_UNDERFLOW",
            gl::OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
            gl::INVALID_FRAMEBUFFER_OPERATION => "GL_INVALID_FRAMEBUFFER_OPERATION",
            gl::CONTEXT_LOST => "GL_CONTEXT_LOST",
            _ => "Unknown GL error",
        };

        log::error!("GL error: {}", error_str)
    }
}

const VERTEX_SHADER: &str = "
#version 330 core

in vec2 aPos;
in vec2 aTexCoord;
in vec4 aColor;

out vec4 rgbaColor;
out vec2 fTexCoord;
out vec2 fMaskCoord;

uniform mat3 transform;
uniform vec2 viewportSize;

void main() {
    // Transform the vertex position.
    vec3 pos = transform * vec3(aPos, 1.0);
    pos /= pos.z;

    // Transform to screen-space coordinates.
    gl_Position = vec4(
        (2.0 * pos.x / viewportSize.x) - 1.0,
        1.0 - (2.0 * pos.y / viewportSize.y),
        0.0,
        1.0
    );

    // Transform to mask-space coordinates.
    fMaskCoord = vec2(
        pos.x / viewportSize.x,
        1.0 - (pos.y / viewportSize.y)
    );

    rgbaColor = aColor / 255.0;
    fTexCoord = aTexCoord;
}
";

const FRAGMENT_SHADER: &str = "
#version 330 core

in vec4 rgbaColor;
in vec2 fTexCoord;
in vec2 fMaskCoord;

uniform sampler2D tex;
uniform sampler2D mask;

void main() {
    vec4 textureColor = texture2D(tex, fTexCoord);
    vec4 mainColor = rgbaColor * textureColor;

    vec4 maskColor = texture2D(mask, fMaskCoord);
    vec4 finalColor = mainColor * maskColor;

    gl_FragColor = finalColor;
}
";
