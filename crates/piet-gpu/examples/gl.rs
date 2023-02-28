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

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;

use glutin_winit::{DisplayBuilder, GlWindow};

use piet_gpu::BufferType;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the winit event loop.
    let event_loop = EventLoop::new();

    let mut size = PhysicalSize::new(600, 400);
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

                    if let Some(renderer) = &renderer {
                        renderer.context().unset_context();
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

fn draw(ctx: &mut impl piet::RenderContext) -> Result<(), piet::Error> {
    ctx.clear(None, piet::Color::AQUA);

    ctx.finish().unwrap();
    ctx.status()
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

        // Get the uniform locations.
        let u_transform = unsafe {
            let name = CString::new("u_transform").unwrap();
            gl::GetUniformLocation(program, name.as_ptr())
        };

        let viewport_size = unsafe {
            let name = CString::new("viewport_size").unwrap();
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

        Self {
            has_context: Cell::new(true),
            render_program: program,
            u_transform,
            viewport_size,
            tex,
            mask
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
        source: &[u8],
    ) -> Result<gl::types::GLuint, GlError> {
        let shader = gl::CreateShader(shader_type);
        gl::ShaderSource(shader, 1, source.as_ptr().cast(), source.len() as _);
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
    type Buffer = gl::types::GLuint;
    type Error = GlError;
    type Texture = gl::types::GLuint;
    type VertexArray = gl::types::GLuint;

    fn clear(&self, color: piet::Color) {
        self.assert_context();
        let (r, g, b, a) = color.as_rgba();

        unsafe {
            gl::ClearColor(r as f32, g as f32, b as f32, a as f32);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    fn flush(&self) -> Result<(), Self::Error> {
        self.assert_context();

        unsafe {
            gl::Flush();
            Ok(())
        }
    }

    fn create_texture(
        &self,
        interpolation: piet::InterpolationMode,
        repeat: piet_gpu::RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error> {
        todo!()
    }

    fn delete_texture(&self, texture: Self::Texture) {
        self.assert_context();

        unsafe {
            gl::DeleteTextures(1, &texture);
        }
    }

    fn write_texture<T: bytemuck::Pod>(
        &self,
        texture: &Self::Texture,
        size: (u32, u32),
        format: piet_gpu::ImageFormat,
        data: Option<&[T]>,
    ) {
        todo!()
    }

    fn write_subtexture<T: bytemuck::Pod>(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet_gpu::ImageFormat,
        data: &[T],
    ) {
        todo!()
    }

    fn set_texture_interpolation(
        &self,
        texture: &Self::Texture,
        interpolation: piet::InterpolationMode,
    ) {
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
            gl::BindTexture(gl::TEXTURE_2D, 0);
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

    fn create_buffer(&self) -> Result<Self::Buffer, Self::Error> {
        self.assert_context();

        unsafe {
            let mut buffer = 0;
            gl::GenBuffers(1, &mut buffer);
            Ok(buffer)
        }
    }

    fn write_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &Self::Buffer,
        data: &[T],
        ty: BufferType,
    ) -> Result<(), Self::Error> {
        self.assert_context();

        unsafe {
            let bind_location = match ty {
                BufferType::Vertex => gl::ARRAY_BUFFER,
                BufferType::Index => gl::ELEMENT_ARRAY_BUFFER,
            };

            gl::BindBuffer(bind_location, *buffer);
            gl::BufferData(
                bind_location,
                (data.len() * std::mem::size_of::<T>()) as isize,
                data.as_ptr() as *const _,
                gl::DYNAMIC_DRAW,
            );
            gl::BindBuffer(bind_location, 0);

            Ok(())
        }
    }

    fn delete_buffer(&self, buffer: Self::Buffer) {
        self.assert_context();

        unsafe {
            gl::DeleteBuffers(1, &buffer);
        }
    }

    fn create_vertex_array(
        &self,
        buffer: &Self::Buffer,
        formats: &[piet_gpu::VertexFormat],
    ) -> Result<Self::VertexArray, Self::Error> {
        todo!()
    }

    fn delete_vertex_array(&self, vertex_array: Self::VertexArray) {
        self.assert_context();

        unsafe {
            gl::DeleteVertexArrays(1, &vertex_array);
        }
    }

    fn push_buffers(
        &self,
        draw_buffers: piet_gpu::DrawBuffers<'_, Self>,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &piet::kurbo::Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error> {
        unsafe {
            // Use our program.
            gl::UseProgram(self.render_program);

            // Set the viewport size.
            let (width, height) = size;
            gl::Viewport(0, 0, width as i32, height as i32);
            gl::Uniform2f(
                self.viewport_size,
                width as f32,
                height as f32,
            );

            // Set the transform.
            let [a, b, c, d, e, f] = transform.as_coeffs();
            let transform = [
                a as f32,
                b as f32,
                0.0,
                c as f32,
                d as f32,
                0.0,
                e as f32,
                f as f32,
                1.0,
            ];
            gl::UniformMatrix3fv(
                self.u_transform,
                1,
                gl::FALSE,
                transform.as_ptr(),
            );

            // Set the texture.
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, *current_texture);
            gl::Uniform1i(self.tex, 0);

            // Set the mask texture.
            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, *mask_texture);
            gl::Uniform1i(self.mask, 1);

            // Set buffers.
            gl::BindBuffer(gl::ARRAY_BUFFER, *draw_buffers.vertex_buffer);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, *draw_buffers.index_buffer);

            // Set vertex attributes.
            gl::BindVertexArray(*draw_buffers.vertex_array);

            // Draw.
            gl::DrawElements(
                gl::TRIANGLES,
                draw_buffers.num_indices as i32,
                gl::UNSIGNED_INT,
                std::ptr::null(),
            );

            // Unbind everything.
            gl::BindVertexArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::UseProgram(0);
        }

        Ok(())
    }
}

const VERTEX_SHADER: &[u8] = b"
#version 330 core

in vec2 aPos;
in vec2 aTexCoord;
in vec4 color;

out vec4 rgbaColor;
out vec2 fTexCoord;
out vec2 fMaskCoord;

uniform mat3 transform;
uniform vec2 viewportSize;

void main() {
    // Transform the vertex position.
    vec2 pos = (transform * vec3(aPos, 1.0)).xy;
    fMaskCoord = pos;

    // Clamp to the viewport size.
    gl_Position = vec4(
        2.0 * posx / viewportSize.x - 1.0,
        1.0 - 2.0 * pos.y / viewportSize.y,
        0.0,
        1.0
    );

    rgbaColor = color / 255.0;
    fTexCoords = aTexCoord;
}
\0";

const FRAGMENT_SHADER: &[u8] = b"
#version 330 core

in vec4 rgbaColor;
in vec2 fTexCoord;
in vec2 fMaskCoord;

uniform sampler2D tex;
uniform sampler2D mask;

void main() {
    vec4 textureColor = texture2D(tex, fTexCoord);
    vec4 = rgbaColor * textureColor;

    float maskAlpha = texture2D(mask, fMaskCoord).a;
    gl_FragColor = vec4(
        finalColor.rgb,
        finalColor.a * maskAlpha
    );
}
\0";
