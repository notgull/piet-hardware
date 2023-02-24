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

//! OpenGL resources that implement `Drop` to free the resources when they go out of scope.

use crate::Error;
use glow::HasContext;
use piet::kurbo::Affine;
use piet::{ImageFormat, InterpolationMode, Text};

use std::borrow::Borrow;
use std::collections::hash_map::{Entry, HashMap};
use std::fmt::{self, Write};
use std::marker::PhantomData;
use std::rc::Rc;

/// A program.
pub(super) struct Program<H: HasContext + ?Sized> {
    /// The OpenGL context.
    context: Rc<H>,

    /// The program ID.
    id: H::Program,

    /// Uniform locations.
    uniforms: HashMap<&'static str, H::UniformLocation>,
}

impl<H: HasContext + ?Sized> fmt::Debug for Program<H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Program").field(&self.id).finish()
    }
}

impl<H: HasContext + ?Sized> Drop for Program<H> {
    fn drop(&mut self) {
        unsafe {
            self.context.delete_program(self.id);
        }
    }
}

impl<H: HasContext + ?Sized> Program<H> {
    /// Create a new program from a vertex and fragment shader.
    pub(super) fn with_vertex_and_fragment(
        vertex: Shader<H, Vertex>,
        fragment: Shader<H, Fragment>,
    ) -> Result<Self, Error> {
        assert!(Rc::ptr_eq(&vertex.context, &fragment.context));

        unsafe {
            // Create the program.
            let id = vertex.context.create_program().map_err(|e| {
                let err = format!("Failed to create program: {e}");
                Error::BackendError(err.into())
            })?;

            // Attach the shaders.
            vertex.context.attach_shader(id, vertex.id);
            vertex.context.attach_shader(id, fragment.id);

            // Link the program.
            vertex.context.link_program(id);

            // Check for errors.
            if vertex.context.get_program_link_status(id) {
                // Detach the shaders.
                vertex.context.detach_shader(id, vertex.id);
                vertex.context.detach_shader(id, fragment.id);

                Ok(Self {
                    context: vertex.context.clone(),
                    id,
                    uniforms: HashMap::new(),
                })
            } else {
                let err = vertex.context.get_program_info_log(id);
                Err(Error::BackendError(err.into()))
            }
        }
    }

    /// Get the location of a uniform.
    pub(super) fn uniform_location(
        &mut self,
        name: &'static str,
    ) -> Result<&mut H::UniformLocation, Error> {
        match self.uniforms.entry(name) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let location = unsafe {
                    self.context
                        .get_uniform_location(self.id, name)
                        .ok_or_else(|| {
                            let err = format!("Failed to get uniform location: {name}");
                            Error::BackendError(err.into())
                        })?
                };

                Ok(entry.insert(location))
            }
        }
    }

    /// Use the program.
    ///
    /// Note: Do NOT call this function reentrantly.
    pub(super) fn bind(&self) -> BoundProgram<'_, H> {
        unsafe {
            self.context.use_program(Some(self.id));

            BoundProgram(self.context.as_ref())
        }
    }
}

/// A bound program.
pub(super) struct BoundProgram<'a, H: HasContext + ?Sized>(&'a H);

impl<H: HasContext + ?Sized> Drop for BoundProgram<'_, H> {
    fn drop(&mut self) {
        unsafe {
            self.0.use_program(None);
        }
    }
}

impl<H: HasContext + ?Sized> BoundProgram<'_, H> {
    /// Register an affine transformation at a `mat3` uniform.
    pub(super) fn register_mat3(&self, location: &H::UniformLocation, transform: &Affine) {
        let matrix = affine_to_gl_matrix(transform);

        unsafe {
            self.0
                .uniform_matrix_3_f32_slice(Some(location), false, &matrix);
        }
    }

    /// Register a four-component color at a `vec4` uniform.
    pub(super) fn register_color(&self, location: &H::UniformLocation, color: piet::Color) {
        let (r, g, b, a) = color.as_rgba();
        let color = [r as f32, g as f32, b as f32, a as f32];

        unsafe {
            self.0.uniform_4_f32_slice(Some(location), &color);
        }
    }

    /// Register a texture as a `sampler2D` uniform.
    pub(super) fn register_texture<B: Borrow<Texture<H>>>(
        &self,
        location: &H::UniformLocation,
        texture: &mut BoundTexture<H, B>,
    ) {
        unsafe {
            let active = texture.active.expect("Texture is not active");
            self.0.uniform_1_u32(Some(location), active);
        }
    }
}

/// A shader.
pub(super) struct Shader<H: HasContext + ?Sized, Ty> {
    /// The OpenGL context.
    context: Rc<H>,

    /// The shader ID.
    id: H::Shader,

    /// The shader type.
    _ty: PhantomData<Ty>,
}

impl<H: HasContext + ?Sized, Ty> Drop for Shader<H, Ty> {
    fn drop(&mut self) {
        unsafe {
            self.context.delete_shader(self.id);
        }
    }
}

impl<H: HasContext + ?Sized, Ty: ShaderType> Shader<H, Ty> {
    /// Create a new shader from the given source.
    pub(super) fn new(context: &Rc<H>, source: &str) -> Result<Self, Error> {
        unsafe {
            // Create the shader.
            let id = context.create_shader(Ty::TYPE).map_err(|e| {
                let err = format!("Failed to create shader: {e}");
                Error::BackendError(err.into())
            })?;

            // Set the source.
            context.shader_source(id, source);

            // Compile the shader.
            context.compile_shader(id);

            // Check for errors.
            if context.get_shader_compile_status(id) {
                Ok(Self {
                    context: context.clone(),
                    id,
                    _ty: PhantomData,
                })
            } else {
                let err = context.get_shader_info_log(id);

                Err(Error::BackendError(
                    ShaderError {
                        error_msg: err,
                        source_code: source.to_string(),
                        shader_ty: Ty::name(),
                    }
                    .into(),
                ))
            }
        }
    }
}

/// A framebuffer.
pub(super) struct Framebuffer<H: HasContext + ?Sized> {
    /// The OpenGL context.
    context: Rc<H>,

    /// The framebuffer ID.
    id: H::Framebuffer,
}

impl<H: HasContext + ?Sized> Drop for Framebuffer<H> {
    fn drop(&mut self) {
        unsafe {
            self.context.delete_framebuffer(self.id);
        }
    }
}

impl<H: HasContext + ?Sized> Framebuffer<H> {
    /// Create a new framebuffer.
    pub fn new(context: &Rc<H>) -> Result<Self, Error> {
        unsafe {
            let id = context.create_framebuffer().map_err(|e| {
                let err = format!("Failed to create framebuffer: {e}");
                Error::BackendError(err.into())
            })?;

            Ok(Self {
                context: context.clone(),
                id,
            })
        }
    }

    /// Bind the framebuffer to `GL_FRAMEBUFFER`.
    pub fn bind(&self) -> BoundFramebuffer<'_, H> {
        unsafe {
            self.context
                .bind_framebuffer(glow::FRAMEBUFFER, Some(self.id));
        }

        BoundFramebuffer {
            context: &self.context,
        }
    }
}

/// A texture.
pub(super) struct Texture<H: HasContext + ?Sized> {
    /// The OpenGL context.
    context: Rc<H>,

    /// The texture ID.
    id: H::Texture,
}

impl<H: HasContext + ?Sized> fmt::Debug for Texture<H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Texture").field(&self.id).finish()
    }
}

impl<H: HasContext + ?Sized> Clone for Texture<H> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            id: self.id,
        }
    }
}

impl<H: HasContext + ?Sized> Drop for Texture<H> {
    fn drop(&mut self) {
        unsafe {
            self.context.delete_texture(self.id);
        }
    }
}

impl<H: HasContext + ?Sized> Texture<H> {
    /// Create a new texture.
    pub(super) fn new(context: &Rc<H>) -> Result<Self, Error> {
        unsafe {
            let id = context.create_texture().map_err(|e| {
                let err = format!("Failed to create texture: {e}");
                Error::BackendError(err.into())
            })?;

            let this = Self {
                context: context.clone(),
                id,
            };
            let _bound = this.bind(None);

            // Set wrap to GL_CLAMP_TO_BORDER and border color to transparent.
            context.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_BORDER as i32,
            );
            context.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_BORDER as i32,
            );
            context.tex_parameter_f32_slice(
                glow::TEXTURE_2D,
                glow::TEXTURE_BORDER_COLOR,
                &[0.0, 0.0, 0.0, 0.0],
            );

            drop(_bound);
            Ok(this)
        }
    }

    /// Run with this texture bound to the `GL_TEXTURE_2D` unit ID.
    ///
    /// `active` should be `Some` to also bind this to `GL_TEXTURE0 + active`.
    pub(super) fn bind(&self, active: Option<u32>) -> BoundTexture<H, &Texture<H>> {
        // If possible, set the active texture.
        if let Some(active) = active {
            unsafe {
                self.context.active_texture(glow::TEXTURE0 + active);
            }
        }

        // Bind to the GL texture slot.
        unsafe {
            self.context.bind_texture(glow::TEXTURE_2D, Some(self.id));
        }

        BoundTexture {
            texture: self,
            active,
            _marker: PhantomData,
        }
    }

    /// Run with this texture bound to the `GL_TEXTURE_2D` unit ID.
    ///
    /// `active` should be `Some` to also bind this to `GL_TEXTURE0 + active`.
    pub(super) fn bind_rc(self: Rc<Self>, active: Option<u32>) -> BoundTexture<H, Rc<Texture<H>>> {
        // If possible, set the active texture.
        if let Some(active) = active {
            unsafe {
                self.context.active_texture(glow::TEXTURE0 + active);
            }
        }

        // Bind to the GL texture slot.
        unsafe {
            self.context.bind_texture(glow::TEXTURE_2D, Some(self.id));
        }

        BoundTexture {
            texture: self,
            active,
            _marker: PhantomData,
        }
    }
}

/// VAO.
pub(super) struct VertexArray<H: HasContext + ?Sized> {
    /// The OpenGL context.
    context: Rc<H>,

    /// The VAO ID.
    id: H::VertexArray,
}

impl<H: HasContext + ?Sized> Drop for VertexArray<H> {
    fn drop(&mut self) {
        unsafe {
            self.context.delete_vertex_array(self.id);
        }
    }
}

impl<H: HasContext + ?Sized> VertexArray<H> {
    /// Create a new VAO.
    pub(super) fn new(context: &Rc<H>) -> Result<Self, Error> {
        unsafe {
            let id = context.create_vertex_array().map_err(|e| {
                let err = format!("Failed to create vertex array: {e}");
                Error::BackendError(err.into())
            })?;

            Ok(Self {
                context: context.clone(),
                id,
            })
        }
    }

    /// Bind this buffer to the active VAO.
    pub(super) fn bind(&self) -> BoundVertexArray<'_, H> {
        unsafe {
            self.context.bind_vertex_array(Some(self.id));
        }

        BoundVertexArray {
            context: &self.context,
        }
    }
}

/// An object representing a vertex buffer.
pub(super) struct VertexBuffer<H: HasContext + ?Sized> {
    /// The OpenGL context.
    context: Rc<H>,

    /// The buffer ID.
    id: H::Buffer,
}

impl<H: HasContext + ?Sized> Drop for VertexBuffer<H> {
    fn drop(&mut self) {
        unsafe {
            self.context.delete_buffer(self.id);
        }
    }
}

impl<H: HasContext + ?Sized> VertexBuffer<H> {
    /// Create a new vertex buffer.
    pub(super) fn new(context: &Rc<H>) -> Result<Self, Error> {
        unsafe {
            let id = context.create_buffer().map_err(|e| {
                let err = format!("Failed to create vertex buffer: {e}");
                Error::BackendError(err.into())
            })?;

            Ok(Self {
                context: context.clone(),
                id,
            })
        }
    }

    /// Bind this buffer to the given target.
    pub(super) fn bind(&self, target: BufferTarget) -> BoundVertexBuffer<'_, H> {
        unsafe {
            self.context.bind_buffer(target as u32, Some(self.id));
        }

        BoundVertexBuffer {
            context: &self.context,
            location: target,
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(u32)]
pub(super) enum BufferTarget {
    Array = glow::ARRAY_BUFFER,
    ElementArray = glow::ELEMENT_ARRAY_BUFFER,
}

/// An object representing a currently bound texture.
pub(super) struct BoundTexture<H: HasContext + ?Sized, B: Borrow<Texture<H>>> {
    texture: B,

    /// If we are bound to an active texture, we're bound here.
    active: Option<u32>,

    _marker: PhantomData<H>,
}

impl<H: HasContext + ?Sized, B: Borrow<Texture<H>>> Drop for BoundTexture<H, B> {
    fn drop(&mut self) {
        unsafe {
            self.context().bind_texture(glow::TEXTURE_2D, None);
        }
    }
}

impl<H: HasContext + ?Sized, B: Borrow<Texture<H>>> BoundTexture<H, B> {
    fn context(&self) -> &H {
        &self.texture.borrow().context
    }

    /// Register this bound texture to a uniform.
    ///
    /// # Safety
    ///
    /// The target uniform must be a `sampler2D`.
    pub(super) unsafe fn register_in_uniform(&mut self, uniform: &H::UniformLocation) {
        self.context().uniform_1_u32(
            Some(uniform),
            self.active
                .expect("Called register_in_uniform for inactive texture"),
        )
    }

    /// Fill this texture with nothing.
    pub(super) fn fill_with_nothing(&mut self, width: i32, height: i32) {
        unsafe {
            self.context().tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB as _,
                width,
                height,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                None,
            );
        }
    }

    /// Fill this texture with an image.
    pub(super) fn fill_with_image(
        &mut self,
        width: i32,
        height: i32,
        format: ImageFormat,
        image: &[u8],
    ) -> Result<(), Error> {
        let format = match format {
            ImageFormat::Grayscale => glow::RED,
            ImageFormat::Rgb => glow::RGB,
            ImageFormat::RgbaSeparate => glow::RGBA,
            _ => return Err(Error::NotSupported),
        };

        unsafe {
            self.context().tex_image_2d(
                glow::TEXTURE_2D,
                0,
                format as _,
                width,
                height,
                0,
                format,
                glow::UNSIGNED_BYTE,
                Some(image),
            );
        }

        Ok(())
    }

    pub(super) fn set_interpolation_mode(&mut self, mode: InterpolationMode) {
        match mode {
            InterpolationMode::NearestNeighbor => self.filtering_nearest(),
            InterpolationMode::Bilinear => self.filtering_linear(),
        }
    }

    /// Set the texture parameters to NEAREST filtering.
    pub(super) fn filtering_nearest(&mut self) {
        unsafe {
            self.context().tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as _,
            );
            self.context().tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as _,
            );
        }
    }

    /// Set the texture parameters to LINEAR filtering.
    pub(super) fn filtering_linear(&mut self) {
        unsafe {
            self.context().tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as _,
            );
            self.context().tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as _,
            );
        }
    }

    /*
        /// Dump the contents of the texture to the disk.
        ///
        /// When uncommenting this procedure, re-enable the "image" crate.
        pub(crate) unsafe fn dump_to_disk(
            &mut self,
            path: &std::path::Path,
            width: u32,
            height: u32,
            format: ImageFormat,
        ) {
            let mut bpp = match format {
                ImageFormat::Grayscale => 1,
                ImageFormat::Rgb => 3,
                ImageFormat::RgbaSeparate | ImageFormat::RgbaPremul => 4,
                _ => panic!("Unsupported format"),
            };

            let mut data = vec![0u8; (width * height * bpp) as usize];

            self.context.get_tex_image(
                glow::TEXTURE_2D,
                0,
                match format {
                    ImageFormat::Grayscale => glow::RED,
                    ImageFormat::Rgb => glow::RGB,
                    ImageFormat::RgbaSeparate | ImageFormat::RgbaPremul => glow::RGBA,
                    _ => panic!("Unsupported format"),
                },
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(&mut data),
            );

            // Create an image from the data.
            let mut img = image::ImageBuffer::new(width, height);

            // Copy the data into the image.
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let offset = (y * width + x) as usize * bpp as usize;

                *pixel = match format {
                    ImageFormat::Grayscale => {
                        let v = data[offset];
                        image::Rgba([v, v, v, 255])
                    }
                    ImageFormat::Rgb => {
                        let r = data[offset];
                        let g = data[offset + 1];
                        let b = data[offset + 2];
                        image::Rgba([r, g, b, 255])
                    }
                    ImageFormat::RgbaSeparate | ImageFormat::RgbaPremul => {
                        let r = data[offset];
                        let g = data[offset + 1];
                        let b = data[offset + 2];
                        let a = data[offset + 3];
                        image::Rgba([r, g, b, a])
                    }
                    _ => panic!("Unsupported format"),
                };
            }

            // Save the image.
            img.save(path).unwrap();
        }
    */
}

/// An object representing a framebuffer bound to GL_FRAMEBUFFER.
pub(super) struct BoundFramebuffer<'a, H: HasContext + ?Sized> {
    context: &'a H,
}

impl<H: HasContext + ?Sized> Drop for BoundFramebuffer<'_, H> {
    fn drop(&mut self) {
        unsafe {
            self.context.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }
}

impl<H: HasContext + ?Sized> BoundFramebuffer<'_, H> {
    /// Bind this framebuffer as the first color attachment.
    pub(super) fn bind_color0(&mut self, tex: &Texture<H>) {
        unsafe {
            self.context.framebuffer_texture(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                Some(tex.id),
                0,
            );
            self.context.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
        }
    }

    /// Check the frame buffers for errors.
    pub(super) fn check_error(&mut self) -> Result<(), Error> {
        unsafe {
            if self.context.check_framebuffer_status(glow::FRAMEBUFFER)
                == glow::FRAMEBUFFER_COMPLETE
            {
                Ok(())
            } else {
                Err(Error::BackendError(
                    "unable to bind framebuffer for mask".into(),
                ))
            }
        }
    }
}

/// An object representing a bound VAO.
pub(super) struct BoundVertexArray<'a, H: HasContext + ?Sized> {
    context: &'a H,
}

impl<'a, H: HasContext + ?Sized> Drop for BoundVertexArray<'a, H> {
    fn drop(&mut self) {
        unsafe {
            self.context.bind_vertex_array(None);
        }
    }
}

impl<H: HasContext + ?Sized> BoundVertexArray<'_, H> {
    /// Add an attribute pointer to the VAO.
    pub(super) fn attribute_ptr(&mut self, _bound_buffer: &BoundVertexBuffer<'_, H>) {
        unsafe {
            self.context.vertex_attrib_pointer_f32(
                0,
                2,
                glow::FLOAT,
                false,
                (2 * std::mem::size_of::<f32>()) as _,
                0,
            );
            self.context.enable_vertex_attrib_array(0);
        }
    }

    /// Draw triangles.
    pub(super) unsafe fn draw_triangles(&mut self, count: usize) {
        self.context
            .draw_elements(glow::TRIANGLES, count as i32, glow::UNSIGNED_INT, 0)
    }
}

/// An object representing a bound vertex buffer.
pub(super) struct BoundVertexBuffer<'a, H: HasContext + ?Sized> {
    context: &'a H,

    /// The location at which the buffer is bound.
    location: BufferTarget,
}

impl<H: HasContext + ?Sized> Drop for BoundVertexBuffer<'_, H> {
    fn drop(&mut self) {
        unsafe {
            self.context.bind_buffer(self.location as u32, None);
        }
    }
}

impl<'a, H: HasContext + ?Sized> BoundVertexBuffer<'a, H> {
    /// Upload floating point data to the buffer.
    pub(super) fn upload_f32(&mut self, data: &[f32]) {
        unsafe {
            self.context.buffer_data_u8_slice(
                self.location as u32,
                bytemuck::cast_slice(data),
                glow::DYNAMIC_DRAW,
            );
        }
    }

    /// Upload unsigned integer data to the buffer.
    pub(super) fn upload_u32(&mut self, data: &[u32]) {
        unsafe {
            self.context.buffer_data_u8_slice(
                self.location as u32,
                bytemuck::cast_slice(data),
                glow::DYNAMIC_DRAW,
            );
        }
    }
}

/// The type of a shader.
///
/// # Safety
///
/// `TYPE` must be a valid shader type.
pub(super) unsafe trait ShaderType {
    /// The shader type.
    const TYPE: u32;

    /// Debugging name.
    fn name() -> &'static str;
}

/// Vertex shader.
pub(super) struct Vertex;

unsafe impl ShaderType for Vertex {
    const TYPE: u32 = glow::VERTEX_SHADER;

    fn name() -> &'static str {
        "vertex"
    }
}

/// Fragment shader.
pub(super) struct Fragment;

unsafe impl ShaderType for Fragment {
    const TYPE: u32 = glow::FRAGMENT_SHADER;

    fn name() -> &'static str {
        "fragment"
    }
}

/// Convert an `Affine` to an OpenGL matrix.
fn affine_to_gl_matrix(aff: &Affine) -> [f32; 9] {
    macro_rules! f {
        ($e:expr) => {
            ($e as f32)
        };
    }

    let [a, b, c, d, e, f] = aff.as_coeffs();
    [f!(a), f!(b), 0.0, f!(c), f!(d), 0.0, f!(e), f!(f), 1.0]
}

struct ShaderError {
    error_msg: String,
    source_code: String,
    shader_ty: &'static str,
}

impl fmt::Debug for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Failed to compile the {} shader: {}",
            self.shader_ty, self.error_msg
        )?;

        if !self.source_code.is_empty() {
            writeln!(f, "Source code:")?;

            // Print a line as long as our longest line.
            let max_len = self.source_code.lines().map(|l| l.len()).max().unwrap_or(0);

            for _ in 0..max_len {
                f.write_char('-')?;
            }

            f.write_char('\n');

            // Print soure along with line numbers.
            for (i, line) in self.source_code.lines().enumerate() {
                writeln!(f, "{:>4} | {}", i + 1, line)?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for ShaderError {}

struct CallOnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for CallOnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}
