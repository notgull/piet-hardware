//! OpenGL resources that implement `Drop` to free the resources when they go out of scope.

use crate::Error;
use glow::HasContext;

use std::collections::hash_map::{Entry, HashMap};
use std::fmt;
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
    pub(super) fn with_program<R>(&self, f: impl FnOnce() -> R) -> R {
        unsafe {
            self.context.use_program(Some(self.id));

            let _guard = CallOnDrop(|| {
                self.context.use_program(None);
            });

            f()
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
                Err(Error::BackendError(err.into()))
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

            Ok(Self {
                context: context.clone(),
                id,
            })
        }
    }

    /// Run with this texture bound to the `GL_TEXTURE_2D` unit ID.
    ///
    /// `active` should be `Some` to also bind this to `GL_TEXTURE0 + active`.
    pub(super) fn bind(&self, active: Option<u32>) -> BoundTexture<'_, H> {
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
            context: &self.context,
            active,
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
pub(super) struct BoundTexture<'a, H: HasContext + ?Sized> {
    context: &'a H,

    /// If we are bound to an active texture, we're bound here.
    active: Option<u32>,
}

impl<H: HasContext + ?Sized> Drop for BoundTexture<'_, H> {
    fn drop(&mut self) {
        unsafe {
            self.context.bind_texture(glow::TEXTURE_2D, None);
        }
    }
}

impl<H: HasContext + ?Sized> BoundTexture<'_, H> {
    /// Register this bound texture to a uniform.
    ///
    /// # Safety
    ///
    /// The target uniform must be a `sampler2D`.
    pub(super) unsafe fn register_in_uniform(&mut self, uniform: &H::UniformLocation) {
        self.context.uniform_1_u32(
            Some(uniform),
            self.active
                .expect("Called register_in_uniform for inactive texture"),
        )
    }

    /// Fill this texture with nothing.
    pub(super) fn fill_with_nothing(&mut self, width: i32, height: i32) {
        unsafe {
            self.context.tex_image_2d(
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

    /// Set the texture parameters to NEAREST filtering.
    pub(super) fn filtering_nearest(&mut self) {
        unsafe {
            self.context.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as _,
            );
            self.context.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as _,
            );
        }
    }
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
}

/// Vertex shader.
pub(super) struct Vertex;

unsafe impl ShaderType for Vertex {
    const TYPE: u32 = glow::VERTEX_SHADER;
}

/// Fragment shader.
pub(super) struct Fragment;

unsafe impl ShaderType for Fragment {
    const TYPE: u32 = glow::FRAGMENT_SHADER;
}

struct CallOnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for CallOnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}
