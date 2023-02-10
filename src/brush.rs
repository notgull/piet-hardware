//! Handles the brush setup.

#![allow(clippy::wrong_self_convention)]

use crate::resources::{Fragment, Program, Shader, Texture, Vertex};
use crate::{Error, GlContext, GlVersion, RenderContext};

use glow::HasContext;
use piet::kurbo::{Affine, Rect};
use piet::{FixedLinearGradient, FixedRadialGradient, IntoBrush};

use std::borrow::Cow;
use std::collections::hash_map::{Entry, HashMap};
use std::fmt::Write;
use std::mem;
use std::rc::Rc;
use std::sync::Arc;

// Various variable/function names used in GLSL.
const IN_POSITION: &str = "position";
const MVP: &str = "mvp";
const MASK_MVP: &str = "maskMvp";
const MASK_COORDS: &str = "maskCoords";
const GET_COLOR: &str = "getColor";
const SOLID_COLOR: &str = "solidColor";
const TEXTURE_MASK: &str = "textureMask";
const GET_MASK_ALPHA: &str = "getMaskAlpha";

/// The brush type used by the [`RenderContext`].
#[derive(Clone)]
pub struct Brush(BrushInner);

#[derive(Clone)]
enum BrushInner {
    Solid(piet::Color),
    LinearGradient(FixedLinearGradient),
    RadialGradient(FixedRadialGradient),
}

impl Brush {
    pub(super) fn solid(color: piet::Color) -> Self {
        Brush(BrushInner::Solid(color))
    }

    pub(super) fn linear_gradient(gradient: FixedLinearGradient) -> Self {
        Brush(BrushInner::LinearGradient(gradient))
    }

    pub(super) fn radial_gradient(gradient: FixedRadialGradient) -> Self {
        Brush(BrushInner::RadialGradient(gradient))
    }

    fn input_type(&self) -> InputType {
        match &self.0 {
            BrushInner::Solid(_) => InputType::Solid,
            BrushInner::LinearGradient(_) => InputType::Linear,
            BrushInner::RadialGradient(_) => InputType::Radial,
        }
    }
}

impl<'a, H: HasContext + ?Sized> IntoBrush<RenderContext<'a, H>> for Brush {
    fn make_brush<'x>(
        &'x self,
        _piet: &mut RenderContext<'a, H>,
        _bbox: impl FnOnce() -> Rect,
    ) -> Cow<'x, Brush> {
        Cow::Borrowed(self)
    }
}

/// The type for a combined mask and mask transform.
#[derive(Debug)]
pub(super) struct Mask<'a, H: HasContext + ?Sized> {
    /// The mask texture.
    pub(super) texture: &'a Texture<H>,

    /// The mask transform.
    pub(super) transform: &'a Affine,
}

/// The type of input for a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InputType {
    Empty,
    Solid,
    Linear,
    Radial,
}

/// Whether or not we use a mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MaskType {
    NoMask,
    Texture,
}

/// Lookup key for a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShaderKey {
    input_type: InputType,
    mask_type: MaskType,
}

/// A cache for brush-related shaders.
#[derive(Debug)]
pub(super) struct Brushes<H: HasContext + ?Sized> {
    /// The map of shaders.
    shaders: HashMap<ShaderKey, Program<H>>,
}

impl<H: HasContext + ?Sized> Brushes<H> {
    pub(super) fn new() -> Self {
        Brushes {
            shaders: HashMap::new(),
        }
    }

    /// Run a closure with the current program set to that of a specific brush.
    ///
    /// This function takes care of uniforms.
    fn with_brush<R>(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        brush: &Brush,
        mvp: &Affine,
        mask: Option<&Mask<'_, H>>,
        f: impl FnOnce() -> Result<R, Error>,
    ) -> Result<R, Error> {
        let mut shader = self.shader_for_brush(context, version, brush, mask)?;

        // Get location for the uniforms we use.
        let mvp_uniform = shader.uniform_location(MVP)?.clone();
        let mask_uniforms = if mask.is_some() {
            Some((
                shader.uniform_location(MASK_MVP)?.clone(),
                shader.uniform_location(TEXTURE_MASK)?.clone(),
            ))
        } else {
            None
        };
        let solid_color_uniform = if matches!(brush.input_type(), InputType::Solid) {
            Some(shader.uniform_location(SOLID_COLOR)?.clone())
        } else {
            None
        };

        shader.with_program(move || {
            // Set the MVP.
            unsafe {
                uniform_affine(&**context, &mvp_uniform, mvp);
            }

            // Set the Mask values.
            if let (Some(mask), Some((mask_mvp_uniform, texture_mask_uniform))) =
                (mask, mask_uniforms)
            {
                unsafe {
                    uniform_affine(&**context, &mask_mvp_uniform, mask.transform);
                }

                mask.texture.bind(Some(0), |bound| unsafe {
                    bound.register_in_uniform(&texture_mask_uniform);
                });
            }

            // Call the function.
            f()
        })
    }

    /// Run a closure with the current program set to the empty program.
    ///
    /// This function takes care of uniforms.
    fn with_empty<R>(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        mvp: &Affine,
        f: impl FnOnce() -> Result<R, Error>,
    ) -> Result<R, Error> {
        let mut shader = self.shader_for_empty(context, version)?;

        // Get location for MVP uniform.
        let mvp_uniform = shader.uniform_location(MVP)?.clone();

        // Enter the program.
        shader.with_program(move || {
            // Set the MVP.
            unsafe {
                uniform_affine(&**context, &mvp_uniform, mvp);
            }

            // Run with the program.
            f()
        })
    }

    fn shader_for_brush(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        brush: &Brush,
        mask: Option<&Mask<'_, H>>,
    ) -> Result<&mut Program<H>, Error> {
        self.fetch_or_create_shader(
            context,
            version,
            brush.input_type(),
            if mask.is_some() {
                MaskType::Texture
            } else {
                MaskType::NoMask
            },
        )
    }

    /// Fetch or create the shader program that just emits black.
    fn shader_for_empty(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
    ) -> Result<&mut Program<H>, Error> {
        self.fetch_or_create_shader(context, version, InputType::Empty, MaskType::NoMask)
    }

    /// Fetch the shader program from the cache or create a new one.
    fn fetch_or_create_shader(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        input_type: InputType,
        mask_type: MaskType,
    ) -> Result<&mut Program<H>, Error> {
        let lookup_key = ShaderKey {
            input_type,
            mask_type,
        };

        // Use the cached version if available, or create a new one.
        match self.shaders.entry(lookup_key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                // Create a new shader and insert it into the cache.
                let vertex = VertexBuilder::new(version)
                    .with_mask(mask_type)
                    .to_shader(context)?;

                let fragment = FragmentBuilder::new(version)
                    .with_mask_type(mask_type)
                    .with_input_type(input_type)
                    .to_shader(context)?;

                let program = Program::with_vertex_and_fragment(vertex, fragment)?;

                Ok(entry.insert(program))
            }
        }
    }
}

const SHADER_SOURCE_CAPACITY: usize = 1024;

/// A builder for the source code of a shader.
struct VertexBuilder {
    /// The source code.
    source: String,

    /// Whether we are using a texture.
    textured_mask: bool,
}

impl VertexBuilder {
    /// Creates a new builder.
    fn new(vesion: GlVersion) -> Self {
        VertexBuilder {
            source: {
                let mut source = String::with_capacity(SHADER_SOURCE_CAPACITY);
                source.push_str(vesion.shader_header());
                source.push('\n');

                // Write text input and MVP transform.
                writeln!(source, "layout(location = 0) in vec2 {IN_POSITION};").unwrap();
                writeln!(source, "uniform mat3 {MVP};").unwrap();

                source
            },
            textured_mask: false,
        }
    }

    /// Adds a mask type.
    fn with_mask(&mut self, mask_type: MaskType) -> &mut Self {
        match mask_type {
            MaskType::NoMask => self,
            MaskType::Texture => self.with_texture_mask(),
        }
    }

    /// Adds a texture mask input.
    fn with_texture_mask(&mut self) -> &mut Self {
        self.textured_mask = true;
        writeln!(self.source, "uniform mat3 {MASK_MVP};").unwrap();
        writeln!(self.source, "out vec2 {MASK_COORDS};").unwrap();
        self
    }

    /// Build the shader source.
    fn to_source(&mut self) -> String {
        let mut source = mem::take(&mut self.source);

        // Write the main function.
        writeln!(source, "void main() {{").unwrap();
        writeln!(
            source,
            "    gl_Position = vec4((mvp * vec3({IN_POSITION}, 1.0)).xy, 0.0, 1.0);"
        )
        .unwrap();

        if self.textured_mask {
            // Set up tex coords.
            writeln!(
                source,
                "    {MASK_COORDS} = ({MASK_MVP} * vec3({IN_POSITION}, 1.0)).xy;"
            )
            .unwrap();
        }

        writeln!(source, "}}").unwrap();

        source
    }

    /// Make a new shader.
    fn to_shader<H: HasContext + ?Sized>(
        &mut self,
        ctx: &Rc<H>,
    ) -> Result<Shader<H, Vertex>, Error> {
        Shader::new(ctx, &self.to_source())
    }
}

/// A builder for the source code of a fragment shader.
struct FragmentBuilder {
    /// The source code.
    source: String,
}

impl FragmentBuilder {
    fn new(version: GlVersion) -> Self {
        Self {
            source: {
                let mut source = String::with_capacity(SHADER_SOURCE_CAPACITY);

                // Write the heaader.
                source.push_str(version.shader_header());
                source.push('\n');

                source
            },
        }
    }

    /// Use with the provided input type.
    fn with_input_type(&mut self, ty: InputType) -> &mut Self {
        match ty {
            InputType::Empty => self.with_empty_color(),
            InputType::Solid => self.with_solid_color(),
            _ => todo!(),
        }
    }

    /// Use an empty color.
    fn with_empty_color(&mut self) -> &mut Self {
        writeln!(
            self.source,
            "
            vec4 {GET_COLOR}() {{
                return vec4(0, 0, 0, 1);
            }}
        "
        )
        .ok();

        self
    }

    /// Use a solid color.
    fn with_solid_color(&mut self) -> &mut Self {
        writeln!(
            self.source,
            "
            uniform vec4 {SOLID_COLOR};

            vec4 {GET_COLOR}() {{
                return {SOLID_COLOR};
            }}
        "
        )
        .ok();

        self
    }

    /// Use with a specific mask type.
    fn with_mask_type(&mut self, ty: MaskType) -> &mut Self {
        match ty {
            MaskType::NoMask => self.with_no_mask(),
            MaskType::Texture => self.with_texture_mask(),
        }
    }

    /// Use with a textured mask.
    fn with_texture_mask(&mut self) -> &mut Self {
        writeln!(
            self.source,
            "
            in vec2 {MASK_COORDS};
            uniform sampler2D {TEXTURE_MASK};

            float {GET_MASK_ALPHA}() {{
                return texture2D({TEXTURE_MASK}, {MASK_COORDS}); 
            }}
        "
        )
        .ok();

        self
    }

    /// Use without a mask.
    fn with_no_mask(&mut self) -> &mut Self {
        writeln!(
            self.source,
            "
            float {GET_MASK_ALPHA}() {{
                return 1;
            }}
        "
        )
        .ok();

        self
    }

    /// Convert to source code.
    fn to_source(&mut self) -> String {
        let mut source = mem::take(&mut self.source);

        // Write the "main" function.
        writeln!(
            source,
            "
            void main() {{
                gl_FragColor = {GET_COLOR}() * {GET_MASK_ALPHA}();
            }}
        "
        )
        .ok();

        source
    }

    /// Convert to a fragment shader.
    fn to_shader<H: HasContext + ?Sized>(
        &mut self,
        ctx: &Rc<H>,
    ) -> Result<Shader<H, Fragment>, Error> {
        Shader::new(ctx, &self.to_source())
    }
}

impl GlVersion {
    /// Returns the header for the shader.
    fn shader_header(&self) -> &'static str {
        match self {
            GlVersion::Gl32 => "#version 330 core",
            GlVersion::Es30 => "#version 300 es",
        }
    }
}

/// Register an `Affine` as a uniform.
///
/// # Safety
///
/// The target uniform must be a `mat3`.
unsafe fn uniform_affine<H: HasContext + ?Sized>(
    context: &H,
    uniform: &H::UniformLocation,
    aff: &Affine,
) {
    let aff = affine_to_gl_matrix(aff);

    context.uniform_matrix_3_f32_slice(Some(uniform), false, &aff);
}

/// Register a `vec4` as a uniform.
///
/// # Safety
///
/// The target uniform must be a `vec4`.
unsafe fn uniform_vec4<H: HasContext + ?Sized>(
    context: &H,
    uniform: &H::UniformLocation,
    vec: &[f32; 4],
) {
    context.uniform_4_f32_slice(Some(uniform), vec)
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
