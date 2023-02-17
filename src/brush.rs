//! Handles the brush setup.

#![allow(clippy::wrong_self_convention)]

use crate::resources::{BoundProgram, Fragment, Program, Shader, Texture, Vertex};
use crate::{Error, GlVersion, Image, RenderContext};

use glow::HasContext;
use piet::kurbo::{Affine, Rect};
use piet::{FixedLinearGradient, FixedRadialGradient, IntoBrush};

use std::borrow::Cow;
use std::collections::hash_map::{Entry, HashMap};
use std::fmt::Write;
use std::mem;
use std::rc::Rc;

// Various variable/function names used in GLSL.
const IN_POSITION: &str = "position";
const OUTPUT_COLOR: &str = "outputColor";
const TEXTURE_COORDS: &str = "textureCoords";

const LINEAR_GRADIENT_START: &str = "linearGradientStart";
const LINEAR_GRADIENT_END: &str = "linearGradientEnd";
const GRADIENT_COLORS: &str = "gradientColors";
const GRADIENT_STOPS: &str = "gradientStops";

const MVP: &str = "mvp";
const SOLID_COLOR: &str = "solidColor";
const SRC_SIZE: &str = "srcSize";
const DST_RECT: &str = "dstRect";

const MASK_MVP: &str = "maskMvp";
const MASK_COORDS: &str = "maskCoords";
const TEXTURE_MASK: &str = "textureMask";

const GET_COLOR: &str = "getColor";
const GET_MASK_ALPHA: &str = "getMaskAlpha";
const GET_GRADIENT_COORD: &str = "getGradientCoord";

/// The brush type used by the [`RenderContext`].
pub struct Brush<H: HasContext + ?Sized>(BrushInner<H>);

enum BrushInner<H: HasContext + ?Sized> {
    /// A solid color.
    Solid(piet::Color),

    /// A linear gradient.
    LinearGradient(FixedLinearGradient),

    /// A radial gradient.
    RadialGradient(FixedRadialGradient),

    /// A texture.
    Texture {
        /// The texture.
        texture: Texture<H>,

        /// The destination rectangle.
        dest: Rect,
    },
}

impl<H: HasContext + ?Sized> Clone for Brush<H> {
    fn clone(&self) -> Self {
        match &self.0 {
            BrushInner::Solid(color) => Brush::solid(*color),
            BrushInner::LinearGradient(gradient) => Brush::linear_gradient(gradient.clone()),
            BrushInner::RadialGradient(gradient) => Brush::radial_gradient(gradient.clone()),
            BrushInner::Texture { texture, dest } => Brush(BrushInner::Texture {
                texture: texture.clone(),
                dest: *dest,
            }),
        }
    }
}

impl<H: HasContext + ?Sized> Brush<H> {
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
            BrushInner::Texture { .. } => InputType::Texture,
        }
    }
}

impl<'a, H: HasContext + ?Sized> IntoBrush<RenderContext<'a, H>> for Brush<H> {
    fn make_brush<'x>(
        &'x self,
        _piet: &mut RenderContext<'a, H>,
        _bbox: impl FnOnce() -> Rect,
    ) -> Cow<'x, Brush<H>> {
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
    Texture,
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
    write_to_mask: bool,
}

pub(super) enum Target<'a, H: HasContext + ?Sized> {
    /// Draw to the surface, using the given brush.
    Surface(&'a Brush<H>),

    /// Draw to a framebuffer, using the empty brush.
    Framebuffer,
}

impl<H: HasContext + ?Sized> Target<'_, H> {
    fn input_type(&self) -> InputType {
        match self {
            Target::Surface(brush) => brush.input_type(),
            Target::Framebuffer => InputType::Empty,
        }
    }
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

    /// Run a closure with the current program set to that of a specific target.
    ///
    /// This function takes care of uniforms.
    pub(super) fn with_target(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        brush: Target<'_, H>,
        mvp: &Affine,
        mask: Option<&Mask<'_, H>>,
    ) -> Result<BoundProgram<'_, H>, Error> {
        let shader = match brush {
            Target::Surface(brush) => self.shader_for_brush(context, version, brush, mask)?,
            Target::Framebuffer => self.shader_for_empty(context, version, mask)?,
        };

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

        let program = shader.bind();

        // Set the MVP.
        program.register_mat3(&mvp_uniform, mvp);

        // Set the Mask values.
        if let (Some(mask), Some((mask_mvp_uniform, texture_mask_uniform))) = (mask, mask_uniforms)
        {
            program.register_mat3(&mask_mvp_uniform, mask.transform);

            let mut bound = mask.texture.bind(Some(0));
            program.register_texture(&texture_mask_uniform, &mut bound);
        }

        // Set the solid color.
        if let (Some(solid_color_uniform), Target::Surface(Brush(BrushInner::Solid(color)))) =
            (solid_color_uniform, &brush)
        {
            program.register_color(&solid_color_uniform, *color);
        }

        Ok(program)
    }

    fn shader_for_brush(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        brush: &Brush<H>,
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
            false,
        )
    }

    /// Fetch or create the shader program that just emits black.
    fn shader_for_empty(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        mask: Option<&Mask<'_, H>>,
    ) -> Result<&mut Program<H>, Error> {
        self.fetch_or_create_shader(
            context,
            version,
            InputType::Empty,
            if mask.is_some() {
                MaskType::Texture
            } else {
                MaskType::NoMask
            },
            true,
        )
    }

    /// Fetch the shader program from the cache or create a new one.
    fn fetch_or_create_shader(
        &mut self,
        context: &Rc<H>,
        version: GlVersion,
        input_type: InputType,
        mask_type: MaskType,
        write_to_mask: bool,
    ) -> Result<&mut Program<H>, Error> {
        let lookup_key = ShaderKey {
            input_type,
            mask_type,
            write_to_mask,
        };

        // Use the cached version if available, or create a new one.
        match self.shaders.entry(lookup_key) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                // Create a new shader and insert it into the cache.
                let vertex = VertexBuilder::new(version)
                    .with_mask(mask_type)
                    .to_shader(context)?;

                let fragment = {
                    let mut builder = FragmentBuilder::new(version);
                    builder.with_mask_type(mask_type);
                    builder.with_input_type(input_type);

                    if write_to_mask {
                        builder.write_to_layout();
                    }

                    builder.to_shader(context)?
                };

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

    /// Whether we are using a texture as the mask.
    textured_mask: bool,

    /// Whether we are using a texture as the input.
    textured_input: bool,
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
            textured_input: false,
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
            "    
                vec2 finalPosition = (mvp * vec3({IN_POSITION}, 1.0)).xy; 
                gl_Position = vec4(finalPosition.x, -finalPosition.y, 0.0, 1.0);
            "
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

    /// Whether or not we write to `gl_FragColor` or just `color`.
    write_to_layout: bool,
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
            write_to_layout: false,
        }
    }

    /// Write to a color layout.
    fn write_to_layout(&mut self) -> &mut Self {
        self.write_to_layout = true;

        writeln!(self.source, "layout(location = 0) out vec4 {OUTPUT_COLOR};").ok();

        self
    }

    /// Use with the provided input type.
    fn with_input_type(&mut self, ty: InputType) -> &mut Self {
        match ty {
            InputType::Empty => self.with_empty_color(),
            InputType::Solid => self.with_solid_color(),
            InputType::Linear => self.with_linear_gradient(),
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

    /// Use with a linear gradient.
    fn with_linear_gradient(&mut self) -> &mut Self {
        writeln!(
            self.source,
            "
            uniform sampler2D {GRADIENT_STOPS};
            uniform sampler2D {GRADIENT_COLORS};
            uniform vec2 {LINEAR_GRADIENT_START};
            uniform vec2 {LINEAR_GRADIENT_END};

            float {GET_GRADIENT_COORD}(vec2 pos) {{
                vec2 start = {LINEAR_GRADIENT_START};
                vec2 end = {LINEAR_GRADIENT_END};
                vec2 diff = end - start;
                float len = length(diff);
                float dot = dot(diff, pos - start);
                return dot / len;
            }}

            vec4 {GET_COLOR}() {{
                float coord = {GET_GRADIENT_COORD}(gl_FragCoord.xy);
                float stop = texture2D({GRADIENT_STOPS}, vec2(coord, 0)).r;
                vec4 color = texture2D({GRADIENT_COLORS}, vec2(stop, 0));
                return color;
            }}
            "
        )
        .ok();

        todo!();

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
                return texture2D({TEXTURE_MASK}, {MASK_COORDS}).a; 
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
                return 1.0;
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
        let color_output = if self.write_to_layout {
            OUTPUT_COLOR
        } else {
            "gl_FragColor"
        };
        writeln!(
            source,
            "
            void main() {{
                {color_output} = ({GET_COLOR}() * {GET_MASK_ALPHA}());
            }}
            ",
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
