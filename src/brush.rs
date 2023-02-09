//! Handles the brush setup.

#![allow(clippy::wrong_self_convention)]

use crate::{RenderContext, GlVersion, Error};
use crate::resources::{Program, Shader, Vertex, Fragment, Texture};

use glow::HasContext;
use piet::{FixedLinearGradient, FixedRadialGradient, IntoBrush};
use piet::kurbo::{Rect, Affine};

use std::borrow::Cow;
use std::collections::hash_map::{HashMap};
use std::fmt::Write;
use std::mem;
use std::rc::Rc;

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
pub(super) struct Mask<H: HasContext + ?Sized> {
    /// The mask texture.
    pub(super) texture: Texture<H>,

    /// The mask transform.
    pub(super) transform: Affine,
}

/// The type of input for a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InputType {
    Solid,
    Linear,
    Radial
}

/// Whether or not we use a mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MaskType {
    NoMask,
    Texture
}

/// Lookup key for a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShaderKey {
    input_type: InputType,
    mask_type: MaskType
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
            shaders: HashMap::new()
        }
    }

    pub(super) fn shader(&mut self,
        brush: &Brush,
        mask: Option<&Mask<H>>,)
        -> Result<&mut Program<H>, Error> {
            todo!()
        }
}

const SHADER_SOURCE_CAPACITY: usize = 1024;

/// A builder for the source code of a shader.
struct VertexBuilder {
    /// The source code.
    source: String,

    /// Whether we are using a texture.
    texture: bool,
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
                writeln!(source, "layout(location = 0) in vec2 position;").unwrap();
                writeln!(source, "uniform mat3 mvp;").unwrap();

                source
            },
            texture: false
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
        self.texture = true;
        writeln!(self.source, "uniform mat3 maskMvp;").unwrap();
        writeln!(self.source, "out vec2 texCoords;").unwrap();
        self
    }

    /// Build the shader source.
    fn to_source(&mut self) -> String {
        let mut source = mem::take(&mut self.source);

        // Write the main function.
        writeln!(source, "void main() {{").unwrap();
        writeln!(source, "    gl_Position = vec4((mvp * vec3(position, 1.0)).xy, 0.0, 1.0);").unwrap();

        if self.texture {
            writeln!(source, "    texCoords = (maskMvp * vec3(position, 1.0)).xy;").unwrap();
        }

        writeln!(source, "}}").unwrap();

        source
    }

    /// Make a new shader.
    fn to_shader<H: HasContext + ?Sized>(&mut self, ctx: &Rc<H>) -> Result<Shader<H, Vertex>, Error> {
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
