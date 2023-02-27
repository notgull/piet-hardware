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

//! Handles the brush setup.
//!
//! TODO: Right now, gradient rendering is done by first rendering the gradient in tiny-skia and
//! then using that as a texture. It's possible that this could be done on-the-fly in the shader.
//! However, I can't figure out a way of doing this that isn't littered with branches, which would
//! probably be slower than the current solution.

#![allow(clippy::wrong_self_convention)]

use crate::resources::{BoundProgram, BoundTexture, Fragment, Program, Shader, Texture, Vertex};
use crate::{Error, GlVersion, RenderContext};

use glow::HasContext;
use piet::kurbo::{Affine, Point, Rect, Size};
use piet::{FixedLinearGradient, FixedRadialGradient, Image as _, InterpolationMode, IntoBrush};

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::hash_map::{Entry, HashMap};
use std::fmt::Write;
use std::mem;
use std::rc::Rc;

use tiny_skia::{GradientStop, LinearGradient, Pixmap, RadialGradient};

// Various variable/function names used in GLSL.
const IN_POSITION: &str = "position";
const OUTPUT_COLOR: &str = "outputColor";
const TEXTURE_COORDS: &str = "textureCoords";

const LINEAR_GRADIENT_START: &str = "linearGradientStart";
const LINEAR_GRADIENT_END: &str = "linearGradientEnd";
const GRADIENT_COLORS: &str = "gradientColors";
const GRADIENT_STOPS: &str = "gradientStops";
const GRADIENT_CURRENT_COORD: &str = "gradientCurrentCoord";
const GRADIENT_STOP_COUNT: &str = "gradientStopCount";

const MVP: &str = "mvp";
const MVP_INVERSE: &str = "mvpInverse";
const SOLID_COLOR: &str = "solidColor";
const SRC_SIZE: &str = "srcSize";
const DST_RECT: &str = "dstRect";

const MASK_MVP: &str = "maskMvp";
const MASK_COORDS: &str = "maskCoords";
const TEXTURE_MASK: &str = "textureMask";

const TEXTURE: &str = "texture";
const TEXTURE_TRANSFORM: &str = "textureTransform";
const TEXTURE_REVERSE_TRANSFORM: &str = "textureRevTransform";
const TEX_COORDS: &str = "texCoords";

const GET_COLOR: &str = "getColor";
const GET_MASK_ALPHA: &str = "getMaskAlpha";
const GET_GRADIENT_COORD: &str = "getGradientCoord";

/// The brush type used by the [`RenderContext`].
pub struct Brush<H: HasContext + ?Sized>(BrushInner<H>);

enum BrushInner<H: HasContext + ?Sized> {
    /// A solid color.
    Solid(piet::Color),

    /// Linear gradient.
    LinearGradient {
        gradient: FixedLinearGradient,
        cached_texture: RefCell<Option<(Rc<Texture<H>>, Size)>>,
    },

    /// Radial gradient.
    RadialGradient {
        gradient: FixedRadialGradient,
        cached_texture: RefCell<Option<(Rc<Texture<H>>, Size)>>,
    },

    /// A texture.
    Texture {
        /// The texture.
        texture: Rc<Texture<H>>,

        /// The matrix mapping the destination rectangle to the source rectangle.
        dst_to_src: Affine,
    },
}

impl<H: HasContext + ?Sized> Clone for Brush<H> {
    fn clone(&self) -> Self {
        match &self.0 {
            BrushInner::Solid(color) => Brush::solid(*color),
            BrushInner::LinearGradient {
                gradient,
                cached_texture,
            } => {
                let cached_texture = cached_texture.borrow().clone();

                Brush(BrushInner::LinearGradient {
                    gradient: gradient.clone(),
                    cached_texture: RefCell::new(cached_texture),
                })
            }
            BrushInner::RadialGradient {
                gradient,
                cached_texture,
            } => {
                let cached_texture = cached_texture.borrow().clone();

                Brush(BrushInner::RadialGradient {
                    gradient: gradient.clone(),
                    cached_texture: RefCell::new(cached_texture),
                })
            }
            BrushInner::Texture {
                texture,
                dst_to_src,
            } => Brush(BrushInner::Texture {
                texture: texture.clone(),
                dst_to_src: *dst_to_src,
            }),
        }
    }
}

impl<H: HasContext + ?Sized> Brush<H> {
    /// Create a new solid color brush.
    pub fn solid(color: piet::Color) -> Self {
        Brush(BrushInner::Solid(color))
    }

    /// Create a new linear gradient brush.
    pub fn linear_gradient(gradient: FixedLinearGradient) -> Self {
        Brush(BrushInner::LinearGradient {
            gradient,
            cached_texture: RefCell::new(None),
        })
    }

    /// Create a new radial gradient brush.
    pub fn radial_gradient(gradient: FixedRadialGradient) -> Self {
        Brush(BrushInner::RadialGradient {
            gradient,
            cached_texture: RefCell::new(None),
        })
    }

    pub(super) fn textured(image: &crate::Image<H>, src: Rect, dst: Rect) -> Self {
        Brush(BrushInner::Texture {
            texture: image.texture.clone(),
            dst_to_src: texture_transform(src, dst, image.size()),
        })
    }

    fn input_type(&self) -> InputType {
        match &self.0 {
            BrushInner::Solid(_) => InputType::Solid,
            _ => InputType::Texture,
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
    Solid,
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
        brush: &Brush<H>,
        mvp: &Affine,
        mask: Option<&Mask<'_, H>>,
        window_size: Size,
    ) -> Result<BoundUniformProgram<'_, H>, Error> {
        let shader = self.shader_for_brush(context, version, brush, mask)?;

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
        let textured_uniforms = if matches!(brush.input_type(), InputType::Texture) {
            Some((
                shader.uniform_location(TEXTURE)?.clone(),
                shader.uniform_location(TEXTURE_TRANSFORM)?.clone(),
            ))
        } else {
            None
        };

        let program = shader.bind();
        let mut bound_texture = None;

        // Set the MVP.
        program.register_mat3(&mvp_uniform, mvp);

        // Set the Mask values.
        if let (Some(mask), Some((mask_mvp_uniform, texture_mask_uniform))) = (mask, mask_uniforms)
        {
            let total_transform = *mask.transform * sized_transform(window_size);
            program.register_mat3(&mask_mvp_uniform, &total_transform);

            let mut bound = mask.texture.bind(Some(0));
            program.register_texture(&texture_mask_uniform, &mut bound);
        }

        // Set the solid color.
        if let (Some(solid_color_uniform), Brush(BrushInner::Solid(color))) =
            (solid_color_uniform, &brush)
        {
            program.register_color(&solid_color_uniform, *color);
        }

        // Set the image transforms.
        if let Some((texture_uniform, texture_transform_uniform)) = textured_uniforms {
            let (texture, dst_to_src) = match brush.0 {
                BrushInner::LinearGradient {
                    ref gradient,
                    ref cached_texture,
                } => {
                    let mut cache = cached_texture.borrow_mut();
                    let texture = match &mut *cache {
                        Some((texture, size)) if size_approx_eq(*size, window_size) => texture,
                        _ => {
                            let texture = linear_gradient_texture(context, gradient, window_size)?;
                            &mut cache.insert((Rc::new(texture), window_size)).0
                        }
                    };

                    (texture.clone(), sized_transform(window_size))
                }

                BrushInner::RadialGradient {
                    ref gradient,
                    ref cached_texture,
                } => {
                    let mut cache = cached_texture.borrow_mut();
                    let texture = match &mut *cache {
                        Some((texture, size)) if size_approx_eq(*size, window_size) => texture,
                        _ => {
                            let texture = radial_gradient_texture(context, gradient, window_size)?;
                            &mut cache.insert((Rc::new(texture), window_size)).0
                        }
                    };

                    (texture.clone(), sized_transform(window_size))
                }

                BrushInner::Texture {
                    ref texture,
                    dst_to_src,
                } => (texture.clone(), dst_to_src),

                _ => unreachable!(),
            };

            {
                let mut bound = texture.bind_rc(Some(0));
                program.register_texture(&texture_uniform, &mut bound);
                bound_texture = Some(bound);
            }

            program.register_mat3(&texture_transform_uniform, &dst_to_src);
        }

        Ok(BoundUniformProgram {
            _bound_program: program,
            _bound_texture: bound_texture,
        })
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
                    .with_input_type(input_type)
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

pub struct BoundUniformProgram<'prog, H: HasContext + ?Sized> {
    _bound_program: BoundProgram<'prog, H>,
    _bound_texture: Option<BoundTexture<H, Rc<Texture<H>>>>,
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

    /// Uses the specified input.
    fn with_input_type(&mut self, input_type: InputType) -> &mut Self {
        match input_type {
            InputType::Texture => self.with_texture_input(),
            _ => self,
        }
    }

    /// Adds a texture color input.
    fn with_texture_input(&mut self) -> &mut Self {
        self.textured_input = true;

        writeln!(self.source, "uniform mat3 {TEXTURE_TRANSFORM};").unwrap();
        writeln!(self.source, "out vec2 {TEX_COORDS};").unwrap();

        self
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
                vec3 finalPosition = {MVP} * vec3({IN_POSITION}, 1.0);
                finalPosition /= finalPosition.z;
                gl_Position = vec4(finalPosition.x, -finalPosition.y, 0.0, 1.0);
            "
        )
        .unwrap();

        if self.textured_input {
            // Set up tex coords.
            writeln!(
                source,
                "
                    vec3 tCoords = {TEXTURE_TRANSFORM} * vec3({IN_POSITION}, 1.0);
                    tCoords /= tCoords.z;
                    {TEX_COORDS} = tCoords.xy;
                "
            )
            .unwrap();
        }

        if self.textured_mask {
            // Set up tex coords.
            writeln!(
                source,
                "    
                    vec3 mCoords = {MASK_MVP} * vec3({IN_POSITION}, 1.0);
                    mCoords /= mCoords.z;
                    {MASK_COORDS} = mCoords.xy;
                "
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
            InputType::Solid => self.with_solid_color(),
            InputType::Texture => self.with_texture_input(),
        }
    }

    /// Use a solid color.
    fn with_solid_color(&mut self) -> &mut Self {
        writeln!(
            self.source,
            "
            uniform vec4 {SOLID_COLOR};

            vec4 {GET_COLOR}() {{
                vec4 color = {SOLID_COLOR};
                return color;
            }}
        "
        )
        .ok();

        self
    }

    /// Use a texture input.
    fn with_texture_input(&mut self) -> &mut Self {
        writeln!(
            self.source,
            "
            in vec2 {TEX_COORDS};
            uniform sampler2D {TEXTURE};

            vec4 {GET_COLOR}() {{
                vec4 texColor = texture2D({TEXTURE}, {TEX_COORDS});
                return texColor;
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
                vec2 coords = {MASK_COORDS};
                return texture2D({TEXTURE_MASK}, coords).r;
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
                vec4 colorOutput = {GET_COLOR}();
                float alphaMask = {GET_MASK_ALPHA}();
                colorOutput.a *= alphaMask;
                {color_output} = colorOutput;
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

/// Create a linear gradient as a texture.
fn linear_gradient_texture<H: HasContext + ?Sized>(
    context: &Rc<H>,
    gradient: &FixedLinearGradient,
    window_size: Size,
) -> Result<Texture<H>, Error> {
    let linear_gradient = tiny_skia::LinearGradient::new(
        convert_point(gradient.start),
        convert_point(gradient.end),
        gradient
            .stops
            .iter()
            .map(|stop| tiny_skia::GradientStop::new(stop.pos, convert_color(stop.color)))
            .collect(),
        tiny_skia::SpreadMode::Pad,
        tiny_skia::Transform::identity(),
    )
    .ok_or_else(|| Error::BackendError("Failed to create linear gradient".into()))?;

    convert_shader(context, linear_gradient, window_size)
}

/// Create a radial gradient as a texture.
fn radial_gradient_texture<H: HasContext + ?Sized>(
    context: &Rc<H>,
    gradient: &FixedRadialGradient,
    window_size: Size,
) -> Result<Texture<H>, Error> {
    let radial_gradient = tiny_skia::RadialGradient::new(
        convert_point(gradient.center),
        {
            let end = gradient.center + gradient.origin_offset;
            convert_point(end)
        },
        gradient.radius as f32,
        gradient
            .stops
            .iter()
            .map(|stop| tiny_skia::GradientStop::new(stop.pos, convert_color(stop.color)))
            .collect(),
        tiny_skia::SpreadMode::Pad,
        tiny_skia::Transform::identity(),
    )
    .ok_or_else(|| Error::BackendError("Failed to create radial gradient".into()))?;

    convert_shader(context, radial_gradient, window_size)
}

/// Convert a `tiny_skia` shader to a texture.
fn convert_shader<H: HasContext + ?Sized>(
    context: &Rc<H>,
    shader: tiny_skia::Shader<'_>,
    window_size: Size,
) -> Result<Texture<H>, Error> {
    let mut pixmap = tiny_skia::Pixmap::new(window_size.width as _, window_size.height as _)
        .ok_or_else(|| Error::BackendError("Failed to create pixmap".into()))?;
    let paint = tiny_skia::Paint {
        shader,
        ..Default::default()
    };

    // Draw the gradient.
    let rect =
        tiny_skia::Rect::from_xywh(0.0, 0.0, window_size.width as _, window_size.height as _)
            .unwrap();
    pixmap
        .fill_rect(rect, &paint, Default::default(), None)
        .ok_or_else(|| Error::BackendError("Failed to draw gradient".into()))?;

    // Get the bytes.
    let bytes = pixmap.data();

    // Create the texture.
    let texture = Texture::new(context)?;

    // Upload the texture.
    texture.bind(None).fill_with_image(
        window_size.width as _,
        window_size.height as _,
        piet::ImageFormat::RgbaSeparate,
        bytes,
    )?;

    Ok(texture)
}

fn sized_transform(size: Size) -> Affine {
    let rect = Rect::from_origin_size((0.0, 0.0), size);
    texture_transform(rect, rect, size)
}

/// A transform that maps a rectangle to another rectangle in a texture.
fn texture_transform(src: Rect, dst: Rect, image_size: Size) -> Affine {
    // First, translate the dst rectangle to the src rectangle.
    let dst_translate = Affine::translate((-dst.x0, -dst.y0));

    // Now, scale the dst rectangle to the src rectangle.
    let dst_scale =
        Affine::scale_non_uniform(src.width() / dst.width(), src.height() / dst.height());

    // Translate the dst rectangle to the draw rectangle.
    let src_translate = Affine::translate((src.x0, src.y0));

    // Finally, scale the dst rectangle to texture coordinates [0..1]
    let src_scale = Affine::scale_non_uniform(1.0 / image_size.width, 1.0 / image_size.height);

    dst_scale * src_scale * dst_translate * src_translate
}

fn convert_point(p: piet::kurbo::Point) -> tiny_skia::Point {
    tiny_skia::Point {
        x: p.x as f32,
        y: p.y as f32,
    }
}

fn convert_color(p: piet::Color) -> tiny_skia::Color {
    let (r, g, b, a) = p.as_rgba();
    tiny_skia::Color::from_rgba(r as _, g as _, b as _, a as _).unwrap()
}

fn size_approx_eq(a: Size, b: Size) -> bool {
    (a.width - b.width).abs() < 0.5 && (a.height - b.height).abs() < 0.5
}
