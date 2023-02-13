// This file is dual licensed under the MIT and Apache 2.0 licenses.

//! Creates a [`piet`] rendering context for an OpenGL context.
//!
//! The centerpiece of this crate is the [`RenderContext`] structure, which takes an OpenGL
//! [`HasContext`] provided by [`glow`] and uses the [`piet::RenderContext`] interface for
//! drawing. This provides a cross-platform alternative to [`piet-common`] for cases where
//! you have an OpenGL context available.
//!
//! The [`RenderContext`] assumes that the context is acquired before any drawing is done.
//!
//! [`piet`]: https://crates.io/crates/piet
//! [`HasContext`]: https://docs.rs/glow/latest/glow/trait.HasContext.html
//! [`glow`]: https://crates.io/crates/glow
//! [`piet::RenderContext`]: https://docs.rs/piet/latest/piet/trait.RenderContext.html
//! [`piet-common`]: https://crates.io/crates/piet-common

mod brush;
mod resources;

use arrayvec::ArrayVec;
use glow::HasContext;
use tinyvec::TinyVec;

use lyon_tessellation::path::geom::euclid::default::Point2D;
use lyon_tessellation::path::PathEvent;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, LineCap, LineJoin,
    StrokeOptions, StrokeTessellator, StrokeVertex, VertexBuffers,
};

use piet::kurbo::{Affine, BezPath, PathEl, Point, Rect, Shape, Size, Vec2};
use piet::{
    Error, FixedGradient, FontFamily, HitTestPoint, HitTestPosition, ImageFormat,
    InterpolationMode, IntoBrush, LineCap as PietLineCap, LineMetric, StrokeStyle, TextAlignment,
    TextAttribute,
};

use std::mem;
use std::ops::RangeBounds;
use std::rc::Rc;

pub use brush::Brush;

/// The OpenGL version that is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GlVersion {
    /// OpenGL 3.2 or higher.
    Gl32,

    /// OpenGL ES 3.0 or higher.
    Es30,
}

/// An OpenGL context paired with information that is useful for rendering.
pub struct GlContext<H: HasContext + ?Sized> {
    /// The OpenGL context.
    context: Rc<H>,

    /// The OpenGL version.
    version: GlVersion,

    /// Cache for shaders.
    shader_cache: brush::Brushes<H>,

    /// Fill-based tessellator.
    fill_tesselator: FillTessellator,

    /// Stroke-based tessellator.
    stroke_tesselator: StrokeTessellator,

    /// Buffer for vertices.
    vertex_buffer: VertexBuffers<Point2D<f32>, u32>,

    /// The VAO for the vertex buffer.
    vao: resources::VertexArray<H>,

    /// The VBO for the vertex buffer.
    vbo: resources::VertexBuffer<H>,

    /// The VBO for the index buffer.
    ibo: resources::VertexBuffer<H>,
}

impl<H: HasContext + ?Sized> GlContext<H> {
    /// Create a new `GlContext` from something that implements [`HasContext`].
    ///
    /// # Safety
    ///
    /// `context` should be the current context. The `context` is allowed to be released
    /// during the lifetime of this structure; however, when used to call `RenderContext::new()`
    /// it should be acquired again.
    ///
    /// [`HasContext`]: https://docs.rs/glow/latest/glow/trait.HasContext.html
    pub unsafe fn new(context: H) -> Result<Self, Error>
    where
        H: Sized,
    {
        Self::from_rc(Rc::new(context))
    }

    unsafe fn from_rc(context: Rc<H>) -> Result<Self, Error> {
        Ok(Self {
            version: if context.version().is_embedded {
                GlVersion::Es30
            } else {
                GlVersion::Gl32
            },
            vao: resources::VertexArray::new(&context)?,
            vbo: resources::VertexBuffer::new(&context)?,
            ibo: resources::VertexBuffer::new(&context)?,
            context,
            shader_cache: brush::Brushes::new(),
            fill_tesselator: FillTessellator::new(),
            stroke_tesselator: StrokeTessellator::new(),
            vertex_buffer: VertexBuffers::new(),
        })
    }

    /// Get a reference to the underlying context.
    pub fn get_ref(&self) -> &H {
        &self.context
    }

    /// Consume this structure and return the underlying context.
    pub fn into_inner(self) -> Option<H>
    where
        H: Sized,
    {
        let Self { context, .. } = self;
        Rc::try_unwrap(context).ok()
    }
}

/// A rendering context that uses OpenGL for rendering.
///
/// See the [crate level documentation](index.html) for more information.
pub struct RenderContext<'a, H: HasContext + ?Sized> {
    /// Reference to the OpenGL context.
    gl: &'a mut GlContext<H>,

    /// The current size of the framebuffer.
    size: (u32, u32),

    /// Tolerance for rasterizing paths.
    tolerance: f64,

    /// Default transform for fitting coordinate space to framebuffer.
    default_transform: Affine,

    /// The text manager.
    text: Text<H>,

    /// The current state of the render context.
    state: TinyVec<[RenderState; 1]>,

    /// The last error recorded, or `Ok(())` if none have occurred.
    last_error: Result<(), Error>,
}

/// The current state of the render context.
#[derive(Default)]
struct RenderState {
    /// The current transform.
    transform: Affine,

    /// The current clip.
    clip: Option<BezPath>,
}

impl<'a, H: HasContext + ?Sized> RenderContext<'a, H> {
    /// Create a new `RenderContext` from a context.
    ///
    /// # Safety
    ///
    /// The `GlContext` must be the current context. While this structure is active,
    /// it should be replaced as the current context.
    pub unsafe fn new(gl: &'a mut GlContext<H>, width: u32, height: u32) -> Self {
        // The transform needs to transform from the coordinate space to the framebuffer.
        // The framebuffer is (0, 0) at the top left, and (width, height) at the bottom right.
        // GL goes from (-1, -1) to (1, 1)
        let default_transform = Affine::translate(Vec2::new(-1.0, -1.0))
            * Affine::scale_non_uniform(2.0 / width as f64, 2.0 / height as f64);

        unsafe {
            gl.context.viewport(0, 0, width as i32, height as i32);
        }

        Self {
            text: Text {
                context: gl.context.clone(),
            },
            size: (width, height),
            gl,
            tolerance: 5.0,
            default_transform,
            state: TinyVec::from([RenderState {
                transform: default_transform,
                clip: None,
            }]),
            last_error: Ok(()),
        }
    }

    fn context(&self) -> &H {
        self.gl.get_ref()
    }

    fn currrent_state(&self) -> &RenderState {
        self.state.last().unwrap()
    }

    fn current_state_mut(&mut self) -> &mut RenderState {
        self.state.last_mut().unwrap()
    }

    fn fill_impl(&mut self, shape: impl Shape, brush: &Brush, fill_rule: FillRule) {
        // Convert the kurbo shape to a lyon path iterator.
        let path_events = convert_path(shape.path_elements(self.tolerance));

        let mut options = FillOptions::default();
        options.fill_rule = fill_rule;
        options.tolerance = self.tolerance as f32;

        self.gl.vertex_buffer.vertices.clear();
        self.gl.vertex_buffer.indices.clear();

        // Tessellate the path.
        if let Err(e) = self.gl.fill_tesselator.tessellate(
            path_events,
            &options,
            &mut BuffersBuilder::new(&mut self.gl.vertex_buffer, |x: FillVertex<'_>| x.position()),
        ) {
            self.last_error = Err(Error::BackendError(e.into()));
            return;
        }

        self.render_buffers(brush);
    }

    fn render_buffers(&mut self, brush: &Brush) {
        // Activate the program for this brush.
        let result = self.gl.shader_cache.with_brush(
            &self.gl.context,
            self.gl.version,
            brush,
            &self.state.last().unwrap().transform,
            None, //todo
            || {
                // Bind the VAO and VBO.
                let mut vao = self.gl.vao.bind();
                let mut vbo = self.gl.vbo.bind(resources::BufferTarget::Array);
                let mut ibo = self.gl.ibo.bind(resources::BufferTarget::ElementArray);

                // Upload the vertex data.
                vbo.upload_f32(bytemuck::cast_slice(&self.gl.vertex_buffer.vertices));
                ibo.upload_u32(&self.gl.vertex_buffer.indices);
                vao.attribute_ptr(&vbo);

                drop(vbo);

                // Draw the triangles.
                unsafe {
                    vao.draw_triangles(self.gl.vertex_buffer.indices.len());
                }

                Ok(())
            },
        );

        if let Err(e) = result {
            self.last_error = Err(e);
        }
    }

    fn bbox(&self) -> impl Fn() -> Rect + 'static {
        let size = self.size;
        move || Rect::from_origin_size((0.0, 0.0), (size.0 as f64, size.1 as f64))
    }
}

impl<'a, H: HasContext + ?Sized> piet::RenderContext for RenderContext<'a, H> {
    type Brush = Brush;
    type Text = Text<H>;
    type TextLayout = TextLayout<H>;
    type Image = Image<H>;

    fn status(&mut self) -> Result<(), Error> {
        mem::replace(&mut self.last_error, Ok(()))
    }

    fn solid_brush(&mut self, color: piet::Color) -> Self::Brush {
        Brush::solid(color)
    }

    fn gradient(&mut self, gradient: impl Into<FixedGradient>) -> Result<Self::Brush, Error> {
        match gradient.into() {
            FixedGradient::Linear(l) => Ok(Brush::linear_gradient(l)),
            FixedGradient::Radial(r) => Ok(Brush::radial_gradient(r)),
        }
    }

    fn clear(&mut self, region: impl Into<Option<Rect>>, color: piet::Color) {
        let region = region.into();

        // Fall back to fill if we're not clearing the entire screen.
        if region.is_some() || self.currrent_state().clip.is_some() {
            let brush = self.solid_brush(color);

            let rect = match region {
                Some(region) => region,
                None => Rect::new(0.0, 0.0, self.size.0 as f64, self.size.1 as f64),
            };

            self.fill(rect, &brush);
        }

        let (r, g, b, a) = color.as_rgba();

        // SAFETY: The GL context has been acquired
        unsafe {
            self.context()
                .clear_color(r as f32, g as f32, b as f32, a as f32);

            self.context().clear(glow::COLOR_BUFFER_BIT);
        }
    }

    fn stroke(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>, width: f64) {
        self.stroke_styled(shape, brush, width, &Default::default());
    }

    fn stroke_styled(
        &mut self,
        shape: impl Shape,
        brush: &impl IntoBrush<Self>,
        width: f64,
        style: &StrokeStyle,
    ) {
        // TODO: Support dashes
        if !style.dash_pattern.is_empty() {
            self.last_error = Err(Error::NotSupported);
            return;
        }

        let cvt_cap = |x: PietLineCap| match x {
            PietLineCap::Butt => LineCap::Butt,
            PietLineCap::Round => LineCap::Round,
            PietLineCap::Square => LineCap::Square,
        };

        // Convert the kurbo shape to a lyon path iterator.
        let path_events = convert_path(shape.path_elements(self.tolerance));

        // Create stroking options based on our style.
        let mut options = StrokeOptions::default();
        options.line_width = width as f32;
        options.start_cap = cvt_cap(style.line_cap);
        options.end_cap = cvt_cap(style.line_cap);

        self.gl.vertex_buffer.vertices.clear();
        self.gl.vertex_buffer.indices.clear();

        // Tessellate the path.
        if let Err(e) = self.gl.stroke_tesselator.tessellate(
            path_events,
            &options,
            &mut BuffersBuilder::new(&mut self.gl.vertex_buffer, |x: StrokeVertex<'_, '_>| {
                x.position()
            }),
        ) {
            self.last_error = Err(Error::BackendError(e.into()));
            return;
        }

        // Render the buffers to the screen.
        let brush = brush.make_brush(self, self.bbox());

        self.render_buffers(brush.as_ref());
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, self.bbox());
        self.fill_impl(shape, brush.as_ref(), FillRule::NonZero);
    }

    fn fill_even_odd(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, self.bbox());
        self.fill_impl(shape, brush.as_ref(), FillRule::EvenOdd);
    }

    fn clip(&mut self, shape: impl Shape) {
        self.current_state_mut().clip = Some(shape.into_path(self.tolerance));
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.text
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<Point>) {
        todo!()
    }

    fn save(&mut self) -> Result<(), Error> {
        // Add a new state to the stack.
        self.state.push(RenderState {
            transform: self.default_transform,
            clip: None,
        });
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Error> {
        if self.state.len() <= 1 {
            return Err(Error::StackUnbalance);
        }

        // Remove the last state from the stack.
        self.state.pop();
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Error> {
        // Flush the GL context.
        unsafe {
            self.context().flush();
        }

        Ok(())
    }

    fn transform(&mut self, transform: Affine) {
        self.current_state_mut().transform *= transform;
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        todo!()
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        todo!()
    }

    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<Rect>,
        dst_rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        todo!()
    }

    fn capture_image_area(&mut self, _src_rect: impl Into<Rect>) -> Result<Self::Image, Error> {
        Err(Error::NotSupported)
    }

    fn blurred_rect(&mut self, _rect: Rect, _blur_radius: f64, _brush: &impl IntoBrush<Self>) {
        self.last_error = Err(Error::NotSupported);
    }

    fn current_transform(&self) -> Affine {
        self.currrent_state().transform
    }
}

/// The text renderer used by the [`RenderContext`].
pub struct Text<H: HasContext + ?Sized> {
    context: Rc<H>,
}

impl<H: HasContext + ?Sized> Clone for Text<H> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
        }
    }
}

impl<H: HasContext + ?Sized> piet::Text for Text<H> {
    type TextLayout = TextLayout<H>;
    type TextLayoutBuilder = TextLayoutBuilder<H>;

    fn font_family(&mut self, family_name: &str) -> Option<FontFamily> {
        todo!()
    }

    fn load_font(&mut self, data: &[u8]) -> Result<FontFamily, Error> {
        todo!()
    }

    fn new_text_layout(&mut self, text: impl piet::TextStorage) -> Self::TextLayoutBuilder {
        todo!()
    }
}

/// The text layout builder used by the [`RenderContext`].
pub struct TextLayoutBuilder<H: HasContext + ?Sized> {
    context: Rc<H>,
}

impl<H: HasContext + ?Sized> Clone for TextLayoutBuilder<H> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
        }
    }
}

impl<H: HasContext + ?Sized> piet::TextLayoutBuilder for TextLayoutBuilder<H> {
    type Out = TextLayout<H>;

    fn alignment(self, alignment: TextAlignment) -> Self {
        todo!()
    }

    fn max_width(self, width: f64) -> Self {
        todo!()
    }

    fn default_attribute(self, attribute: impl Into<TextAttribute>) -> Self {
        todo!()
    }

    fn range_attribute(
        self,
        range: impl RangeBounds<usize>,
        attribute: impl Into<TextAttribute>,
    ) -> Self {
        todo!()
    }

    fn build(self) -> Result<Self::Out, Error> {
        todo!()
    }
}

/// The text layout used by the [`RenderContext`].
pub struct TextLayout<H: HasContext + ?Sized> {
    context: Rc<H>,
}

impl<H: HasContext + ?Sized> Clone for TextLayout<H> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
        }
    }
}

impl<H: HasContext + ?Sized> piet::TextLayout for TextLayout<H> {
    fn size(&self) -> Size {
        todo!()
    }

    fn trailing_whitespace_width(&self) -> f64 {
        todo!()
    }

    fn image_bounds(&self) -> Rect {
        todo!()
    }

    fn text(&self) -> &str {
        todo!()
    }

    fn line_count(&self) -> usize {
        todo!()
    }

    fn line_metric(&self, line_number: usize) -> Option<LineMetric> {
        todo!()
    }

    fn line_text(&self, line_number: usize) -> Option<&str> {
        todo!()
    }

    fn hit_test_point(&self, point: Point) -> HitTestPoint {
        todo!()
    }

    fn hit_test_text_position(&self, idx: usize) -> HitTestPosition {
        todo!()
    }
}

/// The image type used by the [`RenderContext`].
pub struct Image<H: HasContext + ?Sized> {
    context: Rc<H>,
}

impl<H: HasContext + ?Sized> Clone for Image<H> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
        }
    }
}

impl<H: HasContext + ?Sized> piet::Image for Image<H> {
    fn size(&self) -> Size {
        todo!()
    }
}

/// Convert `kurbo` paths to `lyon` paths.
fn convert_path(kurbo_shape: impl IntoIterator<Item = PathEl>) -> impl Iterator<Item = PathEvent> {
    kurbo_shape
        .into_iter()
        .scan(
            (None, None),
            |&mut (ref mut first_point, ref mut last_point), el| {
                Some(match el {
                    PathEl::MoveTo(p) => {
                        // Close off the previous path if we need to.
                        let start_ev = match (first_point.take(), last_point.take()) {
                            (Some(first), Some(last)) => Some(PathEvent::End {
                                last: convert_point(last),
                                first: convert_point(first),
                                close: false,
                            }),
                            _ => None,
                        };

                        *first_point = Some(p);
                        *last_point = Some(p);

                        let mut av = ArrayVec::<PathEvent, 2>::new();
                        av.extend(start_ev);
                        av.push(PathEvent::Begin {
                            at: convert_point(p),
                        });
                        av
                    }

                    PathEl::LineTo(last) => {
                        let first = last_point.replace(last).expect("invalid path");
                        one(PathEvent::Line {
                            from: convert_point(first),
                            to: convert_point(last),
                        })
                    }

                    PathEl::QuadTo(ctrl1, last) => {
                        let first = last_point.replace(last).expect("invalid path");
                        one(PathEvent::Quadratic {
                            from: convert_point(first),
                            ctrl: convert_point(ctrl1),
                            to: convert_point(last),
                        })
                    }

                    PathEl::CurveTo(ctrl1, ctrl2, last) => {
                        let first = last_point.replace(last).expect("invalid path");
                        one(PathEvent::Cubic {
                            from: convert_point(first),
                            ctrl1: convert_point(ctrl1),
                            ctrl2: convert_point(ctrl2),
                            to: convert_point(last),
                        })
                    }

                    PathEl::ClosePath => {
                        let first = first_point.take().expect("invalid path");
                        let last = last_point.replace(first).expect("invalid path");

                        one(PathEvent::End {
                            last: convert_point(last),
                            first: convert_point(first),
                            close: true,
                        })
                    }
                })
            },
        )
        .flatten()
}

fn one<T>(t: T) -> ArrayVec<T, 2> {
    let mut av = ArrayVec::new();
    av.push(t);
    av
}

fn convert_point(point: Point) -> Point2D<f32> {
    Point2D::new(point.x as f32, point.y as f32)
}
