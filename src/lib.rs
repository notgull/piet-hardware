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

#![allow(clippy::uninlined_format_args)]

mod brush;
mod mask;
mod resources;
mod text;

use arrayvec::ArrayVec;
use glow::HasContext;
use tinyvec::TinyVec;

use lyon_tessellation::path::geom::euclid::default::Point2D;
use lyon_tessellation::path::PathEvent;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, LineCap, StrokeOptions,
    StrokeTessellator, StrokeVertex, VertexBuffers,
};

use piet::kurbo::{Affine, PathEl, Point, Rect, Shape, Size, Vec2};
use piet::{
    Error, FixedGradient, ImageFormat, InterpolationMode, IntoBrush, LineCap as PietLineCap,
    StrokeStyle,
};

use std::mem;

use std::rc::Rc;

pub use piet_cosmic_text;

#[doc(inline)]
pub use piet_cosmic_text::{Text, TextLayout, TextLayoutBuilder};

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

    /// Buffers for drawing.
    buffers: Buffers<H>,

    /// Text handler.
    text: Text,
}

struct Buffers<H: HasContext + ?Sized> {
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

impl<H: HasContext + ?Sized> Buffers<H> {
    fn push_buffers(&mut self) -> Result<(), Error> {
        // Bind the VAO and VBO.
        let mut vao = self.vao.bind();
        let mut vbo = self.vbo.bind(resources::BufferTarget::Array);
        let mut ibo = self.ibo.bind(resources::BufferTarget::ElementArray);

        // Upload the vertex data.
        vbo.upload_f32(bytemuck::cast_slice(&self.vertex_buffer.vertices));
        ibo.upload_u32(&self.vertex_buffer.indices);
        vao.attribute_ptr(&vbo);

        drop(vbo);

        // Draw the triangles.
        unsafe {
            vao.draw_triangles(self.vertex_buffer.indices.len());
        }

        Ok(())
    }
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
            buffers: Buffers {
                vao: resources::VertexArray::new(&context)?,
                vbo: resources::VertexBuffer::new(&context)?,
                ibo: resources::VertexBuffer::new(&context)?,
                fill_tesselator: FillTessellator::new(),
                stroke_tesselator: StrokeTessellator::new(),
                vertex_buffer: VertexBuffers::new(),
            },
            text: Text::new(),
            context,
            shader_cache: brush::Brushes::new(),
        })
    }

    /// Get a reference to the underlying context.
    pub fn get_ref(&self) -> &H {
        &self.context
    }

    /// Consume this structure and return the underlying context.
    ///
    /// If there are any live `Text` or `Image` objects, this will return
    /// `None`, since those objects hold a reference to the context.
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

    /// The current state of the render context.
    state: TinyVec<[RenderState<H>; 1]>,

    /// The last error recorded, or `Ok(())` if none have occurred.
    last_error: Result<(), Error>,
}

/// The current state of the render context.
struct RenderState<H: HasContext + ?Sized> {
    /// The current transform with OpenGL scaling in mind.
    gl_transform: Affine,

    /// The current transform without OpenGL scaling in mind.
    pixel_transform: Affine,

    /// The current clip.
    mask: Option<mask::Mask<H>>,
}

impl<H: HasContext + ?Sized> Default for RenderState<H> {
    fn default() -> Self {
        Self {
            gl_transform: Affine::IDENTITY,
            pixel_transform: Affine::IDENTITY,
            mask: None,
        }
    }
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
            gl.context.enable(glow::BLEND);
            gl.context
                .blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
        }

        Self {
            size: (width, height),
            gl,
            tolerance: 5.0,
            default_transform,
            state: TinyVec::from([RenderState {
                gl_transform: default_transform,
                pixel_transform: default_transform.inverse(),
                mask: None,
            }]),
            last_error: Ok(()),
        }
    }

    /// Set the tolerance for rasterizing paths.
    ///
    /// The tolerance is the maximum distance between the path and the rasterized
    /// version. Higher values will result in faster rendering, but lower quality.
    /// Lower values will result in slower rendering, but higher quality.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    fn context(&self) -> &H {
        self.gl.get_ref()
    }

    fn currrent_state(&self) -> &RenderState<H> {
        self.state.last().unwrap()
    }

    fn current_state_mut(&mut self) -> &mut RenderState<H> {
        self.state.last_mut().unwrap()
    }

    fn fill_impl(&mut self, shape: impl Shape, brush: &Brush<H>, fill_rule: FillRule) {
        // Convert the kurbo shape to a lyon path iterator.
        let path_events = convert_path(shape.path_elements(self.tolerance));

        let mut options = FillOptions::default();
        options.fill_rule = fill_rule;
        options.tolerance = self.tolerance as f32;

        self.gl.buffers.vertex_buffer.vertices.clear();
        self.gl.buffers.vertex_buffer.indices.clear();

        // Tessellate the path.
        if let Err(e) = self.gl.buffers.fill_tesselator.tessellate(
            path_events,
            &options,
            &mut BuffersBuilder::new(&mut self.gl.buffers.vertex_buffer, |x: FillVertex<'_>| {
                x.position()
            }),
        ) {
            self.last_error = Err(Error::BackendError(e.into()));
            return;
        }

        self.render_buffers(brush);
    }

    fn fill_rect(&mut self, rect: Rect, brush: &Brush<H>) {
        // Get the vertices and indices for the rectangle.
        const INDICES: &[u32] = &[0, 1, 2, 0, 2, 3];
        let vertices = [
            Point2D::new(rect.x0 as f32, rect.y0 as f32),
            Point2D::new(rect.x1 as f32, rect.y0 as f32),
            Point2D::new(rect.x1 as f32, rect.y1 as f32),
            Point2D::new(rect.x0 as f32, rect.y1 as f32),
        ];

        // Fill the buffers with them.
        self.gl.buffers.vertex_buffer.vertices.clear();
        self.gl.buffers.vertex_buffer.indices.clear();
        self.gl
            .buffers
            .vertex_buffer
            .vertices
            .extend_from_slice(&vertices);
        self.gl
            .buffers
            .vertex_buffer
            .indices
            .extend_from_slice(INDICES);

        self.render_buffers(brush);
    }

    fn render_buffers(&mut self, brush: &Brush<H>) {
        // Get a mask.
        let state = self.state.last_mut().unwrap();
        let RenderState {
            gl_transform, mask, ..
        } = state;

        if let Some(mask) = mask.as_mut() {
            mask.update_texture();
        }

        // Activate the program for this brush.
        let program = self.gl.shader_cache.with_target(
            &self.gl.context,
            self.gl.version,
            brush,
            gl_transform,
            mask.as_mut().map(|m| m.as_brush_mask()).as_ref(),
        );

        // Upload the vertex data.
        let result = program.and_then(|_program| self.gl.buffers.push_buffers());

        if let Err(e) = result {
            self.last_error = Err(e);
        }
    }

    fn bbox<'s>(&self, shape: &'s impl Shape) -> impl Fn() -> Rect + 's {
        let size = self.size;
        move || {
            let our_rect = Rect::from_origin_size((0.0, 0.0), (size.0 as f64, size.1 as f64));
            let shape_rect = shape.bounding_box();

            our_rect.intersect(shape_rect)
        }
    }
}

impl<'a, H: HasContext + ?Sized> piet::RenderContext for RenderContext<'a, H> {
    type Brush = Brush<H>;
    type Text = Text;
    type TextLayout = TextLayout;
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
        if region.is_some() || self.currrent_state().mask.is_some() {
            let brush = self.solid_brush(color);

            let rect = match region {
                Some(region) => region,
                None => Rect::new(0.0, 0.0, self.size.0 as f64, self.size.1 as f64),
            };

            self.fill_rect(rect, &brush);
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

        self.gl.buffers.vertex_buffer.vertices.clear();
        self.gl.buffers.vertex_buffer.indices.clear();

        // Tessellate the path.
        if let Err(e) = self.gl.buffers.stroke_tesselator.tessellate(
            path_events,
            &options,
            &mut BuffersBuilder::new(
                &mut self.gl.buffers.vertex_buffer,
                |x: StrokeVertex<'_, '_>| x.position(),
            ),
        ) {
            self.last_error = Err(Error::BackendError(e.into()));
            return;
        }

        // Render the buffers to the screen.
        let brush = brush.make_brush(self, self.bbox(&shape));
        self.render_buffers(brush.as_ref());
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, self.bbox(&shape));
        self.fill_impl(shape, brush.as_ref(), FillRule::NonZero);
    }

    fn fill_even_odd(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, self.bbox(&shape));
        self.fill_impl(shape, brush.as_ref(), FillRule::EvenOdd);
    }

    fn clip(&mut self, shape: impl Shape) {
        let context = self.gl.context.clone();
        let tolerance = self.tolerance;
        let (width, height) = self.size;
        let transform = self.currrent_state().pixel_transform;

        self.current_state_mut()
            .mask
            .get_or_insert_with(|| {
                match mask::Mask::new(&context, width, height, transform) {
                    Ok(mask) => mask,
                    Err(e) => {
                        // TODO
                        panic!("Failed to create mask: {:?}", e);
                    }
                }
            })
            .add_path(shape, tolerance);
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.gl.text
    }

    fn draw_text(&mut self, _layout: &Self::TextLayout, _pos: impl Into<Point>) {
        todo!()
    }

    fn save(&mut self) -> Result<(), Error> {
        // Add a new state to the stack.
        self.state.push(RenderState {
            gl_transform: self.default_transform,
            pixel_transform: Affine::IDENTITY,
            mask: None,
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
        self.current_state_mut().gl_transform *= transform;
        self.current_state_mut().pixel_transform *= transform;
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        Image::from_image(&self.gl.context, (width, height), format, buf)
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        let dst_rect = dst_rect.into();

        image.texture.bind(None).set_interpolation_mode(interp);

        let textured_brush = Brush::textured(
            image,
            Rect::new(0.0, 0.0, image.size.width, image.size.height),
            dst_rect,
        );

        self.fill_rect(dst_rect, &textured_brush);
    }

    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<Rect>,
        dst_rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        let (src_rect, dst_rect) = (src_rect.into(), dst_rect.into());
        image.texture.bind(None).set_interpolation_mode(interp);
        let textured_brush = Brush::textured(image, src_rect, dst_rect);

        self.fill_rect(dst_rect, &textured_brush);
    }

    fn capture_image_area(&mut self, _src_rect: impl Into<Rect>) -> Result<Self::Image, Error> {
        Err(Error::NotSupported)
    }

    fn blurred_rect(&mut self, _rect: Rect, _blur_radius: f64, _brush: &impl IntoBrush<Self>) {
        self.last_error = Err(Error::NotSupported);
    }

    fn current_transform(&self) -> Affine {
        self.currrent_state().gl_transform
    }
}

struct RestoreMask<'a, 'b, H: HasContext + ?Sized> {
    context: &'a mut RenderContext<'b, H>,
    mask: Option<mask::Mask<H>>,
}

impl<'a, 'b, H: HasContext + ?Sized> RestoreMask<'a, 'b, H> {
    fn new(context: &'a mut RenderContext<'b, H>) -> Self {
        let mask = context.current_state_mut().mask.take();
        Self { context, mask }
    }

    fn update_texture(&mut self) {
        if let Some(mask) = self.mask.as_mut() {
            mask.update_texture();
        }
    }
}

impl<H: HasContext + ?Sized> Drop for RestoreMask<'_, '_, H> {
    fn drop(&mut self) {
        self.context.current_state_mut().mask = self.mask.take();
    }
}

/// The image type used by the [`RenderContext`].
pub struct Image<H: HasContext + ?Sized> {
    /// The underlying texture.
    texture: Rc<resources::Texture<H>>,

    /// The size of the image.
    size: Size,
}

impl<H: HasContext + ?Sized> Clone for Image<H> {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            size: self.size,
        }
    }
}

impl<H: HasContext + ?Sized> piet::Image for Image<H> {
    fn size(&self) -> Size {
        self.size
    }
}

impl<H: HasContext + ?Sized> Image<H> {
    fn from_image(
        context: &Rc<H>,
        size: (usize, usize),
        format: ImageFormat,
        data: &[u8],
    ) -> Result<Self, Error> {
        // Create the texture.
        let tex = resources::Texture::new(context)?;
        let (width, height) = size;

        // Upload the data.
        {
            let mut bound = tex.bind(None);
            bound.fill_with_image(
                width.try_into().expect("width too large"),
                width.try_into().expect("height too large"),
                format,
                data,
            )?;
        }

        Ok(Self {
            texture: Rc::new(tex),
            size: Size::new(width as f64, height as f64),
        })
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
