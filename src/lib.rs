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

use glow::HasContext;
use tinyvec::TinyVec;

use piet::kurbo::{Affine, Point, Rect, Shape, Size, PathEl, BezPath};
use piet::{
    Error, FixedGradient, FontFamily, HitTestPoint, HitTestPosition, ImageFormat,
    InterpolationMode, IntoBrush, LineMetric, StrokeStyle, TextAlignment, TextAttribute,
    FixedLinearGradient, FixedRadialGradient
};

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
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
    pub unsafe fn new(context: H) -> Self
    where
        H: Sized,
    {
        Self::from_rc(Rc::new(context))
    }

    unsafe fn from_rc(context: Rc<H>) -> Self {
        Self {
            version: if context.version().is_embedded {
                GlVersion::Es30
            } else {
                GlVersion::Gl32
            },
            context,
        }
    }

    /// Get a reference to the underlying context.
    pub fn get_ref(&self) -> &H {
        &self.context
    }

    /// Consume this structure and return the underlying context.
    pub fn into_inner(self) -> Result<H, Self>
    where
        H: Sized,
    {
        let Self { context, .. } = self;
        Rc::try_unwrap(context).map_err(move |context| unsafe { Self::from_rc(context) })
    }
}

/// A rendering context that uses OpenGL for rendering.
///
/// See the [crate level documentation](index.html) for more information.
pub struct RenderContext<'a, H: HasContext + ?Sized> {
    /// Reference to the OpenGL context.
    gl: &'a mut GlContext<H>,

    /// Tolerance for rasterizing paths.
    tolerance: f64,

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
    pub unsafe fn new(gl: &'a mut GlContext<H>) -> Self {
        Self {
            text: Text {
                context: gl.context.clone(),
            },
            gl,
            tolerance: 5.0,
            state: TinyVec::from([RenderState::default()]),
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
                None => todo!()
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
        self.stroke_styled(
            shape,
            brush,
            width,
            &Default::default()
        );
    }

    fn stroke_styled(
        &mut self,
        shape: impl Shape,
        brush: &impl IntoBrush<Self>,
        width: f64,
        style: &StrokeStyle,
    ) {
        todo!()
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        todo!()
    }

    fn fill_even_odd(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        todo!()
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
        self.state.push(RenderState::default());
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

    fn capture_image_area(&mut self, src_rect: impl Into<Rect>) -> Result<Self::Image, Error> {
        todo!()
    }

    fn blurred_rect(&mut self, rect: Rect, blur_radius: f64, brush: &impl IntoBrush<Self>) {
        todo!()
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
