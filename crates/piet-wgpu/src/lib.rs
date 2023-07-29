// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the
// terms of either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
//   version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
// * The Patron License (https://github.com/notgull/piet-hardware/blob/main/LICENSE-PATRON.md)
//   for sponsors and contributors, who can ignore the copyleft provisions of the above licenses
//   for this project.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License or the Mozilla Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/>.

//! A GPU-accelerated 2D graphics backend for [`piet`] that uses the [`wgpu`] crate.
//!
//! This crate follows the [`wgpu` middleware pattern], but in a somewhat unique way.
//!
//! - The user creates the [`WgpuContext`] by calling `new()` with a device and expected texture
//!   format.
//! - Before rendering, the user creates a [`RenderContext`] by calling `prepare()` on the
//!   [`WgpuContext`] with a [`Device`] and a [`Queue`]. `prepare()` returns the context,
//!   which is expected to be written to.
//! - Finally, by calling `render` on the [`WgpuContext`], the user renders all of the material
//!   that was written to the [`RenderContext`] using the [`piet`] API.
//!
//! [`piet`]: https://crates.io/crates/piet
//! [`wgpu`]: https://crates.io/crates/wgpu
//! [`wgpu` middleware pattern]: https://github.com/gfx-rs/wgpu/wiki/Encapsulating-Graphics-Work
//! [`WgpuContext`]: struct.WgpuContext.html
//! [`RenderContext`]: struct.RenderContext.html
//! [`Device`]: wgpu::Device
//! [`Queue`]: wgpu::Queue

#![forbid(unsafe_code, rust_2018_idioms)]

pub use piet_hardware::piet;
pub use wgpu;

use piet_hardware::piet::kurbo::Affine;
use piet_hardware::piet::{Color, Error as Pierror, ImageFormat, InterpolationMode};

mod buffer;
mod context;
mod texture;

use context::GpuContext;

/// A wrapper around internal cached state.
#[derive(Debug)]
pub struct WgpuContext {
    /// The internal context.
    source: piet_hardware::Source<GpuContext>,

    /// The text.
    text: Text,
}

impl WgpuContext {
    /// Create a new [`wgpu`]-based drawing context.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        samples: u32,
    ) -> Self {
        let source = piet_hardware::Source::new(
            GpuContext::new(device, format, depth_format, samples),
            device,
            queue,
        )
        .expect("failed to create GPU context");
        let text = Text(source.text().clone());
        Self { source, text }
    }

    /// Prepare rendering by drawing to a [`RenderContext`].
    ///
    /// After this method is called, drawing is expected to be done to the returned context.
    pub fn prepare<'this, 'dev, 'que>(
        &'this mut self,
        device: &'dev wgpu::Device,
        queue: &'que wgpu::Queue,
        width: u32,
        height: u32,
    ) -> RenderContext<'this, 'dev, 'que> {
        RenderContext {
            context: self.source.render_context(device, queue, width, height),
            text: &mut self.text,
        }
    }

    /// Render the contents of the [`RenderContext`] to the provided render pass.
    pub fn render<'this>(&'this self, pass: &mut wgpu::RenderPass<'this>) {
        self.source.context().render(pass);
    }

    /// Call this function after you call `wgpu::Queue::submit` to free up resources.
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        self.source.gpu_flushed();
        self.source.context_mut().gpu_flushed(device);
    }
}

/// The whole point.
#[derive(Debug)]
pub struct RenderContext<'context, 'device, 'queue> {
    context: piet_hardware::RenderContext<'context, 'device, 'queue, GpuContext>,
    text: &'context mut Text,
}

impl RenderContext<'_, '_, '_> {
    /// Get the flattening tolerance.
    #[inline]
    pub fn tolerance(&self) -> f64 {
        self.context.tolerance()
    }

    /// Set the flattening tolerance.
    #[inline]
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.context.set_tolerance(tolerance)
    }

    /// Get the bitmap scale.
    #[inline]
    pub fn bitmap_scale(&self) -> f64 {
        self.context.bitmap_scale()
    }

    /// Set the bitmap scale.
    #[inline]
    pub fn set_bitmap_scale(&mut self, scale: f64) {
        self.context.set_bitmap_scale(scale)
    }
}

impl piet::RenderContext for RenderContext<'_, '_, '_> {
    type Brush = Brush;
    type Image = Image;
    type Text = Text;
    type TextLayout = TextLayout;

    fn blurred_rect(
        &mut self,
        rect: piet::kurbo::Rect,
        blur_radius: f64,
        brush: &impl piet::IntoBrush<Self>,
    ) {
        let brush = brush.make_brush(self, || rect);
        self.context
            .blurred_rect(rect, blur_radius, &brush.as_ref().0)
    }

    fn capture_image_area(
        &mut self,
        src_rect: impl Into<piet::kurbo::Rect>,
    ) -> Result<Self::Image, Pierror> {
        self.context.capture_image_area(src_rect).map(Image)
    }

    fn clear(&mut self, region: impl Into<Option<piet::kurbo::Rect>>, color: Color) {
        self.context.clear(region, color)
    }

    fn clip(&mut self, shape: impl piet::kurbo::Shape) {
        self.context.clip(shape)
    }

    fn current_transform(&self) -> Affine {
        self.context.current_transform()
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<piet::kurbo::Rect>,
        interp: InterpolationMode,
    ) {
        self.context.draw_image(&image.0, dst_rect, interp)
    }

    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<piet::kurbo::Rect>,
        dst_rect: impl Into<piet::kurbo::Rect>,
        interp: InterpolationMode,
    ) {
        self.context
            .draw_image_area(&image.0, src_rect, dst_rect, interp)
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<piet::kurbo::Point>) {
        self.context.draw_text(&layout.0, pos)
    }

    fn fill(&mut self, shape: impl piet::kurbo::Shape, brush: &impl piet::IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context.fill(shape, &brush.as_ref().0)
    }

    fn fill_even_odd(
        &mut self,
        shape: impl piet::kurbo::Shape,
        brush: &impl piet::IntoBrush<Self>,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context.fill_even_odd(shape, &brush.as_ref().0)
    }

    fn finish(&mut self) -> Result<(), Pierror> {
        self.context.finish()
    }

    fn gradient(
        &mut self,
        gradient: impl Into<piet::FixedGradient>,
    ) -> Result<Self::Brush, Pierror> {
        self.context.gradient(gradient).map(Brush)
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image, Pierror> {
        self.context
            .make_image(width, height, buf, format)
            .map(Image)
    }

    fn restore(&mut self) -> Result<(), Pierror> {
        self.context.restore()
    }

    fn save(&mut self) -> Result<(), Pierror> {
        self.context.save()
    }

    fn solid_brush(&mut self, color: Color) -> Self::Brush {
        Brush(self.context.solid_brush(color))
    }

    fn status(&mut self) -> Result<(), Pierror> {
        self.context.status()
    }

    fn stroke(
        &mut self,
        shape: impl piet::kurbo::Shape,
        brush: &impl piet::IntoBrush<Self>,
        width: f64,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context.stroke(shape, &brush.as_ref().0, width)
    }

    fn stroke_styled(
        &mut self,
        shape: impl piet::kurbo::Shape,
        brush: &impl piet::IntoBrush<Self>,
        width: f64,
        style: &piet::StrokeStyle,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        self.context
            .stroke_styled(shape, &brush.as_ref().0, width, style)
    }

    fn text(&mut self) -> &mut Self::Text {
        self.text
    }

    fn transform(&mut self, transform: Affine) {
        self.context.transform(transform)
    }
}

/// The brush type.
#[derive(Debug)]
pub struct Brush(piet_hardware::Brush<GpuContext>);

impl Clone for Brush {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl piet::IntoBrush<RenderContext<'_, '_, '_>> for Brush {
    fn make_brush<'a>(
        &'a self,
        _piet: &mut RenderContext<'_, '_, '_>,
        _bbox: impl FnOnce() -> piet::kurbo::Rect,
    ) -> std::borrow::Cow<'a, Brush> {
        std::borrow::Cow::Borrowed(self)
    }
}

/// The image type.
#[derive(Debug)]
pub struct Image(piet_hardware::Image<GpuContext>);

impl Clone for Image {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl piet::Image for Image {
    fn size(&self) -> piet::kurbo::Size {
        self.0.size()
    }
}

/// The text layout type.
#[derive(Debug, Clone)]
pub struct TextLayout(piet_hardware::TextLayout);

impl piet::TextLayout for TextLayout {
    fn size(&self) -> piet::kurbo::Size {
        self.0.size()
    }

    fn line_text(&self, line_number: usize) -> Option<&str> {
        self.0.line_text(line_number)
    }

    fn line_metric(&self, line_number: usize) -> Option<piet::LineMetric> {
        self.0.line_metric(line_number)
    }

    fn line_count(&self) -> usize {
        self.0.line_count()
    }

    fn hit_test_point(&self, point: piet::kurbo::Point) -> piet::HitTestPoint {
        self.0.hit_test_point(point)
    }

    fn trailing_whitespace_width(&self) -> f64 {
        self.0.trailing_whitespace_width()
    }

    fn image_bounds(&self) -> piet::kurbo::Rect {
        self.0.image_bounds()
    }

    fn text(&self) -> &str {
        self.0.text()
    }

    fn hit_test_text_position(&self, idx: usize) -> piet::HitTestPosition {
        self.0.hit_test_text_position(idx)
    }
}

/// The text layout builder type.
#[derive(Debug)]
pub struct TextLayoutBuilder(piet_hardware::TextLayoutBuilder);

impl piet::TextLayoutBuilder for TextLayoutBuilder {
    type Out = TextLayout;

    fn max_width(self, width: f64) -> Self {
        Self(self.0.max_width(width))
    }

    fn alignment(self, alignment: piet::TextAlignment) -> Self {
        Self(self.0.alignment(alignment))
    }

    fn default_attribute(self, attribute: impl Into<piet::TextAttribute>) -> Self {
        Self(self.0.default_attribute(attribute))
    }

    fn range_attribute(
        self,
        range: impl std::ops::RangeBounds<usize>,
        attribute: impl Into<piet::TextAttribute>,
    ) -> Self {
        Self(self.0.range_attribute(range, attribute))
    }

    fn build(self) -> Result<Self::Out, Pierror> {
        Ok(TextLayout(self.0.build()?))
    }
}

/// The text engine type.
#[derive(Debug, Clone)]
pub struct Text(piet_hardware::Text);

impl Text {
    /// Get the DPI scale.
    pub fn dpi(&self) -> f64 {
        self.0.dpi()
    }

    /// Set the DPI scale.
    pub fn set_dpi(&mut self, dpi: f64) {
        self.0.set_dpi(dpi)
    }
}

impl piet::Text for Text {
    type TextLayoutBuilder = TextLayoutBuilder;
    type TextLayout = TextLayout;

    fn font_family(&mut self, family_name: &str) -> Option<piet::FontFamily> {
        self.0.font_family(family_name)
    }

    fn load_font(&mut self, data: &[u8]) -> Result<piet::FontFamily, Pierror> {
        self.0.load_font(data)
    }

    fn new_text_layout(&mut self, text: impl piet::TextStorage) -> Self::TextLayoutBuilder {
        TextLayoutBuilder(self.0.new_text_layout(text))
    }
}
