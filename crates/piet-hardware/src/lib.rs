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

//! An adaptor for [`piet`] that allows it to take advantage of GPU acceleration.
//!
//! This crate provides common types, traits and functionality that should be useful for
//! implementations of the [`piet`] drawing framework for hardware-accelerated backends
//! like OpenGL, Vulkan and WGPU. It handles things like rasterization, atlas packing and
//! memory management, while leaving the actual implementation of the GPU commands to the
//! backend.
//!
//! To use, first implement the [`GpuContext`] trait on a type of your choice that represents
//! an active GPU context. Wrap this type in the [`Source`] type, and then use that to
//! create a [`RenderContext`]. From here, you can pass that type to your rendering code. It
//! conforms to the [`piet`] API, so you can use it as a drop-in replacement for any [`piet`]
//! backend, including [`piet-common`].
//!
//! Note that this crate generally uses thread-unsafe primitives. This is because UI management is
//! usually pinned to one thread anyways, and it's a bad idea to do drawing outside of that thread.
//!
//! ## Implementation
//!
//! This crate works first and foremost by converting drawing operations to a series of
//! triangles.

#![forbid(unsafe_code, rust_2018_idioms)]

use cosmic_text::LayoutGlyph;
use line_straddler::{LineGenerator, LineType};

use lyon_tessellation::FillRule;

pub use piet;
use piet::kurbo::{Affine, PathEl, Point, Rect, Shape, Size};
use piet::{Error as Pierror, FixedGradient, Image as _, InterpolationMode};

use piet_cosmic_text::Metadata;
use tinyvec::TinyVec;

use std::error::Error as StdError;
use std::fmt;
use std::mem;

mod atlas;
mod brush;
mod gpu_backend;
mod image;
mod mask;
mod rasterizer;
mod resources;
mod text;

pub use self::brush::Brush;
pub use self::gpu_backend::{BufferType, GpuContext, RepeatStrategy, Vertex};
pub use self::image::Image;
pub use self::text::{Text, TextLayout, TextLayoutBuilder};

pub(crate) use atlas::{Atlas, GlyphData};
pub(crate) use mask::MaskSlot;
pub(crate) use rasterizer::{Rasterizer, TessRect};
pub(crate) use resources::{Texture, VertexBuffer};

const UV_WHITE: [f32; 2] = [0.5, 0.5];

/// The source of the GPU renderer.
pub struct Source<C: GpuContext + ?Sized> {
    /// A texture that consists of an endless repeating pattern of a single white pixel.
    ///
    /// This is used for solid-color fills. It is also used as the mask for when a
    /// clipping mask is not defined.
    white_pixel: Texture<C>,

    /// The buffers used by the GPU renderer.
    buffers: Buffers<C>,

    /// The text API.
    text: Text,

    /// The font atlas.
    atlas: Option<Atlas<C>>,

    /// The context to use for the GPU renderer.
    context: C,
}

impl<C: GpuContext + fmt::Debug + ?Sized> fmt::Debug for Source<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Source")
            .field("context", &&self.context)
            .finish_non_exhaustive()
    }
}

struct Buffers<C: GpuContext + ?Sized> {
    /// The rasterizer for the GPU renderer.
    rasterizer: Rasterizer,

    /// The VBO for vertices.
    vbo: VertexBuffer<C>,
}

impl<C: GpuContext + ?Sized> Source<C> {
    /// Create a new source from a context.
    pub fn new(mut context: C, device: &C::Device, queue: &C::Queue) -> Result<Self, Pierror>
    where
        C: Sized,
    {
        const WHITE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];

        // Setup a white pixel texture.
        let texture = Texture::new(
            &mut context,
            device,
            InterpolationMode::NearestNeighbor,
            RepeatStrategy::Repeat,
        )
        .piet_err()?;

        texture.write_texture(
            &mut context,
            device,
            queue,
            (1, 1),
            piet::ImageFormat::RgbaSeparate,
            Some(&WHITE),
        );

        Ok(Self {
            white_pixel: texture,
            buffers: {
                let vbo = VertexBuffer::new(&mut context, device).piet_err()?;

                Buffers {
                    rasterizer: Rasterizer::new(),
                    vbo,
                }
            },
            atlas: Some(Atlas::new(&mut context, device, queue)?),
            context,
            text: Text::new(),
        })
    }

    /// Get a reference to the context.
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Get a mutable reference to the context.
    pub fn context_mut(&mut self) -> &mut C {
        &mut self.context
    }

    /// Create a new rendering context.
    pub fn render_context<'this, 'dev, 'que>(
        &'this mut self,
        device: &'dev C::Device,
        queue: &'que C::Queue,
        width: u32,
        height: u32,
    ) -> RenderContext<'this, 'dev, 'que, C> {
        RenderContext {
            source: self,
            device,
            queue,
            size: (width, height),
            state: TinyVec::from([RenderState::default()]),
            status: Ok(()),
            tolerance: 0.1,
        }
    }

    /// Get a reference to the text backend.
    pub fn text(&self) -> &Text {
        &self.text
    }

    /// Get a mutable reference to the text backend.
    pub fn text_mut(&mut self) -> &mut Text {
        &mut self.text
    }
}

/// The whole point of this crate.
pub struct RenderContext<'src, 'dev, 'que, C: GpuContext + ?Sized> {
    /// The source of the GPU renderer.
    source: &'src mut Source<C>,

    /// The device that we are rendering to.
    device: &'dev C::Device,

    /// The queue that we are rendering to.
    queue: &'que C::Queue,

    /// The width and height of the target.
    size: (u32, u32),

    /// The current state of the renderer.
    state: TinyVec<[RenderState<C>; 1]>,

    /// The result to use for `status`.
    status: Result<(), Pierror>,

    /// Tolerance for tesselation.
    tolerance: f64,
}

struct RenderState<C: GpuContext + ?Sized> {
    /// The current transform in pixel space.
    transform: Affine,

    /// The current clipping mask.
    mask: MaskSlot<C>,
}

impl<C: GpuContext + ?Sized> Default for RenderState<C> {
    fn default() -> Self {
        Self {
            transform: Affine::IDENTITY,
            mask: MaskSlot::new(),
        }
    }
}

impl<C: GpuContext + ?Sized> RenderContext<'_, '_, '_, C> {
    /// Fill in a rectangle.
    fn fill_rects(
        &mut self,
        rects: impl IntoIterator<Item = TessRect>,
        texture: Option<&Texture<C>>,
    ) -> Result<(), Pierror> {
        self.source.buffers.rasterizer.fill_rects(rects);

        // Push the buffers to the GPU.
        self.push_buffers(texture)
    }

    /// Fill in the provided shape.
    fn fill_impl(
        &mut self,
        shape: impl Shape,
        brush: &Brush<C>,
        mode: FillRule,
    ) -> Result<(), Pierror> {
        self.source
            .buffers
            .rasterizer
            .fill_shape(shape, mode, self.tolerance, |vert| {
                let pos = vert.position();
                brush.make_vertex(pos.into())
            })?;

        // Push the incoming buffers.
        self.push_buffers(brush.texture(self.size).as_ref().map(|t| t.texture()))
    }

    fn stroke_impl(
        &mut self,
        shape: impl Shape,
        brush: &Brush<C>,
        width: f64,
        style: &piet::StrokeStyle,
    ) -> Result<(), Pierror> {
        self.source.buffers.rasterizer.stroke_shape(
            shape,
            self.tolerance,
            width,
            style,
            |vert| {
                let pos = vert.position();
                brush.make_vertex(pos.into())
            },
            |vert| {
                let pos = vert.position();
                brush.make_vertex(pos.into())
            },
        )?;

        // Push the incoming buffers.
        self.push_buffers(brush.texture(self.size).as_ref().map(|t| t.texture()))
    }

    /// Push the values currently in the renderer to the GPU.
    fn push_buffers(&mut self, texture: Option<&Texture<C>>) -> Result<(), Pierror> {
        // Upload the vertex and index buffers.
        self.source.buffers.vbo.upload(
            &mut self.source.context,
            self.device,
            self.queue,
            self.source.buffers.rasterizer.vertices(),
            self.source.buffers.rasterizer.indices(),
        );

        // Decide which mask and transform to use.
        let (transform, mask) = {
            let state = self.state.last_mut().unwrap();

            let mask = state
                .mask
                .texture(&mut self.source.context, self.device, self.queue)?
                .unwrap_or(&self.source.white_pixel);

            (&state.transform, mask)
        };

        // Decide the texture to use.
        let texture = texture.unwrap_or(&self.source.white_pixel);

        // Draw!
        self.source
            .context
            .push_buffers(
                self.device,
                self.queue,
                self.source.buffers.vbo.resource(),
                texture.resource(),
                mask.resource(),
                transform,
                self.size,
            )
            .piet_err()?;

        // Clear the original buffers.
        self.source.buffers.rasterizer.clear();

        Ok(())
    }

    /// Get the source of this render context.
    pub fn source(&self) -> &Source<C> {
        self.source
    }

    /// Get a mutable reference to the source of this render context.
    pub fn source_mut(&mut self) -> &mut Source<C> {
        self.source
    }
}

macro_rules! leap {
    ($self:expr, $e:expr) => {{
        match $e {
            Ok(v) => v,
            Err(e) => {
                $self.status = Err(Pierror::BackendError(e.into()));
                return;
            }
        }
    }};
    ($self:expr, $e:expr, $err:expr) => {{
        match $e {
            Ok(v) => v,
            Err(e) => {
                let err = $err;
                $self.status = Err(err.into());
                return;
            }
        }
    }};
}

impl<C: GpuContext + ?Sized> piet::RenderContext for RenderContext<'_, '_, '_, C> {
    type Brush = Brush<C>;
    type Text = Text;
    type TextLayout = TextLayout;
    type Image = Image<C>;

    fn status(&mut self) -> Result<(), Pierror> {
        mem::replace(&mut self.status, Ok(()))
    }

    fn solid_brush(&mut self, color: piet::Color) -> Self::Brush {
        Brush::solid(color)
    }

    fn gradient(&mut self, gradient: impl Into<FixedGradient>) -> Result<Self::Brush, Pierror> {
        match gradient.into() {
            FixedGradient::Linear(linear) => {
                Brush::linear_gradient(&mut self.source.context, self.device, self.queue, linear)
            }
            FixedGradient::Radial(radial) => {
                Brush::radial_gradient(&mut self.source.context, self.device, self.queue, radial)
            }
        }
    }

    fn clear(&mut self, region: impl Into<Option<Rect>>, color: piet::Color) {
        let region = region.into();

        // Use optimized clear if possible.
        if region.is_none() && self.state.last().unwrap().mask.is_empty() {
            self.source.context.clear(self.device, self.queue, color);
            return;
        }

        // Otherwise, fall back to filling in the screen rectangle.
        let result = self.fill_rects(
            {
                let uv_white = Point::new(UV_WHITE[0] as f64, UV_WHITE[1] as f64);
                [TessRect {
                    pos: region.unwrap_or_else(|| {
                        Rect::from_origin_size((0.0, 0.0), (self.size.0 as f64, self.size.1 as f64))
                    }),
                    uv: Rect::from_points(uv_white, uv_white),
                    color,
                }]
            },
            None,
        );

        leap!(self, result);
    }

    fn stroke(&mut self, shape: impl Shape, brush: &impl piet::IntoBrush<Self>, width: f64) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        if let Err(e) =
            self.stroke_impl(shape, brush.as_ref(), width, &piet::StrokeStyle::default())
        {
            self.status = Err(e);
        }
    }

    fn stroke_styled(
        &mut self,
        shape: impl Shape,
        brush: &impl piet::IntoBrush<Self>,
        width: f64,
        style: &piet::StrokeStyle,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        if let Err(e) = self.stroke_impl(shape, brush.as_ref(), width, style) {
            self.status = Err(e);
        }
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl piet::IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        if let Err(e) = self.fill_impl(shape, brush.as_ref(), FillRule::NonZero) {
            self.status = Err(e);
        }
    }

    fn fill_even_odd(&mut self, shape: impl Shape, brush: &impl piet::IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        if let Err(e) = self.fill_impl(shape, brush.as_ref(), FillRule::EvenOdd) {
            self.status = Err(e);
        }
    }

    fn clip(&mut self, shape: impl Shape) {
        let state = self.state.last_mut().unwrap();
        let transform = state.transform;
        leap!(
            self,
            state.mask.clip(
                &mut self.source.context,
                self.device,
                shape,
                self.tolerance,
                transform,
                self.size
            )
        );
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.source.text
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<Point>) {
        struct RestoreAtlas<'a, 'b, 'c, 'd, G: GpuContext + ?Sized> {
            context: &'a mut RenderContext<'b, 'c, 'd, G>,
            atlas: Option<Atlas<G>>,
        }

        impl<G: GpuContext + ?Sized> Drop for RestoreAtlas<'_, '_, '_, '_, G> {
            fn drop(&mut self) {
                self.context.source.atlas = Some(self.atlas.take().unwrap());
            }
        }

        let pos = pos.into();
        let mut restore = RestoreAtlas {
            atlas: self.source.atlas.take(),
            context: self,
        };

        // Iterate over the glyphs and use them to write.
        let texture = restore.atlas.as_ref().unwrap().texture().clone();

        let text = restore.context.text().clone();
        let device = restore.context.device;
        let queue = restore.context.queue;
        let mut line_state = TextProcessingState::new();
        let rects = layout
            .buffer()
            .layout_runs()
            .flat_map(|run| {
                // Combine the run's glyphs and the layout's y position.
                run.glyphs
                    .iter()
                    .map(move |glyph| (glyph, run.line_y as f64))
            })
            .filter_map({
                let atlas = restore.atlas.as_mut().unwrap();
                |(glyph, line_y)| {
                    // Get the rectangle in texture space representing the glyph.
                    let GlyphData {
                        uv_rect,
                        offset,
                        size,
                    } = match text.with_font_system_mut(|fs| {
                        atlas.uv_rect(
                            &mut restore.context.source.context,
                            device,
                            queue,
                            glyph,
                            fs,
                        )
                    }) {
                        Ok(rect) => rect,
                        Err(e) => {
                            tracing::trace!("failed to get uv rect: {}", e);
                            return None;
                        }
                    };

                    // Get the rectangle in screen space representing the glyph.
                    let pos_rect = Rect::from_origin_size(
                        (
                            glyph.x_int as f64 + pos.x + offset.x,
                            glyph.y_int as f64 + line_y + pos.y - offset.y,
                        ),
                        size,
                    );

                    let color = match glyph.color_opt {
                        Some(color) => {
                            let [r, g, b, a] = [color.r(), color.g(), color.b(), color.a()];
                            piet::Color::rgba8(r, g, b, a)
                        }
                        None => piet::util::DEFAULT_TEXT_COLOR,
                    };

                    // Register the glyph in the atlas.
                    line_state.handle_glyph(
                        glyph,
                        line_y as f32 - (f32::from_bits(glyph.cache_key.font_size_bits) * 0.9),
                        color,
                        false,
                    );

                    Some(TessRect {
                        pos: pos_rect,
                        uv: uv_rect,
                        color,
                    })
                }
            })
            .collect::<Vec<_>>();
        let result = restore.context.fill_rects(rects, Some(&texture));

        drop(restore);

        let lines_result = {
            let lines = line_state.lines();
            if lines.is_empty() {
                Ok(())
            } else {
                self.fill_rects(
                    lines.into_iter().map(|line| {
                        let line_straddler::Line {
                            y,
                            start_x,
                            end_x,
                            style,
                            ..
                        } = line;
                        let line_width = 3.0;

                        TessRect {
                            pos: Rect::from_points(
                                Point::new(start_x as f64, y as f64) + pos.to_vec2(),
                                Point::new(end_x as f64, y as f64 + line_width) + pos.to_vec2(),
                            ),
                            uv: Rect::new(0.5, 0.5, 0.5, 0.5),
                            color: {
                                let [r, g, b, a] = [
                                    style.color.red(),
                                    style.color.green(),
                                    style.color.blue(),
                                    style.color.alpha(),
                                ];

                                piet::Color::rgba8(r, g, b, a)
                            },
                        }
                    }),
                    None,
                )
            }
        };

        leap!(self, result);
        leap!(self, lines_result);
    }

    fn save(&mut self) -> Result<(), Pierror> {
        self.state.push(RenderState {
            transform: self.state.last().unwrap().transform,
            mask: MaskSlot::new(),
        });
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Pierror> {
        if self.state.len() <= 1 {
            return Err(Pierror::StackUnbalance);
        }

        self.state.pop();
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Pierror> {
        self.source
            .context
            .flush()
            .map_err(|x| Pierror::BackendError(x.into()))
    }

    fn transform(&mut self, transform: Affine) {
        let slot = &mut self.state.last_mut().unwrap().transform;
        *slot *= transform;
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: piet::ImageFormat,
    ) -> Result<Self::Image, Pierror> {
        let tex = Texture::new(
            &mut self.source.context,
            self.device,
            InterpolationMode::Bilinear,
            RepeatStrategy::Color(piet::Color::TRANSPARENT),
        )
        .piet_err()?;

        tex.write_texture(
            &mut self.source.context,
            self.device,
            self.queue,
            (width as u32, height as u32),
            format,
            Some(buf),
        );

        Ok(Image::new(tex, Size::new(width as f64, height as f64)))
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<Rect>,
        interp: piet::InterpolationMode,
    ) {
        self.draw_image_area(image, Rect::ZERO.with_size(image.size()), dst_rect, interp)
    }

    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<Rect>,
        dst_rect: impl Into<Rect>,
        interp: piet::InterpolationMode,
    ) {
        // Create a rectangle for the destination and a rectangle for UV.
        let pos_rect = dst_rect.into();
        let uv_rect = {
            let scale_x = 1.0 / image.size().width;
            let scale_y = 1.0 / image.size().height;

            let src_rect = src_rect.into();
            Rect::new(
                src_rect.x0 * scale_x,
                src_rect.y0 * scale_y,
                src_rect.x1 * scale_x,
                src_rect.y1 * scale_y,
            )
        };

        // Set the interpolation mode.
        image
            .texture()
            .set_interpolation(&mut self.source.context, self.device, interp);

        // Use this to draw the image.
        if let Err(e) = self.fill_rects(
            [TessRect {
                pos: pos_rect,
                uv: uv_rect,
                color: piet::Color::WHITE,
            }],
            Some(image.texture()),
        ) {
            self.status = Err(e);
        }
    }

    fn capture_image_area(&mut self, src_rect: impl Into<Rect>) -> Result<Self::Image, Pierror> {
        let src_rect = src_rect.into();
        let src_size = src_rect.size();

        // Create a new texture to copy the image to.
        let image = {
            let texture = Texture::new(
                &mut self.source.context,
                self.device,
                InterpolationMode::Bilinear,
                RepeatStrategy::Repeat,
            )
            .piet_err()?;

            Image::new(texture, src_size)
        };

        // Capture the area in the texture.
        let offset = (src_rect.x0 as u32, src_rect.y0 as u32);
        let size = (src_size.width as u32, src_size.height as u32);
        self.source
            .context
            .capture_area(
                self.device,
                self.queue,
                image.texture().resource(),
                offset,
                size,
            )
            .piet_err()?;

        Ok(image)
    }

    fn blurred_rect(
        &mut self,
        _rect: Rect,
        _blur_radius: f64,
        _brush: &impl piet::IntoBrush<Self>,
    ) {
        tracing::warn!("blurred_rect is not supported");
        self.status = Err(Pierror::NotSupported);
    }

    fn current_transform(&self) -> Affine {
        self.state.last().unwrap().transform
    }
}

struct TextProcessingState {
    /// State for the underline.
    underline: LineGenerator,

    /// State for the strikethrough.
    strikethrough: LineGenerator,

    /// The lines to draw.
    lines: Vec<line_straddler::Line>,
}

impl TextProcessingState {
    fn new() -> Self {
        Self {
            underline: LineGenerator::new(LineType::Underline),
            strikethrough: LineGenerator::new(LineType::StrikeThrough),
            lines: Vec::new(),
        }
    }

    fn handle_glyph(
        &mut self,
        glyph: &LayoutGlyph,
        line_y: f32,
        color: piet::Color,
        is_bold: bool,
    ) {
        // Get the metadata.
        let metadata = Metadata::from_raw(glyph.metadata);
        let glyph = line_straddler::Glyph {
            line_y,
            font_size: f32::from_bits(glyph.cache_key.font_size_bits),
            width: glyph.w,
            x: glyph.x,
            style: line_straddler::GlyphStyle {
                bold: is_bold,
                color: match glyph.color_opt {
                    Some(color) => {
                        let [r, g, b, a] = [color.r(), color.g(), color.b(), color.a()];

                        line_straddler::Color::rgba(r, g, b, a)
                    }

                    None => {
                        let (r, g, b, a) = color.as_rgba8();
                        line_straddler::Color::rgba(r, g, b, a)
                    }
                },
            },
        };
        let Self {
            underline,
            strikethrough,
            lines,
        } = self;

        let handle_meta = |generator: &mut LineGenerator, has_it| {
            if has_it {
                generator.add_glyph(glyph)
            } else {
                generator.pop_line()
            }
        };

        let underline = handle_meta(underline, metadata.underline());
        let strikethrough = handle_meta(strikethrough, metadata.strikethrough());

        lines.extend(underline);
        lines.extend(strikethrough);
    }

    fn lines(&mut self) -> Vec<line_straddler::Line> {
        // Pop the last lines.
        let underline = self.underline.pop_line();
        let strikethrough = self.strikethrough.pop_line();
        self.lines.extend(underline);
        self.lines.extend(strikethrough);

        mem::take(&mut self.lines)
    }
}

trait ResultExt<T, E: StdError + 'static> {
    fn piet_err(self) -> Result<T, Pierror>;
}

impl<T, E: StdError + 'static> ResultExt<T, E> for Result<T, E> {
    fn piet_err(self) -> Result<T, Pierror> {
        self.map_err(|e| Pierror::BackendError(Box::new(LibraryError(e))))
    }
}

struct LibraryError<E>(E);

impl<E: fmt::Debug> fmt::Debug for LibraryError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<E: fmt::Display> fmt::Display for LibraryError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<E: StdError> StdError for LibraryError<E> {}

/// Convert a `piet::Shape` to a `tiny_skia` path.
fn shape_to_skia_path(builder: &mut tiny_skia::PathBuilder, shape: impl Shape, tolerance: f64) {
    shape.path_elements(tolerance).for_each(|el| match el {
        PathEl::MoveTo(pt) => builder.move_to(pt.x as f32, pt.y as f32),
        PathEl::LineTo(pt) => builder.line_to(pt.x as f32, pt.y as f32),
        PathEl::QuadTo(p1, p2) => {
            builder.quad_to(p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32)
        }
        PathEl::CurveTo(p1, p2, p3) => builder.cubic_to(
            p1.x as f32,
            p1.y as f32,
            p2.x as f32,
            p2.y as f32,
            p3.x as f32,
            p3.y as f32,
        ),
        PathEl::ClosePath => builder.close(),
    })
}
