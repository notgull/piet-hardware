// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the
// terms of either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
//   version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
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

pub use piet;

use lyon_tessellation::FillRule;

use piet::kurbo::{Affine, PathEl, Point, Rect, Shape, Size};
use piet::{Error as Pierror, FixedGradient, Image as _, InterpolationMode};

use piet_cosmic_text::LineProcessor;
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
mod stroke;
mod text;

pub use self::brush::Brush;
pub use self::gpu_backend::{BufferType, GpuContext, RepeatStrategy, Vertex};
pub use self::image::Image;
pub use self::text::{Text, TextLayout, TextLayoutBuilder};

pub(crate) use atlas::{Atlas, GlyphData};
pub(crate) use mask::{Mask, MaskContext};
pub(crate) use rasterizer::{Rasterizer, TessRect};
pub(crate) use resources::{Texture, VertexBuffer};

const UV_WHITE: [f32; 2] = [0.5, 0.5];

/// Structures that are useful for implementing the `GpuContext` type.
pub mod gpu_types {
    pub use crate::gpu_backend::{AreaCapture, BufferPush, SubtextureWrite, TextureWrite};
}

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

    /// The mask rendering context.
    mask_context: MaskContext<C>,

    /// The cached list of render states.
    ///
    /// This is always empty, but it keeps the memory around.
    render_states: Option<TinyVec<[RenderState<C>; 1]>>,

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
            mask_context: MaskContext::new(),
            render_states: None,
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
            state: {
                let mut list = self.render_states.take().unwrap_or_default();
                list.clear();
                list.push(RenderState::default());
                list
            },
            source: self,
            device,
            queue,
            size: (width, height),
            status: Ok(()),
            tolerance: 0.1,
            ignore_state: false,
            bitmap_scale: 1.0,
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

    /// Indicate that we've flushed the queue and all of the GPU resources can be overwritten.
    pub fn gpu_flushed(&mut self) {
        self.mask_context.gpu_flushed();
    }
}

/// The whole point of this crate.
#[derive(Debug)]
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

    /// Scale to apply for bitmaps.
    bitmap_scale: f64,

    /// Flag to ignore the current state.
    ignore_state: bool,
}

#[derive(Debug)]
struct RenderState<C: GpuContext + ?Sized> {
    /// The current transform in pixel space.
    transform: Affine,

    /// The current clipping mask.
    mask: Option<Mask<C>>,
}

impl<C: GpuContext + ?Sized> Default for RenderState<C> {
    fn default() -> Self {
        Self {
            transform: Affine::IDENTITY,
            mask: None,
        }
    }
}

impl<C: GpuContext + ?Sized> Drop for RenderContext<'_, '_, '_, C> {
    fn drop(&mut self) {
        match &mut self.state {
            TinyVec::Heap(h) => self
                .source
                .mask_context
                .reclaim(h.drain(..).filter_map(|s| s.mask)),
            TinyVec::Inline(i) => self
                .source
                .mask_context
                .reclaim(i.drain(..).filter_map(|s| s.mask)),
        }

        let mut state = mem::take(&mut self.state);
        state.clear();
        self.source.render_states = Some(state);
    }
}

impl<'a, 'b, 'c, C: GpuContext + ?Sized> RenderContext<'a, 'b, 'c, C> {
    /// Temporarily ignore the transform and the clip.
    fn temporarily_ignore_state<'this>(
        &'this mut self,
    ) -> TemporarilyIgnoreState<'this, 'a, 'b, 'c, C> {
        self.ignore_state = true;
        TemporarilyIgnoreState(self)
    }

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
        let (transform, mask_texture, used_mask) = if self.ignore_state {
            (
                Affine::scale(self.bitmap_scale),
                &self.source.white_pixel,
                false,
            )
        } else {
            let state = self.state.last_mut().unwrap();

            let has_mask = state.mask.is_some();
            let mask = state
                .mask
                .as_mut()
                .map(|mask| {
                    self.source.mask_context.texture(
                        mask,
                        &mut self.source.context,
                        self.device,
                        self.queue,
                    )
                })
                .unwrap_or(&self.source.white_pixel);

            (
                Affine::scale(self.bitmap_scale) * state.transform,
                mask,
                has_mask,
            )
        };

        // Decide the texture to use.
        let texture = texture.unwrap_or(&self.source.white_pixel);

        // Draw!
        self.source
            .context
            .push_buffers(gpu_types::BufferPush {
                device: self.device,
                queue: self.queue,
                vertex_buffer: self.source.buffers.vbo.resource(),
                current_texture: texture.resource(),
                mask_texture: mask_texture.resource(),
                transform: &transform,
                viewport_size: self.size,
                clip: None,
            })
            .piet_err()?;

        // Clear the original buffers.
        self.source.buffers.rasterizer.clear();

        // Mark the mask as used so we don't overwrite it.
        if used_mask {
            if let Some(mask) = &mut self.state.last_mut().unwrap().mask {
                self.source.mask_context.mark_used(mask);
            }
        }

        Ok(())
    }

    fn clip_impl(&mut self, shape: impl Shape) {
        let state = self.state.last_mut().unwrap();

        let mask = state
            .mask
            .get_or_insert_with(|| Mask::new(self.size.0, self.size.1));

        self.source
            .mask_context
            .add_path(mask, shape, self.tolerance);
    }

    /// Get the source of this render context.
    pub fn source(&self) -> &Source<C> {
        self.source
    }

    /// Get a mutable reference to the source of this render context.
    pub fn source_mut(&mut self) -> &mut Source<C> {
        self.source
    }

    /// Get the current tolerance for tesselation.
    ///
    /// This is used to convert curves into line segments.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Set the current tolerance for tesselation.
    ///
    /// This is used to convert curves into line segments.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Get the bitmap scale.
    pub fn bitmap_scale(&self) -> f64 {
        self.bitmap_scale
    }

    /// Set the bitmap scale.
    pub fn set_bitmap_scale(&mut self, scale: f64) {
        self.bitmap_scale = scale;
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

    fn clear(&mut self, region: impl Into<Option<Rect>>, mut color: piet::Color) {
        let region = region.into();

        // Premultiply the color.
        let clamp = |x: f64| {
            if x < 0.0 {
                0.0
            } else if x > 1.0 {
                1.0
            } else {
                x
            }
        };
        let (r, g, b, a) = color.as_rgba();
        let r = clamp(r * a);
        let g = clamp(g * a);
        let b = clamp(b * a);
        color = piet::Color::rgba(r, g, b, 1.0);

        // Use optimized clear if possible.
        if region.is_none() {
            self.source.context.clear(self.device, self.queue, color);
            return;
        }

        // Ignore clipping mask and transform.
        let ignore_state = self.temporarily_ignore_state();

        // Otherwise, fall back to filling in the screen rectangle.
        let result = ignore_state.0.fill_rects(
            {
                let uv_white = Point::new(UV_WHITE[0] as f64, UV_WHITE[1] as f64);
                [TessRect {
                    pos: region.unwrap_or_else(|| {
                        Rect::from_origin_size(
                            (0.0, 0.0),
                            (ignore_state.0.size.0 as f64, ignore_state.0.size.1 as f64),
                        )
                    }),
                    uv: Rect::from_points(uv_white, uv_white),
                    color,
                }]
            },
            None,
        );

        leap!(ignore_state.0, result);
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
        // If we have a bitmap scale, scale the image up.
        if (self.bitmap_scale - 1.0).abs() > 0.001 {
            let mut path = shape.into_path(self.tolerance);
            path.apply_affine(Affine::scale(self.bitmap_scale));
            self.clip_impl(path);
        } else {
            self.clip_impl(shape);
        }
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
        let mut line_state = LineProcessor::new();
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
                        Some(Ok(rect)) => rect,
                        Some(Err(e)) => {
                            tracing::trace!("failed to get uv rect: {}", e);
                            return None;
                        }
                        None => {
                            // Still waiting to load.
                            tracing::trace!("font system not loaded yet");
                            return None;
                        }
                    };

                    let physical = glyph.physical((0.0, 0.0), 1.0);

                    // Get the rectangle in screen space representing the glyph.
                    let pos_rect = Rect::from_origin_size(
                        (
                            physical.x as f64 + pos.x + offset.x,
                            physical.y as f64 + line_y + pos.y - offset.y,
                        ),
                        size,
                    );

                    let color = glyph.color_opt.unwrap_or({
                        let piet_color = piet::util::DEFAULT_TEXT_COLOR;
                        let (r, g, b, a) = piet_color.as_rgba8();
                        cosmic_text::Color::rgba(r, g, b, a)
                    });
                    let piet_color =
                        glyph
                            .color_opt
                            .map_or(piet::util::DEFAULT_TEXT_COLOR, |color| {
                                let [r, g, b, a] = [color.r(), color.g(), color.b(), color.a()];
                                piet::Color::rgba8(r, g, b, a)
                            });

                    // Register the glyph in the atlas.
                    line_state.handle_glyph(glyph, line_y as f32, color);

                    Some(TessRect {
                        pos: pos_rect,
                        uv: uv_rect,
                        color: piet_color,
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
                        let mut rect = line.into_rect();
                        rect.x0 += pos.x;
                        rect.y0 += pos.y;
                        rect.x1 += pos.x;
                        rect.y1 += pos.y;
                        TessRect {
                            pos: rect,
                            uv: Rect::new(0.5, 0.5, 0.5, 0.5),
                            color: line.color,
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
        let last = self.state.last().unwrap();
        self.state.push(RenderState {
            transform: last.transform,
            mask: last.mask.clone(),
        });
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Pierror> {
        if self.state.len() <= 1 {
            return Err(Pierror::StackUnbalance);
        }

        let mut state = self.state.pop().unwrap();
        self.source
            .mask_context
            .reclaim(state.mask.take().into_iter());

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
        let src_bitmap_size = Size::new(
            src_size.width * self.bitmap_scale,
            src_size.height * self.bitmap_scale,
        );

        // Create a new texture to copy the image to.
        let image = {
            let texture = Texture::new(
                &mut self.source.context,
                self.device,
                InterpolationMode::Bilinear,
                RepeatStrategy::Repeat,
            )
            .piet_err()?;

            Image::new(texture, src_bitmap_size)
        };

        // Capture the area in the texture.
        let offset = (src_rect.x0 as u32, src_rect.y0 as u32);
        let size = (src_size.width as u32, src_size.height as u32);
        self.source
            .context
            .capture_area(gpu_backend::AreaCapture {
                device: self.device,
                queue: self.queue,
                texture: image.texture().resource(),
                offset,
                size,
                bitmap_scale: self.bitmap_scale,
            })
            .piet_err()?;

        Ok(image)
    }

    fn blurred_rect(
        &mut self,
        input_rect: Rect,
        blur_radius: f64,
        brush: &impl piet::IntoBrush<Self>,
    ) {
        let size = piet::util::size_for_blurred_rect(input_rect, blur_radius);
        let width = size.width as u32;
        let height = size.height as u32;
        if width == 0 || height == 0 {
            return;
        }

        // Compute the blurred rectangle image.
        let (mask, rect_exp) = {
            let mut mask = tiny_skia::Mask::new(width, height).unwrap();

            let rect_exp = piet::util::compute_blurred_rect(
                input_rect,
                blur_radius,
                width.try_into().unwrap(),
                mask.data_mut(),
            );

            (mask, rect_exp)
        };

        // Create an image using this mask.
        let mut image = tiny_skia::Pixmap::new(width, height)
            .expect("Pixmap width/height should be valid clipmask width/height");
        let shader = match brush.make_brush(self, || input_rect).to_shader() {
            Some(shader) => shader,
            None => {
                self.status = Err(Pierror::BackendError("Failed to create shader".into()));
                return;
            }
        };
        image.fill(tiny_skia::Color::TRANSPARENT);
        image.fill_rect(
            tiny_skia::Rect::from_xywh(0., 0., width as f32, height as f32).unwrap(),
            &tiny_skia::Paint {
                shader,
                ..Default::default()
            },
            tiny_skia::Transform::identity(),
            Some(&mask),
        );

        // Draw this image.
        let image = leap!(
            self,
            self.make_image(
                width as usize,
                height as usize,
                image.data(),
                piet::ImageFormat::RgbaSeparate
            )
        );
        self.draw_image(&image, rect_exp, piet::InterpolationMode::Bilinear);
    }

    fn current_transform(&self) -> Affine {
        self.state.last().unwrap().transform
    }
}

struct TemporarilyIgnoreState<'this, 'a, 'b, 'c, C: GpuContext + ?Sized>(
    &'this mut RenderContext<'a, 'b, 'c, C>,
);

impl<C: GpuContext + ?Sized> Drop for TemporarilyIgnoreState<'_, '_, '_, '_, C> {
    fn drop(&mut self) {
        self.0.ignore_state = false;
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
