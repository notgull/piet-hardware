// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-gpu`.
//
// `piet-gpu` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-gpu` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-gpu`. If not, see <https://www.gnu.org/licenses/> or
// <https://www.mozilla.org/en-US/MPL/2.0/>.

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

pub use piet;

use lyon_tessellation::FillRule;

use piet::kurbo::{Affine, Point, Rect, Shape, Size};
use piet::{Error as Pierror, FixedGradient, Image as _, InterpolationMode};

use tinyvec::TinyVec;

use std::error::Error as StdError;
use std::fmt;
use std::mem;
use std::rc::Rc;

mod atlas;
mod brush;
mod gpu_backend;
mod image;
mod mask;
mod rasterizer;
mod resources;
mod text;

pub use self::brush::Brush;
pub use self::gpu_backend::{BufferType, GpuContext, RepeatStrategy, Vertex, VertexFormat};
pub use self::image::Image;
pub use self::text::{Text, TextLayout, TextLayoutBuilder};

pub(crate) use atlas::Atlas;
pub(crate) use mask::MaskSlot;
pub(crate) use rasterizer::{Rasterizer, TessRect};
pub(crate) use resources::{Texture, VertexBuffer};

const UV_WHITE: [f32; 2] = [0.5, 0.5];

/// The source of the GPU renderer.
pub struct Source<C: GpuContext + ?Sized> {
    /// The context to use for the GPU renderer.
    context: Rc<C>,

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
}

impl<C: GpuContext + fmt::Debug + ?Sized> fmt::Debug for Source<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Source")
            .field("context", &self.context)
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
    /// Create a new source from a context wrapped in an `Rc`.
    pub fn from_rc(context: Rc<C>) -> Result<Self, Pierror> {
        Ok(Self {
            white_pixel: {
                const WHITE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];

                // Setup a white pixel texture.
                let texture = Texture::new(
                    &context,
                    InterpolationMode::NearestNeighbor,
                    RepeatStrategy::Repeat,
                )
                .piet_err()?;

                texture.write_texture((1, 1), piet::ImageFormat::RgbaSeparate, Some(&WHITE));

                texture
            },
            buffers: {
                let vbo = VertexBuffer::new(&context).piet_err()?;

                Buffers {
                    rasterizer: Rasterizer::new(),
                    vbo,
                }
            },
            atlas: Some(Atlas::new(&context)?),
            context,
            text: Text::new(),
        })
    }

    /// Create a new source from a context.
    pub fn new(context: C) -> Result<Self, Pierror>
    where
        C: Sized,
    {
        Self::from_rc(Rc::new(context))
    }

    /// Get a reference to the context.
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Create a new rendering context.
    pub fn render_context(&mut self, width: u32, height: u32) -> RenderContext<'_, C> {
        RenderContext {
            source: self,
            size: (width, height),
            state: TinyVec::from([RenderState::default()]),
            status: Ok(()),
            tolerance: 1.0,
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
pub struct RenderContext<'a, C: GpuContext + ?Sized> {
    /// The source of the GPU renderer.
    source: &'a mut Source<C>,

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

impl<C: GpuContext + ?Sized> RenderContext<'_, C> {
    /// Fill in a rectangle.
    fn fill_rects(
        &mut self,
        rects: impl IntoIterator<Item = TessRect>,
        texture: Option<&Texture<C>>,
    ) -> Result<(), Pierror> {
        self.source.buffers.rasterizer.fill_rects(rects);

        // Push the buffers to the GPU.
        // SAFETY: The indices are valid.
        unsafe { self.push_buffers(texture) }
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
        // SAFETY: The indices are valid.
        unsafe { self.push_buffers(brush.texture(self.size).as_ref().map(|t| t.texture())) }
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
        )?;

        // Push the incoming buffers.
        // SAFETY: Buffer indices do not exceed the size of the vertex buffer.
        unsafe { self.push_buffers(brush.texture(self.size).as_ref().map(|t| t.texture())) }
    }

    /// Push the values currently in the renderer to the GPU.
    unsafe fn push_buffers(&mut self, texture: Option<&Texture<C>>) -> Result<(), Pierror> {
        // Upload the vertex and index buffers.
        self.source.buffers.vbo.upload(
            self.source.buffers.rasterizer.vertices(),
            self.source.buffers.rasterizer.indices(),
        );

        // Decide which mask and transform to use.
        let (transform, mask) = {
            let state = self.state.last_mut().unwrap();

            let mask = state.mask.texture()?.unwrap_or(&self.source.white_pixel);

            (&state.transform, mask)
        };

        // Decide the texture to use.
        let texture = texture.unwrap_or(&self.source.white_pixel);

        // Draw!
        self.source
            .context
            .push_buffers(
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

impl<C: GpuContext + ?Sized> piet::RenderContext for RenderContext<'_, C> {
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
            FixedGradient::Linear(linear) => Brush::linear_gradient(&self.source.context, linear),
            FixedGradient::Radial(radial) => Brush::radial_gradient(&self.source.context, radial),
        }
    }

    fn clear(&mut self, region: impl Into<Option<Rect>>, color: piet::Color) {
        let region = region.into();

        // Use optimized clear if possible.
        if region.is_none() && self.state.last().unwrap().mask.is_empty() {
            self.source.context.clear(color);
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
                &self.source.context,
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
        struct RestoreAtlas<'a, 'b, G: GpuContext + ?Sized> {
            context: &'a mut RenderContext<'b, G>,
            atlas: Option<Atlas<G>>,
        }

        impl<G: GpuContext + ?Sized> Drop for RestoreAtlas<'_, '_, G> {
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
        let result = restore.context.fill_rects(
            layout
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
                        let font_data = layout
                            .buffer()
                            .font_system()
                            .get_font(glyph.cache_key.font_id)
                            .expect("font not found");
                        let (uv_rect, offset) = match atlas.uv_rect(glyph, &font_data) {
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
                                glyph.y_int as f64 + line_y + pos.y + offset.y,
                            ),
                            (glyph.w as f64, glyph.cache_key.font_size as f64),
                        );

                        let color = match glyph.color_opt {
                            Some(color) => {
                                let [r, g, b, a] = [color.r(), color.g(), color.b(), color.a()];
                                piet::Color::rgba8(r, g, b, a)
                            }
                            None => piet::Color::WHITE,
                        };

                        Some(TessRect {
                            pos: pos_rect,
                            uv: uv_rect,
                            color,
                        })
                    }
                }),
            Some(&texture),
        );

        drop(restore);
        leap!(self, result);
    }

    fn save(&mut self) -> Result<(), Pierror> {
        self.state.push(Default::default());
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
        *slot = transform * *slot;
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: piet::ImageFormat,
    ) -> Result<Self::Image, Pierror> {
        let tex = Texture::new(
            &self.source.context,
            InterpolationMode::Bilinear,
            RepeatStrategy::Color(piet::Color::TRANSPARENT),
        )
        .piet_err()?;

        tex.write_texture((width as u32, height as u32), format, Some(buf));

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
        image.texture().set_interpolation(interp);

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

    fn capture_image_area(&mut self, _src_rect: impl Into<Rect>) -> Result<Self::Image, Pierror> {
        Err(Pierror::Unimplemented)
    }

    fn blurred_rect(
        &mut self,
        _rect: Rect,
        _blur_radius: f64,
        _brush: &impl piet::IntoBrush<Self>,
    ) {
        self.status = Err(Pierror::NotSupported);
    }

    fn current_transform(&self) -> Affine {
        self.state.last().unwrap().transform
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
