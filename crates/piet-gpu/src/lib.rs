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
//! backend. As a bonus, this means that this crate contains no unsafe code.
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

#![forbid(unsafe_code)]

pub use piet_cosmic_text;

use arrayvec::ArrayVec;
use bytemuck::offset_of;
use etagere::AtlasAllocator;

use lyon_tessellation::path::{Event, PathEvent};
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, StrokeOptions,
    StrokeTessellator, StrokeVertex, VertexBuffers,
};

use piet::kurbo::{Affine, PathEl, Point, Rect, Shape, Size};
use piet::{Error as Pierror, InterpolationMode, LineCap, LineJoin};

use tiny_skia::{ClipMask, PathBuilder, Stroke};
use tinyvec::TinyVec;

use core::num;
use std::borrow::Cow;
use std::cell::RefCell;
use std::error::Error as StdError;
use std::fmt;
use std::mem;
use std::rc::Rc;

pub use piet_cosmic_text::{Text, TextLayout, TextLayoutBuilder};

const UV_WHITE: [f32; 2] = [0.5, 0.5];

/// The backend for the GPU renderer.
pub trait GpuContext {
    /// The type associated with a GPU texture.
    type Texture;

    /// The type associated with a GPU vertex buffer.
    type VertexBuffer;

    /// The type associated with a GPU index buffer.
    type IndexBuffer;

    /// The error type associated with this GPU context.
    type Error: StdError + 'static;

    /// Clear the screen with the given color.
    fn clear(&self, color: piet::Color);

    /// Flush the GPU commands.
    fn flush(&self) -> Result<(), Self::Error>;

    /// Create a new texture.
    fn create_texture(
        &self,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Result<Self::Texture, Self::Error>;

    /// Delete a texture.
    fn delete_texture(&self, texture: Self::Texture);

    /// Write an image to a texture.
    fn write_texture<T: bytemuck::Pod>(
        &self,
        texture: &Self::Texture,
        size: (u32, u32),
        format: ImageFormat,
        data: Option<&[T]>,
    );

    /// Write a sub-image to a texture.
    fn write_subtexture<T: bytemuck::Pod>(
        &self,
        texture: &Self::Texture,
        offset: (u32, u32),
        size: (u32, u32),
        format: ImageFormat,
        data: &[T],
    );

    /// Set the interpolation mode for a texture.
    fn set_texture_interpolation(&self, texture: &Self::Texture, interpolation: InterpolationMode);

    /// Get the maximum texture size.
    fn max_texture_size(&self) -> (u32, u32);

    /// Create a new vertex buffer.
    fn create_vertex_buffer(&self) -> Result<Self::VertexBuffer, Self::Error>;

    /// Delete a vertex buffer.
    fn delete_vertex_buffer(&self, buffer: Self::VertexBuffer);

    /// Write vertices to a vertex buffer.
    fn write_vertices(&self, buffer: &Self::VertexBuffer, vertices: &[Vertex]);

    /// Create a new index buffer.
    fn create_index_buffer(&self) -> Result<Self::IndexBuffer, Self::Error>;

    /// Delete an index buffer.
    fn delete_index_buffer(&self, buffer: Self::IndexBuffer);

    /// Write indices to an index buffer.
    fn write_indices(&self, buffer: &Self::IndexBuffer, indices: &[u32]);

    /// Push buffer data to the GPU.
    fn push_buffers(
        &self,
        vertex_buffer: &Self::VertexBuffer,
        index_buffer: &Self::IndexBuffer,
        num_indices: usize,
        current_texture: &Self::Texture,
        mask_texture: &Self::Texture,
        transform: &Affine,
        size: (u32, u32),
    ) -> Result<(), Self::Error>;
}

/// The image format to use for a texture.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ImageFormat {
    /// Use RGBA pixels.
    Rgba,

    /// Use RGB pixels.
    Rgb,

    /// Use grayscale pixel.
    Grayscale,
}

/// The type of interpolation to use when sampling a texture.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Interpolation {
    /// Use nearest-neighbor interpolation.
    Nearest,

    /// Use Bilinear interpolation.
    Bilinear,
}

/// The strategy to use for repeating.
#[derive(Debug, Copy, Clone, PartialEq)]
#[non_exhaustive]
pub enum RepeatStrategy {
    /// Repeat the image.
    Repeat,

    /// Don't repeat and instead use this color.
    Color(piet::Color),
}

/// The format to be provided to the vertex array.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub struct VertexFormat {
    /// The data type associated with the position.
    pub data_type: DataType,

    /// The data format associated with the position.
    pub format: DataFormat,

    /// The number of components in the position.
    pub num_components: u32,

    /// The offset of the position in the vertex.
    pub offset: u32,

    /// The stride of the vertex.
    pub stride: u32,
}

/// The data format associated with a vertex array.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum DataFormat {
    /// Uses floats.
    Float,

    /// Uses unsigned bytes.
    UnsignedByte,
}

/// The type of the data component.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum DataType {
    /// This represents the location of the component, in screen space.
    Position,

    /// This represents the location of the component, in texture space (0..1).
    Texture,

    /// This represents the color of the component.
    Color,
}

/// The vertex type used by the GPU renderer.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vertex {
    /// The position of the vertex.
    pub pos: [f32; 2],

    /// The coordinate of the vertex in the texture.
    pub uv: [f32; 2],

    /// The color of the vertex, in four SRGB channels.
    pub color: [u8; 4],
}

/// The type of the buffer to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferType {
    /// The buffer is used for vertices.
    Vertex,

    /// The buffer is used for indices.
    Index,
}

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
    //text: Text,

    /// A cached path buffer.
    path_builder: PathBuilder,
}

impl<C: GpuContext + fmt::Debug + ?Sized> fmt::Debug for Source<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Source")
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

struct Buffers<C: GpuContext + ?Sized> {
    /// The buffer for vertices and indices.
    vertex_buffers: VertexBuffers<Vertex, u32>,

    /// The fill tesselator.
    fill_tesselator: FillTessellator,

    /// The stroke tesselator.
    stroke_tesselator: StrokeTessellator,

    /// The VBO for vertices.
    vbo: VertexBuffer<C>,

    /// The EB0 for indices.
    ebo: IndexBuffer<C>,
}

impl<C: GpuContext + ?Sized> Source<C> {
    /// Create a new source from a context wrapped in an `Rc`.
    pub fn from_rc(context: Rc<C>) -> Result<Self, Pierror> {
        Ok(Self {
            white_pixel: {
                // Setup a white pixel texture.
                let texture = Texture::new(
                    &context,
                    InterpolationMode::NearestNeighbor,
                    RepeatStrategy::Repeat,
                )
                .piet_err()?;

                texture.write_texture((500, 500), ImageFormat::Rgba, Some(&[255, 255, 255, 255]));

                texture
            },
            buffers: {
                let vbo = VertexBuffer::new(&context).piet_err()?;
                let ebo = IndexBuffer::new(&context).piet_err()?;

                Buffers {
                    vertex_buffers: VertexBuffers::new(),
                    fill_tesselator: FillTessellator::new(),
                    stroke_tesselator: StrokeTessellator::new(),
                    vbo,
                    ebo,
                }
            },
            context,
            //text: Text::new(),
            path_builder: PathBuilder::new(),
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

impl<C: GpuContext + ?Sized> RenderContext<'_, C> {}

struct Mask<C: GpuContext + ?Sized> {
    /// The texture that is used as the mask.
    texture: Texture<C>,

    /// The pixmap we use as scratch space for drawing.
    pixmap: tiny_skia::Pixmap,

    /// The clipping mask we use to calculate the mask.
    mask: tiny_skia::ClipMask,

    /// Whether the mask contains data that needs to be uploaded to the texture.
    dirty: bool,
}

impl<C: GpuContext + ?Sized> Mask<C> {
    /// Upload the mask to the texture.
    fn upload(&mut self) -> Result<&Texture<C>, Pierror> {
        if self.dirty {
            // First, clear the pixmap.
            self.pixmap.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0));

            // Now, composite the mask onto the pixmap.
            let paint = tiny_skia::Paint {
                shader: tiny_skia::Shader::SolidColor(tiny_skia::Color::from_rgba8(
                    0xFF, 0xFF, 0xFF, 0xFF,
                )),
                ..Default::default()
            };
            let rect = tiny_skia::Rect::from_xywh(
                0.0,
                0.0,
                self.pixmap.width() as f32,
                self.pixmap.height() as f32,
            )
            .unwrap();
            self.pixmap.fill_rect(
                rect,
                &paint,
                tiny_skia::Transform::identity(),
                Some(&self.mask),
            );

            // Finally, upload the pixmap to the texture.
            let data = self.pixmap.data();
            self.texture.write_texture(
                (self.pixmap.width(), self.pixmap.height()),
                ImageFormat::Rgba,
                Some(data),
            );

            self.dirty = false;
        }

        Ok(&self.texture)
    }
}

impl<C: GpuContext + ?Sized> RenderContext<'_, C> {
    /// Fill in a rectangle.
    fn fill_rects(
        &mut self,
        rects_and_uv_rects: &[(Rect, Rect)],
        color: piet::Color,
        texture: Option<&Texture<C>>,
    ) -> Result<(), Pierror> {
        let (r, g, b, a) = color.as_rgba8();
        let color = [r, g, b, a];

        // Get the vertices associated with the rectangles.
        let vertices = |pos_rect: Rect, uv_rect: Rect| {
            let cast = |x: f64| x as f32;

            [
                Vertex {
                    pos: [cast(pos_rect.x0), cast(pos_rect.y0)],
                    uv: [cast(uv_rect.x0), cast(uv_rect.y0)],
                    color,
                },
                Vertex {
                    pos: [cast(pos_rect.x1), cast(pos_rect.y0)],
                    uv: [cast(uv_rect.x1), cast(uv_rect.y0)],
                    color,
                },
                Vertex {
                    pos: [cast(pos_rect.x1), cast(pos_rect.y1)],
                    uv: [cast(uv_rect.x1), cast(uv_rect.y1)],
                    color,
                },
                Vertex {
                    pos: [cast(pos_rect.x0), cast(pos_rect.y1)],
                    uv: [cast(uv_rect.x0), cast(uv_rect.y1)],
                    color,
                },
            ]
        };

        self.source.buffers.vertex_buffers.vertices.extend(
            rects_and_uv_rects
                .iter()
                .copied()
                .flat_map(|(pos_rect, uv_rect)| vertices(pos_rect, uv_rect)),
        );
        self.source.buffers.vertex_buffers.indices.extend(
            (0..rects_and_uv_rects.len() as u32).flat_map(|i| {
                let base = i * 6;
                [base, base + 1, base + 2, base, base + 2, base + 3]
            }),
        );

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
        // Create a new buffers builder.
        let mut builder = BuffersBuilder::new(
            &mut self.source.buffers.vertex_buffers,
            |vertex: FillVertex<'_>| {
                let pos = vertex.position();
                brush.0.make_vertex([pos.x, pos.y])
            },
        );

        // Create fill options.
        let mut options = FillOptions::default();
        options.fill_rule = mode;
        options.tolerance = self.tolerance as f32;

        // Fill the shape.
        self.source
            .buffers
            .fill_tesselator
            .tessellate(
                shape_to_lyon_path(&shape, self.tolerance),
                &options,
                &mut builder,
            )
            .piet_err()?;

        // Push the incoming buffers.
        self.push_buffers(brush.0.texture())
    }

    fn stroke_impl(
        &mut self,
        shape: impl Shape,
        brush: &Brush<C>,
        width: f64,
        style: &piet::StrokeStyle,
    ) -> Result<(), Pierror> {
        // TODO: Support dashing.
        if !style.dash_pattern.is_empty() {
            return Err(Pierror::NotSupported);
        }

        // Create a new buffers builder.
        let mut builder = BuffersBuilder::new(
            &mut self.source.buffers.vertex_buffers,
            |vertex: StrokeVertex<'_, '_>| {
                let pos = vertex.position();
                brush.0.make_vertex([pos.x, pos.y])
            },
        );

        let cvt_line_cap = |cap: LineCap| match cap {
            LineCap::Butt => lyon_tessellation::LineCap::Butt,
            LineCap::Round => lyon_tessellation::LineCap::Round,
            LineCap::Square => lyon_tessellation::LineCap::Square,
        };

        // Create stroke options.
        let mut options = StrokeOptions::default();
        options.tolerance = self.tolerance as f32;
        options.line_width = width as f32;
        options.start_cap = cvt_line_cap(style.line_cap);
        options.end_cap = cvt_line_cap(style.line_cap);
        options.line_join = match style.line_join {
            LineJoin::Bevel => lyon_tessellation::LineJoin::Bevel,
            LineJoin::Round => lyon_tessellation::LineJoin::Round,
            LineJoin::Miter { limit } => {
                options.miter_limit = limit as f32;
                lyon_tessellation::LineJoin::Miter
            }
        };

        // Fill the shape.
        self.source
            .buffers
            .stroke_tesselator
            .tessellate(
                shape_to_lyon_path(&shape, self.tolerance),
                &options,
                &mut builder,
            )
            .piet_err()?;

        // Push the incoming buffers.
        self.push_buffers(brush.0.texture())
    }

    /// Push the values currently in the renderer to the GPU.
    fn push_buffers(&mut self, texture: Option<&Texture<C>>) -> Result<(), Pierror> {
        // Upload the vertex and index buffers.
        self.source
            .buffers
            .vbo
            .upload(&self.source.buffers.vertex_buffers.vertices);

        self.source
            .buffers
            .ebo
            .upload(&self.source.buffers.vertex_buffers.indices);

        // Decide which mask and transform to use.
        let (transform, mask) = {
            let state = self.state.last_mut().unwrap();

            let mask = match state.mask.as_mut() {
                Some(mask) => mask.upload()?,
                None => &self.source.white_pixel,
            };

            (&state.transform, mask)
        };

        // Decide the texture to use.
        let texture = texture.unwrap_or(&self.source.white_pixel);

        let num_indices = self.source.buffers.vertex_buffers.indices.len();

        // Draw!
        self.source
            .context
            .push_buffers(
                self.source.buffers.vbo.resource(),
                self.source.buffers.ebo.resource(),
                num_indices,
                texture.resource(),
                mask.resource(),
                transform,
                self.size,
            )
            .piet_err()?;

        // Clear the original buffers.
        self.source.buffers.vertex_buffers.vertices.clear();
        self.source.buffers.vertex_buffers.indices.clear();

        Ok(())
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
        Brush(BrushInner::Solid(color))
    }

    fn gradient(
        &mut self,
        _gradient: impl Into<piet::FixedGradient>,
    ) -> Result<Self::Brush, Pierror> {
        Err(Pierror::Unimplemented)
    }

    fn clear(&mut self, region: impl Into<Option<Rect>>, color: piet::Color) {
        let region = region.into();

        // Use optimized clear if possible.
        if region.is_none() && self.state.last().unwrap().mask.is_none() {
            self.source.context.clear(color);
            return;
        }

        // Otherwise, fall back to filling in the screen rectangle.
        if let Err(e) = self.fill_rects(
            {
                let uv_white = Point::new(UV_WHITE[0] as f64, UV_WHITE[1] as f64);
                &[(
                    region.unwrap_or_else(|| {
                        Rect::from_origin_size((0.0, 0.0), (self.size.0 as f64, self.size.1 as f64))
                    }),
                    Rect::from_points(uv_white, uv_white),
                )]
            },
            color,
            None,
        ) {
            self.status = Err(e);
        }
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

        // TODO: There has to be a better way of doing this.
        let path = {
            let path = shape.into_path(self.tolerance);
            let transformed = transform * path;

            let mut builder = mem::take(&mut self.source.path_builder);
            shape_to_skia_path(&mut builder, transformed, self.tolerance);
            builder.finish().expect("path builder failed")
        };

        match &mut state.mask {
            Some(mask) => {
                // If there is already a mask, we need to intersect it with the new shape.
                mask.mask
                    .intersect_path(&path, tiny_skia::FillRule::EvenOdd, false);
            }

            slot @ None => {
                // If there is no mask, we need to create one.
                let mut mask = Mask {
                    texture: leap!(
                        self,
                        Texture::new(
                            &self.source.context,
                            InterpolationMode::Bilinear,
                            RepeatStrategy::Color(piet::Color::TRANSPARENT),
                        )
                    ),
                    pixmap: tiny_skia::Pixmap::new(self.size.0, self.size.1).unwrap(),
                    mask: ClipMask::new(),
                    dirty: true,
                };

                mask.mask
                    .set_path(
                        self.size.0,
                        self.size.1,
                        &path,
                        tiny_skia::FillRule::EvenOdd,
                        false,
                    )
                    .expect("failed to set path");

                *slot = Some(mask);
            }
        }

        self.source.path_builder = path.clear();
    }

    fn text(&mut self) -> &mut Self::Text {
        //&mut self.source.text
        todo!()
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<Point>) {
        todo!()
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

        tex.write_texture((width as u32, height as u32), todo!(), Some(buf));

        Ok(Image {
            texture: Rc::new(tex),
            size: Size::new(width as f64, height as f64),
        })
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<Rect>,
        interp: piet::InterpolationMode,
    ) {
        todo!()
    }

    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<Rect>,
        dst_rect: impl Into<Rect>,
        interp: piet::InterpolationMode,
    ) {
        todo!()
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

/// The brush type used by the GPU renderer.
pub struct Brush<C: GpuContext + ?Sized>(BrushInner<C>);

impl<C: GpuContext + ?Sized> Clone for Brush<C> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

enum BrushInner<C: GpuContext + ?Sized> {
    /// A solid color.
    Solid(piet::Color),

    /// A texture to apply.
    Texture {
        /// The image to apply.
        image: Image<C>,

        /// The source rectangle of the image to apply.
        source: Rect,
    },
}

impl<C: GpuContext + ?Sized> piet::IntoBrush<RenderContext<'_, C>> for Brush<C> {
    fn make_brush<'a>(
        &'a self,
        _piet: &mut RenderContext<'_, C>,
        _bbox: impl FnOnce() -> Rect,
    ) -> Cow<'a, <RenderContext<'_, C> as piet::RenderContext>::Brush> {
        Cow::Borrowed(self)
    }
}

impl<C: GpuContext + ?Sized> BrushInner<C> {
    /// Get the texture associated with this brush.
    fn texture(&self) -> Option<&Texture<C>> {
        match self {
            Self::Solid(_) => None,
            Self::Texture { image, .. } => Some(&image.texture),
        }
    }

    /// Transform a two-dimensional point into a vertex using this brush.
    fn make_vertex(&self, point: [f32; 2]) -> Vertex {
        match self {
            Self::Solid(color) => Vertex {
                pos: point,
                uv: UV_WHITE,
                color: {
                    let (r, g, b, a) = color.as_rgba8();
                    [r, g, b, a]
                },
            },

            Self::Texture { image, source } => {
                // Create a transform to convert from image coordinates to
                // UV coordinates.
                Vertex {
                    pos: point,
                    uv: todo!(),
                    color: [0xFF, 0xFF, 0xFF, 0xFF],
                }
            }
        }
    }
}

impl<C: GpuContext + ?Sized> Clone for BrushInner<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Solid(color) => Self::Solid(*color),
            Self::Texture { image, source } => Self::Texture {
                image: image.clone(),
                source: *source,
            },
        }
    }
}

/// The image type used by the GPU renderer.
pub struct Image<C: GpuContext + ?Sized> {
    /// The texture.
    texture: Rc<Texture<C>>,

    /// The size of the image.
    size: Size,
}

impl<C: GpuContext + ?Sized> Clone for Image<C> {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            size: self.size,
        }
    }
}

impl<C: GpuContext + ?Sized> piet::Image for Image<C> {
    fn size(&self) -> Size {
        self.size
    }
}

macro_rules! define_resource_wrappers {
    ($($name:ident($res:ident => $delete:ident)),* $(,)?) => {
        $(
            struct $name<C: GpuContext + ?Sized> {
                context: Rc<C>,
                resource: Option<C::$res>,
            }

            impl<C: GpuContext + ?Sized> Drop for $name<C> {
                fn drop(&mut self) {
                    if let Some(resource) = self.resource.take() {
                        self.context.$delete(resource);
                    }
                }
            }

            impl<C: GpuContext + ?Sized> $name<C> {
                fn from_raw(
                    context: &Rc<C>,
                    resource: C::$res,
                ) -> Self {
                    Self {
                        context: context.clone(),
                        resource: Some(resource),
                    }
                }

                fn resource(&self) -> &C::$res {
                    self.resource.as_ref().unwrap()
                }
            }
        )*
    };
}

define_resource_wrappers! {
    Texture(Texture => delete_texture),
    VertexBuffer(VertexBuffer => delete_vertex_buffer),
    IndexBuffer(IndexBuffer => delete_index_buffer),
}

impl<C: GpuContext + ?Sized> Texture<C> {
    fn new(
        context: &Rc<C>,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Result<Self, C::Error> {
        let resource = context.create_texture(interpolation, repeat)?;

        Ok(Self::from_raw(context, resource))
    }

    fn write_texture<T: bytemuck::Pod>(
        &self,
        size: (u32, u32),
        format: ImageFormat,
        data: Option<&[T]>,
    ) {
        self.context
            .write_texture(self.resource(), size, format, data);
    }

    fn write_subtexture<T: bytemuck::Pod>(
        &self,
        offset: (u32, u32),
        size: (u32, u32),
        format: ImageFormat,
        data: &[T],
    ) {
        self.context
            .write_subtexture(self.resource(), offset, size, format, data);
    }
}

impl<C: GpuContext + ?Sized> VertexBuffer<C> {
    fn new(context: &Rc<C>) -> Result<Self, C::Error> {
        let resource = context.create_vertex_buffer()?;
        Ok(Self::from_raw(context, resource))
    }

    fn upload(&self, data: &[Vertex]) {
        self.context.write_vertices(self.resource(), data)
    }
}

impl<C: GpuContext + ?Sized> IndexBuffer<C> {
    fn new(context: &Rc<C>) -> Result<Self, C::Error> {
        let resource = context.create_index_buffer()?;
        Ok(Self::from_raw(context, resource))
    }

    fn upload(&self, data: &[u32]) {
        self.context.write_indices(self.resource(), data)
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

fn shape_to_skia_path(builder: &mut PathBuilder, shape: impl Shape, tolerance: f64) {
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

fn shape_to_lyon_path(shape: &impl Shape, tolerance: f64) -> impl Iterator<Item = PathEvent> + '_ {
    use std::iter::Fuse;

    fn convert_point(pt: Point) -> lyon_tessellation::path::geom::Point<f32> {
        let (x, y): (f64, f64) = pt.into();
        [x as f32, y as f32].into()
    }

    struct PathConverter<I> {
        /// The iterator over `kurbo` `PathEl`s.
        iter: Fuse<I>,

        /// The last point that we processed.
        last: Option<Point>,

        /// The first point of the current subpath.
        first: Option<Point>,

        // Whether or not we need to close the path.
        needs_close: bool,
    }

    impl<I: Iterator<Item = PathEl>> Iterator for PathConverter<I> {
        type Item = ArrayVec<PathEvent, 2>;

        fn next(&mut self) -> Option<Self::Item> {
            let close = |this: &mut PathConverter<I>, close| {
                if let (Some(first), Some(last)) = (this.first.take(), this.last.take()) {
                    if (!approx_eq(first.x, last.x) || !approx_eq(first.y, last.y))
                        && (this.needs_close || close)
                    {
                        this.needs_close = false;
                        return Some(Event::End {
                            last: convert_point(last),
                            first: convert_point(first),
                            close,
                        });
                    }
                }

                None
            };

            let el = match self.iter.next() {
                Some(el) => el,
                None => {
                    // If we're at the end of the iterator, we need to close the path.
                    return close(self, false).map(one);
                }
            };

            match el {
                PathEl::MoveTo(pt) => {
                    // Close if we need to.
                    let close = close(self, false);

                    // Set the first point.
                    self.first = Some(pt);
                    self.last = Some(pt);

                    let mut v = ArrayVec::new();
                    v.extend(close);
                    v.push(Event::Begin {
                        at: convert_point(pt),
                    });
                    Some(v)
                }

                PathEl::LineTo(pt) => {
                    self.needs_close = true;
                    let from = self.last.replace(pt).expect("last point should be set");

                    Some(one(Event::Line {
                        from: convert_point(from),
                        to: convert_point(pt),
                    }))
                }

                PathEl::QuadTo(ctrl1, pt) => {
                    self.needs_close = true;
                    let from = self.last.replace(pt).expect("last point should be set");

                    Some(one(Event::Quadratic {
                        from: convert_point(from),
                        ctrl: convert_point(ctrl1),
                        to: convert_point(pt),
                    }))
                }

                PathEl::CurveTo(ctrl1, ctrl2, pt) => {
                    self.needs_close = true;
                    let from = self.last.replace(pt).expect("last point should be set");

                    Some(one(Event::Cubic {
                        from: convert_point(from),
                        ctrl1: convert_point(ctrl1),
                        ctrl2: convert_point(ctrl2),
                        to: convert_point(pt),
                    }))
                }

                PathEl::ClosePath => {
                    let mut v = ArrayVec::new();
                    v.extend(close(self, true));
                    Some(v)
                }
            }
        }
    }

    PathConverter {
        iter: shape.path_elements(tolerance).fuse(),
        last: None,
        first: None,
        needs_close: false,
    }
    .flatten()
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.01
}

fn one(p: PathEvent) -> ArrayVec<PathEvent, 2> {
    let mut v = ArrayVec::new();
    v.push(p);
    v
}
