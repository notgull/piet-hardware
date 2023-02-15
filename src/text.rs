//! Uses a text atlas for text rendering.

use crate::resources::Texture;
use cosmic_text::{Buffer, Font, FontSystem, Metrics};
use etagere::{Allocation, AtlasAllocator, Size};
use glow::HasContext;
use once_cell::sync::OnceCell;

use piet::kurbo::{Point, Rect, Size as KurboSize};
use piet::{
    Error, FontFamily, HitTestPoint, HitTestPosition, LineMetric, TextAlignment, TextAttribute,
};

use std::collections::HashMap;
use std::ops::RangeBounds;
use std::rc::Rc;

/// Global font data.
static FONT_DATA: OnceCell<FontSystem> = OnceCell::new();

/// The text renderer used by the [`RenderContext`].
pub struct Text<H: HasContext + ?Sized> {
    pub(super) context: Rc<H>,
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
    fn size(&self) -> KurboSize {
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

/// The text atlas for text rendering.
pub(super) struct Atlas<H: HasContext + ?Sized> {
    /// Reference to the context.
    context: Rc<H>,

    /// The list of textures in our atlas.
    textures: Vec<(AtlasAllocator, Texture<H>)>,

    /// Max size of a texture.
    max_size: Size,

    /// Map keeping track of allocations.
    allocations: HashMap<GlyphKey, GlyphInfo>,

    /// The buffer for text rendering.
    buffer: Buffer<'static>,
}

impl<H: HasContext + ?Sized> Atlas<H> {
    /// Create a new text atlas.
    pub(super) fn new(context: &Rc<H>) -> Result<Self, Error> {
        // Get the maximum texture size.
        let max_size = unsafe {
            let max_dim = context.get_parameter_i32(glow::MAX_TEXTURE_SIZE);
            Size::new(max_dim as _, max_dim as _)
        };

        Ok(Self {
            context: context.clone(),
            textures: Vec::with_capacity(1),
            max_size,
            allocations: HashMap::new(),
            buffer: Buffer::new(FONT_DATA.get_or_init(FontSystem::new), Metrics::new(0, 0)),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct GlyphKey {
    /// The character.
    character: char,
}

struct GlyphInfo {
    /// The key for this glyph.
    key: GlyphKey,

    /// The index into the `textures` vector.
    texture_index: usize,

    /// The allocation in the texture.
    allocation: Allocation,
}
