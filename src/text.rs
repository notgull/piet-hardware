//! Uses a text atlas for text rendering.

use crate::resources::Texture;
use cosmic_text::fontdb::{Database, Family};
use cosmic_text::{
    Buffer, BufferLine, CacheKey as GlyphKey, Font, FontSystem, LayoutGlyph, Metrics,
};
use etagere::{Allocation, AtlasAllocator, Size};
use glow::HasContext;
use once_cell::sync::OnceCell;

use piet::kurbo::{Point, Rect, Size as KurboSize};
use piet::{
    Error, FontFamily, HitTestPoint, HitTestPosition, LineMetric, TextAlignment, TextAttribute,
};

use std::cell::{Cell, RefCell};
use std::collections::hash_map::{Entry, HashMap};
use std::fmt;
use std::mem;
use std::ops::{Bound, Range, RangeBounds};
use std::rc::Rc;

/// The text renderer used by the [`RenderContext`].
pub struct Text {
    /// The database of fonts.
    font_database: Rc<RefCell<FontDatabase>>,

    /// A re-usable buffer for text rendering.
    buffer: Cell<Vec<BufferLine>>,
}

impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Text { .. }")
    }
}

impl Clone for Text {
    fn clone(&self) -> Self {
        Self {
            font_database: self.font_database.clone(),
            // Don't clone the buffer, just make a new one.
            buffer: Cell::new(Vec::new()),
        }
    }
}

impl Text {
    pub(super) fn new() -> Self {
        Self {
            font_database: Rc::new(RefCell::new(FontDatabase::Cosmic(FontSystem::new()))),
            buffer: Cell::new(Vec::new()),
        }
    }

    fn with_buffer<R>(&self, metrics: Metrics, f: impl FnOnce(&mut Buffer<'_>) -> R) -> R {
        let mut line_buffer = self.buffer.take();
        let mut font_database = self.font_database.borrow_mut();

        let mut buffer = Buffer::new(font_database.font_system(), metrics);
        buffer.lines = line_buffer;

        let result = f(&mut buffer);

        // Restore the buffer state.
        buffer.lines.clear();
        self.buffer.set(buffer.lines);

        result
    }
}

impl piet::Text for Text {
    type TextLayout = TextLayout;
    type TextLayoutBuilder = TextLayoutBuilder;

    fn font_family(&mut self, family_name: &str) -> Option<FontFamily> {
        self.font_database.borrow().font_family_by_name(family_name)
    }

    fn load_font(&mut self, data: &[u8]) -> Result<FontFamily, Error> {
        self.font_database.borrow_mut().load_font(data)
    }

    fn new_text_layout(&mut self, text: impl piet::TextStorage) -> Self::TextLayoutBuilder {
        let text = { text.as_str().to_string() };

        TextLayoutBuilder {
            handle: self.clone(),
            string: text.into_boxed_str().into(),
            default_attributes: vec![],
            range_attributes: HashMap::new(),
            alignment: TextAlignment::Start,
            max_width: f64::INFINITY,
        }
    }
}

/// The text layout builder used by the [`RenderContext`].
#[derive(Debug, Clone)]
pub struct TextLayoutBuilder {
    /// Handle to the original `Text` object.
    handle: Text,

    /// The string we're laying out.
    string: Rc<str>,

    /// The default text attributes.
    default_attributes: Vec<TextAttribute>,

    /// The range attributes.
    range_attributes: HashMap<Range<usize>, Vec<TextAttribute>>,

    /// The alignment.
    alignment: TextAlignment,

    /// The allowed buffer size.
    max_width: f64,
}

impl piet::TextLayoutBuilder for TextLayoutBuilder {
    type Out = TextLayout;

    fn alignment(mut self, alignment: TextAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    fn max_width(mut self, width: f64) -> Self {
        self.max_width = width;
        self
    }

    fn default_attribute(mut self, attribute: impl Into<TextAttribute>) -> Self {
        self.default_attributes.push(attribute.into());
        self
    }

    fn range_attribute(
        mut self,
        range: impl RangeBounds<usize>,
        attribute: impl Into<TextAttribute>,
    ) -> Self {
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.string.len(),
        };

        let range = start..end;

        let attributes = match self.range_attributes.entry(range) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(Vec::new()),
        };

        attributes.push(attribute.into());

        self
    }

    fn build(self) -> Result<Self::Out, Error> {
        todo!()
    }
}

/// The text layout used by the [`RenderContext`].
#[derive(Clone)]
pub struct TextLayout {
    /// The original string.
    string: Rc<str>,

    /// The lines combined with the line's Y coordinate.
    lines: Rc<[LayoutLine]>,
}

struct LayoutLine {
    /// The range of text of `string` that this line represents.
    range: Range<usize>,

    /// The glyphs in this line.
    glyphs: Vec<LayoutGlyph>,

    /// The width of the line.
    line_w: i32,

    /// The Y coordinate of the line.
    line_y: i32,
}

impl piet::TextLayout for TextLayout {
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
        &self.string
    }

    fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn line_metric(&self, line_number: usize) -> Option<LineMetric> {
        todo!()
    }

    fn line_text(&self, line_number: usize) -> Option<&str> {
        let range = self.lines.get(line_number)?.range.clone();
        Some(&self.string[range])
    }

    fn hit_test_point(&self, point: Point) -> HitTestPoint {
        todo!()
    }

    fn hit_test_text_position(&self, idx: usize) -> HitTestPosition {
        todo!()
    }
}

/// The text atlas for text rendering.
struct Atlas<H: HasContext + ?Sized> {
    /// Reference to the context.
    context: Rc<H>,

    /// The list of textures in our atlas.
    textures: RefCell<Vec<(AtlasAllocator, Texture<H>)>>,

    /// Max size of a texture.
    max_size: Size,

    /// Map keeping track of allocations.
    allocations: RefCell<HashMap<GlyphKey, GlyphInfo>>,

    /// Vector of buffer lines reused between `Buffer`s.
    buffer_lines: Cell<Vec<BufferLine>>,

    /// The font database.
    font_database: RefCell<FontDatabase>,
}

impl<H: HasContext + ?Sized> Atlas<H> {
    /// Create a new text atlas.
    fn new(context: &Rc<H>) -> Result<Self, Error> {
        // Get the maximum texture size.
        let max_size = unsafe {
            let max_dim = context.get_parameter_i32(glow::MAX_TEXTURE_SIZE);
            Size::new(max_dim as _, max_dim as _)
        };

        Ok(Self {
            context: context.clone(),
            textures: RefCell::new(Vec::with_capacity(1)),
            max_size,
            allocations: RefCell::new(HashMap::new()),
            buffer_lines: Cell::new(Vec::new()),
            font_database: RefCell::new(FontDatabase::Cosmic(FontSystem::new())),
        })
    }

    /// Run with a buffer with the given metrics.
    fn with_buffer<'a, R>(
        &'a self,
        metrics: Metrics,
        f: impl FnOnce(&'a Self, &mut Buffer<'_>) -> R,
    ) -> R {
        // Take the font database and the buffer lines.
        let mut db = self.font_database.borrow_mut();
        let buffer_lines = self.buffer_lines.take();

        // Create a new buffer.
        let mut buffer = Buffer::new(db.font_system(), metrics);

        // Use our buffer lines cache.
        buffer.lines = buffer_lines;

        // Run the function.
        let result = f(self, &mut buffer);

        // Put the buffer lines back.
        self.buffer_lines.set(buffer.lines);

        // Return the result.
        result
    }
}

struct GlyphInfo {
    /// The key for this glyph.
    key: GlyphKey,

    /// The index into the `textures` vector.
    texture_index: usize,

    /// The allocation in the texture.
    allocation: Allocation,
}

enum FontDatabase {
    /// The raw `fontdb` font database.
    ///
    /// This is used for adding new fonts to the system.
    FontDb { locale: String, db: Database },

    /// The `cosmic-text` `FontSystem` structure.
    ///
    /// This is used to render text. It is not cheap to construct, so it should only be
    /// constructed/destructed when we have to add new fonts.
    Cosmic(FontSystem),

    /// Empty hole.
    Empty,
}

impl FontDatabase {
    /// Get a font family by its name.
    fn font_family_by_name(&self, name: &str) -> Option<FontFamily> {
        let db = self.database();

        // Get the font family.
        let family = Family::Name(name);
        let name = db.family_name(&family);

        // Look for the font with that name.
        db.faces()
            .iter()
            .find(|face| face.family == name)
            .map(|face| FontFamily::new_unchecked(face.family.clone()))
    }

    /// Load a font by its raw bytes.
    fn load_font(&mut self, bytes: &[u8]) -> Result<FontFamily, Error> {
        let font_name = font_name(bytes)?;

        // Fast path: try to load the font by its name.
        if let Some(family) = self.font_family_by_name(&font_name) {
            return Ok(family);
        }

        // Slow path: insert the font into the database and then try to load it by its name.
        {
            let db = self.database_mut();
            db.load_font_data(bytes.into());
        }

        self.font_family_by_name(&font_name)
            .ok_or_else(|| Error::FontLoadingFailed)
    }

    /// Get the font system.
    fn font_system(&mut self) -> &mut FontSystem {
        loop {
            match self {
                FontDatabase::FontDb { .. } => {
                    // Replace this database with the corresponding `FontSystem`.
                    let (locale, db) = match mem::replace(self, Self::Empty) {
                        FontDatabase::FontDb { locale, db } => (locale, db),
                        _ => unreachable!(),
                    };

                    // Construct the font system.
                    let font_system = FontSystem::new_with_locale_and_db(locale, db);
                    *self = FontDatabase::Cosmic(font_system);
                }
                FontDatabase::Cosmic(font_system) => return font_system,
                _ => unreachable!("cannot poll an empty hole"),
            }
        }
    }

    /// Get the underlying database.
    ///
    /// This does not mutate the structure.
    fn database(&self) -> &Database {
        match self {
            FontDatabase::FontDb { db, .. } => db,
            FontDatabase::Cosmic(cm) => cm.db(),
            _ => unreachable!("cannot poll an empty hole"),
        }
    }

    /// Get a mutable reference to the database.
    fn database_mut(&mut self) -> &mut Database {
        loop {
            match self {
                FontDatabase::FontDb { db, .. } => return db,
                FontDatabase::Cosmic(_) => {
                    // Replace this database with the corresponding `FontSystem`.
                    let font_system = match mem::replace(self, Self::Empty) {
                        FontDatabase::Cosmic(font_system) => font_system,
                        _ => unreachable!(),
                    };

                    // Construct the font system.
                    let (locale, db) = font_system.into_locale_and_db();
                    *self = FontDatabase::FontDb { locale, db };
                }
                _ => unreachable!("cannot poll an empty hole"),
            }
        }
    }
}

fn font_name(font: &[u8]) -> Result<String, Error> {
    // Parse it using ttf-parser
    let font = ttf_parser::Face::parse(font, 0).map_err(|e| Error::BackendError(e.into()))?;

    // Get the name with the main ID.
    let name = font
        .names()
        .into_iter()
        .find(|n| n.name_id == ttf_parser::name_id::FAMILY)
        .ok_or_else(|| Error::BackendError("font does not have a name with the main ID".into()))?;

    // TODO: Support macintosh encoding.
    name.to_string()
        .ok_or_else(|| Error::BackendError("font name is not valid UTF-16".into()))
}
