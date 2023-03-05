// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the terms of
// either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License or the Mozilla Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/> or
// <https://www.mozilla.org/en-US/MPL/2.0/>.

use piet::kurbo::{Point, Rect, Size};
use piet::Error as Pierror;

use piet_cosmic_text::{
    Text as CosText, TextLayout as CosTextLayout, TextLayoutBuilder as CosTextLayoutBuilder,
};

/// The text layout engine for the GPU renderer.
#[derive(Clone)]
pub struct Text(CosText);

impl Text {
    /// Create a new text layout engine.
    pub(crate) fn new() -> Self {
        Self(CosText::new())
    }
}

impl piet::Text for Text {
    type TextLayout = TextLayout;
    type TextLayoutBuilder = TextLayoutBuilder;

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

/// The text layout builder for the GPU renderer.
pub struct TextLayoutBuilder(CosTextLayoutBuilder);

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

/// The text layout for the GPU renderer.
#[derive(Clone)]
pub struct TextLayout(CosTextLayout);

impl TextLayout {
    pub(crate) fn buffer(&self) -> &cosmic_text::Buffer<'_> {
        self.0.buffer()
    }
}

impl piet::TextLayout for TextLayout {
    fn size(&self) -> Size {
        self.0.size()
    }

    fn trailing_whitespace_width(&self) -> f64 {
        self.0.trailing_whitespace_width()
    }

    fn image_bounds(&self) -> Rect {
        self.0.image_bounds()
    }

    fn text(&self) -> &str {
        self.0.text()
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

    fn hit_test_point(&self, point: Point) -> piet::HitTestPoint {
        self.0.hit_test_point(point)
    }

    fn hit_test_text_position(&self, idx: usize) -> piet::HitTestPosition {
        self.0.hit_test_text_position(idx)
    }
}
