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

//! The `piet-glow` basics.rs example, but with piet-wgpu.

#[path = "util/setup_context.rs"]
mod util;

use piet_hardware::piet::kurbo::{Affine, Vec2};
use piet_hardware::piet::{
    Color, FontFamily, FontStyle, FontWeight, RenderContext as _, Text as _, TextAttribute,
    TextLayoutBuilder as _,
};
use piet_wgpu::RenderContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let draw = move |rc: &mut RenderContext<'_, '_, '_>, _width, _height| {
        static TEXT: &str = r#"Philosophers often behave like little children who scribble some marks on a piece of paper at random and then ask the grown-up "What's that?" — It happened like this: the grown-up had drawn pictures for the child several times and said "this is a man," "this is a house," etc. And then the child makes some marks too and asks: what's this then?"#;

        const RED: Color = Color::rgb8(255, 0, 0);
        const BLUE: Color = Color::rgb8(0, 0, 255);

        rc.clear(None, Color::WHITE);
        rc.transform(Affine::scale(2.0));
        let text = rc.text();
        let courier = text
            .font_family("Courier New")
            .unwrap_or(FontFamily::MONOSPACE);
        let layout = text
            .new_text_layout(TEXT)
            .max_width(200.0)
            .default_attribute(courier)
            .default_attribute(TextAttribute::Underline(true))
            .default_attribute(FontStyle::Italic)
            .default_attribute(TextAttribute::TextColor(RED))
            .default_attribute(FontWeight::BOLD)
            .range_attribute(..200, TextAttribute::TextColor(BLUE))
            .range_attribute(10..100, FontWeight::NORMAL)
            .range_attribute(20..50, TextAttribute::Strikethrough(true))
            .range_attribute(40..300, TextAttribute::Underline(false))
            .range_attribute(60..160, FontStyle::Regular)
            .range_attribute(140..220, FontWeight::NORMAL)
            .range_attribute(240.., FontFamily::SYSTEM_UI)
            .build()
            .unwrap();

        let text_pos = Vec2::new(0.0, 0.0);
        rc.draw_text(&layout, text_pos.to_point());
    };

    util::run(draw)
}
