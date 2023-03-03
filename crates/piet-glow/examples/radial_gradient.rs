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

//! An example with a basic usage of the library.

include!("util/setup_context.rs");

use piet::kurbo::{Point, Rect, Vec2};
use piet::{GradientStop, RenderContext as _};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gradient_image = tiny_skia::Pixmap::new(800, 600).unwrap();
    gradient_image.fill(tiny_skia::Color::from_rgba8(255, 255, 255, 255));
    gradient_image
        .fill_rect(
            tiny_skia::Rect::from_xywh(0.0, 0.0, 800.0, 600.0).unwrap(),
            &tiny_skia::Paint {
                shader: tiny_skia::RadialGradient::new(
                    tiny_skia::Point { x: 300.0, y: 400.0 },
                    tiny_skia::Point { x: 300.0, y: 400.0 },
                    150.0,
                    vec![
                        tiny_skia::GradientStop::new(
                            0.0,
                            tiny_skia::Color::from_rgba8(0, 255, 0, 255),
                        ),
                        tiny_skia::GradientStop::new(
                            0.5,
                            tiny_skia::Color::from_rgba8(128, 0, 0, 255),
                        ),
                        tiny_skia::GradientStop::new(
                            1.0,
                            tiny_skia::Color::from_rgba8(0, 0, 128, 255),
                        ),
                    ],
                    tiny_skia::SpreadMode::Pad,
                    tiny_skia::Transform::identity(),
                )
                .unwrap(),
                ..tiny_skia::Paint::default()
            },
            tiny_skia::Transform::identity(),
            None,
        )
        .unwrap();
    let img_width = gradient_image.width();
    let img_height = gradient_image.height();
    let data = gradient_image.take();

    let mut gradient_brush = None;
    util::with_renderer(move |render_context, width, height| {
        let gradient = gradient_brush.get_or_insert_with(|| {
            /*let grad = piet::FixedRadialGradient {
                center: Point::new(300.0, 400.0),
                origin_offset: Vec2::new(0.0, 0.0),
                radius: 150.0,
                stops: vec![
                    GradientStop {
                        pos: 0.0,
                        color: piet::Color::LIME,
                    },
                    GradientStop {
                        pos: 0.5,
                        color: piet::Color::MAROON,
                    },
                    GradientStop {
                        pos: 1.0,
                        color: piet::Color::NAVY,
                    },
                ],
            };

            render_context.gradient(grad).unwrap()*/

            render_context
                .make_image(
                    img_width as _,
                    img_height as _,
                    &data,
                    piet::ImageFormat::RgbaSeparate,
                )
                .unwrap()
        });

        /*
        render_context.fill(Rect::new(0.0, 0.0, width as _, height as _), gradient);
        */

        render_context.draw_image(
            gradient,
            Rect::new(0.0, 0.0, width as _, height as _),
            piet::InterpolationMode::Bilinear,
        );

        render_context.finish().unwrap();
        render_context.status().unwrap();
    })
}
