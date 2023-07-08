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

use instant::{Duration, Instant};

use piet_hardware::piet::kurbo::{Affine, BezPath, Point, Rect, Vec2};
use piet_hardware::piet::{self, GradientStop, RenderContext as _};
use piet_wgpu::RenderContext;

const ORANGES: &[u8] = include_bytes!("../../piet-glow/examples/assets/oranges.jpg");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A path representing a star.
    let star = generate_five_pointed_star(Point::new(0.0, 0.0), 75.0, 150.0);
    let mut tick = 0;

    // Get the test image.
    let image = image::load_from_memory(ORANGES)?.to_rgba8();

    // Convert the image to a byte buffer.
    let image_size = image.dimensions();
    let image_data = image.into_raw();

    // Drawing function.
    let mut solid_olive = None;
    let mut radial_gradient = None;
    let mut outline = None;
    let mut image = None;

    let mut last_second = Instant::now();
    let mut num_frames = 0;
    let mut current_fps = None;

    let draw = move |rc: &mut RenderContext<'_, '_, '_>, width, height| {
        rc.clear(None, piet::Color::rgb8(0x87, 0xCE, 0xEB));

        let red_star = {
            let rot = (tick % 360) as f64 / 180.0 * std::f64::consts::PI;
            let transform = Affine::translate((200.0, 200.0)) * Affine::rotate(rot);
            transform * (&star)
        };

        // Draw a solid red using the path.
        let solid_olive = solid_olive.get_or_insert_with(|| rc.solid_brush(piet::Color::OLIVE));
        rc.fill(&red_star, solid_olive);

        // Draw a black outline using the path.
        let outline = outline.get_or_insert_with(|| rc.solid_brush(piet::Color::BLACK));
        rc.stroke(&red_star, outline, 5.0);

        // Test the transform.
        rc.with_save(|rc| {
            let rot = ((tick * 2) % 360) as f64 / 180.0 * std::f64::consts::PI;
            let trans = Affine::translate((600.0, 200.0))
                * Affine::rotate(rot)
                * Affine::scale_non_uniform(0.75, 0.75);
            let gradient = radial_gradient.get_or_insert_with(|| {
                let grad = piet::FixedRadialGradient {
                    center: Point::new(0.0, 0.0),
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

                rc.gradient(grad).unwrap()
            });

            rc.transform(trans);
            rc.fill(&star, gradient);
            rc.stroke(&star, outline, 5.0);

            Ok(())
        })
        .unwrap();

        // Create an image and draw it.
        let image = image.get_or_insert_with(|| {
            rc.make_image(
                image_size.0 as _,
                image_size.1 as _,
                &image_data,
                piet::ImageFormat::RgbaSeparate,
            )
            .unwrap()
        });

        let scale = |x: f64| (x + 1.0) * 50.0;
        let posn_shift_x = scale(((tick as f64) / 25.0).cos());
        let posn_shift_y = scale(((tick as f64) / 25.0).sin());
        let posn_x = posn_shift_x + 350.0;
        let posn_y = posn_shift_y + 350.0;

        let size_shift_x = ((tick as f64) / 50.0).cos() * 25.0;
        let size_shift_y = ((tick as f64) / 50.0).sin() * 25.0;

        rc.draw_image(
            image,
            Rect::new(
                posn_x,
                posn_y,
                posn_x + 100.0 + size_shift_x,
                posn_y + 100.0 + size_shift_y,
            ),
            piet::InterpolationMode::Bilinear,
        );

        // Also draw a subregion of the image.
        let out_rect = Rect::new(100.0, 400.0, 200.0, 500.0);
        rc.draw_image_area(
            image,
            Rect::new(
                25.0 + posn_shift_x,
                25.0 + posn_shift_y,
                100.0 + posn_shift_x,
                100.0 + posn_shift_y,
            ),
            out_rect,
            piet::InterpolationMode::Bilinear,
        );
        rc.stroke(out_rect, outline, 3.0);

        // Text isn't supported on WASM yet.
        if cfg!(not(any(target_arch = "wasm32", target_arch = "wasm64"))) {
            use piet::{Text as _, TextLayout as _, TextLayoutBuilder as _};

            // Update the FPS counter, if necessary.
            num_frames += 1;
            let now = Instant::now();
            if now - last_second >= Duration::from_secs(1) {
                let fps_string = format!("Frames per Second: {num_frames}");
                let fps_text = rc
                    .text()
                    .new_text_layout(fps_string)
                    .font(piet::FontFamily::SERIF, 24.0)
                    .build()
                    .unwrap();

                current_fps = Some(fps_text);

                last_second = now;
                num_frames = 0;
            }

            // Draw the FPS counter.
            if let Some(current_fps) = current_fps.as_ref() {
                let size = current_fps.size();
                let pt = (
                    width as f64 - size.width - 10.0,
                    height as f64 - size.height - 10.0,
                );

                if pt.0 > 0.0 || pt.1 > 0.0 {
                    rc.draw_text(current_fps, pt);
                }
            }
        }

        tick += 1;
        rc.finish().unwrap();
    };

    util::run(draw)
}

fn generate_five_pointed_star(center: Point, inner_radius: f64, outer_radius: f64) -> BezPath {
    let point_from_polar = |radius: f64, angle: f64| {
        let x = center.x + radius * angle.cos();
        let y = center.y + radius * angle.sin();
        Point::new(x, y)
    };

    let one_fifth_circle = std::f64::consts::PI * 2.0 / 5.0;

    let outer_points = (0..5).map(|i| point_from_polar(outer_radius, one_fifth_circle * i as f64));
    let inner_points = (0..5).map(|i| {
        point_from_polar(
            inner_radius,
            one_fifth_circle * i as f64 + one_fifth_circle / 2.0,
        )
    });
    let mut points = outer_points.zip(inner_points).flat_map(|(a, b)| [a, b]);

    // Set up the path.
    let mut path = BezPath::new();
    path.move_to(points.next().unwrap());

    // Add the points to the path.
    for point in points {
        path.line_to(point);
    }

    // Close the path.
    path.close_path();
    path
}
