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

use piet::kurbo::{Affine, BezPath, Circle, Point, Rect};
use piet::RenderContext as _;

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A path representing a star.
    let star = generate_five_pointed_star(Point::new(0.0, 0.0), 75.0, 150.0);
    let circle_path = Circle::new(Point::new(200.0, 200.0), 150.0);
    let mut tick = 0;

    // Get the test image at $CRATE_ROOT/examples/assets/test-image.png
    let manifest_root = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest_root).join("examples/assets/test-image.png");
    let image = image::open(path)?.to_rgba8();

    // Convert the image to a byte buffer.
    let size = image.dimensions();
    let image_data = image.into_raw();

    util::with_renderer(move |render_context, width, height| {
        // Clear the screen to a light blue.
        render_context.clear(None, piet::Color::rgb8(0x87, 0xce, 0xeb));

        // Add a clip.
        //render_context.clip(circle_path);

        let red_star = {
            let rot = (tick % 360) as f64 / 180.0 * std::f64::consts::PI;
            let transform = Affine::translate((200.0, 200.0)) * Affine::rotate(rot);
            transform * (&star)
        };

        // Draw a solid red using the path.
        let solid_red = render_context.solid_brush(piet::Color::rgb8(0xff, 0x00, 0x00));
        render_context.fill(Rect::new(0.0, 0.0, 200.0, 200.0), &solid_red);
        render_context.fill(&red_star, &solid_red);

        // Draw a black outline using the path.
        let outline = render_context.solid_brush(piet::Color::rgb8(0x00, 0x00, 0x00));
        render_context.stroke(&red_star, &outline, 5.0);

        // Test the transform.
        render_context
            .with_save(|rc| {
                let rot = ((tick * 2) % 360) as f64 / 180.0 * std::f64::consts::PI;
                let trans =
                    Affine::scale(0.75) * Affine::translate((750.0, 275.0)) * Affine::rotate(rot);
                let solid = rc.solid_brush(piet::Color::rgb8(0x00, 0xff, 0x00));

                rc.transform(trans);
                rc.fill(&star, &solid);
                rc.stroke(&star, &outline, 5.0);

                Ok(())
            })
            .unwrap();

        // Create an image and draw it.
        let image = render_context
            .make_image(
                size.0 as _,
                size.1 as _,
                &image_data,
                piet::ImageFormat::RgbaSeparate,
            )
            .unwrap();

        let posn_shift_x = 0.0;//((tick as f64) / 25.0).cos() * 50.0;
        let posn_shift_y = 0.0;//((tick as f64) / 25.0).sin() * 50.0;
        let posn_x = posn_shift_x + 400.0;
        let posn_y = posn_shift_y + 400.0;

        let size_shift_x = ((tick as f64) / 50.0).cos() * 50.0;
        let size_shift_y = ((tick as f64) / 50.0).cos() * 50.0;       

        render_context.draw_image(
            &image,
            Rect::new(posn_x, posn_y, posn_x + 100.0 + size_shift_x, posn_y + 100.0 + size_shift_y),
            piet::InterpolationMode::Bilinear,
        );

        // Panic on any errors.
        render_context.finish().unwrap();
        render_context.status().unwrap();

        tick += 1;
    })
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
