//! An example with a basic usage of the library.

include!("util/setup_context.rs");

use piet::kurbo::{Affine, BezPath, Circle, Point, Rect};
use piet::RenderContext as _;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A path representing a star.
    let path = generate_five_pointed_star(Point::new(0.0, 0.0), 75.0, 150.0);
    let circle_path = Circle::new(Point::new(200.0, 200.0), 150.0);
    let mut tick = 0;

    util::with_renderer(move |render_context| {
        // Clear the screen to a light blue.
        render_context.clear(None, piet::Color::rgb8(0x87, 0xce, 0xeb));

        // Add a clip.
        render_context.clip(Rect::new(0.0, 0.0, 400.0, 400.0));

        let red_star = {
            let rot = (tick % 360) as f64 / 180.0 * std::f64::consts::PI;
            let transform = Affine::translate((200.0, 200.0)) * Affine::rotate(rot);
            transform * (&path)
        };

        // Draw a solid red using the path.
        let solid_red = render_context.solid_brush(piet::Color::rgb8(0xff, 0x00, 0x00));
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
                rc.fill(&path, &solid);
                rc.stroke(&path, &outline, 5.0);

                Ok(())
            })
            .unwrap();

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
