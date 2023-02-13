//! An example with a basic usage of the library.

include!("util/setup_context.rs");

use piet::kurbo::{BezPath, Point};
use piet::RenderContext as _;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A path representing a star.
    let path = generate_five_pointed_star(
        Point::new(200.0, 200.0),
        100.0,
        200.0
    );

    util::with_renderer(move |render_context| {
        // Clear the screen to a light blue.
        render_context.clear(None, piet::Color::rgb8(0x87, 0xce, 0xeb));

        // Draw a solid red using the path.
        let solid = render_context.solid_brush(piet::Color::rgb8(0xff, 0x00, 0x00));
        render_context.fill(&path, &solid);

        // Draw a black outline using the path.
        let outline = render_context.solid_brush(piet::Color::rgb8(0x00, 0x00, 0x00));
        render_context.stroke(&path, &outline, 5.0);

        // Panic on any errors.
        render_context.finish().unwrap();
        render_context.status().unwrap();
    })
}

fn generate_five_pointed_star(
    center: Point,
    inner_radius: f64,
    outer_radius: f64
) -> BezPath {
    let point_from_polar = |radius: f64, angle: f64| {
        let x = center.x + radius * angle.cos();
        let y = center.y + radius * angle.sin();
        Point::new(x, y)
    };

    let one_fifth_circle = std::f64::consts::PI * 2.0 / 5.0; 

    let outer_points = (0..5)
        .map(|i| point_from_polar(outer_radius, one_fifth_circle * i as f64));
    let inner_points = (0..5)
        .map(|i| point_from_polar(inner_radius, one_fifth_circle * i as f64 + one_fifth_circle / 2.0));
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
