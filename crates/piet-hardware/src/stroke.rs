// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the
// terms of either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
//   version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
// * The Patron License (https://github.com/notgull/piet-hardware/blob/main/LICENSE-PATRON.md)
//   for sponsors and contributors, who can ignore the copyleft provisions of the above licenses
//   for this project.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License or the Mozilla Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/>.

//! Advanced stroke rasterization using `zeno`.

use crate::FillRule;
use kurbo::{BezPath, PathEl, Shape};
use piet::StrokeStyle;
use zeno::{Command, PathData, Scratch};

/// The buffers for stroking a path.
pub(crate) struct StrokeBuffer {
    /// The scratch buffer for rendering.
    scratch: Scratch,

    /// Buffer for rendering the `kurbo` path.
    input_buffer: BezPath,

    /// Buffer for rendering the stroked path.
    output_buffer: Vec<Command>,

    /// The second output buffer for rendering the stroked path.
    kurbo_output: BezPath,

    /// The fill rule to use when stroking.
    fill_rule: FillRule,

    /// Dash length buffer.
    dashes: Vec<f32>,
}

impl Default for StrokeBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl StrokeBuffer {
    /// Create a new `StrokeBuffer`.
    pub(crate) fn new() -> Self {
        Self {
            scratch: Scratch::new(),
            input_buffer: BezPath::new(),
            output_buffer: Vec::new(),
            kurbo_output: BezPath::new(),
            fill_rule: FillRule::NonZero,
            dashes: Vec::new(),
        }
    }

    /// Render the stroked path.
    pub(crate) fn render_into(
        &mut self,
        shape: impl Shape,
        width: f64,
        style: &StrokeStyle,
        tolerance: f64,
    ) {
        // Adapting `Shape` to `zeno::PathData` requires the inner iterator to be `Clone`.
        // Since this can't be guaranteed by arbitrary shapes, we need to collect it into a
        // BezPath.
        //
        // TODO: Once specialization is stable in Rust, we can specialize so that we can use
        // the `Shape` if its iterator is `Clone`, and fall back to collecting into a `BezPath`
        // otherwise.
        self.input_buffer.truncate(0);
        self.input_buffer.extend(shape.path_elements(tolerance));

        // Set up the zeno path style.
        let stroke = {
            let mut stroke = zeno::Stroke::new(width as f32);
            stroke.cap(match style.line_cap {
                piet::LineCap::Butt => zeno::Cap::Butt,
                piet::LineCap::Round => zeno::Cap::Round,
                piet::LineCap::Square => zeno::Cap::Square,
            });

            let join_style = match style.line_join {
                piet::LineJoin::Bevel => zeno::Join::Bevel,
                piet::LineJoin::Miter { limit } => {
                    stroke.miter_limit(limit as f32);
                    zeno::Join::Miter
                }
                piet::LineJoin::Round => zeno::Join::Round,
            };
            stroke.join(join_style);

            // Set up the dash pattern.
            self.dashes.clear();
            self.dashes
                .extend(style.dash_pattern.iter().map(|&d| d as f32));
            stroke.dash(&self.dashes, style.dash_offset as f32);

            stroke
        };

        // Render the stroked path.
        self.output_buffer.clear();
        let fill_rule = self.scratch.apply(
            &KurboShapeAsZenoPathData {
                shape: &self.input_buffer,
                tolerance,
            },
            stroke,
            None,
            &mut self.output_buffer,
        );

        // Convert the zeno path to a kurbo path.
        // TODO: Figure out how to make this part unnecessary.
        self.kurbo_output.truncate(0);
        self.kurbo_output
            .extend(self.output_buffer.iter().map(|&cmd| cvt_command(cmd)));

        // Convert the fill rule.
        self.fill_rule = match fill_rule {
            zeno::Fill::EvenOdd => FillRule::EvenOdd,
            zeno::Fill::NonZero => FillRule::NonZero,
        };
    }

    /// Get the latest output buffer.
    pub(crate) fn output_buffer(&self) -> &BezPath {
        &self.kurbo_output
    }

    /// Get the latest fill rule.
    pub(crate) fn fill_rule(&self) -> FillRule {
        self.fill_rule
    }
}

/// Represent a `kurbo::Shape` as a `zeno::PathData`.
struct KurboShapeAsZenoPathData<'a, S> {
    /// The shape to render.
    shape: &'a S,

    /// The tolerance to use when rendering the shape.
    tolerance: f64,
}

impl<'a, S: Shape> PathData for KurboShapeAsZenoPathData<'a, S>
where
    S::PathElementsIter<'a>: Clone,
{
    type Commands = PathElToCommandIter<S::PathElementsIter<'a>>;

    fn commands(&self) -> Self::Commands {
        PathElToCommandIter(self.shape.path_elements(self.tolerance))
    }
}

/// Iterator that converts `PathEl` to `Command`.
#[derive(Clone)]
struct PathElToCommandIter<I>(I);

impl<I: Iterator<Item = PathEl>> Iterator for PathElToCommandIter<I> {
    type Item = Command;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(cvt_path_el)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn last(self) -> Option<Self::Item> {
        self.0.last().map(cvt_path_el)
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.0.fold(init, move |acc, el| f(acc, cvt_path_el(el)))
    }

    fn all<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.0.all(|el| f(cvt_path_el(el)))
    }

    fn any<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.0.any(|el| f(cvt_path_el(el)))
    }

    fn collect<B: FromIterator<Self::Item>>(self) -> B {
        self.0.map(cvt_path_el).collect()
    }
}

/// Convert a `kurbo::PathEl` into a `zeno::Command`.
fn cvt_path_el(el: PathEl) -> Command {
    match el {
        PathEl::MoveTo(p) => Command::MoveTo(cvt_point(p)),
        PathEl::LineTo(p) => Command::LineTo(cvt_point(p)),
        PathEl::QuadTo(p1, p2) => Command::QuadTo(cvt_point(p1), cvt_point(p2)),
        PathEl::CurveTo(p1, p2, p3) => {
            Command::CurveTo(cvt_point(p1), cvt_point(p2), cvt_point(p3))
        }
        PathEl::ClosePath => Command::Close,
    }
}

/// Convert a `zeno::Command` to a `kurbo::PathEl`.
fn cvt_command(cmd: Command) -> PathEl {
    match cmd {
        Command::MoveTo(p) => PathEl::MoveTo(cvt_vector(p)),
        Command::LineTo(p) => PathEl::LineTo(cvt_vector(p)),
        Command::QuadTo(p1, p2) => PathEl::QuadTo(cvt_vector(p1), cvt_vector(p2)),
        Command::CurveTo(p1, p2, p3) => {
            PathEl::CurveTo(cvt_vector(p1), cvt_vector(p2), cvt_vector(p3))
        }
        Command::Close => PathEl::ClosePath,
    }
}

/// Convert a `kurbo::Point` into a `zeno::Vector`.
fn cvt_point(p: kurbo::Point) -> zeno::Vector {
    zeno::Vector::new(p.x as f32, p.y as f32)
}

/// Convert a `zeno::Vector` into a `kurbo::Point`.
fn cvt_vector(v: zeno::Vector) -> kurbo::Point {
    kurbo::Point::new(v.x as f64, v.y as f64)
}
