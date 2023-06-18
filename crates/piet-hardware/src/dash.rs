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

//! Handles the dashing of lines using the `zeno` crate.

use super::FillRule;
use piet::kurbo::{BezPath, PathEl, Shape};
use piet::StrokeStyle;
use zeno::{Command, PathBuilder, PathData, Vector};

/// A buffer for holding the result of a dash operation.
pub(crate) struct DashBuffer {
    /// The scratch buffer for the dash operation.
    scratch: zeno::Scratch,

    /// The input bezier path for the dash operation.
    input: BezPath,

    /// The output path builder for the dash operation.
    pub(crate) output: BezPath,

    /// The fill rule for the dash operation.
    pub(crate) fill_rule: FillRule,

    /// Dash pattern buffer.
    dash_pattern: Vec<f32>,
}

impl DashBuffer {
    /// Create an empty dash buffer.
    pub(crate) fn new() -> Self {
        Self {
            scratch: zeno::Scratch::new(),
            input: BezPath::new(),
            output: BezPath::new(),
            fill_rule: FillRule::EvenOdd,
            dash_pattern: Vec::new(),
        }
    }

    /// Given an input shape, write the dashed version of it to the output path.
    ///
    /// This overwrites the previous contents of the output path.
    pub(crate) fn write_stroke(
        &mut self,
        shape: impl Shape,
        width: f64,
        style: &StrokeStyle,
        tolerance: f64,
    ) {
        // TODO: Once specialization is in stable Rust, use it to avoid needing to buffer inputs.
        // If the iterator is Clone, we can just pass it directly to `apply()`.
        self.input.truncate(0);
        self.input.extend(shape.path_elements(tolerance));

        // Create the stroking style.
        let mut stroke = zeno::Stroke::new(width as f32);

        let join = match style.line_join {
            piet::LineJoin::Bevel => zeno::Join::Bevel,
            piet::LineJoin::Miter { limit } => {
                stroke.miter_limit(limit as f32);
                zeno::Join::Miter
            }
            piet::LineJoin::Round => zeno::Join::Round,
        };
        stroke.join(join);
        stroke.cap(cvt_cap(style.line_cap));

        if !style.dash_pattern.is_empty() {
            self.dash_pattern.clear();
            self.dash_pattern
                .extend(style.dash_pattern.iter().map(|x| *x as f32));

            stroke.dash(&self.dash_pattern, style.dash_offset as f32);
        }

        // Stroke out the path.
        self.output.truncate(0);
        let fill = self.scratch.apply(
            PietShapeToZenoPathData {
                shape: &self.input,
                tolerance,
            },
            stroke,
            None,
            &mut BezPathBuilder(&mut self.output),
        );

        self.fill_rule = match fill {
            zeno::Fill::EvenOdd => FillRule::EvenOdd,
            zeno::Fill::NonZero => FillRule::NonZero,
        };
    }
}

/// Converts a `piet` `Shape` to a `zeno` `PathData`.
struct PietShapeToZenoPathData<'a, P> {
    shape: &'a P,
    tolerance: f64,
}

impl<'a, P: Shape> PathData for PietShapeToZenoPathData<'a, P>
where
    P::PathElementsIter<'a>: Clone,
{
    type Commands = CommandIter<P::PathElementsIter<'a>>;

    fn commands(&self) -> Self::Commands {
        CommandIter(self.shape.path_elements(self.tolerance))
    }
}

/// The iterator over commands for `PietShapeToZenoPathData`.
#[derive(Clone)]
struct CommandIter<I>(I);

impl<I: Iterator<Item = PathEl>> Iterator for CommandIter<I> {
    type Item = Command;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(path_el_to_command)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n).map(path_el_to_command)
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.0
            .fold(init, move |acc, el| f(acc, path_el_to_command(el)))
    }
}

/// Applies the `PathBuilder` trait to `BezPath`.
struct BezPathBuilder<'a>(&'a mut BezPath);

impl PathBuilder for BezPathBuilder<'_> {
    fn current_point(&self) -> zeno::Point {
        let mut points_iter = self.0.elements().iter().rev();

        match points_iter.next() {
            Some(PathEl::MoveTo(p))
            | Some(PathEl::LineTo(p))
            | Some(PathEl::QuadTo(_, p))
            | Some(PathEl::CurveTo(_, _, p)) => cvt_point_r(*p),
            Some(PathEl::ClosePath) => points_iter
                .find_map(|el| match el {
                    PathEl::MoveTo(p) => Some(cvt_point_r(*p)),
                    _ => None,
                })
                .unwrap_or(zeno::Point::ZERO),
            None => zeno::Point::ZERO,
        }
    }

    fn move_to(&mut self, to: impl Into<zeno::Point>) -> &mut Self {
        self.0.move_to(cvt_point(to.into()));
        self
    }

    fn line_to(&mut self, to: impl Into<zeno::Point>) -> &mut Self {
        self.0.line_to(cvt_point(to.into()));
        self
    }

    fn curve_to(
        &mut self,
        control1: impl Into<zeno::Point>,
        control2: impl Into<zeno::Point>,
        to: impl Into<zeno::Point>,
    ) -> &mut Self {
        self.0.curve_to(
            cvt_point(control1.into()),
            cvt_point(control2.into()),
            cvt_point(to.into()),
        );
        self
    }

    fn quad_to(
        &mut self,
        control1: impl Into<zeno::Point>,
        to: impl Into<zeno::Point>,
    ) -> &mut Self {
        self.0
            .quad_to(cvt_point(control1.into()), cvt_point(to.into()));
        self
    }

    fn close(&mut self) -> &mut Self {
        self.0.close_path();
        self
    }
}

fn path_el_to_command(path_el: PathEl) -> Command {
    match path_el {
        PathEl::MoveTo(p) => Command::MoveTo(Vector {
            x: p.x as f32,
            y: p.y as f32,
        }),
        PathEl::LineTo(p) => Command::LineTo(Vector {
            x: p.x as f32,
            y: p.y as f32,
        }),
        PathEl::QuadTo(p1, p2) => Command::QuadTo(
            Vector {
                x: p1.x as f32,
                y: p1.y as f32,
            },
            Vector {
                x: p2.x as f32,
                y: p2.y as f32,
            },
        ),
        PathEl::CurveTo(p1, p2, p3) => Command::CurveTo(
            Vector {
                x: p1.x as f32,
                y: p1.y as f32,
            },
            Vector {
                x: p2.x as f32,
                y: p2.y as f32,
            },
            Vector {
                x: p3.x as f32,
                y: p3.y as f32,
            },
        ),
        PathEl::ClosePath => Command::Close,
    }
}

fn cvt_point(pt: zeno::Point) -> piet::kurbo::Point {
    piet::kurbo::Point::new(pt.x as f64, pt.y as f64)
}

fn cvt_point_r(pt: piet::kurbo::Point) -> zeno::Point {
    zeno::Point::new(pt.x as f32, pt.y as f32)
}

fn cvt_cap(cap: piet::LineCap) -> zeno::Cap {
    match cap {
        piet::LineCap::Butt => zeno::Cap::Butt,
        piet::LineCap::Round => zeno::Cap::Round,
        piet::LineCap::Square => zeno::Cap::Square,
    }
}
