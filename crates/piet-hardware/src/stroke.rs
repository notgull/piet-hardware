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

//! A backup stroke implementation using the `zeno` crate.
//!
//! `lyon_tesselation` on its own does not support dashing properly, and `tiny-skia` looks like it fails
//! on some sample patterns. Therefore we have to fall back to `zeno`.

use crate::FillRule;
use piet::kurbo::{BezPath, PathEl, Shape};
use zeno::{PathBuilder, PathData, Scratch};

/// A buffer for stroking a path.
pub(crate) struct StrokeBuffer {
    /// Scratch buffer for stroking.
    scratch: Scratch,

    /// The input buffer for stroking.
    input: BezPath,

    /// The output buffer for stroking.
    output: BezPath,

    /// The fill rule to use for filling eventually.
    fill_rule: FillRule,

    /// Buffer to hold dashes.
    dashes: Vec<f32>,
}

impl Default for StrokeBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl StrokeBuffer {
    /// Create a new stroke buffer.
    pub(crate) fn new() -> Self {
        Self {
            scratch: Scratch::new(),
            input: BezPath::new(),
            output: BezPath::new(),
            fill_rule: FillRule::NonZero,
            dashes: vec![],
        }
    }

    /// Draw the stroke style into the output buffer.
    pub(crate) fn draw(
        &mut self,
        shape: impl Shape,
        width: f64,
        style: &piet::StrokeStyle,
        tolerance: f64,
    ) {
        // TODO: Once specialization exists, we can specialize on whether or not `shape`'s iterator
        // is `Clone` to avoid this allocation.
        self.input.extend(shape.path_elements(tolerance));

        // Convert the stroke style to a zeno stroke.
        let mut stroke = zeno::Stroke::new(width as f32);
        stroke.cap(match style.line_cap {
            piet::LineCap::Butt => zeno::Cap::Butt,
            piet::LineCap::Round => zeno::Cap::Round,
            piet::LineCap::Square => zeno::Cap::Square,
        });
        let join = match style.line_join {
            piet::LineJoin::Miter { limit } => {
                stroke.miter_limit(limit as f32);
                zeno::Join::Miter
            }
            piet::LineJoin::Round => zeno::Join::Round,
            piet::LineJoin::Bevel => zeno::Join::Bevel,
        };
        stroke.join(join);

        if !style.dash_pattern.is_empty() {
            self.dashes.clear();
            self.dashes
                .extend(style.dash_pattern.iter().map(|x| *x as f32));

            stroke.dash(&self.dashes, style.dash_offset as f32);
        }

        // Stroke the path.
        self.output.truncate(0);
        let fill = self.scratch.apply(
            PietShapeAsZenoPathData {
                shape: &self.input,
                tolerance,
            },
            stroke,
            None,
            &mut BezPathCommandReceiver(&mut self.output),
        );
        self.fill_rule = match fill {
            zeno::Fill::NonZero => FillRule::NonZero,
            zeno::Fill::EvenOdd => FillRule::EvenOdd,
        };
    }

    /// Get the path data for the stroked path.
    pub(crate) fn path_data(&self) -> &BezPath {
        &self.output
    }

    /// Get the fill rule for the stroked path.
    pub(crate) fn fill_rule(&self) -> FillRule {
        self.fill_rule
    }
}

/// A wrapper around `Shape` that implements `PathData`.
struct PietShapeAsZenoPathData<'a, S> {
    /// The underlying shape.
    shape: &'a S,

    /// The tolerance to render the curve at.
    tolerance: f64,
}

impl<'a, S: Shape> PathData for PietShapeAsZenoPathData<'a, S>
where
    S::PathElementsIter<'a>: Clone,
{
    type Commands = PietShapeToZenoCommands<S::PathElementsIter<'a>>;

    fn commands(&self) -> Self::Commands {
        PietShapeToZenoCommands(self.shape.path_elements(self.tolerance))
    }
}

#[derive(Clone)]
struct PietShapeToZenoCommands<I>(I);

impl<I: Iterator<Item = PathEl>> Iterator for PietShapeToZenoCommands<I> {
    type Item = zeno::Command;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(PathEl::MoveTo(p)) => Some(zeno::Command::MoveTo(cvt_point(p))),
            Some(PathEl::LineTo(p)) => Some(zeno::Command::LineTo(cvt_point(p))),
            Some(PathEl::QuadTo(p1, p2)) => {
                Some(zeno::Command::QuadTo(cvt_point(p1), cvt_point(p2)))
            }
            Some(PathEl::CurveTo(p1, p2, p3)) => Some(zeno::Command::CurveTo(
                cvt_point(p1),
                cvt_point(p2),
                cvt_point(p3),
            )),
            Some(PathEl::ClosePath) => Some(zeno::Command::Close),
            None => None,
        }
    }
}

struct BezPathCommandReceiver<'a>(&'a mut BezPath);

impl PathBuilder for BezPathCommandReceiver<'_> {
    fn current_point(&self) -> zeno::Point {
        let mut segments = self.0.elements().iter().rev();
        match segments.next() {
            None => zeno::Point::ZERO,
            Some(cmd) => match cmd {
                PathEl::MoveTo(p)
                | PathEl::LineTo(p)
                | PathEl::QuadTo(_, p)
                | PathEl::CurveTo(_, _, p) => cvt_point(*p),
                PathEl::ClosePath => segments
                    .find_map(|el| match el {
                        PathEl::MoveTo(p) => Some(cvt_point(*p)),
                        _ => None,
                    })
                    .unwrap_or(zeno::Point::ZERO),
            },
        }
    }

    fn move_to(&mut self, to: impl Into<zeno::Point>) -> &mut Self {
        self.0.move_to(cvt_point_r(to));
        self
    }

    fn line_to(&mut self, to: impl Into<zeno::Point>) -> &mut Self {
        self.0.line_to(cvt_point_r(to));
        self
    }

    fn curve_to(
        &mut self,
        control1: impl Into<zeno::Point>,
        control2: impl Into<zeno::Point>,
        to: impl Into<zeno::Point>,
    ) -> &mut Self {
        self.0.curve_to(
            cvt_point_r(control1),
            cvt_point_r(control2),
            cvt_point_r(to),
        );
        self
    }

    fn quad_to(
        &mut self,
        control1: impl Into<zeno::Point>,
        to: impl Into<zeno::Point>,
    ) -> &mut Self {
        self.0.quad_to(cvt_point_r(control1), cvt_point_r(to));
        self
    }

    fn close(&mut self) -> &mut Self {
        self.0.close_path();
        self
    }
}

fn cvt_point(p: kurbo::Point) -> zeno::Vector {
    zeno::Vector::new(p.x as f32, p.y as f32)
}

fn cvt_point_r(z: impl Into<zeno::Point>) -> kurbo::Point {
    let z = z.into();
    kurbo::Point::new(z.x as f64, z.y as f64)
}
