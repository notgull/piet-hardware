//! Shader for drawing text.

use euclid::default::{Point, Rect, Size};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct GlyphPoint {
    /// The position of the point.
    pub(super) position: Point<f32>,
}
