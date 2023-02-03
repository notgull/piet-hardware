//! Shader manipulation.

use std::rc::Rc;

/// A compiled OpenGL fragment shader.
pub(super) struct CompiledShader<H> {
    context: Rc<H>
}
