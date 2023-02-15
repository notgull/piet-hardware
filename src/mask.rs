//! Handles the mask.

use crate::resources::{BoundFramebuffer, Framebuffer, Texture};
use crate::{Error, RenderContext};

use glow::HasContext;
use piet::kurbo::{Affine, Shape};
use std::rc::Rc;

pub(super) struct Mask<H: HasContext + ?Sized> {
    /// The frame buffer for drawing to the mask.
    framebuffer: Framebuffer<H>,

    /// The texture for this mask.
    texture: Texture<H>,

    /// The transform for this mask.
    transform: Affine,

    /// The width and height of this mask.
    size: (u32, u32),

    /// Is this mask empty?
    empty: bool,
}

impl<H: HasContext + ?Sized> Mask<H> {
    /// Create a new, empty mask.
    pub(crate) fn new(
        context: &Rc<H>,
        width: u32,
        height: u32,
        transform: Affine,
    ) -> Result<Self, Error> {
        // Create a texture and fill it with nothingness.
        let texture = Texture::new(context)?;

        {
            let mut bound = texture.bind(None);
            bound.fill_with_nothing(width as _, height as _);
            bound.filtering_nearest();
        }

        // Create a framebuffer.
        let framebuffer = Framebuffer::new(context)?;

        Ok(Self {
            texture,
            framebuffer,
            transform,
            size: (width, height),
            empty: true,
        })
    }

    /// Set the transform of this mask.
    pub(crate) fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Clear out the mask.
    pub(crate) fn clear(&mut self) {
        let mut bound = self.texture.bind(None);
        bound.fill_with_nothing(self.size.0 as _, self.size.1 as _);
        self.empty = true;
    }

    /// Resize this mask, clearing it in the process.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.size = (width, height);
        self.clear();
    }

    /// Get the size of this mask.
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Get the transform of this mask.
    pub(crate) fn transform(&self) -> &Affine {
        &self.transform
    }

    /// Get the texture.
    pub(crate) fn texture(&self) -> &Texture<H> {
        &self.texture
    }

    /// Bind the framebuffer to the draw buffer.
    fn bind_framebuffer(&self) -> Result<BoundMask<'_, H>, Error> {
        let mut bound = self.framebuffer.bind();

        // Bind the texture as the first attachment.
        bound.bind_color0(&self.texture);

        // Check for errors.
        bound.check_error()?;

        // Keep it bound and unbind it when we're done.
        Ok(BoundMask { _bound: bound })
    }
}

impl<H: HasContext + ?Sized> RenderContext<'_, H> {
    /// Draw to a mask.
    pub(super) fn draw_to_mask(&mut self, mask: &mut Mask<H>, shape: impl Shape) {
        // Bind the framebuffer so we can draw to it.
        let _guard = match mask.bind_framebuffer() {
            Ok(guard) => guard,
            Err(e) => {
                self.last_error = Err(e);
                return;
            }
        };

        // Draw to the current mask.
        let draw = || todo!();

        todo!()
    }
}

struct BoundMask<'a, H: HasContext + ?Sized> {
    _bound: BoundFramebuffer<'a, H>,
}
