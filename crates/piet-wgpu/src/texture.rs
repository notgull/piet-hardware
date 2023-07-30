// SPDX-License-Identifier: LGPL-3.0-or-later OR MPL-2.0
// This file is a part of `piet-hardware`.
//
// `piet-hardware` is free software: you can redistribute it and/or modify it under the
// terms of either:
//
// * GNU Lesser General Public License as published by the Free Software Foundation, either
//   version 3 of the License, or (at your option) any later version.
// * Mozilla Public License as published by the Mozilla Foundation, version 2.
//
// `piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License or the Mozilla Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License and the Mozilla
// Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/>.

//! Convenient wrappers around WGPU textures.

use std::cell::{RefCell, RefMut};
use std::rc::Rc;

use piet_hardware::piet::{Color, ImageFormat, InterpolationMode};
use piet_hardware::RepeatStrategy;

/// The resource representing a WGPU texture.
#[derive(Clone)]
pub(crate) struct WgpuTexture(Rc<RefCell<TextureInner>>);

impl WgpuTexture {
    /// Create a new texture.
    pub(crate) fn create_texture(
        id: usize,
        device: &wgpu::Device,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
        renderable: bool,
    ) -> Self {
        let filter_mode = match interpolation {
            InterpolationMode::Bilinear => wgpu::FilterMode::Linear,
            InterpolationMode::NearestNeighbor => wgpu::FilterMode::Nearest,
        };

        let mut border_color = None;
        let address_mode = match repeat {
            RepeatStrategy::Clamp => wgpu::AddressMode::ClampToEdge,
            RepeatStrategy::Repeat => wgpu::AddressMode::Repeat,
            RepeatStrategy::Color(color) => {
                border_color = Some({
                    if color == Color::TRANSPARENT {
                        wgpu::SamplerBorderColor::TransparentBlack
                    } else if color == Color::BLACK {
                        wgpu::SamplerBorderColor::OpaqueBlack
                    } else if color == Color::WHITE {
                        wgpu::SamplerBorderColor::OpaqueWhite
                    } else {
                        tracing::warn!("Invalid border color for sampler: {:?}", color);
                        wgpu::SamplerBorderColor::OpaqueWhite
                    }
                });

                wgpu::AddressMode::ClampToBorder
            }
            _ => panic!("unknown repeat strategy"),
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("piet-wgpu sampler {id}")),
            compare: None,
            mag_filter: filter_mode,
            min_filter: filter_mode,
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            border_color,
            ..Default::default()
        });

        WgpuTexture(Rc::new(RefCell::new(TextureInner {
            id,
            texture: None,
            format: ImageFormat::Grayscale,
            sampler,
            interpolation,
            border_color,
            address_mode,
            bind_group: None,
            renderable,
        })))
    }

    /// Borrow the inner texture mutably.
    pub(crate) fn borrow_mut(&self) -> BorrowedTextureMut<'_> {
        BorrowedTextureMut(self.0.borrow_mut())
    }

    /// Clone out the bind group for this texture.
    pub(crate) fn bind_group(&self) -> Rc<wgpu::BindGroup> {
        self.0
            .borrow()
            .bind_group
            .as_ref()
            .expect("texture not bound yet")
            .clone()
    }
}

/// Mutably borrowed texture guard.
pub(crate) struct BorrowedTextureMut<'a>(RefMut<'a, TextureInner>);

impl BorrowedTextureMut<'_> {
    /// Get a reference to the inner texture.
    pub(crate) fn texture(&self) -> Option<&wgpu::Texture> {
        self.0.texture.as_ref()
    }

    /// Write data to this texture.
    pub(crate) fn write_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group: &wgpu::BindGroupLayout,
        size: (u32, u32),
        format: ImageFormat,
        data: Option<&[u8]>,
    ) {
        let bytes_per_pixel = bytes_per_pixel(format);

        let size = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };

        let formatted_data;
        let mut data = data;

        let wgpu_format = match format {
            ImageFormat::Grayscale => {
                // Translate to RGBA.
                formatted_data = data.map(|data| {
                    let mut new_data = Vec::with_capacity(data.len() * 4);
                    for &byte in data {
                        new_data.push(byte);
                        new_data.push(byte);
                        new_data.push(byte);
                        new_data.push(255);
                    }
                    new_data
                });
                data = formatted_data.as_deref();
                wgpu::TextureFormat::Rgba8Unorm
            }
            ImageFormat::Rgb => {
                // Translate to RGBA.
                formatted_data = data.map(|data| {
                    let mut new_data = Vec::with_capacity(data.len() * 4);
                    for chunk in data.chunks(3) {
                        new_data.push(chunk[0]);
                        new_data.push(chunk[1]);
                        new_data.push(chunk[2]);
                        new_data.push(255);
                    }
                    new_data
                });
                data = formatted_data.as_deref();
                wgpu::TextureFormat::Rgba8Unorm
            }
            ImageFormat::RgbaPremul => wgpu::TextureFormat::Rgba8Unorm,
            ImageFormat::RgbaSeparate => wgpu::TextureFormat::Rgba8Unorm,
            _ => panic!("Unsupported"),
        };

        let data_len = data.map_or(0, |d| d.len());
        tracing::debug!(?size, ?format, %data_len, "Writing a texture");

        // Get the texture to write to.
        let needs_new_texture = self
            .0
            .texture
            .as_ref()
            .map_or(true, |tex| tex.size() != size)
            || self.0.format != format;
        let texture = if needs_new_texture {
            let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
            if self.0.renderable {
                usage |= wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
            }

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("piet-wgpu texture {}", self.0.id)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage,
                view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            });

            self.0.format = format;
            self.0.texture = Some(texture);

            // Reset the bind group.
            self.0.recompute_bind_group(device, texture_bind_group);

            self.0.texture.as_ref().unwrap()
        } else {
            self.0.texture.as_ref().unwrap()
        };

        let zeroes;
        let data = match data {
            Some(data) => data,
            None => {
                zeroes =
                    vec![0; size.width as usize * size.height as usize * bytes_per_pixel as usize];
                &zeroes
            }
        };

        // Queue a data write to the texture.
        let data_layout = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(size.width * bytes_per_pixel),
            rows_per_image: Some(size.height),
        };

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            data_layout,
            size,
        );
    }

    /// Write to a sub-area of this texture.
    pub(crate) fn write_subtexture(
        &mut self,
        queue: &wgpu::Queue,
        offset: (u32, u32),
        size: (u32, u32),
        format: piet_hardware::piet::ImageFormat,
        data: &[u8],
    ) {
        if self.0.format != format {
            panic!("write_subtexture format mismatch");
        }

        let bytes_per_pixel = bytes_per_pixel(format);

        // Queue a data write to the texture.
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: self.0.texture.as_ref().expect("texture"),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset.0,
                    y: offset.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size.0 * bytes_per_pixel),
                rows_per_image: Some(size.1),
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Change the interpolation mode.
    pub(crate) fn set_texture_interpolation(
        &mut self,
        device: &wgpu::Device,
        texture_bind_group: &wgpu::BindGroupLayout,
        interpolation: InterpolationMode,
    ) {
        if self.0.interpolation != interpolation {
            let interp_mode = match interpolation {
                InterpolationMode::NearestNeighbor => wgpu::FilterMode::Nearest,
                InterpolationMode::Bilinear => wgpu::FilterMode::Linear,
            };

            self.0.interpolation = interpolation;
            self.0.sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(&format!("piet-wgpu sampler {}", self.0.id)),
                compare: None,
                mag_filter: interp_mode,
                min_filter: interp_mode,
                address_mode_u: self.0.address_mode,
                address_mode_v: self.0.address_mode,
                border_color: self.0.border_color,
                ..Default::default()
            });
            self.0.recompute_bind_group(device, texture_bind_group);
        }
    }
}

/// Inner data for a texture.
struct TextureInner {
    /// The texture ID.
    id: usize,

    /// The texture.
    texture: Option<wgpu::Texture>,

    /// The sampler to use.
    sampler: wgpu::Sampler,

    /// The image format we used to render.
    format: ImageFormat,

    /// The interpolation mode.
    interpolation: InterpolationMode,

    /// The address mode.
    address_mode: wgpu::AddressMode,

    /// The border color.
    border_color: Option<wgpu::SamplerBorderColor>,

    /// The bind group to use to bind to the pipeline.
    bind_group: Option<Rc<wgpu::BindGroup>>,

    /// Whether to make the texture renderable.
    renderable: bool,
}

impl TextureInner {
    /// Re-create the `BindGroup` from the current data.
    fn recompute_bind_group(
        &mut self,
        device: &wgpu::Device,
        texture_bind_group: &wgpu::BindGroupLayout,
    ) {
        let texture = match self.texture.as_ref() {
            Some(texture) => texture,
            None => {
                self.bind_group = None;
                return;
            }
        };

        let new_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("piet-wgpu texture bind group {}", self.id)),
            layout: texture_bind_group,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.bind_group = Some(Rc::new(new_bind_group));
    }
}

fn bytes_per_pixel(format: ImageFormat) -> u32 {
    match format {
        ImageFormat::Grayscale => 4u32,
        ImageFormat::Rgb => 4,
        ImageFormat::RgbaPremul => 4,
        ImageFormat::RgbaSeparate => 4,
        _ => panic!("Unsupported"),
    }
}
