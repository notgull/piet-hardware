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

//! Convenient wrappers around WGPU textures.

use super::{DeviceAndQueue, GpuContext};

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use piet_hardware::piet::{Color, ImageFormat, InterpolationMode};
use piet_hardware::RepeatStrategy;

/// The resource representing a WGPU texture.
#[derive(Clone)]
pub(crate) struct WgpuTexture(Rc<RefCell<TextureInner>>);

impl WgpuTexture {
    /// Create a new texture.
    pub(crate) fn create_texture<DaQ: DeviceAndQueue + ?Sized>(
        base: &GpuContext<DaQ>,
        interpolation: InterpolationMode,
        repeat: RepeatStrategy,
    ) -> Self {
        let id = base.next_id();
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

        let sampler = base
            .device_and_queue()
            .device()
            .create_sampler(&wgpu::SamplerDescriptor {
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
        })))
    }

    /// Borrow the inner texture.
    pub(crate) fn borrow(&self) -> BorrowedTexture<'_> {
        BorrowedTexture(self.0.borrow())
    }

    /// Borrow the inner texture mutably.
    pub(crate) fn borrow_mut(&self) -> BorrowedTextureMut<'_> {
        BorrowedTextureMut(self.0.borrow_mut())
    }
}

/// Borrowed texture guard.
pub(crate) struct BorrowedTexture<'a>(Ref<'a, TextureInner>);

impl BorrowedTexture<'_> {
    /// Get the bind group for this texture.
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        self.0.bind_group.as_ref().unwrap()
    }
}

/// Mutably borrowed texture guard.
pub(crate) struct BorrowedTextureMut<'a>(RefMut<'a, TextureInner>);

impl BorrowedTextureMut<'_> {
    /// Write data to this texture.
    pub(crate) fn write_texture<DaQ: DeviceAndQueue + ?Sized>(
        &mut self,
        base: &GpuContext<DaQ>,
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

        let data_len = data.map_or(0, |d| d.len());
        tracing::debug!(size=?size, format=?format, data_len=%data_len, "Writing a texture");

        // Get the texture to write to.
        let texture = if self.0.texture.is_none() || self.0.format != format {
            let texture =
                base.device_and_queue()
                    .device()
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(&format!("piet-wgpu texture {}", self.0.id)),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: match format {
                            ImageFormat::Grayscale => wgpu::TextureFormat::R8Unorm,
                            ImageFormat::Rgb => panic!("Unsupported"),
                            ImageFormat::RgbaPremul => wgpu::TextureFormat::Rgba8Unorm,
                            ImageFormat::RgbaSeparate => wgpu::TextureFormat::Rgba8Unorm,
                            _ => panic!("Unsupported"),
                        },
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                    });

            self.0.format = format;
            self.0.texture = Some(texture);

            // Reset the bind group.
            self.0.recompute_bind_group(base);

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
        base.device_and_queue().queue().write_texture(
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
    pub(crate) fn write_subtexture<DaQ: DeviceAndQueue + ?Sized>(
        &mut self,
        base: &GpuContext<DaQ>,
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
        base.device_and_queue().queue().write_texture(
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
    pub(crate) fn set_texture_interpolation<DaQ: DeviceAndQueue + ?Sized>(
        &mut self,
        base: &GpuContext<DaQ>,
        interpolation: InterpolationMode,
    ) {
        if self.0.interpolation != interpolation {
            let interp_mode = match interpolation {
                InterpolationMode::NearestNeighbor => wgpu::FilterMode::Nearest,
                InterpolationMode::Bilinear => wgpu::FilterMode::Linear,
            };

            self.0.interpolation = interpolation;
            self.0.sampler =
                base.device_and_queue()
                    .device()
                    .create_sampler(&wgpu::SamplerDescriptor {
                        label: Some(&format!("piet-wgpu sampler {}", self.0.id)),
                        compare: None,
                        mag_filter: interp_mode,
                        min_filter: interp_mode,
                        address_mode_u: self.0.address_mode,
                        address_mode_v: self.0.address_mode,
                        border_color: self.0.border_color,
                        ..Default::default()
                    });
            self.0.recompute_bind_group(base);
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
    bind_group: Option<wgpu::BindGroup>,
}

impl TextureInner {
    /// Re-create the `BindGroup` from the current data.
    fn recompute_bind_group<DaQ: DeviceAndQueue + ?Sized>(&mut self, base: &GpuContext<DaQ>) {
        let texture = match self.texture.as_ref() {
            Some(texture) => texture,
            None => {
                self.bind_group = None;
                return;
            }
        };

        let new_bind_group =
            base.device_and_queue()
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("piet-wgpu texture bind group {}", self.id)),
                    layout: base.texture_bind_layout(),
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

        self.bind_group = Some(new_bind_group);
    }
}

fn bytes_per_pixel(format: ImageFormat) -> u32 {
    match format {
        ImageFormat::Grayscale => 1u32,
        ImageFormat::Rgb => 3,
        ImageFormat::RgbaPremul => 4,
        ImageFormat::RgbaSeparate => 4,
        _ => panic!("Unsupported"),
    }
}
