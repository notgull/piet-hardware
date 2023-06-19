# Changelog

This log describes changes in the `piet-hardware`, `piet-glow` and `piet-wgpu` crates.

## piet-glow UNRELEASED

- Update to the newest `piet-hardware` version.

## piet-wgpu UNRELEASED

- **Breaking:** Adapt to the new middleware pattern.
- Update to the newest `piet-hardware` version.

## piet-hardware UNRELEASED

- **Breaking:** Add the `capture_area` method for capturing an area of the screen.
- **Breaking:** Change the `GpuContext` trait to use `&mut self` instead of `&self`.
- Add support for dashed lines.
- Fix some minor bugs.

## piet-wgpu 0.2.2

- Set default texture color space to non-SRGB.

## piet-hardware v0.2.1

- Add support for line decorations. 

## piet-glow 0.1.3

- Upgrade to `piet-hardware` v0.2.0

## piet-wgpu 0.2.1

- Upgrade to `piet-hardware` v0.2.0

## piet-hardware 0.2.0

- **Breaking:** The `push_buffers` method is now safe.