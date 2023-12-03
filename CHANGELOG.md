# Changelog

This log describes changes in the `piet-hardware`, `piet-glow` and `piet-wgpu`
crates.

## piet-hardware 0.4.0

- **Breaking:** Add a new scale parameter to `capture_area`.

## piet-hardware 0.3.0

- **Breaking:** Add the `capture_area` method for capturing an area of the screen.
- **Breaking:** Change the `GpuContext` trait to use `&mut self` instead of `&self`.
- Add support for dashed lines.
- Fix some minor bugs.
- Improved text handling.
- Fix bugs in the clipping code.

## piet-hardware v0.2.1

- Add support for line decorations.

## piet-hardware 0.2.0

- **Breaking:** The `push_buffers` method is now safe.
