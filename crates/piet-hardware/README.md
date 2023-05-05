# piet-hardware

`piet-hardware` is a strategy for implementing the [`piet`] drawing interface using GPU primitives. The goal is to break down the drawing operations to rendering textured triangles. The resulting buffers are than passed to the GPU backend for rendering.

As `piet-hardware` simply implements the high-level strategy, it has no unsafe code. The actual GPU calls are forwarded to an object that implements `GpuContext`. This object is intended to be an interface to OpenGL, Vulkan, Metal, or other GPU APIs.

[`piet`]: https://crates.io/crates/piet

## License

`piet-hardware` is free software: you can redistribute it and/or modify it under the terms of
either:

* GNU Lesser General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.
* Mozilla Public License as published by the Mozilla Foundation, version 2.
* The [Patron License](https://github.com/notgull/piet-hardware/blob/main/LICENSE-PATRON.md) for [sponsors](https://github.com/sponsors/notgull) and [contributors](https://github.com/notgull/async-winit/graphs/contributors), who can ignore the copyleft provisions of the GNU AGPL for this project.

`piet-hardware` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License or the Mozilla Public License for more details.

You should have received a copy of the GNU Lesser General Public License and the Mozilla
Public License along with `piet-hardware`. If not, see <https://www.gnu.org/licenses/> or
<https://www.mozilla.org/en-US/MPL/2.0/>.
