# piet-gpu

`piet-gpu` is a strategy for implementing the [`piet`] drawing interface using GPU primitives. The goal is to break down the drawing operations to rendering textured triangles. The resulting buffers are than passed to the GPU backend for rendering.

As `piet-gpu` simply implements the high-level strategy, it has no unsafe code. The actual GPU calls are forwarded to an object that implements `GpuContext`. This object is intended to be an interface to OpenGL, Vulkan, Metal, or other GPU APIs.

[`piet`]: https://crates.io/crates/piet

## License

`piet-gpu` is free software: you can redistribute it and/or modify it under the terms of
either:

* GNU Lesser General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.
* Mozilla Public License as published by the Mozilla Foundation, version 2.

`piet-gpu` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License or the Mozilla Public License for more details.

You should have received a copy of the GNU Lesser General Public License and the Mozilla
Public License along with `piet-gpu`. If not, see <https://www.gnu.org/licenses/> or
<https://www.mozilla.org/en-US/MPL/2.0/>.
