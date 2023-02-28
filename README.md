# piet-gpu

A set of implementations of [`piet`], Rust's 2D vector graphics library, using GPU primitives. The goal is to provide fast and high quality graphics rendering using a familiar API.

The [`piet-gpu`] crate is the centerpiece of this project. It translates the [`piet`] API calls down to rendering textured triangles. In turn, it sends these rendering calls to a structure implementing the `GpuContext` interface. This trait represents the lower level hardware calls.

In addition to [`piet-gpu`], this project also contains some crates that implement the [`piet`] API using [`piet-gpu`]. These are:

- [`piet-glow`], an implementation of [`piet`] using the [`glow`] crate for OpenGL and WebGL calls.

Crates that are planned in the future:

- `piet-wgpu`, an implementation of [`piet`] using the [`wgpu`] crate.
- `piet-vulkano`, an implementation of [`piet`] using the [`vulkano`] crate.
- `piet-metal`, an implementation of [`piet`] using the Metal API.

[`piet`]: https://crates.io/crates/piet
[`piet-gpu`]: ./crates/piet-gpu/README.md
[`piet-glow`]: ./crates/piet-glow/README.md
[`glow`]: https://crates.io/crates/glow
[`wgpu`]: https://crates.io/crates/wgpu
[`vulkano`]: https://crates.io/crates/vulkano

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
