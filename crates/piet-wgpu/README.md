# piet-wgpu

A GPU-acclerated backend for the [`piet`] API that uses the [`wgpu`] crate.

Given a [`wgpu`] `Device`, `Queue`, and `TextureView` to render to, it can effectively draw vector graphics to the desired target.

[`piet`]: https://crates.io/crates/piet
[`wgpu`]: https://crates.io/crates/wgpu

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
