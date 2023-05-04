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

//! A wrapper around the WGPU buffers.

use super::{DeviceAndQueue, GpuContext};
use piet_hardware::Vertex;

use std::cell::{Ref, RefCell, RefMut};
use std::mem;
use std::rc::Rc;

/// The resource representing a WGPU buffer.
#[derive(Clone)]
pub(crate) struct WgpuVertexBuffer(Rc<BufferInner>);

impl WgpuVertexBuffer {
    /// Create a new vertex buffer.
    pub(crate) fn new<DaQ: DeviceAndQueue + ?Sized>(base: &GpuContext<DaQ>) -> Self {
        const INITIAL_VERTEX_BUFFER_SIZE: usize = 1024 * mem::size_of::<Vertex>();
        const INITIAL_INDEX_BUFFER_SIZE: usize = 1024 * mem::size_of::<u32>();

        let vertex_buffer = Buffer::new(
            base,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            INITIAL_VERTEX_BUFFER_SIZE,
            "vertex",
        );
        let index_buffer = Buffer::new(
            base,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            INITIAL_INDEX_BUFFER_SIZE,
            "index",
        );

        WgpuVertexBuffer(Rc::new(BufferInner {
            id: base.next_id(),
            vertex_buffer: RefCell::new(vertex_buffer),
            index_buffer: RefCell::new(index_buffer),
        }))
    }

    /// Get the ID of this buffer set.
    #[inline]
    pub(crate) fn id(&self) -> usize {
        self.0.id
    }

    /// Borrow the vertex buffer.
    pub(crate) fn borrow_vertex_buffer(&self) -> Ref<'_, Buffer> {
        self.0.vertex_buffer.borrow()
    }

    /// Borrow the index buffer.
    pub(crate) fn borrow_index_buffer(&self) -> Ref<'_, Buffer> {
        self.0.index_buffer.borrow()
    }

    /// Borrow the vertex buffer, mutably.
    pub(crate) fn borrow_vertex_buffer_mut(&self) -> RefMut<'_, Buffer> {
        self.0.vertex_buffer.borrow_mut()
    }

    /// Borrow the index buffer, mutably.
    pub(crate) fn borrow_index_buffer_mut(&self) -> RefMut<'_, Buffer> {
        self.0.index_buffer.borrow_mut()
    }
}

/// Inner data for a buffer.
struct BufferInner {
    /// Unique ID.
    id: usize,

    /// The buffer for vertices.
    vertex_buffer: RefCell<Buffer>,

    /// The buffer for indices.
    index_buffer: RefCell<Buffer>,
}

/// Describes the data for a buffer.
pub(crate) struct Buffer {
    /// The index of the inner WGPU buffer.
    id: usize,

    /// The capacity of the buffer.
    ///
    /// This is the total number of bytes that can be held by `buffer`.
    capacity: usize,

    /// The capacity of the last buffer.
    ///
    /// This is used to determine when to allocate a new buffer.
    last_capacity: usize,

    /// The starting cursor for the buffer.
    ///
    /// This is the start of the current slice and where new writes will begin. It is into the last
    /// buffer.
    start_cursor: usize,

    /// The ending cursor for the buffer.
    ///
    /// This determines the end of the current slice and where new writes will end. It is into the
    /// last buffer.
    end_cursor: usize,

    /// The buffer usages.
    usage: wgpu::BufferUsages,

    /// The identifier for the buffer.
    buffer_id: &'static str,

    /// The inner WGPU buffer.
    buffer: BufferCollection,
}

/// Either a single buffer or a list of them.
///
/// This is used to dynamically reallocate new buffers during rendering.
enum BufferCollection {
    /// A single buffer.
    Single(wgpu::Buffer),

    /// A list of buffers.
    ///
    /// This is only used when the single buffer overflows.
    List(Vec<wgpu::Buffer>),

    /// Empty hole.
    Hole,
}

impl BufferCollection {
    /// Get the buffer at the given index.
    fn get(&self, i: usize) -> Option<&wgpu::Buffer> {
        match (self, i) {
            (BufferCollection::Single(buffer), 0) => Some(buffer),
            (BufferCollection::List(buffers), i) => buffers.get(i),
            _ => None,
        }
    }

    /// Get the last buffer.
    fn last(&self) -> Option<&wgpu::Buffer> {
        match self {
            BufferCollection::Single(buffer) => Some(buffer),
            BufferCollection::List(buffers) => buffers.last(),
            _ => None,
        }
    }

    /// Get the last buffer, mutably.
    fn last_mut(&mut self) -> Option<&mut wgpu::Buffer> {
        match self {
            BufferCollection::Single(buffer) => Some(buffer),
            BufferCollection::List(buffers) => buffers.last_mut(),
            _ => None,
        }
    }

    /// Push a new buffer.
    fn push(&mut self, buffer: wgpu::Buffer) {
        match mem::replace(self, Self::Hole) {
            Self::Hole => *self = Self::Single(buffer),
            Self::Single(old_buffer) => {
                tracing::debug!("using list-based buffering strategy");
                *self = Self::List(vec![old_buffer, buffer])
            }
            Self::List(mut buffers) => {
                buffers.push(buffer);
                *self = Self::List(buffers);
            }
        }
    }

    /// Get the length of the buffer.
    fn len(&self) -> usize {
        match self {
            BufferCollection::Single(_) => 1,
            BufferCollection::List(buffers) => buffers.len(),
            _ => 0,
        }
    }

    /// Get a slice of the buffer.
    pub(crate) fn slice(
        &self,
        slice: BufferSlice,
        granularity: u64,
    ) -> Option<wgpu::BufferSlice<'_>> {
        let map_end = |end| end / granularity;
        let new_range = map_end(slice.range.0)..map_end(slice.range.1);

        self.get(slice.buffer_index).map(|buf| buf.slice(new_range))
    }
}

/// A slice out of the `BufferCollection`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BufferSlice {
    /// The index of the buffer.
    buffer_index: usize,

    /// The range of the slice.
    range: (u64, u64),
}

impl BufferSlice {
    /// Get the length of this slice.
    pub(crate) fn len(&self) -> usize {
        (self.range.1 - self.range.0).try_into().unwrap()
    }
}

impl Buffer {
    /// Create a new buffer.
    fn create_buffer(&self, dev: &wgpu::Device, len: usize) -> wgpu::Buffer {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("piet-wgpu {} buffer {}", self.buffer_id, self.id)),
            usage: self.usage,
            size: len.try_into().expect("buffer too large"),
            mapped_at_creation: false,
        })
    }

    /// Write this data into the buffer.
    pub(crate) fn write_buffer<DaQ: DeviceAndQueue + ?Sized>(
        &mut self,
        base: &GpuContext<DaQ>,
        data: &[u8],
    ) {
        // See if we need to allocate a new buffer.
        let remaining_capacity = self.last_capacity - self.end_cursor;
        if remaining_capacity < data.len() {
            // Round the desired length up to the nearest multiple of 2 to prevent frequent reallocs.
            let new_capacity = data
                .len()
                .checked_add(remaining_capacity)
                .map(|len| len.next_power_of_two())
                .expect("buffer too large");
            let new_buffer = self.create_buffer(base.device_and_queue().device(), new_capacity);

            // If we haven't sliced out this buffer yet, just reallocate in place.
            if self.start_cursor == 0 {
                *self.buffer.last_mut().unwrap() = new_buffer;
                self.capacity -= self.last_capacity;
            } else {
                // Push the buffer to the end.
                self.buffer.push(new_buffer);
                self.start_cursor = 0;
                self.end_cursor = 0;
            }

            self.last_capacity = new_capacity;
            self.capacity += new_capacity;
        }

        // Queue the write to the buffer.
        base.device_and_queue().queue().write_buffer(
            self.buffer.last().unwrap(),
            self.start_cursor.try_into().expect("buffer too large"),
            data,
        );

        // Update the cursor.
        self.end_cursor = self.start_cursor + data.len();
        tracing::debug!(
            "Wrote to {} buffer from {} to {}",
            self.buffer_id,
            self.start_cursor,
            self.end_cursor
        );
    }

    /// Pop off a slice of the buffer.
    pub(crate) fn pop_slice(&mut self) -> BufferSlice {
        let slice = BufferSlice {
            buffer_index: self.buffer.len() - 1,
            range: (self.start_cursor as u64, self.end_cursor as u64),
        };

        tracing::debug!(slice=?slice, "Popped {} buffer slice", self.buffer_id);

        // Update the cursor.
        self.start_cursor = self.end_cursor;

        slice
    }

    /// Empty out the buffer.
    pub(crate) fn clear(&mut self, device: &wgpu::Device) {
        // Reset the cursor.
        self.start_cursor = 0;
        self.end_cursor = 0;

        // If we are using multiple buffers, combine them all into one.
        if matches!(self.buffer, BufferCollection::List(..)) {
            let desired_capacity = self.capacity.next_power_of_two();
            tracing::debug!("Resizing {} buffer to {}", self.buffer_id, desired_capacity);
            let new_buffer = self.create_buffer(device, desired_capacity);
            self.buffer = BufferCollection::Single(new_buffer);
            self.capacity = desired_capacity;
            self.last_capacity = desired_capacity;
        }
    }

    /// Create a new buffer.
    fn new<DaQ: DeviceAndQueue + ?Sized>(
        base: &GpuContext<DaQ>,
        usage: wgpu::BufferUsages,
        starting_size: usize,
        buffer_id: &'static str,
    ) -> Self {
        let starting_size = starting_size.next_power_of_two();
        let mut this = Self {
            id: base.next_id(),
            capacity: starting_size,
            last_capacity: starting_size,
            start_cursor: 0,
            end_cursor: 0,
            buffer_id,
            usage,
            buffer: BufferCollection::Hole,
        };

        this.buffer = BufferCollection::Single(
            this.create_buffer(base.device_and_queue().device(), starting_size),
        );
        this
    }

    /// Get a slice of this buffer.
    pub(crate) fn slice(&self, buffer_slice: BufferSlice) -> Option<wgpu::BufferSlice<'_>> {
        self.buffer.slice(buffer_slice, 1)
    }
}
