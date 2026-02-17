//! Voxel Grid — 3D spatial structure for neuron neighborhoods.
//!
//! The grid divides space into voxels, each containing a contiguous range of
//! neurons in the flat neuron array. This enables O(1) neighbor lookup for
//! wiring, migration, and spatial queries.
//!
//! ## Layout
//!
//! ```text
//! VoxelGrid (dims.x × dims.y × dims.z)
//!   └── VoxelNeighborhood per occupied voxel
//!         └── neuron index range [start..end) in flat array
//! ```

use std::collections::HashMap;
use super::neuron::UnifiedNeuron;

/// A neighborhood within a single voxel — stores a neuron index range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelNeighborhood {
    /// Start index (inclusive) in the flat neuron array.
    pub start: u32,
    /// End index (exclusive) in the flat neuron array.
    pub end: u32,
}

impl VoxelNeighborhood {
    /// Number of neurons in this neighborhood.
    #[inline]
    pub fn count(&self) -> u32 {
        self.end - self.start
    }

    /// Is this neighborhood empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Iterate neuron indices.
    #[inline]
    pub fn indices(&self) -> std::ops::Range<u32> {
        self.start..self.end
    }
}

/// Grid dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridDims {
    pub x: u16,
    pub y: u16,
    pub z: u16,
}

impl GridDims {
    /// Total number of voxels.
    #[inline]
    pub fn total(&self) -> usize {
        self.x as usize * self.y as usize * self.z as usize
    }

    /// Linear index from 3D coordinates. Returns None if out of bounds.
    #[inline]
    pub fn linear_index(&self, x: u16, y: u16, z: u16) -> Option<usize> {
        if x < self.x && y < self.y && z < self.z {
            Some(x as usize + y as usize * self.x as usize + z as usize * self.x as usize * self.y as usize)
        } else {
            None
        }
    }

    /// 3D coordinates from linear index.
    #[inline]
    pub fn coords(&self, linear: usize) -> (u16, u16, u16) {
        let xy = self.x as usize * self.y as usize;
        let z = (linear / xy) as u16;
        let rem = linear % xy;
        let y = (rem / self.x as usize) as u16;
        let x = (rem % self.x as usize) as u16;
        (x, y, z)
    }
}

/// 3D voxel grid providing spatial indexing of neurons.
///
/// Neurons are sorted by voxel position and stored in a flat array.
/// Each voxel maps to a contiguous range in that array.
pub struct VoxelGrid {
    /// Grid dimensions.
    pub dims: GridDims,
    /// Per-voxel neighborhood (neuron index range). Indexed by linear voxel index.
    neighborhoods: Vec<VoxelNeighborhood>,
}

impl VoxelGrid {
    /// Build a grid from neurons. Sorts neurons by voxel and builds the index.
    ///
    /// The `neurons` slice is reordered in-place (sorted by voxel position).
    /// After this call, neurons within the same voxel are contiguous.
    pub fn build(neurons: &mut [UnifiedNeuron]) -> Self {
        // Determine grid bounds from neuron positions
        let (max_x, max_y, max_z) = neurons.iter().fold((0u16, 0u16, 0u16), |(mx, my, mz), n| {
            (mx.max(n.position.voxel.0), my.max(n.position.voxel.1), mz.max(n.position.voxel.2))
        });

        let dims = GridDims {
            x: max_x + 1,
            y: max_y + 1,
            z: max_z + 1,
        };

        Self::build_with_dims(neurons, dims)
    }

    /// Build a grid with explicit dimensions. Neurons outside the grid are clamped.
    pub fn build_with_dims(neurons: &mut [UnifiedNeuron], dims: GridDims) -> Self {
        // Sort neurons by voxel (z-major, then y, then x)
        neurons.sort_by(|a, b| {
            let av = a.position.voxel;
            let bv = b.position.voxel;
            (av.2, av.1, av.0).cmp(&(bv.2, bv.1, bv.0))
        });

        // Build neighborhoods by scanning sorted neurons
        let total = dims.total();
        let mut neighborhoods = vec![VoxelNeighborhood { start: 0, end: 0 }; total];

        // Count neurons per voxel
        let mut counts: HashMap<usize, u32> = HashMap::new();
        for n in neurons.iter() {
            if let Some(idx) = dims.linear_index(n.position.voxel.0, n.position.voxel.1, n.position.voxel.2) {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        // Compute offsets (prefix sum)
        let mut offset = 0u32;
        for i in 0..total {
            let count = counts.get(&i).copied().unwrap_or(0);
            neighborhoods[i] = VoxelNeighborhood {
                start: offset,
                end: offset + count,
            };
            offset += count;
        }

        Self { dims, neighborhoods }
    }

    /// Get the neighborhood for a voxel position.
    pub fn neighborhood(&self, x: u16, y: u16, z: u16) -> Option<&VoxelNeighborhood> {
        self.dims.linear_index(x, y, z).map(|i| &self.neighborhoods[i])
    }

    /// Get neuron indices for a specific voxel.
    pub fn neuron_indices(&self, x: u16, y: u16, z: u16) -> std::ops::Range<u32> {
        match self.neighborhood(x, y, z) {
            Some(n) => n.indices(),
            None => 0..0,
        }
    }

    /// Enumerate 6-connected neighbor voxels (face-adjacent).
    pub fn neighbors_6(&self, x: u16, y: u16, z: u16) -> Vec<(u16, u16, u16)> {
        let mut result = Vec::with_capacity(6);
        let offsets: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];
        for (dx, dy, dz) in offsets {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx >= 0 && nx < self.dims.x as i32
                && ny >= 0 && ny < self.dims.y as i32
                && nz >= 0 && nz < self.dims.z as i32
            {
                result.push((nx as u16, ny as u16, nz as u16));
            }
        }
        result
    }

    /// Enumerate 26-connected neighbor voxels (face + edge + corner adjacent).
    pub fn neighbors_26(&self, x: u16, y: u16, z: u16) -> Vec<(u16, u16, u16)> {
        let mut result = Vec::with_capacity(26);
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx >= 0 && nx < self.dims.x as i32
                        && ny >= 0 && ny < self.dims.y as i32
                        && nz >= 0 && nz < self.dims.z as i32
                    {
                        result.push((nx as u16, ny as u16, nz as u16));
                    }
                }
            }
        }
        result
    }

    /// Collect all neuron indices from a voxel and its 26-connected neighbors.
    ///
    /// Useful for proximity wiring — gives the local candidate set without
    /// scanning the entire neuron array.
    pub fn local_neuron_indices(&self, x: u16, y: u16, z: u16) -> Vec<u32> {
        let mut indices = Vec::new();

        // Self
        if let Some(n) = self.neighborhood(x, y, z) {
            indices.extend(n.indices());
        }

        // 26-connected neighbors
        for (nx, ny, nz) in self.neighbors_26(x, y, z) {
            if let Some(n) = self.neighborhood(nx, ny, nz) {
                indices.extend(n.indices());
            }
        }

        indices
    }

    /// Total neurons indexed by the grid.
    pub fn total_neurons(&self) -> u32 {
        self.neighborhoods.iter().map(|n| n.count()).sum()
    }

    /// Number of occupied voxels (with at least one neuron).
    pub fn occupied_voxels(&self) -> usize {
        self.neighborhoods.iter().filter(|n| !n.is_empty()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::neuron::VoxelPosition;

    fn pos(x: u16, y: u16, z: u16) -> VoxelPosition {
        VoxelPosition::at_center((x, y, z))
    }

    #[test]
    fn grid_dims_linear() {
        let dims = GridDims { x: 4, y: 3, z: 2 };
        assert_eq!(dims.total(), 24);
        assert_eq!(dims.linear_index(0, 0, 0), Some(0));
        assert_eq!(dims.linear_index(3, 2, 1), Some(23));
        assert_eq!(dims.linear_index(4, 0, 0), None); // out of bounds
    }

    #[test]
    fn grid_dims_round_trip() {
        let dims = GridDims { x: 4, y: 3, z: 2 };
        for i in 0..dims.total() {
            let (x, y, z) = dims.coords(i);
            assert_eq!(dims.linear_index(x, y, z), Some(i));
        }
    }

    #[test]
    fn build_grid_from_neurons() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(1, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(1, 1, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);

        assert_eq!(grid.dims, GridDims { x: 2, y: 2, z: 1 });
        assert_eq!(grid.total_neurons(), 4);
        assert_eq!(grid.occupied_voxels(), 3);

        // Voxel (0,0,0) has 2 neurons
        let n = grid.neighborhood(0, 0, 0).unwrap();
        assert_eq!(n.count(), 2);

        // Voxel (1,0,0) has 1 neuron
        let n = grid.neighborhood(1, 0, 0).unwrap();
        assert_eq!(n.count(), 1);
    }

    #[test]
    fn neighbors_6_center() {
        let mut neurons = vec![UnifiedNeuron::pyramidal_at(pos(1, 1, 1))];
        let grid = VoxelGrid::build_with_dims(&mut neurons, GridDims { x: 3, y: 3, z: 3 });

        let n6 = grid.neighbors_6(1, 1, 1);
        assert_eq!(n6.len(), 6);
    }

    #[test]
    fn neighbors_6_corner() {
        let mut neurons = vec![UnifiedNeuron::pyramidal_at(pos(0, 0, 0))];
        let grid = VoxelGrid::build_with_dims(&mut neurons, GridDims { x: 3, y: 3, z: 3 });

        let n6 = grid.neighbors_6(0, 0, 0);
        assert_eq!(n6.len(), 3); // only +x, +y, +z are valid
    }

    #[test]
    fn neighbors_26_center() {
        let mut neurons = vec![UnifiedNeuron::pyramidal_at(pos(1, 1, 1))];
        let grid = VoxelGrid::build_with_dims(&mut neurons, GridDims { x: 3, y: 3, z: 3 });

        let n26 = grid.neighbors_26(1, 1, 1);
        assert_eq!(n26.len(), 26);
    }

    #[test]
    fn neighbors_26_corner() {
        let mut neurons = vec![UnifiedNeuron::pyramidal_at(pos(0, 0, 0))];
        let grid = VoxelGrid::build_with_dims(&mut neurons, GridDims { x: 3, y: 3, z: 3 });

        let n26 = grid.neighbors_26(0, 0, 0);
        assert_eq!(n26.len(), 7); // 2^3 - 1 = 7 valid neighbors at corner
    }

    #[test]
    fn local_neuron_indices() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(1, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 1, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);

        // Local indices for (0,0,0) should include self + adjacent voxels
        let local = grid.local_neuron_indices(0, 0, 0);
        assert_eq!(local.len(), 4); // all 4 neurons are within 1 voxel of (0,0,0)
    }

    #[test]
    fn empty_grid() {
        let mut neurons: Vec<UnifiedNeuron> = Vec::new();
        let grid = VoxelGrid::build_with_dims(&mut neurons, GridDims { x: 2, y: 2, z: 2 });

        assert_eq!(grid.total_neurons(), 0);
        assert_eq!(grid.occupied_voxels(), 0);
        assert_eq!(grid.neuron_indices(0, 0, 0), 0..0);
    }

    #[test]
    fn neurons_sorted_after_build() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(2, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(1, 0, 0)),
        ];

        let _grid = VoxelGrid::build(&mut neurons);

        // After build, neurons should be sorted by voxel position
        assert_eq!(neurons[0].position.voxel.0, 0);
        assert_eq!(neurons[1].position.voxel.0, 1);
        assert_eq!(neurons[2].position.voxel.0, 2);
    }
}
