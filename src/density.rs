#![allow(deprecated)]
//! Density Field — grid-based soma density for spatial queries.
//!
//! Provides O(1) density lookups with trilinear interpolation.
//! Used for axon growth resistance and region boundary detection.
//!
//! ## Tissue Types (Gray vs White Matter)
//!
//! - **Gray matter**: High soma density (>threshold), local processing
//! - **White matter**: Low soma density (<threshold), axon corridors
//!
//! Axons traverse white matter easily but resist entering gray matter,
//! creating natural boundaries between processing regions.

/// Tissue type classification based on density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TissueType {
    /// High density - local processing, resists axon traversal
    GrayMatter,
    /// Low density - axon corridors, easy traversal
    WhiteMatter,
    /// Edge region - boundary between gray and white
    Boundary,
}

/// Conductivity profile for signal propagation.
#[derive(Debug, Clone, Copy)]
pub struct Conductivity {
    /// Base resistance (0.0 = perfect conductor, 1.0 = insulator)
    pub resistance: f32,
    /// Signal attenuation per unit distance
    pub attenuation: f32,
    /// Delay multiplier (1.0 = normal, >1 = slower)
    pub delay_factor: f32,
}

/// Grid-based density field for fast spatial queries.
///
/// Stores soma density at each voxel in a 3D grid. Density is computed
/// from neuron soma positions and updated periodically.
#[derive(Clone)]
pub struct DensityField {
    /// Flattened 3D grid of density values.
    /// Index: x + y * res_x + z * res_x * res_y
    grid: Vec<f32>,
    /// Grid resolution (voxels per dimension).
    resolution: [usize; 3],
    /// Physical bounds of the space.
    bounds: [f32; 3],
    /// Voxel size in each dimension.
    voxel_size: [f32; 3],
}

impl DensityField {
    /// Create a new density field with given resolution and bounds.
    pub fn new(resolution: [usize; 3], bounds: [f32; 3]) -> Self {
        let total = resolution[0] * resolution[1] * resolution[2];
        let voxel_size = [
            bounds[0] / resolution[0] as f32,
            bounds[1] / resolution[1] as f32,
            bounds[2] / resolution[2] as f32,
        ];

        Self {
            grid: vec![0.0; total],
            resolution,
            bounds,
            voxel_size,
        }
    }

    /// Default density field for a given bounds (8x8x8 resolution).
    pub fn default_for_bounds(bounds: [f32; 3]) -> Self {
        Self::new([8, 8, 8], bounds)
    }

    /// Resolution of the grid.
    pub fn resolution(&self) -> [usize; 3] {
        self.resolution
    }

    /// Physical bounds of the space.
    pub fn bounds(&self) -> [f32; 3] {
        self.bounds
    }

    /// Convert position to grid indices, clamped to valid range.
    fn pos_to_indices(&self, pos: [f32; 3]) -> (usize, usize, usize) {
        let ix = ((pos[0] / self.voxel_size[0]).floor() as isize)
            .max(0)
            .min(self.resolution[0] as isize - 1) as usize;
        let iy = ((pos[1] / self.voxel_size[1]).floor() as isize)
            .max(0)
            .min(self.resolution[1] as isize - 1) as usize;
        let iz = ((pos[2] / self.voxel_size[2]).floor() as isize)
            .max(0)
            .min(self.resolution[2] as isize - 1) as usize;
        (ix, iy, iz)
    }

    /// Convert grid indices to flat array index.
    #[inline]
    fn flat_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + iy * self.resolution[0] + iz * self.resolution[0] * self.resolution[1]
    }

    /// Get density at grid position (no interpolation).
    pub fn density_at_voxel(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        if ix >= self.resolution[0] || iy >= self.resolution[1] || iz >= self.resolution[2] {
            return 0.0;
        }
        self.grid[self.flat_index(ix, iy, iz)]
    }

    /// Get density at position using trilinear interpolation.
    pub fn density_at(&self, pos: [f32; 3]) -> f32 {
        // Fractional position within voxels
        let fx = pos[0] / self.voxel_size[0];
        let fy = pos[1] / self.voxel_size[1];
        let fz = pos[2] / self.voxel_size[2];

        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let z0 = fz.floor() as isize;

        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        // Interpolation weights
        let tx = fx - fx.floor();
        let ty = fy - fy.floor();
        let tz = fz - fz.floor();

        // Clamp indices
        let clamp = |v: isize, max: usize| v.max(0).min(max as isize - 1) as usize;
        let x0c = clamp(x0, self.resolution[0]);
        let x1c = clamp(x1, self.resolution[0]);
        let y0c = clamp(y0, self.resolution[1]);
        let y1c = clamp(y1, self.resolution[1]);
        let z0c = clamp(z0, self.resolution[2]);
        let z1c = clamp(z1, self.resolution[2]);

        // Sample 8 corners
        let c000 = self.density_at_voxel(x0c, y0c, z0c);
        let c001 = self.density_at_voxel(x0c, y0c, z1c);
        let c010 = self.density_at_voxel(x0c, y1c, z0c);
        let c011 = self.density_at_voxel(x0c, y1c, z1c);
        let c100 = self.density_at_voxel(x1c, y0c, z0c);
        let c101 = self.density_at_voxel(x1c, y0c, z1c);
        let c110 = self.density_at_voxel(x1c, y1c, z0c);
        let c111 = self.density_at_voxel(x1c, y1c, z1c);

        // Trilinear interpolation
        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    /// Update density from neuron soma positions.
    ///
    /// Each neuron contributes 1.0 to its containing voxel.
    pub fn update_from_positions(&mut self, positions: &[[f32; 3]]) {
        // Clear grid
        for v in &mut self.grid {
            *v = 0.0;
        }

        // Accumulate density
        for pos in positions {
            let (ix, iy, iz) = self.pos_to_indices(*pos);
            let idx = self.flat_index(ix, iy, iz);
            self.grid[idx] += 1.0;
        }

        // Normalize by voxel volume
        let voxel_vol = self.voxel_size[0] * self.voxel_size[1] * self.voxel_size[2];
        if voxel_vol > 0.0 {
            for v in &mut self.grid {
                *v /= voxel_vol;
            }
        }
    }

    /// Find a low-density position below threshold.
    ///
    /// Returns center of first voxel with density < threshold, or None.
    pub fn find_low_density(&self, threshold: f32) -> Option<[f32; 3]> {
        for iz in 0..self.resolution[2] {
            for iy in 0..self.resolution[1] {
                for ix in 0..self.resolution[0] {
                    let idx = self.flat_index(ix, iy, iz);
                    if self.grid[idx] < threshold {
                        // Return voxel center
                        return Some([
                            (ix as f32 + 0.5) * self.voxel_size[0],
                            (iy as f32 + 0.5) * self.voxel_size[1],
                            (iz as f32 + 0.5) * self.voxel_size[2],
                        ]);
                    }
                }
            }
        }
        None
    }

    /// Find the position with lowest density.
    pub fn find_lowest_density(&self) -> [f32; 3] {
        let mut min_idx = 0;
        let mut min_val = f32::MAX;

        for (idx, &val) in self.grid.iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = idx;
            }
        }

        // Convert flat index back to position
        let iz = min_idx / (self.resolution[0] * self.resolution[1]);
        let iy = (min_idx / self.resolution[0]) % self.resolution[1];
        let ix = min_idx % self.resolution[0];

        [
            (ix as f32 + 0.5) * self.voxel_size[0],
            (iy as f32 + 0.5) * self.voxel_size[1],
            (iz as f32 + 0.5) * self.voxel_size[2],
        ]
    }

    /// Average density across the entire field.
    pub fn average_density(&self) -> f32 {
        let sum: f32 = self.grid.iter().sum();
        sum / self.grid.len() as f32
    }

    /// Maximum density in the field.
    pub fn max_density(&self) -> f32 {
        self.grid.iter().cloned().fold(0.0f32, f32::max)
    }

    /// Classify tissue type at position based on density threshold.
    ///
    /// - density > high_threshold → GrayMatter
    /// - density < low_threshold → WhiteMatter
    /// - otherwise → Boundary
    pub fn tissue_type_at(&self, pos: [f32; 3], gray_threshold: f32) -> TissueType {
        let density = self.density_at(pos);
        let boundary_margin = gray_threshold * 0.2; // 20% margin for boundary

        if density > gray_threshold {
            TissueType::GrayMatter
        } else if density < gray_threshold - boundary_margin {
            TissueType::WhiteMatter
        } else {
            TissueType::Boundary
        }
    }

    /// Calculate conductivity at position.
    ///
    /// Higher density = higher resistance (harder for signals to propagate).
    /// White matter has low resistance (fast conduction).
    /// Gray matter has high resistance (local processing, not long-range).
    pub fn conductivity_at(&self, pos: [f32; 3]) -> Conductivity {
        let density = self.density_at(pos);
        let max_density = self.max_density().max(1.0);
        let normalized = density / max_density;

        Conductivity {
            // Resistance scales with density
            resistance: normalized * 0.8, // 0.0-0.8 range
            // Attenuation in gray matter
            attenuation: 0.1 + normalized * 0.3, // 0.1-0.4 per unit
            // Delay increases in gray matter
            delay_factor: 1.0 + normalized * 2.0, // 1.0-3.0x
        }
    }

    /// Find corridor path from start to end through low-density regions.
    ///
    /// Returns a sequence of positions forming a path that minimizes
    /// total resistance. Uses gradient descent toward target.
    pub fn find_corridor(&self, start: [f32; 3], end: [f32; 3], steps: usize) -> Vec<[f32; 3]> {
        let mut path = Vec::with_capacity(steps + 1);
        path.push(start);

        let mut current = start;
        let step_size = 0.5; // Fraction of voxel size

        for _ in 0..steps {
            // Direction toward target
            let dx = end[0] - current[0];
            let dy = end[1] - current[1];
            let dz = end[2] - current[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < self.voxel_size[0] * 0.5 {
                break; // Close enough
            }

            // Normalize direction
            let dir = [dx / dist, dy / dist, dz / dist];

            // Also consider density gradient (prefer lower density)
            let grad = self.density_gradient_at(current);

            // Combine: move toward target but bias toward lower density
            let bias = 0.3; // How much to favor low density
            let next = [
                current[0] + (dir[0] - grad[0] * bias) * step_size * self.voxel_size[0],
                current[1] + (dir[1] - grad[1] * bias) * step_size * self.voxel_size[1],
                current[2] + (dir[2] - grad[2] * bias) * step_size * self.voxel_size[2],
            ];

            // Clamp to bounds
            let next = [
                next[0].max(0.0).min(self.bounds[0]),
                next[1].max(0.0).min(self.bounds[1]),
                next[2].max(0.0).min(self.bounds[2]),
            ];

            path.push(next);
            current = next;
        }

        path.push(end);
        path
    }

    /// Calculate density gradient at position (direction of increasing density).
    ///
    /// Returns normalized direction vector pointing toward higher density.
    /// Negate to get direction toward lower density.
    pub fn density_gradient_at(&self, pos: [f32; 3]) -> [f32; 3] {
        let eps = self.voxel_size[0] * 0.1;

        // Central differences
        let dx = self.density_at([pos[0] + eps, pos[1], pos[2]])
            - self.density_at([pos[0] - eps, pos[1], pos[2]]);
        let dy = self.density_at([pos[0], pos[1] + eps, pos[2]])
            - self.density_at([pos[0], pos[1] - eps, pos[2]]);
        let dz = self.density_at([pos[0], pos[1], pos[2] + eps])
            - self.density_at([pos[0], pos[1], pos[2] - eps]);

        // Normalize
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if mag > 1e-6 {
            [dx / mag, dy / mag, dz / mag]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Find neurons within a spatial neighborhood.
    ///
    /// Returns indices into the positions array for neurons within radius of center.
    pub fn neurons_in_radius(
        &self,
        positions: &[[f32; 3]],
        center: [f32; 3],
        radius: f32,
    ) -> Vec<usize> {
        let radius_sq = radius * radius;
        positions
            .iter()
            .enumerate()
            .filter(|(_, pos)| {
                let dx = pos[0] - center[0];
                let dy = pos[1] - center[1];
                let dz = pos[2] - center[2];
                dx * dx + dy * dy + dz * dz <= radius_sq
            })
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_field_creation() {
        let field = DensityField::new([4, 4, 4], [4.0, 4.0, 4.0]);
        assert_eq!(field.resolution(), [4, 4, 4]);
        assert_eq!(field.bounds(), [4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_density_update() {
        let mut field = DensityField::new([2, 2, 2], [2.0, 2.0, 2.0]);

        // Place two neurons in the same voxel
        let positions = vec![
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ];
        field.update_from_positions(&positions);

        // Voxel (0,0,0) should have density > 0
        assert!(field.density_at_voxel(0, 0, 0) > 0.0);
    }

    #[test]
    fn test_find_low_density() {
        let mut field = DensityField::new([2, 2, 2], [2.0, 2.0, 2.0]);

        // Place neurons in one corner only
        let positions = vec![
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
        ];
        field.update_from_positions(&positions);

        // Should find a low-density position in the other half
        let low_pos = field.find_low_density(0.1);
        assert!(low_pos.is_some());
    }
}
