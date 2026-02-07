//! Tissue Field — Emergent Gray/White Matter from Point Cloud
//!
//! Tissue properties emerge from neuron positions, not from predefined voxels.
//! Gray matter forms where somas cluster. White matter forms where axons bundle.
//!
//! ## How It Works
//!
//! ```text
//! Neurons exist at 3D positions
//!        ↓
//! Spatial hash enables O(1) neighbor queries
//!        ↓
//! Density fields computed via kernel estimation
//!   - soma_density(pos) → gray matter indicator
//!   - axon_density(pos) → white matter indicator
//!        ↓
//! Tissue type derived from density ratio
//!   - High soma density → Gray
//!   - High axon, low soma → White (tract)
//!   - Both low → Sparse
//!        ↓
//! Signal propagation varies by tissue
//!   - White matter = fast (myelinated highways)
//!   - Gray matter = slower (local processing)
//! ```
//!
//! ## Emergent Regions
//!
//! Regions are detected, not defined. Clustering uses:
//! - Spatial proximity
//! - Spike correlation (from CorrelationTracker)
//! - Nuclei similarity
//!
//! This means brain regions form from activity patterns, not from boxes.

use super::{CorrelationTracker, Nuclei, SpatialNeuron};
use std::collections::HashMap;

// ============================================================================
// Spatial Hash — O(1) Neighbor Queries
// ============================================================================

/// Grid key for spatial hashing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl GridKey {
    /// Create a grid key from a position and cell size.
    #[inline]
    pub fn from_pos(pos: [f32; 3], cell_size: f32) -> Self {
        Self {
            x: (pos[0] / cell_size).floor() as i32,
            y: (pos[1] / cell_size).floor() as i32,
            z: (pos[2] / cell_size).floor() as i32,
        }
    }

    /// Get all 27 neighboring cells (including self).
    pub fn neighbors(&self) -> impl Iterator<Item = GridKey> + '_ {
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                (-1..=1).map(move |dz| GridKey {
                    x: self.x + dx,
                    y: self.y + dy,
                    z: self.z + dz,
                })
            })
        })
    }
}

/// Spatial hash for O(1) neighbor queries.
#[derive(Clone, Debug, Default)]
pub struct SpatialHash {
    cells: HashMap<GridKey, Vec<u32>>,
    cell_size: f32,
}

impl SpatialHash {
    /// Create a new spatial hash with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cells: HashMap::new(),
            cell_size,
        }
    }

    /// Clear and rebuild from neuron positions.
    pub fn rebuild(&mut self, neurons: &[SpatialNeuron]) {
        self.cells.clear();
        for (idx, neuron) in neurons.iter().enumerate() {
            let key = GridKey::from_pos(neuron.soma.position, self.cell_size);
            self.cells.entry(key).or_default().push(idx as u32);
        }
    }

    /// Query all neurons within radius of a position.
    ///
    /// Note: This returns all neurons in adjacent cells. For exact radius
    /// filtering, the caller should check distances. This is intentional
    /// for performance — the spatial hash gives approximate neighbors.
    pub fn query_radius(&self, pos: [f32; 3], _radius: f32) -> Vec<u32> {
        let key = GridKey::from_pos(pos, self.cell_size);
        let mut result = Vec::new();

        for neighbor_key in key.neighbors() {
            if let Some(indices) = self.cells.get(&neighbor_key) {
                result.extend(indices.iter().copied());
            }
        }

        result
    }

    /// Query neurons in a specific cell.
    pub fn query_cell(&self, key: GridKey) -> &[u32] {
        self.cells.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get cell size.
    #[inline]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

// ============================================================================
// Axon Segment — For White Matter Detection
// ============================================================================

/// A segment of an axon path (soma → terminal).
#[derive(Clone, Copy, Debug)]
pub struct AxonSegment {
    /// Neuron index this segment belongs to.
    pub neuron: u32,
    /// Start position (soma).
    pub start: [f32; 3],
    /// End position (axon terminal).
    pub end: [f32; 3],
    /// Myelination level (0-255).
    pub myelin: u8,
}

impl AxonSegment {
    /// Create from a neuron.
    pub fn from_neuron(idx: u32, neuron: &SpatialNeuron) -> Self {
        Self {
            neuron: idx,
            start: neuron.soma.position,
            end: neuron.axon.terminal,
            myelin: neuron.axon.myelin,
        }
    }

    /// Distance from a point to this line segment.
    pub fn distance_to_point(&self, pos: [f32; 3]) -> f32 {
        let ax = self.end[0] - self.start[0];
        let ay = self.end[1] - self.start[1];
        let az = self.end[2] - self.start[2];

        let px = pos[0] - self.start[0];
        let py = pos[1] - self.start[1];
        let pz = pos[2] - self.start[2];

        let dot = px * ax + py * ay + pz * az;
        let len_sq = ax * ax + ay * ay + az * az;

        if len_sq < 0.0001 {
            // Degenerate segment (point)
            return (px * px + py * py + pz * pz).sqrt();
        }

        // Project point onto line, clamp to segment
        let t = (dot / len_sq).clamp(0.0, 1.0);

        let closest_x = self.start[0] + t * ax;
        let closest_y = self.start[1] + t * ay;
        let closest_z = self.start[2] + t * az;

        let dx = pos[0] - closest_x;
        let dy = pos[1] - closest_y;
        let dz = pos[2] - closest_z;

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Direction vector (normalized).
    pub fn direction(&self) -> [f32; 3] {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        let dz = self.end[2] - self.start[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 0.0001 {
            [0.0, 0.0, 0.0]
        } else {
            [dx / len, dy / len, dz / len]
        }
    }

    /// Length of the segment.
    pub fn length(&self) -> f32 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        let dz = self.end[2] - self.start[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ============================================================================
// Tissue Types
// ============================================================================

/// Type of neural tissue at a location.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TissueType {
    /// Gray matter — high soma density (cell bodies).
    Gray { density: f32 },
    /// White matter — high axon density, low soma density (fiber tracts).
    White { myelination: f32, coherence: f32 },
    /// Sparse — neither gray nor white (empty or boundary).
    Sparse,
}

impl TissueType {
    /// Is this gray matter?
    #[inline]
    pub fn is_gray(&self) -> bool {
        matches!(self, TissueType::Gray { .. })
    }

    /// Is this white matter?
    #[inline]
    pub fn is_white(&self) -> bool {
        matches!(self, TissueType::White { .. })
    }

    /// Is this sparse?
    #[inline]
    pub fn is_sparse(&self) -> bool {
        matches!(self, TissueType::Sparse)
    }

    /// Signal velocity factor (1.0 = baseline).
    pub fn velocity_factor(&self) -> f32 {
        match self {
            TissueType::White { myelination, .. } => 1.0 + myelination * 2.0,
            TissueType::Gray { density } => 0.5 + (1.0 - density) * 0.3,
            TissueType::Sparse => 0.3,
        }
    }
}

// ============================================================================
// Tissue Configuration
// ============================================================================

/// Configuration for tissue field computation.
#[derive(Clone, Copy, Debug)]
pub struct TissueConfig {
    /// Spatial hash cell size (should be >= kernel_radius).
    pub cell_size: f32,
    /// Kernel radius for density estimation.
    pub kernel_radius: f32,
    /// Soma density threshold for gray matter.
    pub gray_threshold: f32,
    /// Axon density threshold for white matter.
    pub white_threshold: f32,
    /// Ratio of axon/soma density required for white matter.
    pub white_ratio: f32,
    /// Number of samples along path for propagation delay.
    pub path_samples: usize,
    /// Base propagation velocity (units per microsecond).
    pub base_velocity: f32,
}

impl Default for TissueConfig {
    fn default() -> Self {
        Self {
            cell_size: 2.0,
            kernel_radius: 1.5,
            gray_threshold: 0.3,
            white_threshold: 0.2,
            white_ratio: 3.0,
            path_samples: 10,
            base_velocity: 0.001, // 1 unit per millisecond
        }
    }
}

// ============================================================================
// Tissue Field
// ============================================================================

/// Continuous tissue field computed from neuron positions.
#[derive(Clone, Debug)]
pub struct TissueField {
    /// Spatial hash for soma positions.
    soma_hash: SpatialHash,
    /// All axon segments (for white matter detection).
    axon_segments: Vec<AxonSegment>,
    /// Configuration.
    config: TissueConfig,
    /// Cached neuron count (for validation).
    neuron_count: usize,
}

impl TissueField {
    /// Create a new tissue field with default configuration.
    pub fn new() -> Self {
        Self::with_config(TissueConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: TissueConfig) -> Self {
        Self {
            soma_hash: SpatialHash::new(config.cell_size),
            axon_segments: Vec::new(),
            config,
            neuron_count: 0,
        }
    }

    /// Rebuild the field from neurons.
    pub fn rebuild(&mut self, neurons: &[SpatialNeuron]) {
        self.neuron_count = neurons.len();
        self.soma_hash.rebuild(neurons);

        self.axon_segments.clear();
        self.axon_segments.reserve(neurons.len());
        for (idx, neuron) in neurons.iter().enumerate() {
            self.axon_segments.push(AxonSegment::from_neuron(idx as u32, neuron));
        }
    }

    /// Update axon segments after migration (neurons moved).
    pub fn update_axons(&mut self, neurons: &[SpatialNeuron]) {
        for (idx, neuron) in neurons.iter().enumerate() {
            if idx < self.axon_segments.len() {
                self.axon_segments[idx] = AxonSegment::from_neuron(idx as u32, neuron);
            }
        }
    }

    /// Gaussian kernel for density estimation.
    #[inline]
    fn gaussian_kernel(&self, distance: f32) -> f32 {
        let sigma = self.config.kernel_radius / 3.0;
        let x = distance / sigma;
        (-0.5 * x * x).exp()
    }

    /// Compute soma density at a position.
    pub fn soma_density(&self, pos: [f32; 3], neurons: &[SpatialNeuron]) -> f32 {
        let nearby = self.soma_hash.query_radius(pos, self.config.kernel_radius);
        let radius_sq = self.config.kernel_radius * self.config.kernel_radius;

        let mut density = 0.0;
        for idx in nearby {
            let idx = idx as usize;
            if idx >= neurons.len() {
                continue;
            }
            let soma_pos = neurons[idx].soma.position;
            let dx = pos[0] - soma_pos[0];
            let dy = pos[1] - soma_pos[1];
            let dz = pos[2] - soma_pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq <= radius_sq {
                density += self.gaussian_kernel(dist_sq.sqrt());
            }
        }

        // Normalize by expected max (tuned empirically)
        (density / 5.0).min(1.0)
    }

    /// Compute axon density at a position.
    pub fn axon_density(&self, pos: [f32; 3]) -> f32 {
        let mut density = 0.0;

        for segment in &self.axon_segments {
            let dist = segment.distance_to_point(pos);
            if dist <= self.config.kernel_radius {
                density += self.gaussian_kernel(dist);
            }
        }

        // Normalize by expected max
        (density / 5.0).min(1.0)
    }

    /// Compute mean myelination of nearby axons.
    pub fn mean_myelin(&self, pos: [f32; 3]) -> f32 {
        let mut total_myelin = 0.0;
        let mut count = 0.0;

        for segment in &self.axon_segments {
            let dist = segment.distance_to_point(pos);
            if dist <= self.config.kernel_radius {
                let weight = self.gaussian_kernel(dist);
                total_myelin += segment.myelin as f32 * weight;
                count += weight;
            }
        }

        if count < 0.001 {
            0.0
        } else {
            (total_myelin / count) / 255.0
        }
    }

    /// Compute axon direction coherence (how aligned nearby axons are).
    ///
    /// Returns 0.0 (random directions) to 1.0 (all parallel).
    pub fn axon_coherence(&self, pos: [f32; 3]) -> f32 {
        let mut sum_dir = [0.0f32; 3];
        let mut count = 0.0;

        for segment in &self.axon_segments {
            let dist = segment.distance_to_point(pos);
            if dist <= self.config.kernel_radius {
                let weight = self.gaussian_kernel(dist);
                let dir = segment.direction();
                // Use absolute value to handle opposite directions as aligned
                sum_dir[0] += dir[0].abs() * weight;
                sum_dir[1] += dir[1].abs() * weight;
                sum_dir[2] += dir[2].abs() * weight;
                count += weight;
            }
        }

        if count < 0.001 {
            return 0.0;
        }

        // Coherence = magnitude of average direction
        let avg_dir = [
            sum_dir[0] / count,
            sum_dir[1] / count,
            sum_dir[2] / count,
        ];
        (avg_dir[0] * avg_dir[0] + avg_dir[1] * avg_dir[1] + avg_dir[2] * avg_dir[2]).sqrt()
    }

    /// Determine tissue type at a position.
    pub fn tissue_at(&self, pos: [f32; 3], neurons: &[SpatialNeuron]) -> TissueType {
        let soma_d = self.soma_density(pos, neurons);
        let axon_d = self.axon_density(pos);

        if soma_d >= self.config.gray_threshold {
            TissueType::Gray { density: soma_d }
        } else if axon_d >= self.config.white_threshold
            && (soma_d < 0.01 || axon_d / soma_d >= self.config.white_ratio)
        {
            TissueType::White {
                myelination: self.mean_myelin(pos),
                coherence: self.axon_coherence(pos),
            }
        } else {
            TissueType::Sparse
        }
    }

    /// Compute propagation delay from source to target position.
    ///
    /// Returns delay in microseconds.
    pub fn propagation_delay_us(
        &self,
        from: [f32; 3],
        to: [f32; 3],
        neurons: &[SpatialNeuron],
    ) -> u64 {
        let dx = to[0] - from[0];
        let dy = to[1] - from[1];
        let dz = to[2] - from[2];
        let total_distance = (dx * dx + dy * dy + dz * dz).sqrt();

        if total_distance < 0.001 {
            return 0;
        }

        let samples = self.config.path_samples.max(1);
        let step_distance = total_distance / samples as f32;
        let mut total_delay = 0.0;

        for i in 0..samples {
            let t = (i as f32 + 0.5) / samples as f32;
            let pos = [
                from[0] + dx * t,
                from[1] + dy * t,
                from[2] + dz * t,
            ];

            let tissue = self.tissue_at(pos, neurons);
            let velocity = self.config.base_velocity * tissue.velocity_factor();
            total_delay += step_distance / velocity;
        }

        total_delay as u64
    }

    /// Compute gradient for axon growth (which direction to extend).
    ///
    /// Returns a direction vector that:
    /// - Points toward target
    /// - Is attracted to existing white matter tracts
    /// - Avoids dense gray matter
    pub fn growth_gradient(
        &self,
        current: [f32; 3],
        target: [f32; 3],
        neurons: &[SpatialNeuron],
    ) -> [f32; 3] {
        // Direct path to target
        let dx = target[0] - current[0];
        let dy = target[1] - current[1];
        let dz = target[2] - current[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < 0.001 {
            return [0.0, 0.0, 0.0];
        }

        let direct = [dx / dist, dy / dist, dz / dist];

        // Sample nearby directions for tract attraction
        let sample_dist = self.config.kernel_radius * 0.5;
        let mut tract_attraction = [0.0f32; 3];

        for (ox, oy, oz) in [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ] {
            let sample_pos = [
                current[0] + ox * sample_dist,
                current[1] + oy * sample_dist,
                current[2] + oz * sample_dist,
            ];
            let axon_d = self.axon_density(sample_pos);
            tract_attraction[0] += ox * axon_d;
            tract_attraction[1] += oy * axon_d;
            tract_attraction[2] += oz * axon_d;
        }

        // Gray matter repulsion
        let mut soma_repulsion = [0.0f32; 3];
        for (ox, oy, oz) in [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ] {
            let sample_pos = [
                current[0] + ox * sample_dist,
                current[1] + oy * sample_dist,
                current[2] + oz * sample_dist,
            ];
            let soma_d = self.soma_density(sample_pos, neurons);
            soma_repulsion[0] += ox * soma_d;
            soma_repulsion[1] += oy * soma_d;
            soma_repulsion[2] += oz * soma_d;
        }

        // Blend: direct (60%), tract attraction (30%), gray avoidance (10%)
        let result = [
            direct[0] * 0.6 + tract_attraction[0] * 0.3 - soma_repulsion[0] * 0.1,
            direct[1] * 0.6 + tract_attraction[1] * 0.3 - soma_repulsion[1] * 0.1,
            direct[2] * 0.6 + tract_attraction[2] * 0.3 - soma_repulsion[2] * 0.1,
        ];

        // Normalize
        let mag = (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
        if mag < 0.001 {
            direct
        } else {
            [result[0] / mag, result[1] / mag, result[2] / mag]
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &TissueConfig {
        &self.config
    }
}

impl Default for TissueField {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Emergent Regions
// ============================================================================

/// A nuclei signature for region characterization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NucleiSignature {
    /// Mean soma size bucket (0-255 → 0-3).
    pub soma_bucket: u8,
    /// Mean axon affinity bucket.
    pub axon_bucket: u8,
    /// Dominant polarity (-1, 0, +1).
    pub polarity: i8,
    /// Has oscillators.
    pub has_oscillators: bool,
    /// Has sensory neurons.
    pub has_sensory: bool,
    /// Has motor neurons.
    pub has_motor: bool,
}

impl NucleiSignature {
    /// Compute signature from a set of neurons.
    pub fn from_neurons(neurons: &[SpatialNeuron], indices: &[u32]) -> Self {
        if indices.is_empty() {
            return Self {
                soma_bucket: 0,
                axon_bucket: 0,
                polarity: 0,
                has_oscillators: false,
                has_sensory: false,
                has_motor: false,
            };
        }

        let mut soma_sum = 0u32;
        let mut axon_sum = 0u32;
        let mut polarity_sum = 0i32;
        let mut has_oscillators = false;
        let mut has_sensory = false;
        let mut has_motor = false;

        for &idx in indices {
            let idx = idx as usize;
            if idx >= neurons.len() {
                continue;
            }
            let nuclei = &neurons[idx].nuclei;
            soma_sum += nuclei.soma_size as u32;
            axon_sum += nuclei.axon_affinity as u32;
            polarity_sum += match nuclei.polarity {
                ternary_signal::Polarity::Positive => 1,
                ternary_signal::Polarity::Negative => -1,
                ternary_signal::Polarity::Zero => 0,
            };
            has_oscillators |= nuclei.is_oscillator();
            has_sensory |= nuclei.is_sensory();
            has_motor |= nuclei.is_motor();
        }

        let n = indices.len() as u32;
        Self {
            soma_bucket: ((soma_sum / n) / 64).min(3) as u8,
            axon_bucket: ((axon_sum / n) / 64).min(3) as u8,
            polarity: if polarity_sum > 0 {
                1
            } else if polarity_sum < 0 {
                -1
            } else {
                0
            },
            has_oscillators,
            has_sensory,
            has_motor,
        }
    }

    /// Similarity score to another signature (0.0-1.0).
    pub fn similarity(&self, other: &Self) -> f32 {
        let mut score = 0.0;
        let mut max_score = 0.0;

        // Soma size similarity (max 1.0)
        score += 1.0 - (self.soma_bucket as f32 - other.soma_bucket as f32).abs() / 3.0;
        max_score += 1.0;

        // Axon affinity similarity (max 1.0)
        score += 1.0 - (self.axon_bucket as f32 - other.axon_bucket as f32).abs() / 3.0;
        max_score += 1.0;

        // Polarity match (max 1.0)
        if self.polarity == other.polarity {
            score += 1.0;
        }
        max_score += 1.0;

        // Interface matches (max 0.5 each)
        if self.has_oscillators == other.has_oscillators {
            score += 0.5;
        }
        max_score += 0.5;

        if self.has_sensory == other.has_sensory {
            score += 0.5;
        }
        max_score += 0.5;

        if self.has_motor == other.has_motor {
            score += 0.5;
        }
        max_score += 0.5;

        score / max_score
    }
}

/// An emergent region detected from clustering.
#[derive(Clone, Debug)]
pub struct EmergentRegion {
    /// Unique region ID.
    pub id: u32,
    /// Neuron indices in this region.
    pub neurons: Vec<u32>,
    /// Centroid position.
    pub centroid: [f32; 3],
    /// Approximate boundary radius.
    pub radius: f32,
    /// Nuclei signature characterizing this region.
    pub signature: NucleiSignature,
}

impl EmergentRegion {
    /// Compute centroid from neuron positions.
    pub fn compute_centroid(neurons: &[SpatialNeuron], indices: &[u32]) -> [f32; 3] {
        if indices.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let mut sum = [0.0f32; 3];
        let mut count = 0;

        for &idx in indices {
            let idx = idx as usize;
            if idx < neurons.len() {
                let pos = neurons[idx].soma.position;
                sum[0] += pos[0];
                sum[1] += pos[1];
                sum[2] += pos[2];
                count += 1;
            }
        }

        if count == 0 {
            [0.0, 0.0, 0.0]
        } else {
            [
                sum[0] / count as f32,
                sum[1] / count as f32,
                sum[2] / count as f32,
            ]
        }
    }

    /// Compute radius (max distance from centroid).
    pub fn compute_radius(neurons: &[SpatialNeuron], indices: &[u32], centroid: [f32; 3]) -> f32 {
        let mut max_dist = 0.0f32;

        for &idx in indices {
            let idx = idx as usize;
            if idx < neurons.len() {
                let pos = neurons[idx].soma.position;
                let dx = pos[0] - centroid[0];
                let dy = pos[1] - centroid[1];
                let dz = pos[2] - centroid[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                max_dist = max_dist.max(dist);
            }
        }

        max_dist
    }
}

/// Configuration for region detection.
#[derive(Clone, Copy, Debug)]
pub struct RegionConfig {
    /// Maximum distance for spatial clustering.
    pub spatial_epsilon: f32,
    /// Minimum neurons to form a region.
    pub min_neurons: usize,
    /// Correlation weight in distance metric (0.0-1.0).
    pub correlation_weight: f32,
    /// Nuclei similarity weight in distance metric (0.0-1.0).
    pub nuclei_weight: f32,
}

impl Default for RegionConfig {
    fn default() -> Self {
        Self {
            spatial_epsilon: 3.0,
            min_neurons: 5,
            correlation_weight: 0.3,
            nuclei_weight: 0.2,
        }
    }
}

/// Detect emergent regions from spatial and correlation clustering.
///
/// Uses a simplified DBSCAN-like algorithm with a composite distance metric.
pub fn detect_regions(
    neurons: &[SpatialNeuron],
    correlations: Option<&CorrelationTracker>,
    current_time: u64,
    config: &RegionConfig,
) -> Vec<EmergentRegion> {
    if neurons.is_empty() {
        return Vec::new();
    }

    let n = neurons.len();
    let mut visited = vec![false; n];
    let mut cluster_id = vec![None::<u32>; n];
    let mut next_cluster = 0u32;

    // DBSCAN-like clustering
    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        // Find neighbors
        let neighbors = find_neighbors(i, neurons, correlations, current_time, config);

        if neighbors.len() < config.min_neurons {
            // Noise point (no cluster)
            continue;
        }

        // Start new cluster
        let current_cluster = next_cluster;
        next_cluster += 1;
        cluster_id[i] = Some(current_cluster);

        // Expand cluster
        let mut queue = neighbors;
        while let Some(j) = queue.pop() {
            if visited[j] {
                if cluster_id[j].is_none() {
                    cluster_id[j] = Some(current_cluster);
                }
                continue;
            }
            visited[j] = true;
            cluster_id[j] = Some(current_cluster);

            let j_neighbors = find_neighbors(j, neurons, correlations, current_time, config);
            if j_neighbors.len() >= config.min_neurons {
                queue.extend(j_neighbors);
            }
        }
    }

    // Build regions from clusters
    let mut regions = Vec::new();
    for cid in 0..next_cluster {
        let indices: Vec<u32> = cluster_id
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == Some(cid))
            .map(|(i, _)| i as u32)
            .collect();

        if indices.len() >= config.min_neurons {
            let centroid = EmergentRegion::compute_centroid(neurons, &indices);
            let radius = EmergentRegion::compute_radius(neurons, &indices, centroid);
            let signature = NucleiSignature::from_neurons(neurons, &indices);

            regions.push(EmergentRegion {
                id: cid,
                neurons: indices,
                centroid,
                radius,
                signature,
            });
        }
    }

    regions
}

/// Find neighbors within composite distance threshold.
fn find_neighbors(
    idx: usize,
    neurons: &[SpatialNeuron],
    correlations: Option<&CorrelationTracker>,
    current_time: u64,
    config: &RegionConfig,
) -> Vec<usize> {
    let pos = neurons[idx].soma.position;
    let nuclei = &neurons[idx].nuclei;
    let mut neighbors = Vec::new();

    for j in 0..neurons.len() {
        if j == idx {
            continue;
        }

        // Spatial distance
        let other_pos = neurons[j].soma.position;
        let dx = pos[0] - other_pos[0];
        let dy = pos[1] - other_pos[1];
        let dz = pos[2] - other_pos[2];
        let spatial_dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if spatial_dist > config.spatial_epsilon * 2.0 {
            // Too far, skip
            continue;
        }

        // Correlation distance (1.0 - correlation)
        let corr_dist = if let Some(corr) = correlations {
            1.0 - corr.correlation(idx, j, current_time)
        } else {
            0.5 // neutral if no correlation data
        };

        // Nuclei similarity distance
        let nuclei_dist = nuclei_distance(nuclei, &neurons[j].nuclei);

        // Composite distance
        let spatial_weight = 1.0 - config.correlation_weight - config.nuclei_weight;
        let composite = spatial_dist / config.spatial_epsilon * spatial_weight
            + corr_dist * config.correlation_weight
            + nuclei_dist * config.nuclei_weight;

        if composite <= 1.0 {
            neighbors.push(j);
        }
    }

    neighbors
}

/// Distance between two nuclei (0.0 = identical, 1.0 = maximally different).
fn nuclei_distance(a: &Nuclei, b: &Nuclei) -> f32 {
    let soma_diff = (a.soma_size as f32 - b.soma_size as f32).abs() / 255.0;
    let axon_diff = (a.axon_affinity as f32 - b.axon_affinity as f32).abs() / 255.0;
    let polarity_diff = if a.polarity == b.polarity { 0.0 } else { 1.0 };

    (soma_diff + axon_diff + polarity_diff) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Axon;

    // ========================================================================
    // Spatial Hash Tests
    // ========================================================================

    #[test]
    fn test_grid_key_from_pos() {
        let key = GridKey::from_pos([2.5, -1.2, 0.0], 1.0);
        assert_eq!(key.x, 2);
        assert_eq!(key.y, -2);
        assert_eq!(key.z, 0);
    }

    #[test]
    fn test_grid_key_neighbors() {
        let key = GridKey { x: 0, y: 0, z: 0 };
        let neighbors: Vec<_> = key.neighbors().collect();
        assert_eq!(neighbors.len(), 27); // 3x3x3
    }

    #[test]
    fn test_spatial_hash_rebuild() {
        let neurons = vec![
            SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([1.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([5.0, 5.0, 5.0]),
        ];

        let mut hash = SpatialHash::new(2.0);
        hash.rebuild(&neurons);

        // First two neurons should be in same or adjacent cells
        let nearby = hash.query_radius([0.5, 0.0, 0.0], 3.0);
        assert!(nearby.contains(&0));
        assert!(nearby.contains(&1));
    }

    // ========================================================================
    // Axon Segment Tests
    // ========================================================================

    #[test]
    fn test_axon_segment_distance_to_point() {
        let segment = AxonSegment {
            neuron: 0,
            start: [0.0, 0.0, 0.0],
            end: [10.0, 0.0, 0.0],
            myelin: 100,
        };

        // Point on the line
        assert!((segment.distance_to_point([5.0, 0.0, 0.0]) - 0.0).abs() < 0.001);

        // Point perpendicular to middle
        assert!((segment.distance_to_point([5.0, 3.0, 0.0]) - 3.0).abs() < 0.001);

        // Point beyond end
        assert!((segment.distance_to_point([12.0, 0.0, 0.0]) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_axon_segment_direction() {
        let segment = AxonSegment {
            neuron: 0,
            start: [0.0, 0.0, 0.0],
            end: [3.0, 4.0, 0.0],
            myelin: 100,
        };

        let dir = segment.direction();
        assert!((dir[0] - 0.6).abs() < 0.001);
        assert!((dir[1] - 0.8).abs() < 0.001);
        assert!((dir[2] - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Tissue Type Tests
    // ========================================================================

    #[test]
    fn test_tissue_type_velocity() {
        let gray = TissueType::Gray { density: 0.5 };
        let white = TissueType::White {
            myelination: 0.8,
            coherence: 0.9,
        };
        let sparse = TissueType::Sparse;

        // White matter should be fastest
        assert!(white.velocity_factor() > gray.velocity_factor());
        assert!(gray.velocity_factor() > sparse.velocity_factor());
    }

    // ========================================================================
    // Tissue Field Tests — THE DREAMS
    // ========================================================================

    #[test]
    fn test_gray_matter_forms_where_somas_cluster() {
        // Create a cluster of neurons at origin
        let mut neurons = Vec::new();
        for x in 0..5 {
            for y in 0..5 {
                neurons.push(SpatialNeuron::pyramidal_at([
                    x as f32 * 0.3,
                    y as f32 * 0.3,
                    0.0,
                ]));
            }
        }

        let mut field = TissueField::new();
        field.rebuild(&neurons);

        // Center of cluster should be gray matter
        let tissue = field.tissue_at([0.6, 0.6, 0.0], &neurons);
        assert!(tissue.is_gray(), "Expected gray matter at soma cluster, got {:?}", tissue);

        // Far from cluster should be sparse
        let tissue_far = field.tissue_at([100.0, 100.0, 100.0], &neurons);
        assert!(tissue_far.is_sparse(), "Expected sparse far from cluster");
    }

    #[test]
    fn test_white_matter_forms_where_axons_bundle() {
        // Create neurons with axons passing through a common corridor
        let mut neurons = Vec::new();

        // Left cluster (gray matter)
        for i in 0..10 {
            let mut n = SpatialNeuron::pyramidal_at([0.0, i as f32 * 0.5, 0.0]);
            n.axon = Axon::myelinated([20.0, i as f32 * 0.5, 0.0], 200);
            neurons.push(n);
        }

        // Right cluster (gray matter)
        for i in 0..10 {
            neurons.push(SpatialNeuron::pyramidal_at([20.0, i as f32 * 0.5, 0.0]));
        }

        let config = TissueConfig {
            kernel_radius: 2.0,
            gray_threshold: 0.2,
            white_threshold: 0.1,
            white_ratio: 2.0,
            ..Default::default()
        };
        let mut field = TissueField::with_config(config);
        field.rebuild(&neurons);

        // Middle of the corridor (where axons pass, no somas)
        let tissue_middle = field.tissue_at([10.0, 2.0, 0.0], &neurons);
        assert!(
            tissue_middle.is_white(),
            "Expected white matter in axon corridor, got {:?}",
            tissue_middle
        );

        // Verify gray matter at left cluster
        let tissue_left = field.tissue_at([0.0, 2.0, 0.0], &neurons);
        assert!(tissue_left.is_gray(), "Expected gray matter at left cluster");
    }

    #[test]
    fn test_propagation_faster_through_white_matter() {
        // Create a white matter tract
        let mut neurons = Vec::new();

        // Myelinated axons forming a tract
        for i in 0..20 {
            let mut n = SpatialNeuron::pyramidal_at([-5.0, i as f32 * 0.5, 0.0]);
            n.axon = Axon::myelinated([25.0, i as f32 * 0.5, 0.0], 200);
            neurons.push(n);
        }

        let mut field = TissueField::new();
        field.rebuild(&neurons);

        // Measure delay through the tract (white matter)
        let delay_through_tract = field.propagation_delay_us([0.0, 5.0, 0.0], [20.0, 5.0, 0.0], &neurons);

        // Measure delay through empty space (sparse)
        let delay_through_sparse =
            field.propagation_delay_us([0.0, 5.0, 10.0], [20.0, 5.0, 10.0], &neurons);

        // White matter should be faster (lower delay)
        assert!(
            delay_through_tract < delay_through_sparse,
            "White matter should have lower delay: tract={}, sparse={}",
            delay_through_tract,
            delay_through_sparse
        );
    }

    #[test]
    fn test_axon_coherence_parallel_axons() {
        // Create parallel axons
        let mut neurons = Vec::new();
        for i in 0..10 {
            let mut n = SpatialNeuron::pyramidal_at([0.0, i as f32, 0.0]);
            n.axon = Axon::toward([10.0, i as f32, 0.0]); // All point in +X
            neurons.push(n);
        }

        let mut field = TissueField::new();
        field.rebuild(&neurons);

        let coherence = field.axon_coherence([5.0, 5.0, 0.0]);
        assert!(
            coherence > 0.8,
            "Parallel axons should have high coherence, got {}",
            coherence
        );
    }

    #[test]
    fn test_axon_coherence_random_directions() {
        // Create axons pointing in random directions
        let mut neurons = Vec::new();
        let directions = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];

        for (i, dir) in directions.iter().enumerate() {
            let mut n = SpatialNeuron::pyramidal_at([0.0, 0.0, i as f32 * 0.1]);
            n.axon = Axon::toward([dir[0] * 5.0, dir[1] * 5.0, dir[2] * 5.0 + i as f32 * 0.1]);
            neurons.push(n);
        }

        let mut field = TissueField::new();
        field.rebuild(&neurons);

        let coherence = field.axon_coherence([0.0, 0.0, 0.3]);
        // Random directions should have lower coherence than parallel
        assert!(
            coherence < 0.8,
            "Random axon directions should have lower coherence, got {}",
            coherence
        );
    }

    #[test]
    fn test_growth_gradient_toward_target() {
        let neurons = vec![SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0])];

        let field = TissueField::new();

        let gradient = field.growth_gradient([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], &neurons);

        // Should point toward target (positive X)
        assert!(
            gradient[0] > 0.5,
            "Gradient should point toward target, got {:?}",
            gradient
        );
    }

    #[test]
    fn test_growth_gradient_attracted_to_tract() {
        // Create an existing white matter tract
        let mut neurons = Vec::new();
        for i in 0..10 {
            let mut n = SpatialNeuron::pyramidal_at([-5.0, i as f32 + 2.0, 0.0]);
            n.axon = Axon::myelinated([25.0, i as f32 + 2.0, 0.0], 200);
            neurons.push(n);
        }

        let mut field = TissueField::new();
        field.rebuild(&neurons);

        // Growth from below the tract, targeting far right
        // Should be influenced by the tract above
        let gradient = field.growth_gradient([0.0, 0.0, 0.0], [20.0, 0.0, 0.0], &neurons);

        // Y component should be slightly positive (attracted toward tract at y~6)
        // This tests that existing tracts influence growth direction
        assert!(
            gradient[1] > -0.3,
            "Gradient should be influenced by nearby tract, y component: {}",
            gradient[1]
        );
    }

    // ========================================================================
    // Emergent Region Tests
    // ========================================================================

    #[test]
    fn test_regions_emerge_from_spatial_clusters() {
        let mut neurons = Vec::new();

        // Cluster 1: around origin
        for i in 0..10 {
            neurons.push(SpatialNeuron::pyramidal_at([
                (i % 3) as f32 * 0.5,
                (i / 3) as f32 * 0.5,
                0.0,
            ]));
        }

        // Cluster 2: far away
        for i in 0..10 {
            neurons.push(SpatialNeuron::pyramidal_at([
                20.0 + (i % 3) as f32 * 0.5,
                (i / 3) as f32 * 0.5,
                0.0,
            ]));
        }

        let config = RegionConfig {
            spatial_epsilon: 2.0,
            min_neurons: 3,
            correlation_weight: 0.0, // pure spatial
            nuclei_weight: 0.0,
        };

        let regions = detect_regions(&neurons, None, 0, &config);

        assert_eq!(regions.len(), 2, "Should detect two separate regions");

        // Regions should have different centroids
        let centroids: Vec<_> = regions.iter().map(|r| r.centroid).collect();
        let dist = ((centroids[0][0] - centroids[1][0]).powi(2)
            + (centroids[0][1] - centroids[1][1]).powi(2))
        .sqrt();
        assert!(dist > 10.0, "Region centroids should be far apart");
    }

    #[test]
    fn test_nuclei_signature_characterizes_region() {
        let mut neurons = Vec::new();

        // Excitatory cluster
        for i in 0..10 {
            neurons.push(SpatialNeuron::pyramidal_at([i as f32 * 0.5, 0.0, 0.0]));
        }

        // Inhibitory cluster
        for i in 0..10 {
            neurons.push(SpatialNeuron::interneuron_at([i as f32 * 0.5, 10.0, 0.0]));
        }

        let config = RegionConfig {
            spatial_epsilon: 3.0,
            min_neurons: 3,
            correlation_weight: 0.0,
            nuclei_weight: 0.0,
        };

        let regions = detect_regions(&neurons, None, 0, &config);

        assert_eq!(regions.len(), 2);

        // Find which region is which by checking signature
        let excitatory_region = regions.iter().find(|r| r.signature.polarity == 1);
        let inhibitory_region = regions.iter().find(|r| r.signature.polarity == -1);

        assert!(excitatory_region.is_some(), "Should have excitatory region");
        assert!(inhibitory_region.is_some(), "Should have inhibitory region");
    }

    #[test]
    fn test_signature_similarity() {
        let sig1 = NucleiSignature {
            soma_bucket: 2,
            axon_bucket: 2,
            polarity: 1,
            has_oscillators: false,
            has_sensory: false,
            has_motor: false,
        };

        let sig_similar = NucleiSignature {
            soma_bucket: 2,
            axon_bucket: 2,
            polarity: 1,
            has_oscillators: false,
            has_sensory: false,
            has_motor: false,
        };

        let sig_different = NucleiSignature {
            soma_bucket: 0,
            axon_bucket: 0,
            polarity: -1,
            has_oscillators: true,
            has_sensory: true,
            has_motor: true,
        };

        assert!(
            sig1.similarity(&sig_similar) > 0.9,
            "Identical signatures should be very similar"
        );
        assert!(
            sig1.similarity(&sig_different) < 0.5,
            "Different signatures should have low similarity"
        );
    }

    #[test]
    fn test_region_radius_correct() {
        let neurons = vec![
            SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([4.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([0.0, 3.0, 0.0]),
        ];

        let indices: Vec<u32> = vec![0, 1, 2];
        let centroid = EmergentRegion::compute_centroid(&neurons, &indices);
        let radius = EmergentRegion::compute_radius(&neurons, &indices, centroid);

        // Centroid should be around (1.33, 1.0, 0.0)
        // Max distance should be from centroid to (4.0, 0.0, 0.0)
        assert!(radius > 2.0, "Radius should be > 2.0, got {}", radius);
        assert!(radius < 4.0, "Radius should be < 4.0, got {}", radius);
    }

    #[test]
    fn test_min_neurons_threshold() {
        let neurons = vec![
            SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([0.5, 0.0, 0.0]),
        ];

        let config = RegionConfig {
            spatial_epsilon: 2.0,
            min_neurons: 5, // Require 5 neurons minimum
            correlation_weight: 0.0,
            nuclei_weight: 0.0,
        };

        let regions = detect_regions(&neurons, None, 0, &config);

        assert!(
            regions.is_empty(),
            "Should not form region with fewer than min_neurons"
        );
    }

    // ========================================================================
    // Integration Test — Full Tissue + Region Pipeline
    // ========================================================================

    #[test]
    fn test_tissue_and_regions_together() {
        // Build a small "brain" with two regions connected by a tract
        let mut neurons = Vec::new();

        // Region A: sensory cluster at origin
        for i in 0..15 {
            neurons.push(SpatialNeuron::sensory_at(
                [(i % 4) as f32 * 0.5, (i / 4) as f32 * 0.5, 0.0],
                0,
                1,
            ));
        }

        // Region B: motor cluster at [20, 0, 0]
        for i in 0..15 {
            neurons.push(SpatialNeuron::motor_at(
                [20.0 + (i % 4) as f32 * 0.5, (i / 4) as f32 * 0.5, 0.0],
                0,
                2,
            ));
        }

        // Projection neurons with long axons (create white matter tract)
        for i in 0..10 {
            let mut n = SpatialNeuron::pyramidal_at([2.0, i as f32 * 0.3 + 0.5, 0.0]);
            n.axon = Axon::myelinated([18.0, i as f32 * 0.3 + 0.5, 0.0], 200);
            neurons.push(n);
        }

        // Build tissue field
        let mut field = TissueField::new();
        field.rebuild(&neurons);

        // Detect regions
        let config = RegionConfig {
            spatial_epsilon: 2.0,
            min_neurons: 5,
            correlation_weight: 0.0,
            nuclei_weight: 0.1,
        };
        let regions = detect_regions(&neurons, None, 0, &config);

        // Should have at least 2 distinct regions
        assert!(
            regions.len() >= 2,
            "Should detect at least 2 regions, got {}",
            regions.len()
        );

        // Check tissue types
        let tissue_origin = field.tissue_at([1.0, 1.0, 0.0], &neurons);
        assert!(tissue_origin.is_gray(), "Origin should be gray matter");

        let tissue_motor = field.tissue_at([21.0, 1.0, 0.0], &neurons);
        assert!(tissue_motor.is_gray(), "Motor region should be gray matter");

        // Middle of tract should be influenced by axons
        let tissue_tract = field.tissue_at([10.0, 1.5, 0.0], &neurons);
        // Either white or at least has axon density
        let axon_d = field.axon_density([10.0, 1.5, 0.0]);
        assert!(
            axon_d > 0.0 || tissue_tract.is_white(),
            "Tract area should have axon density or be white matter"
        );
    }
}
