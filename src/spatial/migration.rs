//! Neuron Migration — activity-dependent position changes.
//!
//! Neurons migrate based on two forces:
//! 1. **Attraction** toward correlated partners (Hebbian: fire together, cluster together)
//! 2. **Repulsion** from metabolic competitors (same type, no correlation)
//!
//! This creates organic clustering — regions emerge from activity patterns,
//! not from predefined boxes.
//!
//! ## How It Works
//!
//! ```text
//! For each neuron:
//!   1. Find correlated partners (neurons it fires with)
//!   2. Compute attraction toward their center of mass
//!   3. Find competitors (same nuclei type, uncorrelated)
//!   4. Compute repulsion away from their center of mass
//!   5. Apply net force with migration rate
//!   6. Axon terminal follows soma (elastic connection)
//! ```

use super::{SpatialNeuron, TissueField};

/// Configuration for neuron migration.
#[derive(Clone, Copy, Debug)]
pub struct MigrationConfig {
    /// Base migration rate (units per migration step)
    pub migration_rate: f32,
    /// Correlation threshold for "firing together" (0.0-1.0)
    pub correlation_threshold: f32,
    /// Attraction strength toward correlated partners
    pub attraction_strength: f32,
    /// Repulsion strength from competitors
    pub repulsion_strength: f32,
    /// Minimum distance to maintain (prevents collapse)
    pub min_distance: f32,
    /// Maximum migration distance per step
    pub max_step: f32,
    /// Axon elasticity (0.0 = rigid, 1.0 = fully elastic)
    pub axon_elasticity: f32,
    /// Volume exclusion radius — unconditional short-range repulsion (prevents point collapse)
    pub exclusion_radius: f32,
    /// Volume exclusion strength
    pub exclusion_strength: f32,
    /// Origin spring strength — homeostatic anchor toward initial position (0 = disabled)
    pub origin_spring: f32,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            migration_rate: 0.1,
            correlation_threshold: 0.3,
            attraction_strength: 1.0,
            repulsion_strength: 0.5,
            min_distance: 0.5,
            max_step: 0.5,
            axon_elasticity: 0.8,
            exclusion_radius: 0.3,
            exclusion_strength: 2.0,
            origin_spring: 0.0,
        }
    }
}

/// Correlation data for a pair of neurons.
#[derive(Clone, Copy, Debug)]
pub struct CorrelationEntry {
    /// First neuron index
    pub a: u32,
    /// Second neuron index
    pub b: u32,
    /// Correlation score (0.0-1.0)
    pub correlation: f32,
}

/// Tracks spike correlations between neurons.
#[derive(Clone, Debug, Default)]
pub struct CorrelationTracker {
    /// Recent spike times per neuron (circular buffer)
    spike_times: Vec<Vec<u64>>,
    /// Maximum spikes to track per neuron
    max_spikes: usize,
    /// Time window for correlation (μs)
    window_us: u64,
}

impl CorrelationTracker {
    /// Create a new correlation tracker.
    pub fn new(neuron_count: usize, max_spikes: usize, window_us: u64) -> Self {
        Self {
            spike_times: vec![Vec::with_capacity(max_spikes); neuron_count],
            max_spikes,
            window_us,
        }
    }

    /// Record a spike.
    pub fn record_spike(&mut self, neuron: usize, time_us: u64) {
        if neuron >= self.spike_times.len() {
            return;
        }

        let times = &mut self.spike_times[neuron];
        if times.len() >= self.max_spikes {
            times.remove(0);
        }
        times.push(time_us);
    }

    /// Resize for new neuron count.
    pub fn resize(&mut self, neuron_count: usize) {
        self.spike_times.resize_with(neuron_count, || Vec::with_capacity(self.max_spikes));
    }

    /// Compute correlation between two neurons.
    ///
    /// Correlation is based on how many spikes occurred within the time window.
    pub fn correlation(&self, a: usize, b: usize, current_time: u64) -> f32 {
        if a >= self.spike_times.len() || b >= self.spike_times.len() {
            return 0.0;
        }

        let times_a = &self.spike_times[a];
        let times_b = &self.spike_times[b];

        if times_a.is_empty() || times_b.is_empty() {
            return 0.0;
        }

        // Count coincident spikes (within window of each other)
        // Use a reasonable lookback window (100x the correlation window)
        let mut coincident = 0;
        let lookback = self.window_us.saturating_mul(100).max(10_000);
        let cutoff = current_time.saturating_sub(lookback);

        for &ta in times_a.iter().filter(|&&t| t >= cutoff) {
            for &tb in times_b.iter().filter(|&&t| t >= cutoff) {
                let diff = if ta > tb { ta - tb } else { tb - ta };
                if diff <= self.window_us {
                    coincident += 1;
                }
            }
        }

        // Normalize by geometric mean of spike counts
        let count_a = times_a.iter().filter(|&&t| t > cutoff).count() as f32;
        let count_b = times_b.iter().filter(|&&t| t > cutoff).count() as f32;

        if count_a < 1.0 || count_b < 1.0 {
            return 0.0;
        }

        let max_possible = (count_a * count_b).sqrt();
        if max_possible < 1.0 {
            return 0.0;
        }

        (coincident as f32 / max_possible).min(1.0)
    }

    /// Find all correlated partners for a neuron.
    pub fn correlated_partners(
        &self,
        neuron: usize,
        threshold: f32,
        current_time: u64,
    ) -> Vec<(usize, f32)> {
        let mut partners = Vec::new();

        for other in 0..self.spike_times.len() {
            if other == neuron {
                continue;
            }

            let corr = self.correlation(neuron, other, current_time);
            if corr >= threshold {
                partners.push((other, corr));
            }
        }

        partners
    }

    /// Clear all spike history.
    pub fn clear(&mut self) {
        for times in &mut self.spike_times {
            times.clear();
        }
    }
}

/// Compute migration forces for all neurons.
///
/// If a `TissueField` is provided, forces are attenuated by the local
/// tissue resistance at each neuron's position: `force /= (1 + R)`.
/// This makes neurons move slower through stiff tissue and faster through
/// softened corridors.
pub fn compute_migration_forces(
    neurons: &[SpatialNeuron],
    correlations: &CorrelationTracker,
    config: &MigrationConfig,
    current_time: u64,
    origins: Option<&[[f32; 3]]>,
    tissue: Option<&TissueField>,
) -> Vec<[f32; 3]> {
    let mut forces = vec![[0.0f32; 3]; neurons.len()];

    for i in 0..neurons.len() {
        let pos = neurons[i].soma.position;

        // Find correlated partners
        let partners = correlations.correlated_partners(i, config.correlation_threshold, current_time);

        // Attraction toward partners
        if !partners.is_empty() {
            let mut attraction = [0.0f32; 3];
            let mut total_weight = 0.0f32;

            for (partner_idx, corr) in &partners {
                let partner_pos = neurons[*partner_idx].soma.position;
                let dx = partner_pos[0] - pos[0];
                let dy = partner_pos[1] - pos[1];
                let dz = partner_pos[2] - pos[2];

                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist > config.min_distance {
                    let weight = *corr;
                    attraction[0] += dx * weight;
                    attraction[1] += dy * weight;
                    attraction[2] += dz * weight;
                    total_weight += weight;
                }
            }

            if total_weight > 0.0 {
                forces[i][0] += (attraction[0] / total_weight) * config.attraction_strength;
                forces[i][1] += (attraction[1] / total_weight) * config.attraction_strength;
                forces[i][2] += (attraction[2] / total_weight) * config.attraction_strength;
            }
        }

        // Repulsion forces: volume exclusion + competitive repulsion
        let mut repulsion = [0.0f32; 3];
        let mut repulsion_count = 0;

        for j in 0..neurons.len() {
            if i == j {
                continue;
            }

            let other_pos = neurons[j].soma.position;
            let dx = pos[0] - other_pos[0];
            let dy = pos[1] - other_pos[1];
            let dz = pos[2] - other_pos[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let dist = dist_sq.sqrt();

            // Volume exclusion — ALL neurons, unconditional at short range.
            // Neurons are physical objects; even correlated ones can't share a point.
            if config.exclusion_radius > 0.0 && dist < config.exclusion_radius && dist > 0.001 {
                let strength = config.exclusion_strength * (1.0 - dist / config.exclusion_radius);
                let norm = 1.0 / dist;
                forces[i][0] += dx * norm * strength;
                forces[i][1] += dy * norm * strength;
                forces[i][2] += dz * norm * strength;
            }

            // Competitive repulsion — same-type, uncorrelated only
            let same_type = neurons[i].nuclei.polarity == neurons[j].nuclei.polarity
                && neurons[i].nuclei.interface.kind == neurons[j].nuclei.interface.kind;

            if !same_type {
                continue;
            }

            let corr = correlations.correlation(i, j, current_time);
            if corr >= config.correlation_threshold {
                continue; // correlated, not a competitor
            }

            if dist < config.min_distance * 3.0 && dist > 0.01 {
                // Inverse square repulsion
                let strength = 1.0 / (dist_sq + 0.1);
                repulsion[0] += dx * strength;
                repulsion[1] += dy * strength;
                repulsion[2] += dz * strength;
                repulsion_count += 1;
            }
        }

        if repulsion_count > 0 {
            let scale = config.repulsion_strength / repulsion_count as f32;
            forces[i][0] += repulsion[0] * scale;
            forces[i][1] += repulsion[1] * scale;
            forces[i][2] += repulsion[2] * scale;
        }

        // Origin spring — homeostatic anchor toward initial position
        if config.origin_spring > 0.0 {
            if let Some(origins) = origins {
                let origin = origins[i];
                forces[i][0] += (origin[0] - pos[0]) * config.origin_spring;
                forces[i][1] += (origin[1] - pos[1]) * config.origin_spring;
                forces[i][2] += (origin[2] - pos[2]) * config.origin_spring;
            }
        }
    }

    // Clamp forces to max step
    for force in &mut forces {
        let mag = (force[0] * force[0] + force[1] * force[1] + force[2] * force[2]).sqrt();
        if mag > config.max_step {
            let scale = config.max_step / mag;
            force[0] *= scale;
            force[1] *= scale;
            force[2] *= scale;
        }
    }

    // Tissue resistance attenuation: force /= (1 + R)
    // Stiff tissue (high R) damps movement. Softened corridors (low R) let neurons slide.
    if let Some(tissue) = tissue {
        for (i, force) in forces.iter_mut().enumerate() {
            let r = tissue.resistance_at(neurons[i].soma.position, neurons);
            let damping = 1.0 / (1.0 + r);
            force[0] *= damping;
            force[1] *= damping;
            force[2] *= damping;
        }
    }

    forces
}

/// Apply migration forces to neurons.
pub fn apply_migration(
    neurons: &mut [SpatialNeuron],
    forces: &[[f32; 3]],
    config: &MigrationConfig,
) {
    for (neuron, force) in neurons.iter_mut().zip(forces.iter()) {
        // Skip if force is negligible
        let mag = force[0] * force[0] + force[1] * force[1] + force[2] * force[2];
        if mag < 0.0001 {
            continue;
        }

        // Apply to soma
        let delta = [
            force[0] * config.migration_rate,
            force[1] * config.migration_rate,
            force[2] * config.migration_rate,
        ];

        neuron.soma.translate(delta);

        // Axon terminal follows with elasticity
        neuron.axon.terminal[0] += delta[0] * config.axon_elasticity;
        neuron.axon.terminal[1] += delta[1] * config.axon_elasticity;
        neuron.axon.terminal[2] += delta[2] * config.axon_elasticity;
    }
}

/// Perform one migration step.
pub fn migrate_step(
    neurons: &mut [SpatialNeuron],
    correlations: &CorrelationTracker,
    config: &MigrationConfig,
    current_time: u64,
    origins: Option<&[[f32; 3]]>,
    tissue: Option<&TissueField>,
) {
    let forces = compute_migration_forces(neurons, correlations, config, current_time, origins, tissue);
    apply_migration(neurons, &forces, config);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_tracker_creation() {
        let tracker = CorrelationTracker::new(10, 20, 1000);
        assert_eq!(tracker.spike_times.len(), 10);
    }

    #[test]
    fn test_record_spike() {
        let mut tracker = CorrelationTracker::new(2, 5, 1000);
        tracker.record_spike(0, 100);
        tracker.record_spike(0, 200);
        tracker.record_spike(1, 150);

        assert_eq!(tracker.spike_times[0].len(), 2);
        assert_eq!(tracker.spike_times[1].len(), 1);
    }

    #[test]
    fn test_correlation_coincident() {
        let mut tracker = CorrelationTracker::new(2, 10, 100);

        // Fire at nearly the same time
        tracker.record_spike(0, 1000);
        tracker.record_spike(1, 1050);
        tracker.record_spike(0, 2000);
        tracker.record_spike(1, 2030);

        let corr = tracker.correlation(0, 1, 3000);
        assert!(corr > 0.5); // should be highly correlated
    }

    #[test]
    fn test_correlation_uncorrelated() {
        let mut tracker = CorrelationTracker::new(2, 10, 100);

        // Fire far apart
        tracker.record_spike(0, 1000);
        tracker.record_spike(1, 5000);
        tracker.record_spike(0, 10000);
        tracker.record_spike(1, 15000);

        let corr = tracker.correlation(0, 1, 20000);
        assert!(corr < 0.3); // should be uncorrelated
    }

    #[test]
    fn test_correlated_partners() {
        let mut tracker = CorrelationTracker::new(3, 10, 100);

        // 0 and 1 fire together, 2 fires alone
        tracker.record_spike(0, 1000);
        tracker.record_spike(1, 1020);
        tracker.record_spike(0, 2000);
        tracker.record_spike(1, 2030);
        tracker.record_spike(2, 5000);
        tracker.record_spike(2, 6000);

        let partners = tracker.correlated_partners(0, 0.3, 7000);
        assert!(partners.iter().any(|(idx, _)| *idx == 1));
        assert!(!partners.iter().any(|(idx, _)| *idx == 2));
    }

    #[test]
    fn test_migration_attraction() {
        let config = MigrationConfig::default();

        let mut tracker = CorrelationTracker::new(2, 10, 1000);
        // Make neurons correlated
        for i in 0..5 {
            tracker.record_spike(0, i * 1000);
            tracker.record_spike(1, i * 1000 + 50);
        }

        let neurons = vec![
            SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([5.0, 0.0, 0.0]),
        ];

        let forces = compute_migration_forces(&neurons, &tracker, &config, 10000, None, None);

        // Neuron 0 should be attracted toward neuron 1 (positive x)
        assert!(forces[0][0] > 0.0);
        // Neuron 1 should be attracted toward neuron 0 (negative x)
        assert!(forces[1][0] < 0.0);
    }

    #[test]
    fn test_migration_repulsion() {
        let config = MigrationConfig::default();

        // Uncorrelated neurons of same type
        let tracker = CorrelationTracker::new(2, 10, 1000);

        let neurons = vec![
            SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([0.5, 0.0, 0.0]), // close together
        ];

        let forces = compute_migration_forces(&neurons, &tracker, &config, 10000, None, None);

        // Should repel each other
        assert!(forces[0][0] < 0.0); // pushed away from 1
        assert!(forces[1][0] > 0.0); // pushed away from 0
    }

    #[test]
    fn test_apply_migration() {
        let config = MigrationConfig {
            migration_rate: 1.0,
            axon_elasticity: 0.5,
            ..Default::default()
        };

        let mut neurons = vec![SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0])];
        neurons[0].axon.terminal = [1.0, 0.0, 0.0];

        let forces = vec![[1.0, 0.0, 0.0]];
        apply_migration(&mut neurons, &forces, &config);

        assert_eq!(neurons[0].soma.position[0], 1.0);
        assert_eq!(neurons[0].axon.terminal[0], 1.5); // 1.0 + 1.0 * 0.5
    }

    #[test]
    fn test_max_step_clamping() {
        let config = MigrationConfig {
            max_step: 0.1,
            ..Default::default()
        };

        let tracker = CorrelationTracker::new(2, 10, 1000);
        let neurons = vec![
            SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([100.0, 0.0, 0.0]), // very far
        ];

        let forces = compute_migration_forces(&neurons, &tracker, &config, 0, None, None);

        // Force magnitude should be clamped
        for force in &forces {
            let mag = (force[0] * force[0] + force[1] * force[1] + force[2] * force[2]).sqrt();
            assert!(mag <= config.max_step + 0.001);
        }
    }
}
