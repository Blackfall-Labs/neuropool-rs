#![allow(deprecated)]
//! Synaptic Pruning — connections die, neurons persist.
//!
//! In spatial neurons, neurons never die (except in pathology).
//! Only synapses die through pruning. This matches biology where
//! synapses are constantly pruned and regrown.
//!
//! ## Pruning Mechanisms
//!
//! 1. **Axon health decay** — inactive axons lose health
//! 2. **Dormancy timeout** — synapses at magnitude 0 for too long are removed
//! 3. **Metabolic pressure** — expensive synapses (long distance) decay faster
//!
//! ## What Happens When Pruned
//!
//! ```text
//! Synapse pruned:
//!   - Removed from SynapseStore
//!   - Axon health decremented (accumulated damage)
//!   - Neuron persists (can regrow connections later)
//!
//! Axon health → 0:
//!   - Axon retracts toward soma
//!   - All outgoing synapses pruned
//!   - Neuron becomes isolated (but still exists)
//! ```

use super::{SpatialNeuron, SpatialSynapseStore};

/// Configuration for synaptic pruning.
#[derive(Clone, Copy, Debug)]
pub struct PruningConfig {
    /// Health decay per pruning cycle for inactive axons
    pub inactivity_decay: u8,
    /// Health decay per unit distance (metabolic cost)
    pub distance_decay_per_unit: f32,
    /// Dormancy threshold (cycles at magnitude 0 before pruning)
    pub dormancy_threshold: u8,
    /// Minimum synapse magnitude to avoid dormancy counting
    pub activity_threshold: u8,
    /// Axon retraction rate (fraction per cycle when health is 0)
    pub retraction_rate: f32,
    /// Health boost when synapse is active
    pub activity_boost: u8,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            inactivity_decay: 1,
            distance_decay_per_unit: 0.1,
            dormancy_threshold: 10,
            activity_threshold: 5,
            retraction_rate: 0.1,
            activity_boost: 2,
        }
    }
}

/// Tracks dormancy duration for synapses.
#[derive(Clone, Debug, Default)]
pub struct DormancyTracker {
    /// Dormancy count per synapse (indexed by synapse index in store)
    dormancy_counts: Vec<u8>,
}

impl DormancyTracker {
    /// Create a new dormancy tracker.
    pub fn new(synapse_count: usize) -> Self {
        Self {
            dormancy_counts: vec![0; synapse_count],
        }
    }

    /// Resize for new synapse count.
    pub fn resize(&mut self, synapse_count: usize) {
        self.dormancy_counts.resize(synapse_count, 0);
    }

    /// Get dormancy count for a synapse.
    pub fn get(&self, idx: usize) -> u8 {
        self.dormancy_counts.get(idx).copied().unwrap_or(0)
    }

    /// Increment dormancy count.
    pub fn increment(&mut self, idx: usize) {
        if let Some(count) = self.dormancy_counts.get_mut(idx) {
            *count = count.saturating_add(1);
        }
    }

    /// Reset dormancy count (synapse became active).
    pub fn reset(&mut self, idx: usize) {
        if let Some(count) = self.dormancy_counts.get_mut(idx) {
            *count = 0;
        }
    }

    /// Clear all counts.
    pub fn clear(&mut self) {
        for count in &mut self.dormancy_counts {
            *count = 0;
        }
    }
}

/// Result of a pruning cycle.
#[derive(Clone, Debug, Default)]
pub struct PruningResult {
    /// Number of synapses pruned
    pub synapses_pruned: usize,
    /// Number of axons that hit zero health
    pub axons_depleted: usize,
    /// Number of synapses that became dormant
    pub synapses_dormant: usize,
    /// Number of synapses that were active
    pub synapses_active: usize,
}

/// Update dormancy counts and identify synapses to prune.
///
/// Returns indices of synapses that should be pruned.
pub fn identify_prunable_synapses(
    synapses: &SpatialSynapseStore,
    dormancy: &mut DormancyTracker,
    config: &PruningConfig,
) -> Vec<usize> {
    let mut to_prune = Vec::new();

    for (idx, syn) in synapses.iter().enumerate() {
        if syn.signal.magnitude < config.activity_threshold {
            // Dormant or nearly so
            dormancy.increment(idx);

            if dormancy.get(idx) >= config.dormancy_threshold {
                to_prune.push(idx);
            }
        } else {
            // Active
            dormancy.reset(idx);
        }
    }

    to_prune
}

/// Apply axon health decay based on activity and distance.
pub fn decay_axon_health(
    neurons: &mut [SpatialNeuron],
    synapses: &SpatialSynapseStore,
    config: &PruningConfig,
) -> usize {
    let mut depleted = 0;

    for (idx, neuron) in neurons.iter_mut().enumerate() {
        // Interface neurons (sensory/motor) are terminals — they don't project
        // outward through the network. Don't decay their axons for "inactivity".
        if neuron.nuclei.is_motor() || neuron.nuclei.is_sensory() {
            continue;
        }

        let outgoing = synapses.outgoing(idx as u32);

        if outgoing.is_empty() {
            // No outgoing synapses = inactive axon
            neuron.axon.decay(config.inactivity_decay);
        } else {
            // Check activity
            let active_count = outgoing.iter()
                .filter(|s| s.signal.magnitude >= config.activity_threshold)
                .count();

            if active_count == 0 {
                // All synapses dormant
                neuron.axon.decay(config.inactivity_decay);
            } else {
                // Active — boost health
                neuron.axon.boost(config.activity_boost);
            }

            // Apply distance-based decay
            let axon_length = neuron.axon.length(neuron.soma.position);
            let distance_decay = (axon_length * config.distance_decay_per_unit) as u8;
            neuron.axon.decay(distance_decay);
        }

        if !neuron.axon.is_alive() {
            depleted += 1;
        }
    }

    depleted
}

/// Retract axons that have zero health.
pub fn retract_dead_axons(neurons: &mut [SpatialNeuron], config: &PruningConfig) {
    for neuron in neurons.iter_mut() {
        if !neuron.axon.is_alive() {
            neuron.axon.retract_toward(neuron.soma.position, config.retraction_rate);
        }
    }
}

/// Perform one pruning cycle.
///
/// This is a "soft" pruning that marks synapses as dormant.
/// Call `hard_prune` to actually remove them from the store.
pub fn pruning_cycle(
    neurons: &mut [SpatialNeuron],
    synapses: &SpatialSynapseStore,
    dormancy: &mut DormancyTracker,
    config: &PruningConfig,
) -> PruningResult {
    let mut result = PruningResult::default();

    // Count active vs dormant
    for syn in synapses.iter() {
        if syn.signal.magnitude >= config.activity_threshold {
            result.synapses_active += 1;
        } else {
            result.synapses_dormant += 1;
        }
    }

    // Identify synapses to prune
    let to_prune = identify_prunable_synapses(synapses, dormancy, config);
    result.synapses_pruned = to_prune.len();

    // Decay axon health
    result.axons_depleted = decay_axon_health(neurons, synapses, config);

    // Retract dead axons
    retract_dead_axons(neurons, config);

    result
}

/// Hard prune: actually remove dormant synapses from the store.
///
/// Call this periodically to clean up the synapse store.
pub fn hard_prune(
    synapses: &mut SpatialSynapseStore,
    dormancy: &mut DormancyTracker,
    neuron_count: usize,
) -> usize {
    let before = synapses.len();
    synapses.prune_dormant(neuron_count);
    let after = synapses.len();

    // Resize dormancy tracker
    dormancy.resize(after);

    before - after
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::SpatialSynapse;
    use ternary_signal::Signal;

    fn make_test_synapse(magnitude: u8) -> SpatialSynapse {
        SpatialSynapse::with_signal(0, 1, Signal::positive(magnitude), 100)
    }

    #[test]
    fn test_dormancy_tracker() {
        let mut tracker = DormancyTracker::new(3);

        tracker.increment(0);
        tracker.increment(0);
        assert_eq!(tracker.get(0), 2);

        tracker.reset(0);
        assert_eq!(tracker.get(0), 0);
    }

    #[test]
    fn test_identify_prunable() {
        let mut store = SpatialSynapseStore::new(2);
        store.add(make_test_synapse(100)); // active
        store.add(make_test_synapse(0));   // dormant
        store.rebuild_index(2);

        let mut dormancy = DormancyTracker::new(2);
        let config = PruningConfig {
            dormancy_threshold: 3,
            activity_threshold: 5,
            ..Default::default()
        };

        // First cycle
        let prunable = identify_prunable_synapses(&store, &mut dormancy, &config);
        assert!(prunable.is_empty()); // not yet at threshold

        // More cycles
        for _ in 0..3 {
            identify_prunable_synapses(&store, &mut dormancy, &config);
        }

        let prunable = identify_prunable_synapses(&store, &mut dormancy, &config);
        assert_eq!(prunable.len(), 1); // dormant synapse should be prunable
    }

    #[test]
    fn test_axon_decay_inactive() {
        let mut neurons = vec![SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0])];
        neurons[0].axon.health = 100;

        let store = SpatialSynapseStore::new(1); // no synapses
        let config = PruningConfig {
            inactivity_decay: 10,
            ..Default::default()
        };

        decay_axon_health(&mut neurons, &store, &config);
        assert_eq!(neurons[0].axon.health, 90);
    }

    #[test]
    fn test_axon_boost_active() {
        let mut neurons = vec![SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0])];
        neurons[0].axon.health = 100;

        let mut store = SpatialSynapseStore::new(2);
        store.add(make_test_synapse(100)); // active synapse
        store.rebuild_index(2);

        let config = PruningConfig {
            activity_boost: 5,
            activity_threshold: 5,
            distance_decay_per_unit: 0.0, // disable for this test
            ..Default::default()
        };

        decay_axon_health(&mut neurons, &store, &config);
        assert!(neurons[0].axon.health > 100);
    }

    #[test]
    fn test_retract_dead_axon() {
        let mut neurons = vec![SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0])];
        neurons[0].axon.terminal = [10.0, 0.0, 0.0];
        neurons[0].axon.health = 0;

        let config = PruningConfig {
            retraction_rate: 0.5,
            ..Default::default()
        };

        retract_dead_axons(&mut neurons, &config);

        // Should have retracted halfway
        assert!((neurons[0].axon.terminal[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_hard_prune() {
        let mut store = SpatialSynapseStore::new(2);
        store.add(make_test_synapse(100)); // active
        store.add(SpatialSynapse::dormant(0, 1, 100)); // dormant
        store.rebuild_index(2);

        let mut dormancy = DormancyTracker::new(2);

        let pruned = hard_prune(&mut store, &mut dormancy, 2);
        assert_eq!(pruned, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_pruning_cycle() {
        let mut neurons = vec![
            SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]),
            SpatialNeuron::pyramidal_at([1.0, 0.0, 0.0]),
        ];

        let mut store = SpatialSynapseStore::new(2);
        store.add(make_test_synapse(100)); // active
        store.add(make_test_synapse(0));   // dormant
        store.rebuild_index(2);

        let mut dormancy = DormancyTracker::new(2);
        let config = PruningConfig::default();

        let result = pruning_cycle(&mut neurons, &store, &mut dormancy, &config);

        assert_eq!(result.synapses_active, 1);
        assert_eq!(result.synapses_dormant, 1);
    }

    #[test]
    fn test_distance_decay() {
        let mut neurons = vec![SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0])];
        neurons[0].axon.terminal = [10.0, 0.0, 0.0]; // 10 units away
        neurons[0].axon.health = 100;

        let mut store = SpatialSynapseStore::new(2);
        store.add(make_test_synapse(100));
        store.rebuild_index(2);

        let config = PruningConfig {
            distance_decay_per_unit: 1.0, // 1 health per unit
            activity_threshold: 5,
            activity_boost: 0, // disable for this test
            ..Default::default()
        };

        decay_axon_health(&mut neurons, &store, &config);

        // Should have lost 10 health (10 units * 1.0)
        assert_eq!(neurons[0].axon.health, 90);
    }
}
