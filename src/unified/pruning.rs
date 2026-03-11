//! Synaptic Pruning for Unified Neurons — connections die, neurons persist.
//!
//! Adapted from `spatial::pruning` for unified neuron types. Adds synapse
//! health-based decay: synapses that haven't conducted recently lose health
//! and eventually die. This is the use-it-or-lose-it mechanism.
//!
//! ## Pruning Mechanisms
//!
//! 1. **Synapse health decay** — synapses that haven't conducted within the
//!    inactivity window lose health each cycle
//! 2. **Dormancy timeout** — synapses at low magnitude for too long are marked prunable
//! 3. **Axon health decay** — neurons with no active outgoing synapses lose axon health
//! 4. **Axon retraction** — dead axons retract toward soma
//!
//! ## What Happens When Pruned
//!
//! ```text
//! Synapse health → 0:
//!   - Signal zeroed out (dormant)
//!   - Synapse persists structurally until hard_prune removes it
//!
//! Axon health → 0:
//!   - Axon retracts toward soma
//!   - All outgoing synapses effectively dead
//!   - Neuron becomes isolated (but still exists)
//! ```

use super::neuron::UnifiedNeuron;
use super::synapse::UnifiedSynapseStore;

/// Configuration for synaptic pruning.
#[derive(Clone, Copy, Debug)]
pub struct PruningConfig {
    /// Health decay per pruning cycle for inactive axons.
    pub inactivity_decay: u8,
    /// Dormancy threshold (cycles at low magnitude before pruning).
    pub dormancy_threshold: u8,
    /// Minimum synapse magnitude to avoid dormancy counting.
    pub activity_threshold: u8,
    /// Health boost when synapse is active.
    pub activity_boost: u8,
    /// Axon retraction rate (fraction per cycle when health is 0).
    pub retraction_rate: f32,
    /// Health decay per unit distance (metabolic cost of long axons).
    pub distance_decay_per_unit: f32,
    /// Inactivity window in microseconds. Synapses that haven't conducted
    /// within this window lose health each pruning cycle.
    pub inactivity_window_us: u64,
    /// Health decay amount per cycle for inactive synapses.
    pub synapse_health_decay: u8,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            inactivity_decay: 1,
            dormancy_threshold: 10,
            activity_threshold: 5,
            activity_boost: 2,
            retraction_rate: 0.1,
            distance_decay_per_unit: 0.1,
            inactivity_window_us: 2_000_000, // 2 seconds
            synapse_health_decay: 5,
        }
    }
}

/// Tracks dormancy duration for synapses.
#[derive(Clone, Debug, Default)]
pub struct DormancyTracker {
    /// Dormancy count per synapse (indexed by synapse index in store).
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
}

/// Result of a pruning cycle.
#[derive(Clone, Debug, Default)]
pub struct PruningResult {
    /// Number of synapses identified as prunable.
    pub synapses_pruned: usize,
    /// Number of axons that hit zero health.
    pub axons_depleted: usize,
    /// Number of synapses that were dormant (low magnitude).
    pub synapses_dormant: usize,
    /// Number of synapses that were active.
    pub synapses_active: usize,
    /// Number of synapses that lost health due to inactivity.
    pub synapses_decayed: usize,
}

/// Decay synapse health based on conduction recency.
///
/// For each synapse: if `current_time_us - last_conducted_us > inactivity_window_us`,
/// decay health. This is the use-it-or-lose-it mechanism that the spatial pruning
/// system doesn't have (spatial tracks dormancy counts, not conduction time).
pub fn decay_synapse_health(
    synapses: &mut UnifiedSynapseStore,
    current_time_us: u64,
    config: &PruningConfig,
) -> usize {
    let mut decayed = 0;
    for syn in synapses.iter_mut() {
        if syn.health == 0 || syn.maturity == 255 {
            continue; // already dead or frozen
        }
        let inactive_duration = current_time_us.saturating_sub(syn.last_conducted_us);
        if inactive_duration > config.inactivity_window_us {
            syn.decay_health(config.synapse_health_decay);
            decayed += 1;
        }
    }
    decayed
}

/// Update dormancy counts and identify synapses to prune.
///
/// Returns indices of synapses that should be pruned.
pub fn identify_prunable_synapses(
    synapses: &UnifiedSynapseStore,
    dormancy: &mut DormancyTracker,
    config: &PruningConfig,
) -> Vec<usize> {
    let mut to_prune = Vec::new();

    for (idx, syn) in synapses.iter().enumerate() {
        // Frozen synapses are never prunable
        if syn.maturity == 255 {
            continue;
        }
        if syn.signal.magnitude < config.activity_threshold || syn.health == 0 {
            dormancy.increment(idx);
            if dormancy.get(idx) >= config.dormancy_threshold {
                to_prune.push(idx);
            }
        } else {
            dormancy.reset(idx);
        }
    }

    to_prune
}

/// Apply axon health decay based on outgoing synapse activity.
pub fn decay_axon_health(
    neurons: &mut [UnifiedNeuron],
    synapses: &UnifiedSynapseStore,
    config: &PruningConfig,
) -> usize {
    let mut depleted = 0;

    for (idx, neuron) in neurons.iter_mut().enumerate() {
        // Interface neurons (sensory/motor) are terminals — don't decay their axons.
        if neuron.nuclei.is_motor() || neuron.nuclei.is_sensory() {
            continue;
        }

        let outgoing = synapses.outgoing(idx as u32);

        if outgoing.is_empty() {
            neuron.axon.decay(config.inactivity_decay);
        } else {
            let active_count = outgoing
                .iter()
                .filter(|s| s.signal.magnitude >= config.activity_threshold && s.health > 0)
                .count();

            if active_count == 0 {
                neuron.axon.decay(config.inactivity_decay);
            } else {
                neuron.axon.boost(config.activity_boost);
            }

            // Distance-based decay: use axon terminal distance from soma position
            let soma_pos = [
                neuron.position.voxel.0 as f32,
                neuron.position.voxel.1 as f32,
                neuron.position.voxel.2 as f32,
            ];
            let axon_length = neuron.axon.length(soma_pos);
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
pub fn retract_dead_axons(neurons: &mut [UnifiedNeuron], config: &PruningConfig) {
    for neuron in neurons.iter_mut() {
        if !neuron.axon.is_alive() {
            let soma_pos = [
                neuron.position.voxel.0 as f32,
                neuron.position.voxel.1 as f32,
                neuron.position.voxel.2 as f32,
            ];
            neuron.axon.retract_toward(soma_pos, config.retraction_rate);
        }
    }
}

/// Perform one pruning cycle.
///
/// This is a "soft" pruning that decays health and identifies prunable synapses.
/// Call `hard_prune` to actually remove dead synapses from the store.
pub fn pruning_cycle(
    neurons: &mut [UnifiedNeuron],
    synapses: &mut UnifiedSynapseStore,
    dormancy: &mut DormancyTracker,
    current_time_us: u64,
    config: &PruningConfig,
) -> PruningResult {
    let mut result = PruningResult::default();

    // Count active vs dormant
    for syn in synapses.iter() {
        if syn.signal.magnitude >= config.activity_threshold && syn.health > 0 {
            result.synapses_active += 1;
        } else {
            result.synapses_dormant += 1;
        }
    }

    // Decay synapse health based on conduction recency
    result.synapses_decayed = decay_synapse_health(synapses, current_time_us, config);

    // Identify synapses to prune
    let to_prune = identify_prunable_synapses(synapses, dormancy, config);
    result.synapses_pruned = to_prune.len();

    // Decay axon health
    result.axons_depleted = decay_axon_health(neurons, synapses, config);

    // Retract dead axons
    retract_dead_axons(neurons, config);

    result
}

/// Hard prune: remove dead synapses (health=0 or dormant signal) from the store.
///
/// Rebuilds the CSR index after removal. Returns the number of synapses removed.
pub fn hard_prune(
    synapses: &mut UnifiedSynapseStore,
    dormancy: &mut DormancyTracker,
    neuron_count: usize,
) -> usize {
    let pruned = synapses.prune_dormant();
    if pruned > 0 {
        synapses.rebuild_index(neuron_count as u32);
    }
    dormancy.resize(synapses.len());
    pruned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::synapse::UnifiedSynapse;
    use crate::unified::zone::DendriticZone;

    #[test]
    fn dormancy_tracker_increment_reset() {
        let mut tracker = DormancyTracker::new(3);
        tracker.increment(0);
        tracker.increment(0);
        assert_eq!(tracker.get(0), 2);
        tracker.reset(0);
        assert_eq!(tracker.get(0), 0);
    }

    #[test]
    fn synapse_health_decay_inactive() {
        let mut store = UnifiedSynapseStore::new();
        let mut syn = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        syn.last_conducted_us = 0; // never conducted
        store.add(syn);
        store.rebuild_index(2);

        let config = PruningConfig {
            inactivity_window_us: 1_000_000, // 1s
            synapse_health_decay: 10,
            ..Default::default()
        };

        let decayed = decay_synapse_health(&mut store, 2_000_000, &config);
        assert_eq!(decayed, 1);

        // Health should have decreased from 200 to 190
        let s = store.iter().next().unwrap();
        assert_eq!(s.health, 190);
    }

    #[test]
    fn synapse_health_decay_recent_conduction_unaffected() {
        let mut store = UnifiedSynapseStore::new();
        let mut syn = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        syn.last_conducted_us = 1_500_000; // conducted recently
        store.add(syn);
        store.rebuild_index(2);

        let config = PruningConfig {
            inactivity_window_us: 1_000_000,
            synapse_health_decay: 10,
            ..Default::default()
        };

        let decayed = decay_synapse_health(&mut store, 2_000_000, &config);
        assert_eq!(decayed, 0);

        let s = store.iter().next().unwrap();
        assert_eq!(s.health, 200); // unchanged
    }

    #[test]
    fn synapse_health_decay_kills_synapse() {
        let mut store = UnifiedSynapseStore::new();
        let mut syn = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        syn.health = 3; // nearly dead
        syn.last_conducted_us = 0;
        store.add(syn);
        store.rebuild_index(2);

        let config = PruningConfig {
            inactivity_window_us: 1_000_000,
            synapse_health_decay: 5,
            ..Default::default()
        };

        decay_synapse_health(&mut store, 2_000_000, &config);

        let s = store.iter().next().unwrap();
        assert_eq!(s.health, 0);
        assert!(s.is_dormant());
        assert!(!s.is_active());
    }

    #[test]
    fn identify_prunable_after_threshold() {
        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500)); // active
        store.add(UnifiedSynapse::dormant(0, 2, DendriticZone::Context, 300)); // dormant
        store.rebuild_index(3);

        let mut dormancy = DormancyTracker::new(2);
        let config = PruningConfig {
            dormancy_threshold: 3,
            activity_threshold: 5,
            ..Default::default()
        };

        // Not yet at threshold
        let prunable = identify_prunable_synapses(&store, &mut dormancy, &config);
        assert!(prunable.is_empty());

        // Run enough cycles to cross threshold
        for _ in 0..3 {
            identify_prunable_synapses(&store, &mut dormancy, &config);
        }

        let prunable = identify_prunable_synapses(&store, &mut dormancy, &config);
        assert_eq!(prunable.len(), 1);
    }

    #[test]
    fn hard_prune_removes_dead() {
        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.add(UnifiedSynapse::dormant(0, 2, DendriticZone::Context, 300));
        store.rebuild_index(3);

        let mut dormancy = DormancyTracker::new(2);
        let pruned = hard_prune(&mut store, &mut dormancy, 3);
        assert_eq!(pruned, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn pruning_cycle_reports_counts() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(crate::unified::VoxelPosition::at_center((0, 0, 0))),
            UnifiedNeuron::pyramidal_at(crate::unified::VoxelPosition::at_center((1, 0, 0))),
        ];

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        let mut dormant_syn = UnifiedSynapse::excitatory(0, 1, DendriticZone::Context, 2, 300);
        dormant_syn.last_conducted_us = 0;
        store.add(dormant_syn);
        store.rebuild_index(2);

        let mut dormancy = DormancyTracker::new(2);
        let config = PruningConfig::default();

        let result = pruning_cycle(&mut neurons, &mut store, &mut dormancy, 5_000_000, &config);
        assert_eq!(result.synapses_active, 1);
        assert!(result.synapses_dormant >= 1);
    }

    #[test]
    fn conduct_boosts_health() {
        let mut syn = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        let initial_health = syn.health;
        syn.conduct(1_000_000);
        assert_eq!(syn.last_conducted_us, 1_000_000);
        assert_eq!(syn.health, initial_health + 2);
    }

    #[test]
    fn axon_decay_no_outgoing() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(crate::unified::VoxelPosition::at_center((0, 0, 0))),
        ];
        neurons[0].axon.health = 100;

        let store = UnifiedSynapseStore::new(); // no synapses
        let config = PruningConfig {
            inactivity_decay: 10,
            ..Default::default()
        };

        let depleted = decay_axon_health(&mut neurons, &store, &config);
        assert_eq!(neurons[0].axon.health, 90);
        assert_eq!(depleted, 0); // 90 > 0, not depleted
    }
}
