//! Proximity-Based Wiring — connect neurons by spatial proximity.
//!
//! Creates synapses based on physical distance between axon terminals
//! and target somas, replacing brittle index-based wiring patterns.
//!
//! ## How It Works
//!
//! For each neuron with a projecting axon:
//! 1. Find all neurons whose soma falls within `max_distance` of the axon terminal
//! 2. Sort candidates by distance (nearest first)
//! 3. Create synapses up to `max_fanout` per source, `max_fanin` per target
//! 4. Synapse polarity follows source neuron's nuclei polarity
//! 5. All delays are 0 (computed from spatial distance at runtime)
//!
//! Motor neurons are skipped as sources — they're output terminals.
//! Any neuron can be a target (including sensory, for feedback circuits).

use super::{SpatialNeuron, SpatialSynapse, SpatialSynapseStore};

/// Configuration for proximity-based wiring.
#[derive(Clone, Copy, Debug)]
pub struct WiringConfig {
    /// Maximum distance from axon terminal to target soma for connection
    pub max_distance: f32,
    /// Maximum outgoing synapses per neuron
    pub max_fanout: u16,
    /// Maximum incoming synapses per neuron (dendrite spine limit)
    pub max_fanin: u16,
    /// Default synapse magnitude for new connections
    pub default_magnitude: u8,
}

impl Default for WiringConfig {
    fn default() -> Self {
        Self {
            max_distance: 5.0,
            max_fanout: 8,
            max_fanin: 20,
            default_magnitude: 100,
        }
    }
}

/// Wire neurons by spatial proximity.
///
/// For each neuron with a projecting axon, finds nearby neurons whose soma
/// falls within `max_distance` of the axon terminal and creates synapses.
/// Respects fanout/fanin caps. Motor neurons are skipped as sources.
///
/// Returns a fully indexed `SpatialSynapseStore`.
pub fn wire_by_proximity(
    neurons: &[SpatialNeuron],
    config: &WiringConfig,
) -> SpatialSynapseStore {
    let n = neurons.len();
    let mut store = SpatialSynapseStore::new(n);
    let mut fanout = vec![0u16; n];
    let mut fanin = vec![0u16; n];

    for i in 0..n {
        // Motor neurons don't project
        if neurons[i].nuclei.is_motor() {
            continue;
        }

        // Dead axons don't project
        if !neurons[i].axon.is_alive() {
            continue;
        }

        let terminal = neurons[i].axon.terminal;

        // Collect candidates sorted by distance
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            let soma = neurons[j].soma.position;
            let dx = terminal[0] - soma[0];
            let dy = terminal[1] - soma[1];
            let dz = terminal[2] - soma[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist <= config.max_distance {
                candidates.push((j, dist));
            }
        }

        // Sort by distance (nearest first)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create synapses to nearest candidates within limits
        for (j, _dist) in candidates {
            if fanout[i] >= config.max_fanout {
                break;
            }
            if fanin[j] >= config.max_fanin {
                continue;
            }

            let syn = if neurons[i].nuclei.is_inhibitory() {
                SpatialSynapse::inhibitory(
                    i as u32,
                    j as u32,
                    config.default_magnitude,
                    0, // delay computed from spatial distance at runtime
                )
            } else {
                SpatialSynapse::excitatory(
                    i as u32,
                    j as u32,
                    config.default_magnitude,
                    0,
                )
            };

            store.add(syn);
            fanout[i] += 1;
            fanin[j] += 1;
        }
    }

    store.rebuild_index(n);
    store
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Axon;

    #[test]
    fn test_proximity_wiring_basic() {
        // Two neurons: source axon terminal near target soma
        let mut source = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        source.axon = Axon::myelinated([5.0, 0.0, 0.0], 100);

        let target = SpatialNeuron::pyramidal_at([5.0, 0.5, 0.0]);

        let neurons = vec![source, target];
        let config = WiringConfig {
            max_distance: 2.0,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0).len(), 1);
        assert_eq!(store.outgoing(0)[0].target, 1);
    }

    #[test]
    fn test_proximity_wiring_out_of_range() {
        // Axon terminal too far from target soma
        let mut source = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        source.axon = Axon::myelinated([5.0, 0.0, 0.0], 100);

        let target = SpatialNeuron::pyramidal_at([20.0, 0.0, 0.0]);

        let neurons = vec![source, target];
        let config = WiringConfig {
            max_distance: 3.0,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0).len(), 0);
    }

    #[test]
    fn test_motor_neurons_dont_project() {
        let motor = SpatialNeuron::motor_at([0.0, 0.0, 0.0], 0, 1);
        let target = SpatialNeuron::pyramidal_at([0.5, 0.0, 0.0]);

        let neurons = vec![motor, target];
        let config = WiringConfig::default();

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0).len(), 0);
    }

    #[test]
    fn test_motor_neurons_receive_input() {
        let mut source = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        source.axon = Axon::myelinated([5.0, 0.0, 0.0], 100);

        let motor = SpatialNeuron::motor_at([5.0, 0.0, 0.0], 0, 1);

        let neurons = vec![source, motor];
        let config = WiringConfig {
            max_distance: 2.0,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0).len(), 1);
        assert_eq!(store.outgoing(0)[0].target, 1);
    }

    #[test]
    fn test_fanout_limit() {
        let mut source = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        source.axon = Axon::myelinated([5.0, 0.0, 0.0], 100);

        // Create many targets near axon terminal
        let mut neurons = vec![source];
        for i in 0..20 {
            neurons.push(SpatialNeuron::pyramidal_at([
                5.0 + (i as f32) * 0.1,
                0.0,
                0.0,
            ]));
        }

        let config = WiringConfig {
            max_distance: 5.0,
            max_fanout: 4,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0).len(), 4);
    }

    #[test]
    fn test_fanin_limit() {
        // Many sources, one target, low fanin cap
        let target = SpatialNeuron::pyramidal_at([10.0, 0.0, 0.0]);
        let mut neurons = vec![target];

        for i in 0..10 {
            let mut src = SpatialNeuron::pyramidal_at([i as f32, 0.0, 0.0]);
            src.axon = Axon::myelinated([10.0, (i as f32) * 0.1, 0.0], 100);
            neurons.push(src);
        }

        let config = WiringConfig {
            max_distance: 3.0,
            max_fanin: 3,
            max_fanout: 8,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);

        // Count incoming to target (neuron 0)
        let incoming: usize = (1..neurons.len())
            .map(|i| {
                store
                    .outgoing(i as u32)
                    .iter()
                    .filter(|s| s.target == 0)
                    .count()
            })
            .sum();
        assert!(incoming <= 3, "fanin should be capped at 3, got {}", incoming);
    }

    #[test]
    fn test_inhibitory_wiring() {
        let mut inh = SpatialNeuron::interneuron_at([0.0, 0.0, 0.0]);
        inh.axon = Axon::myelinated([2.0, 0.0, 0.0], 100);

        let target = SpatialNeuron::pyramidal_at([2.0, 0.0, 0.0]);

        let neurons = vec![inh, target];
        let config = WiringConfig {
            max_distance: 2.0,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0).len(), 1);
        assert!(store.outgoing(0)[0].is_inhibitory());
    }

    #[test]
    fn test_nearest_first_preference() {
        let mut source = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        source.axon = Axon::myelinated([5.0, 0.0, 0.0], 100);

        let near = SpatialNeuron::pyramidal_at([5.0, 0.0, 0.0]); // distance 0
        let mid = SpatialNeuron::pyramidal_at([6.0, 0.0, 0.0]); // distance 1
        let far = SpatialNeuron::pyramidal_at([7.0, 0.0, 0.0]); // distance 2

        let neurons = vec![source, near, mid, far];
        let config = WiringConfig {
            max_distance: 5.0,
            max_fanout: 2,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);
        let out = store.outgoing(0);
        assert_eq!(out.len(), 2);
        // Nearest targets should be selected
        assert_eq!(out[0].target, 1); // near
        assert_eq!(out[1].target, 2); // mid (not far)
    }

    #[test]
    fn test_dead_axon_skipped() {
        let mut source = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        source.axon = Axon::myelinated([5.0, 0.0, 0.0], 100);
        source.axon.health = 0; // dead

        let target = SpatialNeuron::pyramidal_at([5.0, 0.0, 0.0]);

        let neurons = vec![source, target];
        let config = WiringConfig::default();

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0).len(), 0);
    }

    #[test]
    fn test_delay_is_zero() {
        let mut source = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        source.axon = Axon::myelinated([2.0, 0.0, 0.0], 100);

        let target = SpatialNeuron::pyramidal_at([2.0, 0.0, 0.0]);

        let neurons = vec![source, target];
        let config = WiringConfig {
            max_distance: 3.0,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &config);
        assert_eq!(store.outgoing(0)[0].delay_us, 0);
    }
}
