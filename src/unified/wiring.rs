#![allow(deprecated)]
//! Proximity-Based Wiring for Unified Neurons — zone-aware, voxel-accelerated.
//!
//! Port of `spatial::wiring` to the unified neuron system. Uses `VoxelGrid`
//! for O(1) neighbor lookup instead of scanning the full neuron array.
//! Each synapse is assigned a `DendriticZone` based on the spatial relationship
//! between source and target.

use super::grid::VoxelGrid;
use super::neuron::UnifiedNeuron;
use super::synapse::{UnifiedSynapse, UnifiedSynapseStore};
use super::zone::DendriticZone;

/// Configuration for proximity-based wiring.
#[derive(Clone, Copy, Debug)]
pub struct UnifiedWiringConfig {
    /// Maximum squared distance for connection (in voxel-scaled integer units).
    /// Distance is measured from source position to target position.
    /// VoxelPosition uses 16 local units per voxel axis, so adjacent voxel
    /// centers are 16 units apart → distance_sq = 256.
    pub max_distance_sq: u64,
    /// Maximum outgoing synapses per neuron.
    pub max_fanout: u16,
    /// Maximum incoming synapses per neuron.
    pub max_fanin: u16,
    /// Default synapse magnitude for new connections.
    pub default_magnitude: u8,
    /// Zone assignment strategy.
    pub zone_strategy: ZoneStrategy,
}

/// How to assign dendritic zones to new synapses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoneStrategy {
    /// All synapses target feedforward zone.
    AllFeedforward,
    /// Assign zone based on relative voxel z-position:
    /// - source.z < target.z → Feedforward (bottom-up)
    /// - source.z == target.z → Context (lateral)
    /// - source.z > target.z → Feedback (top-down)
    ZLayer,
    /// Same voxel → Context, different voxel → Feedforward.
    /// Oscillators always target Context (entrainment).
    LocalContext,
}

impl Default for UnifiedWiringConfig {
    fn default() -> Self {
        Self {
            max_distance_sq: 256 * 4, // ~2 voxels
            max_fanout: 8,
            max_fanin: 20,
            default_magnitude: 100,
            zone_strategy: ZoneStrategy::ZLayer,
        }
    }
}

/// Determine the dendritic zone for a synapse based on source and target positions.
fn assign_zone(
    source: &UnifiedNeuron,
    target: &UnifiedNeuron,
    strategy: ZoneStrategy,
) -> DendriticZone {
    match strategy {
        ZoneStrategy::AllFeedforward => DendriticZone::Feedforward,
        ZoneStrategy::ZLayer => {
            let sz = source.position.voxel.2;
            let tz = target.position.voxel.2;
            if sz < tz {
                DendriticZone::Feedforward
            } else if sz == tz {
                DendriticZone::Context
            } else {
                DendriticZone::Feedback
            }
        }
        ZoneStrategy::LocalContext => {
            if source.nuclei.is_oscillator() {
                DendriticZone::Context
            } else if source.position.voxel == target.position.voxel {
                DendriticZone::Context
            } else {
                DendriticZone::Feedforward
            }
        }
    }
}

/// Wire neurons by spatial proximity using the voxel grid for acceleration.
///
/// For each non-motor neuron with a live axon, finds nearby neurons within
/// `max_distance_sq` using the grid's local neighborhood lookup, then creates
/// zone-aware synapses up to fanout/fanin caps.
///
/// Returns a fully indexed `UnifiedSynapseStore`.
pub fn wire_by_proximity(
    neurons: &[UnifiedNeuron],
    grid: &VoxelGrid,
    config: &UnifiedWiringConfig,
) -> UnifiedSynapseStore {
    let n = neurons.len();
    let mut store = UnifiedSynapseStore::with_capacity(n * config.max_fanout as usize, n);
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

        let voxel = neurons[i].position.voxel;

        // Get candidates from local neighborhood (self + 26-connected)
        let candidate_indices = grid.local_neuron_indices(voxel.0, voxel.1, voxel.2);

        // Compute distances and filter
        let mut candidates: Vec<(u32, u64)> = candidate_indices
            .into_iter()
            .filter(|&j| j != i as u32)
            .filter_map(|j| {
                let dist_sq = neurons[i].position.distance_sq(&neurons[j as usize].position);
                if dist_sq <= config.max_distance_sq {
                    Some((j, dist_sq))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance (nearest first)
        candidates.sort_by_key(|&(_, d)| d);

        // Create synapses
        for (j, _dist_sq) in candidates {
            let j_idx = j as usize;
            if fanout[i] >= config.max_fanout {
                break;
            }
            if fanin[j_idx] >= config.max_fanin {
                continue;
            }

            let zone = assign_zone(&neurons[i], &neurons[j_idx], config.zone_strategy);

            let syn = if neurons[i].nuclei.is_inhibitory() {
                UnifiedSynapse::inhibitory(
                    i as u32,
                    j,
                    zone,
                    config.default_magnitude,
                    0, // delay computed from distance at runtime
                )
            } else {
                UnifiedSynapse::excitatory(
                    i as u32,
                    j,
                    zone,
                    config.default_magnitude,
                    0,
                )
            };

            store.add(syn);
            fanout[i] += 1;
            fanin[j_idx] += 1;
        }
    }

    store.rebuild_index(n as u32);
    store
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Axon;
    use crate::unified::neuron::VoxelPosition;

    fn pos(x: u16, y: u16, z: u16) -> VoxelPosition {
        VoxelPosition::at_center((x, y, z))
    }

    fn grid_dims(x: u16, y: u16, z: u16) -> super::super::grid::GridDims {
        super::super::grid::GridDims { x, y, z }
    }

    #[test]
    fn basic_wiring() {
        // Two adjacent neurons in same voxel
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig {
            max_distance_sq: 256, // within same voxel
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &grid, &config);
        // Both neurons at same center → distance = 0, should wire
        assert_eq!(store.outgoing(0).len(), 1);
        assert_eq!(store.outgoing(0)[0].target, 1);
    }

    #[test]
    fn out_of_range() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(10, 0, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig {
            max_distance_sq: 256,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &grid, &config);
        assert_eq!(store.outgoing(0).len(), 0);
    }

    #[test]
    fn motor_neurons_dont_project() {
        let mut neurons = vec![
            UnifiedNeuron::motor_at(pos(0, 0, 0), 0, 1),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig::default();

        let store = wire_by_proximity(&neurons, &grid, &config);
        // Motor neuron is at index 0 or 1 (after sort). Find it.
        let motor_idx = neurons.iter().position(|n| n.nuclei.is_motor()).unwrap();
        assert_eq!(store.outgoing(motor_idx as u32).len(), 0);
    }

    #[test]
    fn motor_neurons_receive_input() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::motor_at(pos(0, 0, 0), 0, 1),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig::default();

        let store = wire_by_proximity(&neurons, &grid, &config);
        // Pyramidal should project to motor
        let pyr_idx = neurons.iter().position(|n| !n.nuclei.is_motor()).unwrap();
        assert_eq!(store.outgoing(pyr_idx as u32).len(), 1);
    }

    #[test]
    fn fanout_limit() {
        // One source, many targets in same voxel
        let mut neurons = vec![UnifiedNeuron::pyramidal_at(pos(0, 0, 0))];
        for _ in 0..20 {
            neurons.push(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));
        }

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig {
            max_fanout: 4,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &grid, &config);
        // Each neuron should have at most 4 outgoing
        for i in 0..neurons.len() {
            assert!(
                store.outgoing(i as u32).len() <= 4,
                "neuron {} has {} outgoing, expected <= 4",
                i,
                store.outgoing(i as u32).len()
            );
        }
    }

    #[test]
    fn inhibitory_wiring() {
        let mut neurons = vec![
            UnifiedNeuron::interneuron_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig::default();

        let store = wire_by_proximity(&neurons, &grid, &config);
        let inh_idx = neurons.iter().position(|n| n.nuclei.is_inhibitory()).unwrap();
        let out = store.outgoing(inh_idx as u32);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_inhibitory());
    }

    #[test]
    fn z_layer_zone_assignment() {
        // Source at z=0, target at z=1 → Feedforward
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 1)),
        ];

        let grid = VoxelGrid::build_with_dims(&mut neurons, grid_dims(1, 1, 2));
        let config = UnifiedWiringConfig {
            zone_strategy: ZoneStrategy::ZLayer,
            max_distance_sq: 256 * 4,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &grid, &config);

        // Find the neuron at z=0 (after sort)
        let z0_idx = neurons.iter().position(|n| n.position.voxel.2 == 0).unwrap();
        let z1_idx = neurons.iter().position(|n| n.position.voxel.2 == 1).unwrap();

        let out = store.outgoing(z0_idx as u32);
        if !out.is_empty() {
            assert_eq!(out[0].zone, DendriticZone::Feedforward, "z=0 → z=1 should be feedforward");
        }

        // z=1 → z=0 should be Feedback
        let out_fb = store.outgoing(z1_idx as u32);
        if !out_fb.is_empty() {
            assert_eq!(out_fb[0].zone, DendriticZone::Feedback, "z=1 → z=0 should be feedback");
        }
    }

    #[test]
    fn same_z_is_context() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(1, 0, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig {
            zone_strategy: ZoneStrategy::ZLayer,
            max_distance_sq: 256 * 4,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &grid, &config);
        // Same z → Context
        let out = store.outgoing(0);
        if !out.is_empty() {
            assert_eq!(out[0].zone, DendriticZone::Context, "same z should be context");
        }
    }

    #[test]
    fn local_context_strategy() {
        // Same voxel → Context, different voxel → Feedforward
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)), // same voxel
            UnifiedNeuron::pyramidal_at(pos(1, 0, 0)), // different voxel
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig {
            zone_strategy: ZoneStrategy::LocalContext,
            max_distance_sq: 256 * 4,
            ..Default::default()
        };

        let store = wire_by_proximity(&neurons, &grid, &config);

        // Find neuron at (0,0,0) — it should have both context and feedforward targets
        for i in 0..neurons.len() {
            for syn in store.outgoing(i as u32) {
                let src_voxel = neurons[i].position.voxel;
                let tgt_voxel = neurons[syn.target as usize].position.voxel;
                if src_voxel == tgt_voxel {
                    assert_eq!(syn.zone, DendriticZone::Context, "same voxel should be context");
                } else {
                    assert_eq!(syn.zone, DendriticZone::Feedforward, "different voxel should be feedforward");
                }
            }
        }
    }

    #[test]
    fn dead_axon_skipped() {
        let mut source = UnifiedNeuron::pyramidal_at(pos(0, 0, 0));
        source.axon = Axon::default();
        source.axon.health = 0; // dead

        let mut neurons = vec![
            source,
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig::default();

        let store = wire_by_proximity(&neurons, &grid, &config);
        // Find the dead-axon neuron
        let dead_idx = neurons.iter().position(|n| n.axon.health == 0).unwrap();
        assert_eq!(store.outgoing(dead_idx as u32).len(), 0);
    }

    #[test]
    fn delay_is_zero() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
            UnifiedNeuron::pyramidal_at(pos(0, 0, 0)),
        ];

        let grid = VoxelGrid::build(&mut neurons);
        let config = UnifiedWiringConfig::default();

        let store = wire_by_proximity(&neurons, &grid, &config);
        for i in 0..neurons.len() {
            for syn in store.outgoing(i as u32) {
                assert_eq!(syn.delay_us, 0, "delay should be 0 (computed at runtime)");
            }
        }
    }
}
