#![allow(deprecated)]
//! Incubation — grow a neuron pool from an imaginal disc.
//!
//! `incubate()` takes a disc blueprint, a seed, and a neuron count, then:
//! 1. Seeds neurons in the voxel grid according to the disc's archetype
//! 2. Assigns nuclei biased by the disc's distribution
//! 3. Wires synapses using archetype-specific zone rules
//! 4. Runs oscillator-only cascade settling to let dynamics reach equilibrium
//! 5. Prunes dormant synapses
//!
//! The result is a settled pool with topology-appropriate structure.

use super::cascade::{CascadeConfig, CascadeEngine};
use super::disc::{ImaginalDisc, RegionArchetype};
use super::grid::{GridDims, VoxelGrid};
use super::neuron::{UnifiedNeuron, VoxelPosition};
use super::synapse::{UnifiedSynapse, UnifiedSynapseStore};
use super::zone::DendriticZone;

/// Result of incubation — a settled pool ready for external input.
pub struct IncubatedPool {
    /// All neurons (sorted by voxel position after grid build).
    pub neurons: Vec<UnifiedNeuron>,
    /// All synapses (zone-aware, CSR indexed).
    pub synapses: UnifiedSynapseStore,
    /// Spatial grid index.
    pub grid: VoxelGrid,
    /// The disc that produced this pool.
    pub disc: ImaginalDisc,
    /// Number of settling steps run during incubation.
    pub settling_steps: u32,
    /// Number of synapses pruned during incubation.
    pub pruned_count: usize,
}

/// Incubation configuration.
#[derive(Clone, Copy, Debug)]
pub struct IncubateConfig {
    /// Number of settling cascade steps (oscillator-only).
    pub settling_steps: u32,
    /// Time per settling step in microseconds.
    pub step_duration_us: u64,
    /// Whether to prune dormant synapses after settling.
    pub prune_after_settling: bool,
}

impl Default for IncubateConfig {
    fn default() -> Self {
        Self {
            settling_steps: 50,
            step_duration_us: 10_000, // 10ms
            prune_after_settling: true,
        }
    }
}

/// Deterministic LCG for reproducible placement and assignment.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    /// Next u32 in [0, 2^32).
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }

    /// Random value in [0, max).
    fn next_range(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        self.next_u32() % max
    }

    /// Random value in [min, max].
    fn next_range_inclusive(&mut self, min: u32, max: u32) -> u32 {
        if min >= max {
            return min;
        }
        min + self.next_range(max - min + 1)
    }
}

/// Grow a neuron pool from an imaginal disc.
///
/// The `seed` controls all randomness — same seed + same disc = same pool.
pub fn incubate(
    disc: &ImaginalDisc,
    seed: u64,
    neuron_count: u32,
    config: &IncubateConfig,
) -> IncubatedPool {
    let mut rng = Rng::new(seed);

    // 1. Seed neurons in the voxel grid
    let mut neurons = seed_neurons(disc, neuron_count, &mut rng);

    // 2. Build spatial grid (sorts neurons by voxel)
    let grid_dims = GridDims {
        x: disc.grid_dims.0,
        y: disc.grid_dims.1,
        z: disc.grid_dims.2,
    };
    let grid = VoxelGrid::build_with_dims(&mut neurons, grid_dims);

    // 3. Wire synapses using archetype rules
    let synapses = wire_archetype(&neurons, &grid, disc, &mut rng);

    // 4. Settle with oscillator-only cascade
    let mut engine = CascadeEngine::with_network(
        neurons,
        synapses,
        CascadeConfig {
            coincidence_boost: 0.3,
            ..CascadeConfig::default()
        },
    );

    for step in 0..config.settling_steps {
        let time = step as u64 * config.step_duration_us;
        engine.sim_time_us = time;
        engine.check_oscillators();
        engine.run_until(time + config.step_duration_us);
        engine.decay_traces();
        engine.recover_stamina(config.step_duration_us);
    }

    // 5. Prune dormant synapses
    let pruned_count = if config.prune_after_settling {
        engine.synapses.prune_dormant()
    } else {
        0
    };

    // Rebuild index after pruning
    if pruned_count > 0 {
        engine.synapses.rebuild_index(engine.neurons.len() as u32);
    }

    IncubatedPool {
        neurons: engine.neurons,
        synapses: engine.synapses,
        grid,
        disc: disc.clone(),
        settling_steps: config.settling_steps,
        pruned_count,
    }
}

/// Seed neurons with positions and nuclei based on the disc.
fn seed_neurons(
    disc: &ImaginalDisc,
    count: u32,
    rng: &mut Rng,
) -> Vec<UnifiedNeuron> {
    let mut neurons = Vec::with_capacity(count as usize);
    let (gx, gy, gz) = disc.grid_dims;
    let dist = &disc.distribution;
    let osc_range = disc.wiring.oscillator_period_range;

    for i in 0..count {
        // Position: distribute across the grid
        let vx = (i as u16 % gx).min(gx.saturating_sub(1));
        let vy = ((i as u16 / gx) % gy).min(gy.saturating_sub(1));
        let vz = ((i as u16 / (gx * gy)) % gz).min(gz.saturating_sub(1));

        // Local offset: random within voxel
        let lx = rng.next_range(16) as u8;
        let ly = rng.next_range(16) as u8;
        let lz = rng.next_range(16) as u8;

        let pos = VoxelPosition::new((vx, vy, vz), (lx, ly, lz));

        // Assign nuclei based on distribution
        let roll = rng.next_range(100) as u8;
        let neuron = assign_nuclei(pos, roll, dist, osc_range, rng);
        neurons.push(neuron);
    }

    neurons
}

/// Assign nuclei to a neuron based on a random roll and the distribution.
fn assign_nuclei(
    pos: VoxelPosition,
    roll: u8,
    dist: &super::disc::NucleiDistribution,
    osc_range: (u32, u32),
    rng: &mut Rng,
) -> UnifiedNeuron {
    let mut threshold = dist.pyramidal;
    if roll < threshold {
        return UnifiedNeuron::pyramidal_at(pos);
    }
    threshold += dist.interneuron;
    if roll < threshold {
        return UnifiedNeuron::interneuron_at(pos);
    }
    threshold += dist.gate;
    if roll < threshold {
        return UnifiedNeuron::gate_at(pos);
    }
    threshold += dist.relay;
    if roll < threshold {
        return UnifiedNeuron::relay_at(pos);
    }
    threshold += dist.oscillator;
    if roll < threshold {
        let period = rng.next_range_inclusive(osc_range.0, osc_range.1);
        return UnifiedNeuron::oscillator_at(pos, period);
    }
    // Remaining: memory
    let bank_id = rng.next_range(256) as u16;
    UnifiedNeuron::memory_at(pos, bank_id)
}

/// Wire neurons using archetype-specific rules.
fn wire_archetype(
    neurons: &[UnifiedNeuron],
    grid: &VoxelGrid,
    disc: &ImaginalDisc,
    rng: &mut Rng,
) -> UnifiedSynapseStore {
    let n = neurons.len();
    let rules = &disc.wiring;
    let mut store = UnifiedSynapseStore::with_capacity(n * rules.max_fanout as usize, n);
    let mut fanout = vec![0u16; n];
    let mut fanin = vec![0u16; n];

    // Compute total zone probability for weighted selection
    let total_prob: u32 = rules.zone_biases.iter().map(|b| b.probability as u32).sum();

    for i in 0..n {
        if neurons[i].nuclei.is_motor() {
            continue;
        }
        if !neurons[i].axon.is_alive() {
            continue;
        }

        let voxel = neurons[i].position.voxel;
        let candidate_indices = grid.local_neuron_indices(voxel.0, voxel.1, voxel.2);

        // Compute distances and filter
        let mut candidates: Vec<(u32, u64)> = candidate_indices
            .into_iter()
            .filter(|&j| j != i as u32)
            .filter_map(|j| {
                let dist_sq = neurons[i].position.distance_sq(&neurons[j as usize].position);
                if dist_sq <= rules.max_distance_sq {
                    Some((j, dist_sq))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by_key(|&(_, d)| d);

        // Dense lateral: for hippocampal CA3 and brainstem, same-z neurons get
        // extra context connections
        let dense_z = if rules.dense_lateral {
            Some(neurons[i].position.voxel.2)
        } else {
            None
        };

        for (j, _dist_sq) in candidates {
            let j_idx = j as usize;
            if fanout[i] >= rules.max_fanout {
                break;
            }
            if fanin[j_idx] >= rules.max_fanin {
                continue;
            }

            // Determine zone
            let zone = select_zone(
                &neurons[i],
                &neurons[j_idx],
                disc.archetype,
                dense_z,
                rng,
                &rules.zone_biases,
                total_prob,
            );

            // Get magnitude from zone bias
            let magnitude = rules.zone_biases[zone.index()].magnitude;

            let syn = if neurons[i].nuclei.is_inhibitory() {
                UnifiedSynapse::inhibitory(i as u32, j, zone, magnitude, 0)
            } else {
                UnifiedSynapse::excitatory(i as u32, j, zone, magnitude, 0)
            };

            store.add(syn);
            fanout[i] += 1;
            fanin[j_idx] += 1;
        }
    }

    store.rebuild_index(n as u32);
    store
}

/// Select a dendritic zone for a synapse based on archetype rules.
fn select_zone(
    source: &UnifiedNeuron,
    target: &UnifiedNeuron,
    archetype: RegionArchetype,
    dense_z: Option<u16>,
    rng: &mut Rng,
    zone_biases: &[super::disc::ZoneBias; 3],
    total_prob: u32,
) -> DendriticZone {
    // Oscillators always target context (entrainment)
    if source.nuclei.is_oscillator() {
        return DendriticZone::Context;
    }

    match archetype {
        RegionArchetype::Cortical => {
            // Columnar: z-layer determines zone
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
        RegionArchetype::Hippocampal => {
            // Pipeline: lower z → higher z = feedforward
            // Same z with dense_lateral = context (CA3 recurrence)
            // Higher z → lower z = feedback
            let sz = source.position.voxel.2;
            let tz = target.position.voxel.2;
            if sz == tz && dense_z == Some(sz) {
                DendriticZone::Context
            } else if sz < tz {
                DendriticZone::Feedforward
            } else if sz > tz {
                DendriticZone::Feedback
            } else {
                DendriticZone::Context
            }
        }
        _ => {
            // Probabilistic zone selection based on biases
            let roll = rng.next_range(total_prob);
            let mut accum = 0u32;
            for bias in zone_biases {
                accum += bias.probability as u32;
                if roll < accum {
                    return bias.zone;
                }
            }
            DendriticZone::Feedforward
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incubate_cortical() {
        let disc = ImaginalDisc::new(RegionArchetype::Cortical, 3, 3);
        let config = IncubateConfig {
            settling_steps: 10,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };

        let pool = incubate(&disc, 42, 64, &config);

        assert_eq!(pool.neurons.len(), 64);
        assert!(pool.synapses.len() > 0, "should have synapses");
        assert_eq!(pool.settling_steps, 10);

        // Should have multiple nuclei types
        let pyramidal_count = pool.neurons.iter().filter(|n| n.nuclei.is_excitatory() && !n.nuclei.is_oscillator()).count();
        let inhibitory_count = pool.neurons.iter().filter(|n| n.nuclei.is_inhibitory()).count();
        assert!(pyramidal_count > 0, "should have pyramidal neurons");
        assert!(inhibitory_count > 0, "should have inhibitory neurons");
    }

    #[test]
    fn incubate_thalamic() {
        let disc = ImaginalDisc::new(RegionArchetype::Thalamic, 3, 3);
        let config = IncubateConfig {
            settling_steps: 5,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };

        let pool = incubate(&disc, 99, 48, &config);
        assert_eq!(pool.neurons.len(), 48);
        assert!(pool.synapses.len() > 0);

        // Thalamic should have gate/relay neurons
        let gate_count = pool.neurons.iter().filter(|n| n.nuclei.leak >= 200 && n.nuclei.is_excitatory()).count();
        assert!(gate_count > 0, "thalamic should have gate neurons");
    }

    #[test]
    fn incubate_hippocampal() {
        let disc = ImaginalDisc::new(RegionArchetype::Hippocampal, 3, 3);
        let config = IncubateConfig {
            settling_steps: 5,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };

        let pool = incubate(&disc, 7, 80, &config);
        assert_eq!(pool.neurons.len(), 80);
        assert!(pool.synapses.len() > 0);

        // Should have memory neurons
        let memory_count = pool.neurons.iter().filter(|n| n.nuclei.is_memory()).count();
        assert!(memory_count > 0, "hippocampal should have memory neurons");
    }

    #[test]
    fn incubate_brainstem() {
        let disc = ImaginalDisc::new(RegionArchetype::Brainstem, 3, 3);
        let config = IncubateConfig {
            settling_steps: 5,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };

        let pool = incubate(&disc, 123, 40, &config);
        assert_eq!(pool.neurons.len(), 40);

        // Brainstem should be oscillator-heavy
        let osc_count = pool.neurons.iter().filter(|n| n.nuclei.is_oscillator()).count();
        assert!(
            osc_count >= 10,
            "brainstem should have many oscillators, got {}",
            osc_count
        );
    }

    #[test]
    fn deterministic_incubation() {
        let disc = ImaginalDisc::new(RegionArchetype::Cortical, 2, 2);
        let config = IncubateConfig {
            settling_steps: 5,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };

        let pool1 = incubate(&disc, 42, 32, &config);
        let pool2 = incubate(&disc, 42, 32, &config);

        // Same seed → same topology
        assert_eq!(pool1.neurons.len(), pool2.neurons.len());
        assert_eq!(pool1.synapses.len(), pool2.synapses.len());
    }

    #[test]
    fn different_seeds_different_pools() {
        let disc = ImaginalDisc::new(RegionArchetype::Cortical, 2, 2);
        let config = IncubateConfig {
            settling_steps: 5,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };

        let pool1 = incubate(&disc, 42, 32, &config);
        let pool2 = incubate(&disc, 99, 32, &config);

        // Different seeds should produce different wiring
        // (neurons are sorted by voxel, so positions might be similar,
        // but synapse targets should differ)
        let synapses1: Vec<(u32, u32)> = pool1.synapses.iter().map(|s| (s.source, s.target)).collect();
        let synapses2: Vec<(u32, u32)> = pool2.synapses.iter().map(|s| (s.source, s.target)).collect();
        assert_ne!(synapses1, synapses2, "different seeds should produce different wiring");
    }

    #[test]
    fn cortical_zone_assignment() {
        // Cortical should use z-layer based zone assignment
        let disc = ImaginalDisc::new(RegionArchetype::Cortical, 2, 2);
        let config = IncubateConfig {
            settling_steps: 0,
            step_duration_us: 0,
            prune_after_settling: false,
        };

        let pool = incubate(&disc, 42, 64, &config);

        // Check that synapses crossing z-layers have appropriate zones
        let mut ff_count = 0;
        let mut ctx_count = 0;
        for syn in pool.synapses.iter() {
            if !syn.is_active() {
                continue;
            }
            match syn.zone {
                DendriticZone::Feedforward => ff_count += 1,
                DendriticZone::Context => ctx_count += 1,
                DendriticZone::Feedback => {}
            }
        }

        // Cortical should have all three zones represented
        assert!(ff_count > 0, "cortical should have feedforward synapses");
        assert!(ctx_count > 0, "cortical should have context synapses");
    }

    #[test]
    fn pruning_removes_dormant() {
        let disc = ImaginalDisc::new(RegionArchetype::Cortical, 2, 2);
        let config = IncubateConfig {
            settling_steps: 20,
            step_duration_us: 10_000,
            prune_after_settling: true,
        };

        let pool = incubate(&disc, 42, 32, &config);

        // All remaining synapses should be active
        assert_eq!(
            pool.synapses.count_dormant(),
            0,
            "pruning should remove all dormant synapses"
        );
    }

    #[test]
    fn for_neuron_count_factory() {
        let disc = ImaginalDisc::for_neuron_count(RegionArchetype::Cerebellar, 512);
        assert_eq!(disc.archetype, RegionArchetype::Cerebellar);
        assert_eq!(disc.z_layers, 1); // cerebellar is flat
        assert!(disc.grid_dims.0 * disc.grid_dims.1 * disc.grid_dims.2 > 0);
    }
}
