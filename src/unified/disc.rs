//! Imaginal Discs — blueprints for growing region-specific neuron pools.
//!
//! Borrowed from developmental biology: imaginal discs are undifferentiated
//! tissue that develops into specific organs. Here, each disc specifies a
//! region archetype, nuclei distribution, grid dimensions, and wiring biases.
//! The `incubate` module uses discs to seed, wire, and settle pools.

use super::zone::DendriticZone;

/// Region archetype — determines topology, nuclei distribution, and wiring rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegionArchetype {
    /// Cortical: 5 z-layers (columnar), pyramidal-heavy, vertical feedforward,
    /// lateral context, top-down feedback.
    Cortical,
    /// Thalamic: flat clusters, gate/relay-heavy, bidirectional projection.
    Thalamic,
    /// Hippocampal: pipeline topology (DG→CA3→CA1→EC), dense recurrence in CA3.
    Hippocampal,
    /// Basal ganglia: direct/indirect pathways, gate-heavy, competition dynamics.
    BasalGanglia,
    /// Cerebellar: parallel fiber topology, oscillator-heavy, motor correction.
    Cerebellar,
    /// Brainstem: dense small nuclei, oscillator-dominant, broad projection.
    Brainstem,
}

/// Target percentages for nuclei types within a region.
///
/// All values are percentages (0-100). Must sum to 100.
/// Sensory/motor neurons are placed explicitly by the caller, not by distribution.
#[derive(Clone, Copy, Debug)]
pub struct NucleiDistribution {
    /// Pyramidal (excitatory projection) neurons.
    pub pyramidal: u8,
    /// Interneuron (inhibitory, local circuit) neurons.
    pub interneuron: u8,
    /// Gate (coincidence detection, high-leak) neurons.
    pub gate: u8,
    /// Relay (fast pass-through) neurons.
    pub relay: u8,
    /// Oscillator neurons. Period determined by archetype.
    pub oscillator: u8,
    /// Memory (bank-coupled) neurons.
    pub memory: u8,
}

impl NucleiDistribution {
    /// Default distribution for an archetype.
    pub fn for_archetype(archetype: RegionArchetype) -> Self {
        match archetype {
            RegionArchetype::Cortical => Self {
                pyramidal: 60,
                interneuron: 20,
                gate: 5,
                relay: 5,
                oscillator: 5,
                memory: 5,
            },
            RegionArchetype::Thalamic => Self {
                pyramidal: 10,
                interneuron: 15,
                gate: 35,
                relay: 30,
                oscillator: 5,
                memory: 5,
            },
            RegionArchetype::Hippocampal => Self {
                pyramidal: 50,
                interneuron: 15,
                gate: 5,
                relay: 5,
                oscillator: 5,
                memory: 20,
            },
            RegionArchetype::BasalGanglia => Self {
                pyramidal: 20,
                interneuron: 30,
                gate: 30,
                relay: 10,
                oscillator: 5,
                memory: 5,
            },
            RegionArchetype::Cerebellar => Self {
                pyramidal: 25,
                interneuron: 20,
                gate: 5,
                relay: 10,
                oscillator: 30,
                memory: 10,
            },
            RegionArchetype::Brainstem => Self {
                pyramidal: 15,
                interneuron: 10,
                gate: 10,
                relay: 15,
                oscillator: 40,
                memory: 10,
            },
        }
    }

    /// Validate that distribution sums to 100.
    pub fn is_valid(&self) -> bool {
        let sum = self.pyramidal as u16
            + self.interneuron as u16
            + self.gate as u16
            + self.relay as u16
            + self.oscillator as u16
            + self.memory as u16;
        sum == 100
    }
}

/// Wiring bias for a specific zone — controls connection probability and magnitude.
#[derive(Clone, Copy, Debug)]
pub struct ZoneBias {
    /// Which zone this bias configures.
    pub zone: DendriticZone,
    /// Relative probability weight for connections targeting this zone (0-255).
    pub probability: u8,
    /// Default magnitude for synapses targeting this zone.
    pub magnitude: u8,
}

/// Archetype-specific wiring rules.
#[derive(Clone, Debug)]
pub struct WiringRules {
    /// Maximum distance squared for connections within this archetype.
    pub max_distance_sq: u64,
    /// Maximum fanout per neuron.
    pub max_fanout: u16,
    /// Maximum fanin per neuron.
    pub max_fanin: u16,
    /// Zone biases (probability and magnitude per zone).
    pub zone_biases: [ZoneBias; 3],
    /// Whether to use dense recurrent (context) wiring within same z-layer.
    pub dense_lateral: bool,
    /// Oscillator period range (min_us, max_us) for this archetype.
    pub oscillator_period_range: (u32, u32),
}

impl WiringRules {
    /// Default wiring rules for an archetype.
    pub fn for_archetype(archetype: RegionArchetype) -> Self {
        match archetype {
            RegionArchetype::Cortical => Self {
                max_distance_sq: 256 * 3, // ~1.7 voxels
                max_fanout: 8,
                max_fanin: 20,
                zone_biases: [
                    ZoneBias { zone: DendriticZone::Feedforward, probability: 150, magnitude: 120 },
                    ZoneBias { zone: DendriticZone::Context, probability: 80, magnitude: 80 },
                    ZoneBias { zone: DendriticZone::Feedback, probability: 40, magnitude: 60 },
                ],
                dense_lateral: false,
                oscillator_period_range: (8_000, 20_000), // 50-125 Hz (gamma/beta)
            },
            RegionArchetype::Thalamic => Self {
                max_distance_sq: 256 * 4,
                max_fanout: 12,
                max_fanin: 16,
                zone_biases: [
                    ZoneBias { zone: DendriticZone::Feedforward, probability: 120, magnitude: 140 },
                    ZoneBias { zone: DendriticZone::Context, probability: 60, magnitude: 60 },
                    ZoneBias { zone: DendriticZone::Feedback, probability: 120, magnitude: 100 },
                ],
                dense_lateral: false,
                oscillator_period_range: (10_000, 25_000), // 40-100 Hz
            },
            RegionArchetype::Hippocampal => Self {
                max_distance_sq: 256 * 5,
                max_fanout: 10,
                max_fanin: 25,
                zone_biases: [
                    ZoneBias { zone: DendriticZone::Feedforward, probability: 130, magnitude: 100 },
                    ZoneBias { zone: DendriticZone::Context, probability: 140, magnitude: 120 },
                    ZoneBias { zone: DendriticZone::Feedback, probability: 60, magnitude: 80 },
                ],
                dense_lateral: true, // CA3 recurrence
                oscillator_period_range: (100_000, 250_000), // 4-10 Hz (theta)
            },
            RegionArchetype::BasalGanglia => Self {
                max_distance_sq: 256 * 4,
                max_fanout: 10,
                max_fanin: 20,
                zone_biases: [
                    ZoneBias { zone: DendriticZone::Feedforward, probability: 160, magnitude: 130 },
                    ZoneBias { zone: DendriticZone::Context, probability: 40, magnitude: 60 },
                    ZoneBias { zone: DendriticZone::Feedback, probability: 80, magnitude: 100 },
                ],
                dense_lateral: false,
                oscillator_period_range: (15_000, 50_000), // 20-67 Hz (beta)
            },
            RegionArchetype::Cerebellar => Self {
                max_distance_sq: 256 * 6,
                max_fanout: 16,
                max_fanin: 30,
                zone_biases: [
                    ZoneBias { zone: DendriticZone::Feedforward, probability: 140, magnitude: 100 },
                    ZoneBias { zone: DendriticZone::Context, probability: 100, magnitude: 80 },
                    ZoneBias { zone: DendriticZone::Feedback, probability: 40, magnitude: 60 },
                ],
                dense_lateral: false,
                oscillator_period_range: (5_000, 15_000), // 67-200 Hz (fast)
            },
            RegionArchetype::Brainstem => Self {
                max_distance_sq: 256 * 8,
                max_fanout: 20,
                max_fanin: 30,
                zone_biases: [
                    ZoneBias { zone: DendriticZone::Feedforward, probability: 100, magnitude: 100 },
                    ZoneBias { zone: DendriticZone::Context, probability: 100, magnitude: 100 },
                    ZoneBias { zone: DendriticZone::Feedback, probability: 80, magnitude: 80 },
                ],
                dense_lateral: true, // dense interconnection
                oscillator_period_range: (500_000, 2_000_000), // 0.5-2 Hz (slow)
            },
        }
    }
}

/// An imaginal disc — blueprint for growing a region-specific neuron pool.
#[derive(Clone, Debug)]
pub struct ImaginalDisc {
    /// Region archetype.
    pub archetype: RegionArchetype,
    /// Target nuclei distribution (percentages).
    pub distribution: NucleiDistribution,
    /// Wiring rules for this archetype.
    pub wiring: WiringRules,
    /// Grid dimensions for the pool (x × y × z voxels).
    pub grid_dims: (u16, u16, u16),
    /// Number of z-layers (for cortical: 5 canonical layers).
    /// For non-cortical archetypes, this is typically 1-2.
    pub z_layers: u16,
}

impl ImaginalDisc {
    /// Create a disc with default parameters for an archetype.
    pub fn new(archetype: RegionArchetype, grid_x: u16, grid_y: u16) -> Self {
        let z_layers = match archetype {
            RegionArchetype::Cortical => 5,
            RegionArchetype::Hippocampal => 4, // DG, CA3, CA1, EC
            RegionArchetype::BasalGanglia => 3, // striatum, GPe/STN, GPi
            _ => 1,
        };

        Self {
            archetype,
            distribution: NucleiDistribution::for_archetype(archetype),
            wiring: WiringRules::for_archetype(archetype),
            grid_dims: (grid_x, grid_y, z_layers),
            z_layers,
        }
    }

    /// Create with custom neuron count constraint.
    /// Grid dimensions are computed to fit approximately `neuron_count` neurons
    /// with ~8 neurons per voxel.
    pub fn for_neuron_count(archetype: RegionArchetype, neuron_count: u32) -> Self {
        let neurons_per_voxel = 8u32;
        let total_voxels = (neuron_count + neurons_per_voxel - 1) / neurons_per_voxel;

        let z_layers = match archetype {
            RegionArchetype::Cortical => 5u16,
            RegionArchetype::Hippocampal => 4,
            RegionArchetype::BasalGanglia => 3,
            _ => 1,
        };

        let xy_voxels = (total_voxels / z_layers as u32).max(1);
        let side = (xy_voxels as f32).sqrt().ceil() as u16;

        Self::new(archetype, side.max(2), side.max(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_distributions_valid() {
        let archetypes = [
            RegionArchetype::Cortical,
            RegionArchetype::Thalamic,
            RegionArchetype::Hippocampal,
            RegionArchetype::BasalGanglia,
            RegionArchetype::Cerebellar,
            RegionArchetype::Brainstem,
        ];

        for arch in &archetypes {
            let dist = NucleiDistribution::for_archetype(*arch);
            assert!(dist.is_valid(), "{:?} distribution doesn't sum to 100", arch);
        }
    }

    #[test]
    fn cortical_pyramidal_dominant() {
        let dist = NucleiDistribution::for_archetype(RegionArchetype::Cortical);
        assert!(dist.pyramidal >= 50, "cortical should be pyramidal-heavy");
    }

    #[test]
    fn thalamic_gate_relay_dominant() {
        let dist = NucleiDistribution::for_archetype(RegionArchetype::Thalamic);
        assert!(
            dist.gate + dist.relay >= 50,
            "thalamic should be gate/relay-heavy"
        );
    }

    #[test]
    fn brainstem_oscillator_dominant() {
        let dist = NucleiDistribution::for_archetype(RegionArchetype::Brainstem);
        assert!(
            dist.oscillator >= 30,
            "brainstem should be oscillator-heavy"
        );
    }

    #[test]
    fn hippocampal_memory_present() {
        let dist = NucleiDistribution::for_archetype(RegionArchetype::Hippocampal);
        assert!(dist.memory >= 15, "hippocampal should have memory neurons");
    }

    #[test]
    fn disc_for_neuron_count() {
        let disc = ImaginalDisc::for_neuron_count(RegionArchetype::Cortical, 256);
        assert_eq!(disc.z_layers, 5);
        // 256 neurons / 8 per voxel = 32 voxels / 5 layers = ~6.4 → ceil(sqrt(7)) = 3
        assert!(disc.grid_dims.0 >= 2);
        assert!(disc.grid_dims.1 >= 2);
        assert_eq!(disc.grid_dims.2, 5);
    }

    #[test]
    fn disc_default_cortical() {
        let disc = ImaginalDisc::new(RegionArchetype::Cortical, 4, 4);
        assert_eq!(disc.archetype, RegionArchetype::Cortical);
        assert_eq!(disc.z_layers, 5);
        assert_eq!(disc.grid_dims, (4, 4, 5));
        assert!(disc.distribution.is_valid());
    }

    #[test]
    fn wiring_rules_per_archetype() {
        for arch in &[
            RegionArchetype::Cortical,
            RegionArchetype::Thalamic,
            RegionArchetype::Hippocampal,
            RegionArchetype::BasalGanglia,
            RegionArchetype::Cerebellar,
            RegionArchetype::Brainstem,
        ] {
            let rules = WiringRules::for_archetype(*arch);
            assert!(rules.max_fanout > 0);
            assert!(rules.max_fanin > 0);
            assert!(rules.oscillator_period_range.0 < rules.oscillator_period_range.1);
        }
    }
}
