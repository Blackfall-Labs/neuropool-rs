//! Pool inspection and diagnostics.

use crate::neuron::{NeuronType, flags};
use crate::pool::{NeuronPool, SpatialDims};
use crate::synapse::{ThermalState, maturity};

/// Distribution of synapses across thermal states.
#[derive(Clone, Debug, Default)]
pub struct ThermalDistribution {
    pub hot: usize,
    pub warm: usize,
    pub cool: usize,
    pub cold: usize,
    pub dead: usize,
}

impl ThermalDistribution {
    pub fn total(&self) -> usize {
        self.hot + self.warm + self.cool + self.cold + self.dead
    }
}

impl std::fmt::Display for ThermalDistribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HOT={} WARM={} COOL={} COLD={} DEAD={}",
            self.hot, self.warm, self.cool, self.cold, self.dead
        )
    }
}

/// Distribution of neurons across NeuronType variants.
#[derive(Clone, Debug, Default)]
pub struct TypeDistribution {
    pub computational: u32,
    pub sensory: u32,
    pub motor: u32,
    pub memory_reader: u32,
    pub memory_matcher: u32,
    pub gate: u32,
    pub relay: u32,
    pub oscillator: u32,
}

/// Summary statistics for a neuron pool.
#[derive(Clone, Debug)]
pub struct PoolStats {
    pub name: String,
    pub dims: SpatialDims,
    pub n_neurons: u32,
    pub n_excitatory: u32,
    pub n_inhibitory: u32,
    pub n_synapses: usize,
    pub tick_count: u64,
    pub last_spike_count: u32,
    pub thermal: ThermalDistribution,
    pub types: TypeDistribution,
    pub mean_weight_magnitude: f32,
    pub mean_eligibility_magnitude: f32,
    // --- Growth engine fields (A6) ---
    /// Initial neuron count at construction (genome baseline).
    pub initial_neuron_count: u32,
    /// Current / initial neuron count ratio.
    pub growth_ratio: f32,
    /// Ratio of neurons that spiked at least once since last reset.
    pub active_ratio: f32,
    /// Average synapses per neuron.
    pub synapses_per_neuron: f32,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pool '{}': {} neurons ({}E/{}I), {} synapses, dims={}x{}x{}",
            self.name, self.n_neurons, self.n_excitatory, self.n_inhibitory,
            self.n_synapses, self.dims.w, self.dims.h, self.dims.d)?;
        writeln!(f, "  Tick: {}, Last spikes: {}", self.tick_count, self.last_spike_count)?;
        writeln!(f, "  Thermal: {}", self.thermal)?;
        writeln!(f, "  Mean |weight|: {:.1}, Mean |eligibility|: {:.1}",
            self.mean_weight_magnitude, self.mean_eligibility_magnitude)?;
        writeln!(f, "  Growth: {:.2}x (initial={}), Active: {:.1}%, Syn/neuron: {:.1}",
            self.growth_ratio, self.initial_neuron_count,
            self.active_ratio * 100.0, self.synapses_per_neuron)?;
        Ok(())
    }
}

impl NeuronPool {
    /// Compute thermal distribution of all synapses.
    pub fn thermal_distribution(&self) -> ThermalDistribution {
        let mut dist = ThermalDistribution::default();

        for syn in &self.synapses.synapses {
            if maturity::is_dead(syn.maturity) {
                dist.dead += 1;
            } else {
                match syn.thermal_state() {
                    ThermalState::Hot => dist.hot += 1,
                    ThermalState::Warm => dist.warm += 1,
                    ThermalState::Cool => dist.cool += 1,
                    ThermalState::Cold => dist.cold += 1,
                }
            }
        }

        dist
    }

    /// Compute neuron type distribution.
    pub fn type_distribution(&self) -> TypeDistribution {
        let mut dist = TypeDistribution::default();
        for &f in &self.neurons.flags {
            match flags::neuron_type(f) {
                NeuronType::Computational => dist.computational += 1,
                NeuronType::Sensory => dist.sensory += 1,
                NeuronType::Motor => dist.motor += 1,
                NeuronType::MemoryReader => dist.memory_reader += 1,
                NeuronType::MemoryMatcher => dist.memory_matcher += 1,
                NeuronType::Gate => dist.gate += 1,
                NeuronType::Relay => dist.relay += 1,
                NeuronType::Oscillator => dist.oscillator += 1,
            }
        }
        dist
    }

    /// Compute comprehensive pool statistics.
    pub fn stats(&self) -> PoolStats {
        let thermal = self.thermal_distribution();
        let types = self.type_distribution();

        let n_syn = self.synapses.total_synapses();
        let (weight_sum, elig_sum) = self.synapses.synapses.iter().fold((0u64, 0u64), |(ws, es), s| {
            (ws + s.weight.unsigned_abs() as u64, es + s.eligibility.unsigned_abs() as u64)
        });

        let mean_weight = if n_syn > 0 { weight_sum as f32 / n_syn as f32 } else { 0.0 };
        let mean_elig = if n_syn > 0 { elig_sum as f32 / n_syn as f32 } else { 0.0 };

        let active_ratio = if self.n_neurons > 0 {
            self.active_neuron_count() as f32 / self.n_neurons as f32
        } else {
            0.0
        };
        let synapses_per_neuron = if self.n_neurons > 0 {
            n_syn as f32 / self.n_neurons as f32
        } else {
            0.0
        };

        PoolStats {
            name: self.name.clone(),
            dims: self.dims,
            n_neurons: self.n_neurons,
            n_excitatory: self.n_excitatory,
            n_inhibitory: self.n_inhibitory,
            n_synapses: n_syn,
            tick_count: self.tick_count,
            last_spike_count: self.last_spike_count,
            thermal,
            types,
            mean_weight_magnitude: mean_weight,
            mean_eligibility_magnitude: mean_elig,
            initial_neuron_count: self.initial_neuron_count,
            growth_ratio: self.growth_ratio(),
            active_ratio,
            synapses_per_neuron,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::PoolConfig;

    #[test]
    fn stats_empty_pool() {
        let pool = NeuronPool::new("test", 100, PoolConfig::default());
        let stats = pool.stats();
        assert_eq!(stats.n_neurons, 100);
        assert_eq!(stats.n_synapses, 0);
        assert_eq!(stats.thermal.total(), 0);
    }

    #[test]
    fn stats_reports_growth_fields() {
        let mut pool = NeuronPool::new("test", 32, PoolConfig::default());

        // Generate some activity
        let mut input = vec![0i16; 32];
        input[0] = 10000;
        input[1] = 10000;
        pool.tick_simple(&input);

        let stats = pool.stats();
        assert_eq!(stats.initial_neuron_count, 32);
        assert_eq!(stats.growth_ratio, 1.0);
        assert!(stats.active_ratio > 0.0, "some neurons should be active");
        assert_eq!(stats.synapses_per_neuron, 0.0, "no connectivity = 0 syn/neuron");

        // Grow and check ratio changes
        pool.grow_neurons_seeded(32, 42);
        let stats2 = pool.stats();
        assert_eq!(stats2.initial_neuron_count, 32);
        assert!((stats2.growth_ratio - 2.0).abs() < 0.01, "64/32 = 2.0");
    }

    #[test]
    fn thermal_distribution_counts() {
        let pool = NeuronPool::with_random_connectivity("test", 50, 0.05, PoolConfig::default());
        let dist = pool.thermal_distribution();
        // All new synapses start as HOT
        assert_eq!(dist.hot, pool.synapse_count());
        assert_eq!(dist.warm, 0);
        assert_eq!(dist.cool, 0);
        assert_eq!(dist.cold, 0);
    }
}
