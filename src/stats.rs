//! Pool inspection and diagnostics.

use crate::pool::NeuronPool;
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

/// Summary statistics for a neuron pool.
#[derive(Clone, Debug)]
pub struct PoolStats {
    pub name: String,
    pub n_neurons: u32,
    pub n_excitatory: u32,
    pub n_inhibitory: u32,
    pub n_synapses: usize,
    pub tick_count: u64,
    pub last_spike_count: u32,
    pub thermal: ThermalDistribution,
    pub mean_weight_magnitude: f32,
    pub mean_eligibility_magnitude: f32,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pool '{}': {} neurons ({}E/{}I), {} synapses",
            self.name, self.n_neurons, self.n_excitatory, self.n_inhibitory, self.n_synapses)?;
        writeln!(f, "  Tick: {}, Last spikes: {}", self.tick_count, self.last_spike_count)?;
        writeln!(f, "  Thermal: {}", self.thermal)?;
        writeln!(f, "  Mean |weight|: {:.1}, Mean |eligibility|: {:.1}",
            self.mean_weight_magnitude, self.mean_eligibility_magnitude)?;
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

    /// Compute comprehensive pool statistics.
    pub fn stats(&self) -> PoolStats {
        let thermal = self.thermal_distribution();

        let n_syn = self.synapses.total_synapses();
        let (weight_sum, elig_sum) = self.synapses.synapses.iter().fold((0u64, 0u64), |(ws, es), s| {
            (ws + s.weight.unsigned_abs() as u64, es + s.eligibility.unsigned_abs() as u64)
        });

        let mean_weight = if n_syn > 0 { weight_sum as f32 / n_syn as f32 } else { 0.0 };
        let mean_elig = if n_syn > 0 { elig_sum as f32 / n_syn as f32 } else { 0.0 };

        PoolStats {
            name: self.name.clone(),
            n_neurons: self.n_neurons,
            n_excitatory: self.n_excitatory,
            n_inhibitory: self.n_inhibitory,
            n_synapses: n_syn,
            tick_count: self.tick_count,
            last_spike_count: self.last_spike_count,
            thermal,
            mean_weight_magnitude: mean_weight,
            mean_eligibility_magnitude: mean_elig,
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
