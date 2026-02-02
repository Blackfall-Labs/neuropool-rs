//! Three-factor plasticity — the core credit assignment mechanism.
//!
//! Three-factor learning rule: eligibility_trace * neuromodulator = weight_change.
//! Only synapses with nonzero eligibility traces are affected — this is what
//! thermograms couldn't do.

use crate::pool::NeuronPool;
use crate::synapse::{Synapse, ThermalState};

/// Result of a growth cycle decision.
#[derive(Clone, Debug, Default)]
pub struct GrowthResult {
    /// Number of neurons grown this cycle.
    pub neurons_grown: u32,
    /// Number of neurons pruned this cycle.
    pub neurons_pruned: u32,
    /// Pool size after the cycle.
    pub new_size: u32,
}

impl NeuronPool {
    /// Apply neuromodulator-gated plasticity to all eligible synapses.
    ///
    /// Three-factor rule: eligibility_trace * modulator_signal = weight_change.
    /// Only synapses with nonzero eligibility traces are affected.
    ///
    /// - DA above 128 reinforces (strengthens causal synapses, weakens anti-causal)
    /// - Cortisol above 30 weakens (weakens causal synapses, strengthens anti-causal)
    /// - ACh above 140 gates synaptogenesis (handled by `synaptogenesis()`)
    ///
    /// Returns (reinforced_count, weakened_count).
    pub fn apply_modulation(&mut self, da: u8, cortisol: u8, _ach: u8) -> (usize, usize) {
        let reinforce = (da as i16 - 128).max(0);
        let weaken = (cortisol as i16 - 30).max(0);

        if reinforce == 0 && weaken == 0 {
            return (0, 0);
        }

        let mut reinforced = 0usize;
        let mut weakened = 0usize;

        for syn in self.synapses.synapses.iter_mut() {
            if syn.eligibility == 0 {
                continue; // No trace = no credit to assign
            }
            if syn.thermal_state() == ThermalState::Cold {
                continue; // Frozen synapses are immutable
            }

            let trace = syn.eligibility as i16;

            // Three-factor credit assignment:
            // Positive trace + DA = strengthen (pre caused post, outcome was good)
            // Negative trace + DA = weaken (anti-causal during good outcome)
            // Positive trace + Cortisol = weaken (pre caused post, outcome was bad)
            // Negative trace + Cortisol = strengthen (anti-causal during bad outcome)
            let delta = (trace * reinforce) / 64 - (trace * weaken) / 64;

            if delta == 0 {
                continue;
            }

            // Apply weight change, clamping to valid range
            let new_weight = (syn.weight as i16 + delta).clamp(-127, 127) as i8;

            // Dale's Law enforcement: excitatory synapses stay >= 0, inhibitory stay <= 0
            // We track this based on the sign of the original weight
            if syn.weight > 0 && new_weight < 0 {
                syn.weight = 0; // Don't flip excitatory to inhibitory
            } else if syn.weight < 0 && new_weight > 0 {
                syn.weight = 0; // Don't flip inhibitory to excitatory
            } else {
                syn.weight = new_weight;
            }

            // Update maturity based on whether weight got stronger or weaker
            if delta > 0 {
                syn.increment_maturity();
                reinforced += 1;
            } else {
                syn.decrement_maturity();
                weakened += 1;
            }
        }

        if reinforced > 0 || weakened > 0 {
            log::debug!(
                "[PLASTICITY] {}: +{} reinforced, -{} weakened (DA={}, Cort={})",
                self.name, reinforced, weakened, da, cortisol
            );
        }

        (reinforced, weakened)
    }

    /// Create new synapses between co-active neurons (ACh-gated).
    ///
    /// Looks for neuron pairs where both spiked recently (nonzero trace) but
    /// are not yet connected. Creates HOT synapses with small random weights.
    ///
    /// Returns count of new synapses created.
    pub fn synaptogenesis(&mut self, ach: u8) -> usize {
        if ach < 140 {
            return 0; // ACh gates new connection formation
        }

        let n = self.n_neurons as usize;
        let max_syn = self.config.max_synapses_per_neuron;

        // Find neurons with recent activity (nonzero post-synaptic trace)
        let active_neurons: Vec<u32> = (0..n)
            .filter(|&i| self.neurons.trace[i].abs() > 5)
            .map(|i| i as u32)
            .collect();

        if active_neurons.len() < 2 {
            return 0;
        }

        // Simple LCG for weight/delay randomization
        let mut rng_state: u64 = self.tick_count.wrapping_mul(0xBEEF) ^ 0xCAFE;
        let lcg_next = |state: &mut u64| -> u32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as u32
        };

        let mut created = 0usize;
        let max_new_per_tick = 16u32; // Don't create too many at once

        for &src in &active_neurons {
            if created >= max_new_per_tick as usize {
                break;
            }

            let current_count = self.synapses.outgoing(src).len() as u16;
            if current_count >= max_syn {
                continue;
            }

            let src_flags = self.neurons.flags[src as usize];

            // Distance cutoff for spatial pools: skip targets beyond ~2σ
            let sigma = self.dims.default_sigma();
            let max_synaptogenesis_d_sq = if self.dims.h > 1 || self.dims.d > 1 {
                // Spatial pool: limit new connections to ~2σ radius
                (2.0 * sigma * 2.0 * sigma) as u32
            } else {
                u32::MAX // Flat pool: no distance restriction
            };

            for &tgt in &active_neurons {
                if src == tgt { continue; }
                if created >= max_new_per_tick as usize { break; }

                // Distance filter for spatial pools
                if max_synaptogenesis_d_sq < u32::MAX {
                    let d_sq = self.dims.distance_sq(src, tgt);
                    if d_sq > max_synaptogenesis_d_sq {
                        continue;
                    }
                }

                // Check if connection already exists
                let already_connected = self.synapses.outgoing(src)
                    .iter()
                    .any(|s| s.target == tgt as u16);

                if already_connected {
                    continue;
                }

                // Create new HOT synapse with small weight
                let magnitude = (lcg_next(&mut rng_state) % 20 + 5) as u8; // 5-24
                // Distance-proportional delay for spatial pools
                let delay = if self.dims.h > 1 || self.dims.d > 1 {
                    let d_sq = self.dims.distance_sq(src, tgt);
                    let max_d_sq = self.dims.max_distance_sq();
                    let sqrt_max = (max_d_sq as f32).sqrt();
                    let ratio = if sqrt_max > 0.0 { (d_sq as f32).sqrt() / sqrt_max } else { 0.0 };
                    let d = 1 + (ratio * (self.config.max_delay - 1) as f32) as u8;
                    d.min(self.config.max_delay).max(1)
                } else {
                    (lcg_next(&mut rng_state) % self.config.max_delay as u32 + 1) as u8
                };
                let syn = Synapse::new(tgt as u16, magnitude, delay, src_flags);

                self.synapses.add_synapse(src, syn);
                created += 1;
            }
        }

        if created > 0 {
            log::debug!(
                "[SYNAPTOGENESIS] {}: created {} new synapses (ACh={}, active_neurons={})",
                self.name, created, ach, active_neurons.len()
            );
        }

        created
    }

    /// Prune dead synapses (maturity counter = 0 in HOT state).
    ///
    /// Returns count of pruned synapses.
    pub fn prune_dead(&mut self) -> usize {
        let pruned = self.synapses.prune_dead();
        if pruned > 0 {
            log::debug!("[PRUNE] {}: pruned {} dead synapses", self.name, pruned);
        }
        pruned
    }

    /// Chemical-gated growth cycle — decides whether to grow or prune neurons.
    ///
    /// Growth signal is computed from DA, ACh, and NE above baseline (128).
    /// Prune signal is computed from cortisol above baseline + silence ratio.
    /// Dead neurons (zero spikes since last reset) are pruned first.
    ///
    /// Resets spike_counts after making the decision.
    pub fn growth_cycle(&mut self, da: u8, ach: u8, ne: u8, cortisol: u8, seed: u64) -> GrowthResult {
        // Copy config values to avoid borrow-checker conflict with &mut self calls
        let max_neurons = self.config.growth.max_neurons;
        let min_neurons = self.config.growth.min_neurons;
        let growth_per_cycle = self.config.growth.growth_per_cycle;
        let prune_per_cycle = self.config.growth.prune_per_cycle;
        let growth_threshold = self.config.growth.growth_threshold;
        let prune_threshold = self.config.growth.prune_threshold;

        // Compute growth signal from excitatory modulators above baseline
        let growth_signal = (da.saturating_sub(128) as u16)
            + (ach.saturating_sub(128) as u16)
            + (ne.saturating_sub(128) as u16);

        // Compute prune signal from cortisol + silence ratio
        let silence_ratio = if self.n_neurons > 0 {
            let silent = self.n_neurons - self.active_neuron_count();
            (silent * 100 / self.n_neurons) as u16
        } else {
            0
        };
        let prune_signal = (cortisol.saturating_sub(128) as u16) + silence_ratio;

        let mut result = GrowthResult::default();

        // Prune if signal exceeds threshold and above minimum
        if prune_signal > prune_threshold && self.n_neurons > min_neurons {
            // Identify dead neurons (zero spikes)
            let mut dead: Vec<u32> = self.spike_counts.iter()
                .enumerate()
                .filter(|(_, &c)| c == 0)
                .map(|(i, _)| i as u32)
                .collect();

            // Limit to prune_per_cycle
            dead.truncate(prune_per_cycle as usize);

            // Don't prune below minimum
            let max_prunable = self.n_neurons.saturating_sub(min_neurons);
            dead.truncate(max_prunable as usize);

            if !dead.is_empty() {
                result.neurons_pruned = self.prune_neurons(&dead);
            }
        }

        // Grow if signal exceeds threshold and below maximum
        if growth_signal > growth_threshold && self.n_neurons < max_neurons {
            let headroom = max_neurons.saturating_sub(self.n_neurons);
            let to_grow = (growth_per_cycle as u32).min(headroom);

            if to_grow > 0 {
                result.neurons_grown = self.grow_neurons_seeded(to_grow, seed);
            }
        }

        result.new_size = self.n_neurons;

        // Reset activity counters for next observation window
        self.reset_activities();

        if result.neurons_grown > 0 || result.neurons_pruned > 0 {
            log::info!(
                "[GROWTH] {}: +{} grown, -{} pruned → {} neurons (growth_sig={}, prune_sig={})",
                self.name, result.neurons_grown, result.neurons_pruned,
                result.new_size, growth_signal, prune_signal
            );
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::PoolConfig;

    #[test]
    fn modulation_no_effect_at_baseline() {
        let mut pool = NeuronPool::with_random_connectivity("test", 50, 0.1, Default::default());

        // Drive some activity to build eligibility traces
        for _ in 0..10 {
            let input: Vec<i16> = (0..50).map(|i| if i < 10 { 8000 } else { 0 }).collect();
            pool.tick_simple(&input);
        }

        // Baseline chemicals — no modulation should happen
        let (r, w) = pool.apply_modulation(128, 30, 100);
        assert_eq!(r, 0);
        assert_eq!(w, 0);
    }

    #[test]
    fn da_reinforces_eligible_synapses() {
        let mut pool = NeuronPool::with_random_connectivity("test", 50, 0.1, Default::default());

        // Drive activity to build eligibility traces
        for _ in 0..20 {
            let input: Vec<i16> = (0..50).map(|i| if i < 15 { 8000 } else { 0 }).collect();
            pool.tick_simple(&input);
        }

        // Count synapses with nonzero eligibility before
        let eligible_count = pool.synapses.synapses.iter()
            .filter(|s| s.eligibility != 0)
            .count();

        // High DA = reward signal
        let (reinforced, weakened) = pool.apply_modulation(200, 30, 100);

        if eligible_count > 0 {
            assert!(reinforced + weakened > 0, "DA should affect eligible synapses");
        }
    }

    #[test]
    fn cortisol_weakens_eligible_synapses() {
        let mut pool = NeuronPool::with_random_connectivity("test", 50, 0.1, Default::default());

        // Drive activity
        for _ in 0..20 {
            let input: Vec<i16> = (0..50).map(|i| if i < 15 { 8000 } else { 0 }).collect();
            pool.tick_simple(&input);
        }

        // High cortisol = punishment signal
        let (reinforced, weakened) = pool.apply_modulation(128, 80, 100);

        let eligible_count = pool.synapses.synapses.iter()
            .filter(|s| s.eligibility != 0)
            .count();

        // Cortisol should cause weakening of eligible synapses
        if eligible_count > 0 {
            assert!(reinforced + weakened > 0, "Cortisol should affect eligible synapses");
        }
    }

    #[test]
    fn synaptogenesis_creates_connections() {
        let mut pool = NeuronPool::new("test", 20, Default::default());
        assert_eq!(pool.synapse_count(), 0);

        // Drive activity to build traces
        for _ in 0..10 {
            let input: Vec<i16> = (0..20).map(|i| if i < 8 { 8000 } else { 0 }).collect();
            pool.tick_simple(&input);
        }

        // High ACh = gate synaptogenesis
        let created = pool.synaptogenesis(200);
        assert!(created > 0, "ACh should create new synapses between co-active neurons");
    }

    #[test]
    fn synaptogenesis_blocked_without_ach() {
        let mut pool = NeuronPool::new("test", 20, Default::default());

        // Drive activity
        for _ in 0..10 {
            let input: Vec<i16> = (0..20).map(|i| if i < 8 { 8000 } else { 0 }).collect();
            pool.tick_simple(&input);
        }

        // Low ACh = no synaptogenesis
        let created = pool.synaptogenesis(100);
        assert_eq!(created, 0, "Low ACh should block synaptogenesis");
    }

    // ====================================================================
    // Growth cycle tests (A4)
    // ====================================================================

    #[test]
    fn growth_cycle_respects_bounds() {
        use crate::pool::GrowthConfig;

        let mut config = PoolConfig::default();
        config.growth = GrowthConfig {
            max_neurons: 40,
            min_neurons: 8,
            growth_per_cycle: 4,
            prune_per_cycle: 4,
            growth_threshold: 10,
            prune_threshold: 10,
        };
        let mut pool = NeuronPool::new("test", 32, config);

        // Mark all neurons as active so pruning doesn't fire
        pool.spike_counts.fill(5);

        // High DA/ACh/NE should trigger growth
        let result = pool.growth_cycle(200, 200, 200, 128, 42);
        assert_eq!(result.neurons_grown, 4, "should grow 4 (growth_per_cycle)");
        assert_eq!(result.neurons_pruned, 0, "all active, no prune");
        assert_eq!(result.new_size, 36);

        // growth_cycle resets activities, so re-mark as active
        pool.spike_counts.fill(5);

        // Grow again — should hit ceiling at 40
        let result2 = pool.growth_cycle(200, 200, 200, 128, 43);
        assert_eq!(result2.neurons_grown, 4, "should grow exactly to cap");
        assert_eq!(result2.new_size, 40);

        pool.spike_counts.fill(5);

        // One more — already at cap
        let result3 = pool.growth_cycle(200, 200, 200, 128, 44);
        assert_eq!(result3.neurons_grown, 0, "at cap, no growth");
    }

    #[test]
    fn growth_cycle_chemical_gated() {
        let mut pool = NeuronPool::new("test", 32, PoolConfig::default());

        // Baseline chemicals — no growth, no pruning
        let result = pool.growth_cycle(128, 128, 128, 128, 42);
        assert_eq!(result.neurons_grown, 0);
        assert_eq!(result.neurons_pruned, 0);
        assert_eq!(result.new_size, 32);
    }

    #[test]
    fn growth_cycle_prunes_dead_neurons() {
        use crate::pool::GrowthConfig;

        let mut config = PoolConfig::default();
        config.growth = GrowthConfig {
            max_neurons: 200,
            min_neurons: 4,
            growth_per_cycle: 4,
            prune_per_cycle: 8,
            growth_threshold: 30,
            prune_threshold: 10,
        };
        let mut pool = NeuronPool::new("test", 32, config);

        // Simulate: only neurons 0..8 are active, rest are dead
        for i in 0..8 {
            pool.spike_counts[i] = 10;
        }
        // Neurons 8..32 have 0 spikes — all dead

        // High cortisol + many silent neurons should trigger pruning
        let result = pool.growth_cycle(128, 128, 128, 200, 42);
        assert!(result.neurons_pruned > 0, "should prune dead neurons: got {}", result.neurons_pruned);
        assert!(result.new_size < 32);
    }

    #[test]
    fn prune_removes_dead() {
        let mut pool = NeuronPool::with_random_connectivity("test", 20, 0.1, Default::default());
        let initial = pool.synapse_count();
        assert!(initial > 0);

        // Kill some synapses by zeroing their maturity
        for syn in pool.synapses.synapses.iter_mut().take(3) {
            syn.maturity = 0x00; // Dead
        }

        let pruned = pool.prune_dead();
        assert_eq!(pruned, 3);
        assert_eq!(pool.synapse_count(), initial - 3);
    }
}
