//! NeuronPool — the top-level container and tick loop.
//!
//! Integer-only LIF (Leaky Integrate-and-Fire) dynamics with synaptic delay
//! buffers, homeostatic threshold adjustment, and Signal facade for I/O.

use std::ops::Range;

use ternary_signal::Signal;

use crate::neuron::NeuronArrays;
use crate::synapse::{Synapse, SynapseStore};

/// Pool configuration — all Q8.8 fixed-point where noted.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Resting membrane potential (Q8.8). Default: -70 * 256 = -17920
    pub resting_potential: i16,
    /// Spike threshold (Q8.8). Default: -55 * 256 = -14080
    pub spike_threshold: i16,
    /// Reset potential after spike (Q8.8). Default: -65 * 256 = -16640
    pub reset_potential: i16,
    /// Refractory period in ticks. Default: 2
    pub refractory_ticks: u8,
    /// Eligibility trace decay factor (0-255). trace *= decay / 256 each tick.
    /// Default: 230 (~90% retention per tick, ~50% after 7 ticks)
    pub trace_decay: u8,
    /// Homeostatic threshold adjustment rate. Default: 1
    pub homeostatic_rate: u8,
    /// Maximum outgoing synapses per neuron. Default: 128
    pub max_synapses_per_neuron: u16,
    /// STDP positive trace bump (pre fires before post). Default: 20
    pub stdp_positive: i8,
    /// STDP negative trace bump (post fires before pre). Default: -10
    pub stdp_negative: i8,
    /// Maximum axonal delay in ticks. Default: 8
    pub max_delay: u8,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            resting_potential: -17920, // -70 * 256
            spike_threshold: -14080,   // -55 * 256
            reset_potential: -16640,   // -65 * 256
            refractory_ticks: 2,
            trace_decay: 230,
            homeostatic_rate: 1,
            max_synapses_per_neuron: 128,
            stdp_positive: 20,
            stdp_negative: -10,
            max_delay: 8,
        }
    }
}

/// Delay buffer for synaptic transmission. Ring buffer per delay slot.
pub(crate) struct DelayBuffer {
    /// `buffers[d]` holds currents to be delivered after `d+1` ticks.
    /// Each inner Vec has length n_neurons.
    buffers: Vec<Vec<i16>>,
    /// Current write position (advances each tick, wraps around)
    write_head: usize,
    /// Number of delay slots (= max_delay)
    n_slots: usize,
}

impl DelayBuffer {
    pub(crate) fn new(n_neurons: usize, max_delay: u8) -> Self {
        let n_slots = max_delay.max(1) as usize;
        Self {
            buffers: vec![vec![0i16; n_neurons]; n_slots],
            write_head: 0,
            n_slots,
        }
    }

    /// Queue current delivery to `target` neuron after `delay` ticks.
    #[inline]
    fn queue(&mut self, target: usize, delay: u8, current: i16) {
        let slot = (self.write_head + delay.max(1) as usize - 1) % self.n_slots;
        self.buffers[slot][target] = self.buffers[slot][target].saturating_add(current);
    }

    /// Read and clear the current delivery slot (advances head).
    fn drain_current_slot(&mut self) -> &[i16] {
        let slot = self.write_head;
        self.write_head = (self.write_head + 1) % self.n_slots;
        // Return the slot contents (caller reads them), then zero it
        // We return a reference — caller must read before next drain
        &self.buffers[slot]
    }

    /// Zero out the slot that was just drained.
    fn clear_drained_slot(&mut self) {
        // The slot that was just read is at (write_head - 1 + n_slots) % n_slots
        let slot = (self.write_head + self.n_slots - 1) % self.n_slots;
        for v in &mut self.buffers[slot] {
            *v = 0;
        }
    }
}

/// A biological neuron pool — LIF neurons + CSR synapses + delay buffers.
pub struct NeuronPool {
    /// Human-readable name for this pool (e.g. "frontal_excitatory")
    pub name: String,
    /// SoA neuron arrays
    pub neurons: NeuronArrays,
    /// CSR synapse storage
    pub synapses: SynapseStore,
    /// Number of neurons in this pool
    pub n_neurons: u32,
    /// Number of excitatory neurons (first n_excitatory in arrays)
    pub n_excitatory: u32,
    /// Number of inhibitory neurons (last n_inhibitory in arrays)
    pub n_inhibitory: u32,
    /// Monotonic tick counter
    pub tick_count: u64,
    /// Configuration parameters
    pub config: PoolConfig,
    /// Synaptic delay buffer
    pub(crate) delay_buf: DelayBuffer,
    /// Scratch buffer for synaptic currents (avoids allocation in tick)
    pub(crate) synaptic_current: Vec<i16>,
    /// Spike count for the most recent tick
    pub(crate) last_spike_count: u32,
    /// Homeostatic spike rate tracker: running average per neuron (Q8.8)
    pub(crate) spike_rate: Vec<u16>,
}

impl NeuronPool {
    /// Create a new pool with `n` neurons (no connections).
    ///
    /// Dale's Law: 80% excitatory, 20% inhibitory.
    pub fn new(name: &str, n_neurons: u32, config: PoolConfig) -> Self {
        let n_exc = (n_neurons * 4) / 5; // 80% excitatory
        let n_inh = n_neurons - n_exc;

        let neurons = NeuronArrays::new(
            n_neurons,
            n_exc,
            config.resting_potential,
            config.spike_threshold,
        );
        let synapses = SynapseStore::empty(n_neurons);
        let delay_buf = DelayBuffer::new(n_neurons as usize, config.max_delay);

        Self {
            name: name.to_string(),
            neurons,
            synapses,
            n_neurons,
            n_excitatory: n_exc,
            n_inhibitory: n_inh,
            tick_count: 0,
            config,
            delay_buf,
            synaptic_current: vec![0i16; n_neurons as usize],
            last_spike_count: 0,
            spike_rate: vec![0u16; n_neurons as usize],
        }
    }

    /// Create a pool with random connectivity (default seed).
    ///
    /// Convenience wrapper around `with_random_connectivity_seeded` with a fixed seed.
    pub fn with_random_connectivity(
        name: &str,
        n_neurons: u32,
        density: f32,
        config: PoolConfig,
    ) -> Self {
        Self::with_random_connectivity_seeded(name, n_neurons, density, config, 0xDEAD_BEEF_CAFE_1337)
    }

    /// Create a pool with random connectivity using a deterministic seed.
    ///
    /// `density` is the probability of a connection existing between any two neurons
    /// (0.0 = no connections, 1.0 = fully connected). Typical: 0.01-0.05.
    /// `seed` drives the LCG — same seed + same params = identical pool.
    pub fn with_random_connectivity_seeded(
        name: &str,
        n_neurons: u32,
        density: f32,
        config: PoolConfig,
        seed: u64,
    ) -> Self {
        let mut pool = Self::new(name, n_neurons, config);

        // Simple LCG for deterministic random without pulling in rand crate
        let mut rng_state: u64 = seed ^ (n_neurons as u64);
        let lcg_next = |state: &mut u64| -> u32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as u32
        };

        let max_syn = pool.config.max_synapses_per_neuron;
        let density_threshold = (density * u32::MAX as f32) as u32;
        let mut edges: Vec<(u32, Synapse)> = Vec::new();
        let mut per_neuron_count = vec![0u16; n_neurons as usize];

        for src in 0..n_neurons {
            let src_flags = pool.neurons.flags[src as usize];
            for tgt in 0..n_neurons {
                if src == tgt { continue; } // No self-connections
                if per_neuron_count[src as usize] >= max_syn { break; }

                if lcg_next(&mut rng_state) < density_threshold {
                    let magnitude = (lcg_next(&mut rng_state) % 60 + 10) as u8; // 10-69
                    let delay = (lcg_next(&mut rng_state) % pool.config.max_delay as u32 + 1) as u8;
                    edges.push((src, Synapse::new(tgt as u16, magnitude, delay, src_flags)));
                    per_neuron_count[src as usize] += 1;
                }
            }
        }

        pool.synapses = SynapseStore::from_edges(n_neurons, edges);
        pool
    }

    /// Step the pool forward one tick.
    ///
    /// `input_currents` provides external stimulation (length must equal n_neurons).
    /// Pass an empty slice or all-zeros for no external input.
    pub fn tick(&mut self, input_currents: &[i16]) {
        let n = self.n_neurons as usize;

        // 1. Read delayed synaptic currents from delay buffer
        let delayed = self.delay_buf.drain_current_slot();
        for i in 0..n {
            self.synaptic_current[i] = delayed[i];
        }
        self.delay_buf.clear_drained_slot();

        // 2. Integrate each neuron
        let mut spike_count = 0u32;

        for i in 0..n {
            // 2a. Refractory check
            if self.neurons.refract_remaining[i] > 0 {
                self.neurons.refract_remaining[i] -= 1;
                self.neurons.spike_out[i] = false;
                continue;
            }

            // 2b. Leak: membrane += (resting - membrane) >> (8 - leak.min(7))
            let leak_shift = 8u32.saturating_sub(self.neurons.leak[i].min(7) as u32);
            let leak_current = (self.config.resting_potential as i32 - self.neurons.membrane[i] as i32) >> leak_shift;
            self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(leak_current as i16);

            // 2c. Add external input
            if i < input_currents.len() {
                self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(input_currents[i]);
            }

            // 2d. Add synaptic current
            self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(self.synaptic_current[i]);

            // 2e. Spike check
            if self.neurons.membrane[i] >= self.neurons.threshold[i] {
                self.neurons.spike_out[i] = true;
                self.neurons.membrane[i] = self.config.reset_potential;
                self.neurons.refract_remaining[i] = self.config.refractory_ticks;

                // Post-synaptic trace bump (this neuron just fired)
                self.neurons.trace[i] = self.neurons.trace[i].saturating_add(self.config.stdp_positive);

                spike_count += 1;
            } else {
                self.neurons.spike_out[i] = false;
            }

            // 2f. Decay eligibility trace
            self.neurons.trace[i] = ((self.neurons.trace[i] as i16 * self.config.trace_decay as i16) / 256) as i8;
        }

        // 3. Propagate spikes through synapses
        for i in 0..n {
            if !self.neurons.spike_out[i] {
                continue;
            }

            let outgoing = self.synapses.outgoing_mut(i as u32);
            for syn in outgoing.iter_mut() {
                let target = syn.target as usize;
                if target >= n { continue; }

                // STDP: pre-synaptic neuron just fired.
                // Check if target ALSO fired this tick (anti-causal / simultaneous).
                // We use spike_out (this-tick) not trace (slow-decaying) because the
                // trace never decays to zero during sustained firing (refractory=2,
                // fire every 3 ticks, trace_decay=94%/tick → always > 0). Using trace
                // makes ALL interactions anti-causal during sustained activity, which
                // causes DA reward to WEAKEN pathways instead of strengthening them.
                if self.neurons.spike_out[target] {
                    syn.eligibility = syn.eligibility.saturating_add(self.config.stdp_negative);
                } else {
                    // Pre fired, target hasn't fired this tick → causal marking
                    syn.eligibility = syn.eligibility.saturating_add(self.config.stdp_positive);
                }

                // Queue current delivery with delay
                let current = syn.weight as i16 * 64; // Scale weight to Q8.8 current range
                self.delay_buf.queue(target, syn.delay, current);
            }
        }

        // 4. Homeostatic threshold adjustment
        if self.config.homeostatic_rate > 0 && self.tick_count % 100 == 0 {
            let rate = self.config.homeostatic_rate as i16;
            for i in 0..n {
                // Update running spike rate (exponential moving average)
                let spiked = if self.neurons.spike_out[i] { 256u16 } else { 0 };
                self.spike_rate[i] = ((self.spike_rate[i] as u32 * 255 + spiked as u32) / 256) as u16;

                // Target rate: ~5% of ticks should produce a spike (= ~13 in Q8.8)
                let target_rate = 13u16;
                if self.spike_rate[i] > target_rate * 2 {
                    // Firing too much — raise threshold
                    self.neurons.threshold[i] = self.neurons.threshold[i].saturating_add(rate);
                } else if self.spike_rate[i] < target_rate / 2 {
                    // Too silent — lower threshold
                    self.neurons.threshold[i] = self.neurons.threshold[i].saturating_sub(rate);
                }
            }
        }

        // 5. Decay synapse eligibility traces
        for syn in self.synapses.synapses.iter_mut() {
            syn.eligibility = ((syn.eligibility as i16 * self.config.trace_decay as i16) / 256) as i8;
        }

        self.last_spike_count = spike_count;
        self.tick_count += 1;
    }

    /// Inject an external signal into a range of neurons.
    ///
    /// Signal polarity determines excitation (+) or inhibition (-).
    /// Magnitude is scaled to Q8.8 current.
    pub fn inject(&mut self, neuron_range: Range<u32>, signal: Signal) {
        let current = signal.as_signed_i32() as i16 * 128; // Scale to Q8.8
        let start = neuron_range.start as usize;
        let end = (neuron_range.end as usize).min(self.n_neurons as usize);

        for i in start..end {
            self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(current);
        }
    }

    /// Read output from a range of neurons as Signal vector.
    ///
    /// Neurons that spiked this tick produce a positive Signal with magnitude
    /// proportional to their suprathreshold membrane potential before reset.
    pub fn read_output(&self, neuron_range: Range<u32>) -> Vec<Signal> {
        let start = neuron_range.start as usize;
        let end = (neuron_range.end as usize).min(self.n_neurons as usize);

        (start..end)
            .map(|i| {
                if self.neurons.spike_out[i] {
                    Signal::positive(255) // Binary spike — full magnitude
                } else {
                    Signal::zero()
                }
            })
            .collect()
    }

    /// Number of spikes in the most recent tick.
    #[inline]
    pub fn spike_count(&self) -> u32 {
        self.last_spike_count
    }

    /// Total number of synapses.
    #[inline]
    pub fn synapse_count(&self) -> usize {
        self.synapses.total_synapses()
    }

    /// Mean activation as a Signal (average spike rate as magnitude).
    pub fn mean_activation(&self) -> Signal {
        if self.n_neurons == 0 {
            return Signal::zero();
        }
        let rate = (self.last_spike_count as u32 * 255) / self.n_neurons;
        if rate > 0 {
            Signal::positive(rate.min(255) as u8)
        } else {
            Signal::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_creation() {
        let pool = NeuronPool::new("test", 100, PoolConfig::default());
        assert_eq!(pool.n_neurons, 100);
        assert_eq!(pool.n_excitatory, 80);
        assert_eq!(pool.n_inhibitory, 20);
        assert_eq!(pool.tick_count, 0);
        assert_eq!(pool.spike_count(), 0);
        assert_eq!(pool.synapse_count(), 0);
    }

    #[test]
    fn pool_with_connectivity() {
        let pool = NeuronPool::with_random_connectivity("test", 100, 0.05, PoolConfig::default());
        assert!(pool.synapse_count() > 0, "should have some connections");
        // With 100 neurons and 5% density, expect ~500 synapses (100*99*0.05)
        assert!(pool.synapse_count() > 100, "should have reasonable connectivity");
    }

    #[test]
    fn seeded_connectivity_is_deterministic() {
        let a = NeuronPool::with_random_connectivity_seeded("a", 64, 0.05, PoolConfig::default(), 42);
        let b = NeuronPool::with_random_connectivity_seeded("b", 64, 0.05, PoolConfig::default(), 42);
        assert_eq!(a.synapse_count(), b.synapse_count(), "same seed = same synapse count");
        // Different seed = different topology
        let c = NeuronPool::with_random_connectivity_seeded("c", 64, 0.05, PoolConfig::default(), 99);
        // Extremely unlikely to match (but possible in theory)
        assert_ne!(a.synapse_count(), c.synapse_count(), "different seed should differ");
    }

    #[test]
    fn tick_no_input() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());
        pool.tick(&[]);
        assert_eq!(pool.spike_count(), 0, "no spikes without input");
        assert_eq!(pool.tick_count, 1);
    }

    #[test]
    fn tick_with_strong_input_causes_spike() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());
        // Inject massive current to force a spike
        let mut input = vec![0i16; 10];
        input[0] = 10000; // Way above threshold

        pool.tick(&input);
        assert!(pool.spike_count() > 0, "strong input should cause spikes");
        // Neuron 0 should have spiked
        assert!(pool.neurons.spike_out[0], "neuron 0 should have spiked");
    }

    #[test]
    fn refractory_period() {
        let mut pool = NeuronPool::new("test", 1, PoolConfig::default());
        let input = vec![10000i16]; // Force spike

        pool.tick(&input);
        assert!(pool.neurons.spike_out[0], "should spike on first tick");

        // During refractory period, even strong input shouldn't spike
        pool.tick(&input);
        assert!(!pool.neurons.spike_out[0], "should NOT spike during refractory");

        pool.tick(&input);
        assert!(!pool.neurons.spike_out[0], "should NOT spike during refractory (tick 2)");

        // After refractory period (2 ticks), should spike again
        pool.tick(&input);
        assert!(pool.neurons.spike_out[0], "should spike again after refractory");
    }

    #[test]
    fn signal_inject_and_read() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());

        // Inject excitatory signal
        pool.inject(0..5, Signal::positive(255));
        pool.tick(&[]);

        let output = pool.read_output(0..10);
        assert_eq!(output.len(), 10);

        // At least some of the injected neurons should have spiked
        let spiked: usize = output.iter().filter(|s| s.is_positive()).count();
        assert!(spiked > 0, "injected neurons should produce output spikes");
    }
}
