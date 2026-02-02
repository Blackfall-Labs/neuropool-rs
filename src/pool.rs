//! NeuronPool — the top-level container and tick loop.
//!
//! Integer-only LIF (Leaky Integrate-and-Fire) dynamics with synaptic delay
//! buffers, homeostatic threshold adjustment, and Signal facade for I/O.

use std::ops::Range;

use ternary_signal::Signal;

use crate::binding::BindingTable;
use crate::io::{NeuronIO, NullIO};
use crate::neuron::{NeuronArrays, NeuronType, flags};
use crate::synapse::{Synapse, SynapseStore};

/// 3D spatial dimensions for a neuron pool grid.
///
/// Neurons are arranged on a fixed grid where position is implicit from index:
///   x = i % w, y = (i / w) % h, z = i / (w * h)
///
/// No per-neuron position storage — coordinates are computed on demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialDims {
    pub w: u16,
    pub h: u16,
    pub d: u16,
}

impl SpatialDims {
    /// Create spatial dimensions. Panics if any dimension is zero.
    pub fn new(w: u16, h: u16, d: u16) -> Self {
        assert!(w > 0 && h > 0 && d > 0, "spatial dimensions must be non-zero");
        Self { w, h, d }
    }

    /// Flat dimensions (1D array of n neurons, no spatial structure).
    pub fn flat(n: u32) -> Self {
        Self { w: n as u16, h: 1, d: 1 }
    }

    /// Total number of voxels in the grid.
    #[inline]
    pub fn total(&self) -> u32 {
        self.w as u32 * self.h as u32 * self.d as u32
    }

    /// Convert flat index to (x, y, z) coordinates.
    #[inline]
    pub fn coords(&self, i: u32) -> (u16, u16, u16) {
        let w = self.w as u32;
        let h = self.h as u32;
        let x = (i % w) as u16;
        let y = ((i / w) % h) as u16;
        let z = (i / (w * h)) as u16;
        (x, y, z)
    }

    /// Squared Euclidean distance between two neuron indices.
    #[inline]
    pub fn distance_sq(&self, i: u32, j: u32) -> u32 {
        let (xi, yi, zi) = self.coords(i);
        let (xj, yj, zj) = self.coords(j);
        let dx = xi as i32 - xj as i32;
        let dy = yi as i32 - yj as i32;
        let dz = zi as i32 - zj as i32;
        (dx * dx + dy * dy + dz * dz) as u32
    }

    /// Maximum possible squared distance in this grid (corner to corner).
    #[inline]
    pub fn max_distance_sq(&self) -> u32 {
        let dx = (self.w as u32).saturating_sub(1);
        let dy = (self.h as u32).saturating_sub(1);
        let dz = (self.d as u32).saturating_sub(1);
        dx * dx + dy * dy + dz * dz
    }

    /// Default sigma for Gaussian connectivity: max(w, h, d) / 3.
    #[inline]
    pub fn default_sigma(&self) -> f32 {
        self.w.max(self.h).max(self.d) as f32 / 3.0
    }

    /// Whether this neuron index is on the grid boundary.
    #[inline]
    pub fn is_boundary(&self, i: u32) -> bool {
        let (x, y, z) = self.coords(i);
        x == 0 || x == self.w - 1
            || y == 0 || y == self.h - 1
            || z == 0 || z == self.d - 1
    }
}

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
    /// 3D spatial dimensions of the neuron grid
    pub dims: SpatialDims,
    /// SoA neuron arrays
    pub neurons: NeuronArrays,
    /// CSR synapse storage
    pub synapses: SynapseStore,
    /// Binding configurations for specialized neurons
    pub bindings: BindingTable,
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
    /// Spike window accumulator — tracks which neurons spiked at ANY point
    /// since the last `inject()` call. This allows `read_output()` to report
    /// activity from the entire tick sequence, not just the final tick.
    /// Biological basis: a neuron that fired during a processing window is
    /// "active" even if currently in refractory period.
    pub(crate) spike_window: Vec<bool>,
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
            dims: SpatialDims::flat(n_neurons),
            neurons,
            synapses,
            bindings: BindingTable::new(),
            n_neurons,
            n_excitatory: n_exc,
            n_inhibitory: n_inh,
            tick_count: 0,
            config,
            delay_buf,
            synaptic_current: vec![0i16; n_neurons as usize],
            last_spike_count: 0,
            spike_rate: vec![0u16; n_neurons as usize],
            spike_window: vec![false; n_neurons as usize],
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

    /// Create a pool with spatial (Gaussian distance-weighted) connectivity.
    ///
    /// Neurons are laid out on a 3D grid defined by `dims`. Connection probability
    /// falls off with Gaussian distance: p(i,j) = norm * exp(-d_sq / 2σ²).
    /// Delay is proportional to distance: nearby = fast, far = slow.
    ///
    /// Boundary neurons are marked as sentinels: inhibitory FastSpiking with
    /// strong inhibitory connections to interior neighbors, preventing spike escape.
    pub fn with_spatial_connectivity_seeded(
        name: &str,
        dims: SpatialDims,
        density: f32,
        config: PoolConfig,
        seed: u64,
    ) -> Self {
        let n_neurons = dims.total();
        let mut pool = Self::new(name, n_neurons, config);
        pool.dims = dims;

        // --- Build Gaussian LUT ---
        // For each possible d_sq value, precompute a u32 threshold for connection.
        let sigma = dims.default_sigma();
        let two_sigma_sq = 2.0 * sigma * sigma;
        let max_d_sq = dims.max_distance_sq();

        // Compute normalization: sum of exp(-d_sq / 2σ²) for all i!=j pairs
        // For small pools we can compute exactly; for large pools we sample.
        let mut gaussian_sum = 0.0f64;
        let n = n_neurons as u64;
        if n <= 512 {
            // Exact computation for pools up to 512 neurons
            for i in 0..n_neurons {
                for j in 0..n_neurons {
                    if i == j { continue; }
                    let d_sq = dims.distance_sq(i, j);
                    gaussian_sum += (-(d_sq as f64) / two_sigma_sq as f64).exp();
                }
            }
        } else {
            // Sample-based estimation for larger pools
            let mut rng = seed ^ 0x1234_5678;
            let samples = 10000u64;
            for _ in 0..samples {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let i = ((rng >> 33) as u32) % n_neurons;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = ((rng >> 33) as u32) % n_neurons;
                if i == j { continue; }
                let d_sq = dims.distance_sq(i, j);
                gaussian_sum += (-(d_sq as f64) / two_sigma_sq as f64).exp();
            }
            // Scale up to full population
            gaussian_sum *= (n * (n - 1)) as f64 / samples as f64;
        }

        // norm = desired_edges / gaussian_sum
        let desired_edges = density as f64 * (n * (n - 1)) as f64;
        let norm = if gaussian_sum > 0.0 { desired_edges / gaussian_sum } else { density as f64 };

        // Build LUT: for each d_sq value, compute threshold = norm * exp(-d_sq/2σ²) * u32::MAX
        let max_d_sq_usize = (max_d_sq as usize).min(512); // cap LUT size
        let mut lut = vec![0u32; max_d_sq_usize + 1];
        for d_sq in 0..=max_d_sq_usize {
            let prob = norm * (-(d_sq as f64) / two_sigma_sq as f64).exp();
            lut[d_sq] = (prob.min(1.0) * u32::MAX as f64) as u32;
        }

        // --- Generate edges using LUT ---
        let mut rng_state: u64 = seed ^ (n_neurons as u64);
        let lcg_next = |state: &mut u64| -> u32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as u32
        };

        let max_syn = pool.config.max_synapses_per_neuron;
        let max_delay = pool.config.max_delay;
        let sqrt_max_d = (max_d_sq as f32).sqrt();
        let mut edges: Vec<(u32, Synapse)> = Vec::new();
        let mut per_neuron_count = vec![0u16; n_neurons as usize];

        for src in 0..n_neurons {
            let src_flags = pool.neurons.flags[src as usize];
            for tgt in 0..n_neurons {
                if src == tgt { continue; }
                if per_neuron_count[src as usize] >= max_syn { break; }

                let d_sq = dims.distance_sq(src, tgt);
                let d_sq_idx = (d_sq as usize).min(max_d_sq_usize);
                let threshold = lut[d_sq_idx];

                if lcg_next(&mut rng_state) < threshold {
                    let magnitude = (lcg_next(&mut rng_state) % 60 + 10) as u8; // 10-69
                    // Distance-proportional delay
                    let dist_ratio = if sqrt_max_d > 0.0 {
                        (d_sq as f32).sqrt() / sqrt_max_d
                    } else {
                        0.0
                    };
                    let delay = 1 + (dist_ratio * (max_delay - 1) as f32) as u8;
                    let delay = delay.min(max_delay).max(1);

                    edges.push((src, Synapse::new(tgt as u16, magnitude, delay, src_flags)));
                    per_neuron_count[src as usize] += 1;
                }
            }
        }

        pool.synapses = SynapseStore::from_edges(n_neurons, edges);

        // --- Place sentinels at boundary ---
        pool.place_sentinels();

        pool
    }

    /// Mark boundary neurons as inhibitory sentinels and wire strong
    /// inhibitory connections to nearby interior neurons.
    ///
    /// Sentinels prevent spike chains from escaping the region boundary,
    /// acting as the brain's "negative space" for traversal containment.
    fn place_sentinels(&mut self) {
        let n = self.n_neurons;
        let dims = self.dims;

        // Skip sentinel placement for flat (1D) pools — no spatial boundary
        if dims.h == 1 && dims.d == 1 {
            return;
        }

        // Skip sentinels for small grids where boundary neurons dominate.
        // For a grid to benefit from sentinels, each axis must be >= 6 so
        // the interior (axis-2 per dim) has meaningful volume. Below this,
        // sentinels over-inhibit and crush all activity.
        if dims.w < 6 || dims.h < 6 || (dims.d > 1 && dims.d < 6) {
            return;
        }

        for i in 0..n {
            if !dims.is_boundary(i) { continue; }

            let idx = i as usize;

            // Overwrite flags: inhibitory + FastSpiking + Computational
            self.neurons.flags[idx] = flags::encode_full(
                true,
                crate::neuron::NeuronProfile::FastSpiking,
                NeuronType::Computational,
            );
            self.neurons.leak[idx] = crate::neuron::NeuronProfile::FastSpiking.default_leak();

            // Lower threshold by 4mV (Q8.8: 4 * 256 = 1024)
            self.neurons.threshold[idx] = self.neurons.threshold[idx].saturating_sub(1024);
        }

        // Wire sentinel→interior inhibitory connections
        // For each sentinel, connect to interior neighbors within radius 2
        let mut sentinel_edges: Vec<(u32, Synapse)> = Vec::new();
        let radius_sq = 4u32 + 1; // distance_sq <= 4 (radius ~2 voxels)

        for i in 0..n {
            if !dims.is_boundary(i) { continue; }

            let src_flags = self.neurons.flags[i as usize];
            for j in 0..n {
                if i == j { continue; }
                if dims.is_boundary(j) { continue; } // sentinel→sentinel not needed

                let d_sq = dims.distance_sq(i, j);
                if d_sq <= radius_sq {
                    // Strong inhibitory synapse: weight -80, delay 1
                    sentinel_edges.push((i, Synapse::new(j as u16, 80, 1, src_flags)));
                }
            }
        }

        // Merge sentinel edges into existing synapse store
        if !sentinel_edges.is_empty() {
            // Collect existing edges
            let mut all_edges: Vec<(u32, Synapse)> = Vec::new();
            for src in 0..n {
                let outgoing = self.synapses.outgoing(src);
                for syn in outgoing {
                    all_edges.push((src, syn.clone()));
                }
            }
            all_edges.extend(sentinel_edges);
            self.synapses = SynapseStore::from_edges(n, all_edges);
        }
    }

    /// Step the pool forward one tick (no external I/O — backward compatible).
    ///
    /// Equivalent to `tick(input_currents, &mut NullIO)`. All specialized neuron
    /// types behave as standard Computational neurons.
    pub fn tick_simple(&mut self, input_currents: &[i16]) {
        self.tick(input_currents, &mut NullIO);
    }

    /// Step the pool forward one tick with external I/O support.
    ///
    /// `input_currents` provides external stimulation (length must equal n_neurons).
    /// Pass an empty slice or all-zeros for no external input.
    /// `io` provides the NeuronIO interface for specialized neuron types.
    pub fn tick(&mut self, input_currents: &[i16], io: &mut dyn NeuronIO) {
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

            // 2e. Type-specific pre-spike behavior
            let ntype = flags::neuron_type(self.neurons.flags[i]);
            let binding_slot = self.neurons.binding_slot[i];
            match ntype {
                NeuronType::Sensory if binding_slot > 0 => {
                    if let Some(cfg) = self.bindings.get(binding_slot) {
                        let sensory_current = io.read_sensory(cfg.target, cfg.sensory_offset());
                        self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(sensory_current);
                    }
                }
                NeuronType::Gate if binding_slot > 0 => {
                    if let Some(cfg) = self.bindings.get(binding_slot) {
                        let chem_level = io.read_chemical(cfg.target);
                        // Sensitivity modulates threshold: high chemical → lower threshold
                        let modulation = (chem_level as i32 * cfg.param_a as i32) / 256;
                        self.neurons.threshold[i] = self.neurons.threshold[i].saturating_sub(modulation as i16);
                    }
                }
                NeuronType::Oscillator if binding_slot > 0 => {
                    if let Some(cfg) = self.bindings.get(binding_slot) {
                        let period = cfg.target.max(1) as u64;
                        let amplitude = cfg.param_a as i16 * 64; // Scale to Q8.8
                        let phase_offset = cfg.param_b as u64;
                        // Ramp depolarization based on tick phase
                        let phase = (self.tick_count.wrapping_add(phase_offset)) % period;
                        let ramp = (amplitude * phase as i16) / period as i16;
                        self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(ramp);
                    }
                }
                NeuronType::MemoryMatcher if binding_slot > 0 => {
                    // Non-spike check: compare synaptic input pattern against memory
                    if self.synaptic_current[i] != 0 {
                        if let Some(cfg) = self.bindings.get(binding_slot) {
                            let pattern = self.gather_local_pattern(i);
                            let boost = io.memory_match(cfg.target, &pattern);
                            self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(boost);
                        }
                    }
                }
                _ => {} // Computational, Relay, Motor, MemoryReader — no pre-spike action
            }

            // 2f. Spike check
            if self.neurons.membrane[i] >= self.neurons.threshold[i] {
                self.neurons.spike_out[i] = true;
                self.spike_window[i] = true; // Accumulate into measurement window
                self.neurons.membrane[i] = self.config.reset_potential;
                self.neurons.refract_remaining[i] = self.config.refractory_ticks;

                // Post-synaptic trace bump (this neuron just fired)
                self.neurons.trace[i] = self.neurons.trace[i].saturating_add(self.config.stdp_positive);

                // 2g. Type-specific post-spike behavior
                match ntype {
                    NeuronType::Motor if binding_slot > 0 => {
                        if let Some(cfg) = self.bindings.get(binding_slot) {
                            let magnitude = cfg.param_a as i16 * 128; // Scale to output range
                            io.write_motor(cfg.target, magnitude);
                        }
                    }
                    NeuronType::MemoryReader if binding_slot > 0 => {
                        if let Some(cfg) = self.bindings.get(binding_slot) {
                            let pattern = self.gather_local_pattern(i);
                            let result = io.memory_query(cfg.target, &pattern, cfg.param_b);
                            // Inject query result as current on next tick via direct membrane add
                            self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(result);
                        }
                    }
                    _ => {} // No post-spike action for other types
                }

                spike_count += 1;
            } else {
                self.neurons.spike_out[i] = false;
            }

            // 2h. Decay eligibility trace
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

    /// Gather a local activity pattern from neurons synaptically connected to `neuron_idx`.
    ///
    /// Returns up to 8 membrane potential values from the neuron's synaptic neighbors.
    /// Used by MemoryReader and MemoryMatcher to create query vectors.
    fn gather_local_pattern(&self, neuron_idx: usize) -> Vec<i16> {
        let outgoing = self.synapses.outgoing(neuron_idx as u32);
        outgoing.iter()
            .take(8)
            .map(|syn| {
                let t = syn.target as usize;
                if t < self.n_neurons as usize {
                    self.neurons.membrane[t]
                } else {
                    0
                }
            })
            .collect()
    }

    /// Inject an external signal into a range of neurons.
    ///
    /// Signal polarity determines excitation (+) or inhibition (-).
    /// Magnitude is scaled to Q8.8 current.
    pub fn inject(&mut self, neuron_range: Range<u32>, signal: Signal) {
        let current = signal.as_signed_i32() as i16 * 128; // Scale to Q8.8
        let start = neuron_range.start as usize;
        let end = (neuron_range.end as usize).min(self.n_neurons as usize);

        // Clear spike window — new measurement window begins with each injection
        self.spike_window.fill(false);

        for i in start..end {
            self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(current);
        }
    }

    /// Read output from a range of neurons as Signal vector.
    ///
    /// Reports any neuron that spiked at ANY point during the current
    /// measurement window (since the last `inject()` call). This captures
    /// activity from the entire tick sequence, not just the final tick.
    ///
    /// Biological basis: a neuron that fired during a processing window
    /// is "active" regardless of current refractory state.
    pub fn read_output(&self, neuron_range: Range<u32>) -> Vec<Signal> {
        let start = neuron_range.start as usize;
        let end = (neuron_range.end as usize).min(self.n_neurons as usize);

        (start..end)
            .map(|i| {
                if self.spike_window[i] {
                    Signal::positive(255) // Fired during this window
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
        pool.tick_simple(&[]);
        assert_eq!(pool.spike_count(), 0, "no spikes without input");
        assert_eq!(pool.tick_count, 1);
    }

    #[test]
    fn tick_with_strong_input_causes_spike() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());
        // Inject massive current to force a spike
        let mut input = vec![0i16; 10];
        input[0] = 10000; // Way above threshold

        pool.tick_simple(&input);
        assert!(pool.spike_count() > 0, "strong input should cause spikes");
        // Neuron 0 should have spiked
        assert!(pool.neurons.spike_out[0], "neuron 0 should have spiked");
    }

    #[test]
    fn refractory_period() {
        let mut pool = NeuronPool::new("test", 1, PoolConfig::default());
        let input = vec![10000i16]; // Force spike

        pool.tick_simple(&input);
        assert!(pool.neurons.spike_out[0], "should spike on first tick");

        // During refractory period, even strong input shouldn't spike
        pool.tick_simple(&input);
        assert!(!pool.neurons.spike_out[0], "should NOT spike during refractory");

        pool.tick_simple(&input);
        assert!(!pool.neurons.spike_out[0], "should NOT spike during refractory (tick 2)");

        // After refractory period (2 ticks), should spike again
        pool.tick_simple(&input);
        assert!(pool.neurons.spike_out[0], "should spike again after refractory");
    }

    #[test]
    fn signal_inject_and_read() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());

        // Inject excitatory signal
        pool.inject(0..5, Signal::positive(255));
        pool.tick_simple(&[]);

        let output = pool.read_output(0..10);
        assert_eq!(output.len(), 10);

        // At least some of the injected neurons should have spiked
        let spiked: usize = output.iter().filter(|s| s.is_positive()).count();
        assert!(spiked > 0, "injected neurons should produce output spikes");
    }

    #[test]
    fn propagation_diagnostic_256n() {
        // Simulates the regulation shared pool: 256 neurons, 5% density,
        // inject into 0..32, read from 57..64 after 10 ticks.
        let mut pool = NeuronPool::with_random_connectivity_seeded(
            "regulation", 256, 0.05, PoolConfig::default(), 0xDEAD_BEEF,
        );
        eprintln!("Pool: {} neurons, {} synapses", pool.n_neurons, pool.synapse_count());

        // Inject strong signal (magnitude 40 → current 5120 > threshold gap 3840)
        pool.inject(0..32, Signal::positive(40));

        for t in 0..10 {
            pool.tick_simple(&[]);
            let spikes = pool.spike_count();
            let out = pool.read_output(57..64);
            let out_fired: usize = out.iter().filter(|s| s.is_positive()).count();
            eprintln!("  tick {}: spikes={} output_57_63_fired={}", t, spikes, out_fired);
        }

        let output = pool.read_output(57..64);
        let fired: usize = output.iter().filter(|s| s.is_positive()).count();
        eprintln!("Final output 57-63: {:?} ({} fired)",
            output.iter().map(|s| s.as_signed_i32()).collect::<Vec<_>>(), fired);

        // Second injection (simulating program 2 injecting into same range)
        pool.inject(0..32, Signal::positive(40));
        for t in 0..10 {
            pool.tick_simple(&[]);
            let spikes = pool.spike_count();
            let out = pool.read_output(57..64);
            let out_fired: usize = out.iter().filter(|s| s.is_positive()).count();
            eprintln!("  pass2 tick {}: spikes={} output_57_63_fired={}", t, spikes, out_fired);
        }

        let output2 = pool.read_output(57..64);
        let fired2: usize = output2.iter().filter(|s| s.is_positive()).count();
        eprintln!("Pass2 output 57-63: {:?} ({} fired)",
            output2.iter().map(|s| s.as_signed_i32()).collect::<Vec<_>>(), fired2);
    }

    // ====================================================================
    // Spatial tests
    // ====================================================================

    #[test]
    fn spatial_dims_coords() {
        let dims = SpatialDims::new(4, 4, 4);
        assert_eq!(dims.total(), 64);
        assert_eq!(dims.coords(0), (0, 0, 0));
        assert_eq!(dims.coords(1), (1, 0, 0));
        assert_eq!(dims.coords(4), (0, 1, 0));
        assert_eq!(dims.coords(16), (0, 0, 1));
        assert_eq!(dims.coords(63), (3, 3, 3));
    }

    #[test]
    fn spatial_dims_distance() {
        let dims = SpatialDims::new(8, 8, 8);
        // Same position = 0
        assert_eq!(dims.distance_sq(0, 0), 0);
        // Adjacent in x
        assert_eq!(dims.distance_sq(0, 1), 1);
        // Adjacent in y (8 apart in flat index)
        assert_eq!(dims.distance_sq(0, 8), 1);
        // Adjacent in z (64 apart)
        assert_eq!(dims.distance_sq(0, 64), 1);
        // Diagonal in x,y
        assert_eq!(dims.distance_sq(0, 9), 2);
        // Corner to corner: (0,0,0) to (7,7,7) = 49+49+49 = 147
        assert_eq!(dims.distance_sq(0, 511), 147);
        assert_eq!(dims.max_distance_sq(), 147);
    }

    #[test]
    fn spatial_connectivity_deterministic() {
        let dims = SpatialDims::new(4, 4, 4);
        let a = NeuronPool::with_spatial_connectivity_seeded("a", dims, 0.05, PoolConfig::default(), 42);
        let b = NeuronPool::with_spatial_connectivity_seeded("b", dims, 0.05, PoolConfig::default(), 42);
        assert_eq!(a.synapse_count(), b.synapse_count(), "same seed = same topology");
        assert_eq!(a.dims, dims);
    }

    #[test]
    fn spatial_local_bias() {
        // Nearby neurons should be connected more than distant ones
        let dims = SpatialDims::new(8, 8, 4);
        let pool = NeuronPool::with_spatial_connectivity_seeded("test", dims, 0.05, PoolConfig::default(), 99);

        // Count connections from neuron 0 to neighbors (d_sq <= 3) vs far (d_sq > 20)
        let outgoing = pool.synapses.outgoing(0);
        let mut near = 0usize;
        let mut far = 0usize;
        for syn in outgoing {
            let d_sq = dims.distance_sq(0, syn.target as u32);
            if d_sq <= 3 {
                near += 1;
            } else if d_sq > 20 {
                far += 1;
            }
        }
        eprintln!("[SPATIAL] Neuron 0: near={}, far={}, total={}", near, far, outgoing.len());
        // With Gaussian falloff, near should dominate
        assert!(near >= far, "Gaussian: near connections ({}) should >= far connections ({})", near, far);
    }

    #[test]
    fn spatial_average_density() {
        let dims = SpatialDims::new(4, 4, 4);
        let n = dims.total() as f32;
        let target_density = 0.05f32;
        let pool = NeuronPool::with_spatial_connectivity_seeded("test", dims, target_density, PoolConfig::default(), 77);

        let actual_density = pool.synapse_count() as f32 / (n * (n - 1.0));
        eprintln!("[SPATIAL] Target density: {}, Actual density: {:.4}, synapses: {}", target_density, actual_density, pool.synapse_count());
        // Gaussian normalization targets the requested density. Small grids may
        // deviate more due to discrete sampling. Allow 3x tolerance.
        assert!(
            actual_density > target_density * 0.3 && actual_density < target_density * 3.0,
            "Density {:.4} should be in reasonable range of target {}", actual_density, target_density
        );
    }

    #[test]
    fn flat_constructor_unchanged() {
        // Old constructor should produce flat dims and identical results
        let a = NeuronPool::with_random_connectivity_seeded("a", 64, 0.05, PoolConfig::default(), 42);
        assert_eq!(a.dims, SpatialDims::flat(64));
        assert!(a.synapse_count() > 0);
    }

    #[test]
    fn sentinel_inhibits_boundary() {
        // Sentinels require all axes >= 6 (otherwise skipped to avoid over-inhibition)
        let dims = SpatialDims::new(6, 6, 6);
        let pool = NeuronPool::with_spatial_connectivity_seeded("test", dims, 0.05, PoolConfig::default(), 55);

        // Boundary neurons should be inhibitory
        let boundary_count = (0..dims.total()).filter(|&i| dims.is_boundary(i)).count();
        let interior_count = dims.total() as usize - boundary_count;
        eprintln!("[SENTINEL] Boundary: {}, Interior: {} (6x6x6 = {})", boundary_count, interior_count, dims.total());

        // Check all boundary neurons are inhibitory
        for i in 0..dims.total() {
            if dims.is_boundary(i) {
                assert!(
                    crate::neuron::flags::is_inhibitory(pool.neurons.flags[i as usize]),
                    "Boundary neuron {} should be inhibitory", i
                );
            }
        }

        // Sentinels should have connections to interior neighbors
        let sentinel_synapses: usize = (0..dims.total())
            .filter(|&i| dims.is_boundary(i))
            .map(|i| pool.synapses.outgoing(i).len())
            .sum();
        assert!(sentinel_synapses > 0, "Sentinels should have outgoing connections");
    }

    #[test]
    fn small_grid_skips_sentinels() {
        // Grids with any axis < 6 skip sentinel placement to avoid over-inhibition
        let dims = SpatialDims::new(4, 4, 4);
        let pool = NeuronPool::with_spatial_connectivity_seeded("test", dims, 0.05, PoolConfig::default(), 55);

        // No boundary neurons should be marked inhibitory by sentinels
        // (some may be randomly inhibitory from the base constructor)
        let boundary_flags: Vec<u8> = (0..dims.total())
            .filter(|&i| dims.is_boundary(i))
            .map(|i| pool.neurons.flags[i as usize])
            .collect();

        // At least some boundary neurons should be excitatory (not all forced inhibitory)
        let exc_boundary = boundary_flags.iter()
            .filter(|&&f| !crate::neuron::flags::is_inhibitory(f))
            .count();
        assert!(exc_boundary > 0, "Small grid should have excitatory boundary neurons (no sentinels)");
    }

    // ====================================================================
    // Neuron type tests
    // ====================================================================

    #[test]
    fn neuron_type_encoding() {
        use crate::neuron::{NeuronProfile, NeuronType, flags};

        // Round-trip all types
        for ntype in [
            NeuronType::Computational, NeuronType::Sensory, NeuronType::Motor,
            NeuronType::MemoryReader, NeuronType::MemoryMatcher,
            NeuronType::Gate, NeuronType::Relay, NeuronType::Oscillator,
        ] {
            let f = flags::encode_full(false, NeuronProfile::RegularSpiking, ntype);
            assert_eq!(flags::neuron_type(f), ntype, "Round-trip failed for {:?}", ntype);
            assert!(flags::is_excitatory(f));
            assert_eq!(NeuronProfile::from_flags(f), NeuronProfile::RegularSpiking);
        }

        // Inhibitory + FastSpiking + Gate
        let f = flags::encode_full(true, NeuronProfile::FastSpiking, NeuronType::Gate);
        assert!(flags::is_inhibitory(f));
        assert_eq!(NeuronProfile::from_flags(f), NeuronProfile::FastSpiking);
        assert_eq!(flags::neuron_type(f), NeuronType::Gate);
    }

    #[test]
    fn tick_with_null_io_same_as_tick_simple() {
        let mut pool_a = NeuronPool::with_random_connectivity_seeded("a", 32, 0.05, PoolConfig::default(), 42);
        let mut pool_b = NeuronPool::with_random_connectivity_seeded("b", 32, 0.05, PoolConfig::default(), 42);

        let input = vec![5000i16; 32];
        pool_a.tick_simple(&input);
        pool_b.tick(&input, &mut NullIO);

        // Both should produce identical results
        assert_eq!(pool_a.spike_count(), pool_b.spike_count());
        for i in 0..32 {
            assert_eq!(pool_a.neurons.membrane[i], pool_b.neurons.membrane[i],
                "membrane mismatch at neuron {}", i);
        }
    }

    #[test]
    fn sensory_neuron_reads_input() {
        use crate::binding::BindingConfig;
        use crate::neuron::{NeuronProfile, NeuronType, flags};

        let mut pool = NeuronPool::new("test", 4, PoolConfig::default());

        // Make neuron 0 a Sensory neuron
        pool.neurons.flags[0] = flags::encode_full(false, NeuronProfile::RegularSpiking, NeuronType::Sensory);
        let slot = pool.bindings.add(BindingConfig::sensory(1, 0)).unwrap();
        pool.neurons.binding_slot[0] = slot;

        // Custom IO that returns a known sensory value
        struct TestIO;
        impl NeuronIO for TestIO {
            fn read_sensory(&self, _field_id: u8, _offset: u16) -> i16 { 5000 }
            fn write_motor(&mut self, _: u8, _: i16) {}
            fn memory_query(&mut self, _: u8, _: &[i16], _: u8) -> i16 { 0 }
            fn memory_match(&self, _: u8, _: &[i16]) -> i16 { 0 }
            fn read_chemical(&self, _: u8) -> u8 { 0 }
        }

        let baseline_membrane = pool.neurons.membrane[0];
        pool.tick(&[], &mut TestIO);

        // Sensory neuron should have received the sensory current
        // (it might have spiked, so check either membrane changed or spike occurred)
        let spiked = pool.neurons.spike_out[0];
        let membrane_changed = pool.neurons.membrane[0] != baseline_membrane;
        assert!(spiked || membrane_changed,
            "Sensory neuron should respond to read_sensory input");
    }

    #[test]
    fn motor_neuron_fires_output() {
        use crate::binding::BindingConfig;
        use crate::neuron::{NeuronProfile, NeuronType, flags};
        use std::cell::Cell;

        let mut pool = NeuronPool::new("test", 4, PoolConfig::default());

        // Make neuron 0 a Motor neuron
        pool.neurons.flags[0] = flags::encode_full(false, NeuronProfile::RegularSpiking, NeuronType::Motor);
        let slot = pool.bindings.add(BindingConfig::motor(0, 128)).unwrap();
        pool.neurons.binding_slot[0] = slot;

        struct MotorTestIO { wrote: Cell<bool> }
        impl NeuronIO for MotorTestIO {
            fn read_sensory(&self, _: u8, _: u16) -> i16 { 0 }
            fn write_motor(&mut self, _channel: u8, _magnitude: i16) {
                self.wrote.set(true);
            }
            fn memory_query(&mut self, _: u8, _: &[i16], _: u8) -> i16 { 0 }
            fn memory_match(&self, _: u8, _: &[i16]) -> i16 { 0 }
            fn read_chemical(&self, _: u8) -> u8 { 0 }
        }

        let mut io = MotorTestIO { wrote: Cell::new(false) };

        // Force neuron 0 to spike
        let mut input = vec![0i16; 4];
        input[0] = 10000;
        pool.tick(&input, &mut io);

        assert!(pool.neurons.spike_out[0], "Motor neuron should spike");
        assert!(io.wrote.get(), "Motor neuron should have called write_motor on spike");
    }

    #[test]
    fn gate_neuron_chemical_modulation() {
        use crate::binding::BindingConfig;
        use crate::neuron::{NeuronProfile, NeuronType, flags};

        let mut pool = NeuronPool::new("test", 4, PoolConfig::default());

        // Make neuron 0 a Gate neuron
        pool.neurons.flags[0] = flags::encode_full(false, NeuronProfile::RegularSpiking, NeuronType::Gate);
        let slot = pool.bindings.add(BindingConfig::gate(0, 200)).unwrap();
        pool.neurons.binding_slot[0] = slot;

        struct ChemIO;
        impl NeuronIO for ChemIO {
            fn read_sensory(&self, _: u8, _: u16) -> i16 { 0 }
            fn write_motor(&mut self, _: u8, _: i16) {}
            fn memory_query(&mut self, _: u8, _: &[i16], _: u8) -> i16 { 0 }
            fn memory_match(&self, _: u8, _: &[i16]) -> i16 { 0 }
            fn read_chemical(&self, _chem: u8) -> u8 { 200 } // High chemical level
        }

        let threshold_before = pool.neurons.threshold[0];
        pool.tick(&[], &mut ChemIO);
        let threshold_after = pool.neurons.threshold[0];

        // High chemical + high sensitivity should lower threshold
        assert!(threshold_after < threshold_before,
            "Gate neuron threshold should decrease with high chemical (before={}, after={})",
            threshold_before, threshold_after);
    }

    #[test]
    fn oscillator_fires_periodically() {
        use crate::binding::BindingConfig;
        use crate::neuron::{NeuronProfile, NeuronType, flags};

        let mut pool = NeuronPool::new("test", 1, PoolConfig::default());

        // Make neuron 0 an Oscillator with period=5, high amplitude
        pool.neurons.flags[0] = flags::encode_full(false, NeuronProfile::RegularSpiking, NeuronType::Oscillator);
        let slot = pool.bindings.add(BindingConfig::oscillator(5, 200, 0)).unwrap();
        pool.neurons.binding_slot[0] = slot;

        let mut spike_ticks = Vec::new();
        for t in 0..30 {
            pool.tick_simple(&[]);
            if pool.neurons.spike_out[0] {
                spike_ticks.push(t);
            }
        }

        eprintln!("[OSCILLATOR] Spike ticks: {:?}", spike_ticks);
        // Oscillator should fire multiple times over 30 ticks
        assert!(spike_ticks.len() >= 2,
            "Oscillator should fire multiple times in 30 ticks (fired {} times)", spike_ticks.len());
    }

    #[test]
    fn codec_v2_round_trip() {
        let dims = SpatialDims::new(4, 4, 4);
        let mut pool = NeuronPool::with_spatial_connectivity_seeded("spatial", dims, 0.05, PoolConfig::default(), 42);

        // Add a binding
        use crate::binding::BindingConfig;
        use crate::neuron::{NeuronProfile, NeuronType, flags};
        pool.neurons.flags[0] = flags::encode_full(false, NeuronProfile::RegularSpiking, NeuronType::Sensory);
        let slot = pool.bindings.add(BindingConfig::sensory(3, 256)).unwrap();
        pool.neurons.binding_slot[0] = slot;

        let path = std::env::temp_dir().join("neuropool_test_v2.pool");
        pool.save(&path).expect("save failed");
        let loaded = NeuronPool::load(&path).expect("load failed");

        assert_eq!(loaded.dims, dims);
        assert_eq!(loaded.n_neurons, pool.n_neurons);
        assert_eq!(loaded.synapse_count(), pool.synapse_count());
        assert_eq!(loaded.neurons.binding_slot[0], slot);
        let binding = loaded.bindings.get(slot).unwrap();
        assert_eq!(binding.target, 3);
        assert_eq!(binding.sensory_offset(), 256);
        assert_eq!(flags::neuron_type(loaded.neurons.flags[0]), NeuronType::Sensory);

        std::fs::remove_file(&path).ok();
    }
}
