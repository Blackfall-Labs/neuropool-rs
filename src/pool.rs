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
use crate::template::{SignalType, TemplateType, TemplateRequest, TemplateInstance, TemplateRegistry};

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

    /// Grow the grid by `additional` neurons, extending the deepest dimension.
    ///
    /// The aspect ratio is maintained by growing the dimension with the most
    /// room. For flat pools (h=1, d=1), width is extended.
    pub fn grow(&mut self, additional: u32) {
        // For flat pools just extend w
        if self.h == 1 && self.d == 1 {
            self.w = self.w.saturating_add(additional as u16);
            return;
        }
        // Extend the deepest dimension (d) by enough layers to fit `additional` neurons
        let layer_size = self.w as u32 * self.h as u32;
        if layer_size == 0 { return; }
        let extra_layers = (additional + layer_size - 1) / layer_size;
        self.d = self.d.saturating_add(extra_layers as u16);
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

/// Growth engine configuration — bounds and thresholds for neurogenesis/pruning.
#[derive(Clone, Debug)]
pub struct GrowthConfig {
    /// Maximum neuron count ceiling. Default: initial * 4
    pub max_neurons: u32,
    /// Minimum neuron count floor. Default: initial / 2
    pub min_neurons: u32,
    /// Maximum neurons to grow per cycle. Default: 8
    pub growth_per_cycle: u16,
    /// Maximum neurons to prune per cycle. Default: 16
    pub prune_per_cycle: u16,
    /// Chemical signal threshold to trigger growth. Default: 30
    pub growth_threshold: u16,
    /// Chemical signal threshold to trigger pruning. Default: 20
    pub prune_threshold: u16,
}

impl GrowthConfig {
    /// Create a growth config with defaults derived from initial neuron count.
    pub fn from_initial(n_neurons: u32) -> Self {
        Self {
            max_neurons: n_neurons.saturating_mul(4),
            min_neurons: n_neurons / 2,
            growth_per_cycle: 8,
            prune_per_cycle: 16,
            growth_threshold: 30,
            prune_threshold: 20,
        }
    }
}

impl Default for GrowthConfig {
    fn default() -> Self {
        Self::from_initial(256)
    }
}

/// Neuron type distribution specification — percentages of excitatory neurons
/// to seed as each specialized type.
///
/// Percentages are applied to excitatory non-boundary neurons only.
/// Boundary neurons are always Relay (boundary sentinels).
/// Remainder becomes Computational. Total should not exceed 100.
#[derive(Clone, Debug)]
pub struct TypeDistributionSpec {
    /// Percentage seeded as Gate (chemical-modulated threshold).
    pub gate_pct: u8,
    /// Percentage seeded as Oscillator (autonomous depolarization ramp).
    pub oscillator_pct: u8,
    /// Percentage seeded as Sensory (reads external field).
    pub sensory_pct: u8,
    /// Percentage seeded as Motor (writes output on spike).
    pub motor_pct: u8,
    /// Percentage seeded as MemoryReader + MemoryMatcher (split 50/50).
    pub memory_pct: u8,
}

impl TypeDistributionSpec {
    /// All Computational (no specialization). Used as default.
    pub fn all_computational() -> Self {
        Self { gate_pct: 0, oscillator_pct: 0, sensory_pct: 0, motor_pct: 0, memory_pct: 0 }
    }

    /// Total allocated percentage (must not exceed 100).
    pub fn total_pct(&self) -> u8 {
        self.gate_pct.saturating_add(self.oscillator_pct)
            .saturating_add(self.sensory_pct)
            .saturating_add(self.motor_pct)
            .saturating_add(self.memory_pct)
    }
}

impl Default for TypeDistributionSpec {
    fn default() -> Self {
        Self::all_computational()
    }
}

/// Parameters for binding specialized neurons to external resources.
///
/// After `seed_type_distribution()` assigns neuron types, `seed_bindings()` uses
/// this spec to create `BindingConfig` entries and set `binding_slot` on each
/// specialized neuron. Without bindings, all specialized neurons silently behave
/// as plain Computational neurons (the `if binding_slot > 0` guards in tick skip them).
#[derive(Clone, Debug)]
pub struct BindingSpec {
    /// Sensory: which field to read (field_id for NeuronIO::read_sensory).
    pub sensory_field_id: u8,
    /// Sensory: total field width — offsets are distributed evenly across neurons.
    pub sensory_field_width: u16,
    /// Motor: output channel ID for NeuronIO::write_motor.
    pub motor_channel_id: u8,
    /// Motor: magnitude scale factor (0-255).
    pub motor_scale: u8,
    /// Gate: chemical IDs to cycle across gate neurons (NeuronIO::read_chemical).
    pub gate_chemicals: Vec<u8>,
    /// Gate: sensitivity to chemical modulation (0-255).
    pub gate_sensitivity: u8,
    /// Oscillator: base period in ticks (varied per neuron with jitter).
    pub oscillator_base_period: u8,
    /// Oscillator: peak depolarization amplitude (0-255, scaled by 64 in tick).
    pub oscillator_amplitude: u8,
    /// MemoryReader: bank slot for NeuronIO::memory_query.
    pub memory_bank_slot: u8,
    /// MemoryReader: query dimension (typically 8 = gather_local_pattern size).
    pub memory_query_dim: u8,
    /// MemoryReader: number of top results to consider.
    pub memory_top_k: u8,
    /// MemoryMatcher: minimum match threshold (0-255).
    pub memory_match_threshold: u8,
}

/// Input signals for computing region fitness during evolution.
///
/// These are collected by the brain runtime (astromind-v3) and passed down
/// to the pool's evolution engine. The pool doesn't know where the signals
/// come from — it just scores them.
#[derive(Clone, Copy, Debug)]
pub struct FitnessInput {
    /// Singular coherence for this region's division (0-255).
    pub coherence: u8,
    /// Dopamine level (0-255, baseline ~128).
    pub da: u8,
    /// Cortisol level (0-255, baseline ~30).
    pub cortisol: u8,
    /// Fraction of neurons that spiked in recent window (0.0 - 1.0).
    pub active_ratio: f32,
    /// Fraction of max possible synapses that exist (0.0 - 1.0).
    pub synapse_density: f32,
}

/// Type of structural mutation applied during evolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationType {
    NeuronGrow,
    NeuronPrune,
    SynapsePrune,
    SynapseRegrow,
    TypeRebalance,
    WeightPerturbation,
}

/// A single mutation event recorded in the journal.
#[derive(Clone, Copy, Debug)]
pub struct MutationEntry {
    pub generation: u32,
    pub mutation_type: MutationType,
    /// Signed magnitude — positive = added, negative = removed.
    pub magnitude: i16,
    pub fitness_before: u8,
    pub fitness_after: u8,
}

/// Ring buffer of recent mutation events. Keeps the last N entries.
#[derive(Clone, Debug)]
pub struct MutationJournal {
    entries: Vec<MutationEntry>,
    max_entries: usize,
}

impl MutationJournal {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries,
        }
    }

    /// Record a mutation. Drops oldest entry if full.
    pub fn record(&mut self, entry: MutationEntry) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// All recorded entries, oldest first.
    pub fn entries(&self) -> &[MutationEntry] {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether journal is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Average fitness delta across recent entries. Positive = improving.
    pub fn avg_fitness_delta(&self) -> i16 {
        if self.entries.is_empty() {
            return 0;
        }
        let sum: i32 = self.entries.iter()
            .map(|e| e.fitness_after as i32 - e.fitness_before as i32)
            .sum();
        (sum / self.entries.len() as i32) as i16
    }
}

/// Evolution engine configuration — bounds and thresholds for structural mutation.
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    /// Whether evolution is enabled. Default: true
    pub enabled: bool,
    /// Maximum mutations per evolution call. Default: 4
    pub mutation_budget: u8,
    /// Fitness above this → low mutation rate (elite). Default: 180
    pub elite_threshold: u8,
    /// Fitness below this → high mutation rate (struggling). Default: 80
    pub struggle_threshold: u8,
    /// Mutation probability /256 for elite regions. Default: 25 (~10%)
    pub elite_mutation_prob: u8,
    /// Mutation probability /256 for struggling regions. Default: 128 (~50%)
    pub struggle_mutation_prob: u8,
    /// Max weight delta per synapse during perturbation. Default: 10
    pub weight_perturbation_max: i8,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mutation_budget: 4,
            elite_threshold: 180,
            struggle_threshold: 80,
            elite_mutation_prob: 25,
            struggle_mutation_prob: 128,
            weight_perturbation_max: 10,
        }
    }
}

/// Result of a single evolution call.
#[derive(Clone, Copy, Debug)]
pub struct EvolutionResult {
    pub mutations_applied: u8,
    pub fitness_before: u8,
    pub fitness_after: u8,
    pub generation: u32,
}

/// Lightweight structural snapshot for evolution rollback.
///
/// Captures only what evolution can change — neuron structure and synapse edges.
/// Does NOT include membrane potentials, traces, or delay buffers (those reset
/// on rollback since the pre-mutation dynamic state is meaningless).
#[derive(Clone)]
pub struct PoolCheckpoint {
    pub generation: u32,
    pub n_neurons: u32,
    pub n_excitatory: u32,
    pub n_inhibitory: u32,
    pub flags: Vec<u8>,
    pub thresholds: Vec<i16>,
    pub leak: Vec<u8>,
    pub edges: Vec<(u32, Synapse)>,
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
    /// Growth engine configuration. Default: derived from 256 neurons.
    pub growth: GrowthConfig,
    /// Evolution engine configuration.
    pub evolution: EvolutionConfig,
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
            growth: GrowthConfig::default(),
            evolution: EvolutionConfig::default(),
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

    /// Extend all delay slots to accommodate `additional` new neurons (zeroed).
    fn extend(&mut self, additional: usize) {
        for buf in &mut self.buffers {
            buf.extend(std::iter::repeat(0i16).take(additional));
        }
    }

    /// Shrink all delay slots to `new_size` neurons.
    fn shrink_to(&mut self, new_size: usize) {
        for buf in &mut self.buffers {
            buf.truncate(new_size);
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
    /// Projection currents from inter-regional spike projections.
    /// Set by the region thread before tick(), consumed and zeroed during tick().
    pub projection_current: Vec<i16>,
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
    /// Per-neuron spike count within the current measurement window (since last inject).
    /// Rate-coded output: 0 = never fired, 4 = fired every 3rd tick (max with refractory=2).
    pub spike_window_count: Vec<u8>,
    /// Per-neuron spike counts since last reset. Used by the growth engine
    /// to identify dead (zero-spike) neurons for pruning and active boundaries
    /// for neurogenesis. Reset after each growth_cycle call.
    pub spike_counts: Vec<u32>,
    /// Neuron count at construction — the genome baseline. Never changes after
    /// creation, even as neurons are grown/pruned. Used to compute growth_ratio.
    pub initial_neuron_count: u32,
    /// Rolling chemical exposure per neuron — exponential moving average of
    /// chemical levels seen during growth_cycle. Used by type mutation to detect
    /// neurons consistently exposed to specific chemicals. 0 = no exposure.
    pub chem_exposure: Vec<u8>,
    /// Evolution generation counter. Increments each time evolve_structure runs.
    pub generation: u32,
    /// Journal of recent structural mutations for tracking evolution history.
    pub journal: MutationJournal,
    /// Template registry for computational circuit motifs.
    pub templates: TemplateRegistry,
    /// Spatial bounds for template placement (set by init_spatial).
    pub spatial_bounds: Option<[f32; 3]>,
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
            projection_current: vec![0i16; n_neurons as usize],
            last_spike_count: 0,
            spike_rate: vec![0u16; n_neurons as usize],
            spike_window: vec![false; n_neurons as usize],
            spike_window_count: vec![0u8; n_neurons as usize],
            spike_counts: vec![0u32; n_neurons as usize],
            initial_neuron_count: n_neurons,
            chem_exposure: vec![0u8; n_neurons as usize],
            generation: 0,
            journal: MutationJournal::new(16),
            templates: TemplateRegistry::new(),
            spatial_bounds: None,
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

    /// Seed neuron type distribution across excitatory non-boundary neurons.
    ///
    /// Assigns specialized `NeuronType` to a percentage of eligible neurons
    /// according to the spec. Boundary neurons (Relay sentinels) and inhibitory
    /// neurons are never reassigned. Uses seeded RNG for determinism.
    ///
    /// Returns the number of neurons that were specialized.
    pub fn seed_type_distribution(&mut self, spec: &TypeDistributionSpec, seed: u64) -> u32 {
        if spec.total_pct() == 0 {
            return 0;
        }

        // Collect eligible neuron indices: excitatory + non-boundary + Computational
        let eligible: Vec<usize> = (0..self.n_neurons as usize)
            .filter(|&i| {
                let f = self.neurons.flags[i];
                flags::is_excitatory(f)
                    && flags::neuron_type(f) == NeuronType::Computational
                    && !(self.dims.h > 1 && self.dims.is_boundary(i as u32))
            })
            .collect();

        if eligible.is_empty() {
            return 0;
        }

        // Shuffle eligible indices deterministically
        let mut shuffled = eligible.clone();
        let mut rng = seed;
        let lcg = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s >> 33
        };
        for i in (1..shuffled.len()).rev() {
            let j = lcg(&mut rng) as usize % (i + 1);
            shuffled.swap(i, j);
        }

        let total = shuffled.len();
        let mut cursor = 0usize;

        // Helper: assign `pct` percentage of `total` neurons to `ntype`
        let mut assign = |pct: u8, ntype: NeuronType, count: &mut u32| {
            let n = (total * pct as usize) / 100;
            let end = (cursor + n).min(shuffled.len());
            for &idx in &shuffled[cursor..end] {
                let old_flags = self.neurons.flags[idx];
                let inhibitory = flags::is_inhibitory(old_flags);
                let profile = crate::neuron::NeuronProfile::from_flags(old_flags);
                self.neurons.flags[idx] = flags::encode_full(inhibitory, profile, ntype);
                *count += 1;
            }
            cursor = end;
        };

        let mut specialized = 0u32;
        assign(spec.gate_pct, NeuronType::Gate, &mut specialized);
        assign(spec.oscillator_pct, NeuronType::Oscillator, &mut specialized);
        assign(spec.sensory_pct, NeuronType::Sensory, &mut specialized);
        assign(spec.motor_pct, NeuronType::Motor, &mut specialized);

        // Memory split: half MemoryReader, half MemoryMatcher
        if spec.memory_pct > 0 {
            let reader_pct = spec.memory_pct / 2;
            let matcher_pct = spec.memory_pct - reader_pct;
            assign(reader_pct, NeuronType::MemoryReader, &mut specialized);
            assign(matcher_pct, NeuronType::MemoryMatcher, &mut specialized);
        }

        if specialized > 0 {
            log::debug!(
                "[TYPE_SEED] {}: specialized {} of {} eligible neurons",
                self.name, specialized, total
            );
        }

        specialized
    }

    /// Create BindingConfig entries for all typed neurons that lack a binding_slot.
    ///
    /// Must be called AFTER `seed_type_distribution()`. Iterates all neurons,
    /// creates a `BindingConfig` for each specialized type based on the `BindingSpec`,
    /// and assigns the neuron's `binding_slot` to the new table entry.
    ///
    /// Returns the number of neurons successfully bound.
    pub fn seed_bindings(&mut self, spec: &BindingSpec, seed: u64) -> u32 {
        use crate::binding::BindingConfig;

        let n = self.n_neurons as usize;
        let mut bound = 0u32;

        // LCG for deterministic jitter (oscillator period variation)
        let mut rng = seed;
        let lcg = |s: &mut u64| -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s >> 33
        };

        // First pass: count neurons of each type for offset distribution
        let mut sensory_total = 0u32;
        let mut gate_total = 0u32;
        let mut oscillator_total = 0u32;
        for i in 0..n {
            if self.neurons.binding_slot[i] != 0 { continue; }
            match flags::neuron_type(self.neurons.flags[i]) {
                NeuronType::Sensory => sensory_total += 1,
                NeuronType::Gate => gate_total += 1,
                NeuronType::Oscillator => oscillator_total += 1,
                _ => {}
            }
        }

        // Second pass: create bindings
        let mut sensory_idx = 0u32;
        let mut gate_idx = 0u32;
        let mut oscillator_idx = 0u32;

        for i in 0..n {
            if self.neurons.binding_slot[i] != 0 { continue; }

            let ntype = flags::neuron_type(self.neurons.flags[i]);
            let config = match ntype {
                NeuronType::Sensory => {
                    // Distribute offsets evenly across field width
                    let offset = if sensory_total > 1 {
                        (sensory_idx as u32 * spec.sensory_field_width as u32
                            / sensory_total) as u16
                    } else {
                        0
                    };
                    sensory_idx += 1;
                    Some(BindingConfig::sensory(spec.sensory_field_id, offset))
                }
                NeuronType::Motor => {
                    Some(BindingConfig::motor(spec.motor_channel_id, spec.motor_scale))
                }
                NeuronType::Gate => {
                    // Cycle through available chemicals
                    let chem = if spec.gate_chemicals.is_empty() {
                        0
                    } else {
                        spec.gate_chemicals[gate_idx as usize % spec.gate_chemicals.len()]
                    };
                    gate_idx += 1;
                    Some(BindingConfig::gate(chem, spec.gate_sensitivity))
                }
                NeuronType::Oscillator => {
                    // Base period with deterministic jitter (±25%)
                    let base = spec.oscillator_base_period.max(2) as u32;
                    let jitter_range = base / 4;
                    let jitter = if jitter_range > 0 {
                        (lcg(&mut rng) % (jitter_range as u64 * 2 + 1)) as u32
                    } else {
                        0
                    };
                    let period = (base - jitter_range + jitter).clamp(2, 255) as u8;

                    // Spread phase offsets evenly across neurons
                    let phase = if oscillator_total > 1 {
                        ((oscillator_idx as u32 * period as u32) / oscillator_total) as u8
                    } else {
                        0
                    };
                    oscillator_idx += 1;
                    Some(BindingConfig::oscillator(period, spec.oscillator_amplitude, phase))
                }
                NeuronType::MemoryReader => {
                    Some(BindingConfig::memory_reader(
                        spec.memory_bank_slot,
                        spec.memory_query_dim,
                        spec.memory_top_k,
                    ))
                }
                NeuronType::MemoryMatcher => {
                    Some(BindingConfig::memory_matcher(
                        spec.memory_bank_slot,
                        spec.memory_match_threshold,
                    ))
                }
                _ => None, // Computational, Relay — no binding needed
            };

            if let Some(cfg) = config {
                if let Some(slot) = self.bindings.add(cfg) {
                    self.neurons.binding_slot[i] = slot;
                    bound += 1;
                } else {
                    // BindingTable full (255 entries) — stop binding
                    log::warn!(
                        "[BINDING] {}: binding table full after {} entries",
                        self.name, bound
                    );
                    break;
                }
            }
        }

        if bound > 0 {
            log::debug!(
                "[BINDING] {}: bound {} neurons (S={}, G={}, O={}, M+M={}+{})",
                self.name, bound,
                sensory_idx, gate_idx, oscillator_idx,
                sensory_total.min(bound), // approximate — counts include unbound
                gate_total.min(bound),
            );
        }

        bound
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

            // 2d2. Add inter-regional projection current
            if self.projection_current[i] != 0 {
                self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(self.projection_current[i]);
                self.projection_current[i] = 0;
            }

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
                        let amplitude = cfg.param_a as i32 * 64; // Scale to Q8.8
                        let phase_offset = cfg.param_b as u64;
                        // Ramp depolarization based on tick phase
                        let phase = (self.tick_count.wrapping_add(phase_offset)) % period;
                        let ramp = ((amplitude * phase as i32) / period as i32).clamp(-32768, 32767) as i16;
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
                self.spike_window[i] = true;
                self.spike_window_count[i] = self.spike_window_count[i].saturating_add(1);
                self.neurons.membrane[i] = self.config.reset_potential;
                // Per-profile refractory: 0 = use profile default, non-zero = global override
                self.neurons.refract_remaining[i] = if self.config.refractory_ticks == 0 {
                    crate::neuron::NeuronProfile::from_flags(self.neurons.flags[i]).default_refractory()
                } else {
                    self.config.refractory_ticks
                };

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
                self.spike_counts[i] = self.spike_counts[i].saturating_add(1);
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

        // 4. Homeostatic threshold adjustment (per-profile scaled)
        if self.config.homeostatic_rate > 0 && self.tick_count % 100 == 0 {
            let rate = self.config.homeostatic_rate as i16;
            for i in 0..n {
                // Update running spike rate (exponential moving average)
                let spiked = if self.neurons.spike_out[i] { 256u16 } else { 0 };
                self.spike_rate[i] = ((self.spike_rate[i] as u32 * 255 + spiked as u32) / 256) as u16;

                // Per-profile adaptation scale:
                //   FastSpiking → 2x (interneurons adapt quickly to maintain fast inhibition)
                //   IntrinsicBursting → 0 (no homeostatic adaptation, preserves bursting capability)
                //   RegularSpiking → 1x (normal)
                let profile = crate::neuron::NeuronProfile::from_flags(self.neurons.flags[i]);
                let adapt_scale: i16 = match profile {
                    crate::neuron::NeuronProfile::FastSpiking => 2,
                    crate::neuron::NeuronProfile::IntrinsicBursting => 0,
                    _ => 1,
                };

                if adapt_scale == 0 { continue; }
                let scaled_rate = rate * adapt_scale;

                // Target rate: ~5% of ticks should produce a spike (= ~13 in Q8.8)
                let target_rate = 13u16;
                if self.spike_rate[i] > target_rate * 2 {
                    // Firing too much — raise threshold
                    self.neurons.threshold[i] = self.neurons.threshold[i].saturating_add(scaled_rate);
                } else if self.spike_rate[i] < target_rate / 2 {
                    // Too silent — lower threshold
                    self.neurons.threshold[i] = self.neurons.threshold[i].saturating_sub(scaled_rate);
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
        self.spike_window_count.fill(0);

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
                let count = self.spike_window_count[i];
                if count > 0 {
                    // Rate-coded: scale spike count to magnitude.
                    // With refractory=2, max ~4 spikes in 12 ticks.
                    let magnitude = (count as u16 * 64).min(255) as u8;
                    Signal::positive(magnitude)
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

    // --- Activity tracking for growth engine ---

    /// Per-neuron spike counts since last reset.
    ///
    /// Used by the growth engine to identify dead neurons (zero spikes)
    /// and active boundaries (high spike counts) for neurogenesis/pruning.
    #[inline]
    pub fn neuron_activities(&self) -> &[u32] {
        &self.spike_counts
    }

    /// Reset all per-neuron spike counts to zero.
    ///
    /// Called after growth_cycle to start a new observation window.
    pub fn reset_activities(&mut self) {
        self.spike_counts.fill(0);
    }

    /// Count of neurons that spiked at least once since last reset.
    pub fn active_neuron_count(&self) -> u32 {
        self.spike_counts.iter().filter(|&&c| c > 0).count() as u32
    }

    /// Growth ratio: current neuron count / initial (genome) count.
    ///
    /// Returns 1.0 if no growth/pruning has occurred.
    pub fn growth_ratio(&self) -> f32 {
        if self.initial_neuron_count == 0 {
            return 1.0;
        }
        self.n_neurons as f32 / self.initial_neuron_count as f32
    }

    // --- Neurogenesis (A2) ---

    /// Grow `additional` neurons into the pool using deterministic RNG.
    ///
    /// New neurons are initialized at resting potential with Dale's Law ratio
    /// (80% excitatory, 20% inhibitory). They have no synapses — synaptogenesis
    /// must run separately to wire them in.
    ///
    /// For spatial pools, the grid's deepest dimension is extended to fit.
    /// Returns the number of neurons actually added.
    pub fn grow_neurons_seeded(&mut self, additional: u32, _seed: u64) -> u32 {
        if additional == 0 { return 0; }

        let n_add = additional as usize;
        let n_exc = (additional * 4) / 5; // 80% excitatory
        let n_inh = additional - n_exc;

        // Extend SoA neuron arrays
        self.neurons.extend(
            additional,
            n_exc,
            self.config.resting_potential,
            self.config.spike_threshold,
        );

        // Extend auxiliary per-neuron arrays
        self.synaptic_current.extend(std::iter::repeat(0i16).take(n_add));
        self.projection_current.extend(std::iter::repeat(0i16).take(n_add));
        self.spike_rate.extend(std::iter::repeat(0u16).take(n_add));
        self.spike_window.extend(std::iter::repeat(false).take(n_add));
        self.spike_window_count.extend(std::iter::repeat(0u8).take(n_add));
        self.spike_counts.extend(std::iter::repeat(0u32).take(n_add));
        self.chem_exposure.extend(std::iter::repeat(0u8).take(n_add));

        // Extend delay buffer
        self.delay_buf.extend(n_add);

        // Extend CSR row_ptr — new neurons have zero outgoing synapses
        let last_ptr = *self.synapses.row_ptr.last().unwrap_or(&0);
        for _ in 0..additional {
            self.synapses.row_ptr.push(last_ptr);
        }

        // Update counts
        self.n_neurons += additional;
        self.n_excitatory += n_exc;
        self.n_inhibitory += n_inh;

        // Update spatial dims
        self.dims.grow(additional);

        additional
    }

    // --- Neuron Pruning (A3) ---

    /// Remove neurons at the given indices from the pool.
    ///
    /// Uses swap-remove for O(1) per-element removal from SoA arrays.
    /// Rebuilds the CSR synapse store to remap targets and remove dangling edges.
    /// Returns the number of neurons actually pruned.
    pub fn prune_neurons(&mut self, indices: &[u32]) -> u32 {
        if indices.is_empty() { return 0; }

        let n_before = self.n_neurons as usize;

        // Sort descending for safe swap-remove
        let mut sorted: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        sorted.dedup();
        // Filter out-of-bounds
        sorted.retain(|&i| i < n_before);

        if sorted.is_empty() { return 0; }

        let n_removed = sorted.len() as u32;

        // Build a removal set for synapse remapping
        let mut removed_set = vec![false; n_before];
        for &idx in &sorted {
            removed_set[idx] = true;
        }

        // Build index remap: old_index → new_index (u32::MAX if removed)
        // After swap-remove in descending order, element at `idx` is replaced by
        // the last element, then the vec shrinks. This is complex to track with CSR
        // so we rebuild CSR from scratch instead.
        let mut remap = vec![0u32; n_before];
        let mut new_idx = 0u32;
        for old in 0..n_before {
            if removed_set[old] {
                remap[old] = u32::MAX;
            } else {
                remap[old] = new_idx;
                new_idx += 1;
            }
        }
        let n_after = new_idx;

        // Rebuild CSR: collect surviving edges with remapped source/target indices
        let mut new_edges: Vec<(u32, Synapse)> = Vec::new();
        for src in 0..n_before {
            if removed_set[src] { continue; }
            let outgoing = self.synapses.outgoing(src as u32);
            for syn in outgoing {
                let tgt = syn.target as usize;
                if tgt >= n_before || removed_set[tgt] { continue; }
                let mut new_syn = *syn;
                new_syn.target = remap[tgt] as u16;
                new_edges.push((remap[src], new_syn));
            }
        }
        self.synapses = SynapseStore::from_edges(n_after, new_edges);

        // Compact SoA arrays: keep only non-removed neurons in order
        // (We can't use swap_remove because CSR remap assumes stable ordering)
        let keep: Vec<usize> = (0..n_before).filter(|i| !removed_set[*i]).collect();

        macro_rules! compact_vec {
            ($vec:expr) => {
                $vec = keep.iter().map(|&i| $vec[i]).collect();
            };
        }

        compact_vec!(self.neurons.membrane);
        compact_vec!(self.neurons.threshold);
        compact_vec!(self.neurons.leak);
        compact_vec!(self.neurons.refract_remaining);
        compact_vec!(self.neurons.flags);
        compact_vec!(self.neurons.trace);
        compact_vec!(self.neurons.spike_out);
        compact_vec!(self.neurons.binding_slot);
        compact_vec!(self.synaptic_current);
        compact_vec!(self.projection_current);
        compact_vec!(self.spike_rate);
        compact_vec!(self.spike_window);
        compact_vec!(self.spike_window_count);
        compact_vec!(self.spike_counts);
        compact_vec!(self.chem_exposure);

        // Shrink delay buffer
        self.delay_buf.shrink_to(n_after as usize);

        // Recount excitatory/inhibitory
        let mut exc = 0u32;
        let mut inh = 0u32;
        for &f in &self.neurons.flags {
            if flags::is_inhibitory(f) { inh += 1; } else { exc += 1; }
        }
        self.n_neurons = n_after;
        self.n_excitatory = exc;
        self.n_inhibitory = inh;

        // Note: dims is NOT shrunk — the grid retains its shape even when
        // neurons are pruned. The indices just become "empty slots" conceptually.
        // This matches biological reality: cortical columns don't resize when
        // neurons die, the space remains.

        n_removed
    }

    // --- Neuron Migration (A5) ---

    /// Migrate neurons toward activity hotspots by swapping grid positions.
    ///
    /// For spatial pools only. Low-activity neurons adjacent to high-activity
    /// neurons swap positions, causing active clusters to attract neighbors.
    /// "Migration" is position-swapping — neurons exchange grid positions,
    /// which means swapping all their SoA fields and remapping synapse targets.
    ///
    /// Max migrations per call: `n_neurons / 32`. Seeded RNG for tie-breaking.
    /// Returns count of swaps performed.
    pub fn migrate_neurons(&mut self, seed: u64) -> u32 {
        // Skip for flat (1D) pools — no spatial structure
        if self.dims.h == 1 && self.dims.d == 1 {
            return 0;
        }

        let n = self.n_neurons as usize;
        if n < 4 { return 0; }

        let max_swaps = (n / 32).max(1);
        let mut swaps = 0u32;

        // Simple LCG for tie-breaking
        let mut rng_state: u64 = seed ^ (self.tick_count.wrapping_mul(0xBEEF));
        let lcg_next = |state: &mut u64| -> u32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as u32
        };

        // Find candidate pairs: low-activity neuron adjacent to high-activity neuron
        // We iterate in a randomized order to avoid bias
        let mut candidates: Vec<(usize, usize)> = Vec::new();

        for i in 0..n {
            let my_count = self.spike_counts[i];
            // Look for neighbors with much higher activity
            for j in 0..n {
                if i == j { continue; }
                let d_sq = self.dims.distance_sq(i as u32, j as u32);
                if d_sq > 3 { continue; } // Only immediate neighbors (distance <= sqrt(3))

                let neighbor_count = self.spike_counts[j];
                // Swap if neighbor has >2x activity and I have low activity
                if neighbor_count > my_count.saturating_mul(2) + 1 && my_count < 3 {
                    candidates.push((i, j));
                }
            }
        }

        // Shuffle candidates using LCG
        for k in (1..candidates.len()).rev() {
            let swap_idx = lcg_next(&mut rng_state) as usize % (k + 1);
            candidates.swap(k, swap_idx);
        }

        // Track which neurons have already been swapped this cycle
        let mut swapped = vec![false; n];

        for (low_idx, high_idx) in candidates {
            if swaps >= max_swaps as u32 { break; }
            if swapped[low_idx] || swapped[high_idx] { continue; }

            // Swap all SoA fields between the two neurons
            self.neurons.membrane.swap(low_idx, high_idx);
            self.neurons.threshold.swap(low_idx, high_idx);
            self.neurons.leak.swap(low_idx, high_idx);
            self.neurons.refract_remaining.swap(low_idx, high_idx);
            self.neurons.flags.swap(low_idx, high_idx);
            self.neurons.trace.swap(low_idx, high_idx);
            self.neurons.spike_out.swap(low_idx, high_idx);
            self.neurons.binding_slot.swap(low_idx, high_idx);
            self.synaptic_current.swap(low_idx, high_idx);
            self.projection_current.swap(low_idx, high_idx);
            self.spike_rate.swap(low_idx, high_idx);
            self.spike_window.swap(low_idx, high_idx);
            self.spike_window_count.swap(low_idx, high_idx);
            self.spike_counts.swap(low_idx, high_idx);

            // Remap synapse targets: any synapse targeting low_idx now targets high_idx and vice versa
            let a = low_idx as u16;
            let b = high_idx as u16;
            for syn in self.synapses.synapses.iter_mut() {
                if syn.target == a {
                    syn.target = b;
                } else if syn.target == b {
                    syn.target = a;
                }
            }

            // Swap the CSR row segments (outgoing synapses from each neuron)
            // Since CSR is indexed by neuron, swapping the neurons means we need
            // to swap their outgoing synapse blocks too. This is handled by
            // swapping the actual synapse data in the contiguous array.
            // For CSR, we need to extract, swap, and rebuild the two rows.
            let start_a = self.synapses.row_ptr[low_idx] as usize;
            let end_a = self.synapses.row_ptr[low_idx + 1] as usize;
            let start_b = self.synapses.row_ptr[high_idx] as usize;
            let end_b = self.synapses.row_ptr[high_idx + 1] as usize;

            let syns_a: Vec<_> = self.synapses.synapses[start_a..end_a].to_vec();
            let syns_b: Vec<_> = self.synapses.synapses[start_b..end_b].to_vec();
            let len_a = syns_a.len();
            let len_b = syns_b.len();

            if len_a != len_b {
                // Different number of outgoing synapses — need to rebuild CSR
                // This is the expensive path, but migration is infrequent
                let mut edges: Vec<(u32, crate::synapse::Synapse)> = Vec::new();
                for src in 0..n {
                    let outgoing = if src == low_idx {
                        &syns_b[..]
                    } else if src == high_idx {
                        &syns_a[..]
                    } else {
                        let s = self.synapses.row_ptr[src] as usize;
                        let e = self.synapses.row_ptr[src + 1] as usize;
                        &self.synapses.synapses[s..e]
                    };
                    for syn in outgoing {
                        edges.push((src as u32, *syn));
                    }
                }
                self.synapses = SynapseStore::from_edges(self.n_neurons, edges);
            } else {
                // Same length — can swap in place
                for k in 0..len_a {
                    self.synapses.synapses[start_a + k] = syns_b[k];
                    self.synapses.synapses[start_b + k] = syns_a[k];
                }
            }

            swapped[low_idx] = true;
            swapped[high_idx] = true;
            swaps += 1;
        }

        if swaps > 0 {
            log::debug!(
                "[MIGRATION] {}: {} position swaps performed",
                self.name, swaps
            );
        }

        swaps
    }

    /// Take a structural checkpoint for potential rollback.
    ///
    /// Captures neuron flags, thresholds, leak rates, and all synapse edges.
    /// Cheap for small pools (hundreds of neurons), proportional to pool size.
    pub fn checkpoint(&self) -> PoolCheckpoint {
        let n = self.n_neurons as usize;
        let mut edges = Vec::new();
        for src in 0..self.n_neurons {
            for syn in self.synapses.outgoing(src) {
                edges.push((src, *syn));
            }
        }
        PoolCheckpoint {
            generation: self.generation,
            n_neurons: self.n_neurons,
            n_excitatory: self.n_excitatory,
            n_inhibitory: self.n_inhibitory,
            flags: self.neurons.flags[..n].to_vec(),
            thresholds: self.neurons.threshold[..n].to_vec(),
            leak: self.neurons.leak[..n].to_vec(),
            edges,
        }
    }

    /// Rollback structural state from a checkpoint.
    ///
    /// Restores neuron flags, thresholds, leak, and synapse edges.
    /// Resets membrane, traces, and activity to resting state (clean slate).
    pub fn rollback(&mut self, cp: &PoolCheckpoint) {
        let n = cp.n_neurons as usize;
        self.n_neurons = cp.n_neurons;
        self.n_excitatory = cp.n_excitatory;
        self.n_inhibitory = cp.n_inhibitory;
        self.dims = SpatialDims::flat(cp.n_neurons);

        // Resize arrays to checkpoint size
        self.neurons.flags.resize(n, 0);
        self.neurons.threshold.resize(n, self.config.spike_threshold);
        self.neurons.leak.resize(n, 0u8);
        self.neurons.membrane.resize(n, self.config.resting_potential);
        self.neurons.refract_remaining.resize(n, 0);
        self.neurons.trace.resize(n, 0);
        self.spike_rate.resize(n, 0);
        self.spike_window.resize(n, false);
        self.spike_window_count.resize(n, 0);
        self.spike_counts.resize(n, 0);
        self.chem_exposure.resize(n, 0);
        self.synaptic_current.resize(n, 0);
        self.projection_current.resize(n, 0);

        // Restore structural state
        self.neurons.flags[..n].copy_from_slice(&cp.flags);
        self.neurons.threshold[..n].copy_from_slice(&cp.thresholds);
        self.neurons.leak[..n].copy_from_slice(&cp.leak);

        // Reset dynamic state to resting
        for i in 0..n {
            self.neurons.membrane[i] = self.config.resting_potential;
            self.neurons.refract_remaining[i] = 0;
            self.neurons.trace[i] = 0;
            self.spike_rate[i] = 0;
            self.spike_window[i] = false;
            self.spike_window_count[i] = 0;
            self.spike_counts[i] = 0;
            self.chem_exposure[i] = 0;
        }

        // Restore synapse edges
        self.synapses = SynapseStore::from_edges(cp.n_neurons, cp.edges.clone());

        // Rebuild delay buffer for new size
        self.delay_buf = DelayBuffer::new(n, self.config.max_delay);

        log::debug!(
            "[ROLLBACK] {}: restored to gen {} ({} neurons, {} edges)",
            self.name, cp.generation, cp.n_neurons, cp.edges.len()
        );
    }

    /// Evolve the pool's structure based on fitness signals.
    ///
    /// Regions that perform well (high fitness) get conservative mutations.
    /// Struggling regions (low fitness) get aggressive rewiring.
    /// Mutations are bounded by `EvolutionConfig::mutation_budget`.
    ///
    /// Returns an `EvolutionResult` with before/after fitness and generation.
    pub fn evolve_structure(&mut self, fitness: &FitnessInput, seed: u64) -> EvolutionResult {
        let evo = &self.config.evolution.clone();
        if !evo.enabled {
            return EvolutionResult {
                mutations_applied: 0,
                fitness_before: self.compute_fitness(fitness),
                fitness_after: self.compute_fitness(fitness),
                generation: self.generation,
            };
        }

        let fitness_before = self.compute_fitness(fitness);

        // Determine mutation probability based on fitness tier
        let mutation_prob = if fitness_before >= evo.elite_threshold {
            evo.elite_mutation_prob
        } else if fitness_before <= evo.struggle_threshold {
            evo.struggle_mutation_prob
        } else {
            // Linear interpolation between struggle and elite
            let range = evo.elite_threshold.saturating_sub(evo.struggle_threshold) as u16;
            let pos = fitness_before.saturating_sub(evo.struggle_threshold) as u16;
            let prob_range = evo.struggle_mutation_prob.saturating_sub(evo.elite_mutation_prob) as u16;
            (evo.struggle_mutation_prob as u16 - (pos * prob_range / range.max(1))) as u8
        };

        // Simple LCG seeded PRNG
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let lcg_next = |state: &mut u64| -> u32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as u32
        };

        // Roll dice — skip if no mutation this cycle
        let roll = (lcg_next(&mut rng) & 0xFF) as u8;
        if roll >= mutation_prob {
            self.generation += 1;
            return EvolutionResult {
                mutations_applied: 0,
                fitness_before,
                fitness_after: fitness_before,
                generation: self.generation,
            };
        }

        // Take checkpoint before mutations for potential rollback
        let checkpoint = self.checkpoint();

        // Select and apply mutations based on pool state
        let mut mutations_applied = 0u8;
        let budget = evo.mutation_budget;

        for _ in 0..budget {
            let mutation_roll = lcg_next(&mut rng) % 100;

            // Weighted selection based on what the pool needs
            let mutation_type = if fitness.active_ratio < 0.05 && mutation_roll < 40 {
                // Nearly silent → grow neurons
                MutationType::NeuronGrow
            } else if fitness.active_ratio > 0.50 && fitness_before < evo.struggle_threshold && mutation_roll < 30 {
                // Hyperactive + low fitness → prune dead neurons
                MutationType::NeuronPrune
            } else if fitness.synapse_density < 0.15 && mutation_roll < 40 {
                // Very sparse → regrow synapses
                MutationType::SynapseRegrow
            } else if fitness.synapse_density > 0.70 && fitness_before < evo.struggle_threshold && mutation_roll < 30 {
                // Dense but struggling → prune weak synapses
                MutationType::SynapsePrune
            } else if mutation_roll < 60 {
                // Default mutation: weight perturbation (most conservative)
                MutationType::WeightPerturbation
            } else if mutation_roll < 80 {
                MutationType::SynapseRegrow
            } else {
                MutationType::TypeRebalance
            };

            let magnitude = match mutation_type {
                MutationType::NeuronGrow => {
                    let to_grow = ((lcg_next(&mut rng) % 4) + 1).min(
                        self.config.growth.max_neurons.saturating_sub(self.n_neurons)
                    );
                    if to_grow > 0 {
                        let grown = self.grow_neurons_seeded(to_grow, lcg_next(&mut rng) as u64);
                        grown as i16
                    } else {
                        0
                    }
                }
                MutationType::NeuronPrune => {
                    // Find dead neurons (zero spike count) and prune some
                    let dead: Vec<u32> = (0..self.n_neurons)
                        .filter(|&i| self.spike_counts[i as usize] == 0)
                        .take(4)
                        .collect();
                    if !dead.is_empty() {
                        let pruned = self.prune_neurons(&dead);
                        -(pruned as i16)
                    } else {
                        0
                    }
                }
                MutationType::SynapsePrune => {
                    let pruned = self.prune_dead();
                    -(pruned as i16)
                }
                MutationType::SynapseRegrow => {
                    // Bypass ACh gate — evolve_structure forces synaptogenesis
                    let created = self.synaptogenesis(255); // max ACh to bypass gate
                    created as i16
                }
                MutationType::TypeRebalance => {
                    // Use existing type_plasticity with elevated DA
                    let mutations = self.type_plasticity(200, lcg_next(&mut rng) as u64);
                    mutations as i16
                }
                MutationType::WeightPerturbation => {
                    let max_delta = evo.weight_perturbation_max;
                    let mut perturbed = 0i16;
                    let n_synapses = self.synapses.synapses.len();
                    // Perturb up to 16 random synapses
                    let n_to_perturb = 16.min(n_synapses);
                    for _ in 0..n_to_perturb {
                        let idx = lcg_next(&mut rng) as usize % n_synapses.max(1);
                        if idx < n_synapses {
                            let syn = &mut self.synapses.synapses[idx];
                            if syn.thermal_state() == crate::synapse::ThermalState::Cold {
                                continue; // Don't perturb frozen synapses
                            }
                            let delta = ((lcg_next(&mut rng) % (2 * max_delta as u32 + 1)) as i8)
                                .wrapping_sub(max_delta);
                            let old = syn.weight as i16;
                            let new_w = (old + delta as i16).clamp(-127, 127) as i8;
                            syn.weight = new_w;
                            perturbed += 1;
                        }
                    }
                    perturbed
                }
            };

            if magnitude != 0 {
                mutations_applied += 1;
                self.journal.record(MutationEntry {
                    generation: self.generation,
                    mutation_type,
                    magnitude,
                    fitness_before,
                    fitness_after: 0, // filled in below
                });
            }
        }

        let fitness_after = self.compute_fitness(fitness);

        // Rollback if significant regression (fitness dropped by more than 10)
        if mutations_applied > 0 && fitness_before > fitness_after.saturating_add(10) {
            self.rollback(&checkpoint);
            self.generation += 1;

            log::debug!(
                "[EVOLVE] {}: gen {} ROLLBACK (fitness {} → {} regressed)",
                self.name, self.generation, fitness_before, fitness_after
            );

            return EvolutionResult {
                mutations_applied: 0,
                fitness_before,
                fitness_after: fitness_before, // restored
                generation: self.generation,
            };
        }

        // Update journal entries with actual fitness_after
        for entry in self.journal.entries.iter_mut().rev().take(mutations_applied as usize) {
            entry.fitness_after = fitness_after;
        }

        self.generation += 1;

        log::debug!(
            "[EVOLVE] {}: gen {} fitness {} → {} ({} mutations)",
            self.name, self.generation, fitness_before, fitness_after, mutations_applied
        );

        EvolutionResult {
            mutations_applied,
            fitness_before,
            fitness_after,
            generation: self.generation,
        }
    }

    /// Compute a fitness score (0-255) from external signals.
    ///
    /// Higher fitness = region is performing well. Formula components:
    /// - **Coherence** (0-255): direct measure of useful computation
    /// - **DA reward** (0-127): dopamine above baseline signals success
    /// - **Cortisol penalty** (0-127): cortisol above baseline signals distress
    /// - **Activity health** (0-64): penalizes both silent and saturated pools
    /// - **Density health** (0-64): penalizes both too-sparse and too-dense wiring
    pub fn compute_fitness(&self, input: &FitnessInput) -> u8 {
        // Coherence: worth up to 128 points (half the score)
        let coherence_score = (input.coherence as u16) / 2;

        // DA reward: above baseline 128 → positive, below → 0
        let da_reward = input.da.saturating_sub(128) as u16; // 0-127

        // Cortisol penalty: above baseline 30 → negative
        let cortisol_penalty = input.cortisol.saturating_sub(30) as u16; // 0-225

        // Activity health: bell curve centered at 0.15-0.40 (healthy range)
        let activity_health = if input.active_ratio < 0.05 {
            // Nearly silent — unhealthy
            (input.active_ratio * 20.0 * 64.0) as u16
        } else if input.active_ratio <= 0.40 {
            64 // Sweet spot
        } else if input.active_ratio <= 0.80 {
            // Increasingly saturated
            ((1.0 - (input.active_ratio - 0.40) / 0.40) * 64.0) as u16
        } else {
            0 // Epileptic-level firing — very unhealthy
        };

        // Density health: bell curve centered at 0.20-0.80 (healthy range)
        let density_health = if input.synapse_density < 0.10 {
            // Too sparse — limited connectivity
            (input.synapse_density * 10.0 * 64.0) as u16
        } else if input.synapse_density <= 0.80 {
            64 // Healthy range
        } else {
            // Too dense — saturated, no room for growth
            ((1.0 - (input.synapse_density - 0.80) / 0.20).max(0.0) * 64.0) as u16
        };

        // Combine: max theoretical = 128 + 127 + 64 + 64 = 383
        // Scale to 0-255
        let raw = coherence_score
            .saturating_add(da_reward)
            .saturating_add(activity_health)
            .saturating_add(density_health)
            .saturating_sub(cortisol_penalty);

        // Scale: 383 → 255 (multiply by 2/3)
        let scaled = (raw * 170) / 256; // 170/256 ≈ 0.664
        scaled.min(255) as u8
    }

    // =========================================================================
    // v4: Spatial Brain Methods
    // =========================================================================

    /// Initialize spatial positions from grid dimensions.
    ///
    /// Must be called after pool creation to enable spatial operations.
    pub fn init_spatial(&mut self, bounds: [f32; 3]) {
        self.neurons.init_spatial_from_grid(bounds, (self.dims.w, self.dims.h, self.dims.d));
        self.spatial_bounds = Some(bounds);
    }

    /// Initialize spatial positions with random jitter.
    pub fn init_spatial_random(&mut self, bounds: [f32; 3], seed: u64) {
        self.neurons.init_spatial_random(bounds, seed);
    }

    /// Find neurons within radius of position.
    pub fn neurons_in_radius(&self, pos: [f32; 3], radius: f32) -> Vec<usize> {
        self.neurons.neurons_in_radius(pos, radius)
    }

    /// Compute density at position.
    pub fn density_at(&self, pos: [f32; 3], sigma: f32) -> f32 {
        self.neurons.density_at(pos, sigma)
    }

    /// Inject current to neurons within radius of position.
    ///
    /// Current is distributed based on distance (closer = stronger).
    pub fn inject_spatial(&mut self, pos: [f32; 3], radius: f32, value: i16) {
        let nearby = self.neurons.neurons_in_radius(pos, radius);
        if nearby.is_empty() {
            return;
        }

        let r2 = radius * radius;
        for &idx in &nearby {
            let dx = self.neurons.soma_position[idx][0] - pos[0];
            let dy = self.neurons.soma_position[idx][1] - pos[1];
            let dz = self.neurons.soma_position[idx][2] - pos[2];
            let d2 = dx * dx + dy * dy + dz * dz;

            // Distance-weighted injection: full at center, zero at radius
            let weight = 1.0 - (d2 / r2).sqrt();
            let current = (value as f32 * weight) as i16;

            self.neurons.membrane[idx] = self.neurons.membrane[idx].saturating_add(current);
        }

        // Clear spike window for fresh read
        for &idx in &nearby {
            self.spike_window[idx] = false;
            self.spike_window_count[idx] = 0;
        }
    }

    /// Read activity from neurons within radius of position.
    ///
    /// Returns weighted sum of spike_window_count (rate-coded output).
    pub fn read_spatial(&self, pos: [f32; 3], radius: f32) -> Vec<i32> {
        let nearby = self.neurons.neurons_in_radius(pos, radius);
        if nearby.is_empty() {
            return vec![0];
        }

        let r2 = radius * radius;
        let mut result = Vec::with_capacity(nearby.len());

        for &idx in &nearby {
            let dx = self.neurons.soma_position[idx][0] - pos[0];
            let dy = self.neurons.soma_position[idx][1] - pos[1];
            let dz = self.neurons.soma_position[idx][2] - pos[2];
            let d2 = dx * dx + dy * dy + dz * dz;

            // Distance-weighted output
            let weight = 1.0 - (d2 / r2).sqrt();
            let activity = (self.spike_window_count[idx] as f32 * weight * 64.0) as i32;
            result.push(activity);
        }

        result
    }

    /// Get soma position of a neuron.
    pub fn soma_position(&self, idx: usize) -> Option<[f32; 3]> {
        if idx < self.neurons.len() {
            Some(self.neurons.soma_position[idx])
        } else {
            None
        }
    }

    /// Get axon terminal position of a neuron.
    pub fn axon_terminal(&self, idx: usize) -> Option<[f32; 3]> {
        if idx < self.neurons.len() {
            Some(self.neurons.axon_terminal[idx])
        } else {
            None
        }
    }

    /// Get all soma positions (for density field updates).
    pub fn all_soma_positions(&self) -> &[[f32; 3]] {
        &self.neurons.soma_position
    }

    /// Decay axon health for all neurons.
    ///
    /// Call periodically to apply survival pressure to unused connections.
    pub fn decay_axon_health(&mut self, decay_rate: u8) {
        for health in &mut self.neurons.axon_health {
            *health = health.saturating_sub(decay_rate);
        }
    }

    /// Boost axon health for neurons that spiked (activity = survival).
    pub fn boost_active_axon_health(&mut self, boost: u8) {
        for idx in 0..self.neurons.len() {
            if self.spike_window[idx] {
                self.neurons.axon_health[idx] = self.neurons.axon_health[idx].saturating_add(boost);
            }
        }
    }

    /// Prune neurons with dead axons (health = 0).
    ///
    /// Returns number of neurons pruned.
    pub fn prune_dead_axons(&mut self) -> usize {
        let mut dead_indices: Vec<usize> = self.neurons.axon_health.iter()
            .enumerate()
            .filter(|(_, &h)| h == 0)
            .map(|(i, _)| i)
            .collect();

        // Sort descending for safe swap-remove
        dead_indices.sort_by(|a, b| b.cmp(a));

        let count = self.neurons.remove_descending(&dead_indices);

        // Update pool counts
        self.n_neurons = self.neurons.len() as u32;
        // Note: excitatory/inhibitory counts need recomputation if precise tracking needed

        count
    }

    // =========================================================================
    // v4: Template System (Computational Circuit Motifs)
    // =========================================================================

    /// Request a template instantiation.
    ///
    /// The template is spawned at the hint position (or auto-placed in low-density
    /// area if None). Returns the template instance ID, or None if spawn failed.
    pub fn spawn_template(&mut self, request: TemplateRequest, seed: u64) -> Option<u32> {
        let bounds = self.spatial_bounds?;

        // Find spawn position
        let centroid = match request.position_hint {
            Some(pos) => pos,
            None => self.find_low_density_position(bounds, seed)?,
        };

        // Get neuron type distribution for this template (for future use)
        let (_n_comp, _n_gate, _n_osc, _n_mem, _n_sens, _n_mot) = request.template_type.type_distribution();
        let total_neurons = request.template_type.neuron_count();

        if total_neurons == 0 {
            return None;
        }

        // Record starting index for new neurons
        let start_idx = self.neurons.len();

        // Spawn neurons based on template type
        let mut neuron_indices = Vec::with_capacity(total_neurons);
        let mut rng = seed;

        // Helper for LCG random
        let mut next_rand = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 32) as u32) as f32 / u32::MAX as f32
        };

        // Spawn arrangement depends on template type
        match request.template_type {
            TemplateType::LateralInhibition { scale, surround_ratio } => {
                // Center neurons in a cluster
                for i in 0..scale {
                    let offset = [
                        (next_rand() - 0.5) * 0.5,
                        (next_rand() - 0.5) * 0.5,
                        (next_rand() - 0.5) * 0.5,
                    ];
                    let pos = [
                        centroid[0] + offset[0],
                        centroid[1] + offset[1],
                        centroid[2] + offset[2],
                    ];
                    self.spawn_neuron_at(pos, NeuronType::Computational);
                    neuron_indices.push(start_idx + i as usize);
                }

                // Gate neurons surrounding each center
                for i in 0..scale {
                    let center_pos = self.neurons.soma_position[start_idx + i as usize];
                    for j in 0..surround_ratio {
                        let angle = (j as f32 / surround_ratio as f32) * std::f32::consts::TAU;
                        let radius = 0.3;
                        let pos = [
                            center_pos[0] + angle.cos() * radius,
                            center_pos[1] + angle.sin() * radius,
                            center_pos[2],
                        ];
                        self.spawn_neuron_at(pos, NeuronType::Gate);
                        neuron_indices.push(self.neurons.len() - 1);
                    }
                }
            }

            TemplateType::AttractorMemory { capacity } => {
                // Memory neurons in a compact cluster
                for i in 0..capacity {
                    let offset = [
                        (next_rand() - 0.5) * 0.4,
                        (next_rand() - 0.5) * 0.4,
                        (next_rand() - 0.5) * 0.4,
                    ];
                    let pos = [
                        centroid[0] + offset[0],
                        centroid[1] + offset[1],
                        centroid[2] + offset[2],
                    ];
                    self.spawn_neuron_at(pos, NeuronType::MemoryReader);
                    neuron_indices.push(start_idx + i as usize);
                }
            }

            TemplateType::TemporalChain { length } => {
                // Chain neurons in a line
                let step = 0.2;
                let direction = [1.0f32, 0.0, 0.0]; // Default: extend in X
                for i in 0..length {
                    let pos = [
                        centroid[0] + direction[0] * step * i as f32,
                        centroid[1] + direction[1] * step * i as f32,
                        centroid[2] + direction[2] * step * i as f32,
                    ];
                    self.spawn_neuron_at(pos, NeuronType::Computational);
                    neuron_indices.push(start_idx + i as usize);
                }
                // Gate neuron at end for control
                let gate_pos = [
                    centroid[0] + direction[0] * step * length as f32,
                    centroid[1] + direction[1] * step * length as f32,
                    centroid[2] + direction[2] * step * length as f32,
                ];
                self.spawn_neuron_at(gate_pos, NeuronType::Gate);
                neuron_indices.push(self.neurons.len() - 1);
            }

            TemplateType::OscillatorNetwork { follower_count, .. } => {
                // Pacemaker at center
                self.spawn_neuron_at(centroid, NeuronType::Oscillator);
                neuron_indices.push(start_idx);

                // Followers in a sphere around pacemaker
                for i in 0..follower_count {
                    let theta = next_rand() * std::f32::consts::TAU;
                    let phi = (next_rand() * 2.0 - 1.0).acos();
                    let r = 0.3 + next_rand() * 0.2;
                    let pos = [
                        centroid[0] + r * phi.sin() * theta.cos(),
                        centroid[1] + r * phi.sin() * theta.sin(),
                        centroid[2] + r * phi.cos(),
                    ];
                    self.spawn_neuron_at(pos, NeuronType::Computational);
                    neuron_indices.push(start_idx + 1 + i as usize);
                }
            }

            TemplateType::DisinhibitionGate => {
                // Two gate neurons close together
                self.spawn_neuron_at(centroid, NeuronType::Gate);
                neuron_indices.push(start_idx);
                let pos2 = [centroid[0] + 0.1, centroid[1], centroid[2]];
                self.spawn_neuron_at(pos2, NeuronType::Gate);
                neuron_indices.push(start_idx + 1);
            }

            TemplateType::WinnerTakeAll { competitors } => {
                // Competitors in a ring, each with inhibitory partner
                for i in 0..competitors {
                    let angle = (i as f32 / competitors as f32) * std::f32::consts::TAU;
                    let r = 0.3;
                    let pos = [
                        centroid[0] + r * angle.cos(),
                        centroid[1] + r * angle.sin(),
                        centroid[2],
                    ];
                    self.spawn_neuron_at(pos, NeuronType::Computational);
                    neuron_indices.push(start_idx + (i * 2) as usize);

                    // Inhibitory partner slightly inside
                    let inh_pos = [
                        centroid[0] + (r * 0.7) * angle.cos(),
                        centroid[1] + (r * 0.7) * angle.sin(),
                        centroid[2],
                    ];
                    self.spawn_neuron_at(inh_pos, NeuronType::Gate);
                    neuron_indices.push(start_idx + (i * 2 + 1) as usize);
                }
            }

            TemplateType::SensoryArray { dimensions } => {
                // Sensory neurons in a line - like a cochlea or retina strip
                // Each position corresponds to one input dimension
                // Neuron at index i responds to input value i
                let step = 0.05; // Spacing between neurons
                let neurons_per_row = 16u16;
                for i in 0..dimensions {
                    let row = i / neurons_per_row;
                    let col = i % neurons_per_row;
                    let pos = [
                        centroid[0] + (col as f32 - neurons_per_row as f32 / 2.0) * step,
                        centroid[1] + (row as f32) * step,
                        centroid[2],
                    ];
                    self.spawn_neuron_at(pos, NeuronType::Sensory);
                    neuron_indices.push(start_idx + i as usize);
                }
            }
        }

        // Create and register the template instance
        let instance = TemplateInstance::new(
            0, // ID assigned by registry
            request.template_type,
            neuron_indices,
            centroid,
            request.input_signal,
            request.output_signal,
        );

        let id = self.templates.register(instance);
        Some(id)
    }

    /// Spawn a single neuron at the given position with the given type.
    fn spawn_neuron_at(&mut self, pos: [f32; 3], ntype: NeuronType) {
        // Extend neurons array
        let is_excitatory = ntype != NeuronType::Gate; // Gates are inhibitory
        let n_exc = if is_excitatory { 1 } else { 0 };
        self.neurons.extend(1, n_exc, self.config.resting_potential, self.config.spike_threshold);

        let idx = self.neurons.len() - 1;

        // Set position
        self.neurons.soma_position[idx] = pos;
        self.neurons.axon_terminal[idx] = pos; // Starts at soma

        // Set neuron type in flags
        let mut f = self.neurons.flags[idx];
        f &= 0b00000111; // Clear type bits
        f |= (ntype as u8) << 3;
        self.neurons.flags[idx] = f;

        // Extend auxiliary arrays
        self.synaptic_current.push(0);
        self.projection_current.push(0);
        self.spike_rate.push(0);
        self.spike_window.push(false);
        self.spike_window_count.push(0);
        self.spike_counts.push(0);
        self.chem_exposure.push(0);
        self.delay_buf.extend(1);

        // Extend synapse store to cover new neuron
        self.synapses.extend(1);

        // Update counts
        self.n_neurons = self.neurons.len() as u32;
        if is_excitatory {
            self.n_excitatory += 1;
        } else {
            self.n_inhibitory += 1;
        }
    }

    /// Find a low-density position for template placement.
    fn find_low_density_position(&self, bounds: [f32; 3], seed: u64) -> Option<[f32; 3]> {
        // Sample random positions and pick the one with lowest density
        let mut rng = seed;
        let mut best_pos = [bounds[0] / 2.0, bounds[1] / 2.0, bounds[2] / 2.0];
        let mut best_density = f32::MAX;

        for _ in 0..16 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rx = ((rng >> 32) as u32) as f32 / u32::MAX as f32;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let ry = ((rng >> 32) as u32) as f32 / u32::MAX as f32;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rz = ((rng >> 32) as u32) as f32 / u32::MAX as f32;

            let pos = [bounds[0] * rx, bounds[1] * ry, bounds[2] * rz];
            let density = self.neurons.density_at(pos, 0.5);

            if density < best_density {
                best_density = density;
                best_pos = pos;
            }
        }

        Some(best_pos)
    }

    /// Get a template by ID.
    pub fn get_template(&self, id: u32) -> Option<&TemplateInstance> {
        self.templates.get(id)
    }

    /// Get a mutable template by ID.
    pub fn get_template_mut(&mut self, id: u32) -> Option<&mut TemplateInstance> {
        self.templates.get_mut(id)
    }

    /// List all active template IDs.
    pub fn list_template_ids(&self) -> Vec<u32> {
        self.templates.all().iter().map(|t| t.id).collect()
    }

    /// Update template fitness based on activity correlation.
    ///
    /// Call periodically to measure how well templates are performing.
    pub fn update_template_fitness(&mut self, alpha: f32) {
        for template in self.templates.all_mut() {
            // Measure correlation: did template neurons fire when receiving input?
            let mut activity = 0u64;
            for &idx in &template.neuron_indices {
                if idx < self.spike_window.len() && self.spike_window[idx] {
                    activity += 1;
                }
            }

            // Fitness based on activity ratio
            let activity_ratio = if template.neuron_indices.is_empty() {
                0.0
            } else {
                activity as f32 / template.neuron_indices.len() as f32
            };

            // Update with EMA
            template.update_fitness(activity_ratio, alpha);
            template.cumulative_activity += activity;
            template.tick();
        }
    }

    /// Prune templates below fitness threshold.
    ///
    /// Returns IDs of pruned templates. Note: Does NOT remove the neurons;
    /// they remain in the pool but are no longer associated with a template.
    pub fn prune_unfit_templates(&mut self, threshold: f32, grace_period: u64) -> Vec<u32> {
        self.templates.prune_unfit(threshold, grace_period)
    }

    /// Find templates that need a specific signal type.
    pub fn templates_needing(&self, signal: SignalType) -> Vec<u32> {
        self.templates.templates_needing(signal)
    }

    /// Find templates that offer a specific signal type.
    pub fn templates_offering(&self, signal: SignalType) -> Vec<u32> {
        self.templates.templates_offering(signal)
    }

    /// Get template centroid position.
    pub fn template_centroid(&self, id: u32) -> Option<[f32; 3]> {
        self.templates.get(id).map(|t| t.centroid)
    }

    /// Get template fitness score.
    pub fn template_fitness(&self, id: u32) -> Option<f32> {
        self.templates.get(id).map(|t| t.fitness)
    }

    /// Number of active templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Inject sensory input through a SensoryArray template.
    ///
    /// This is how sensory organs work:
    /// - Cochlea: frequency → hair cell position → neural activation
    /// - Retina: light position → ganglion cell → neural activation
    /// - SensoryArray: byte value → sensory neuron at that index
    ///
    /// For input bytes [b0, b1, b2, ...]:
    /// - Sensory neuron at index b0 is activated
    /// - Sensory neuron at index b1 is activated
    /// - etc.
    ///
    /// The SAME sensory organ (template), different input "shapes" pass through.
    /// "cat" [99, 97, 116] activates neurons 99, 97, 116.
    /// "dog" [100, 111, 103] activates neurons 100, 111, 103.
    ///
    /// Returns number of neurons stimulated.
    pub fn inject_sensory(
        &mut self,
        template_id: u32,
        input_bytes: &[u8],
        current: i16,
    ) -> usize {
        let template = match self.templates.get(template_id) {
            Some(t) => t,
            None => return 0,
        };

        // Template must have enough neurons for the byte range
        let neurons = template.neuron_indices.clone();
        let n = neurons.len();
        if n == 0 || input_bytes.is_empty() {
            return 0;
        }

        let mut stimulated = 0;

        // Each byte value maps directly to a sensory neuron index
        // This is the fixed topology of the sensory organ
        for &byte in input_bytes {
            if byte == 0 {
                continue; // Skip null bytes
            }

            let sensory_idx = byte as usize;
            if sensory_idx < n {
                // Activate the sensory neuron at this index
                let neuron_idx = neurons[sensory_idx];
                self.neurons.membrane[neuron_idx] =
                    self.neurons.membrane[neuron_idx].saturating_add(current);
                stimulated += 1;
            }
        }

        // Clear spike window for fresh measurement
        for &idx in &neurons {
            self.spike_window[idx] = false;
            self.spike_window_count[idx] = 0;
        }

        stimulated
    }

    /// Wire inter-template connections based on need/offer matching and spatial proximity.
    ///
    /// This is the "axon growth" step: neurons from offering templates form synapses
    /// to neurons in needing templates. Connection probability is based purely on:
    /// - Spatial distance (closer = more likely)
    /// - Need/offer signal matching
    ///
    /// No template knows about other templates. Only pressure gradients (need/offer)
    /// and physical proximity drive wiring.
    ///
    /// Returns number of synapses created.
    pub fn wire_inter_template(&mut self, max_distance: f32, base_probability: f32, seed: u64) -> usize {
        // Collect all need/offer pairs
        let mut connections_to_make: Vec<(usize, usize, f32)> = Vec::new();
        let mut rng = seed;

        // Get all template data upfront to avoid borrow issues
        let template_data: Vec<(u32, Vec<usize>, [f32; 3], SignalType, SignalType)> = self.templates
            .all()
            .iter()
            .map(|t| (t.id, t.neuron_indices.clone(), t.centroid, t.input_signal, t.output_signal))
            .collect();

        // For each pair of templates, check if one offers what the other needs
        for (offer_id, offer_neurons, offer_centroid, _, offer_signal) in &template_data {
            for (need_id, need_neurons, need_centroid, need_signal, _) in &template_data {
                // Skip self-connections (within same template)
                if offer_id == need_id {
                    continue;
                }

                // Check if offer matches need
                if offer_signal != need_signal {
                    continue;
                }

                // Distance between template centroids
                let dx = offer_centroid[0] - need_centroid[0];
                let dy = offer_centroid[1] - need_centroid[1];
                let dz = offer_centroid[2] - need_centroid[2];
                let centroid_dist = (dx * dx + dy * dy + dz * dz).sqrt();

                // Skip if too far
                if centroid_dist > max_distance * 2.0 {
                    continue;
                }

                // Create connections from offering neurons to needing neurons
                // Probability decreases with distance
                for &src_idx in offer_neurons {
                    for &tgt_idx in need_neurons {
                        // Neuron-to-neuron distance
                        let src_pos = self.neurons.soma_position[src_idx];
                        let tgt_pos = self.neurons.soma_position[tgt_idx];
                        let ndx = src_pos[0] - tgt_pos[0];
                        let ndy = src_pos[1] - tgt_pos[1];
                        let ndz = src_pos[2] - tgt_pos[2];
                        let neuron_dist = (ndx * ndx + ndy * ndy + ndz * ndz).sqrt();

                        if neuron_dist > max_distance {
                            continue;
                        }

                        // Probability: higher when closer
                        let dist_factor = 1.0 - (neuron_dist / max_distance);
                        let prob = base_probability * dist_factor * dist_factor; // Quadratic falloff

                        // Random check
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                        if roll < prob {
                            connections_to_make.push((src_idx, tgt_idx, dist_factor));
                        }
                    }
                }
            }
        }

        // Now create the synapses
        let mut created = 0;
        for (src, tgt, strength) in connections_to_make {
            // Weight based on distance factor (closer = stronger)
            let weight = (strength * 80.0) as i8; // Max weight ~80

            let syn = crate::synapse::Synapse {
                target: tgt as u16,
                weight,
                delay: 1,
                eligibility: 0,
                maturity: crate::synapse::maturity::encode(crate::synapse::ThermalState::Hot, 0),
                _reserved: [0, 0],
            };

            self.synapses.add_synapse(src as u32, syn);
            created += 1;
        }

        created
    }

    /// Wire intra-template recurrent connections for a specific template.
    ///
    /// Creates lateral inhibition dynamics within a template:
    /// - Nearby neurons have weak excitatory connections (co-activation)
    /// - Distant neurons have inhibitory connections (competition)
    ///
    /// This creates attractor dynamics where small input differences get
    /// amplified into distinct stable states. Active neurons inhibit others,
    /// causing winner-take-all dynamics within subpopulations.
    ///
    /// Returns number of synapses created.
    pub fn wire_intra_template(
        &mut self,
        template_id: u32,
        connectivity: f32, // 0.0-1.0, fraction of possible connections
        seed: u64,
    ) -> usize {
        let template = match self.templates.get(template_id) {
            Some(t) => t,
            None => return 0,
        };

        let neurons = template.neuron_indices.clone();
        let n = neurons.len();
        if n < 2 {
            return 0;
        }

        let mut rng = seed;
        let mut created = 0;

        // Create recurrent connections within the template
        // Close neurons: weak excitation (co-activate)
        // Distant neurons: inhibition (compete)
        for (i, &src_idx) in neurons.iter().enumerate() {
            for (j, &tgt_idx) in neurons.iter().enumerate() {
                if i == j {
                    continue; // No self-loops
                }

                // Probabilistic connection based on connectivity parameter
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                if roll < connectivity {
                    // Distance between neurons
                    let src_pos = self.neurons.soma_position[src_idx];
                    let tgt_pos = self.neurons.soma_position[tgt_idx];
                    let dx = src_pos[0] - tgt_pos[0];
                    let dy = src_pos[1] - tgt_pos[1];
                    let dz = src_pos[2] - tgt_pos[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    // Lateral inhibition profile (Mexican hat):
                    // Very close neurons: weak excitation (co-activation)
                    // All other neurons: strong inhibition (competition)
                    // This creates winner-take-all dynamics
                    let excite_radius = 0.08; // Very tight excitation
                    let max_dist = 0.5; // Cluster diameter

                    let weight = if dist < excite_radius {
                        // Very close: weak excitation
                        let proximity = 1.0 - (dist / excite_radius);
                        (proximity * 12.0) as i8 // Max +12
                    } else {
                        // All others: strong inhibition
                        let remoteness = ((dist - excite_radius) / (max_dist - excite_radius)).min(1.0);
                        let inh = 20 + (remoteness * 40.0) as i8; // -20 to -60
                        -(inh.min(60))
                    };

                    let syn = crate::synapse::Synapse {
                        target: tgt_idx as u16,
                        weight,
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Hot,
                            0,
                        ),
                        _reserved: [0, 0],
                    };

                    self.synapses.add_synapse(src_idx as u32, syn);
                    created += 1;
                }
            }
        }

        created
    }

    /// Wire lateral inhibition using inhibitory neurons as the inhibitory pool.
    ///
    /// Creates feedback inhibition where:
    /// 1. All excitatory neurons project TO inhibitory neurons (they integrate activity)
    /// 2. All inhibitory neurons project TO all excitatory neurons (global suppression)
    ///
    /// This creates winner-take-all dynamics:
    /// - High activity → strong inhibition → only strongest survive
    /// - Enforces sparse coding (2-4% active neurons)
    /// - Pattern separation: similar inputs activate different "winners"
    ///
    /// # Arguments
    /// - `excite_to_inhib_prob`: Connection probability from excitatory → inhibitory (0.5-1.0)
    /// - `inhib_to_excite_prob`: Connection probability from inhibitory → excitatory (0.5-1.0)
    /// - `inhib_weight`: Inhibitory weight magnitude (negative, typically -40 to -80)
    /// - `seed`: RNG seed for reproducibility
    ///
    /// # Returns
    /// Number of synapses created.
    pub fn wire_lateral_inhibition(
        &mut self,
        excite_to_inhib_prob: f32,
        inhib_to_excite_prob: f32,
        inhib_weight: i8,
        seed: u64,
    ) -> usize {
        let n = self.n_neurons as usize;
        if n < 2 {
            return 0;
        }

        // Collect excitatory and inhibitory neuron indices
        let mut excitatory: Vec<usize> = Vec::new();
        let mut inhibitory: Vec<usize> = Vec::new();

        for i in 0..n {
            if crate::neuron::flags::is_inhibitory(self.neurons.flags[i]) {
                inhibitory.push(i);
            } else {
                excitatory.push(i);
            }
        }

        if inhibitory.is_empty() || excitatory.is_empty() {
            return 0;
        }

        let mut rng = seed;
        let mut created = 0;

        // Phase 1: Excitatory → Inhibitory (integration)
        // Each inhibitory neuron samples the population activity
        for &exc_idx in &excitatory {
            for &inh_idx in &inhibitory {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                if roll < excite_to_inhib_prob {
                    // Excitatory synapse to inhibitory neuron
                    // Weight: moderate positive (inhibitory neuron integrates activity)
                    let syn = crate::synapse::Synapse {
                        target: inh_idx as u16,
                        weight: 20, // Moderate excitation to integrator
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(exc_idx as u32, syn);
                    created += 1;
                }
            }
        }

        // Phase 2: Inhibitory → Excitatory (suppression)
        // Each inhibitory neuron suppresses all excitatory neurons
        for &inh_idx in &inhibitory {
            for &exc_idx in &excitatory {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                if roll < inhib_to_excite_prob {
                    // Inhibitory synapse (negative weight)
                    let syn = crate::synapse::Synapse {
                        target: exc_idx as u16,
                        weight: inhib_weight, // Strong negative weight
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(inh_idx as u32, syn);
                    created += 1;
                }
            }
        }

        created
    }

    /// Wire lateral inhibition within a specific template.
    ///
    /// Like `wire_lateral_inhibition` but scoped to neurons in a single template.
    /// Useful for creating local competition within a computational motif.
    ///
    /// Uses Gate neurons within the template as the inhibitory pool.
    /// If no Gate neurons exist, uses inhibitory neurons instead.
    ///
    /// # Returns
    /// Number of synapses created.
    pub fn wire_template_lateral_inhibition(
        &mut self,
        template_id: u32,
        inhib_weight: i8,
        seed: u64,
    ) -> usize {
        let template = match self.templates.get(template_id) {
            Some(t) => t,
            None => return 0,
        };

        let neurons = template.neuron_indices.clone();
        if neurons.len() < 2 {
            return 0;
        }

        // Find Gate neurons or inhibitory neurons within template
        let mut excitatory: Vec<usize> = Vec::new();
        let mut inhibitory: Vec<usize> = Vec::new();

        for &idx in &neurons {
            let flags = self.neurons.flags[idx];
            let ntype = crate::neuron::NeuronType::from_flags(flags);

            // Gate neurons are our primary inhibitory pool
            if ntype == crate::neuron::NeuronType::Gate {
                inhibitory.push(idx);
            } else if crate::neuron::flags::is_inhibitory(flags) {
                inhibitory.push(idx);
            } else {
                excitatory.push(idx);
            }
        }

        if inhibitory.is_empty() || excitatory.is_empty() {
            return 0;
        }

        let mut rng = seed;
        let mut created = 0;

        // All excitatory → all inhibitory (high probability, they integrate)
        for &exc_idx in &excitatory {
            for &inh_idx in &inhibitory {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                if roll < 0.8 {
                    let syn = crate::synapse::Synapse {
                        target: inh_idx as u16,
                        weight: 15,
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(exc_idx as u32, syn);
                    created += 1;
                }
            }
        }

        // All inhibitory → all excitatory (strong suppression)
        for &inh_idx in &inhibitory {
            for &exc_idx in &excitatory {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                if roll < 0.9 {
                    let syn = crate::synapse::Synapse {
                        target: exc_idx as u16,
                        weight: inhib_weight,
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(inh_idx as u32, syn);
                    created += 1;
                }
            }
        }

        created
    }

    /// Wire LOCAL lateral inhibition based on spatial proximity.
    ///
    /// Unlike `wire_lateral_inhibition` which creates global competition,
    /// this version only connects neurons within a spatial neighborhood.
    /// This prevents spurious overlap between spatially distant input patterns.
    ///
    /// # Parameters
    /// - `radius`: Spatial radius for inhibition neighborhood
    /// - `inhib_weight`: Negative weight for suppression synapses
    /// - `seed`: RNG seed
    ///
    /// # Returns
    /// Number of synapses created.
    pub fn wire_local_lateral_inhibition(
        &mut self,
        radius: f32,
        inhib_weight: i8,
        seed: u64,
    ) -> usize {
        let n = self.n_neurons as usize;
        if n < 2 {
            return 0;
        }

        // Collect positions and classify neurons
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut excitatory: Vec<usize> = Vec::new();
        let mut inhibitory: Vec<usize> = Vec::new();

        for i in 0..n {
            positions.push(self.neurons.soma_position[i]);
            if crate::neuron::flags::is_inhibitory(self.neurons.flags[i]) {
                inhibitory.push(i);
            } else {
                excitatory.push(i);
            }
        }

        if inhibitory.is_empty() || excitatory.is_empty() {
            return 0;
        }

        let mut rng = seed;
        let mut created = 0;
        let radius_sq = radius * radius;

        // For each inhibitory neuron, find nearby excitatory neurons
        for &inh_idx in &inhibitory {
            let inh_pos = positions[inh_idx];

            // Find excitatory neurons within radius
            let nearby_exc: Vec<usize> = excitatory
                .iter()
                .filter(|&&exc_idx| {
                    let exc_pos = positions[exc_idx];
                    let dx = exc_pos[0] - inh_pos[0];
                    let dy = exc_pos[1] - inh_pos[1];
                    let dz = exc_pos[2] - inh_pos[2];
                    dx * dx + dy * dy + dz * dz <= radius_sq
                })
                .copied()
                .collect();

            // Wire excitatory → this inhibitory (integration)
            for &exc_idx in &nearby_exc {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                if roll < 0.6 {
                    let syn = crate::synapse::Synapse {
                        target: inh_idx as u16,
                        weight: 15, // Moderate excitation
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(exc_idx as u32, syn);
                    created += 1;
                }
            }

            // Wire this inhibitory → nearby excitatory (suppression)
            for &exc_idx in &nearby_exc {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                if roll < 0.8 {
                    let syn = crate::synapse::Synapse {
                        target: exc_idx as u16,
                        weight: inhib_weight,
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(inh_idx as u32, syn);
                    created += 1;
                }
            }
        }

        created
    }

    /// Create explicit conductive chains between neuron groups.
    ///
    /// Establishes strong directional connections from source neurons to target neurons,
    /// with signal intensity determined by the weight. Chains are pruned based on
    /// activity correlation - unused chains decay via axon_health.
    ///
    /// # Parameters
    /// - `sources`: Indices of source neurons
    /// - `targets`: Indices of target neurons
    /// - `weight`: Synapse weight (positive for excitation)
    /// - `delay`: Synaptic delay in ticks
    /// - `seed`: RNG seed
    ///
    /// # Returns
    /// Number of synapses created.
    pub fn wire_conductive_chain(
        &mut self,
        sources: &[usize],
        targets: &[usize],
        weight: i8,
        delay: u8,
        seed: u64,
    ) -> usize {
        let mut rng = seed;
        let mut created = 0;

        for &src in sources {
            for &tgt in targets {
                if src >= self.n_neurons as usize || tgt >= self.n_neurons as usize {
                    continue;
                }

                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let roll = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

                // High probability for chain connections
                if roll < 0.7 {
                    let syn = crate::synapse::Synapse {
                        target: tgt as u16,
                        weight,
                        delay,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(src as u32, syn);
                    created += 1;
                }
            }
        }

        created
    }

    /// Wire oscillatory synchronization between neurons.
    ///
    /// Creates bidirectional connections between oscillator neurons to enable
    /// phase locking. Non-oscillator neurons within the group receive one-way
    /// connections from oscillators for rhythm entrainment.
    ///
    /// # Parameters
    /// - `neuron_indices`: Indices of neurons to synchronize
    /// - `coupling_weight`: Weight for coupling synapses (typically low, 5-15)
    /// - `seed`: RNG seed
    ///
    /// # Returns
    /// Number of synapses created.
    pub fn wire_oscillatory_sync(
        &mut self,
        neuron_indices: &[usize],
        coupling_weight: i8,
        seed: u64,
    ) -> usize {
        if neuron_indices.len() < 2 {
            return 0;
        }

        // Find oscillators and followers
        let mut oscillators: Vec<usize> = Vec::new();
        let mut followers: Vec<usize> = Vec::new();

        for &idx in neuron_indices {
            if idx >= self.n_neurons as usize {
                continue;
            }
            let flags = self.neurons.flags[idx];
            let ntype = crate::neuron::NeuronType::from_flags(flags);

            if ntype == crate::neuron::NeuronType::Oscillator {
                oscillators.push(idx);
            } else {
                followers.push(idx);
            }
        }

        let mut rng = seed;
        let mut created = 0;

        // Oscillator ↔ Oscillator (bidirectional coupling)
        for i in 0..oscillators.len() {
            for j in (i + 1)..oscillators.len() {
                let src = oscillators[i];
                let tgt = oscillators[j];

                // Forward
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 33) as f32 / (u32::MAX >> 1) as f32 > 0.3 {
                    let syn = crate::synapse::Synapse {
                        target: tgt as u16,
                        weight: coupling_weight,
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(src as u32, syn);
                    created += 1;
                }

                // Backward
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 33) as f32 / (u32::MAX >> 1) as f32 > 0.3 {
                    let syn = crate::synapse::Synapse {
                        target: src as u16,
                        weight: coupling_weight,
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(tgt as u32, syn);
                    created += 1;
                }
            }
        }

        // Oscillator → Follower (entrainment)
        for &osc_idx in &oscillators {
            for &fol_idx in &followers {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 33) as f32 / (u32::MAX >> 1) as f32 > 0.4 {
                    let syn = crate::synapse::Synapse {
                        target: fol_idx as u16,
                        weight: coupling_weight,
                        delay: 1,
                        eligibility: 0,
                        maturity: crate::synapse::maturity::encode(
                            crate::synapse::ThermalState::Warm,
                            0,
                        ),
                        _reserved: [0, 0],
                    };
                    self.synapses.add_synapse(osc_idx as u32, syn);
                    created += 1;
                }
            }
        }

        created
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

    // ====================================================================
    // Activity tracking tests (A1)
    // ====================================================================

    #[test]
    fn activity_tracking_counts_spikes() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());

        // Initially all zeros
        assert_eq!(pool.neuron_activities().len(), 10);
        assert!(pool.neuron_activities().iter().all(|&c| c == 0));
        assert_eq!(pool.active_neuron_count(), 0);

        // Force neuron 0 to spike with massive input
        let mut input = vec![0i16; 10];
        input[0] = 10000;
        pool.tick_simple(&input);

        assert!(pool.neurons.spike_out[0], "neuron 0 should have spiked");
        assert_eq!(pool.neuron_activities()[0], 1, "spike_count for neuron 0 should be 1");
        assert!(pool.active_neuron_count() >= 1);

        // Tick again — neuron 0 is in refractory, so no spike
        pool.tick_simple(&input);
        assert!(!pool.neurons.spike_out[0], "neuron 0 should be refractory");
        assert_eq!(pool.neuron_activities()[0], 1, "spike_count should stay 1 (no new spike)");

        // Wait out refractory (2 ticks), then spike again
        pool.tick_simple(&input); // tick 3 — still refractory
        pool.tick_simple(&input); // tick 4 — should spike again
        assert!(pool.neurons.spike_out[0], "neuron 0 should spike after refractory");
        assert_eq!(pool.neuron_activities()[0], 2, "spike_count should now be 2");

        // Reset activities
        pool.reset_activities();
        assert!(pool.neuron_activities().iter().all(|&c| c == 0));
        assert_eq!(pool.active_neuron_count(), 0);
    }

    #[test]
    fn growth_ratio_tracks_initial_size() {
        let pool = NeuronPool::new("test", 100, PoolConfig::default());
        assert_eq!(pool.growth_ratio(), 1.0, "initial growth ratio should be 1.0");
    }

    // ====================================================================
    // Growth tests (A2)
    // ====================================================================

    #[test]
    fn grow_neurons_extends_all_arrays() {
        let mut pool = NeuronPool::new("test", 32, PoolConfig::default());
        assert_eq!(pool.n_neurons, 32);
        assert_eq!(pool.initial_neuron_count, 32);

        let added = pool.grow_neurons_seeded(8, 42);
        assert_eq!(added, 8);
        assert_eq!(pool.n_neurons, 40);
        assert_eq!(pool.initial_neuron_count, 32, "initial should not change");

        // All SoA arrays should have correct length
        assert_eq!(pool.neurons.membrane.len(), 40);
        assert_eq!(pool.neurons.threshold.len(), 40);
        assert_eq!(pool.neurons.leak.len(), 40);
        assert_eq!(pool.neurons.flags.len(), 40);
        assert_eq!(pool.neurons.trace.len(), 40);
        assert_eq!(pool.neurons.spike_out.len(), 40);
        assert_eq!(pool.neurons.binding_slot.len(), 40);
        assert_eq!(pool.synaptic_current.len(), 40);
        assert_eq!(pool.projection_current.len(), 40);
        assert_eq!(pool.spike_rate.len(), 40);
        assert_eq!(pool.spike_window.len(), 40);
        assert_eq!(pool.spike_window_count.len(), 40);
        assert_eq!(pool.spike_counts.len(), 40);

        // New neurons should be at resting potential
        for i in 32..40 {
            assert_eq!(pool.neurons.membrane[i], pool.config.resting_potential);
            assert_eq!(pool.neurons.threshold[i], pool.config.spike_threshold);
        }

        // Dale's Law: 80% excitatory of new neurons
        let new_exc = (32..40).filter(|&i| crate::neuron::flags::is_excitatory(pool.neurons.flags[i])).count();
        assert_eq!(new_exc, 6, "80% of 8 = 6 excitatory"); // (8*4)/5 = 6

        // CSR should be extended — new neurons have 0 outgoing synapses
        assert_eq!(pool.synapses.row_ptr.len(), 41); // n_neurons + 1
        for i in 33..=40 {
            assert_eq!(pool.synapses.outgoing((i - 1) as u32).len(), 0, "new neuron should have no synapses");
        }

        // Growth ratio should reflect the change
        assert!((pool.growth_ratio() - 1.25).abs() < 0.01, "40/32 = 1.25");
    }

    #[test]
    fn grow_neurons_pool_still_ticks() {
        let mut pool = NeuronPool::new("test", 16, PoolConfig::default());
        pool.grow_neurons_seeded(8, 42);

        // Should be able to tick without panic
        let input = vec![0i16; 24]; // 16 + 8
        pool.tick_simple(&input);
        assert_eq!(pool.tick_count, 1);

        // Force spike on a new neuron
        let mut input2 = vec![0i16; 24];
        input2[20] = 10000;
        pool.tick_simple(&input2);
        assert!(pool.neurons.spike_out[20], "new neuron should spike with strong input");
    }

    #[test]
    fn grow_spatial_extends_depth() {
        let dims = SpatialDims::new(4, 4, 4);
        let mut pool = NeuronPool::with_spatial_connectivity_seeded("test", dims, 0.05, PoolConfig::default(), 42);
        assert_eq!(pool.n_neurons, 64);

        pool.grow_neurons_seeded(16, 42);
        assert_eq!(pool.n_neurons, 80);
        // Depth should have increased: 16 / (4*4) = 1 extra layer → d=5
        assert_eq!(pool.dims.d, 5);
    }

    // ====================================================================
    // Pruning tests (A3)
    // ====================================================================

    #[test]
    fn prune_neurons_reduces_count() {
        let mut pool = NeuronPool::with_random_connectivity_seeded("test", 32, 0.05, PoolConfig::default(), 42);
        let syn_before = pool.synapse_count();
        assert!(syn_before > 0);

        let pruned = pool.prune_neurons(&[5, 10, 15]);
        assert_eq!(pruned, 3);
        assert_eq!(pool.n_neurons, 29);
        assert_eq!(pool.neurons.membrane.len(), 29);
        assert_eq!(pool.synapses.row_ptr.len(), 30); // 29 + 1

        // Synapse count should be reduced (some edges removed)
        assert!(pool.synapse_count() <= syn_before);

        // Should still be tickable
        let input = vec![0i16; 29];
        pool.tick_simple(&input);
    }

    #[test]
    fn prune_neurons_remaps_synapses() {
        let mut pool = NeuronPool::with_random_connectivity_seeded("test", 16, 0.1, PoolConfig::default(), 42);

        // Prune the middle neurons
        pool.prune_neurons(&[4, 5, 6, 7]);
        assert_eq!(pool.n_neurons, 12);

        // CSR integrity: all targets should be < n_neurons
        for src in 0..pool.n_neurons {
            for syn in pool.synapses.outgoing(src) {
                assert!((syn.target as u32) < pool.n_neurons,
                    "target {} out of range for {} neurons", syn.target, pool.n_neurons);
            }
        }

        // row_ptr should be monotonically non-decreasing
        for i in 0..pool.synapses.row_ptr.len() - 1 {
            assert!(pool.synapses.row_ptr[i] <= pool.synapses.row_ptr[i + 1],
                "row_ptr not monotonic at {}", i);
        }
    }

    #[test]
    fn prune_neurons_exc_inh_recount() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());
        // 10 neurons: 8 exc + 2 inh (indices 8, 9)
        assert_eq!(pool.n_excitatory, 8);
        assert_eq!(pool.n_inhibitory, 2);

        // Prune one excitatory neuron
        pool.prune_neurons(&[0]);
        assert_eq!(pool.n_neurons, 9);
        assert_eq!(pool.n_excitatory, 7);
        assert_eq!(pool.n_inhibitory, 2);

        // Prune one inhibitory neuron (originally at index 8, now at index 7 after pruning)
        // After pruning index 0 from [0..7 exc, 8..9 inh], we have [1..7 exc, 8..9 inh] → remapped to [0..6 exc, 7..8 inh]
        pool.prune_neurons(&[7]);
        assert_eq!(pool.n_neurons, 8);
        assert_eq!(pool.n_excitatory, 7);
        assert_eq!(pool.n_inhibitory, 1);
    }

    #[test]
    fn prune_out_of_bounds_ignored() {
        let mut pool = NeuronPool::new("test", 10, PoolConfig::default());
        let pruned = pool.prune_neurons(&[100, 200]);
        assert_eq!(pruned, 0);
        assert_eq!(pool.n_neurons, 10);
    }

    // ====================================================================
    // Migration tests (A5)
    // ====================================================================

    #[test]
    fn migration_flat_pool_noop() {
        let mut pool = NeuronPool::new("test", 32, PoolConfig::default());
        pool.spike_counts[0] = 100;
        let swaps = pool.migrate_neurons(42);
        assert_eq!(swaps, 0, "flat pool should not migrate");
    }

    #[test]
    fn migration_moves_toward_activity() {
        let dims = SpatialDims::new(4, 4, 4);
        let mut pool = NeuronPool::with_spatial_connectivity_seeded("test", dims, 0.05, PoolConfig::default(), 42);

        // Set high activity in one corner (neurons near index 0)
        for i in 0..8 {
            pool.spike_counts[i] = 50;
        }
        // Rest have zero activity

        let swaps = pool.migrate_neurons(42);
        // Some swaps should have occurred (low-activity neurons adjacent to high-activity)
        // The exact count depends on grid topology, but with 56 silent neurons
        // and 8 active ones, there should be some candidates
        eprintln!("[MIGRATION TEST] swaps: {}", swaps);
        // At minimum, the function shouldn't panic with spatial pools
        assert!(swaps <= (64 / 32 + 1) as u32, "should not exceed max_swaps");
    }

    #[test]
    fn migration_preserves_pool_integrity() {
        let dims = SpatialDims::new(4, 4, 4);
        let mut pool = NeuronPool::with_spatial_connectivity_seeded("test", dims, 0.05, PoolConfig::default(), 42);

        for i in 0..16 {
            pool.spike_counts[i] = 20;
        }

        pool.migrate_neurons(42);

        // After migration, pool should still be tickable
        assert_eq!(pool.n_neurons, 64);
        let input = vec![0i16; 64];
        pool.tick_simple(&input);

        // CSR integrity
        for src in 0..pool.n_neurons {
            for syn in pool.synapses.outgoing(src) {
                assert!((syn.target as u32) < pool.n_neurons,
                    "target {} out of range after migration", syn.target);
            }
        }
    }

    #[test]
    fn grow_then_prune_round_trip() {
        let mut pool = NeuronPool::with_random_connectivity_seeded("test", 32, 0.05, PoolConfig::default(), 42);

        // Grow
        pool.grow_neurons_seeded(8, 42);
        assert_eq!(pool.n_neurons, 40);

        // Prune the newly added neurons
        let pruned = pool.prune_neurons(&[32, 33, 34, 35, 36, 37, 38, 39]);
        assert_eq!(pruned, 8);
        assert_eq!(pool.n_neurons, 32);

        // Pool should still tick correctly
        let input = vec![5000i16; 32];
        pool.tick_simple(&input);
        assert!(pool.spike_count() > 0);
    }

    #[test]
    fn codec_v3_round_trip_with_activity() {
        let mut pool = NeuronPool::new("test", 16, PoolConfig::default());

        // Generate some activity
        let mut input = vec![0i16; 16];
        input[0] = 10000;
        input[5] = 10000;
        pool.tick_simple(&input);

        let count_0 = pool.neuron_activities()[0];
        let count_5 = pool.neuron_activities()[5];
        assert!(count_0 > 0 || count_5 > 0, "at least one neuron should have spiked");

        let path = std::env::temp_dir().join("neuropool_test_v3_activity.pool");
        pool.save(&path).expect("save failed");
        let loaded = NeuronPool::load(&path).expect("load failed");

        assert_eq!(loaded.neuron_activities().len(), 16);
        assert_eq!(loaded.neuron_activities()[0], count_0);
        assert_eq!(loaded.neuron_activities()[5], count_5);
        assert_eq!(loaded.growth_ratio(), 1.0);

        std::fs::remove_file(&path).ok();
    }

    // ====================================================================
    // Per-type refractory tests (Gap 1)
    // ====================================================================

    #[test]
    fn per_type_refractory_periods() {
        use crate::neuron::{NeuronProfile, flags};

        // Use refractory_ticks = 0 to enable per-profile defaults
        let mut config = PoolConfig::default();
        config.refractory_ticks = 0;
        let mut pool = NeuronPool::new("test", 4, config);

        // Neuron 0: FastSpiking (refractory = 1)
        pool.neurons.flags[0] = flags::encode(true, NeuronProfile::FastSpiking);
        // Neuron 1: IntrinsicBursting (refractory = 3)
        pool.neurons.flags[1] = flags::encode(false, NeuronProfile::IntrinsicBursting);
        // Neuron 2: RegularSpiking (refractory = 2) — default excitatory
        // Neuron 3: RegularSpiking (refractory = 2) — default excitatory

        // Force both neurons above threshold
        let strong_input = vec![8000i16; 4];

        // Tick 1: both spike
        pool.tick_simple(&strong_input);
        assert!(pool.neurons.spike_out[0], "n0 should spike");
        assert!(pool.neurons.spike_out[1], "n1 should spike");

        // Check refractory periods set correctly
        assert_eq!(pool.neurons.refract_remaining[0], 1, "FastSpiking should have refract=1");
        assert_eq!(pool.neurons.refract_remaining[1], 3, "IntrinsicBursting should have refract=3");
        assert_eq!(pool.neurons.refract_remaining[2], 2, "RegularSpiking should have refract=2");

        // Tick 2: n0 refractory (refract=1 → decrements to 0, skips), n1 refractory (refract=3 → 2)
        pool.tick_simple(&strong_input);
        assert!(!pool.neurons.spike_out[0], "FastSpiking still in 1-tick refractory");
        assert!(!pool.neurons.spike_out[1], "IntrinsicBursting still refractory");

        // Tick 3: n0 recovers (refract now 0), n1 still (refract=2 → 1)
        pool.tick_simple(&strong_input);
        assert!(pool.neurons.spike_out[0], "FastSpiking should fire after 1 dead tick");
        assert!(!pool.neurons.spike_out[1], "IntrinsicBursting still refractory");

        // Tick 4: n1 still refractory (refract=1 → 0)
        pool.tick_simple(&strong_input);
        assert!(!pool.neurons.spike_out[1], "IntrinsicBursting still refractory");

        // Tick 5: n1 recovers (refract now 0)
        pool.tick_simple(&strong_input);
        assert!(pool.neurons.spike_out[1], "IntrinsicBursting should fire after 3 dead ticks");
    }

    // ====================================================================
    // Per-type threshold adaptation tests (Gap 2)
    // ====================================================================

    #[test]
    fn threshold_adaptation_intrinsic_bursting_stable() {
        use crate::neuron::{NeuronProfile, flags};

        let mut config = PoolConfig::default();
        config.homeostatic_rate = 1;
        let mut pool = NeuronPool::new("test", 4, config);

        // Set neuron 0 to IntrinsicBursting
        pool.neurons.flags[0] = flags::encode(false, NeuronProfile::IntrinsicBursting);
        pool.neurons.leak[0] = NeuronProfile::IntrinsicBursting.default_leak();

        let initial_threshold = pool.neurons.threshold[0];

        // Force lots of activity to trigger homeostatic adjustment
        // (homeostatic runs every 100 ticks)
        let strong_input = vec![8000i16; 4];
        for _ in 0..200 {
            pool.tick_simple(&strong_input);
        }

        // IntrinsicBursting should NOT have its threshold adjusted (adapt_scale = 0)
        assert_eq!(
            pool.neurons.threshold[0], initial_threshold,
            "IntrinsicBursting threshold should not adapt homeostatically"
        );
    }

    #[test]
    fn fitness_score_rewards_coherent_da() {
        let pool = NeuronPool::new("test", 64, PoolConfig::default());

        // High coherence + high DA + healthy activity/density = high fitness
        let good = FitnessInput {
            coherence: 200,
            da: 200,          // 72 above baseline
            cortisol: 20,     // below baseline — no penalty
            active_ratio: 0.20,
            synapse_density: 0.50,
        };
        let good_score = pool.compute_fitness(&good);

        // Low coherence + low DA + high cortisol = low fitness
        let bad = FitnessInput {
            coherence: 30,
            da: 80,           // below baseline — no reward
            cortisol: 200,    // high stress
            active_ratio: 0.01,  // nearly silent
            synapse_density: 0.05, // almost no synapses
        };
        let bad_score = pool.compute_fitness(&bad);

        assert!(
            good_score > bad_score + 50,
            "healthy region ({}) should score much higher than struggling ({})",
            good_score, bad_score
        );
        assert!(good_score >= 140, "healthy region should score well: {}", good_score);
    }

    #[test]
    fn evolve_structure_applies_mutations() {
        // Struggling pool: low coherence, low DA, sparse synapses
        let mut config = PoolConfig::default();
        config.evolution.struggle_threshold = 100;
        config.evolution.struggle_mutation_prob = 255; // always mutate when struggling
        config.evolution.mutation_budget = 4;

        let mut pool = NeuronPool::with_random_connectivity_seeded("test", 64, 0.05, config, 42);

        let struggling = FitnessInput {
            coherence: 20,
            da: 90,
            cortisol: 180,
            active_ratio: 0.02,
            synapse_density: 0.10,
        };

        let result = pool.evolve_structure(&struggling, 12345);
        assert_eq!(result.generation, 1, "generation should increment");
        assert!(result.mutations_applied > 0, "struggling pool should receive mutations");
        assert!(!pool.journal.is_empty(), "journal should have entries");

        // Elite pool: high fitness → much lower mutation rate
        let mut elite_config = PoolConfig::default();
        elite_config.evolution.elite_threshold = 50; // very low bar so pool qualifies
        elite_config.evolution.elite_mutation_prob = 0; // never mutate elites

        let mut elite_pool = NeuronPool::with_random_connectivity_seeded("elite", 64, 0.05, elite_config, 42);
        let elite_fitness = FitnessInput {
            coherence: 250,
            da: 230,
            cortisol: 10,
            active_ratio: 0.20,
            synapse_density: 0.50,
        };

        let elite_result = elite_pool.evolve_structure(&elite_fitness, 99999);
        assert_eq!(elite_result.mutations_applied, 0, "elite pool should skip mutations");
        assert_eq!(elite_result.generation, 1, "generation still increments");
    }

    #[test]
    fn journal_records_mutations() {
        let mut config = PoolConfig::default();
        config.evolution.struggle_mutation_prob = 255;
        config.evolution.mutation_budget = 4;

        let mut pool = NeuronPool::with_random_connectivity_seeded("test", 64, 0.05, config, 42);

        let input = FitnessInput {
            coherence: 20,
            da: 90,
            cortisol: 180,
            active_ratio: 0.02,
            synapse_density: 0.10,
        };

        // Run evolution multiple times
        for i in 0..5 {
            pool.evolve_structure(&input, i * 1000 + 42);
        }

        assert_eq!(pool.generation, 5, "generation should be 5 after 5 calls");
        assert!(pool.journal.len() > 0, "journal should have entries after evolution");
        assert!(pool.journal.len() <= 16, "journal should not exceed max 16 entries");
    }

    #[test]
    fn rollback_restores_structure() {
        let mut pool = NeuronPool::with_random_connectivity_seeded("test", 64, 0.05, PoolConfig::default(), 42);

        // Capture original state
        let orig_n = pool.n_neurons;
        let orig_n_exc = pool.n_excitatory;
        let orig_flags_0 = pool.neurons.flags[0];
        let orig_threshold_0 = pool.neurons.threshold[0];
        let orig_synapse_count = pool.synapses.synapses.len();

        let cp = pool.checkpoint();

        // Mutate: grow neurons, change flags
        pool.grow_neurons_seeded(8, 999);
        assert_eq!(pool.n_neurons, 72, "should have grown 8 neurons");

        // Change some flags
        pool.neurons.flags[0] = 0xFF;
        pool.neurons.threshold[0] = 12345;

        // Rollback
        pool.rollback(&cp);

        assert_eq!(pool.n_neurons, orig_n, "n_neurons restored");
        assert_eq!(pool.n_excitatory, orig_n_exc, "n_excitatory restored");
        assert_eq!(pool.neurons.flags[0], orig_flags_0, "flags restored");
        assert_eq!(pool.neurons.threshold[0], orig_threshold_0, "threshold restored");

        // Synapse count should match
        assert_eq!(pool.synapses.synapses.len(), orig_synapse_count, "synapse count restored");

        // Dynamic state should be reset to resting
        assert_eq!(pool.neurons.membrane[0], pool.config.resting_potential, "membrane reset to resting");
        assert_eq!(pool.neurons.trace[0], 0, "trace reset");
    }
}
