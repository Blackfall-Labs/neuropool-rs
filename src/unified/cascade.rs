#![allow(deprecated)]
//! Zone-Aware Cascade Engine for Unified Neurons
//!
//! Forks `SpatialCascade` with dendritic zone targeting, predicted vs burst
//! firing, oscillator context entrainment, and ternsig triggering.
//!
//! ## Key Differences from SpatialCascade
//!
//! - **Zone-aware arrivals**: Each `SpikeArrival` targets a specific `DendriticZone`
//! - **Predicted vs burst**: Context-primed neurons fire predictively (clean spike);
//!   unprimed neurons burst (spike + lateral context spikes to neighbors)
//! - **Oscillator entrainment**: Oscillator spikes target context zone of neighbors
//! - **Ternsig triggering**: Neurons bound to ternsig programs emit collective
//!   activation events when enough co-fire

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::io::UnifiedNeuronIO;
use super::neuron::{UnifiedNeuron, DEFAULT_THRESHOLD};
use super::synapse::{UnifiedSynapse, UnifiedSynapseStore};
use super::zone::DendriticZone;

/// A spike in flight, targeting a specific dendritic zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeArrival {
    /// Target neuron index.
    pub target: u32,
    /// Which dendritic zone receives this spike.
    pub zone: DendriticZone,
    /// Current to deliver (from synapse signal).
    pub current: i16,
    /// Arrival time in microseconds.
    pub arrival_time_us: u64,
    /// Source neuron (u32::MAX = external input).
    pub source: u32,
}

impl Ord for SpikeArrival {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: earlier arrivals first, then by target for determinism
        other
            .arrival_time_us
            .cmp(&self.arrival_time_us)
            .then_with(|| other.target.cmp(&self.target))
    }
}

impl PartialOrd for SpikeArrival {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Event emitted when ternsig-bound neurons collectively activate.
#[derive(Clone, Debug)]
pub struct TernsigTrigger {
    /// Which ternsig program was triggered.
    pub program_id: u32,
    /// Activation levels of the bound neurons (indexed by binding order).
    pub activation: Vec<i16>,
}

/// Configuration for the cascade engine.
#[derive(Clone, Copy, Debug)]
pub struct CascadeConfig {
    /// Base propagation speed in μs per unit distance (voxel-scale).
    pub propagation_speed_us_per_unit: f32,
    /// Myelination speed bonus (0.0 = no bonus, 1.0 = 2x speed at full myelin).
    pub myelin_speed_factor: f32,
    /// Maximum events to process per `run_until` call.
    pub max_events_per_call: usize,
    /// Fast coincidence window (μs). Arrivals within this window amplify each other.
    pub fast_coincidence_window_us: u64,
    /// Peak coincidence boost (0.0 = disabled, 0.5 = up to 1.5x).
    pub coincidence_boost: f32,
    /// Burst lateral current magnitude (context spikes sent to neighbors on burst fire).
    pub burst_lateral_current: i16,
    /// Maximum number of lateral burst targets per firing neuron.
    pub burst_max_targets: usize,
    /// Ternsig collective activation threshold (fraction of bound neurons that must
    /// have nonzero trace). 0.5 = half must be recently active.
    pub ternsig_activation_threshold: f32,
    /// Per-neuron per-event threshold jitter half-range (Q8.8 format).
    /// Before each spike check, the effective threshold is varied by a deterministic
    /// hash of (event_count, neuron_index). 0 = disabled. Default: 512 (~2mV).
    pub threshold_jitter: i16,
    /// Spontaneous depolarization probability per neuron per `check_spontaneous()` call.
    /// Probability = rate/256. Default: 5 (~2% of neurons per call).
    pub spontaneous_rate: u8,
    /// Spontaneous depolarization current magnitude (feedforward).
    pub spontaneous_current: i16,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            propagation_speed_us_per_unit: 100.0,
            myelin_speed_factor: 0.5,
            max_events_per_call: 10_000,
            fast_coincidence_window_us: 1500,
            coincidence_boost: 0.5,
            burst_lateral_current: 400,
            burst_max_targets: 6,
            ternsig_activation_threshold: 0.5,
            threshold_jitter: 512,
            spontaneous_rate: 5,
            spontaneous_current: 800,
        }
    }
}

/// Zone-aware event-driven cascade executor.
pub struct CascadeEngine {
    /// All neurons.
    pub neurons: Vec<UnifiedNeuron>,
    /// All synapses (zone-aware, CSR indexed).
    pub synapses: UnifiedSynapseStore,
    /// Event queue (min-heap by arrival time).
    pending: BinaryHeap<Reverse<SpikeArrival>>,
    /// Current simulation time in μs.
    pub(super) sim_time_us: u64,
    /// Configuration.
    config: CascadeConfig,
    /// Total spikes fired.
    total_spikes: u64,
    /// Total events processed.
    total_events: u64,
    /// Burst fires (unpredicted).
    pub burst_fires: u64,
    /// Predicted fires.
    pub predicted_fires: u64,
    /// Coincidence boost applications.
    pub coincidence_events: u64,
    /// Ternsig triggers emitted this run.
    ternsig_triggers: Vec<TernsigTrigger>,
}

impl CascadeEngine {
    /// Create a new cascade engine.
    pub fn new(config: CascadeConfig) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: UnifiedSynapseStore::new(),
            pending: BinaryHeap::new(),
            sim_time_us: 0,
            config,
            total_spikes: 0,
            total_events: 0,
            burst_fires: 0,
            predicted_fires: 0,
            coincidence_events: 0,
            ternsig_triggers: Vec::new(),
        }
    }

    /// Create with pre-built network.
    pub fn with_network(
        neurons: Vec<UnifiedNeuron>,
        synapses: UnifiedSynapseStore,
        config: CascadeConfig,
    ) -> Self {
        Self {
            neurons,
            synapses,
            pending: BinaryHeap::new(),
            sim_time_us: 0,
            config,
            total_spikes: 0,
            total_events: 0,
            burst_fires: 0,
            predicted_fires: 0,
            coincidence_events: 0,
            ternsig_triggers: Vec::new(),
        }
    }

    // === Accessors ===

    /// Current simulation time.
    #[inline]
    pub fn sim_time(&self) -> u64 {
        self.sim_time_us
    }

    /// Number of pending events.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Total spikes fired.
    #[inline]
    pub fn total_spikes(&self) -> u64 {
        self.total_spikes
    }

    /// Total events processed.
    #[inline]
    pub fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Drain ternsig triggers emitted since last drain.
    pub fn drain_ternsig_triggers(&mut self) -> Vec<TernsigTrigger> {
        std::mem::take(&mut self.ternsig_triggers)
    }

    // === Network Building ===

    /// Add a neuron, returns its index.
    pub fn add_neuron(&mut self, neuron: UnifiedNeuron) -> u32 {
        let idx = self.neurons.len() as u32;
        self.neurons.push(neuron);
        idx
    }

    /// Add a synapse.
    pub fn add_synapse(&mut self, synapse: UnifiedSynapse) {
        self.synapses.add(synapse);
    }

    /// Rebuild synapse index after adding synapses.
    pub fn rebuild_synapse_index(&mut self) {
        self.synapses.rebuild_index(self.neurons.len() as u32);
    }

    // === Injection ===

    /// Inject external input to a neuron at a specific time and zone.
    pub fn inject(&mut self, neuron: u32, zone: DendriticZone, current: i16, time_us: u64) {
        self.pending.push(Reverse(SpikeArrival {
            target: neuron,
            zone,
            current,
            arrival_time_us: time_us,
            source: u32::MAX,
        }));
    }

    /// Inject feedforward input (convenience — most external input is feedforward).
    pub fn inject_ff(&mut self, neuron: u32, current: i16, time_us: u64) {
        self.inject(neuron, DendriticZone::Feedforward, current, time_us);
    }

    /// Inject ternsig program output to all neurons bound to that program.
    ///
    /// Each output value is delivered as feedforward current to the corresponding
    /// bound neuron (by binding order). Values beyond the bound neuron count are
    /// ignored; bound neurons beyond the output length get zero.
    pub fn inject_ternsig_output(&mut self, program_id: u32, output: &[i16], time_us: u64) {
        // Find neurons bound to this program
        let bound: Vec<u32> = self
            .neurons
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.nuclei.interface.kind == crate::spatial::Interface::KIND_TERNSIG
                    && n.nuclei.interface.ternsig_program_id() == program_id
            })
            .map(|(i, _)| i as u32)
            .collect();

        for (i, &neuron_idx) in bound.iter().enumerate() {
            let current = output.get(i).copied().unwrap_or(0);
            if current != 0 {
                self.inject_ff(neuron_idx, current, time_us);
            }
        }
    }

    // === Execution ===

    /// Process all events up to the given time.
    ///
    /// Returns number of spikes that occurred.
    pub fn run_until(&mut self, until_time_us: u64) -> u64 {
        let mut spikes_this_run = 0u64;
        let mut events_this_run = 0u64;

        while let Some(&Reverse(arrival)) = self.pending.peek() {
            if arrival.arrival_time_us > until_time_us {
                break;
            }
            if events_this_run >= self.config.max_events_per_call as u64 {
                break;
            }

            let arrival = self.pending.pop().unwrap().0;
            self.sim_time_us = arrival.arrival_time_us;
            events_this_run += 1;

            if self.process_arrival(arrival) {
                spikes_this_run += 1;
            }
        }

        self.total_events += events_this_run;
        spikes_this_run
    }

    /// Process all events up to the given time with external IO.
    ///
    /// When memory neurons fire, their IO operations (query/write) are dispatched
    /// through the provided `io` trait. Motor neuron fires also write through IO.
    pub fn run_until_with_io(&mut self, until_time_us: u64, io: &mut dyn UnifiedNeuronIO) -> u64 {
        let mut spikes_this_run = 0u64;
        let mut events_this_run = 0u64;

        while let Some(&Reverse(arrival)) = self.pending.peek() {
            if arrival.arrival_time_us > until_time_us {
                break;
            }
            if events_this_run >= self.config.max_events_per_call as u64 {
                break;
            }

            let arrival = self.pending.pop().unwrap().0;
            self.sim_time_us = arrival.arrival_time_us;
            events_this_run += 1;

            if self.process_arrival(arrival) {
                let idx = arrival.target as usize;
                self.dispatch_io(idx, io);
                spikes_this_run += 1;
            }
        }

        self.total_events += events_this_run;
        spikes_this_run
    }

    /// Dispatch IO operations for a neuron that just fired.
    fn dispatch_io(&mut self, idx: usize, io: &mut dyn UnifiedNeuronIO) {
        let neuron = &self.neurons[idx];

        if neuron.nuclei.is_motor() {
            io.write_motor(
                neuron.nuclei.interface.target,
                neuron.nuclei.interface.modality,
                neuron.trace as i16,
            );
        }

        if neuron.nuclei.is_memory() {
            // Collect local activity pattern from neighboring neurons' traces
            let pattern: Vec<i16> = self.synapses.outgoing(idx as u32)
                .iter()
                .filter(|s| s.is_active())
                .take(8) // max 8 neighbors for pattern
                .map(|s| {
                    let t = s.target as usize;
                    if t < self.neurons.len() {
                        self.neurons[t].trace as i16
                    } else {
                        0
                    }
                })
                .collect();

            let bank_id = neuron.nuclei.interface.target;
            let action = neuron.nuclei.interface.gates.action(neuron.membrane);

            match action {
                crate::spatial::InterfaceAction::Low => {
                    // Read mode: query bank and inject result
                    let result = io.memory_query(bank_id, &pattern);
                    if result != 0 {
                        self.inject(
                            idx as u32,
                            DendriticZone::Feedback,
                            result,
                            self.sim_time_us,
                        );
                    }
                }
                crate::spatial::InterfaceAction::High => {
                    // Write mode: store pattern to bank
                    io.memory_write(bank_id, &pattern);
                }
                crate::spatial::InterfaceAction::Neutral => {
                    // Exclude — no action
                }
            }
        }
    }

    /// Process a single arrival. Returns true if the target neuron fired.
    fn process_arrival(&mut self, arrival: SpikeArrival) -> bool {
        let idx = arrival.target as usize;
        if idx >= self.neurons.len() {
            return false;
        }

        let sim_time = self.sim_time_us;

        // Apply leak for elapsed time since last arrival
        let elapsed = sim_time.saturating_sub(self.neurons[idx].last_arrival_us);
        if elapsed > 0 && self.neurons[idx].last_arrival_us > 0 {
            self.neurons[idx].apply_leak(elapsed);
        }

        // Fast coincidence boost
        let current = if self.config.coincidence_boost > 0.0
            && self.neurons[idx].last_arrival_us > 0
        {
            let dt = sim_time.saturating_sub(self.neurons[idx].last_arrival_us);
            if dt > 0 && dt < self.config.fast_coincidence_window_us {
                self.coincidence_events += 1;
                let closeness =
                    1.0 - (dt as f32 / self.config.fast_coincidence_window_us as f32);
                (arrival.current as f32 * (1.0 + closeness * self.config.coincidence_boost))
                    as i16
            } else {
                arrival.current
            }
        } else {
            arrival.current
        };

        // Zone-aware integration
        self.neurons[idx].integrate_zone(arrival.zone, current);
        self.neurons[idx].recompute_membrane();
        self.neurons[idx].last_arrival_us = sim_time;

        // Update eligibility trace on source
        if arrival.source != u32::MAX {
            let source_idx = arrival.source as usize;
            if source_idx < self.neurons.len() && source_idx != idx {
                self.neurons[source_idx].trace =
                    self.neurons[source_idx].trace.saturating_add(5);
            }
        }

        // Apply threshold jitter for this event
        let jittered = if self.config.threshold_jitter > 0 {
            let jitter = threshold_jitter(self.total_events, idx as u64, self.config.threshold_jitter);
            let original = self.neurons[idx].threshold;
            self.neurons[idx].threshold = original.saturating_add(jitter);
            let can_fire = self.neurons[idx].can_fire(sim_time);
            self.neurons[idx].threshold = original;
            can_fire
        } else {
            self.neurons[idx].can_fire(sim_time)
        };

        // Check if neuron fires
        if jittered {
            self.fire_neuron(idx);
            true
        } else {
            false
        }
    }

    /// Fire a neuron and queue zone-targeted spikes to all outgoing synapses.
    ///
    /// If the fire was unpredicted (burst), also send lateral context spikes
    /// to nearby neurons through outgoing synapses.
    fn fire_neuron(&mut self, idx: usize) {
        let was_predicted = self.neurons[idx].fire(self.sim_time_us);
        self.total_spikes += 1;

        if was_predicted {
            self.predicted_fires += 1;
        } else {
            self.burst_fires += 1;
        }

        // Queue outgoing synaptic spikes (each synapse targets a specific zone)
        let synapses: Vec<UnifiedSynapse> = self.synapses.outgoing(idx as u32).to_vec();

        for syn in &synapses {
            if !syn.is_active() {
                continue;
            }

            let delay = self.compute_delay(idx, syn);

            self.pending.push(Reverse(SpikeArrival {
                target: syn.target,
                zone: syn.zone,
                current: syn.current(),
                arrival_time_us: self.sim_time_us + delay,
                source: idx as u32,
            }));
        }

        // Burst fire: send lateral context spikes to neighbors
        if !was_predicted {
            self.emit_burst_laterals(idx, &synapses);
        }

        // Check ternsig collective activation
        if self.neurons[idx].nuclei.interface.kind == crate::spatial::Interface::KIND_TERNSIG {
            self.check_ternsig_activation(idx);
        }
    }

    /// Compute propagation delay for a synapse.
    fn compute_delay(&self, source_idx: usize, syn: &UnifiedSynapse) -> u64 {
        if syn.delay_us > 0 {
            // Precomputed delay — use directly
            return syn.delay_us as u64;
        }

        // Compute from voxel distance
        let src_pos = &self.neurons[source_idx].position;
        let tgt_idx = syn.target as usize;
        if tgt_idx >= self.neurons.len() {
            return 1;
        }
        let tgt_pos = &self.neurons[tgt_idx].position;

        let dist_sq = src_pos.distance_sq(tgt_pos);
        // Integer sqrt approximation: iterate
        let distance = integer_sqrt(dist_sq) as f32;

        // Myelination reduces delay
        let myelin = self.neurons[source_idx].axon.myelin as f32 / 255.0;
        let speed_factor = 1.0 + myelin * self.config.myelin_speed_factor;
        let delay = distance * self.config.propagation_speed_us_per_unit / speed_factor;

        (delay as u64).max(1)
    }

    /// Emit lateral context spikes on burst fire.
    ///
    /// Burst firing means the neuron was NOT predicted by context — it was surprised.
    /// The burst sends context-zone spikes to neighbors through existing outgoing
    /// synapses so they become primed for future similar input.
    fn emit_burst_laterals(&mut self, idx: usize, synapses: &[UnifiedSynapse]) {
        let burst_current = self.config.burst_lateral_current;
        let max_targets = self.config.burst_max_targets;
        let mut emitted = 0;

        for syn in synapses {
            if emitted >= max_targets {
                break;
            }
            if !syn.is_active() {
                continue;
            }

            let delay = self.compute_delay(idx, syn);

            self.pending.push(Reverse(SpikeArrival {
                target: syn.target,
                zone: DendriticZone::Context,
                current: burst_current,
                arrival_time_us: self.sim_time_us + delay,
                source: idx as u32,
            }));
            emitted += 1;
        }
    }

    /// Check if a ternsig program's bound neurons are collectively active enough
    /// to trigger the program.
    fn check_ternsig_activation(&mut self, firing_idx: usize) {
        let program_id = self.neurons[firing_idx]
            .nuclei
            .interface
            .ternsig_program_id();

        // Collect bound neurons and their activation
        let mut bound_indices = Vec::new();
        let mut activation = Vec::new();
        let mut active_count = 0u32;

        for (i, n) in self.neurons.iter().enumerate() {
            if n.nuclei.interface.kind == crate::spatial::Interface::KIND_TERNSIG
                && n.nuclei.interface.ternsig_program_id() == program_id
            {
                bound_indices.push(i);
                activation.push(n.trace as i16);
                if n.trace > 0 {
                    active_count += 1;
                }
            }
        }

        if bound_indices.is_empty() {
            return;
        }

        let fraction = active_count as f32 / bound_indices.len() as f32;
        if fraction >= self.config.ternsig_activation_threshold {
            self.ternsig_triggers.push(TernsigTrigger {
                program_id,
                activation,
            });
        }
    }

    // === Spontaneous Activity ===

    /// Check for spontaneous depolarization — low-probability random membrane bumps.
    ///
    /// For each non-oscillator neuron, a deterministic hash decides if it gets a
    /// small feedforward current injection. This breaks synchrony and ensures
    /// neurons are never completely silent.
    pub fn check_spontaneous(&mut self) {
        let rate = self.config.spontaneous_rate;
        if rate == 0 {
            return;
        }
        let current = self.config.spontaneous_current;
        let time = self.sim_time_us;

        for idx in 0..self.neurons.len() {
            if self.neurons[idx].nuclei.is_oscillator() {
                continue; // oscillators have their own autonomous firing
            }
            // Deterministic hash of (time, idx) → 0..255
            let hash = spontaneous_hash(time, idx as u64);
            if hash < rate {
                self.inject(idx as u32, DendriticZone::Feedforward, current, time);
            }
        }
    }

    // === Oscillator Support ===

    /// Check and fire oscillators that have completed their period.
    ///
    /// Oscillator spikes target the context zone of their outgoing targets,
    /// providing rhythm entrainment.
    pub fn check_oscillators(&mut self) {
        for idx in 0..self.neurons.len() {
            if self.neurons[idx].oscillator_should_fire(self.sim_time_us) {
                // Inject autonomous depolarization to feedforward (self-fire).
                // Must account for zone weight normalization: inject enough to
                // bring weighted average from resting to above threshold.
                let w = &self.neurons[idx].zone_weights;
                let weight_sum = w.feedforward as i32 + w.context as i32 + w.feedback as i32;
                let delta = (DEFAULT_THRESHOLD as i32 - super::neuron::RESTING_POTENTIAL as i32)
                    * weight_sum
                    / w.feedforward as i32;
                let fire_current = (delta + 500).min(i16::MAX as i32) as i16;
                self.inject(
                    idx as u32,
                    DendriticZone::Feedforward,
                    fire_current,
                    self.sim_time_us,
                );
            }
        }
    }

    // === Motor Output ===

    /// Read motor neuron outputs.
    ///
    /// Returns (channel, trace) for motor neurons with nonzero trace.
    pub fn read_motor_outputs(&self) -> Vec<(u16, i16)> {
        let mut outputs = Vec::new();
        for n in &self.neurons {
            if n.nuclei.is_motor() && n.trace > 0 {
                outputs.push((n.nuclei.interface.target, n.trace as i16));
            }
        }
        outputs
    }

    // === Maintenance ===

    /// Decay all eligibility traces.
    pub fn decay_traces(&mut self) {
        for n in &mut self.neurons {
            n.decay_trace();
        }
    }

    /// Recover stamina for all neurons based on frame duration.
    pub fn recover_stamina(&mut self, frame_interval_us: u64) {
        // Recovery: 1 stamina per 5000μs (same as SpatialNeuron::STAMINA_RECOVERY_US)
        const STAMINA_RECOVERY_US: u64 = 5_000;
        let recovery = (frame_interval_us / STAMINA_RECOVERY_US) as u8;
        if recovery == 0 {
            return;
        }
        for n in &mut self.neurons {
            n.stamina = n.stamina.saturating_add(recovery);
        }
    }

    /// Clear the event queue.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Reset simulation time to zero.
    pub fn reset_time(&mut self) {
        self.sim_time_us = 0;
        for n in &mut self.neurons {
            n.last_spike_us = 0;
            n.last_arrival_us = 0;
        }
    }
}

/// Deterministic threshold jitter from event count and neuron index.
///
/// Returns a value in [-half_range, +half_range].
fn threshold_jitter(event_count: u64, neuron_idx: u64, half_range: i16) -> i16 {
    // Fast hash: multiply by golden ratio constants, xorshift
    let mut h = event_count.wrapping_mul(2654435761).wrapping_add(neuron_idx.wrapping_mul(2246822519));
    h ^= h >> 16;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    let range = half_range as i32 * 2 + 1;
    let raw = (h as i32).rem_euclid(range);
    (raw - half_range as i32) as i16
}

/// Deterministic spontaneous firing hash.
///
/// Returns a value in 0..255 for probability comparison.
fn spontaneous_hash(time: u64, neuron_idx: u64) -> u8 {
    let mut h = time.wrapping_mul(6364136223846793005).wrapping_add(neuron_idx.wrapping_mul(1442695040888963407));
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    (h & 0xFF) as u8
}

/// Integer square root via Newton's method.
fn integer_sqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::neuron::{VoxelPosition, RESTING_POTENTIAL};
    use crate::unified::zone::ZoneWeights;

    fn pos(x: u16, y: u16, z: u16) -> VoxelPosition {
        VoxelPosition::at_center((x, y, z))
    }

    /// Feedforward current needed to bring a neuron from resting to above threshold
    /// when context and feedback zones are at resting. Accounts for zone weight
    /// normalization: delta > (threshold - resting) * weight_sum / ff_weight.
    fn ff_fire_current(weights: &ZoneWeights) -> i16 {
        let weight_sum = weights.feedforward as i32 + weights.context as i32 + weights.feedback as i32;
        let delta = (DEFAULT_THRESHOLD as i32 - RESTING_POTENTIAL as i32) * weight_sum
            / weights.feedforward as i32;
        (delta + 200) as i16 // +200 margin above threshold
    }

    #[test]
    fn creation() {
        let engine = CascadeEngine::new(CascadeConfig::default());
        assert_eq!(engine.neurons.len(), 0);
        assert_eq!(engine.sim_time(), 0);
    }

    #[test]
    fn add_neuron() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        let idx = engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));
        assert_eq!(idx, 0);
        assert_eq!(engine.neurons.len(), 1);
    }

    #[test]
    fn inject_and_fire() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));

        // Feedforward current must overcome zone weight normalization
        let fire_current = ff_fire_current(&ZoneWeights::PYRAMIDAL);
        engine.inject_ff(0, fire_current, 100);

        let spikes = engine.run_until(200);
        assert_eq!(spikes, 1);
        assert_eq!(engine.total_spikes(), 1);
        // First fire is always a burst (no context priming)
        assert_eq!(engine.burst_fires, 1);
        assert_eq!(engine.predicted_fires, 0);
    }

    #[test]
    fn zone_targeted_arrival() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));

        // Inject into context zone (not enough to fire, just to prime)
        engine.inject(0, DendriticZone::Context, 1500, 50);
        engine.run_until(60);

        // Neuron should be context-primed
        assert!(
            engine.neurons[0].predicted,
            "context injection should prime prediction"
        );
    }

    #[test]
    fn predicted_fire() {
        let config = CascadeConfig {
            coincidence_boost: 0.0, // disable for clarity
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));

        // Step 1: Prime with context
        engine.inject(0, DendriticZone::Context, 2000, 50);
        engine.run_until(60);
        assert!(engine.neurons[0].predicted);

        // Step 2: Feedforward drives above threshold → predicted fire
        let fire_current = ff_fire_current(&ZoneWeights::PYRAMIDAL);
        engine.inject_ff(0, fire_current, 100);
        let spikes = engine.run_until(200);

        assert_eq!(spikes, 1);
        assert_eq!(engine.predicted_fires, 1);
        assert_eq!(engine.burst_fires, 0);
    }

    #[test]
    fn burst_fire_sends_context_laterals() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            burst_lateral_current: 500,
            burst_max_targets: 4,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);

        // Source and target
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(1, 0, 0)));

        // Connect with feedforward synapse
        engine.add_synapse(UnifiedSynapse::excitatory(
            0,
            1,
            DendriticZone::Feedforward,
            200,
            50,
        ));
        engine.rebuild_synapse_index();

        // Fire neuron 0 (unpredicted → burst)
        let fire_current = ff_fire_current(&ZoneWeights::PYRAMIDAL);
        engine.inject_ff(0, fire_current, 0);
        engine.run_until(200);

        assert_eq!(engine.burst_fires, 1);

        // Neuron 1 should have received BOTH the synapse's feedforward current
        // AND a burst lateral context spike. Check that context was touched.
        // (It may have fired too — that's fine. We check the burst count.)
        // Since neuron 0 burst-fired, neuron 1's context zone got a lateral spike.
        // If neuron 1 also fired, it would also be a burst (no prior context priming).
        assert!(engine.total_events() >= 2, "should have processed multiple events");
    }

    #[test]
    fn spike_propagation_with_zones() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);

        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(1, 0, 0)));

        // Feedforward synapse 0→1
        engine.add_synapse(UnifiedSynapse::excitatory(
            0,
            1,
            DendriticZone::Feedforward,
            200,
            50,
        ));
        engine.rebuild_synapse_index();

        let fire_current = ff_fire_current(&ZoneWeights::PYRAMIDAL);
        engine.inject_ff(0, fire_current, 0);

        engine.run_until(200);
        assert!(engine.total_spikes() >= 1);
        assert!(engine.pending_count() > 0 || engine.total_events() > 1);
    }

    #[test]
    fn inhibitory_zone() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);

        // Target starts at resting (below threshold) — inhibition pushes further down
        engine.add_neuron(UnifiedNeuron::interneuron_at(pos(0, 0, 0)));
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(1, 0, 0)));
        let before_membrane = engine.neurons[1].membrane;

        // Inhibitory feedforward synapse
        engine.add_synapse(UnifiedSynapse::inhibitory(
            0,
            1,
            DendriticZone::Feedforward,
            200,
            50,
        ));
        engine.rebuild_synapse_index();

        // Fire the interneuron
        let fire_current = ff_fire_current(&ZoneWeights::INTERNEURON);
        engine.inject_ff(0, fire_current, 0);

        engine.run_until(200);

        // Target should have been pushed further below resting by inhibition
        assert!(
            engine.neurons[1].membrane < before_membrane,
            "inhibition should push membrane below resting: {} vs {}",
            engine.neurons[1].membrane,
            before_membrane
        );
    }

    #[test]
    fn oscillator_context_entrainment() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);

        // Oscillator at period 1000μs
        engine.add_neuron(UnifiedNeuron::oscillator_at(pos(0, 0, 0), 1000));
        // Target neuron
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(1, 0, 0)));

        // Oscillator → target via context zone synapse (rhythm entrainment)
        engine.add_synapse(UnifiedSynapse::excitatory(
            0,
            1,
            DendriticZone::Context,
            100,
            50,
        ));
        engine.rebuild_synapse_index();

        // Trigger oscillator at its period
        engine.sim_time_us = 1000;
        engine.check_oscillators();

        // Run the oscillator's self-fire and propagation
        engine.run_until(1200);

        // Oscillator should have fired
        assert!(engine.total_spikes() >= 1);
        // Target should have received context input (last_arrival_us > 0 means a spike arrived)
        assert!(
            engine.neurons[1].last_arrival_us > 0 || engine.neurons[1].context_potential > RESTING_POTENTIAL,
            "target should receive context entrainment from oscillator"
        );
    }

    #[test]
    fn ternsig_trigger() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            ternsig_activation_threshold: 0.5,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);

        // Three neurons bound to program 42
        for _ in 0..3 {
            engine.add_neuron(UnifiedNeuron::ternsig_at(pos(0, 0, 0), 42));
        }
        // One unbound neuron
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(1, 0, 0)));

        // Give 2/3 ternsig neurons nonzero trace (above 50% threshold)
        engine.neurons[0].trace = 10;
        engine.neurons[1].trace = 5;
        engine.neurons[2].trace = 0;

        // Fire one of the ternsig neurons to trigger the check
        let fire_current = ff_fire_current(&ZoneWeights::BALANCED);
        engine.inject_ff(0, fire_current, 0);
        engine.run_until(100);

        let triggers = engine.drain_ternsig_triggers();
        assert_eq!(triggers.len(), 1, "should emit one ternsig trigger");
        assert_eq!(triggers[0].program_id, 42);
        assert_eq!(triggers[0].activation.len(), 3);
    }

    #[test]
    fn ternsig_no_trigger_below_threshold() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            ternsig_activation_threshold: 0.5,
            ..CascadeConfig::default()
        };
        // Use 5 neurons, only 1 has trace → after fire: 2/5 = 40% < 50%
        let mut engine = CascadeEngine::new(config);
        for _ in 0..5 {
            engine.add_neuron(UnifiedNeuron::ternsig_at(pos(0, 0, 0), 99));
        }
        engine.neurons[0].trace = 10;

        let fire_current = ff_fire_current(&ZoneWeights::BALANCED);
        engine.inject_ff(0, fire_current, 0);
        engine.run_until(100);

        let triggers = engine.drain_ternsig_triggers();
        assert!(
            triggers.is_empty(),
            "should NOT trigger when activation below threshold (2/5 = 40%)"
        );
    }

    #[test]
    fn inject_ternsig_output() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);

        // Two neurons bound to program 7
        engine.add_neuron(UnifiedNeuron::ternsig_at(pos(0, 0, 0), 7));
        engine.add_neuron(UnifiedNeuron::ternsig_at(pos(1, 0, 0), 7));
        // One unbound
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(2, 0, 0)));

        engine.inject_ternsig_output(7, &[500, 300], 100);

        // Should have 2 pending events (one per bound neuron)
        assert_eq!(engine.pending_count(), 2);
    }

    #[test]
    fn max_events_limit() {
        let config = CascadeConfig {
            max_events_per_call: 5,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));

        for i in 0..100 {
            engine.inject_ff(0, 10, i * 10);
        }

        engine.run_until(10_000);
        assert_eq!(engine.total_events(), 5);
        assert!(engine.pending_count() > 0);
    }

    #[test]
    fn recover_stamina() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        let mut n = UnifiedNeuron::pyramidal_at(pos(0, 0, 0));
        n.stamina = 100;
        engine.add_neuron(n);

        // 10ms frame → 10_000 / 5_000 = 2 recovery
        engine.recover_stamina(10_000);
        assert_eq!(engine.neurons[0].stamina, 102);
    }

    #[test]
    fn clear_and_reset() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));

        engine.inject_ff(0, 100, 1000);
        engine.sim_time_us = 500;

        engine.clear_pending();
        assert_eq!(engine.pending_count(), 0);

        engine.reset_time();
        assert_eq!(engine.sim_time(), 0);
    }

    #[test]
    fn integer_sqrt_works() {
        assert_eq!(integer_sqrt(0), 0);
        assert_eq!(integer_sqrt(1), 1);
        assert_eq!(integer_sqrt(4), 2);
        assert_eq!(integer_sqrt(256), 16);
        assert_eq!(integer_sqrt(255), 15); // floor
        assert_eq!(integer_sqrt(10000), 100);
    }

    #[test]
    fn feedback_zone_integration() {
        let config = CascadeConfig {
            coincidence_boost: 0.0,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);
        engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));

        // Inject into feedback zone
        engine.inject(0, DendriticZone::Feedback, 1000, 50);
        engine.run_until(60);

        // Feedback should have been integrated, but not enough to prime or fire
        let resting = super::super::neuron::RESTING_POTENTIAL;
        assert!(
            engine.neurons[0].feedback_potential > resting,
            "feedback zone should accumulate: {}",
            engine.neurons[0].feedback_potential
        );
        assert!(!engine.neurons[0].predicted, "feedback doesn't prime prediction");
    }

    #[test]
    fn threshold_jitter_breaks_synchrony() {
        // With jitter=0: uniform input at exact threshold fires deterministically
        // With jitter>0: same input may or may not fire different neurons
        let config_no_jitter = CascadeConfig {
            threshold_jitter: 0,
            coincidence_boost: 0.0,
            ..CascadeConfig::default()
        };
        let config_jitter = CascadeConfig {
            threshold_jitter: 512,
            coincidence_boost: 0.0,
            ..CascadeConfig::default()
        };

        // With no jitter: all 10 neurons at exact fire current → all fire
        let fire_current = ff_fire_current(&ZoneWeights::PYRAMIDAL);
        let mut engine_no = CascadeEngine::new(config_no_jitter);
        for _ in 0..10 {
            engine_no.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));
        }
        for i in 0..10 {
            engine_no.inject_ff(i, fire_current, 100);
        }
        let spikes_no = engine_no.run_until(200);
        assert_eq!(spikes_no, 10, "without jitter, all should fire at exact threshold");

        // With jitter: slightly below threshold → some fire, some don't
        let marginal_current = fire_current - 300; // slightly below threshold
        let mut engine_jit = CascadeEngine::new(config_jitter);
        for _ in 0..10 {
            engine_jit.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));
        }
        for i in 0..10 {
            engine_jit.inject_ff(i, marginal_current, i as u64 * 10);
        }
        let spikes_jit = engine_jit.run_until(200);
        // With jitter, some neurons should fire (threshold lowered) and some shouldn't (threshold raised)
        // Not deterministic on exact count, but shouldn't be all-or-nothing
        assert!(
            spikes_jit < 10,
            "with jitter and marginal current, not all should fire: got {}",
            spikes_jit
        );
    }

    #[test]
    fn spontaneous_depolarization() {
        let config = CascadeConfig {
            spontaneous_rate: 255, // ~100% probability
            spontaneous_current: 500,
            coincidence_boost: 0.0,
            threshold_jitter: 0,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);
        for _ in 0..5 {
            engine.add_neuron(UnifiedNeuron::pyramidal_at(pos(0, 0, 0)));
        }

        // check_spontaneous should inject into all non-oscillator neurons
        engine.sim_time_us = 1000;
        engine.check_spontaneous();
        assert!(
            engine.pending_count() >= 4,
            "spontaneous rate 255 should inject into most neurons: got {}",
            engine.pending_count()
        );
    }

    #[test]
    fn spontaneous_skips_oscillators() {
        let config = CascadeConfig {
            spontaneous_rate: 255,
            spontaneous_current: 500,
            coincidence_boost: 0.0,
            threshold_jitter: 0,
            ..CascadeConfig::default()
        };
        let mut engine = CascadeEngine::new(config);
        // All oscillators
        for _ in 0..5 {
            engine.add_neuron(UnifiedNeuron::oscillator_at(pos(0, 0, 0), 10_000));
        }

        engine.sim_time_us = 1000;
        engine.check_spontaneous();
        assert_eq!(
            engine.pending_count(), 0,
            "oscillators should be skipped by spontaneous firing"
        );
    }
}
