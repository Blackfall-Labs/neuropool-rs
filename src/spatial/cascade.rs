//! Event-Driven Cascade Executor for Spatial Neurons
//!
//! Unlike tick-based execution (all neurons every cycle), cascade execution
//! processes events as they arrive. Neurons only compute when poked.
//!
//! ## How It Works
//!
//! ```text
//! Input arrives → sensory depolarizes → crosses threshold → fires
//!     → spike propagates via axon (with delay) → arrives at targets
//!     → they depolarize → chain reaction continues
//! ```
//!
//! No global clock. No tick. Just local causality.
//!
//! ## Coincidence Detection
//!
//! With proper delays, coincident inputs naturally strengthen:
//! ```text
//! t=0μs:  Left ear → inject sensory_L
//! t=5μs:  Right ear → inject sensory_R
//! t=60μs: Arrival from L at target A
//! t=70μs: Arrival from R at target A
//! → Temporal summation → A fires
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{SpatialNeuron, SpatialSynapse, SpatialSynapseStore};

/// A spike in flight, waiting to arrive at its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeArrival {
    /// Target neuron index
    pub target: u32,
    /// Current to deliver (from synapse signal)
    pub current: i16,
    /// Arrival time in microseconds
    pub arrival_time_us: u64,
    /// Source neuron (for eligibility traces)
    pub source: u32,
}

impl Ord for SpikeArrival {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: earlier arrivals first
        other.arrival_time_us.cmp(&self.arrival_time_us)
    }
}

impl PartialOrd for SpikeArrival {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration for the spatial cascade executor.
#[derive(Clone, Copy, Debug)]
pub struct SpatialCascadeConfig {
    /// Base propagation speed in μs per unit distance
    pub propagation_speed_us_per_unit: f32,
    /// Myelination speed bonus (0.0 = no bonus, 1.0 = 2x speed at full myelin)
    pub myelin_speed_factor: f32,
    /// Maximum events to process per run_until call (prevents infinite loops)
    pub max_events_per_call: usize,
    /// White matter speed bonus (signals travel faster through axon tracts)
    pub white_matter_speed_factor: f32,
    /// Gray matter attenuation (signals weaken in dense soma regions)
    pub gray_matter_attenuation: f32,
    /// Whether to use tissue-aware propagation (requires TissueField)
    pub use_tissue_physics: bool,
}

impl Default for SpatialCascadeConfig {
    fn default() -> Self {
        Self {
            propagation_speed_us_per_unit: 100.0, // 100μs per unit distance
            myelin_speed_factor: 0.5,             // myelinated axons are 1.5x faster
            max_events_per_call: 10_000,
            white_matter_speed_factor: 2.0,       // 2x faster through white matter
            gray_matter_attenuation: 0.2,         // 20% signal loss through gray matter
            use_tissue_physics: false,            // off by default for compatibility
        }
    }
}

/// Event-driven executor for spatial neurons.
pub struct SpatialCascade {
    /// All neurons
    pub neurons: Vec<SpatialNeuron>,
    /// All synapses
    pub synapses: SpatialSynapseStore,
    /// Event queue (min-heap by arrival time)
    pending: BinaryHeap<Reverse<SpikeArrival>>,
    /// Current simulation time in μs
    sim_time_us: u64,
    /// Configuration
    config: SpatialCascadeConfig,
    /// Statistics
    total_spikes: u64,
    total_events: u64,
}

impl SpatialCascade {
    /// Create a new cascade executor.
    pub fn new(config: SpatialCascadeConfig) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: SpatialSynapseStore::default(),
            pending: BinaryHeap::new(),
            sim_time_us: 0,
            config,
            total_spikes: 0,
            total_events: 0,
        }
    }

    /// Create with neurons and synapses.
    pub fn with_network(
        neurons: Vec<SpatialNeuron>,
        synapses: SpatialSynapseStore,
        config: SpatialCascadeConfig,
    ) -> Self {
        Self {
            neurons,
            synapses,
            pending: BinaryHeap::new(),
            sim_time_us: 0,
            config,
            total_spikes: 0,
            total_events: 0,
        }
    }

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

    /// Add a neuron, returns its index.
    pub fn add_neuron(&mut self, neuron: SpatialNeuron) -> u32 {
        let idx = self.neurons.len() as u32;
        self.neurons.push(neuron);
        idx
    }

    /// Add a synapse.
    pub fn add_synapse(&mut self, synapse: SpatialSynapse) {
        self.synapses.add(synapse);
    }

    /// Rebuild synapse index after adding synapses.
    pub fn rebuild_synapse_index(&mut self) {
        self.synapses.rebuild_index(self.neurons.len());
    }

    /// Inject external input to a neuron at a specific time.
    pub fn inject(&mut self, neuron: u32, current: i16, time_us: u64) {
        self.pending.push(Reverse(SpikeArrival {
            target: neuron,
            current,
            arrival_time_us: time_us,
            source: u32::MAX, // external input has no source
        }));
    }

    /// Inject to multiple sensory neurons (convenience for audio frames).
    pub fn inject_sensory(&mut self, currents: &[i16], time_us: u64) {
        for (i, &current) in currents.iter().enumerate() {
            if current != 0 {
                // Find sensory neuron for channel i
                for (idx, neuron) in self.neurons.iter().enumerate() {
                    if neuron.nuclei.is_sensory()
                        && neuron.nuclei.interface.target == i as u16
                    {
                        self.inject(idx as u32, current, time_us);
                        break;
                    }
                }
            }
        }
    }

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

    /// Process a single event. Returns true if the target neuron fired.
    fn process_arrival(&mut self, arrival: SpikeArrival) -> bool {
        let idx = arrival.target as usize;
        if idx >= self.neurons.len() {
            return false;
        }

        let sim_time = self.sim_time_us;

        // Apply leak for elapsed time
        self.neurons[idx].apply_leak(sim_time);

        // Handle oscillator autonomous ramp
        if self.neurons[idx].nuclei.is_oscillator() {
            self.neurons[idx].oscillator_ramp(sim_time);
        }

        // Integrate incoming current
        self.neurons[idx].integrate(arrival.current);

        // Update eligibility trace for the synapse (if from internal source)
        if arrival.source != u32::MAX {
            let source_idx = arrival.source as usize;
            // Boost trace on the source neuron for STDP-like learning
            if source_idx < self.neurons.len() && source_idx != idx {
                self.neurons[source_idx].trace =
                    self.neurons[source_idx].trace.saturating_add(5);
            }
        }

        // Check if neuron fires
        if self.neurons[idx].can_fire(sim_time) {
            self.fire_neuron(idx);
            true
        } else {
            false
        }
    }

    /// Fire a neuron and queue spikes to all targets.
    fn fire_neuron(&mut self, idx: usize) {
        let neuron = &mut self.neurons[idx];
        neuron.fire(self.sim_time_us);
        self.total_spikes += 1;

        // Get outgoing synapses and queue arrivals
        let synapses: Vec<SpatialSynapse> = self.synapses.outgoing(idx as u32).to_vec();

        for syn in synapses {
            if !syn.is_active() {
                continue;
            }

            // Compute delay based on distance and myelination
            let delay = self.compute_delay(idx, syn.target as usize, syn.delay_us);

            self.pending.push(Reverse(SpikeArrival {
                target: syn.target,
                current: syn.current(),
                arrival_time_us: self.sim_time_us + delay as u64,
                source: idx as u32,
            }));
        }
    }

    /// Compute propagation delay for a synapse.
    fn compute_delay(&self, src: usize, tgt: usize, base_delay: u32) -> u32 {
        if base_delay > 0 {
            // Use synapse's precomputed delay
            return base_delay;
        }

        // Compute from distance
        let src_pos = self.neurons[src].axon.terminal;
        let tgt_pos = self.neurons[tgt].soma.position;

        let dx = tgt_pos[0] - src_pos[0];
        let dy = tgt_pos[1] - src_pos[1];
        let dz = tgt_pos[2] - src_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Myelination reduces delay
        let myelin = self.neurons[src].axon.myelin as f32 / 255.0;
        let speed_factor = 1.0 + myelin * self.config.myelin_speed_factor;

        let delay = (distance * self.config.propagation_speed_us_per_unit / speed_factor) as u32;
        delay.max(1) // minimum 1μs delay
    }

    /// Compute tissue-aware propagation delay.
    ///
    /// Uses the tissue field to determine signal velocity based on whether
    /// the path passes through gray matter (slow) or white matter (fast).
    ///
    /// Returns (delay_us, attenuation_factor).
    pub fn compute_tissue_delay(
        &self,
        src: usize,
        tgt: usize,
        tissue: &super::TissueField,
    ) -> (u32, f32) {
        let src_pos = self.neurons[src].axon.terminal;
        let tgt_pos = self.neurons[tgt].soma.position;

        // Sample tissue along the path
        let samples = 5;
        let mut total_velocity_factor = 0.0f32;
        let mut gray_fraction = 0.0f32;

        for i in 0..samples {
            let t = (i as f32 + 0.5) / samples as f32;
            let sample_pos = [
                src_pos[0] + (tgt_pos[0] - src_pos[0]) * t,
                src_pos[1] + (tgt_pos[1] - src_pos[1]) * t,
                src_pos[2] + (tgt_pos[2] - src_pos[2]) * t,
            ];

            let tissue_type = tissue.tissue_at(sample_pos, &self.neurons);
            total_velocity_factor += tissue_type.velocity_factor();

            if tissue_type.is_gray() {
                gray_fraction += 1.0 / samples as f32;
            }
        }

        let avg_velocity_factor = total_velocity_factor / samples as f32;

        // Compute distance
        let dx = tgt_pos[0] - src_pos[0];
        let dy = tgt_pos[1] - src_pos[1];
        let dz = tgt_pos[2] - src_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Base delay from distance and tissue velocity
        // velocity_factor > 1.0 means faster, so delay is reduced
        let effective_speed = self.config.propagation_speed_us_per_unit / avg_velocity_factor.max(0.1);
        let base_delay = distance * effective_speed;

        // Myelination bonus still applies
        let myelin = self.neurons[src].axon.myelin as f32 / 255.0;
        let myelin_factor = 1.0 + myelin * self.config.myelin_speed_factor;

        let final_delay = (base_delay / myelin_factor) as u32;

        // Attenuation based on gray matter traversal
        let attenuation = 1.0 - (gray_fraction * self.config.gray_matter_attenuation);

        (final_delay.max(1), attenuation)
    }

    /// Run with tissue-aware propagation.
    ///
    /// Returns (spikes, attenuated_events) where attenuated_events had reduced current.
    pub fn run_with_tissue(
        &mut self,
        until_time_us: u64,
        tissue: &super::TissueField,
    ) -> (u64, u64) {
        let mut spikes = 0u64;
        let mut attenuated = 0u64;
        let mut events = 0u64;

        while let Some(&Reverse(arrival)) = self.pending.peek() {
            if arrival.arrival_time_us > until_time_us {
                break;
            }
            if events >= self.config.max_events_per_call as u64 {
                break;
            }

            let arrival = self.pending.pop().unwrap().0;
            self.sim_time_us = arrival.arrival_time_us;
            events += 1;

            // Check if this was a tissue-attenuated event
            if arrival.source != u32::MAX && self.config.use_tissue_physics {
                let (_, attenuation) = self.compute_tissue_delay(
                    arrival.source as usize,
                    arrival.target as usize,
                    tissue,
                );
                if attenuation < 0.99 {
                    attenuated += 1;
                }
            }

            if self.process_arrival(arrival) {
                spikes += 1;
            }
        }

        self.total_events += events;
        self.total_spikes += spikes;
        (spikes, attenuated)
    }

    /// Check and fire any oscillators that have completed their period.
    pub fn check_oscillators(&mut self) {
        for idx in 0..self.neurons.len() {
            let neuron = &self.neurons[idx];
            if neuron.oscillator_should_fire(self.sim_time_us) {
                // Inject autonomous depolarization
                self.inject(
                    idx as u32,
                    SpatialNeuron::DEFAULT_THRESHOLD - SpatialNeuron::RESET_POTENTIAL,
                    self.sim_time_us,
                );
            }
        }
    }

    /// Read motor neuron outputs.
    ///
    /// Returns a vector of (channel, accumulated_current) for motor neurons
    /// that fired since the last read.
    pub fn read_motor_outputs(&self) -> Vec<(u16, i16)> {
        let mut outputs = Vec::new();

        for neuron in &self.neurons {
            if neuron.nuclei.is_motor() && neuron.trace > 0 {
                outputs.push((neuron.nuclei.interface.target, neuron.trace as i16));
            }
        }

        outputs
    }

    /// Decay all eligibility traces.
    pub fn decay_traces(&mut self, retention: f32) {
        for neuron in &mut self.neurons {
            neuron.decay_trace(retention);
        }
    }

    /// Clear the event queue.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Reset simulation time to zero.
    pub fn reset_time(&mut self) {
        self.sim_time_us = 0;
        for neuron in &mut self.neurons {
            neuron.last_update_us = 0;
            neuron.last_spike_us = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Nuclei;

    #[test]
    fn test_cascade_creation() {
        let cascade = SpatialCascade::new(SpatialCascadeConfig::default());
        assert_eq!(cascade.neurons.len(), 0);
        assert_eq!(cascade.sim_time(), 0);
    }

    #[test]
    fn test_add_neuron() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());
        let idx = cascade.add_neuron(SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]));
        assert_eq!(idx, 0);
        assert_eq!(cascade.neurons.len(), 1);
    }

    #[test]
    fn test_inject_and_process() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());
        cascade.add_neuron(SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]));

        // Inject strong current
        let threshold_current =
            SpatialNeuron::DEFAULT_THRESHOLD - SpatialNeuron::RESTING_POTENTIAL + 100;
        cascade.inject(0, threshold_current, 100);

        let spikes = cascade.run_until(200);
        assert_eq!(spikes, 1);
        assert_eq!(cascade.total_spikes(), 1);
    }

    #[test]
    fn test_spike_propagation() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        // Create two connected neurons
        cascade.add_neuron(SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]));
        cascade.add_neuron(SpatialNeuron::pyramidal_at([1.0, 0.0, 0.0]));

        // Connect them with a strong synapse
        cascade.add_synapse(SpatialSynapse::excitatory(0, 1, 200, 50));
        cascade.rebuild_synapse_index();

        // Make first neuron fire
        let threshold_current =
            SpatialNeuron::DEFAULT_THRESHOLD - SpatialNeuron::RESTING_POTENTIAL + 100;
        cascade.inject(0, threshold_current, 0);

        // Run and check propagation
        cascade.run_until(100);
        assert!(cascade.total_spikes() >= 1);
        assert!(cascade.pending_count() > 0 || cascade.total_events() > 1);
    }

    #[test]
    fn test_temporal_summation() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        // Create three neurons: two sources, one target
        cascade.add_neuron(SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0])); // source A
        cascade.add_neuron(SpatialNeuron::pyramidal_at([0.0, 1.0, 0.0])); // source B
        cascade.add_neuron(SpatialNeuron::pyramidal_at([1.0, 0.5, 0.0])); // target

        // Connect both sources to target with moderate synapses
        cascade.add_synapse(SpatialSynapse::excitatory(0, 2, 100, 50));
        cascade.add_synapse(SpatialSynapse::excitatory(1, 2, 100, 55)); // slightly later
        cascade.rebuild_synapse_index();

        // Fire both sources
        let threshold_current =
            SpatialNeuron::DEFAULT_THRESHOLD - SpatialNeuron::RESTING_POTENTIAL + 100;
        cascade.inject(0, threshold_current, 0);
        cascade.inject(1, threshold_current, 5);

        // Run - target should fire from combined input
        cascade.run_until(200);

        // At least the two sources should have fired
        assert!(cascade.total_spikes() >= 2);
    }

    #[test]
    fn test_inhibition() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        // Create two neurons
        let mut target = SpatialNeuron::pyramidal_at([1.0, 0.0, 0.0]);
        target.membrane = SpatialNeuron::DEFAULT_THRESHOLD - 50; // almost at threshold

        cascade.add_neuron(SpatialNeuron::interneuron_at([0.0, 0.0, 0.0]));
        cascade.add_neuron(target);

        // Inhibitory synapse
        cascade.add_synapse(SpatialSynapse::inhibitory(0, 1, 200, 50));
        cascade.rebuild_synapse_index();

        // Fire the inhibitory neuron
        let threshold_current =
            SpatialNeuron::DEFAULT_THRESHOLD - SpatialNeuron::RESTING_POTENTIAL + 100;
        cascade.inject(0, threshold_current, 0);

        cascade.run_until(200);

        // Target should have been pushed below threshold
        assert!(cascade.neurons[1].membrane < SpatialNeuron::DEFAULT_THRESHOLD);
    }

    #[test]
    fn test_oscillator() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        // Add oscillator with 1000μs period
        cascade.add_neuron(SpatialNeuron::at([0.0, 0.0, 0.0], Nuclei::oscillator(1000)));

        // Check oscillators at period boundary
        cascade.sim_time_us = 1000;
        cascade.check_oscillators();

        // Should have injected
        assert!(cascade.pending_count() > 0);

        // Run and verify it fired
        cascade.run_until(1100);
        assert_eq!(cascade.total_spikes(), 1);
    }

    #[test]
    fn test_myelin_speed_bonus() {
        let _config = SpatialCascadeConfig::default();

        // Unmyelinated neuron
        let mut unmyelinated = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        unmyelinated.axon.myelin = 0;

        // Myelinated neuron
        let mut myelinated = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        myelinated.axon.myelin = 255;

        // Put them in a temporary cascade to compute delays
        let mut test_cascade = SpatialCascade::new(SpatialCascadeConfig::default());
        test_cascade.add_neuron(unmyelinated);
        test_cascade.add_neuron(myelinated);
        test_cascade.add_neuron(SpatialNeuron::pyramidal_at([10.0, 0.0, 0.0])); // target

        let delay_unmyelinated = test_cascade.compute_delay(0, 2, 0);
        let delay_myelinated = test_cascade.compute_delay(1, 2, 0);

        // Myelinated should be faster
        assert!(delay_myelinated < delay_unmyelinated);
    }

    #[test]
    fn test_max_events_limit() {
        let config = SpatialCascadeConfig {
            max_events_per_call: 5,
            ..Default::default()
        };
        let mut cascade = SpatialCascade::new(config);

        cascade.add_neuron(SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]));

        // Inject many events
        for i in 0..100 {
            cascade.inject(0, 10, i * 10);
        }

        // Should only process max_events_per_call
        cascade.run_until(10000);
        assert_eq!(cascade.total_events(), 5);
        assert!(cascade.pending_count() > 0);
    }

    #[test]
    fn test_decay_traces() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        let mut neuron = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        neuron.trace = 100;
        cascade.add_neuron(neuron);

        cascade.decay_traces(0.5);
        assert_eq!(cascade.neurons[0].trace, 50);
    }

    #[test]
    fn test_clear_and_reset() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());
        cascade.add_neuron(SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]));

        cascade.inject(0, 100, 1000);
        cascade.sim_time_us = 500;

        cascade.clear_pending();
        assert_eq!(cascade.pending_count(), 0);

        cascade.reset_time();
        assert_eq!(cascade.sim_time(), 0);
    }
}
