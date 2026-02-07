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
    /// How much low conductivity slows signals (0 = tissue ignored, 1.0 = C=0 doubles delay)
    pub tissue_delay_factor: f32,
    /// Resistance-distance attenuation coefficient (higher = more signal loss)
    pub tissue_attenuation_factor: f32,
    /// Number of samples along axon→soma path for tissue field queries
    pub tissue_path_samples: usize,
    /// Fast coincidence window (μs). Arrivals within this window amplify each other.
    /// Models AMPA-like temporal sensitivity — makes μs-scale timing matter.
    pub fast_coincidence_window_us: u64,
    /// Peak coincidence boost (0.0 = disabled, 0.5 = up to 1.5x for simultaneous arrivals)
    pub coincidence_boost: f32,
}

impl Default for SpatialCascadeConfig {
    fn default() -> Self {
        Self {
            propagation_speed_us_per_unit: 100.0, // 100μs per unit distance
            myelin_speed_factor: 0.5,             // myelinated axons are 1.5x faster
            max_events_per_call: 10_000,
            tissue_delay_factor: 1.0,             // low-C tissue doubles delay at C=0
            tissue_attenuation_factor: 0.2,       // real attenuation: 60% loss through stiff tissue over 15 units
            tissue_path_samples: 3,               // 3-point path sampling
            fast_coincidence_window_us: 1500,     // 1.5ms fast window (AMPA-like)
            coincidence_boost: 0.5,               // up to 1.5x for near-simultaneous arrivals
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
    /// Diagnostic: how many times coincidence boost was applied
    pub coincidence_events: u64,
    /// Diagnostic: how many spikes went through tissue attenuation
    pub tissue_attenuated: u64,
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
            coincidence_events: 0,
            tissue_attenuated: 0,
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
            coincidence_events: 0,
            tissue_attenuated: 0,
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

    /// Inject scaled sensory input with optional neighborhood activation.
    ///
    /// For each coefficient at index `i`:
    /// - Skip if `|coeff| < silence_threshold`
    /// - Inject `coeff * scale` as current to sensory neuron for channel `i`
    /// - If `|coeff| > neighbor_threshold`: inject `coeff * neighbor_scale`
    ///   to channels `i-1` and `i+1` (coincidence detection)
    ///
    /// This moves the MFCC injection logic from test code into the runtime.
    pub fn inject_sensory_scaled(
        &mut self,
        coefficients: &[f32],
        scale: f32,
        neighbor_scale: f32,
        neighbor_threshold: f32,
        silence_threshold: f32,
        time_us: u64,
    ) {
        for (i, &coeff) in coefficients.iter().enumerate() {
            if coeff.abs() < silence_threshold {
                continue;
            }

            let current = (coeff * scale) as i16;
            self.inject(i as u32, current, time_us);

            if coeff.abs() > neighbor_threshold {
                let neighbor_current = (coeff * neighbor_scale) as i16;
                if i > 0 {
                    self.inject((i - 1) as u32, neighbor_current, time_us);
                }
                if i < coefficients.len() - 1 {
                    self.inject((i + 1) as u32, neighbor_current, time_us);
                }
            }
        }
    }

    /// Process all events up to the given time.
    ///
    /// Returns number of spikes that occurred.
    pub fn run_until(&mut self, until_time_us: u64) -> u64 {
        self.run_cascade(until_time_us, None)
    }

    /// Process all events with tissue-aware propagation.
    ///
    /// Tissue conductivity modulates spike delay (low C → slower signals).
    /// Tissue resistance attenuates spike current (high R × distance → weaker signals).
    /// This closes the causal loop: tissue shapes timing → timing shapes correlations
    /// → correlations shape migration → migration shapes tissue.
    pub fn run_until_with_tissue(
        &mut self,
        until_time_us: u64,
        tissue: &super::TissueField,
    ) -> u64 {
        self.run_cascade(until_time_us, Some(tissue))
    }

    /// Core cascade loop, optionally tissue-aware.
    fn run_cascade(
        &mut self,
        until_time_us: u64,
        tissue: Option<&super::TissueField>,
    ) -> u64 {
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

            if self.process_arrival(arrival, tissue) {
                spikes_this_run += 1;
            }
        }

        self.total_events += events_this_run;
        spikes_this_run
    }

    /// Process a single event. Returns true if the target neuron fired.
    fn process_arrival(
        &mut self,
        arrival: SpikeArrival,
        tissue: Option<&super::TissueField>,
    ) -> bool {
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

        // Fast coincidence: amplify current when arrivals are temporally coincident.
        // If another spike arrived at this neuron within the fast window, the new
        // arrival gets boosted — models AMPA-like temporal sensitivity so that
        // μs-scale delay differences from tissue actually change who fires.
        let current = if self.config.coincidence_boost > 0.0
            && self.neurons[idx].last_arrival_us > 0
        {
            let dt = sim_time.saturating_sub(self.neurons[idx].last_arrival_us);
            if dt > 0 && dt < self.config.fast_coincidence_window_us {
                self.coincidence_events += 1;
                let closeness = 1.0 - (dt as f32 / self.config.fast_coincidence_window_us as f32);
                (arrival.current as f32 * (1.0 + closeness * self.config.coincidence_boost)) as i16
            } else {
                arrival.current
            }
        } else {
            arrival.current
        };
        self.neurons[idx].integrate(current);
        self.neurons[idx].last_arrival_us = sim_time;

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
            self.fire_neuron(idx, tissue);
            true
        } else {
            false
        }
    }

    /// Fire a neuron and queue spikes to all targets.
    ///
    /// When tissue is provided, each outgoing spike is:
    /// - Delayed by tissue conductivity along the path (low C → slower arrival)
    /// - Attenuated by tissue resistance along the path (high R × distance → weaker current)
    fn fire_neuron(&mut self, idx: usize, tissue: Option<&super::TissueField>) {
        let neuron = &mut self.neurons[idx];
        neuron.fire(self.sim_time_us);
        self.total_spikes += 1;

        // Get outgoing synapses and queue arrivals
        let synapses: Vec<SpatialSynapse> = self.synapses.outgoing(idx as u32).to_vec();

        for syn in synapses {
            if !syn.is_active() {
                continue;
            }

            let tgt = syn.target as usize;

            if syn.delay_us > 0 {
                // Precomputed delay — bypass tissue physics
                self.pending.push(Reverse(SpikeArrival {
                    target: syn.target,
                    current: syn.current(),
                    arrival_time_us: self.sim_time_us + syn.delay_us as u64,
                    source: idx as u32,
                }));
                continue;
            }

            // Compute from distance
            let src_pos = self.neurons[idx].axon.terminal;
            let tgt_pos = self.neurons[tgt].soma.position;
            let dx = tgt_pos[0] - src_pos[0];
            let dy = tgt_pos[1] - src_pos[1];
            let dz = tgt_pos[2] - src_pos[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // Myelination reduces delay
            let myelin = self.neurons[idx].axon.myelin as f32 / 255.0;
            let speed_factor = 1.0 + myelin * self.config.myelin_speed_factor;
            let mut delay = distance * self.config.propagation_speed_us_per_unit / speed_factor;
            let mut current = syn.current();

            // Tissue physics: conductivity modulates delay, resistance attenuates current
            if let Some(tissue) = tissue {
                self.tissue_attenuated += 1;
                let (avg_c, avg_r) = self.sample_tissue_path(src_pos, tgt_pos, tissue);

                // Low conductivity → slower propagation
                // delay *= (1 + k * (1 - C))  so C=1 → unchanged, C=0 → doubles
                delay *= 1.0 + self.config.tissue_delay_factor * (1.0 - avg_c);

                // High resistance × distance → weaker signal
                // attenuation = 1 / (1 + k * R * distance)
                let attenuation = 1.0
                    / (1.0 + self.config.tissue_attenuation_factor * avg_r * distance);
                current = (current as f32 * attenuation) as i16;
            }

            self.pending.push(Reverse(SpikeArrival {
                target: syn.target,
                current,
                arrival_time_us: self.sim_time_us + (delay as u32).max(1) as u64,
                source: idx as u32,
            }));
        }
    }

    /// Sample average conductivity and resistance along the axon→soma path.
    ///
    /// Takes N evenly-spaced samples between source axon terminal and target soma,
    /// querying the tissue field's kernel-interpolated continuous values at each point.
    fn sample_tissue_path(
        &self,
        src_pos: [f32; 3],
        tgt_pos: [f32; 3],
        tissue: &super::TissueField,
    ) -> (f32, f32) {
        let samples = self.config.tissue_path_samples;
        let mut total_c = 0.0f32;
        let mut total_r = 0.0f32;

        for i in 0..samples {
            let t = (i as f32 + 0.5) / samples as f32;
            let pos = [
                src_pos[0] + (tgt_pos[0] - src_pos[0]) * t,
                src_pos[1] + (tgt_pos[1] - src_pos[1]) * t,
                src_pos[2] + (tgt_pos[2] - src_pos[2]) * t,
            ];
            total_c += tissue.conductivity_at(pos, &self.neurons);
            total_r += tissue.resistance_at(pos, &self.neurons);
        }

        (total_c / samples as f32, total_r / samples as f32)
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

    /// Recover stamina for all neurons based on frame duration.
    ///
    /// Called once per frame from the runtime. Recovery is proportional
    /// to elapsed time: 1 stamina point per STAMINA_RECOVERY_US microseconds.
    /// This is intentionally NOT in apply_leak() — event-driven recovery
    /// let depleted neurons stutter-fire via arrival-triggered leak.
    pub fn recover_stamina(&mut self, frame_interval_us: u64) {
        let recovery = (frame_interval_us / SpatialNeuron::STAMINA_RECOVERY_US) as u8;
        if recovery == 0 {
            return;
        }
        for neuron in &mut self.neurons {
            neuron.stamina = neuron.stamina.saturating_add(recovery);
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
            neuron.last_arrival_us = 0;
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
        // Myelinated neuron should propagate spikes faster than unmyelinated.
        // Both source→target pairs have same distance (10 units).
        // The myelinated target should receive its spike first.
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        // Source unmyelinated at origin (axon terminal at soma)
        let mut unmyelinated = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        unmyelinated.axon = crate::spatial::Axon::toward([0.0, 0.0, 0.0]);
        unmyelinated.axon.myelin = 0;
        cascade.add_neuron(unmyelinated); // idx 0

        // Source myelinated (axon terminal at soma)
        let mut myelinated = SpatialNeuron::pyramidal_at([0.0, 2.0, 0.0]);
        myelinated.axon = crate::spatial::Axon::toward([0.0, 2.0, 0.0]);
        myelinated.axon.myelin = 255;
        cascade.add_neuron(myelinated); // idx 1

        // Targets at distance 10 from each source
        cascade.add_neuron(SpatialNeuron::pyramidal_at([10.0, 0.0, 0.0])); // idx 2
        cascade.add_neuron(SpatialNeuron::pyramidal_at([10.0, 2.0, 0.0])); // idx 3

        cascade.add_synapse(SpatialSynapse::excitatory(0, 2, 200, 0));
        cascade.add_synapse(SpatialSynapse::excitatory(1, 3, 200, 0));
        cascade.rebuild_synapse_index();

        // Fire both sources
        let fire_current =
            SpatialNeuron::DEFAULT_THRESHOLD - SpatialNeuron::RESTING_POTENTIAL + 100;
        cascade.inject(0, fire_current, 0);
        cascade.inject(1, fire_current, 0);

        // Run the full cascade — both targets will receive spikes
        cascade.run_until(2000);

        // Both targets should have been depolarized
        let target_unmyel = &cascade.neurons[2];
        let target_myel = &cascade.neurons[3];

        // The myelinated target should have received its spike earlier.
        // last_arrival_us records when the spike arrived.
        assert!(target_myel.last_arrival_us > 0, "myelinated target should receive spike");
        assert!(target_unmyel.last_arrival_us > 0, "unmyelinated target should receive spike");
        assert!(
            target_myel.last_arrival_us < target_unmyel.last_arrival_us,
            "myelinated spike (arrived {}μs) should arrive before unmyelinated (arrived {}μs)",
            target_myel.last_arrival_us,
            target_unmyel.last_arrival_us,
        );
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

    #[test]
    fn test_recover_stamina() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        let mut n = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        n.stamina = 100;
        cascade.add_neuron(n);

        // 10ms frame → 10_000 / 5_000 = 2 recovery
        cascade.recover_stamina(10_000);
        assert_eq!(cascade.neurons[0].stamina, 102);

        // Saturates at 255
        cascade.neurons[0].stamina = 254;
        cascade.recover_stamina(10_000);
        assert_eq!(cascade.neurons[0].stamina, 255);
    }

    #[test]
    fn test_recover_stamina_short_frame() {
        let mut cascade = SpatialCascade::new(SpatialCascadeConfig::default());

        let mut n = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        n.stamina = 100;
        cascade.add_neuron(n);

        // Frame shorter than recovery period → no recovery
        cascade.recover_stamina(4_000);
        assert_eq!(cascade.neurons[0].stamina, 100);
    }
}
