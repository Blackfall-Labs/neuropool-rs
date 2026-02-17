#![allow(deprecated)]
//! Event-driven cascade execution model.
//!
//! Unlike the synchronous `tick()` model where all neurons update every cycle,
//! the cascade model processes spike arrivals as events. Only neurons that
//! receive input do work. This is biologically accurate — neurons have no
//! global clock, only local membrane state and incoming currents.
//!
//! ## Key differences from NeuronPool
//!
//! | Aspect | NeuronPool | CascadePool |
//! |--------|------------|-------------|
//! | Execution | `tick()` — all neurons | `run_until()` — only active |
//! | Time | Discrete ticks | Continuous μs |
//! | Delay | Per-synapse fixed | Computed from distance + tissue |
//! | Leak | Per-tick shift | Delta-time exponential |

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::density::DensityField;
use crate::neuron::NeuronArrays;
use crate::synapse::SynapseStore;

/// A spike in flight, waiting to arrive at its target neuron.
#[derive(Clone, Copy, Debug)]
pub struct SpikeArrival {
    /// Target neuron index.
    pub target: u32,
    /// Current to inject on arrival (signed, includes weight scaling).
    pub current: i16,
    /// Arrival time in microseconds since simulation start.
    pub arrival_time: u64,
    /// Source neuron (for STDP tracking).
    pub source: u32,
}

impl PartialEq for SpikeArrival {
    fn eq(&self, other: &Self) -> bool {
        self.arrival_time == other.arrival_time
    }
}

impl Eq for SpikeArrival {}

impl PartialOrd for SpikeArrival {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpikeArrival {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (earlier times first)
        other.arrival_time.cmp(&self.arrival_time)
    }
}

/// Configuration for cascade execution.
#[derive(Clone, Debug)]
pub struct CascadeConfig {
    /// Resting membrane potential (Q8.8). Default: -70 * 256 = -17920
    pub resting_potential: i16,
    /// Reset potential after spike (Q8.8). Default: -65 * 256 = -16640
    pub reset_potential: i16,
    /// Spike threshold (Q8.8). Default: -55 * 256 = -14080
    pub spike_threshold: i16,
    /// Propagation speed in μs per unit distance. Default: 10.0
    pub propagation_speed: f32,
    /// Leak time constant in μs. Membrane decays to 1/e in this time. Default: 20000 (20ms)
    pub leak_tau: u64,
    /// Refractory period in μs. Default: 2000 (2ms)
    pub refractory_us: u64,
    /// Weight scaling factor (synapse weight × this = current). Default: 64
    pub weight_scale: i16,
    /// Gray matter density threshold for tissue classification. Default: 0.5
    pub gray_threshold: f32,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            resting_potential: -17920, // -70mV in Q8.8
            reset_potential: -16640,   // -65mV in Q8.8
            spike_threshold: -14080,   // -55mV in Q8.8
            propagation_speed: 10.0,   // 10μs per unit distance
            leak_tau: 20_000,          // 20ms time constant
            refractory_us: 2_000,      // 2ms refractory
            weight_scale: 64,
            gray_threshold: 0.5,
        }
    }
}

/// Event-driven neuron pool — no global tick.
///
/// Neurons only do work when they receive spike arrivals. Membrane potential
/// decays based on elapsed time since last update (delta-time leak). Axon
/// delays are computed from spatial distance and tissue conductivity.
pub struct CascadePool {
    /// Pool name for identification.
    pub name: String,
    /// Neuron state arrays (SoA layout).
    pub neurons: NeuronArrays,
    /// Synapse storage (CSR format).
    pub synapses: SynapseStore,
    /// Density field for tissue classification.
    pub density: DensityField,

    /// Event queue — min-heap sorted by arrival_time.
    pending: BinaryHeap<SpikeArrival>,

    /// Per-neuron: when was membrane last updated (μs).
    last_update: Vec<u64>,
    /// Per-neuron: when refractory period ends (μs). 0 = not refractory.
    refractory_until: Vec<u64>,

    /// Current simulation time (μs).
    pub sim_time: u64,
    /// Total spikes fired (for metrics).
    pub total_spikes: u64,
    /// Total events processed (for metrics).
    pub total_events: u64,

    /// Configuration.
    pub config: CascadeConfig,
}

impl CascadePool {
    /// Create a new cascade pool with given neurons, synapses, and bounds.
    pub fn new(
        name: impl Into<String>,
        neurons: NeuronArrays,
        synapses: SynapseStore,
        bounds: [f32; 3],
        config: CascadeConfig,
    ) -> Self {
        let n = neurons.len();
        let density = DensityField::new([8, 8, 8], bounds);

        Self {
            name: name.into(),
            neurons,
            synapses,
            density,
            pending: BinaryHeap::new(),
            last_update: vec![0; n],
            refractory_until: vec![0; n],
            sim_time: 0,
            total_spikes: 0,
            total_events: 0,
            config,
        }
    }

    /// Create from an existing NeuronPool (migration path).
    pub fn from_pool(pool: crate::pool::NeuronPool, config: CascadeConfig) -> Self {
        let bounds = pool.spatial_bounds.unwrap_or([10.0, 10.0, 10.0]);
        Self::new(pool.name, pool.neurons, pool.synapses, bounds, config)
    }

    /// Number of neurons.
    #[inline]
    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Number of pending events.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Inject external stimulus at specific time.
    ///
    /// Use this for sensory input. The current will be delivered to the
    /// target neuron at the specified time.
    pub fn inject(&mut self, neuron: u32, current: i16, time: u64) {
        self.pending.push(SpikeArrival {
            target: neuron,
            current,
            arrival_time: time,
            source: u32::MAX, // External source marker
        });
    }

    /// Process all events up to (and including) the given time.
    ///
    /// Returns the number of events processed.
    pub fn run_until(&mut self, until_time: u64) -> usize {
        let mut processed = 0;

        while let Some(arrival) = self.pending.peek() {
            if arrival.arrival_time > until_time {
                break;
            }
            let arrival = self.pending.pop().unwrap();
            self.sim_time = arrival.arrival_time;
            self.process_arrival(arrival);
            processed += 1;
        }

        self.total_events += processed as u64;
        processed
    }

    /// Process a single event (for debugging/tracing).
    ///
    /// Returns the processed arrival, or None if queue is empty.
    pub fn step(&mut self) -> Option<SpikeArrival> {
        let arrival = self.pending.pop()?;
        self.sim_time = arrival.arrival_time;
        self.process_arrival(arrival);
        self.total_events += 1;
        Some(arrival)
    }

    /// Process a spike arrival at a neuron.
    fn process_arrival(&mut self, arrival: SpikeArrival) {
        let i = arrival.target as usize;
        if i >= self.neurons.len() {
            return;
        }

        // Check refractory period
        if self.refractory_until[i] > self.sim_time {
            return; // Neuron is refractory, ignore input
        }

        // Apply leak for elapsed time (delta-time)
        let dt = self.sim_time.saturating_sub(self.last_update[i]);
        self.apply_leak(i, dt);
        self.last_update[i] = self.sim_time;

        // Integrate incoming current
        self.neurons.membrane[i] = self.neurons.membrane[i].saturating_add(arrival.current);

        // Check threshold
        if self.neurons.membrane[i] >= self.config.spike_threshold {
            self.fire(i);
        }
    }

    /// Apply exponential leak for elapsed time.
    ///
    /// Uses integer approximation: membrane moves toward resting by
    /// (membrane - resting) * dt / tau per step.
    fn apply_leak(&mut self, neuron: usize, dt: u64) {
        if dt == 0 || self.config.leak_tau == 0 {
            return;
        }

        let membrane = self.neurons.membrane[neuron] as i32;
        let resting = self.config.resting_potential as i32;
        let diff = membrane - resting;

        // Exponential decay approximation: new = old - (old - rest) * (1 - e^(-dt/tau))
        // For small dt/tau: 1 - e^(-x) ≈ x, so decay ≈ diff * dt / tau
        // Cap dt at tau to prevent over-decay
        let effective_dt = dt.min(self.config.leak_tau);
        let decay = (diff * effective_dt as i32) / self.config.leak_tau as i32;

        self.neurons.membrane[neuron] = (membrane - decay).clamp(-32768, 32767) as i16;
    }

    /// Fire a neuron: reset membrane, enter refractory, queue spikes to targets.
    fn fire(&mut self, neuron: usize) {
        // Reset membrane
        self.neurons.membrane[neuron] = self.config.reset_potential;

        // Enter refractory period
        self.refractory_until[neuron] = self.sim_time + self.config.refractory_us;

        // Increment trace (for STDP)
        self.neurons.trace[neuron] = self.neurons.trace[neuron].saturating_add(30);

        // Queue spikes to all synaptic targets
        for syn in self.synapses.outgoing(neuron as u32) {
            let tgt = syn.target as usize;
            if tgt >= self.neurons.len() {
                continue;
            }

            // Compute delay from spatial distance and tissue
            let delay = self.axon_delay(neuron, tgt);

            // Compute attenuated current
            let base_current = syn.weight as i16 * self.config.weight_scale;
            let current = self.attenuate_current(base_current, neuron, tgt);

            self.pending.push(SpikeArrival {
                target: syn.target as u32,
                current,
                arrival_time: self.sim_time + delay,
                source: neuron as u32,
            });
        }

        self.total_spikes += 1;
    }

    /// Compute axon delay from spatial positions and tissue conductivity.
    ///
    /// Delay = distance × propagation_speed × tissue_delay_factor
    pub fn axon_delay(&self, src: usize, tgt: usize) -> u64 {
        let src_pos = self.neurons.axon_terminal[src];
        let tgt_pos = self.neurons.soma_position[tgt];

        // Euclidean distance
        let dx = tgt_pos[0] - src_pos[0];
        let dy = tgt_pos[1] - src_pos[1];
        let dz = tgt_pos[2] - src_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Sample conductivity at midpoint
        let mid = [
            (src_pos[0] + tgt_pos[0]) / 2.0,
            (src_pos[1] + tgt_pos[1]) / 2.0,
            (src_pos[2] + tgt_pos[2]) / 2.0,
        ];
        let cond = self.density.conductivity_at(mid);

        // Delay = distance × speed × tissue factor
        // White matter: delay_factor ≈ 1.0 (fast)
        // Gray matter: delay_factor ≈ 3.0 (slow)
        let delay = distance * self.config.propagation_speed * cond.delay_factor;

        // Minimum 1μs delay to ensure causality
        (delay as u64).max(1)
    }

    /// Attenuate current based on tissue resistance along path.
    ///
    /// White matter: low attenuation (long-range preserved)
    /// Gray matter: high attenuation (local only)
    pub fn attenuate_current(&self, current: i16, src: usize, tgt: usize) -> i16 {
        let src_pos = self.neurons.axon_terminal[src];
        let tgt_pos = self.neurons.soma_position[tgt];

        let mid = [
            (src_pos[0] + tgt_pos[0]) / 2.0,
            (src_pos[1] + tgt_pos[1]) / 2.0,
            (src_pos[2] + tgt_pos[2]) / 2.0,
        ];
        let cond = self.density.conductivity_at(mid);

        // Scale by (1 - attenuation)
        let scale = 1.0 - cond.attenuation;
        ((current as f32) * scale) as i16
    }

    /// Update density field from current neuron positions.
    pub fn update_density(&mut self) {
        self.density.update_from_positions(&self.neurons.soma_position);
    }

    /// Migrate a neuron toward correlated partners, away from competitors.
    ///
    /// Call periodically during learning/development phase.
    pub fn migrate(&mut self, neuron: usize, rate: f32) {
        if neuron >= self.neurons.len() {
            return;
        }

        // Find neurons we're connected to (outgoing synapses)
        let outgoing = self.synapses.outgoing(neuron as u32);
        if outgoing.is_empty() {
            return;
        }

        // Compute center of mass of connected targets (attraction)
        let mut attraction = [0.0f32; 3];
        let mut count = 0.0;

        for syn in outgoing {
            let tgt = syn.target as usize;
            if tgt < self.neurons.len() {
                attraction[0] += self.neurons.soma_position[tgt][0];
                attraction[1] += self.neurons.soma_position[tgt][1];
                attraction[2] += self.neurons.soma_position[tgt][2];
                count += 1.0;
            }
        }

        if count > 0.0 {
            attraction[0] /= count;
            attraction[1] /= count;
            attraction[2] /= count;

            // Direction toward attraction
            let pos = self.neurons.soma_position[neuron];
            let dx = (attraction[0] - pos[0]) * rate;
            let dy = (attraction[1] - pos[1]) * rate;
            let dz = (attraction[2] - pos[2]) * rate;

            // Apply migration
            self.neurons.soma_position[neuron][0] += dx;
            self.neurons.soma_position[neuron][1] += dy;
            self.neurons.soma_position[neuron][2] += dz;

            // Axon terminal follows (elastic)
            self.neurons.axon_terminal[neuron][0] += dx * 0.5;
            self.neurons.axon_terminal[neuron][1] += dy * 0.5;
            self.neurons.axon_terminal[neuron][2] += dz * 0.5;
        }
    }

    /// Reset simulation state (time, events) but keep neurons and synapses.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.last_update.fill(0);
        self.refractory_until.fill(0);
        self.sim_time = 0;
        self.total_spikes = 0;
        self.total_events = 0;

        // Reset membrane to resting
        for m in &mut self.neurons.membrane {
            *m = self.config.resting_potential;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::NeuronArrays;
    use crate::synapse::{Synapse, SynapseStore};

    fn make_test_pool() -> CascadePool {
        // Create 3 neurons
        let neurons = NeuronArrays::new(3, 3, -17920, -14080);
        let synapses = SynapseStore::empty(3);

        CascadePool::new("test", neurons, synapses, [10.0, 10.0, 10.0], CascadeConfig::default())
    }

    #[test]
    fn test_cascade_creation() {
        let pool = make_test_pool();
        assert_eq!(pool.n_neurons(), 3);
        assert_eq!(pool.pending_count(), 0);
        assert_eq!(pool.sim_time, 0);
    }

    #[test]
    fn test_inject_and_process() {
        let mut pool = make_test_pool();

        // Inject strong current to neuron 0
        pool.inject(0, 5000, 100);

        assert_eq!(pool.pending_count(), 1);

        // Process
        let processed = pool.run_until(100);
        assert_eq!(processed, 1);
        assert_eq!(pool.sim_time, 100);
    }

    #[test]
    fn test_spike_propagation() {
        // Create 2 neurons with synapse from 0 to 1
        let mut neurons = NeuronArrays::new(2, 2, -17920, -14080);
        neurons.soma_position[0] = [0.0, 0.0, 0.0];
        neurons.soma_position[1] = [1.0, 0.0, 0.0];
        neurons.axon_terminal[0] = [0.5, 0.0, 0.0];
        neurons.axon_terminal[1] = [1.0, 0.0, 0.0];

        let exc_flags = crate::neuron::flags::encode(false, crate::neuron::NeuronProfile::RegularSpiking);
        let edges = vec![
            (0, Synapse::new(1, 100, 1, exc_flags)),
        ];
        let synapses = SynapseStore::from_edges(2, edges);

        let mut pool = CascadePool::new("test", neurons, synapses, [10.0, 10.0, 10.0], CascadeConfig::default());

        // Inject enough current to make neuron 0 spike
        pool.inject(0, 5000, 0);

        // Process until neuron 0 fires
        pool.run_until(0);

        // Should have queued a spike to neuron 1
        assert!(pool.pending_count() > 0 || pool.total_spikes > 0);
    }

    #[test]
    fn test_spike_arrival_ordering() {
        let a1 = SpikeArrival { target: 0, current: 100, arrival_time: 50, source: 1 };
        let a2 = SpikeArrival { target: 1, current: 100, arrival_time: 100, source: 1 };

        // Earlier time should be "greater" for min-heap
        assert!(a1 > a2);
    }

    #[test]
    fn test_leak_decay() {
        let mut pool = make_test_pool();

        // Set membrane above resting
        pool.neurons.membrane[0] = -10000;
        pool.last_update[0] = 0;

        // Apply leak for half the time constant
        pool.apply_leak(0, pool.config.leak_tau / 2);

        // Should have decayed toward resting
        assert!(pool.neurons.membrane[0] < -10000);
        assert!(pool.neurons.membrane[0] > pool.config.resting_potential);
    }
}
