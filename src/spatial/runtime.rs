#![allow(deprecated)]
//! SpatialRuntime — orchestrates cascade, tissue, migration, pruning, and mastery.
//!
//! Extracts the runtime loop from test code into a reusable struct.
//! A consumer (Hush v2, etc.) creates a runtime, injects sensory input,
//! calls `step()`, and reads motor output. The runtime handles all
//! subsystem choreography internally.
//!
//! ## Usage
//!
//! ```ignore
//! let runtime = SpatialRuntime::new(neurons, synapses, config);
//! // Per frame:
//! runtime.inject_sensory_scaled(&mfcc_coeffs, 2000.0, 1500.0, 0.3, 0.1, time);
//! let spikes = runtime.step(10_000); // 10ms frame
//! let outputs = runtime.read_motors();
//! ```

use super::{
    SpatialCascade, SpatialCascadeConfig, SpatialNeuron, SpatialSynapseStore,
    TissueConfig, TissueField,
    MigrationConfig, CorrelationTracker, migrate_step,
    PruningConfig, DormancyTracker, pruning_cycle, hard_prune,
    MasteryConfig, MasteryState, PolarityChange, HubTracker,
};

/// Configuration for the spatial runtime.
#[derive(Clone, Debug)]
pub struct SpatialRuntimeConfig {
    pub cascade: SpatialCascadeConfig,
    pub tissue: TissueConfig,
    pub migration: MigrationConfig,
    pub pruning: PruningConfig,
    pub mastery: MasteryConfig,
    /// Frames between structural maintenance (tissue + migration)
    pub structural_interval: u32,
    /// Frames between pruning cycles (relative to structural_interval)
    pub pruning_interval: u32,
    /// Frames between hard prune (relative to structural_interval)
    pub hard_prune_interval: u32,
    /// Frames between mastery learning cycles
    pub mastery_interval: u32,
    /// Metabolic budget added per mastery cycle
    pub mastery_budget_per_cycle: u32,
    /// Co-firing window for mastery Hebbian pressure (μs)
    pub mastery_learning_window_us: u64,
    /// Sub-threshold membrane level for gentle strengthening
    pub sub_threshold_level: i16,
    /// Sub-threshold activity scaling (0.0-1.0)
    pub sub_threshold_scale: f32,
    /// Correlation tracker: max spikes per neuron to retain
    pub correlation_max_spikes: usize,
    /// Correlation tracker: time window for spike retention (μs)
    pub correlation_window_us: u64,
}

impl Default for SpatialRuntimeConfig {
    fn default() -> Self {
        Self {
            cascade: SpatialCascadeConfig::default(),
            tissue: TissueConfig::default(),
            migration: MigrationConfig {
                migration_rate: 0.05,
                correlation_threshold: 0.2,
                attraction_strength: 0.5,
                repulsion_strength: 0.3,
                min_distance: 0.3,
                max_step: 0.2,
                axon_elasticity: 0.8,
                exclusion_radius: 0.3,
                exclusion_strength: 2.0,
                origin_spring: 0.05,
            },
            pruning: PruningConfig::default(),
            mastery: MasteryConfig {
                pressure_threshold: 20,
                participation_threshold: 0.15,
                magnitude_cost: 3,
                flip_penalty: 40,
                pressure_scale: 2.5,
                hub_threshold: 15,
                hub_decay_rate: 0.1,
                flip_cooldown_us: 50_000,
            },
            structural_interval: 100,
            pruning_interval: 5000,
            hard_prune_interval: 10000,
            mastery_interval: 5,
            mastery_budget_per_cycle: 200,
            mastery_learning_window_us: 100_000,
            sub_threshold_level: -6500,
            sub_threshold_scale: 0.3,
            correlation_max_spikes: 100,
            correlation_window_us: 10_000,
        }
    }
}

/// Diagnostic counters for learning events.
#[derive(Clone, Copy, Debug, Default)]
pub struct LearningCounters {
    pub strengthened: u32,
    pub weakened: u32,
    pub dormant: u32,
    pub awakened: u32,
    pub flipped: u32,
    pub cycles: u32,
}

/// Diagnostic counters for structural maintenance.
#[derive(Clone, Copy, Debug, Default)]
pub struct StructuralCounters {
    pub migration_steps: u32,
    pub tissue_updates: u32,
    pub pruning_cycles: u32,
    pub hard_pruned: u32,
}

/// Orchestrates all spatial neuron subsystems.
pub struct SpatialRuntime {
    /// The cascade executor (public for direct neuron/synapse access).
    pub cascade: SpatialCascade,
    tissue: TissueField,
    correlations: CorrelationTracker,
    initial_positions: Vec<[f32; 3]>,
    dormancy: DormancyTracker,
    mastery: MasteryState,
    hub_tracker: HubTracker,
    config: SpatialRuntimeConfig,
    /// Current simulation time in μs.
    time_us: u64,
    /// Frame counter (for interval-based subsystem scheduling).
    frame_count: u64,
    /// Learning diagnostics.
    pub learning: LearningCounters,
    /// Structural maintenance diagnostics.
    pub structural: StructuralCounters,
}

impl SpatialRuntime {
    /// Create a new runtime from neurons and synapses.
    pub fn new(
        neurons: Vec<SpatialNeuron>,
        synapses: SpatialSynapseStore,
        config: SpatialRuntimeConfig,
    ) -> Self {
        let neuron_count = neurons.len();
        let synapse_count = synapses.len();

        let cascade = SpatialCascade::with_network(neurons, synapses, config.cascade);

        let initial_positions: Vec<[f32; 3]> =
            cascade.neurons.iter().map(|n| n.soma.position).collect();

        let mut tissue = TissueField::with_config(config.tissue.clone());
        tissue.rebuild(&cascade.neurons);

        let correlations = CorrelationTracker::new(
            neuron_count,
            config.correlation_max_spikes,
            config.correlation_window_us,
        );

        let mastery = MasteryState::new(synapse_count, config.mastery, 10_000);

        let mut hub_tracker = HubTracker::new(neuron_count);
        for syn in cascade.synapses.iter() {
            hub_tracker.record_connection(syn.target);
        }

        let dormancy = DormancyTracker::new(synapse_count);

        Self {
            cascade,
            tissue,
            correlations,
            initial_positions,
            dormancy,
            mastery,
            hub_tracker,
            config,
            time_us: 0,
            frame_count: 0,
            learning: LearningCounters::default(),
            structural: StructuralCounters::default(),
        }
    }

    /// Inject external current to a specific neuron.
    #[inline]
    pub fn inject(&mut self, neuron: u32, current: i16) {
        self.cascade.inject(neuron, current, self.time_us);
    }

    /// Inject scaled sensory input with neighborhood activation.
    ///
    /// For each coefficient:
    /// - Skip if |coeff| < `silence_threshold`
    /// - Inject `coeff * scale` as current to the matching sensory neuron
    /// - If |coeff| > `neighbor_threshold`: inject `coeff * neighbor_scale`
    ///   to adjacent sensory channels
    pub fn inject_sensory_scaled(
        &mut self,
        coefficients: &[f32],
        scale: f32,
        neighbor_scale: f32,
        neighbor_threshold: f32,
        silence_threshold: f32,
    ) {
        self.cascade.inject_sensory_scaled(
            coefficients,
            scale,
            neighbor_scale,
            neighbor_threshold,
            silence_threshold,
            self.time_us,
        );
    }

    /// Run one frame of the simulation.
    ///
    /// Advances time by `frame_interval_us`, runs cascade with tissue,
    /// records spike correlations, and triggers structural maintenance
    /// at configured intervals.
    ///
    /// Returns the number of spikes this frame.
    pub fn step(&mut self, frame_interval_us: u64) -> u64 {
        let target_time = self.time_us + frame_interval_us;

        // Cascade propagation with tissue physics
        let spikes = self.cascade.run_until_with_tissue(target_time, &self.tissue);

        // Per-frame stamina recovery (before correlation tracking)
        self.cascade.recover_stamina(frame_interval_us);

        // Record spikes for correlation tracking
        for (idx, neuron) in self.cascade.neurons.iter().enumerate() {
            if neuron.last_spike_us > self.time_us.saturating_sub(frame_interval_us) {
                self.correlations.record_spike(idx, neuron.last_spike_us);
            }
        }

        // Mastery learning
        if self.frame_count % self.config.mastery_interval as u64
            == (self.config.mastery_interval as u64 - 1)
        {
            self.run_mastery_cycle();
        }

        // Structural maintenance (tissue + migration + pruning)
        if self.frame_count % self.config.structural_interval as u64
            == (self.config.structural_interval as u64 - 1)
        {
            self.run_structural_maintenance();
        }

        self.time_us = target_time;
        self.frame_count += 1;
        spikes
    }

    /// Read motor neuron outputs.
    ///
    /// Returns `(channel, trace)` for motor neurons that have fired.
    pub fn read_motors(&self) -> Vec<(u16, i16)> {
        self.cascade.read_motor_outputs()
    }

    /// Current simulation time in μs.
    #[inline]
    pub fn time_us(&self) -> u64 {
        self.time_us
    }

    /// Total frames processed.
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Access tissue field (for diagnostics/snapshots).
    pub fn tissue(&self) -> &TissueField {
        &self.tissue
    }

    /// Access initial positions (for displacement calculations).
    pub fn initial_positions(&self) -> &[[f32; 3]] {
        &self.initial_positions
    }

    /// Access correlation tracker (for region detection, etc.).
    pub fn correlations(&self) -> &CorrelationTracker {
        &self.correlations
    }

    /// Set the mastery budget added per learning cycle.
    ///
    /// Higher budget → more synaptic changes per cycle (consolidation).
    /// Lower budget → fewer changes (exploration / loosening).
    #[inline]
    pub fn set_mastery_budget(&mut self, budget: u32) {
        self.config.mastery_budget_per_cycle = budget;
    }

    // =========================================================================
    // Internal subsystem orchestration
    // =========================================================================

    /// Run one mastery learning cycle.
    ///
    /// Sub-threshold aware Hebbian pressure → hub pressure → apply learning.
    fn run_mastery_cycle(&mut self) {
        self.mastery.set_time(self.time_us);
        let window = self.config.mastery_learning_window_us;

        // Collect synapse info (source, target) to avoid borrow conflict
        let syn_info: Vec<(u32, u32)> = self
            .cascade
            .synapses
            .iter()
            .map(|s| (s.source, s.target))
            .collect();

        // Phase 1: Sub-threshold aware Hebbian pressure
        for (syn_idx, &(src, tgt)) in syn_info.iter().enumerate() {
            let src_fired =
                self.cascade.neurons[src as usize].last_spike_us > self.time_us.saturating_sub(window);
            if !src_fired {
                continue;
            }

            let activity = self.cascade.neurons[src as usize].trace as f32 / 255.0;
            let tgt_fired =
                self.cascade.neurons[tgt as usize].last_spike_us > self.time_us.saturating_sub(window);
            let tgt_membrane = self.cascade.neurons[tgt as usize].membrane;

            let (direction, eff_activity): (i8, f32) = if tgt_fired {
                (1, activity)
            } else if tgt_membrane > self.config.sub_threshold_level {
                (1, activity * self.config.sub_threshold_scale)
            } else {
                (-1, activity)
            };

            self.mastery.accumulate_pressure(syn_idx, eff_activity, direction);
            self.hub_tracker.record_activation(tgt, 1);
        }

        // Phase 2: Hub pressure (weaken synapses targeting hubs)
        let hub_targets = self
            .hub_tracker
            .hub_synapses_to_weaken(self.config.mastery.hub_threshold);
        for (syn_idx, &(_, tgt)) in syn_info.iter().enumerate() {
            if hub_targets.contains(&tgt) {
                self.mastery.accumulate_pressure(syn_idx, 0.5, -1);
            }
        }

        // Phase 3: Apply learning to all synapses
        for (syn_idx, syn) in self.cascade.synapses.iter_mut().enumerate() {
            if let Some(change) = self.mastery.apply_learning(syn_idx, syn) {
                match change {
                    PolarityChange::Strengthened => self.learning.strengthened += 1,
                    PolarityChange::Weakened => self.learning.weakened += 1,
                    PolarityChange::GoneDormant => self.learning.dormant += 1,
                    PolarityChange::Awakened => self.learning.awakened += 1,
                    PolarityChange::Flipped => self.learning.flipped += 1,
                }
            }
        }

        self.hub_tracker.clear_activation();
        self.mastery.add_budget(self.config.mastery_budget_per_cycle);
        self.learning.cycles += 1;
    }

    /// Run structural maintenance: tissue plasticity, migration, pruning.
    fn run_structural_maintenance(&mut self) {
        // Tissue plasticity: active neurons soften local tissue
        let active_mask: Vec<bool> = self
            .cascade
            .neurons
            .iter()
            .map(|n| n.last_spike_us > self.time_us.saturating_sub(1_000_000))
            .collect();
        self.tissue.update_plasticity(&active_mask);
        self.tissue.rebuild(&self.cascade.neurons);
        self.structural.tissue_updates += 1;

        // Pruning cycle (less frequent)
        if self.frame_count % self.config.pruning_interval as u64
            == (self.config.pruning_interval as u64 - 1)
        {
            let _result = pruning_cycle(
                &mut self.cascade.neurons,
                &self.cascade.synapses,
                &mut self.dormancy,
                &self.config.pruning,
            );
            self.structural.pruning_cycles += 1;

            // Hard prune (even less frequent)
            if self.frame_count % self.config.hard_prune_interval as u64
                == (self.config.hard_prune_interval as u64 - 1)
            {
                let removed = hard_prune(
                    &mut self.cascade.synapses,
                    &mut self.dormancy,
                    self.cascade.neurons.len(),
                );
                self.structural.hard_pruned += removed as u32;
                if removed > 0 {
                    self.cascade.rebuild_synapse_index();
                }
            }
        }

        // Migration
        migrate_step(
            &mut self.cascade.neurons,
            &self.correlations,
            &self.config.migration,
            self.time_us,
            Some(&self.initial_positions),
            Some(&self.tissue),
        );
        self.structural.migration_steps += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Axon, SpatialNeuron, SpatialSynapse, SpatialSynapseStore};

    fn make_test_runtime() -> SpatialRuntime {
        let mut neurons = Vec::new();

        // 2 sensory neurons
        let mut s0 = SpatialNeuron::sensory_at([0.0, 0.0, 0.0], 0, 1);
        s0.axon = Axon::myelinated([3.0, 0.0, 0.0], 100);
        neurons.push(s0);

        let mut s1 = SpatialNeuron::sensory_at([1.0, 0.0, 0.0], 1, 1);
        s1.axon = Axon::myelinated([3.0, 1.0, 0.0], 100);
        neurons.push(s1);

        // 2 interneurons
        let mut i0 = SpatialNeuron::pyramidal_at([3.0, 0.5, 0.0]);
        i0.axon = Axon::myelinated([6.0, 0.5, 0.0], 80);
        neurons.push(i0);

        let mut i1 = SpatialNeuron::pyramidal_at([4.0, 0.5, 0.0]);
        i1.axon = Axon::myelinated([6.0, 1.0, 0.0], 80);
        neurons.push(i1);

        // 1 motor neuron
        neurons.push(SpatialNeuron::motor_at([6.0, 0.5, 0.0], 0, 1));

        let mut store = SpatialSynapseStore::new(5);
        // sensory → inter
        store.add(SpatialSynapse::excitatory(0, 2, 100, 0));
        store.add(SpatialSynapse::excitatory(1, 3, 100, 0));
        // inter → motor
        store.add(SpatialSynapse::excitatory(2, 4, 80, 0));
        store.add(SpatialSynapse::excitatory(3, 4, 80, 0));
        store.rebuild_index(5);

        SpatialRuntime::new(neurons, store, SpatialRuntimeConfig::default())
    }

    #[test]
    fn test_runtime_creation() {
        let rt = make_test_runtime();
        assert_eq!(rt.cascade.neurons.len(), 5);
        assert_eq!(rt.frame_count(), 0);
        assert_eq!(rt.time_us(), 0);
    }

    #[test]
    fn test_runtime_step() {
        let mut rt = make_test_runtime();

        // Inject strong current to sensory neuron
        rt.inject(0, 2000);
        let spikes = rt.step(10_000);

        assert!(rt.time_us() > 0);
        assert_eq!(rt.frame_count(), 1);
        assert!(spikes > 0 || rt.cascade.total_events() > 0);
    }

    #[test]
    fn test_runtime_multi_step() {
        let mut rt = make_test_runtime();

        for frame in 0..200 {
            rt.inject(0, 1500);
            if frame % 3 == 0 {
                rt.inject(1, 1200);
            }
            rt.step(10_000);
        }

        assert_eq!(rt.frame_count(), 200);
        assert!(rt.cascade.total_spikes() > 0);
        // Structural maintenance should have run at least once
        assert!(rt.structural.tissue_updates > 0);
    }

    #[test]
    fn test_runtime_read_motors() {
        let mut rt = make_test_runtime();

        // Run several frames with strong input
        for _ in 0..100 {
            rt.inject(0, 2000);
            rt.inject(1, 2000);
            rt.step(10_000);
        }

        let motors = rt.read_motors();
        // Motors may or may not have fired depending on convergence,
        // but the API should work
        let _ = motors;
    }

    #[test]
    fn test_runtime_learning_counters() {
        let mut rt = make_test_runtime();

        for _ in 0..50 {
            rt.inject(0, 2000);
            rt.step(10_000);
        }

        // After 50 frames with mastery_interval=5, should have 10 learning cycles
        assert_eq!(rt.learning.cycles, 10);
    }

    #[test]
    fn test_runtime_initial_positions() {
        let rt = make_test_runtime();
        let positions = rt.initial_positions();
        assert_eq!(positions.len(), 5);
        assert_eq!(positions[0], [0.0, 0.0, 0.0]); // first sensory
    }
}
