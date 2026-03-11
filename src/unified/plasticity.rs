//! Activity-Dependent Synaptic Plasticity for Unified Neurons.
//!
//! Implements spike-timing-dependent plasticity (STDP) — the biological
//! mechanism by which synapses that causally contribute to postsynaptic
//! firing are strengthened, and ineffective synapses are weakened.
//!
//! ## Biological Basis
//!
//! During development (especially brainstem maturation), synaptic refinement
//! follows a Hebbian rule:
//!
//! - **Pre fires → Post fires within ~20ms**: synapse was *causal*. The
//!   presynaptic spike contributed to the postsynaptic neuron crossing
//!   threshold. Strengthen (long-term potentiation).
//!
//! - **Pre fires → Post doesn't fire**: synapse was *ineffective*. The
//!   presynaptic spike wasn't enough to drive the postsynaptic neuron.
//!   Weaken (long-term depression).
//!
//! This runs ungated during developmental settling — pure activity-dependent
//! refinement. In the mature brain, neuromodulators gate plasticity depth.
//!
//! ## Implementation
//!
//! The plasticity sweep examines each synapse's source and target neurons:
//! - Source `last_spike_us` = when the presynaptic neuron last fired
//! - Target `last_spike_us` = when the postsynaptic neuron last fired
//! - `last_conducted_us` = when the synapse last carried a spike
//!
//! If pre fired, then post fired within the causal window → strengthen.
//! If pre fired but post never fired (or fired too long ago) → weaken.
//! Mature synapses resist modification. Inhibitory synapses follow the
//! same rule (anti-Hebbian: effective inhibition is preserved).

use super::synapse::{UnifiedSynapse, UnifiedSynapseStore};
use super::neuron::UnifiedNeuron;

/// Configuration for activity-dependent plasticity.
#[derive(Clone, Copy, Debug)]
pub struct PlasticityConfig {
    /// Causal timing window in microseconds. If the postsynaptic neuron
    /// fires within this window after the presynaptic neuron, the synapse
    /// is considered causal and gets strengthened.
    pub causal_window_us: u64,
    /// Magnitude increase for causal synapses (LTP).
    pub strengthen_amount: u8,
    /// Magnitude decrease for ineffective synapses (LTD).
    pub weaken_amount: u8,
    /// Maturity increase for causal synapses. Mature synapses resist
    /// future modification.
    pub maturity_boost: u8,
    /// Maturity threshold above which plasticity is reduced. Synapses
    /// with maturity >= this value get halved strengthen/weaken amounts.
    pub maturity_gate: u8,
    /// Minimum time since last pre spike to consider the synapse "recently
    /// active" for plasticity evaluation. Synapses whose source neuron
    /// hasn't fired at all (last_spike_us == 0) are skipped.
    pub min_pre_activity_us: u64,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            causal_window_us: 20_000,   // 20ms — standard STDP window
            strengthen_amount: 1,
            weaken_amount: 1,
            maturity_boost: 1,
            maturity_gate: 150,
            min_pre_activity_us: 0,
        }
    }
}

/// Result of a plasticity sweep.
#[derive(Clone, Debug, Default)]
pub struct PlasticityResult {
    /// Number of synapses strengthened (causal: pre→post within window).
    pub strengthened: usize,
    /// Number of synapses weakened (pre fired, post didn't respond).
    pub weakened: usize,
    /// Number of synapses skipped (source never fired, or mature).
    pub skipped: usize,
    /// Number of synapses that went dormant from weakening.
    pub went_dormant: usize,
}

/// Run one plasticity sweep across all synapses.
///
/// For each synapse, checks spike timing between source and target neurons:
/// - Causal (pre before post within window) → strengthen
/// - Ineffective (pre fired, post didn't follow) → weaken
///
/// This should be called after cascade execution during settling.
pub fn plasticity_sweep(
    neurons: &[UnifiedNeuron],
    synapses: &mut UnifiedSynapseStore,
    current_time_us: u64,
    config: &PlasticityConfig,
) -> PlasticityResult {
    let mut result = PlasticityResult::default();

    for syn in synapses.iter_mut() {
        // Skip dead or frozen synapses
        if syn.health == 0 || syn.maturity == 255 {
            result.skipped += 1;
            continue;
        }

        let src = syn.source as usize;
        let tgt = syn.target as usize;
        if src >= neurons.len() || tgt >= neurons.len() {
            result.skipped += 1;
            continue;
        }

        let pre_spike = neurons[src].last_spike_us;
        let post_spike = neurons[tgt].last_spike_us;

        // Source never fired — nothing to evaluate
        if pre_spike == 0 {
            result.skipped += 1;
            continue;
        }

        // Source fired too long ago — not relevant to current dynamics
        if config.min_pre_activity_us > 0
            && current_time_us.saturating_sub(pre_spike) > config.min_pre_activity_us
        {
            result.skipped += 1;
            continue;
        }

        // Compute effective amounts (halved for mature synapses)
        let (str_amt, wk_amt) = if syn.maturity >= config.maturity_gate {
            (config.strengthen_amount / 2, config.weaken_amount / 2)
        } else {
            (config.strengthen_amount, config.weaken_amount)
        };

        // Check causal timing: pre fired, then post fired within window
        if post_spike > 0 && post_spike >= pre_spike {
            let dt = post_spike - pre_spike;
            if dt <= config.causal_window_us {
                // Causal: strengthen
                syn.strengthen(str_amt.max(1));
                syn.mature(config.maturity_boost);
                result.strengthened += 1;
                continue;
            }
        }

        // Pre fired but post didn't follow within window → weaken
        if wk_amt > 0 {
            let before_mag = syn.signal.magnitude;
            syn.weaken(wk_amt);
            if before_mag > 0 && syn.signal.magnitude == 0 {
                result.went_dormant += 1;
            }
            result.weakened += 1;
        } else {
            result.skipped += 1;
        }
    }

    result
}

// === Mastery Learning ===
//
// Pressure-based plasticity with participation gating, metabolic budgets,
// hub anti-Hebbian dynamics, and polarity flips. Supersedes the simple STDP
// sweep above for production use.

/// Configuration for mastery learning.
#[derive(Clone, Copy, Debug)]
pub struct MasteryConfig {
    /// Pressure threshold before modification is applied.
    pub pressure_threshold: i16,
    /// Magnitude step for strengthen/weaken.
    pub magnitude_step: u8,
    /// Metabolic cost of a polarity flip.
    pub flip_penalty: u32,
    /// Minimum time (μs) between polarity flips on a single synapse.
    pub flip_cooldown_us: u64,
    /// Fan-in count above which hub anti-Hebbian pressure applies.
    pub hub_threshold: u16,
    /// STDP causal timing window (μs).
    pub causal_window_us: u64,
    /// Membrane level below which sub-threshold learning applies.
    pub sub_threshold_level: i16,
    /// Sub-threshold pressure scale (/256). Reduces pressure for near-miss cases.
    pub sub_threshold_scale: u8,
    /// Metabolic budget per sweep. Limits total plasticity per cycle.
    pub metabolic_budget: u32,
    /// Minimum trace (/255) on source neuron to count as participating.
    pub participation_min: u8,
    /// Maturity threshold above which plasticity amounts are halved.
    pub maturity_gate: u8,
    /// Maturity boost on strengthen.
    pub maturity_boost: u8,
}

impl Default for MasteryConfig {
    fn default() -> Self {
        Self {
            pressure_threshold: 50,
            magnitude_step: 5,
            flip_penalty: 50,
            flip_cooldown_us: 100_000,
            hub_threshold: 20,
            causal_window_us: 20_000,
            sub_threshold_level: -6500,
            sub_threshold_scale: 64,
            metabolic_budget: 500,
            participation_min: 64,
            maturity_gate: 150,
            maturity_boost: 1,
        }
    }
}

/// Per-sweep mutable state for mastery learning.
pub struct MasteryState {
    /// Remaining metabolic budget for the current sweep.
    pub metabolic_budget: u32,
    /// Per-synapse last-flip timestamp (μs).
    pub flip_cooldown: Vec<u64>,
    /// Per-neuron incoming connection count (computed at sweep start).
    pub fan_in: Vec<u16>,
    /// Budget spent in the current sweep.
    pub budget_spent: u32,
}

impl MasteryState {
    /// Create a new mastery state for the given synapse count.
    pub fn new(synapse_count: usize) -> Self {
        Self {
            metabolic_budget: 0,
            flip_cooldown: vec![0u64; synapse_count],
            fan_in: Vec::new(),
            budget_spent: 0,
        }
    }

    /// Resize for a new synapse count (after synaptogenesis or pruning).
    pub fn resize(&mut self, synapse_count: usize) {
        self.flip_cooldown.resize(synapse_count, 0);
    }
}

/// Result of a mastery learning sweep.
#[derive(Clone, Debug, Default)]
pub struct MasteryResult {
    pub strengthened: usize,
    pub weakened: usize,
    pub flipped: usize,
    pub awakened: usize,
    pub went_dormant: usize,
    pub skipped: usize,
    pub budget_spent: u32,
    pub hub_pressured: usize,
}

/// Run one mastery learning sweep across all synapses.
///
/// Three phases:
/// 1. Pressure accumulation — causal/anti-causal/sub-threshold/hub dynamics
/// 2. Apply learning — strengthen, weaken, flip, awaken, go dormant
/// 3. Budget accounting
///
/// All integer math. No floats.
pub fn mastery_sweep(
    neurons: &[UnifiedNeuron],
    synapses: &mut UnifiedSynapseStore,
    current_time_us: u64,
    state: &mut MasteryState,
    config: &MasteryConfig,
) -> MasteryResult {
    let mut result = MasteryResult::default();
    state.metabolic_budget = config.metabolic_budget;
    state.budget_spent = 0;

    // Ensure flip_cooldown is sized correctly
    if state.flip_cooldown.len() != synapses.len() {
        state.flip_cooldown.resize(synapses.len(), 0);
    }

    // Compute fan-in per neuron (one pass over all synapses)
    state.fan_in.clear();
    state.fan_in.resize(neurons.len(), 0);
    for syn in synapses.iter() {
        if syn.is_active() {
            let tgt = syn.target as usize;
            if tgt < state.fan_in.len() {
                state.fan_in[tgt] = state.fan_in[tgt].saturating_add(1);
            }
        }
    }

    // Phase 1: Pressure accumulation
    for (syn_idx, syn) in synapses.iter_mut().enumerate() {
        // Skip dead/dormant or frozen synapses
        if syn.health == 0 || syn.maturity == 255 {
            result.skipped += 1;
            continue;
        }

        let src = syn.source as usize;
        let tgt = syn.target as usize;
        if src >= neurons.len() || tgt >= neurons.len() {
            result.skipped += 1;
            continue;
        }

        // Participation gate: source neuron must have sufficient trace
        let src_trace = neurons[src].trace;
        if (src_trace as u8) < config.participation_min && src_trace >= 0 {
            // Source didn't participate — no pressure change
            continue;
        }

        let pre_spike = neurons[src].last_spike_us;
        let post_spike = neurons[tgt].last_spike_us;

        // Source never fired — skip
        if pre_spike == 0 {
            continue;
        }

        // Causal: post fired within window after pre
        if post_spike > 0 && post_spike >= pre_spike {
            let dt = post_spike - pre_spike;
            if dt <= config.causal_window_us {
                syn.accumulate_pressure(config.magnitude_step as i16);
                continue;
            }
        }

        // Sub-threshold: pre fired, post membrane above sub_threshold_level but didn't fire
        // (or fired outside the window). Weak positive pressure.
        if pre_spike > 0 && neurons[tgt].membrane > config.sub_threshold_level {
            let scaled = (config.magnitude_step as i16 * config.sub_threshold_scale as i16) / 256;
            if scaled > 0 {
                syn.accumulate_pressure(scaled);
                continue;
            }
        }

        // Anti-causal: pre fired, post didn't fire or fired before pre
        if pre_spike > 0 {
            syn.accumulate_pressure(-(config.magnitude_step as i16));
        }

        // Hub anti-Hebbian: if target fan-in exceeds threshold, apply extra negative pressure
        let _ = syn_idx; // used for flip_cooldown indexing in phase 2
        if state.fan_in[tgt] > config.hub_threshold {
            syn.accumulate_pressure(-1);
            result.hub_pressured += 1;
        }
    }

    // Phase 2: Apply learning where |pressure| >= threshold
    for (syn_idx, syn) in synapses.iter_mut().enumerate() {
        if syn.health == 0 || syn.maturity == 255 {
            continue;
        }
        if state.metabolic_budget == 0 {
            break; // budget exhausted
        }

        let abs_pressure = syn.pressure.unsigned_abs();
        if abs_pressure < config.pressure_threshold as u16 {
            continue;
        }

        let positive = syn.pressure > 0;

        // Effective amounts (halved for mature synapses)
        let step = if syn.maturity >= config.maturity_gate {
            (config.magnitude_step / 2).max(1)
        } else {
            config.magnitude_step
        };

        if syn.is_dormant() && syn.health > 0 {
            // Dormant but structurally alive — awaken in desired direction
            if positive {
                syn.signal.polarity = 1;
            } else {
                syn.signal.polarity = -1;
            }
            syn.signal.magnitude = 10;
            syn.pressure = 0;
            state.metabolic_budget = state.metabolic_budget.saturating_sub(step as u32);
            state.budget_spent += step as u32;
            result.awakened += 1;
            continue;
        }

        let is_aligned = (positive && syn.is_excitatory()) || (!positive && syn.is_inhibitory());

        if is_aligned {
            // Pressure aligns with polarity — strengthen
            syn.strengthen(step);
            syn.mature(config.maturity_boost);
            syn.pressure = 0;
            state.metabolic_budget = state.metabolic_budget.saturating_sub(step as u32);
            state.budget_spent += step as u32;
            result.strengthened += 1;
        } else {
            // Pressure opposes polarity — weaken first
            let before_mag = syn.signal.magnitude;
            syn.weaken(step);
            state.metabolic_budget = state.metabolic_budget.saturating_sub(step as u32);
            state.budget_spent += step as u32;

            if syn.signal.magnitude == 0 && before_mag > 0 {
                // Magnitude depleted — check if we can flip
                let cooldown_ok = syn_idx < state.flip_cooldown.len()
                    && current_time_us.saturating_sub(state.flip_cooldown[syn_idx])
                        >= config.flip_cooldown_us;
                let budget_ok = state.metabolic_budget >= config.flip_penalty;

                if cooldown_ok && budget_ok {
                    // Polarity flip
                    syn.signal.polarity = if positive { 1 } else { -1 };
                    syn.signal.magnitude = 10;
                    state.metabolic_budget = state.metabolic_budget.saturating_sub(config.flip_penalty);
                    state.budget_spent += config.flip_penalty;
                    if syn_idx < state.flip_cooldown.len() {
                        state.flip_cooldown[syn_idx] = current_time_us;
                    }
                    result.flipped += 1;
                } else {
                    // Can't flip — go dormant
                    result.went_dormant += 1;
                }
            } else {
                result.weakened += 1;
            }
            syn.pressure = 0;
        }
    }

    result.budget_spent = state.budget_spent;
    result
}

/// Configuration for ACh-gated synaptogenesis.
#[derive(Clone, Copy, Debug)]
pub struct SynaptogenesisConfig {
    /// ACh level must exceed this for synaptogenesis to proceed.
    pub ach_threshold: u8,
    /// Co-activation window (μs): both neurons must have fired within this window.
    pub coactivation_window_us: u64,
    /// Maximum new synapses created per sweep.
    pub max_new_per_sweep: u16,
    /// Initial magnitude of newborn synapses.
    pub initial_magnitude: u8,
    /// Initial health of newborn synapses.
    pub initial_health: u8,
}

impl Default for SynaptogenesisConfig {
    fn default() -> Self {
        Self {
            ach_threshold: 140,
            coactivation_window_us: 20_000,
            max_new_per_sweep: 16,
            initial_magnitude: 10,
            initial_health: 200,
        }
    }
}

/// Create new synapses between co-active neuron pairs.
///
/// Gated by ACh level (attention). Only fires when ACh exceeds threshold.
/// Finds recently co-active neuron pairs that lack direct connections and
/// wires them with small excitatory synapses.
pub fn synaptogenesis(
    neurons: &[UnifiedNeuron],
    synapses: &mut UnifiedSynapseStore,
    current_time_us: u64,
    ach_level: u8,
    config: &SynaptogenesisConfig,
) -> usize {
    if ach_level < config.ach_threshold {
        return 0;
    }

    // Collect recently-fired neuron indices
    let recently_fired: Vec<u32> = neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            n.last_spike_us > 0
                && current_time_us.saturating_sub(n.last_spike_us) <= config.coactivation_window_us
        })
        .map(|(i, _)| i as u32)
        .collect();

    if recently_fired.len() < 2 {
        return 0;
    }

    let mut created = 0u16;

    // For each pair, check if a direct synapse exists
    for i in 0..recently_fired.len() {
        if created >= config.max_new_per_sweep {
            break;
        }
        for j in (i + 1)..recently_fired.len() {
            if created >= config.max_new_per_sweep {
                break;
            }
            let a = recently_fired[i];
            let b = recently_fired[j];

            // Check if A→B already exists
            let a_to_b_exists = synapses.outgoing(a).iter().any(|s| s.target == b);
            if !a_to_b_exists {
                // Create A→B excitatory feedforward synapse
                let mut new_syn = UnifiedSynapse::excitatory(
                    a,
                    b,
                    super::zone::DendriticZone::Feedforward,
                    config.initial_magnitude,
                    500, // 500μs default delay
                );
                new_syn.health = config.initial_health;
                synapses.add(new_syn);
                created += 1;
            }
        }
    }

    if created > 0 {
        synapses.rebuild_index(neurons.len() as u32);
    }

    created as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::neuron::VoxelPosition;
    use crate::unified::synapse::UnifiedSynapse;
    use crate::unified::zone::DendriticZone;

    fn pos(x: u16) -> VoxelPosition {
        VoxelPosition::at_center((x, 0, 0))
    }

    #[test]
    fn causal_synapse_strengthened() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        // Pre fires at t=1000, post fires at t=1010 (within 20ms window)
        neurons[0].last_spike_us = 1000;
        neurons[1].last_spike_us = 1010;

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 2000, &config);

        assert_eq!(result.strengthened, 1);
        assert_eq!(result.weakened, 0);

        let syn = store.iter().next().unwrap();
        assert_eq!(syn.signal.magnitude, 101); // 100 + 1
        assert_eq!(syn.maturity, 1);
    }

    #[test]
    fn ineffective_synapse_weakened() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        // Pre fires at t=1000, post never fires
        neurons[0].last_spike_us = 1000;
        neurons[1].last_spike_us = 0;

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 2000, &config);

        assert_eq!(result.strengthened, 0);
        assert_eq!(result.weakened, 1);

        let syn = store.iter().next().unwrap();
        assert_eq!(syn.signal.magnitude, 99); // 100 - 1
    }

    #[test]
    fn post_fires_outside_window_weakens() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        // Pre fires at t=1000, post fires at t=50000 (30ms later, outside 20ms window)
        neurons[0].last_spike_us = 1000;
        neurons[1].last_spike_us = 50_000;

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 60_000, &config);

        assert_eq!(result.weakened, 1);
        assert_eq!(result.strengthened, 0);
    }

    #[test]
    fn source_never_fired_skipped() {
        let neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        // Neither has fired
        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 1000, &config);

        assert_eq!(result.skipped, 1);
        assert_eq!(result.strengthened, 0);
        assert_eq!(result.weakened, 0);
    }

    #[test]
    fn mature_synapse_reduced_plasticity() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 1000;
        neurons[1].last_spike_us = 1010;

        let mut store = UnifiedSynapseStore::new();
        let mut syn = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        syn.maturity = 200; // above gate (150)
        store.add(syn);
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 2000, &config);

        assert_eq!(result.strengthened, 1);
        // strengthen_amount=1, halved=0, max(0, 1) = 1
        let syn = store.iter().next().unwrap();
        assert_eq!(syn.signal.magnitude, 101); // 100 + 1 (halved from 2)
    }

    #[test]
    fn weakening_to_dormancy() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 1000;
        neurons[1].last_spike_us = 0;

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 1, 500));
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 2000, &config);

        assert_eq!(result.weakened, 1);
        assert_eq!(result.went_dormant, 1);

        let syn = store.iter().next().unwrap();
        assert_eq!(syn.signal.magnitude, 0);
        assert!(syn.is_dormant());
    }

    #[test]
    fn inhibitory_synapses_also_learn() {
        let mut neurons = vec![
            UnifiedNeuron::interneuron_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        // Causal: inhibitory pre fires, post fires (inhibition didn't block it,
        // but the timing is causal → strengthen the inhibitory connection)
        neurons[0].last_spike_us = 1000;
        neurons[1].last_spike_us = 1005;

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::inhibitory(0, 1, DendriticZone::Feedforward, 50, 500));
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 2000, &config);

        assert_eq!(result.strengthened, 1);
        let syn = store.iter().next().unwrap();
        assert_eq!(syn.signal.magnitude, 51); // 50 + 1
        assert!(syn.is_inhibitory());
    }

    #[test]
    fn dead_synapse_skipped() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 1000;
        neurons[1].last_spike_us = 1010;

        let mut store = UnifiedSynapseStore::new();
        let mut syn = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        syn.health = 0; // dead
        store.add(syn);
        store.rebuild_index(2);

        let config = PlasticityConfig::default();
        let result = plasticity_sweep(&neurons, &mut store, 2000, &config);

        assert_eq!(result.skipped, 1);
        assert_eq!(result.strengthened, 0);
    }

    // === Mastery Learning Tests ===

    #[test]
    fn mastery_causal_accumulates_pressure() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 1000;
        neurons[0].trace = 127; // above participation_min (64)
        neurons[1].last_spike_us = 1010; // causal: within 20ms

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = MasteryConfig::default();
        let mut state = MasteryState::new(1);

        // Run enough sweeps to accumulate pressure above threshold (50)
        for _ in 0..11 {
            mastery_sweep(&neurons, &mut store, 2000, &mut state, &config);
        }

        let syn = store.iter().next().unwrap();
        // After 10 causal sweeps: pressure += 5 each = 50, then applied → strengthen
        assert!(syn.signal.magnitude > 100, "synapse should be strengthened: mag={}", syn.signal.magnitude);
    }

    #[test]
    fn mastery_anti_causal_weakens() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 1000;
        neurons[0].trace = 127;
        neurons[1].last_spike_us = 0; // post never fired

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = MasteryConfig::default();
        let mut state = MasteryState::new(1);

        for _ in 0..11 {
            mastery_sweep(&neurons, &mut store, 2000, &mut state, &config);
        }

        let syn = store.iter().next().unwrap();
        assert!(syn.signal.magnitude < 100, "synapse should be weakened: mag={}", syn.signal.magnitude);
    }

    #[test]
    fn mastery_participation_gate() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 1000;
        neurons[0].trace = 10; // below participation_min (64)
        neurons[1].last_spike_us = 1010;

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = MasteryConfig::default();
        let mut state = MasteryState::new(1);
        mastery_sweep(&neurons, &mut store, 2000, &mut state, &config);

        let syn = store.iter().next().unwrap();
        assert_eq!(syn.pressure, 0, "no pressure should accumulate when source didn't participate");
        assert_eq!(syn.signal.magnitude, 100, "magnitude unchanged");
    }

    #[test]
    fn mastery_budget_limits_changes() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
            UnifiedNeuron::pyramidal_at(pos(2)),
        ];
        for n in &mut neurons {
            n.last_spike_us = 1000;
            n.trace = 127;
        }
        neurons[1].last_spike_us = 1010;
        neurons[2].last_spike_us = 1010;

        let mut store = UnifiedSynapseStore::new();
        // Two synapses, both with pressure above threshold
        let mut syn1 = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        syn1.pressure = 60; // above threshold
        store.add(syn1);
        let mut syn2 = UnifiedSynapse::excitatory(0, 2, DendriticZone::Feedforward, 100, 500);
        syn2.pressure = 60;
        store.add(syn2);
        store.rebuild_index(3);

        let config = MasteryConfig {
            metabolic_budget: 5, // very low budget — only enough for 1 modification
            ..MasteryConfig::default()
        };
        let mut state = MasteryState::new(2);
        let result = mastery_sweep(&neurons, &mut store, 2000, &mut state, &config);

        // Only 1 should have been modified (budget of 5, step of 5)
        assert_eq!(result.strengthened, 1, "budget should limit to 1 modification");
    }

    // === Synaptogenesis Tests ===

    #[test]
    fn synaptogenesis_below_ach_threshold() {
        let neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        let mut store = UnifiedSynapseStore::new();
        store.rebuild_index(2);

        let config = SynaptogenesisConfig::default();
        let created = synaptogenesis(&neurons, &mut store, 1000, 100, &config); // ACh=100 < 140

        assert_eq!(created, 0);
    }

    #[test]
    fn synaptogenesis_creates_new_connections() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 990;
        neurons[1].last_spike_us = 995;

        let mut store = UnifiedSynapseStore::new();
        store.rebuild_index(2);

        let config = SynaptogenesisConfig::default();
        let created = synaptogenesis(&neurons, &mut store, 1000, 200, &config); // ACh=200 > 140

        assert_eq!(created, 1);
        assert_eq!(store.len(), 1);
        let syn = store.iter().next().unwrap();
        assert_eq!(syn.source, 0);
        assert_eq!(syn.target, 1);
        assert_eq!(syn.signal.magnitude, 10);
    }

    #[test]
    fn synaptogenesis_skips_existing_connections() {
        let mut neurons = vec![
            UnifiedNeuron::pyramidal_at(pos(0)),
            UnifiedNeuron::pyramidal_at(pos(1)),
        ];
        neurons[0].last_spike_us = 990;
        neurons[1].last_spike_us = 995;

        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.rebuild_index(2);

        let config = SynaptogenesisConfig::default();
        let created = synaptogenesis(&neurons, &mut store, 1000, 200, &config);

        assert_eq!(created, 0, "should not duplicate existing connection");
    }
}
