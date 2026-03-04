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

use super::synapse::UnifiedSynapseStore;
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
        // Skip dead synapses
        if syn.health == 0 {
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
}
