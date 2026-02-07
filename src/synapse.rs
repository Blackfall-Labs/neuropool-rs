//! Synapse data structures — 8-byte compact synapses with CSR storage.
//!
//! Each synapse is 8 bytes: target(2) + weight(1) + delay(1) + eligibility(1)
//! + maturity(1) + reserved(2). Stored in CSR (Compressed Sparse Row) format
//! for cache-friendly iteration during spike propagation.

use crate::neuron::flags;

/// Thermal state of a synapse, encoded in bits 0-1 of the maturity byte.
///
/// Mirrors the thermogram thermal lifecycle but at the individual synapse level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ThermalState {
    /// Newly formed, high plasticity, vulnerable to pruning
    Hot = 0b00,
    /// Established, moderate plasticity
    Warm = 0b01,
    /// Proven, low plasticity, protected from weakening
    Cool = 0b10,
    /// Frozen, genome-level, immutable
    Cold = 0b11,
}

impl ThermalState {
    #[inline]
    pub fn from_maturity(maturity: u8) -> Self {
        match maturity & 0b11 {
            0b00 => Self::Hot,
            0b01 => Self::Warm,
            0b10 => Self::Cool,
            0b11 => Self::Cold,
            _ => unreachable!(),
        }
    }
}

/// Maturity byte encoding utilities.
///
/// Layout: `[counter:6][state:2]`
/// - Bits 0-1: ThermalState
/// - Bits 2-7: Reinforcement counter (0-63)
pub mod maturity {
    use super::ThermalState;

    /// Promotion thresholds (counter value at which state advances)
    pub const HOT_TO_WARM: u8 = 8;
    pub const WARM_TO_COOL: u8 = 24;
    pub const COOL_TO_COLD: u8 = 63;

    /// Encode maturity byte from state and counter
    #[inline]
    pub fn encode(state: ThermalState, counter: u8) -> u8 {
        (counter.min(63) << 2) | (state as u8)
    }

    /// Decode state from maturity byte
    #[inline]
    pub fn state(m: u8) -> ThermalState {
        ThermalState::from_maturity(m)
    }

    /// Decode counter from maturity byte
    #[inline]
    pub fn counter(m: u8) -> u8 {
        m >> 2
    }

    /// Increment counter, promoting state if threshold reached.
    /// Returns new maturity byte.
    #[inline]
    pub fn increment(m: u8) -> u8 {
        let s = state(m);
        let c = counter(m);

        if s == ThermalState::Cold {
            return m; // Frozen, no change
        }

        let new_c = c.saturating_add(1).min(63);
        let new_state = match s {
            ThermalState::Hot if new_c >= HOT_TO_WARM => ThermalState::Warm,
            ThermalState::Warm if new_c >= WARM_TO_COOL => ThermalState::Cool,
            ThermalState::Cool if new_c >= COOL_TO_COLD => ThermalState::Cold,
            other => other,
        };

        // Reset counter on promotion
        if new_state != s {
            encode(new_state, 0)
        } else {
            encode(s, new_c)
        }
    }

    /// Decrement counter. If counter reaches 0 in HOT state, synapse is dead.
    /// Returns new maturity byte. Dead synapses have maturity == 0x00.
    #[inline]
    pub fn decrement(m: u8) -> u8 {
        let s = state(m);
        let c = counter(m);

        if s == ThermalState::Cold {
            return m; // Frozen, no change
        }

        if c == 0 {
            // Demote state
            let new_state = match s {
                ThermalState::Cool => ThermalState::Warm,
                ThermalState::Warm => ThermalState::Hot,
                ThermalState::Hot => return 0x00, // DEAD — maturity 0 means prunable
                ThermalState::Cold => return m,
            };
            // Start counter at half the promotion threshold for the new (lower) state
            let new_c = match new_state {
                ThermalState::Warm => WARM_TO_COOL / 2,
                ThermalState::Hot => HOT_TO_WARM / 2,
                _ => 0,
            };
            encode(new_state, new_c)
        } else {
            encode(s, c - 1)
        }
    }

    /// Check if a synapse is dead (HOT state with counter 0)
    #[inline]
    pub fn is_dead(m: u8) -> bool {
        m == 0x00
    }
}

/// A single synapse — 8 bytes, repr(C) for binary persistence.
///
/// Weight sign is constrained by Dale's Law: excitatory source neurons produce
/// positive weights only, inhibitory neurons produce negative weights only.
/// This is enforced at creation time, not checked in the hot path.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Synapse {
    /// Post-synaptic neuron index (max 65535)
    pub target: u16,
    /// Signed weight (-127..+127). Sign constrained by Dale's Law.
    pub weight: i8,
    /// Axonal delay in ticks (1-8). 0 is invalid.
    pub delay: u8,
    /// Eligibility trace: STDP tag, decays toward 0 each tick.
    /// Positive = causal (pre before post), negative = anti-causal.
    pub eligibility: i8,
    /// Thermal lifecycle: bits 0-1 = state, bits 2-7 = reinforcement counter
    pub maturity: u8,
    /// Reserved for future use (neuromodulator sensitivity, branch tag)
    pub _reserved: [u8; 2],
}

impl Synapse {
    /// Create a new synapse respecting Dale's Law.
    ///
    /// `source_flags` is the flags byte of the pre-synaptic neuron.
    /// Weight magnitude is provided; sign is determined by neuron type.
    pub fn new(target: u16, weight_magnitude: u8, delay: u8, source_flags: u8) -> Self {
        let signed_weight = if flags::is_inhibitory(source_flags) {
            -(weight_magnitude.min(127) as i8)
        } else {
            weight_magnitude.min(127) as i8
        };

        Self {
            target,
            weight: signed_weight,
            delay: delay.max(1).min(8),
            eligibility: 0,
            maturity: maturity::encode(ThermalState::Hot, 4), // Start at HOT with some initial counter
            _reserved: [0; 2],
        }
    }

    /// Create a frozen (COLD) synapse for genome-level connectivity.
    pub fn frozen(target: u16, weight: i8, delay: u8) -> Self {
        Self {
            target,
            weight,
            delay: delay.max(1).min(8),
            eligibility: 0,
            maturity: maturity::encode(ThermalState::Cold, 63),
            _reserved: [0; 2],
        }
    }

    /// Current thermal state
    #[inline]
    pub fn thermal_state(&self) -> ThermalState {
        maturity::state(self.maturity)
    }

    /// Whether this synapse is dead and should be pruned
    #[inline]
    pub fn is_dead(&self) -> bool {
        maturity::is_dead(self.maturity)
    }

    /// Increment maturity (toward promotion)
    #[inline]
    pub fn increment_maturity(&mut self) {
        self.maturity = maturity::increment(self.maturity);
    }

    /// Decrement maturity (toward demotion/death)
    #[inline]
    pub fn decrement_maturity(&mut self) {
        self.maturity = maturity::decrement(self.maturity);
    }
}

/// Compressed Sparse Row synapse storage.
///
/// All outgoing synapses for neuron `i` are at indices `row_ptr[i]..row_ptr[i+1]`
/// in the `synapses` array. This gives cache-friendly iteration during spike
/// propagation — all targets of a spiking neuron are contiguous in memory.
pub struct SynapseStore {
    /// Index into `synapses` for each neuron. Length = n_neurons + 1.
    /// `row_ptr[i]` = start index, `row_ptr[i+1]` = end index (exclusive).
    pub row_ptr: Vec<u32>,
    /// All synapses, grouped contiguously by source neuron.
    pub synapses: Vec<Synapse>,
}

impl SynapseStore {
    /// Create empty store for `n` neurons (no connections).
    pub fn empty(n_neurons: u32) -> Self {
        Self {
            row_ptr: vec![0; (n_neurons + 1) as usize],
            synapses: Vec::new(),
        }
    }

    /// Build CSR from a list of (source_neuron, Synapse) pairs.
    ///
    /// The pairs do NOT need to be sorted — this function sorts them internally.
    pub fn from_edges(n_neurons: u32, mut edges: Vec<(u32, Synapse)>) -> Self {
        edges.sort_unstable_by_key(|(src, _)| *src);

        let n = n_neurons as usize;
        let mut row_ptr = vec![0u32; n + 1];

        // Count synapses per neuron
        for (src, _) in &edges {
            let idx = (*src as usize).min(n - 1);
            row_ptr[idx + 1] += 1;
        }

        // Prefix sum
        for i in 1..=n {
            row_ptr[i] += row_ptr[i - 1];
        }

        let synapses: Vec<Synapse> = edges.into_iter().map(|(_, syn)| syn).collect();

        Self { row_ptr, synapses }
    }

    /// Get outgoing synapses for a given source neuron.
    #[inline]
    pub fn outgoing(&self, neuron: u32) -> &[Synapse] {
        let start = self.row_ptr[neuron as usize] as usize;
        let end = self.row_ptr[neuron as usize + 1] as usize;
        &self.synapses[start..end]
    }

    /// Get mutable outgoing synapses for a given source neuron.
    #[inline]
    pub fn outgoing_mut(&mut self, neuron: u32) -> &mut [Synapse] {
        let start = self.row_ptr[neuron as usize] as usize;
        let end = self.row_ptr[neuron as usize + 1] as usize;
        &mut self.synapses[start..end]
    }

    /// Total number of synapses across all neurons.
    #[inline]
    pub fn total_synapses(&self) -> usize {
        self.synapses.len()
    }

    /// Number of neurons this store covers.
    #[inline]
    pub fn n_neurons(&self) -> u32 {
        (self.row_ptr.len().saturating_sub(1)) as u32
    }

    /// Remove dead synapses (maturity == 0x00) and rebuild CSR.
    /// Returns count of pruned synapses.
    pub fn prune_dead(&mut self) -> usize {
        let n = self.n_neurons() as usize;
        let mut new_synapses = Vec::with_capacity(self.synapses.len());
        let mut new_row_ptr = vec![0u32; n + 1];
        let mut pruned = 0usize;

        for i in 0..n {
            let start = self.row_ptr[i] as usize;
            let end = self.row_ptr[i + 1] as usize;

            for syn in &self.synapses[start..end] {
                if syn.is_dead() {
                    pruned += 1;
                } else {
                    new_synapses.push(*syn);
                }
            }
            new_row_ptr[i + 1] = new_synapses.len() as u32;
        }

        self.synapses = new_synapses;
        self.row_ptr = new_row_ptr;
        pruned
    }

    /// Add a synapse from `source` to the given target. Rebuilds the CSR row.
    /// This is expensive — batch additions and rebuild when possible.
    pub fn add_synapse(&mut self, source: u32, syn: Synapse) {
        let idx = source as usize;
        let insert_pos = self.row_ptr[idx + 1] as usize;

        self.synapses.insert(insert_pos, syn);

        // Update row_ptr for all neurons after source
        for ptr in &mut self.row_ptr[(idx + 1)..] {
            *ptr += 1;
        }
    }

    /// Extend the store to accommodate additional neurons (with no synapses).
    ///
    /// Used when dynamically spawning neurons from templates.
    pub fn extend(&mut self, count: usize) {
        let last_ptr = *self.row_ptr.last().unwrap_or(&0);
        for _ in 0..count {
            self.row_ptr.push(last_ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synapse_size() {
        assert_eq!(std::mem::size_of::<Synapse>(), 8);
    }

    #[test]
    fn maturity_lifecycle() {
        // Start at HOT with counter 0
        let mut m = maturity::encode(ThermalState::Hot, 0);
        assert_eq!(maturity::state(m), ThermalState::Hot);
        assert_eq!(maturity::counter(m), 0);

        // Increment to HOT->WARM promotion
        for _ in 0..maturity::HOT_TO_WARM {
            m = maturity::increment(m);
        }
        assert_eq!(maturity::state(m), ThermalState::Warm);
        assert_eq!(maturity::counter(m), 0); // Reset on promotion

        // Increment to WARM->COOL promotion
        for _ in 0..maturity::WARM_TO_COOL {
            m = maturity::increment(m);
        }
        assert_eq!(maturity::state(m), ThermalState::Cool);

        // Increment to COOL->COLD promotion
        for _ in 0..maturity::COOL_TO_COLD {
            m = maturity::increment(m);
        }
        assert_eq!(maturity::state(m), ThermalState::Cold);

        // Cold is frozen — increment has no effect
        let m2 = maturity::increment(m);
        assert_eq!(m2, m);
    }

    #[test]
    fn maturity_death() {
        let m = maturity::encode(ThermalState::Hot, 0);
        let dead = maturity::decrement(m);
        assert!(maturity::is_dead(dead));
    }

    #[test]
    fn maturity_demotion() {
        // Start at COOL with counter 0
        let m = maturity::encode(ThermalState::Cool, 0);
        let demoted = maturity::decrement(m);
        assert_eq!(maturity::state(demoted), ThermalState::Warm);
        // Should start with some counter to avoid immediate further demotion
        assert!(maturity::counter(demoted) > 0);
    }

    #[test]
    fn dale_law_excitatory() {
        let exc_flags = crate::neuron::flags::encode(false, crate::neuron::NeuronProfile::RegularSpiking);
        let syn = Synapse::new(42, 100, 2, exc_flags);
        assert!(syn.weight > 0);
    }

    #[test]
    fn dale_law_inhibitory() {
        let inh_flags = crate::neuron::flags::encode(true, crate::neuron::NeuronProfile::FastSpiking);
        let syn = Synapse::new(42, 100, 2, inh_flags);
        assert!(syn.weight < 0);
    }

    #[test]
    fn csr_basic() {
        let exc_flags = crate::neuron::flags::encode(false, crate::neuron::NeuronProfile::RegularSpiking);
        let edges = vec![
            (0, Synapse::new(1, 50, 1, exc_flags)),
            (0, Synapse::new(2, 30, 1, exc_flags)),
            (1, Synapse::new(0, 40, 1, exc_flags)),
        ];
        let store = SynapseStore::from_edges(3, edges);

        assert_eq!(store.outgoing(0).len(), 2);
        assert_eq!(store.outgoing(1).len(), 1);
        assert_eq!(store.outgoing(2).len(), 0);
        assert_eq!(store.total_synapses(), 3);
    }

    #[test]
    fn csr_prune_dead() {
        let mut store = SynapseStore::empty(2);
        let exc_flags = crate::neuron::flags::encode(false, crate::neuron::NeuronProfile::RegularSpiking);

        // Add some synapses
        store.add_synapse(0, Synapse::new(1, 50, 1, exc_flags));
        let mut dead_syn = Synapse::new(1, 30, 1, exc_flags);
        dead_syn.maturity = 0x00; // Dead
        store.add_synapse(0, dead_syn);

        assert_eq!(store.total_synapses(), 2);
        let pruned = store.prune_dead();
        assert_eq!(pruned, 1);
        assert_eq!(store.total_synapses(), 1);
    }
}
