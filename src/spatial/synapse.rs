//! SpatialSynapse — ternary-signal synaptic connections.
//!
//! Unlike voxel synapses (i8 weights), spatial synapses use `ternary_signal::Signal`
//! which provides genuine polarity (inhibitory/neutral/excitatory) × magnitude.
//!
//! ## Polarity Meanings
//!
//! | Polarity | Biological | Effect |
//! |----------|------------|--------|
//! | +1 | Glutamatergic (excitatory) | Depolarizes target |
//! | 0 | Silent/Dormant | Synapse exists but inactive |
//! | -1 | GABAergic (inhibitory) | Hyperpolarizes target |

use ternary_signal::Signal;

/// A synaptic connection using ternary signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialSynapse {
    /// Source neuron index
    pub source: u32,
    /// Target neuron index
    pub target: u32,
    /// Ternary signal (polarity × magnitude)
    pub signal: Signal,
    /// Transmission delay in microseconds (computed from distance + tissue)
    pub delay_us: u32,
    /// Synapse maturity (0=nascent, 255=mature)
    /// Mature synapses resist modification
    pub maturity: u8,
    /// Accumulated learning pressure (for weaken-before-flip)
    pub pressure: i16,
}

impl SpatialSynapse {
    /// Create a new synapse with positive signal.
    #[inline]
    pub const fn excitatory(source: u32, target: u32, magnitude: u8, delay_us: u32) -> Self {
        Self {
            source,
            target,
            signal: Signal::positive(magnitude),
            delay_us,
            maturity: 0,
            pressure: 0,
        }
    }

    /// Create a new synapse with negative signal.
    #[inline]
    pub const fn inhibitory(source: u32, target: u32, magnitude: u8, delay_us: u32) -> Self {
        Self {
            source,
            target,
            signal: Signal::negative(magnitude),
            delay_us,
            maturity: 0,
            pressure: 0,
        }
    }

    /// Create a dormant synapse (polarity 0).
    #[inline]
    pub const fn dormant(source: u32, target: u32, delay_us: u32) -> Self {
        Self {
            source,
            target,
            signal: Signal::ZERO,
            delay_us,
            maturity: 0,
            pressure: 0,
        }
    }

    /// Create with specific signal.
    #[inline]
    pub const fn with_signal(source: u32, target: u32, signal: Signal, delay_us: u32) -> Self {
        Self {
            source,
            target,
            signal,
            delay_us,
            maturity: 0,
            pressure: 0,
        }
    }

    /// Is this synapse active (non-zero polarity and magnitude)?
    #[inline]
    pub fn is_active(&self) -> bool {
        self.signal.is_active()
    }

    /// Is this synapse dormant?
    #[inline]
    pub fn is_dormant(&self) -> bool {
        !self.signal.is_active()
    }

    /// Is this synapse excitatory?
    #[inline]
    pub fn is_excitatory(&self) -> bool {
        self.signal.is_positive()
    }

    /// Is this synapse inhibitory?
    #[inline]
    pub fn is_inhibitory(&self) -> bool {
        self.signal.is_negative()
    }

    /// Is this synapse mature (resists modification)?
    #[inline]
    pub const fn is_mature(&self) -> bool {
        self.maturity > 200
    }

    /// Get the current as i16 (for integration).
    ///
    /// Scales the signal magnitude by 8 for biologically appropriate membrane effects.
    /// With this scaling:
    /// - magnitude 200 → 1600 current (can fire alone if threshold delta is 1500)
    /// - magnitude 100 → 800 current (needs ~2 converging)
    /// - magnitude 50  → 400 current (needs ~4 converging)
    #[inline]
    pub fn current(&self) -> i16 {
        (self.signal.as_signed_i32() * 8) as i16
    }

    /// Strengthen the synapse (increase magnitude, keep polarity).
    #[inline]
    pub fn strengthen(&mut self, amount: u8) {
        self.signal.magnitude = self.signal.magnitude.saturating_add(amount);
    }

    /// Weaken the synapse (decrease magnitude).
    ///
    /// If magnitude reaches 0, the synapse becomes dormant.
    #[inline]
    pub fn weaken(&mut self, amount: u8) {
        self.signal.magnitude = self.signal.magnitude.saturating_sub(amount);
        if self.signal.magnitude == 0 {
            self.signal.polarity = 0;
        }
    }

    /// Apply decay to magnitude.
    #[inline]
    pub fn decay(&mut self, retention: f32) {
        self.signal.decay(retention);
    }

    /// Increase maturity (makes synapse more resistant to change).
    #[inline]
    pub fn mature(&mut self, amount: u8) {
        self.maturity = self.maturity.saturating_add(amount);
    }

    /// Accumulate learning pressure.
    #[inline]
    pub fn accumulate_pressure(&mut self, delta: i16) {
        self.pressure = self.pressure.saturating_add(delta);
    }

    /// Clear accumulated pressure.
    #[inline]
    pub fn clear_pressure(&mut self) {
        self.pressure = 0;
    }
}

/// Collection of spatial synapses with CSR-like access.
#[derive(Clone, Debug, Default)]
pub struct SpatialSynapseStore {
    /// All synapses, sorted by source for efficient iteration.
    synapses: Vec<SpatialSynapse>,
    /// Index into synapses: row_ptr[i] is the first synapse from neuron i.
    row_ptr: Vec<usize>,
}

impl SpatialSynapseStore {
    /// Create an empty synapse store for n neurons.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            synapses: Vec::new(),
            row_ptr: vec![0; neuron_count + 1],
        }
    }

    /// Number of synapses.
    #[inline]
    pub fn len(&self) -> usize {
        self.synapses.len()
    }

    /// Is the store empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.synapses.is_empty()
    }

    /// Add a synapse. Call `rebuild_index()` after adding multiple synapses.
    pub fn add(&mut self, synapse: SpatialSynapse) {
        self.synapses.push(synapse);
    }

    /// Rebuild the CSR index after adding synapses.
    pub fn rebuild_index(&mut self, neuron_count: usize) {
        // Sort by source
        self.synapses.sort_by_key(|s| s.source);

        // Build row pointers
        self.row_ptr = vec![0; neuron_count + 1];
        for syn in &self.synapses {
            self.row_ptr[syn.source as usize + 1] += 1;
        }
        for i in 1..self.row_ptr.len() {
            self.row_ptr[i] += self.row_ptr[i - 1];
        }
    }

    /// Get outgoing synapses from a neuron.
    #[inline]
    pub fn outgoing(&self, neuron: u32) -> &[SpatialSynapse] {
        let start = self.row_ptr.get(neuron as usize).copied().unwrap_or(0);
        let end = self.row_ptr.get(neuron as usize + 1).copied().unwrap_or(0);
        &self.synapses[start..end]
    }

    /// Get mutable outgoing synapses from a neuron.
    #[inline]
    pub fn outgoing_mut(&mut self, neuron: u32) -> &mut [SpatialSynapse] {
        let start = self.row_ptr.get(neuron as usize).copied().unwrap_or(0);
        let end = self.row_ptr.get(neuron as usize + 1).copied().unwrap_or(0);
        &mut self.synapses[start..end]
    }

    /// Iterate all synapses.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &SpatialSynapse> {
        self.synapses.iter()
    }

    /// Iterate all synapses mutably.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SpatialSynapse> {
        self.synapses.iter_mut()
    }

    /// Remove dormant synapses (soft pruning → hard pruning).
    pub fn prune_dormant(&mut self, neuron_count: usize) {
        self.synapses.retain(|s| s.is_active());
        self.rebuild_index(neuron_count);
    }

    /// Count active synapses.
    pub fn count_active(&self) -> usize {
        self.synapses.iter().filter(|s| s.is_active()).count()
    }

    /// Count dormant synapses.
    pub fn count_dormant(&self) -> usize {
        self.synapses.iter().filter(|s| s.is_dormant()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_creation() {
        let exc = SpatialSynapse::excitatory(0, 1, 100, 1000);
        assert!(exc.is_excitatory());
        assert!(exc.is_active());

        let inh = SpatialSynapse::inhibitory(0, 1, 50, 1000);
        assert!(inh.is_inhibitory());

        let dorm = SpatialSynapse::dormant(0, 1, 1000);
        assert!(dorm.is_dormant());
    }

    #[test]
    fn test_weaken_to_dormant() {
        let mut syn = SpatialSynapse::excitatory(0, 1, 10, 1000);
        syn.weaken(10);
        assert!(syn.is_dormant());
    }

    #[test]
    fn test_synapse_store() {
        let mut store = SpatialSynapseStore::new(3);
        store.add(SpatialSynapse::excitatory(0, 1, 100, 1000));
        store.add(SpatialSynapse::excitatory(0, 2, 100, 1000));
        store.add(SpatialSynapse::excitatory(1, 2, 100, 1000));
        store.rebuild_index(3);

        assert_eq!(store.outgoing(0).len(), 2);
        assert_eq!(store.outgoing(1).len(), 1);
        assert_eq!(store.outgoing(2).len(), 0);
    }

    #[test]
    fn test_synapse_current() {
        // current() applies 8x scaling for biologically appropriate membrane effects
        let exc = SpatialSynapse::excitatory(0, 1, 100, 1000);
        assert_eq!(exc.current(), 800); // 100 * 8

        let inh = SpatialSynapse::inhibitory(0, 1, 50, 1000);
        assert_eq!(inh.current(), -400); // -50 * 8

        let dorm = SpatialSynapse::dormant(0, 1, 1000);
        assert_eq!(dorm.current(), 0);
    }

    #[test]
    fn test_synapse_strengthen_weaken() {
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);

        syn.strengthen(50);
        assert_eq!(syn.signal.magnitude, 150);

        syn.weaken(30);
        assert_eq!(syn.signal.magnitude, 120);

        // Weaken to zero → goes dormant
        syn.weaken(200);
        assert!(syn.is_dormant());
        assert_eq!(syn.signal.polarity, 0);
    }

    #[test]
    fn test_synapse_maturity() {
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);
        assert!(!syn.is_mature());

        syn.mature(250);
        assert!(syn.is_mature());
    }

    #[test]
    fn test_synapse_decay() {
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);
        syn.decay(0.5);
        assert_eq!(syn.signal.magnitude, 50);

        // Decay to dormant
        for _ in 0..10 {
            syn.decay(0.5);
        }
        assert!(syn.is_dormant());
    }

    #[test]
    fn test_synapse_pressure() {
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);
        assert_eq!(syn.pressure, 0);

        syn.accumulate_pressure(50);
        assert_eq!(syn.pressure, 50);

        syn.accumulate_pressure(-30);
        assert_eq!(syn.pressure, 20);

        syn.clear_pressure();
        assert_eq!(syn.pressure, 0);
    }

    #[test]
    fn test_synapse_store_prune() {
        let mut store = SpatialSynapseStore::new(3);
        store.add(SpatialSynapse::excitatory(0, 1, 100, 1000));
        store.add(SpatialSynapse::dormant(0, 2, 1000)); // dormant
        store.add(SpatialSynapse::excitatory(1, 2, 100, 1000));
        store.rebuild_index(3);

        assert_eq!(store.len(), 3);
        assert_eq!(store.count_active(), 2);
        assert_eq!(store.count_dormant(), 1);

        store.prune_dormant(3);
        assert_eq!(store.len(), 2);
        assert_eq!(store.count_dormant(), 0);
    }

    #[test]
    fn test_synapse_store_empty_neuron() {
        let store = SpatialSynapseStore::new(5);
        // All neurons should have 0 outgoing synapses
        for i in 0..5 {
            assert_eq!(store.outgoing(i).len(), 0);
        }
    }

    #[test]
    fn test_signal_with_custom() {
        let sig = Signal::new(-1, 200);
        let syn = SpatialSynapse::with_signal(0, 1, sig, 500);
        assert!(syn.is_inhibitory());
        assert_eq!(syn.current(), -1600); // -200 * 8 scaling
    }
}
