//! UnifiedSynapse — zone-aware ternary synaptic connections.
//!
//! Extends `SpatialSynapse` with a `DendriticZone` field that determines
//! which compartment of the target neuron receives the signal.
//! Storage is CSR (Compressed Sparse Row) for efficient outgoing iteration.

use ternary_signal::Signal;
use super::zone::DendriticZone;

/// A synaptic connection with dendritic zone targeting.
///
/// Same fields as `SpatialSynapse` plus `zone`. The zone determines which
/// dendritic compartment of the target neuron receives this synapse's current.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnifiedSynapse {
    /// Source neuron index.
    pub source: u32,
    /// Target neuron index.
    pub target: u32,
    /// Which dendritic zone this synapse targets on the receiving neuron.
    pub zone: DendriticZone,
    /// Ternary signal (polarity x magnitude x multiplier).
    pub signal: Signal,
    /// Transmission delay in microseconds.
    pub delay_us: u32,
    /// Synapse maturity (0=nascent, 255=mature). Mature synapses resist modification.
    pub maturity: u8,
    /// Accumulated learning pressure (for weaken-before-flip mastery learning).
    pub pressure: i16,
}

impl UnifiedSynapse {
    /// Create an excitatory synapse targeting a specific zone.
    #[inline]
    pub const fn excitatory(
        source: u32,
        target: u32,
        zone: DendriticZone,
        magnitude: u8,
        delay_us: u32,
    ) -> Self {
        Self {
            source,
            target,
            zone,
            signal: Signal::positive(magnitude),
            delay_us,
            maturity: 0,
            pressure: 0,
        }
    }

    /// Create an inhibitory synapse targeting a specific zone.
    #[inline]
    pub const fn inhibitory(
        source: u32,
        target: u32,
        zone: DendriticZone,
        magnitude: u8,
        delay_us: u32,
    ) -> Self {
        Self {
            source,
            target,
            zone,
            signal: Signal::negative(magnitude),
            delay_us,
            maturity: 0,
            pressure: 0,
        }
    }

    /// Create a dormant synapse.
    #[inline]
    pub const fn dormant(source: u32, target: u32, zone: DendriticZone, delay_us: u32) -> Self {
        Self {
            source,
            target,
            zone,
            signal: Signal::ZERO,
            delay_us,
            maturity: 0,
            pressure: 0,
        }
    }

    /// Create with specific signal.
    #[inline]
    pub const fn with_signal(
        source: u32,
        target: u32,
        zone: DendriticZone,
        signal: Signal,
        delay_us: u32,
    ) -> Self {
        Self {
            source,
            target,
            zone,
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
        self.signal.polarity > 0 && self.signal.magnitude > 0
    }

    /// Is this synapse inhibitory?
    #[inline]
    pub fn is_inhibitory(&self) -> bool {
        self.signal.polarity < 0 && self.signal.magnitude > 0
    }

    /// Is this synapse mature (maturity > 200)?
    #[inline]
    pub fn is_mature(&self) -> bool {
        self.maturity > 200
    }

    /// Synaptic current: signal scaled for membrane effect.
    ///
    /// Uses `signal.as_signed_i32()` scaled by 8 for membrane-level effect.
    /// Same scaling as `SpatialSynapse::current()`.
    #[inline]
    pub fn current(&self) -> i16 {
        (self.signal.as_signed_i32() * 8) as i16
    }

    /// Strengthen this synapse (increase magnitude).
    #[inline]
    pub fn strengthen(&mut self, amount: u8) {
        self.signal.magnitude = self.signal.magnitude.saturating_add(amount);
    }

    /// Weaken this synapse (decrease magnitude). Zero magnitude → dormant.
    #[inline]
    pub fn weaken(&mut self, amount: u8) {
        self.signal.magnitude = self.signal.magnitude.saturating_sub(amount);
        if self.signal.magnitude == 0 {
            self.signal.polarity = 0; // dormant
        }
    }

    /// Increase maturity.
    #[inline]
    pub fn mature(&mut self, amount: u8) {
        self.maturity = self.maturity.saturating_add(amount);
    }

    /// Accumulate learning pressure.
    #[inline]
    pub fn accumulate_pressure(&mut self, delta: i16) {
        self.pressure = self.pressure.saturating_add(delta);
    }

    /// Clear learning pressure.
    #[inline]
    pub fn clear_pressure(&mut self) {
        self.pressure = 0;
    }
}

/// CSR (Compressed Sparse Row) storage for zone-aware synapses.
///
/// Same structure as `SpatialSynapseStore` but holds `UnifiedSynapse`.
/// Indexed by source neuron for efficient outgoing iteration.
pub struct UnifiedSynapseStore {
    /// All synapses, grouped by source neuron.
    synapses: Vec<UnifiedSynapse>,
    /// CSR row pointers: `offsets[i]..offsets[i+1]` gives synapse range for neuron i.
    offsets: Vec<usize>,
    /// Whether the CSR index needs rebuilding.
    dirty: bool,
}

impl UnifiedSynapseStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            synapses: Vec::new(),
            offsets: vec![0],
            dirty: false,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(synapse_capacity: usize, neuron_count: usize) -> Self {
        let mut offsets = Vec::with_capacity(neuron_count + 1);
        offsets.push(0);
        Self {
            synapses: Vec::with_capacity(synapse_capacity),
            offsets,
            dirty: false,
        }
    }

    /// Add a synapse. Marks index as dirty.
    pub fn add(&mut self, synapse: UnifiedSynapse) {
        self.synapses.push(synapse);
        self.dirty = true;
    }

    /// Rebuild the CSR index. Must be called after adding synapses.
    pub fn rebuild_index(&mut self, neuron_count: u32) {
        // Sort by source for CSR
        self.synapses.sort_by_key(|s| s.source);

        let n = neuron_count as usize;
        self.offsets = vec![0usize; n + 1];

        // Count synapses per source
        for s in &self.synapses {
            if (s.source as usize) < n {
                self.offsets[s.source as usize + 1] += 1;
            }
        }

        // Prefix sum
        for i in 1..=n {
            self.offsets[i] += self.offsets[i - 1];
        }

        self.dirty = false;
    }

    /// Get outgoing synapses for a source neuron.
    ///
    /// Returns empty slice if index is dirty or neuron has no outgoing synapses.
    pub fn outgoing(&self, source: u32) -> &[UnifiedSynapse] {
        if self.dirty {
            return &[];
        }
        let idx = source as usize;
        if idx + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[idx];
        let end = self.offsets[idx + 1];
        &self.synapses[start..end]
    }

    /// Mutable access to outgoing synapses for a source neuron.
    pub fn outgoing_mut(&mut self, source: u32) -> &mut [UnifiedSynapse] {
        if self.dirty {
            return &mut [];
        }
        let idx = source as usize;
        if idx + 1 >= self.offsets.len() {
            return &mut [];
        }
        let start = self.offsets[idx];
        let end = self.offsets[idx + 1];
        &mut self.synapses[start..end]
    }

    /// Total number of synapses.
    pub fn len(&self) -> usize {
        self.synapses.len()
    }

    /// Is the store empty?
    pub fn is_empty(&self) -> bool {
        self.synapses.is_empty()
    }

    /// Count of active (non-dormant) synapses.
    pub fn count_active(&self) -> usize {
        self.synapses.iter().filter(|s| s.is_active()).count()
    }

    /// Count of dormant synapses.
    pub fn count_dormant(&self) -> usize {
        self.synapses.iter().filter(|s| s.is_dormant()).count()
    }

    /// Iterate all synapses.
    pub fn iter(&self) -> impl Iterator<Item = &UnifiedSynapse> {
        self.synapses.iter()
    }

    /// Mutable iteration over all synapses.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut UnifiedSynapse> {
        self.synapses.iter_mut()
    }

    /// Remove dormant synapses (pruning). Marks index as dirty.
    pub fn prune_dormant(&mut self) -> usize {
        let before = self.synapses.len();
        self.synapses.retain(|s| s.is_active());
        let pruned = before - self.synapses.len();
        if pruned > 0 {
            self.dirty = true;
        }
        pruned
    }
}

impl Default for UnifiedSynapseStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excitatory_synapse() {
        let s = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        assert!(s.is_active());
        assert!(s.is_excitatory());
        assert!(!s.is_inhibitory());
        assert_eq!(s.zone, DendriticZone::Feedforward);
        assert_eq!(s.current(), 100 * 8); // magnitude * 8 (multiplier=1)
    }

    #[test]
    fn inhibitory_synapse() {
        let s = UnifiedSynapse::inhibitory(0, 1, DendriticZone::Feedback, 50, 300);
        assert!(s.is_active());
        assert!(s.is_inhibitory());
        assert_eq!(s.zone, DendriticZone::Feedback);
        assert_eq!(s.current(), -50 * 8);
    }

    #[test]
    fn dormant_synapse() {
        let s = UnifiedSynapse::dormant(0, 1, DendriticZone::Context, 1000);
        assert!(s.is_dormant());
        assert!(!s.is_active());
        assert_eq!(s.current(), 0);
    }

    #[test]
    fn weaken_to_dormancy() {
        let mut s = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 10, 500);
        s.weaken(10);
        assert!(s.is_dormant());
    }

    #[test]
    fn csr_store() {
        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.add(UnifiedSynapse::excitatory(0, 2, DendriticZone::Context, 80, 300));
        store.add(UnifiedSynapse::excitatory(1, 3, DendriticZone::Feedforward, 60, 200));
        store.rebuild_index(4);

        assert_eq!(store.outgoing(0).len(), 2);
        assert_eq!(store.outgoing(1).len(), 1);
        assert_eq!(store.outgoing(2).len(), 0);
        assert_eq!(store.outgoing(3).len(), 0);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn prune_dormant_synapses() {
        let mut store = UnifiedSynapseStore::new();
        store.add(UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500));
        store.add(UnifiedSynapse::dormant(0, 2, DendriticZone::Context, 300));
        store.add(UnifiedSynapse::excitatory(1, 3, DendriticZone::Feedforward, 60, 200));

        let pruned = store.prune_dormant();
        assert_eq!(pruned, 1);
        assert_eq!(store.len(), 2);
        assert_eq!(store.count_active(), 2);
    }

    #[test]
    fn maturity() {
        let mut s = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        assert!(!s.is_mature());
        s.mature(201);
        assert!(s.is_mature());
    }

    #[test]
    fn pressure_accumulation() {
        let mut s = UnifiedSynapse::excitatory(0, 1, DendriticZone::Feedforward, 100, 500);
        s.accumulate_pressure(10);
        s.accumulate_pressure(-3);
        assert_eq!(s.pressure, 7);
        s.clear_pressure();
        assert_eq!(s.pressure, 0);
    }
}
