//! Binding configuration for specialized neurons.
//!
//! Each non-Computational neuron has a binding_slot (1 byte) that indexes into
//! a BindingTable. The table stores compact 4-byte BindingConfig entries that
//! parameterize type-specific behavior (which field to read, which channel to
//! write, which bank to query, etc.).

/// Packed binding configuration for a specialized neuron.
///
/// Interpretation depends on the neuron's NeuronType:
///
/// | Type          | target       | param_a        | param_b       | flags    |
/// |---------------|-------------|----------------|---------------|----------|
/// | Sensory       | field_id    | offset_lo      | offset_hi     | reserved |
/// | Motor         | channel_id  | scale          | reserved      | reserved |
/// | MemoryReader  | bank_slot   | query_dim      | top_k         | reserved |
/// | MemoryMatcher | bank_slot   | threshold      | reserved      | reserved |
/// | Gate          | chemical_id | sensitivity    | reserved      | reserved |
/// | Oscillator    | period      | amplitude      | phase_offset  | reserved |
/// | Relay         | (unused)    | (unused)       | (unused)      | reserved |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct BindingConfig {
    /// Primary target: field_id, channel_id, bank_slot, chemical_id, or period.
    pub target: u8,
    /// First parameter: offset_lo, scale, query_dim, threshold, sensitivity, or amplitude.
    pub param_a: u8,
    /// Second parameter: offset_hi, top_k, phase_offset, etc.
    pub param_b: u8,
    /// Reserved for future use.
    pub flags: u8,
}

impl BindingConfig {
    /// Sensory neuron binding: reads from `field_id` at `offset`.
    pub fn sensory(field_id: u8, offset: u16) -> Self {
        Self {
            target: field_id,
            param_a: offset as u8,
            param_b: (offset >> 8) as u8,
            flags: 0,
        }
    }

    /// Motor neuron binding: writes to `channel_id` with `scale` factor.
    pub fn motor(channel_id: u8, scale: u8) -> Self {
        Self {
            target: channel_id,
            param_a: scale,
            param_b: 0,
            flags: 0,
        }
    }

    /// MemoryReader binding: queries `bank_slot` with `query_dim` dimensions, returns `top_k`.
    pub fn memory_reader(bank_slot: u8, query_dim: u8, top_k: u8) -> Self {
        Self {
            target: bank_slot,
            param_a: query_dim,
            param_b: top_k,
            flags: 0,
        }
    }

    /// MemoryMatcher binding: compares patterns against `bank_slot` with `threshold`.
    pub fn memory_matcher(bank_slot: u8, threshold: u8) -> Self {
        Self {
            target: bank_slot,
            param_a: threshold,
            param_b: 0,
            flags: 0,
        }
    }

    /// Gate neuron binding: modulated by `chemical_id` with `sensitivity`.
    pub fn gate(chemical_id: u8, sensitivity: u8) -> Self {
        Self {
            target: chemical_id,
            param_a: sensitivity,
            param_b: 0,
            flags: 0,
        }
    }

    /// Oscillator binding: fires with `period` ticks, `amplitude` depolarization, `phase_offset`.
    pub fn oscillator(period: u8, amplitude: u8, phase_offset: u8) -> Self {
        Self {
            target: period,
            param_a: amplitude,
            param_b: phase_offset,
            flags: 0,
        }
    }

    /// Offset for sensory binding (reconstructed from param_a + param_b).
    #[inline]
    pub fn sensory_offset(&self) -> u16 {
        self.param_a as u16 | ((self.param_b as u16) << 8)
    }
}

/// Sparse table of binding configurations.
///
/// Slot indices 1-255 are valid (0 means "no binding"). The table only stores
/// entries that have been allocated, so a pool with 20 specialized neurons uses
/// ~80 bytes for the table (20 entries * 4 bytes each).
pub struct BindingTable {
    /// Indexed by slot number (1-based). entries[0] is unused but present for
    /// direct indexing without offset math.
    entries: Vec<BindingConfig>,
    /// Next available slot index.
    next_slot: u8,
}

impl BindingTable {
    /// Create an empty binding table.
    pub fn new() -> Self {
        Self {
            entries: vec![BindingConfig::default()], // slot 0 = unused sentinel
            next_slot: 1,
        }
    }

    /// Allocate a new binding slot and return its index.
    ///
    /// Returns `None` if the table is full (255 entries max).
    pub fn add(&mut self, config: BindingConfig) -> Option<u8> {
        if self.next_slot == 0 {
            return None; // wrapped around — table full
        }
        let slot = self.next_slot;
        if self.entries.len() <= slot as usize {
            self.entries.resize(slot as usize + 1, BindingConfig::default());
        }
        self.entries[slot as usize] = config;
        self.next_slot = self.next_slot.checked_add(1).unwrap_or(0);
        Some(slot)
    }

    /// Look up a binding config by slot index.
    ///
    /// Returns `None` for slot 0 or out-of-range slots.
    #[inline]
    pub fn get(&self, slot: u8) -> Option<&BindingConfig> {
        if slot == 0 || slot as usize >= self.entries.len() {
            None
        } else {
            Some(&self.entries[slot as usize])
        }
    }

    /// Number of allocated entries (not counting the sentinel at slot 0).
    pub fn len(&self) -> usize {
        if self.entries.len() <= 1 { 0 } else { self.entries.len() - 1 }
    }

    /// Whether the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All entries as a slice (including sentinel at index 0).
    pub fn entries(&self) -> &[BindingConfig] {
        &self.entries
    }

    /// Rebuild from a raw entries slice (including sentinel at index 0).
    pub fn from_entries(entries: Vec<BindingConfig>) -> Self {
        let next_slot = if entries.len() >= 256 { 0 } else { entries.len() as u8 };
        Self { entries, next_slot }
    }
}

impl Default for BindingTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_table_add_get() {
        let mut table = BindingTable::new();
        assert!(table.is_empty());

        let cfg = BindingConfig::sensory(3, 128);
        let slot = table.add(cfg).unwrap();
        assert_eq!(slot, 1);
        assert_eq!(table.len(), 1);

        let retrieved = table.get(slot).unwrap();
        assert_eq!(retrieved.target, 3);
        assert_eq!(retrieved.sensory_offset(), 128);

        // Slot 0 returns None
        assert!(table.get(0).is_none());
    }

    #[test]
    fn binding_config_constructors() {
        let s = BindingConfig::sensory(5, 300);
        assert_eq!(s.target, 5);
        assert_eq!(s.sensory_offset(), 300);

        let m = BindingConfig::motor(2, 128);
        assert_eq!(m.target, 2);
        assert_eq!(m.param_a, 128);

        let mr = BindingConfig::memory_reader(1, 8, 3);
        assert_eq!(mr.target, 1);
        assert_eq!(mr.param_a, 8);
        assert_eq!(mr.param_b, 3);

        let mm = BindingConfig::memory_matcher(0, 200);
        assert_eq!(mm.target, 0);
        assert_eq!(mm.param_a, 200);

        let g = BindingConfig::gate(4, 100);
        assert_eq!(g.target, 4);
        assert_eq!(g.param_a, 100);

        let o = BindingConfig::oscillator(20, 50, 10);
        assert_eq!(o.target, 20);
        assert_eq!(o.param_a, 50);
        assert_eq!(o.param_b, 10);
    }
}
