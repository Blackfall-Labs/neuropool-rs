//! External I/O trait for specialized neurons.
//!
//! NeuronIO provides an abstraction layer so that neuropool-rs doesn't depend
//! on external crates (databank, astromind, etc.). The host environment
//! implements this trait and passes it to `tick()`.

/// External I/O interface for specialized neuron types.
///
/// Implemented by the host environment (e.g., astromind-v3's RegionIO) to
/// provide sensory input, motor output, memory access, and chemical readings.
pub trait NeuronIO {
    /// Read a sensory value from a field.
    ///
    /// `field_id`: which substrate field to read
    /// `offset`: byte offset into that field
    /// Returns: Q8.8 current to add to membrane potential
    fn read_sensory(&self, field_id: u8, offset: u16) -> i16;

    /// Write a motor output on spike.
    ///
    /// `channel_id`: which output channel
    /// `magnitude`: spike-scaled output value (Q8.8)
    fn write_motor(&mut self, channel_id: u8, magnitude: i16);

    /// Query a memory bank (MemoryReader post-spike).
    ///
    /// `bank_slot`: which bank to query
    /// `query`: local pattern extracted from neighboring membrane potentials
    /// `top_k`: number of results to consider
    /// Returns: Q8.8 current to inject (similarity-weighted)
    fn memory_query(&mut self, bank_slot: u8, query: &[i16], top_k: u8) -> i16;

    /// Pattern match against a memory bank (MemoryMatcher).
    ///
    /// `bank_slot`: which bank to compare against
    /// `pattern`: local activity pattern
    /// Returns: Q8.8 boost to membrane if pattern matches (0 if no match)
    fn memory_match(&self, bank_slot: u8, pattern: &[i16]) -> i16;

    /// Read a neuromodulator chemical level.
    ///
    /// `chemical_id`: which chemical (0=DA, 1=5HT, 2=NE, 3=GABA, etc.)
    /// Returns: 0-255 level
    fn read_chemical(&self, chemical_id: u8) -> u8;
}

/// No-op NeuronIO for pools without specialized neurons.
///
/// All reads return 0, all writes are silently dropped.
/// Used by `tick_simple()` for backward compatibility.
pub struct NullIO;

impl NeuronIO for NullIO {
    #[inline]
    fn read_sensory(&self, _field_id: u8, _offset: u16) -> i16 { 0 }
    #[inline]
    fn write_motor(&mut self, _channel_id: u8, _magnitude: i16) {}
    #[inline]
    fn memory_query(&mut self, _bank_slot: u8, _query: &[i16], _top_k: u8) -> i16 { 0 }
    #[inline]
    fn memory_match(&self, _bank_slot: u8, _pattern: &[i16]) -> i16 { 0 }
    #[inline]
    fn read_chemical(&self, _chemical_id: u8) -> u8 { 0 }
}
