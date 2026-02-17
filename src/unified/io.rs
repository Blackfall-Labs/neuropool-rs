//! Unified NeuronIO — external interface for specialized neurons during cascade.
//!
//! When a memory neuron fires, the cascade engine calls into the IO trait
//! to query or write to external data banks. Sensory and motor neurons
//! similarly route through this trait.
//!
//! The host environment (e.g., sentinel kernel) implements this trait.

/// External I/O interface for unified cascade execution.
///
/// Called by the cascade engine when specialized neurons fire or receive input.
/// All operations are synchronous within the cascade step.
pub trait UnifiedNeuronIO {
    /// Read sensory input for a sensory neuron.
    ///
    /// `channel`: which sensory channel (from interface.target)
    /// `modality`: sensory modality (from interface.modality)
    /// Returns: Q8.8 current to inject as feedforward input.
    fn read_sensory(&self, channel: u16, modality: u8) -> i16;

    /// Write motor output when a motor neuron fires.
    ///
    /// `channel`: which motor channel (from interface.target)
    /// `modality`: motor modality (from interface.modality)
    /// `magnitude`: spike-scaled output value
    fn write_motor(&mut self, channel: u16, modality: u8, magnitude: i16);

    /// Query a memory bank when a memory neuron fires.
    ///
    /// `bank_id`: which bank (from interface.target)
    /// `query`: local activity pattern from neighboring membrane potentials
    /// Returns: Q8.8 similarity-weighted current to inject
    fn memory_query(&mut self, bank_id: u16, query: &[i16]) -> i16;

    /// Write to a memory bank when a memory neuron fires with high energy.
    ///
    /// `bank_id`: which bank (from interface.target)
    /// `pattern`: local activity pattern to store
    fn memory_write(&mut self, bank_id: u16, pattern: &[i16]);

    /// Read a neuromodulator chemical level.
    ///
    /// `chemical_id`: which chemical (0=DA, 1=5HT, 2=NE, 3=GABA, etc.)
    /// Returns: 0-255 level
    fn read_chemical(&self, chemical_id: u8) -> u8;
}

/// No-op IO for pools without external connections.
///
/// All reads return 0, all writes are silently dropped.
pub struct NullUnifiedIO;

impl UnifiedNeuronIO for NullUnifiedIO {
    #[inline]
    fn read_sensory(&self, _channel: u16, _modality: u8) -> i16 { 0 }
    #[inline]
    fn write_motor(&mut self, _channel: u16, _modality: u8, _magnitude: i16) {}
    #[inline]
    fn memory_query(&mut self, _bank_id: u16, _query: &[i16]) -> i16 { 0 }
    #[inline]
    fn memory_write(&mut self, _bank_id: u16, _pattern: &[i16]) {}
    #[inline]
    fn read_chemical(&self, _chemical_id: u8) -> u8 { 0 }
}
