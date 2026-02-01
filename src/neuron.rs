//! Neuron data structures — SoA (Structure of Arrays) layout for cache performance.
//!
//! Each neuron field is stored as a separate contiguous array so that the tick
//! hot path iterates over dense memory. A single neuron consumes 8 bytes across
//! all arrays combined.

/// Neuron firing profile, encoded in bits 1-2 of the flags byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NeuronProfile {
    /// Standard excitatory cortical neuron — moderate leak, regular firing
    RegularSpiking = 0b00,
    /// Fast inhibitory interneuron — high leak, rapid response
    FastSpiking = 0b01,
    /// Burst-capable neuron — low leak, fires in clusters
    IntrinsicBursting = 0b10,
    /// Reserved for Izhikevich upgrade
    Reserved = 0b11,
}

impl NeuronProfile {
    /// Decode from flags byte (bits 1-2)
    #[inline]
    pub fn from_flags(flags: u8) -> Self {
        match (flags >> 1) & 0b11 {
            0b00 => Self::RegularSpiking,
            0b01 => Self::FastSpiking,
            0b10 => Self::IntrinsicBursting,
            _ => Self::Reserved,
        }
    }

    /// Default leak rate for this profile (higher = faster decay toward resting)
    #[inline]
    pub fn default_leak(self) -> u8 {
        match self {
            Self::RegularSpiking => 4,    // moderate leak
            Self::FastSpiking => 8,       // fast leak — quick return to rest
            Self::IntrinsicBursting => 2, // slow leak — sustains depolarization
            Self::Reserved => 4,
        }
    }
}

/// Flags byte encoding:
/// - Bit 0: 0 = excitatory, 1 = inhibitory
/// - Bits 1-2: NeuronProfile
/// - Bits 3-7: reserved
pub mod flags {
    pub const INHIBITORY_BIT: u8 = 0x01;

    #[inline]
    pub fn is_inhibitory(f: u8) -> bool {
        f & INHIBITORY_BIT != 0
    }

    #[inline]
    pub fn is_excitatory(f: u8) -> bool {
        !is_inhibitory(f)
    }

    #[inline]
    pub fn encode(inhibitory: bool, profile: super::NeuronProfile) -> u8 {
        let mut f = 0u8;
        if inhibitory {
            f |= INHIBITORY_BIT;
        }
        f |= (profile as u8) << 1;
        f
    }
}

/// SoA neuron storage — each field is a separate contiguous array.
///
/// For N neurons, each Vec has exactly N elements. Total memory per neuron: 8 bytes.
///
/// Layout per neuron across arrays:
/// - membrane:          2 bytes (i16, Q8.8 fixed-point)
/// - threshold:         2 bytes (i16, Q8.8 adaptive threshold)
/// - leak:              1 byte  (u8, leak rate)
/// - refract_remaining: 1 byte  (u8, refractory countdown)
/// - flags:             1 byte  (u8, excitatory/inhibitory + profile)
/// - trace:             1 byte  (i8, post-synaptic eligibility trace)
pub struct NeuronArrays {
    /// Q8.8 fixed-point membrane potential. Resting ~ -17920 (-70 * 256).
    pub membrane: Vec<i16>,
    /// Q8.8 adaptive threshold. Baseline ~ -14080 (-55 * 256).
    /// Adjusted by homeostatic plasticity.
    pub threshold: Vec<i16>,
    /// Leak rate — higher values mean faster return to resting potential.
    /// Applied as: membrane += (resting - membrane) >> (8 - leak.min(7))
    pub leak: Vec<u8>,
    /// Refractory ticks remaining. Neuron cannot fire while > 0.
    pub refract_remaining: Vec<u8>,
    /// Bit flags: bit 0 = inhibitory, bits 1-2 = NeuronProfile.
    pub flags: Vec<u8>,
    /// Post-synaptic eligibility trace. Bumped on spike, decays toward 0.
    /// Used for STDP credit assignment.
    pub trace: Vec<i8>,
    /// Did this neuron spike this tick? Cleared at start of each tick.
    pub spike_out: Vec<bool>,
}

impl NeuronArrays {
    /// Allocate arrays for `n` neurons, all initialized to resting state.
    ///
    /// Dale's Law: first `n_excitatory` neurons are excitatory (flag bit 0 = 0),
    /// remaining are inhibitory (flag bit 0 = 1).
    pub fn new(n: u32, n_excitatory: u32, resting: i16, threshold: i16) -> Self {
        let n = n as usize;
        let n_exc = n_excitatory as usize;

        let mut flags_vec = vec![0u8; n];
        for i in 0..n {
            let inhibitory = i >= n_exc;
            let profile = if inhibitory {
                NeuronProfile::FastSpiking
            } else {
                NeuronProfile::RegularSpiking
            };
            flags_vec[i] = flags::encode(inhibitory, profile);
        }

        let mut leak_vec = vec![0u8; n];
        for i in 0..n {
            leak_vec[i] = NeuronProfile::from_flags(flags_vec[i]).default_leak();
        }

        Self {
            membrane: vec![resting; n],
            threshold: vec![threshold; n],
            leak: leak_vec,
            refract_remaining: vec![0; n],
            flags: flags_vec,
            trace: vec![0i8; n],
            spike_out: vec![false; n],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.membrane.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.membrane.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_arrays_dale_law() {
        let arr = NeuronArrays::new(100, 80, -17920, -14080);
        assert_eq!(arr.len(), 100);

        // First 80 are excitatory
        for i in 0..80 {
            assert!(flags::is_excitatory(arr.flags[i]), "neuron {i} should be excitatory");
        }
        // Last 20 are inhibitory
        for i in 80..100 {
            assert!(flags::is_inhibitory(arr.flags[i]), "neuron {i} should be inhibitory");
        }
    }

    #[test]
    fn profile_encoding() {
        let f = flags::encode(false, NeuronProfile::IntrinsicBursting);
        assert!(flags::is_excitatory(f));
        assert_eq!(NeuronProfile::from_flags(f), NeuronProfile::IntrinsicBursting);

        let f2 = flags::encode(true, NeuronProfile::FastSpiking);
        assert!(flags::is_inhibitory(f2));
        assert_eq!(NeuronProfile::from_flags(f2), NeuronProfile::FastSpiking);
    }

    #[test]
    fn resting_state() {
        let arr = NeuronArrays::new(10, 8, -17920, -14080);
        for &m in &arr.membrane {
            assert_eq!(m, -17920);
        }
        for &t in &arr.threshold {
            assert_eq!(t, -14080);
        }
        for &r in &arr.refract_remaining {
            assert_eq!(r, 0);
        }
        for &tr in &arr.trace {
            assert_eq!(tr, 0);
        }
    }
}
