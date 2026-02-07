//! Nuclei — the capability machine of a spatial neuron.
//!
//! **Not enums. Not categories. Physical machines with properties that determine capabilities.**
//!
//! A nuclei isn't "what type of neuron this is" — it's "what this neuron's physical
//! machinery can do." Physical properties constrain behavior, they don't categorize it.
//!
//! ## Physical Properties → Emergent Capabilities
//!
//! | Property | Low Value | High Value |
//! |----------|-----------|------------|
//! | `soma_size` | Few connections, local | Many connections, hub |
//! | `axon_affinity` | Short-range only | Long-range projections |
//! | `myelin_affinity` | Slow conduction | Fast conduction |
//! | `metabolic_rate` | Cheap, survives pruning | Expensive, needs activity |
//! | `leak` | Holds charge, integrates slowly | Fast decay, needs coincidence |
//! | `oscillation_period` | Not rhythmic (0) | Pacemaker |

use super::Interface;
use ternary_signal::Polarity;

/// Physical properties of the nucleus — capabilities, not categories.
///
/// Factory functions (e.g., `pyramidal()`, `interneuron()`) are presets,
/// not types. You can create custom nuclei with any combination of properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nuclei {
    // === Physical Constraints ===
    /// Soma volume → synaptic capacity (larger soma = more dendritic spines)
    pub soma_size: u8,

    /// Axon growth potential (higher = can sustain longer axons metabolically)
    pub axon_affinity: u8,

    /// Myelination receptivity (higher = axon myelinates faster/easier)
    pub myelin_affinity: u8,

    /// Metabolic burn rate (higher = more expensive to keep alive)
    pub metabolic_rate: u8,

    /// Membrane leak rate (higher = faster decay to resting)
    pub leak: u8,

    /// Refractory period in microseconds
    pub refractory: u32,

    // === Autonomous Machinery ===
    /// Oscillation period in microseconds (0 = no autonomous rhythm)
    pub oscillation_period: u32,

    // === External Interface ===
    /// What this neuron physically connects to (if anything)
    pub interface: Interface,

    // === Default Polarity ===
    /// Neurotransmitter type determines default polarity
    pub polarity: Polarity,
}

impl Nuclei {
    /// Create a custom nuclei with all properties specified.
    #[inline]
    pub const fn new(
        soma_size: u8,
        axon_affinity: u8,
        myelin_affinity: u8,
        metabolic_rate: u8,
        leak: u8,
        refractory: u32,
        oscillation_period: u32,
        interface: Interface,
        polarity: Polarity,
    ) -> Self {
        Self {
            soma_size,
            axon_affinity,
            myelin_affinity,
            metabolic_rate,
            leak,
            refractory,
            oscillation_period,
            interface,
            polarity,
        }
    }

    // =========================================================================
    // Factory Functions — Presets, Not Categories
    // =========================================================================

    /// Pyramidal cell — large, long-range, excitatory.
    ///
    /// The workhorses of cortical processing. Large soma supports many
    /// connections, high axon affinity enables long-range projections.
    pub const fn pyramidal() -> Self {
        Self {
            soma_size: 200,
            axon_affinity: 200,
            myelin_affinity: 180,
            metabolic_rate: 100,
            leak: 80,
            refractory: 2000,
            oscillation_period: 0,
            interface: Interface::none(),
            polarity: Polarity::Positive,
        }
    }

    /// Interneuron — small, local, inhibitory, fast.
    ///
    /// Local circuit neurons that provide inhibition. Small soma limits
    /// connections, low axon affinity keeps them local. Fast leak means
    /// they need coincident input to fire.
    pub const fn interneuron() -> Self {
        Self {
            soma_size: 80,
            axon_affinity: 40,
            myelin_affinity: 20,
            metabolic_rate: 60,
            leak: 200, // fast leak = needs coincidence
            refractory: 500, // fast recovery
            oscillation_period: 0,
            interface: Interface::none(),
            polarity: Polarity::Negative,
        }
    }

    /// Memory neuron — energy-gated databank interface.
    ///
    /// Connects to a databank. Low energy = read, high energy = write,
    /// neutral = exclude from query.
    pub const fn memory(bank_id: u16) -> Self {
        Self {
            soma_size: 120,
            axon_affinity: 100,
            myelin_affinity: 100,
            metabolic_rate: 80,
            leak: 100,
            refractory: 1500,
            oscillation_period: 0,
            interface: Interface::memory(bank_id),
            polarity: Polarity::Positive,
        }
    }

    /// Sensory neuron — transduces external input.
    ///
    /// Receives input from an external channel (microphone, camera, etc.)
    /// and converts it to neural activity.
    pub const fn sensory(channel: u16, modality: u8) -> Self {
        Self {
            soma_size: 100,
            axon_affinity: 150,
            myelin_affinity: 120,
            metabolic_rate: 90,
            leak: 120,
            refractory: 1000,
            oscillation_period: 0,
            interface: Interface::sensory(channel, modality),
            polarity: Polarity::Positive,
        }
    }

    /// Auditory sensory neuron.
    pub const fn auditory(channel: u16) -> Self {
        Self::sensory(channel, Interface::MODALITY_AUDITORY)
    }

    /// Visual sensory neuron.
    pub const fn visual(channel: u16) -> Self {
        Self::sensory(channel, Interface::MODALITY_VISUAL)
    }

    /// Motor neuron — outputs to actuator.
    ///
    /// Sends signals to external actuators (speakers, motors, etc.).
    /// Large soma and high axon affinity for reliable output.
    pub const fn motor(channel: u16, modality: u8) -> Self {
        Self {
            soma_size: 180,
            axon_affinity: 220,
            myelin_affinity: 200,
            metabolic_rate: 110,
            leak: 70,
            refractory: 2000,
            oscillation_period: 0,
            interface: Interface::motor(channel, modality),
            polarity: Polarity::Positive,
        }
    }

    /// Oscillator — autonomous rhythm generator.
    ///
    /// Pacemaker neurons that fire periodically without external input.
    /// Provides temporal reference for downstream neurons.
    pub const fn oscillator(period_us: u32) -> Self {
        Self {
            soma_size: 100,
            axon_affinity: 150,
            myelin_affinity: 150,
            metabolic_rate: 120, // rhythms cost energy
            leak: 60,
            refractory: 1000,
            oscillation_period: period_us,
            interface: Interface::none(),
            polarity: Polarity::Positive,
        }
    }

    /// Chemical modulator — releases neuromodulator.
    ///
    /// Affects plasticity and global state rather than direct excitation.
    /// Zero polarity because modulators don't directly excite/inhibit.
    pub const fn modulator(chemical_id: u8) -> Self {
        Self {
            soma_size: 90,
            axon_affinity: 80,
            myelin_affinity: 40,
            metabolic_rate: 70,
            leak: 100,
            refractory: 3000,
            oscillation_period: 0,
            interface: Interface::chemical(chemical_id),
            polarity: Polarity::Zero, // modulators don't directly excite/inhibit
        }
    }

    /// Relay neuron — passes signals with minimal processing.
    ///
    /// High throughput, low metabolic cost. Used in thalamic relays.
    pub const fn relay() -> Self {
        Self {
            soma_size: 140,
            axon_affinity: 180,
            myelin_affinity: 220, // heavily myelinated for speed
            metabolic_rate: 50,   // efficient
            leak: 150,
            refractory: 800,
            oscillation_period: 0,
            interface: Interface::none(),
            polarity: Polarity::Positive,
        }
    }

    /// Gate neuron — modulates signal flow.
    ///
    /// High leak rate means it needs strong, coincident input.
    /// Acts as a conditional gate for downstream activation.
    pub const fn gate() -> Self {
        Self {
            soma_size: 100,
            axon_affinity: 100,
            myelin_affinity: 100,
            metabolic_rate: 80,
            leak: 220, // very fast decay
            refractory: 1200,
            oscillation_period: 0,
            interface: Interface::none(),
            polarity: Polarity::Positive,
        }
    }

    // =========================================================================
    // Property Queries
    // =========================================================================

    /// Is this an oscillating neuron?
    #[inline]
    pub const fn is_oscillator(&self) -> bool {
        self.oscillation_period > 0
    }

    /// Is this a sensory input neuron?
    #[inline]
    pub const fn is_sensory(&self) -> bool {
        self.interface.is_sensory()
    }

    /// Is this a motor output neuron?
    #[inline]
    pub const fn is_motor(&self) -> bool {
        self.interface.is_motor()
    }

    /// Is this a memory interface neuron?
    #[inline]
    pub const fn is_memory(&self) -> bool {
        self.interface.is_memory()
    }

    /// Is this an internal (non-interface) neuron?
    #[inline]
    pub const fn is_internal(&self) -> bool {
        !self.interface.is_external()
    }

    /// Is this an excitatory neuron (positive polarity)?
    #[inline]
    pub const fn is_excitatory(&self) -> bool {
        matches!(self.polarity, Polarity::Positive)
    }

    /// Is this an inhibitory neuron (negative polarity)?
    #[inline]
    pub const fn is_inhibitory(&self) -> bool {
        matches!(self.polarity, Polarity::Negative)
    }

    /// Maximum number of dendritic connections this neuron can support.
    ///
    /// Derived from soma_size (larger soma = more spines).
    #[inline]
    pub const fn max_dendrite_connections(&self) -> u16 {
        // Scale: soma_size 0-255 → 10-500 connections
        10 + (self.soma_size as u16 * 2)
    }

    /// Maximum axon length this neuron can metabolically sustain.
    ///
    /// Derived from axon_affinity and metabolic_rate.
    #[inline]
    pub fn max_axon_length(&self) -> f32 {
        // Higher affinity, lower metabolic rate = longer sustainable axon
        let affinity_factor = self.axon_affinity as f32 / 255.0;
        let metabolic_factor = 1.0 - (self.metabolic_rate as f32 / 510.0); // 0.5-1.0
        affinity_factor * metabolic_factor * 20.0 // max ~20 units
    }

    /// Conduction velocity factor (0.0-1.0, higher = faster).
    ///
    /// Derived from myelin_affinity.
    #[inline]
    pub fn conduction_velocity(&self) -> f32 {
        0.2 + (self.myelin_affinity as f32 / 255.0) * 0.8
    }

    /// Leak time constant in microseconds.
    ///
    /// Higher leak value = faster decay = shorter time constant.
    #[inline]
    pub const fn leak_tau_us(&self) -> u32 {
        // leak 0 = 100ms tau, leak 255 = 1ms tau
        100_000 - (self.leak as u32 * 390)
    }
}

impl Default for Nuclei {
    fn default() -> Self {
        Self::pyramidal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_presets() {
        let pyr = Nuclei::pyramidal();
        assert!(pyr.is_excitatory());
        assert!(pyr.is_internal());

        let inter = Nuclei::interneuron();
        assert!(inter.is_inhibitory());

        let osc = Nuclei::oscillator(1000);
        assert!(osc.is_oscillator());

        let mem = Nuclei::memory(0);
        assert!(mem.is_memory());

        let sens = Nuclei::auditory(0);
        assert!(sens.is_sensory());

        let mot = Nuclei::motor(0, 1);
        assert!(mot.is_motor());
    }

    #[test]
    fn test_derived_properties() {
        let pyr = Nuclei::pyramidal();
        assert!(pyr.max_dendrite_connections() > 100);
        assert!(pyr.max_axon_length() > 5.0);
        assert!(pyr.conduction_velocity() > 0.5);
    }

    #[test]
    fn test_leak_tau() {
        let fast_leak = Nuclei::interneuron();
        let slow_leak = Nuclei::pyramidal();
        assert!(fast_leak.leak_tau_us() < slow_leak.leak_tau_us());
    }
}
