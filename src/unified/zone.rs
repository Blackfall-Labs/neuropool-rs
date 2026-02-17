//! Dendritic zones — compartmental model for spike integration.
//!
//! Each neuron has three dendritic zones that accumulate potential independently:
//! - **Feedforward**: bottom-up input (sensory, thalamic relay, excitatory drive)
//! - **Context**: lateral input (same-region neighbors, temporal context)
//! - **Feedback**: top-down input (modulatory, predictive, attentional)
//!
//! The combined membrane potential is a weighted sum of zone potentials,
//! with weights determined by the neuron's nuclei type.

/// Which dendritic zone a synapse targets on the receiving neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DendriticZone {
    /// Bottom-up / outside input. Primary driver of firing.
    Feedforward = 0,
    /// Lateral / same-region neighbors. Provides temporal context and priming.
    Context = 1,
    /// Top-down / modulatory. Modulates gain, gates attention.
    Feedback = 2,
}

impl DendriticZone {
    /// Number of zones.
    pub const COUNT: usize = 3;

    /// All zones in order.
    pub const ALL: [DendriticZone; 3] = [
        DendriticZone::Feedforward,
        DendriticZone::Context,
        DendriticZone::Feedback,
    ];

    /// Convert from u8.
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Feedforward),
            1 => Some(Self::Context),
            2 => Some(Self::Feedback),
            _ => None,
        }
    }

    /// Convert to array index.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Per-zone integration weights. Determines how each dendritic zone
/// contributes to the combined membrane potential.
///
/// Expressed as Q8.8 fixed-point: 256 = 1.0x, 128 = 0.5x, 384 = 1.5x.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneWeights {
    /// Feedforward zone contribution weight (Q8.8).
    pub feedforward: u16,
    /// Context zone contribution weight (Q8.8).
    pub context: u16,
    /// Feedback zone contribution weight (Q8.8).
    pub feedback: u16,
}

impl ZoneWeights {
    /// Standard cortical pyramidal: feedforward-dominant, context for priming,
    /// feedback for modulation.
    pub const PYRAMIDAL: Self = Self {
        feedforward: 256, // 1.0x
        context: 128,     // 0.5x
        feedback: 64,     // 0.25x
    };

    /// Gate neuron: feedback-dominant (top-down gating).
    pub const GATE: Self = Self {
        feedforward: 128, // 0.5x
        context: 64,      // 0.25x
        feedback: 256,    // 1.0x
    };

    /// Relay neuron: feedforward-only (pass-through).
    pub const RELAY: Self = Self {
        feedforward: 256, // 1.0x
        context: 32,      // 0.125x
        feedback: 32,     // 0.125x
    };

    /// Interneuron: context-sensitive (lateral inhibition).
    pub const INTERNEURON: Self = Self {
        feedforward: 192, // 0.75x
        context: 256,     // 1.0x
        feedback: 64,     // 0.25x
    };

    /// Oscillator: balanced (entrainment from all zones).
    pub const OSCILLATOR: Self = Self {
        feedforward: 192,
        context: 192,
        feedback: 192,
    };

    /// Memory: context-heavy (association-driven recall).
    pub const MEMORY: Self = Self {
        feedforward: 192, // 0.75x
        context: 256,     // 1.0x
        feedback: 128,    // 0.5x
    };

    /// Balanced: equal weight for all zones.
    pub const BALANCED: Self = Self {
        feedforward: 256,
        context: 256,
        feedback: 256,
    };

    /// Compute weighted membrane potential from zone potentials.
    ///
    /// Weighted average normalized by weight sum. When all zones have the same
    /// value V, combine returns V regardless of weights. This keeps resting
    /// potential correct across nuclei types.
    ///
    /// Returns combined potential as i32 to avoid overflow during computation.
    /// Caller truncates/saturates to i16 for membrane.
    #[inline]
    pub fn combine(&self, ff: i16, ctx: i16, fb: i16) -> i32 {
        let weight_sum = self.feedforward as i32 + self.context as i32 + self.feedback as i32;
        if weight_sum == 0 {
            return 0;
        }
        let numerator = ff as i32 * self.feedforward as i32
            + ctx as i32 * self.context as i32
            + fb as i32 * self.feedback as i32;
        numerator / weight_sum
    }
}

impl Default for ZoneWeights {
    fn default() -> Self {
        Self::PYRAMIDAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_round_trip() {
        for z in &DendriticZone::ALL {
            assert_eq!(DendriticZone::from_u8(*z as u8), Some(*z));
        }
        assert_eq!(DendriticZone::from_u8(3), None);
    }

    #[test]
    fn pyramidal_weights_feedforward_dominant() {
        let w = ZoneWeights::PYRAMIDAL;
        // Same input on all zones → normalized average equals that value
        let combined = w.combine(1000, 1000, 1000);
        assert_eq!(combined, 1000);
        // Different inputs: ff dominates
        let combined = w.combine(2000, 0, 0);
        let ff_contrib = 2000 * 256 / (256 + 128 + 64);
        assert_eq!(combined, ff_contrib);
    }

    #[test]
    fn gate_weights_feedback_dominant() {
        let w = ZoneWeights::GATE;
        // Same input → normalized average
        let combined = w.combine(1000, 1000, 1000);
        assert_eq!(combined, 1000);
        // Feedback dominates when only fb has input
        let fb_only = w.combine(0, 0, 2000);
        let ff_only = w.combine(2000, 0, 0);
        assert!(fb_only > ff_only, "gate feedback should dominate");
    }

    #[test]
    fn relay_mostly_feedforward() {
        let w = ZoneWeights::RELAY;
        let ff_only = w.combine(1000, 0, 0);
        let ctx_only = w.combine(0, 1000, 0);
        assert!(ff_only > ctx_only * 4, "relay should be feedforward-dominant");
    }

    #[test]
    fn zero_input_zero_output() {
        let w = ZoneWeights::PYRAMIDAL;
        assert_eq!(w.combine(0, 0, 0), 0);
    }

    #[test]
    fn negative_potentials() {
        let w = ZoneWeights::BALANCED;
        // All same → normalized average equals that value
        let combined = w.combine(-1000, -1000, -1000);
        assert_eq!(combined, -1000);
    }
}
