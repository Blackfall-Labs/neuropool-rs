#![allow(deprecated)]
//! UnifiedNeuron — merged voxel structure + spatial biology + dendritic zones.
//!
//! Reuses `Dendrite`, `Axon`, `Nuclei` from spatial system directly.
//! Replaces `Soma { position: [f32; 3] }` with integer voxel + local coords.
//! Adds three dendritic zone potentials and predicted/burst firing state.

use crate::spatial::{Axon, Dendrite, Nuclei};
use super::zone::{DendriticZone, ZoneWeights};

/// Position within the voxel grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoxelPosition {
    /// Which voxel neighborhood (grid coordinates).
    pub voxel: (u16, u16, u16),
    /// Position within the voxel (0-15 per axis, sub-voxel resolution).
    pub local: (u8, u8, u8),
}

impl VoxelPosition {
    /// Create a position at a specific voxel with local offset.
    #[inline]
    pub const fn new(voxel: (u16, u16, u16), local: (u8, u8, u8)) -> Self {
        Self { voxel, local }
    }

    /// Create a position at a voxel center (local = 8,8,8).
    #[inline]
    pub const fn at_center(voxel: (u16, u16, u16)) -> Self {
        Self { voxel, local: (8, 8, 8) }
    }

    /// Squared distance to another position (integer arithmetic).
    ///
    /// Voxel distance is scaled by 16 (sub-voxel resolution per axis),
    /// then local offset added.
    #[inline]
    pub fn distance_sq(&self, other: &VoxelPosition) -> u64 {
        let dx = (self.voxel.0 as i32 * 16 + self.local.0 as i32)
            - (other.voxel.0 as i32 * 16 + other.local.0 as i32);
        let dy = (self.voxel.1 as i32 * 16 + self.local.1 as i32)
            - (other.voxel.1 as i32 * 16 + other.local.1 as i32);
        let dz = (self.voxel.2 as i32 * 16 + self.local.2 as i32)
            - (other.voxel.2 as i32 * 16 + other.local.2 as i32);
        (dx as i64 * dx as i64 + dy as i64 * dy as i64 + dz as i64 * dz as i64) as u64
    }
}

impl Default for VoxelPosition {
    fn default() -> Self {
        Self::at_center((0, 0, 0))
    }
}

/// A unified neuron with dendritic zone compartments.
///
/// Anatomy (`Dendrite`, `Axon`, `Nuclei`) reused directly from spatial system.
/// Position uses integer voxel coordinates instead of float `Soma`.
pub struct UnifiedNeuron {
    // === Position ===
    /// Where in the voxel grid this neuron lives.
    pub position: VoxelPosition,

    // === Anatomy (from spatial, unchanged) ===
    /// Reception apparatus.
    pub dendrite: Dendrite,
    /// Transmission apparatus.
    pub axon: Axon,
    /// Capability machine (MUTABLE via structural plasticity).
    pub nuclei: Nuclei,

    // === Dendritic Zone Potentials (NEW) ===
    /// Bottom-up / outside input potential.
    pub feedforward_potential: i16,
    /// Lateral / same-region neighbor potential.
    pub context_potential: i16,
    /// Top-down / modulatory potential.
    pub feedback_potential: i16,

    // === Zone integration weights (from nuclei type) ===
    /// How zones combine into membrane. Set from nuclei at creation,
    /// updated on `mutate_to()`.
    pub zone_weights: ZoneWeights,

    // === Electrical State ===
    /// Combined membrane potential (computed from zone potentials).
    pub membrane: i16,
    /// Firing threshold (per-neuron, from spatial).
    pub threshold: i16,
    /// Eligibility trace for learning.
    pub trace: i8,
    /// Context primed this neuron — next feedforward arrival fires predictively
    /// instead of bursting. Reset after firing.
    pub predicted: bool,

    // === Metabolic State ===
    /// Energy budget (255=full, 0=depleted). Depletes on fire, recovers over time.
    pub stamina: u8,

    // === Timing (microsecond precision) ===
    /// Last time this neuron spiked.
    pub last_spike_us: u64,
    /// Last time a spike arrived at this neuron.
    pub last_arrival_us: u64,
}

/// Default threshold (same as SpatialNeuron).
pub const DEFAULT_THRESHOLD: i16 = -5500;
/// Resting potential (same as SpatialNeuron).
pub const RESTING_POTENTIAL: i16 = -7000;
/// Reset potential after firing (same as SpatialNeuron).
pub const RESET_POTENTIAL: i16 = -8000;
/// Context potential threshold for predictive priming.
pub const CONTEXT_PRIMING_THRESHOLD: i16 = -6000;

/// Map nuclei type to zone weights.
fn weights_for_nuclei(nuclei: &Nuclei) -> ZoneWeights {
    if nuclei.is_oscillator() {
        ZoneWeights::OSCILLATOR
    } else if nuclei.is_memory() {
        ZoneWeights::MEMORY
    } else if nuclei.interface.kind == crate::spatial::Interface::KIND_TERNSIG {
        ZoneWeights::BALANCED
    } else if nuclei.leak >= 200 {
        // High leak → gate or interneuron
        if nuclei.is_inhibitory() {
            ZoneWeights::INTERNEURON
        } else {
            ZoneWeights::GATE
        }
    } else if nuclei.myelin_affinity >= 200 && nuclei.metabolic_rate <= 60 {
        // Fast, cheap → relay
        ZoneWeights::RELAY
    } else {
        ZoneWeights::PYRAMIDAL
    }
}

impl UnifiedNeuron {
    /// Create a neuron with explicit anatomy.
    pub fn new(
        position: VoxelPosition,
        dendrite: Dendrite,
        axon: Axon,
        nuclei: Nuclei,
    ) -> Self {
        let zone_weights = weights_for_nuclei(&nuclei);
        Self {
            position,
            dendrite,
            axon,
            nuclei,
            feedforward_potential: RESTING_POTENTIAL,
            context_potential: RESTING_POTENTIAL,
            feedback_potential: RESTING_POTENTIAL,
            zone_weights,
            membrane: RESTING_POTENTIAL,
            threshold: DEFAULT_THRESHOLD,
            trace: 0,
            predicted: false,
            stamina: 255,
            last_spike_us: 0,
            last_arrival_us: 0,
        }
    }

    // === Factory functions (same anatomy presets as SpatialNeuron) ===

    /// Pyramidal cell at a voxel position.
    pub fn pyramidal_at(pos: VoxelPosition) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::pyramidal())
    }

    /// Interneuron at a voxel position.
    pub fn interneuron_at(pos: VoxelPosition) -> Self {
        Self::new(pos, Dendrite::new(0.5, 50), Axon::default(), Nuclei::interneuron())
    }

    /// Sensory neuron at a voxel position.
    pub fn sensory_at(pos: VoxelPosition, channel: u16, modality: u8) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::sensory(channel, modality))
    }

    /// Motor neuron at a voxel position.
    pub fn motor_at(pos: VoxelPosition, channel: u16, modality: u8) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::motor(channel, modality))
    }

    /// Oscillator neuron at a voxel position.
    pub fn oscillator_at(pos: VoxelPosition, period_us: u32) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::oscillator(period_us))
    }

    /// Gate neuron at a voxel position.
    pub fn gate_at(pos: VoxelPosition) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::gate())
    }

    /// Relay neuron at a voxel position.
    pub fn relay_at(pos: VoxelPosition) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::relay())
    }

    /// Memory neuron at a voxel position.
    pub fn memory_at(pos: VoxelPosition, bank_id: u16) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::memory(bank_id))
    }

    /// Ternsig-bound neuron at a voxel position.
    pub fn ternsig_at(pos: VoxelPosition, program_id: u32) -> Self {
        Self::new(pos, Dendrite::standard(), Axon::default(), Nuclei::ternsig(program_id))
    }

    // === Zone Integration ===

    /// Integrate current into a specific dendritic zone.
    #[inline]
    pub fn integrate_zone(&mut self, zone: DendriticZone, current: i16) {
        match zone {
            DendriticZone::Feedforward => {
                self.feedforward_potential = self.feedforward_potential.saturating_add(current);
            }
            DendriticZone::Context => {
                self.context_potential = self.context_potential.saturating_add(current);
                // Context above priming threshold → predict
                if self.context_potential > CONTEXT_PRIMING_THRESHOLD {
                    self.predicted = true;
                }
            }
            DendriticZone::Feedback => {
                self.feedback_potential = self.feedback_potential.saturating_add(current);
            }
        }
    }

    /// Recompute membrane from zone potentials using zone weights.
    #[inline]
    pub fn recompute_membrane(&mut self) {
        let combined = self.zone_weights.combine(
            self.feedforward_potential,
            self.context_potential,
            self.feedback_potential,
        );
        // Saturate to i16
        self.membrane = combined.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }

    /// Is the membrane above threshold?
    #[inline]
    pub fn above_threshold(&self) -> bool {
        self.membrane > self.threshold
    }

    /// Is this neuron in refractory period?
    #[inline]
    pub fn in_refractory(&self, current_time_us: u64) -> bool {
        // last_spike_us == 0 means never spiked — not in refractory
        self.last_spike_us > 0
            && current_time_us < self.last_spike_us + self.nuclei.refractory as u64
    }

    /// Can this neuron fire right now?
    #[inline]
    pub fn can_fire(&self, current_time_us: u64) -> bool {
        self.above_threshold()
            && !self.in_refractory(current_time_us)
            && self.stamina > 0
            && self.axon.is_alive()
    }

    /// Fire the neuron. Returns whether this was a predicted (primed) fire.
    ///
    /// - Resets membrane to RESET_POTENTIAL
    /// - Resets all zone potentials
    /// - Updates spike timing
    /// - Boosts eligibility trace
    /// - Drains stamina
    /// - Clears predicted flag
    pub fn fire(&mut self, current_time_us: u64) -> bool {
        let was_predicted = self.predicted;

        // Reset electrical state
        self.membrane = RESET_POTENTIAL;
        self.feedforward_potential = RESTING_POTENTIAL;
        self.context_potential = RESTING_POTENTIAL;
        self.feedback_potential = RESTING_POTENTIAL;

        // Timing
        self.last_spike_us = current_time_us;

        // Eligibility trace boost
        self.trace = self.trace.saturating_add(50);

        // Stamina drain (from nuclei metabolic rate)
        let drain = (self.nuclei.metabolic_rate / 10).max(5);
        self.stamina = self.stamina.saturating_sub(drain);

        // Clear prediction
        self.predicted = false;

        was_predicted
    }

    /// Apply leak to all zone potentials.
    ///
    /// Each zone decays toward RESTING_POTENTIAL at the nuclei's leak rate.
    /// `elapsed_us` is time since last update.
    pub fn apply_leak(&mut self, elapsed_us: u64) {
        let tau = self.nuclei.leak_tau_us() as u64;
        if tau == 0 || elapsed_us == 0 {
            return;
        }

        // Exponential decay approximation: potential moves toward rest by
        // fraction = elapsed / tau (clamped)
        let fraction = ((elapsed_us * 256) / tau).min(256) as i32;

        for potential in [
            &mut self.feedforward_potential,
            &mut self.context_potential,
            &mut self.feedback_potential,
        ] {
            let distance = RESTING_POTENTIAL as i32 - *potential as i32;
            let decay = (distance * fraction) / 256;
            *potential = (*potential as i32 + decay).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }

        // Recompute membrane after leak
        self.recompute_membrane();
    }

    /// Decay eligibility trace.
    #[inline]
    pub fn decay_trace(&mut self) {
        self.trace = self.trace.saturating_sub(1);
    }

    /// Structural plasticity: change nuclei type.
    ///
    /// Preserves position, anatomy, electrical state. Changes capability
    /// machine and updates zone weights.
    pub fn mutate_to(&mut self, new_nuclei: Nuclei) {
        self.nuclei = new_nuclei;
        self.zone_weights = weights_for_nuclei(&new_nuclei);
    }

    /// Is this an oscillator that should autonomously fire?
    #[inline]
    pub fn oscillator_should_fire(&self, current_time_us: u64) -> bool {
        self.nuclei.is_oscillator()
            && current_time_us >= self.last_spike_us + self.nuclei.oscillation_period as u64
            && self.stamina > 0
    }

    /// Squared distance to another neuron (integer).
    #[inline]
    pub fn distance_sq(&self, other: &UnifiedNeuron) -> u64 {
        self.position.distance_sq(&other.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramidal_creation() {
        let n = UnifiedNeuron::pyramidal_at(VoxelPosition::at_center((5, 5, 5)));
        assert_eq!(n.membrane, RESTING_POTENTIAL);
        assert_eq!(n.threshold, DEFAULT_THRESHOLD);
        assert_eq!(n.stamina, 255);
        assert!(!n.predicted);
        assert!(n.nuclei.is_excitatory());
        assert_eq!(n.zone_weights, ZoneWeights::PYRAMIDAL);
    }

    #[test]
    fn gate_gets_gate_weights() {
        let n = UnifiedNeuron::gate_at(VoxelPosition::default());
        assert_eq!(n.zone_weights, ZoneWeights::GATE);
    }

    #[test]
    fn relay_gets_relay_weights() {
        let n = UnifiedNeuron::relay_at(VoxelPosition::default());
        assert_eq!(n.zone_weights, ZoneWeights::RELAY);
    }

    #[test]
    fn interneuron_gets_interneuron_weights() {
        let n = UnifiedNeuron::interneuron_at(VoxelPosition::default());
        assert_eq!(n.zone_weights, ZoneWeights::INTERNEURON);
    }

    #[test]
    fn zone_integration() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());
        let base_ff = n.feedforward_potential;

        // Inject feedforward current
        n.integrate_zone(DendriticZone::Feedforward, 500);
        assert_eq!(n.feedforward_potential, base_ff + 500);

        // Context unchanged
        assert_eq!(n.context_potential, RESTING_POTENTIAL);
    }

    #[test]
    fn context_priming() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());
        assert!(!n.predicted);

        // Push context above priming threshold
        n.integrate_zone(DendriticZone::Context, 2000);
        assert!(n.predicted, "context above threshold should prime prediction");
    }

    #[test]
    fn fire_returns_predicted_state() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());

        // Unpredicted fire
        n.membrane = n.threshold + 100;
        let was_predicted = n.fire(1000);
        assert!(!was_predicted);

        // Predicted fire
        n.predicted = true;
        n.membrane = n.threshold + 100;
        n.stamina = 255;
        let was_predicted = n.fire(5000);
        assert!(was_predicted);
        assert!(!n.predicted, "predicted flag should clear after fire");
    }

    #[test]
    fn fire_resets_zones() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());
        n.feedforward_potential = 0;
        n.context_potential = 0;
        n.feedback_potential = 0;

        n.fire(1000);

        assert_eq!(n.feedforward_potential, RESTING_POTENTIAL);
        assert_eq!(n.context_potential, RESTING_POTENTIAL);
        assert_eq!(n.feedback_potential, RESTING_POTENTIAL);
        assert_eq!(n.membrane, RESET_POTENTIAL);
    }

    #[test]
    fn fire_drains_stamina() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());
        let before = n.stamina;
        n.fire(1000);
        assert!(n.stamina < before, "fire should drain stamina");
    }

    #[test]
    fn refractory_period() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());
        n.fire(1000);

        assert!(n.in_refractory(1500));
        assert!(!n.in_refractory(1000 + n.nuclei.refractory as u64 + 1));
    }

    #[test]
    fn mutate_to_changes_weights() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());
        assert_eq!(n.zone_weights, ZoneWeights::PYRAMIDAL);

        n.mutate_to(Nuclei::gate());
        assert_eq!(n.zone_weights, ZoneWeights::GATE);
        assert!(n.nuclei.leak >= 200); // gate has high leak for coincidence detection
    }

    #[test]
    fn voxel_distance() {
        let a = VoxelPosition::at_center((0, 0, 0));
        let b = VoxelPosition::at_center((1, 0, 0));
        let dist = a.distance_sq(&b);
        // (1*16 - 0*16)^2 = 256
        assert_eq!(dist, 256);
    }

    #[test]
    fn leak_decays_toward_rest() {
        let mut n = UnifiedNeuron::pyramidal_at(VoxelPosition::default());
        // Push feedforward above rest
        n.feedforward_potential = -5000; // above resting (-7000)

        n.apply_leak(10_000); // 10ms

        // Should have moved toward resting
        assert!(
            n.feedforward_potential < -5000,
            "should decay toward rest: {}",
            n.feedforward_potential
        );
    }

    #[test]
    fn oscillator_fires_on_period() {
        let n = UnifiedNeuron::oscillator_at(VoxelPosition::default(), 10_000);
        assert!(n.oscillator_should_fire(10_001));
        assert!(!n.oscillator_should_fire(5_000));
    }
}
