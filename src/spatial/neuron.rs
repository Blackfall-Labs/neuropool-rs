//! SpatialNeuron — the complete biological neuron.
//!
//! Combines anatomy (soma, dendrite, axon), nuclei (capabilities),
//! and electrical state (membrane, threshold, trace) into a single unit.

use super::{Axon, Dendrite, Nuclei, Soma};

/// A biological neuron — anatomy + capabilities + electrical state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialNeuron {
    // === Anatomy ===
    /// Cell body position
    pub soma: Soma,
    /// Reception apparatus
    pub dendrite: Dendrite,
    /// Transmission apparatus
    pub axon: Axon,

    // === Capabilities ===
    /// Physical properties that determine behavior
    pub nuclei: Nuclei,

    // === Electrical State ===
    /// Current membrane potential (mV scaled to i16)
    pub membrane: i16,
    /// Firing threshold
    pub threshold: i16,
    /// Eligibility trace for learning
    pub trace: i8,

    // === Timing ===
    /// Last update time in microseconds (for delta-time leak)
    pub last_update_us: u64,
    /// Last spike time in microseconds (for refractory)
    pub last_spike_us: u64,
}

impl SpatialNeuron {
    /// Resting membrane potential.
    pub const RESTING_POTENTIAL: i16 = -7000;
    /// Reset potential after firing.
    pub const RESET_POTENTIAL: i16 = -8000;
    /// Default firing threshold.
    pub const DEFAULT_THRESHOLD: i16 = -5500;

    /// Create a new spatial neuron.
    #[inline]
    pub const fn new(soma: Soma, dendrite: Dendrite, axon: Axon, nuclei: Nuclei) -> Self {
        Self {
            soma,
            dendrite,
            axon,
            nuclei,
            membrane: Self::RESTING_POTENTIAL,
            threshold: Self::DEFAULT_THRESHOLD,
            trace: 0,
            last_update_us: 0,
            last_spike_us: 0,
        }
    }

    /// Create a neuron with just position and nuclei (default anatomy).
    #[inline]
    pub fn at(position: [f32; 3], nuclei: Nuclei) -> Self {
        Self::new(
            Soma::at(position),
            Dendrite::default(),
            Axon::toward(position), // axon starts at soma
            nuclei,
        )
    }

    /// Create a pyramidal neuron at position.
    #[inline]
    pub fn pyramidal_at(position: [f32; 3]) -> Self {
        Self::at(position, Nuclei::pyramidal())
    }

    /// Create an interneuron at position.
    #[inline]
    pub fn interneuron_at(position: [f32; 3]) -> Self {
        Self::at(position, Nuclei::interneuron())
    }

    /// Create a sensory neuron at position.
    #[inline]
    pub fn sensory_at(position: [f32; 3], channel: u16, modality: u8) -> Self {
        Self::at(position, Nuclei::sensory(channel, modality))
    }

    /// Create a motor neuron at position.
    #[inline]
    pub fn motor_at(position: [f32; 3], channel: u16, modality: u8) -> Self {
        Self::at(position, Nuclei::motor(channel, modality))
    }

    // =========================================================================
    // Electrical Operations
    // =========================================================================

    /// Is the neuron above threshold?
    #[inline]
    pub const fn above_threshold(&self) -> bool {
        self.membrane >= self.threshold
    }

    /// Is the neuron in refractory period?
    ///
    /// A neuron that has never spiked (last_spike_us == 0) is not in refractory.
    #[inline]
    pub const fn in_refractory(&self, current_time_us: u64) -> bool {
        // Never spiked = not in refractory
        if self.last_spike_us == 0 {
            return false;
        }
        current_time_us < self.last_spike_us + self.nuclei.refractory as u64
    }

    /// Can the neuron fire right now?
    #[inline]
    pub const fn can_fire(&self, current_time_us: u64) -> bool {
        self.above_threshold() && !self.in_refractory(current_time_us)
    }

    /// Apply leak for elapsed time.
    ///
    /// Uses exponential decay toward resting potential based on
    /// the nuclei's leak rate.
    pub fn apply_leak(&mut self, current_time_us: u64) {
        if current_time_us <= self.last_update_us {
            return;
        }

        let dt_us = current_time_us - self.last_update_us;
        let tau = self.nuclei.leak_tau_us() as f32;

        // Exponential decay: V = V_rest + (V - V_rest) * e^(-dt/tau)
        let decay = (-(dt_us as f32) / tau).exp();
        let delta_from_rest = self.membrane - Self::RESTING_POTENTIAL;
        self.membrane = Self::RESTING_POTENTIAL + (delta_from_rest as f32 * decay) as i16;

        self.last_update_us = current_time_us;
    }

    /// Integrate incoming current.
    #[inline]
    pub fn integrate(&mut self, current: i16) {
        self.membrane = self.membrane.saturating_add(current);
    }

    /// Fire the neuron (reset membrane, update spike time, boost trace).
    pub fn fire(&mut self, current_time_us: u64) {
        self.membrane = Self::RESET_POTENTIAL;
        self.last_spike_us = current_time_us;
        self.trace = self.trace.saturating_add(30); // boost trace on fire
        self.axon.boost(10); // activity keeps axon healthy
    }

    /// Decay the eligibility trace.
    #[inline]
    pub fn decay_trace(&mut self, retention: f32) {
        self.trace = (self.trace as f32 * retention) as i8;
    }

    // =========================================================================
    // Oscillator Support
    // =========================================================================

    /// If this is an oscillator, check if it should fire autonomously.
    ///
    /// Returns true if the oscillator has completed a period and should fire.
    pub fn oscillator_should_fire(&self, current_time_us: u64) -> bool {
        if !self.nuclei.is_oscillator() {
            return false;
        }

        let period = self.nuclei.oscillation_period as u64;
        current_time_us >= self.last_spike_us + period
    }

    /// Apply autonomous depolarization for oscillators.
    ///
    /// Call this before checking threshold to allow oscillators to
    /// reach threshold through autonomous ramp.
    pub fn oscillator_ramp(&mut self, current_time_us: u64) {
        if !self.nuclei.is_oscillator() {
            return;
        }

        let dt_us = current_time_us.saturating_sub(self.last_update_us);
        let period = self.nuclei.oscillation_period;

        if period == 0 {
            return;
        }

        // Ramp rate: reach threshold in one period
        let ramp_total = Self::DEFAULT_THRESHOLD - Self::RESET_POTENTIAL;
        let ramp_per_us = ramp_total as f32 / period as f32;
        let ramp = (dt_us as f32 * ramp_per_us) as i16;

        self.membrane = self.membrane.saturating_add(ramp);
    }

    // =========================================================================
    // Spatial Operations
    // =========================================================================

    /// Distance from this neuron's soma to another position.
    #[inline]
    pub fn distance_to(&self, pos: [f32; 3]) -> f32 {
        let dx = self.soma.position[0] - pos[0];
        let dy = self.soma.position[1] - pos[1];
        let dz = self.soma.position[2] - pos[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Axon length (soma to terminal).
    #[inline]
    pub fn axon_length(&self) -> f32 {
        self.axon.length(self.soma.position)
    }

    /// Is this neuron's axon alive?
    #[inline]
    pub const fn axon_alive(&self) -> bool {
        self.axon.is_alive()
    }

    /// Migrate soma by delta, dragging axon terminal elastically.
    pub fn migrate(&mut self, delta: [f32; 3]) {
        self.soma.translate(delta);
        // Axon terminal follows with some lag (elastic connection)
        self.axon.terminal[0] += delta[0] * 0.8;
        self.axon.terminal[1] += delta[1] * 0.8;
        self.axon.terminal[2] += delta[2] * 0.8;
    }
}

impl Default for SpatialNeuron {
    fn default() -> Self {
        Self::at([0.0, 0.0, 0.0], Nuclei::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Interface;

    #[test]
    fn test_neuron_creation() {
        let n = SpatialNeuron::pyramidal_at([1.0, 2.0, 3.0]);
        assert_eq!(n.soma.position, [1.0, 2.0, 3.0]);
        assert!(n.nuclei.is_excitatory());
    }

    #[test]
    fn test_firing() {
        let mut n = SpatialNeuron::default();
        n.membrane = SpatialNeuron::DEFAULT_THRESHOLD + 100;
        assert!(n.can_fire(0));
        n.fire(100);
        assert!(!n.can_fire(100)); // in refractory
        assert!(!n.in_refractory(100 + n.nuclei.refractory as u64 + 1));
    }

    #[test]
    fn test_leak() {
        let mut n = SpatialNeuron::default();
        n.membrane = -4000; // above resting
        n.last_update_us = 0;
        n.apply_leak(50_000); // 50ms
        assert!(n.membrane < -4000); // should have decayed
        assert!(n.membrane > SpatialNeuron::RESTING_POTENTIAL);
    }

    #[test]
    fn test_oscillator() {
        let mut osc = SpatialNeuron::at([0.0, 0.0, 0.0], Nuclei::oscillator(10_000));
        osc.last_spike_us = 0;
        assert!(!osc.oscillator_should_fire(5_000));
        assert!(osc.oscillator_should_fire(10_000));
    }

    #[test]
    fn test_factory_methods() {
        let inter = SpatialNeuron::interneuron_at([0.0, 0.0, 0.0]);
        assert!(inter.nuclei.is_inhibitory());

        let sens = SpatialNeuron::sensory_at([0.0, 0.0, 0.0], 5, Interface::MODALITY_AUDITORY);
        assert!(sens.nuclei.is_sensory());

        let mot = SpatialNeuron::motor_at([0.0, 0.0, 0.0], 10, Interface::MODALITY_AUDITORY);
        assert!(mot.nuclei.is_motor());
    }

    #[test]
    fn test_integration() {
        let mut n = SpatialNeuron::default();

        // Integrate current
        let initial = n.membrane;
        n.integrate(500);
        assert_eq!(n.membrane, initial + 500);

        // Saturating add
        n.membrane = i16::MAX - 100;
        n.integrate(200);
        assert_eq!(n.membrane, i16::MAX);
    }

    #[test]
    fn test_trace_decay() {
        let mut n = SpatialNeuron::default();
        n.trace = 100;
        n.decay_trace(0.5);
        assert_eq!(n.trace, 50);

        n.decay_trace(0.5);
        assert_eq!(n.trace, 25);
    }

    #[test]
    fn test_trace_boost_on_fire() {
        let mut n = SpatialNeuron::default();
        n.membrane = SpatialNeuron::DEFAULT_THRESHOLD + 100;
        let initial_trace = n.trace;

        n.fire(1000);
        assert!(n.trace > initial_trace); // trace boosted on fire
    }

    #[test]
    fn test_axon_health_boost_on_fire() {
        let mut n = SpatialNeuron::default();
        n.axon.health = 100;
        n.membrane = SpatialNeuron::DEFAULT_THRESHOLD + 100;

        n.fire(1000);
        assert!(n.axon.health > 100); // health boosted
    }

    #[test]
    fn test_distance_to() {
        let n = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        assert!((n.distance_to([3.0, 4.0, 0.0]) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_axon_length() {
        let mut n = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        n.axon = Axon::toward([3.0, 4.0, 0.0]);
        assert!((n.axon_length() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_axon_alive() {
        let mut n = SpatialNeuron::default();
        assert!(n.axon_alive());

        n.axon.health = 0;
        assert!(!n.axon_alive());
    }

    #[test]
    fn test_migrate() {
        let mut n = SpatialNeuron::pyramidal_at([5.0, 5.0, 5.0]);
        n.axon = Axon::toward([7.0, 5.0, 5.0]);

        n.migrate([1.0, 0.0, 0.0]);

        assert_eq!(n.soma.position[0], 6.0);
        // Axon follows with 0.8 elasticity
        assert!((n.axon.terminal[0] - 7.8).abs() < 0.001);
    }

    #[test]
    fn test_refractory_never_spiked() {
        let n = SpatialNeuron::default();
        // Never spiked (last_spike_us = 0), should not be in refractory
        assert!(!n.in_refractory(0));
        assert!(!n.in_refractory(100));
    }

    #[test]
    fn test_refractory_after_spike() {
        let mut n = SpatialNeuron::default();
        n.last_spike_us = 1000;

        // In refractory period
        assert!(n.in_refractory(1000));
        assert!(n.in_refractory(1000 + n.nuclei.refractory as u64 - 1));

        // After refractory
        assert!(!n.in_refractory(1000 + n.nuclei.refractory as u64));
    }

    #[test]
    fn test_oscillator_ramp() {
        let mut osc = SpatialNeuron::at([0.0, 0.0, 0.0], Nuclei::oscillator(10_000));
        osc.membrane = SpatialNeuron::RESET_POTENTIAL;
        osc.last_update_us = 0;

        // After half a period, membrane should be roughly halfway to threshold
        osc.oscillator_ramp(5_000);
        assert!(osc.membrane > SpatialNeuron::RESET_POTENTIAL);
        assert!(osc.membrane < SpatialNeuron::DEFAULT_THRESHOLD);
    }

    #[test]
    fn test_non_oscillator_no_ramp() {
        let mut n = SpatialNeuron::pyramidal_at([0.0, 0.0, 0.0]);
        n.membrane = SpatialNeuron::RESET_POTENTIAL;
        let initial = n.membrane;

        n.oscillator_ramp(5_000);
        assert_eq!(n.membrane, initial); // no change for non-oscillator
    }

    #[test]
    fn test_leak_no_time_change() {
        let mut n = SpatialNeuron::default();
        n.membrane = -4000;
        n.last_update_us = 100;

        // Same time should not change membrane
        n.apply_leak(100);
        assert_eq!(n.membrane, -4000);

        // Earlier time should not change membrane
        n.apply_leak(50);
        assert_eq!(n.membrane, -4000);
    }

    #[test]
    fn test_above_threshold() {
        let mut n = SpatialNeuron::default();

        n.membrane = SpatialNeuron::DEFAULT_THRESHOLD - 1;
        assert!(!n.above_threshold());

        n.membrane = SpatialNeuron::DEFAULT_THRESHOLD;
        assert!(n.above_threshold());

        n.membrane = SpatialNeuron::DEFAULT_THRESHOLD + 1;
        assert!(n.above_threshold());
    }

    #[test]
    fn test_new_with_anatomy() {
        let soma = Soma::at([1.0, 2.0, 3.0]);
        let dendrite = Dendrite::new(2.0, 200);
        let axon = Axon::myelinated([5.0, 5.0, 5.0], 200);
        let nuclei = Nuclei::pyramidal();

        let n = SpatialNeuron::new(soma, dendrite, axon, nuclei);

        assert_eq!(n.soma.position, [1.0, 2.0, 3.0]);
        assert_eq!(n.dendrite.radius, 2.0);
        assert_eq!(n.dendrite.spine_count, 200);
        assert_eq!(n.axon.myelin, 200);
        assert!(n.nuclei.is_excitatory());
    }
}
