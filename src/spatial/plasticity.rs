//! Mastery Learning — weaken-before-flip plasticity for spatial synapses.
//!
//! Based on the 2026-01-23 real-time mastery learning validation:
//! - 556 bytes, 8.7ms, 100% accuracy
//! - Weaken-before-flip, sustained pressure, participation threshold
//!
//! ## Key Principles
//!
//! 1. **Sustained pressure** — threshold before any change (prevents noise)
//! 2. **Weaken before flip** — deplete magnitude, then flip polarity
//! 3. **Participation threshold** — only top 25% activity updates
//! 4. **Signum direction** — ±1, not raw difference magnitude
//! 5. **Metabolic cost** — flip costs more than strengthen
//! 6. **Anti-Hebbian hub pressure** — overused targets get weakened
//! 7. **Flip cooldown** — polarity can't thrash rapidly
//!
//! ## Why This Works
//!
//! - Prevents single-sample polarity thrashing
//! - Dormancy (magnitude=0) acts as soft pruning
//! - Polarity flip requires sustained anti-correlation
//! - Learned patterns don't evaporate from noise
//! - Super-attractors get metabolically penalized
//! - Flips require cooldown period between changes

use super::SpatialSynapse;
use ternary_signal::Signal;
use std::collections::HashMap;

/// Configuration for mastery learning.
#[derive(Clone, Copy, Debug)]
pub struct MasteryConfig {
    /// Pressure threshold before any modification occurs
    pub pressure_threshold: i16,
    /// Activity threshold for participation (0.0-1.0, top fraction)
    pub participation_threshold: f32,
    /// Base cost for magnitude changes
    pub magnitude_cost: u8,
    /// Penalty cost for polarity flips
    pub flip_penalty: u8,
    /// Scale factor for pressure accumulation
    pub pressure_scale: f32,
    /// Hub threshold: fan-in count above which anti-Hebbian kicks in
    pub hub_threshold: u16,
    /// Anti-Hebbian decay rate for overused hubs (0.0-1.0)
    pub hub_decay_rate: f32,
    /// Flip cooldown in microseconds (prevents rapid polarity thrashing)
    pub flip_cooldown_us: u64,
}

impl Default for MasteryConfig {
    fn default() -> Self {
        Self {
            pressure_threshold: 50,
            participation_threshold: 0.25,
            magnitude_cost: 5,
            flip_penalty: 50,
            pressure_scale: 1.0,
            hub_threshold: 20,      // More than 20 incoming = hub
            hub_decay_rate: 0.1,    // 10% decay per cycle for hubs
            flip_cooldown_us: 100_000, // 100ms between flips
        }
    }
}

// ============================================================================
// Hub Tracking — Anti-Hebbian pressure for overused targets
// ============================================================================

/// Tracks incoming connection counts (fan-in) per neuron.
///
/// Neurons with high fan-in become "hubs" and receive anti-Hebbian pressure
/// to prevent runaway co-wiring where everything converges on a few nodes.
#[derive(Clone, Debug, Default)]
pub struct HubTracker {
    /// Fan-in count per neuron (indexed by target neuron)
    fan_in: Vec<u16>,
    /// Total activation received per neuron this cycle
    activation: Vec<u32>,
}

impl HubTracker {
    /// Create a new hub tracker for n neurons.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            fan_in: vec![0; neuron_count],
            activation: vec![0; neuron_count],
        }
    }

    /// Resize for new neuron count.
    pub fn resize(&mut self, neuron_count: usize) {
        self.fan_in.resize(neuron_count, 0);
        self.activation.resize(neuron_count, 0);
    }

    /// Record a synapse connection (increments fan-in for target).
    pub fn record_connection(&mut self, target: u32) {
        if let Some(count) = self.fan_in.get_mut(target as usize) {
            *count = count.saturating_add(1);
        }
    }

    /// Remove a synapse connection (decrements fan-in for target).
    pub fn remove_connection(&mut self, target: u32) {
        if let Some(count) = self.fan_in.get_mut(target as usize) {
            *count = count.saturating_sub(1);
        }
    }

    /// Record activation received by a target.
    pub fn record_activation(&mut self, target: u32, magnitude: u16) {
        if let Some(act) = self.activation.get_mut(target as usize) {
            *act = act.saturating_add(magnitude as u32);
        }
    }

    /// Get fan-in count for a neuron.
    pub fn fan_in(&self, neuron: u32) -> u16 {
        self.fan_in.get(neuron as usize).copied().unwrap_or(0)
    }

    /// Get activation for a neuron this cycle.
    pub fn activation(&self, neuron: u32) -> u32 {
        self.activation.get(neuron as usize).copied().unwrap_or(0)
    }

    /// Check if a neuron is a hub (exceeds threshold).
    pub fn is_hub(&self, neuron: u32, threshold: u16) -> bool {
        self.fan_in(neuron) > threshold
    }

    /// Clear activation counts (call at end of each cycle).
    pub fn clear_activation(&mut self) {
        for a in &mut self.activation {
            *a = 0;
        }
    }

    /// Apply anti-Hebbian decay to synapses targeting hubs.
    ///
    /// Returns indices of synapses that should be weakened.
    pub fn hub_synapses_to_weaken(&self, threshold: u16) -> Vec<u32> {
        self.fan_in
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > threshold)
            .map(|(idx, _)| idx as u32)
            .collect()
    }
}

// ============================================================================
// Flip Cooldown — Prevents rapid polarity thrashing
// ============================================================================

/// Tracks last flip time per synapse to enforce cooldown.
#[derive(Clone, Debug, Default)]
pub struct FlipCooldown {
    /// Last flip time per synapse (indexed by synapse index)
    last_flip: HashMap<usize, u64>,
}

impl FlipCooldown {
    /// Create a new flip cooldown tracker.
    pub fn new() -> Self {
        Self {
            last_flip: HashMap::new(),
        }
    }

    /// Check if a synapse can flip (cooldown elapsed).
    pub fn can_flip(&self, synapse_idx: usize, current_time: u64, cooldown_us: u64) -> bool {
        match self.last_flip.get(&synapse_idx) {
            Some(&last) => current_time >= last + cooldown_us,
            None => true, // Never flipped before
        }
    }

    /// Record a flip event.
    pub fn record_flip(&mut self, synapse_idx: usize, time: u64) {
        self.last_flip.insert(synapse_idx, time);
    }

    /// Get time since last flip (or u64::MAX if never flipped).
    pub fn time_since_flip(&self, synapse_idx: usize, current_time: u64) -> u64 {
        match self.last_flip.get(&synapse_idx) {
            Some(&last) => current_time.saturating_sub(last),
            None => u64::MAX,
        }
    }

    /// Clear old entries (housekeeping).
    pub fn prune_old(&mut self, older_than: u64) {
        self.last_flip.retain(|_, &mut time| time >= older_than);
    }
}

/// State tracker for mastery learning.
#[derive(Clone, Debug)]
pub struct MasteryState {
    /// Configuration
    config: MasteryConfig,
    /// Per-synapse pressure accumulation (indexed by synapse index)
    pressure: Vec<i16>,
    /// Metabolic budget remaining
    metabolic_budget: u32,
    /// Flip cooldown tracker
    flip_cooldown: FlipCooldown,
    /// Current simulation time (for cooldown checks)
    current_time: u64,
}

impl MasteryState {
    /// Create a new mastery state for n synapses.
    pub fn new(synapse_count: usize, config: MasteryConfig, initial_budget: u32) -> Self {
        Self {
            config,
            pressure: vec![0; synapse_count],
            metabolic_budget: initial_budget,
            flip_cooldown: FlipCooldown::new(),
            current_time: 0,
        }
    }

    /// Update current simulation time.
    pub fn set_time(&mut self, time_us: u64) {
        self.current_time = time_us;
    }

    /// Resize pressure vector if synapse count changes.
    pub fn resize(&mut self, synapse_count: usize) {
        self.pressure.resize(synapse_count, 0);
    }

    /// Add metabolic budget.
    pub fn add_budget(&mut self, amount: u32) {
        self.metabolic_budget = self.metabolic_budget.saturating_add(amount);
    }

    /// Get remaining metabolic budget.
    pub fn budget(&self) -> u32 {
        self.metabolic_budget
    }

    /// Accumulate pressure for a synapse based on activity and direction.
    ///
    /// - `synapse_idx`: index into synapses
    /// - `activity`: how strongly this synapse participated (0.0-1.0)
    /// - `direction`: signum of (target - output), i.e., -1, 0, or +1
    pub fn accumulate_pressure(&mut self, synapse_idx: usize, activity: f32, direction: i8) {
        if synapse_idx >= self.pressure.len() {
            return;
        }

        // Only top participants accumulate pressure
        if activity < self.config.participation_threshold {
            return;
        }

        let delta = (direction as f32 * activity * self.config.pressure_scale * 10.0) as i16;
        self.pressure[synapse_idx] = self.pressure[synapse_idx].saturating_add(delta);
    }

    /// Apply mastery learning to a synapse based on accumulated pressure.
    ///
    /// Returns the change that occurred (if any).
    pub fn apply_learning(&mut self, synapse_idx: usize, synapse: &mut SpatialSynapse) -> Option<PolarityChange> {
        if synapse_idx >= self.pressure.len() {
            return None;
        }

        let pressure = self.pressure[synapse_idx];
        let abs_pressure = pressure.abs();

        // Not enough pressure yet
        if abs_pressure < self.config.pressure_threshold {
            return None;
        }

        // Determine desired direction
        let desired_positive = pressure > 0;
        let current_positive = synapse.signal.polarity > 0;
        let current_negative = synapse.signal.polarity < 0;

        // Calculate modification cost
        let change = if synapse.signal.polarity == 0 {
            // Currently dormant: wake up in the desired direction
            let cost = self.config.magnitude_cost;
            if !self.try_spend(cost as u32) {
                return None;
            }
            synapse.signal.polarity = if desired_positive { 1 } else { -1 };
            synapse.signal.magnitude = synapse.signal.magnitude.saturating_add(10);
            PolarityChange::Awakened
        } else if (desired_positive && current_positive) || (!desired_positive && current_negative) {
            // Aligned: strengthen
            let cost = self.config.magnitude_cost;
            if !self.try_spend(cost as u32) {
                return None;
            }
            synapse.signal.magnitude = synapse.signal.magnitude.saturating_add(5);
            PolarityChange::Strengthened
        } else {
            // Opposed: weaken first, then flip if depleted
            let cost = self.config.magnitude_cost;
            if !self.try_spend(cost as u32) {
                return None;
            }

            if synapse.signal.magnitude > 5 {
                // Weaken
                synapse.signal.magnitude = synapse.signal.magnitude.saturating_sub(5);
                PolarityChange::Weakened
            } else {
                // Magnitude depleted: flip polarity (expensive!)
                // But first check cooldown
                if !self.flip_cooldown.can_flip(synapse_idx, self.current_time, self.config.flip_cooldown_us) {
                    // Can't flip yet — go dormant instead
                    synapse.signal.magnitude = 0;
                    synapse.signal.polarity = 0;
                    return Some(PolarityChange::GoneDormant);
                }

                let flip_cost = self.config.flip_penalty;
                if !self.try_spend(flip_cost as u32) {
                    synapse.signal.magnitude = 0;
                    synapse.signal.polarity = 0;
                    return Some(PolarityChange::GoneDormant);
                }

                synapse.signal.polarity = if desired_positive { 1 } else { -1 };
                synapse.signal.magnitude = 10;
                self.flip_cooldown.record_flip(synapse_idx, self.current_time);
                PolarityChange::Flipped
            }
        };

        // Clear pressure after update
        self.pressure[synapse_idx] = 0;
        synapse.mature(1); // modifications increase maturity

        Some(change)
    }

    /// Try to spend metabolic budget. Returns false if insufficient.
    fn try_spend(&mut self, cost: u32) -> bool {
        if self.metabolic_budget >= cost {
            self.metabolic_budget -= cost;
            true
        } else {
            false
        }
    }

    /// Clear all accumulated pressure.
    pub fn clear_pressure(&mut self) {
        for p in &mut self.pressure {
            *p = 0;
        }
    }

    /// Get pressure for a synapse.
    pub fn get_pressure(&self, synapse_idx: usize) -> i16 {
        self.pressure.get(synapse_idx).copied().unwrap_or(0)
    }
}

/// What kind of change occurred during learning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolarityChange {
    /// Synapse was strengthened (magnitude increased)
    Strengthened,
    /// Synapse was weakened (magnitude decreased)
    Weakened,
    /// Synapse went dormant (magnitude → 0)
    GoneDormant,
    /// Synapse was awakened from dormancy
    Awakened,
    /// Synapse polarity was flipped (expensive!)
    Flipped,
}

/// Calculate metabolic cost for a proposed signal change.
///
/// Higher cost = more resistance to change.
pub fn modification_cost(current: &Signal, proposed: &Signal) -> u8 {
    if current.polarity == proposed.polarity {
        // Same polarity, just magnitude change — cheap
        let mag_diff = (current.magnitude as i16 - proposed.magnitude as i16).abs();
        (mag_diff / 10) as u8 // ~0-25 cost
    } else if proposed.polarity == 0 {
        // Going dormant — moderate cost
        current.magnitude / 2 // Half the current strength
    } else if current.polarity == 0 {
        // Waking from dormant — moderate cost
        proposed.magnitude / 2
    } else {
        // Polarity flip — EXPENSIVE
        // Must pay: current magnitude + flip penalty + new magnitude
        current
            .magnitude
            .saturating_add(50)
            .saturating_add(proposed.magnitude)
    }
}

/// Compute learning direction from target vs output.
///
/// Returns -1, 0, or +1 (signum only, not magnitude).
#[inline]
pub fn learning_direction(target: i16, output: i16) -> i8 {
    let diff = target as i32 - output as i32;
    if diff > 0 {
        1
    } else if diff < 0 {
        -1
    } else {
        0
    }
}

/// Check if a neuron participated enough to update.
///
/// Uses top-k percentile of activity.
#[inline]
pub fn is_participant(activity: f32, threshold: f32) -> bool {
    activity >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_direction() {
        assert_eq!(learning_direction(100, 50), 1);
        assert_eq!(learning_direction(50, 100), -1);
        assert_eq!(learning_direction(50, 50), 0);
    }

    #[test]
    fn test_modification_cost() {
        let same_pol = Signal::positive(100);
        let stronger = Signal::positive(150);
        assert!(modification_cost(&same_pol, &stronger) < 10);

        let pos = Signal::positive(100);
        let neg = Signal::negative(100);
        assert!(modification_cost(&pos, &neg) > 100);
    }

    #[test]
    fn test_weaken_before_flip() {
        let mut state = MasteryState::new(1, MasteryConfig::default(), 1000);
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);

        // Apply opposing pressure
        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, -1); // want inhibitory
        }

        // First application should weaken, not flip
        let change = state.apply_learning(0, &mut syn);
        assert!(matches!(change, Some(PolarityChange::Weakened)));
        assert!(syn.is_excitatory()); // still positive

        // Keep weakening until flip
        syn.signal.magnitude = 3; // almost depleted
        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, -1);
        }
        let change = state.apply_learning(0, &mut syn);
        assert!(matches!(change, Some(PolarityChange::Flipped)));
        assert!(syn.is_inhibitory()); // now negative
    }

    #[test]
    fn test_metabolic_budget() {
        let mut state = MasteryState::new(1, MasteryConfig::default(), 10);
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);

        // Accumulate pressure
        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, 1);
        }

        // First change succeeds
        let change = state.apply_learning(0, &mut syn);
        assert!(change.is_some());

        // Exhaust budget
        state.metabolic_budget = 0;

        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, 1);
        }

        // Now changes fail
        let change = state.apply_learning(0, &mut syn);
        assert!(change.is_none());
    }

    #[test]
    fn test_participation_threshold() {
        let config = MasteryConfig {
            participation_threshold: 0.5,
            ..Default::default()
        };
        let mut state = MasteryState::new(2, config, 1000);

        // Low activity should not accumulate pressure
        state.accumulate_pressure(0, 0.2, 1);
        assert_eq!(state.get_pressure(0), 0);

        // High activity should accumulate
        state.accumulate_pressure(1, 0.8, 1);
        assert!(state.get_pressure(1) > 0);
    }

    #[test]
    fn test_pressure_threshold() {
        let config = MasteryConfig {
            pressure_threshold: 100,
            ..Default::default()
        };
        let mut state = MasteryState::new(1, config, 1000);
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);

        // Small pressure shouldn't trigger change
        state.accumulate_pressure(0, 1.0, 1);
        state.accumulate_pressure(0, 1.0, 1);
        let change = state.apply_learning(0, &mut syn);
        assert!(change.is_none()); // not enough pressure

        // More pressure should trigger
        for _ in 0..20 {
            state.accumulate_pressure(0, 1.0, 1);
        }
        let change = state.apply_learning(0, &mut syn);
        assert!(change.is_some());
    }

    #[test]
    fn test_dormant_synapse_awakening() {
        let mut state = MasteryState::new(1, MasteryConfig::default(), 1000);
        let mut syn = SpatialSynapse::dormant(0, 1, 1000);

        // Positive pressure should awaken as excitatory
        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, 1);
        }
        let change = state.apply_learning(0, &mut syn);
        assert!(matches!(change, Some(PolarityChange::Awakened)));
        assert!(syn.is_excitatory());

        // Reset for negative test
        let mut syn2 = SpatialSynapse::dormant(0, 1, 1000);
        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, -1);
        }
        let change = state.apply_learning(0, &mut syn2);
        assert!(matches!(change, Some(PolarityChange::Awakened)));
        assert!(syn2.is_inhibitory());
    }

    #[test]
    fn test_strengthening_aligned() {
        let mut state = MasteryState::new(1, MasteryConfig::default(), 1000);
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);
        let original_mag = syn.signal.magnitude;

        // Positive pressure on positive synapse → strengthen
        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, 1);
        }
        let change = state.apply_learning(0, &mut syn);
        assert!(matches!(change, Some(PolarityChange::Strengthened)));
        assert!(syn.signal.magnitude > original_mag);
    }

    #[test]
    fn test_maturity_increases_on_change() {
        let mut state = MasteryState::new(1, MasteryConfig::default(), 1000);
        let mut syn = SpatialSynapse::excitatory(0, 1, 100, 1000);
        let original_maturity = syn.maturity;

        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, 1);
        }
        state.apply_learning(0, &mut syn);
        assert!(syn.maturity > original_maturity);
    }

    #[test]
    fn test_clear_pressure() {
        let mut state = MasteryState::new(3, MasteryConfig::default(), 1000);

        state.accumulate_pressure(0, 1.0, 1);
        state.accumulate_pressure(1, 1.0, -1);
        state.accumulate_pressure(2, 1.0, 1);

        assert!(state.get_pressure(0) > 0);
        assert!(state.get_pressure(1) < 0);

        state.clear_pressure();

        assert_eq!(state.get_pressure(0), 0);
        assert_eq!(state.get_pressure(1), 0);
        assert_eq!(state.get_pressure(2), 0);
    }

    #[test]
    fn test_resize() {
        let mut state = MasteryState::new(2, MasteryConfig::default(), 1000);
        state.accumulate_pressure(0, 1.0, 1);

        // Resize to larger
        state.resize(5);
        assert_eq!(state.get_pressure(4), 0); // new slots initialized to 0

        // Resize to smaller (keeps existing data)
        state.resize(1);
        assert!(state.get_pressure(0) > 0); // existing data preserved
    }

    #[test]
    fn test_add_budget() {
        let mut state = MasteryState::new(1, MasteryConfig::default(), 100);
        assert_eq!(state.budget(), 100);

        state.add_budget(50);
        assert_eq!(state.budget(), 150);
    }

    #[test]
    fn test_modification_cost_dormant() {
        let dormant = Signal::zero();
        let active = Signal::positive(100);

        // Waking from dormant has moderate cost
        let cost = modification_cost(&dormant, &active);
        assert!(cost > 0);
        assert!(cost < 100); // not as expensive as flip
    }

    #[test]
    fn test_is_participant_boundary() {
        // At threshold
        assert!(is_participant(0.25, 0.25));
        // Just below
        assert!(!is_participant(0.24, 0.25));
        // Above
        assert!(is_participant(0.5, 0.25));
    }

    #[test]
    fn test_flip_cost_insufficient_budget() {
        // Start with just enough for weaken, not flip
        let mut state = MasteryState::new(1, MasteryConfig::default(), 10);
        let mut syn = SpatialSynapse::excitatory(0, 1, 3, 1000); // low magnitude

        // Apply opposing pressure
        for _ in 0..10 {
            state.accumulate_pressure(0, 1.0, -1);
        }

        // Should go dormant instead of flipping (insufficient budget for flip)
        let change = state.apply_learning(0, &mut syn);
        assert!(matches!(change, Some(PolarityChange::GoneDormant)));
        assert!(syn.is_dormant());
    }
}
