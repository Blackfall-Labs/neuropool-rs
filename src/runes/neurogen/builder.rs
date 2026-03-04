//! NeurogenBuilder — accumulator state for neurogen program evaluation.
//!
//! The builder is the host type for all neurogen modules. It accumulates
//! region definitions, trophic gradients, differentiation discs, and tract
//! declarations during program evaluation. After evaluation completes, the
//! cold-boot harness converts these specs into neuropool imaginal disc calls.

use crate::RegionArchetype;

// ---------------------------------------------------------------------------
// Spec types — accumulated during program evaluation
// ---------------------------------------------------------------------------

/// A trophic gradient point within a region.
#[derive(Clone, Debug)]
pub struct GradientSpec {
    /// Gradient name (referenced by differentiation discs via `near`).
    pub name: String,
    /// Gradient strength (arbitrary units, higher = stronger attraction).
    pub strength: i64,
    /// Influence radius as percentage of region volume (0-100).
    pub radius: i64,
}

/// Target neuron type for a differentiation disc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscTarget {
    Pyramidal,
    Interneuron,
    Gate,
    Relay,
    Oscillator,
    Memory,
    Sensory,
    Motor,
    Computational,
}

impl DiscTarget {
    /// Parse from a Runes symbol string.
    pub fn from_symbol(s: &str) -> Option<Self> {
        match s {
            "pyramidal" => Some(Self::Pyramidal),
            "interneuron" => Some(Self::Interneuron),
            "gate" => Some(Self::Gate),
            "relay" => Some(Self::Relay),
            "oscillator" => Some(Self::Oscillator),
            "memory" => Some(Self::Memory),
            "sensory" => Some(Self::Sensory),
            "motor" => Some(Self::Motor),
            "computational" => Some(Self::Computational),
            _ => None,
        }
    }
}

/// A differentiation disc specification — defines what neurons become.
#[derive(Clone, Debug)]
pub struct DiscSpec {
    /// Disc name (for diagnostics and testing).
    pub name: String,
    /// Gradient this disc is attracted to (set via `near`).
    pub near_gradient: Option<String>,
    /// What neurons in this disc differentiate into.
    pub target: DiscTarget,
    /// Pressure threshold for differentiation (lower = earlier).
    pub threshold: i64,
    /// Maximum percentage of region neurons this disc can claim (0-100).
    pub population_cap: i64,
    /// Runes program to bind to neurons that differentiate here.
    pub bind_program: Option<String>,
    /// Oscillator period range in microseconds (only for oscillator targets).
    pub period_range: Option<(u32, u32)>,
    /// Spatial distribution: None = near gradient, Some("even") = spread across region.
    pub spread: Option<String>,
}

impl DiscSpec {
    /// Create a new empty disc spec with the given name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            near_gradient: None,
            target: DiscTarget::Pyramidal,
            threshold: 50,
            population_cap: 10,
            bind_program: None,
            period_range: None,
            spread: None,
        }
    }
}

/// Whitematter tract type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TractType {
    /// Intra-hemispheric connection between adjacent cortical regions.
    Association,
    /// Cortical ↔ subcortical pathway.
    Projection,
    /// Inter-hemispheric connection.
    Commissural,
}

impl TractType {
    pub fn from_symbol(s: &str) -> Option<Self> {
        match s {
            "association" => Some(Self::Association),
            "projection" => Some(Self::Projection),
            "commissural" => Some(Self::Commissural),
            _ => None,
        }
    }
}

/// A whitematter tract declaration.
#[derive(Clone, Debug)]
pub struct TractSpec {
    /// Tract name (e.g., "thalamocortical").
    pub name: String,
    /// Source region name.
    pub from: String,
    /// Target region name.
    pub to: String,
    /// Tract type.
    pub tract_type: TractType,
    /// Fiber count (optional — harness assigns defaults).
    pub fiber_count: Option<u32>,
}

/// Developmental phase durations in frames.
#[derive(Clone, Copy, Debug)]
pub struct PhaseDurations {
    /// Self-organization with no disc pressure.
    pub genesis: u32,
    /// Sensors feeding, discs accumulate pressure.
    pub exposure: u32,
    /// Neurons transform irreversibly.
    pub differentiation: u32,
    /// Structure freezes, plasticity drops.
    pub crystallization: u32,
}

impl Default for PhaseDurations {
    fn default() -> Self {
        Self {
            genesis: 500,
            exposure: 2000,
            differentiation: 1000,
            crystallization: 1000,
        }
    }
}

/// A complete region specification accumulated during neurogen evaluation.
#[derive(Clone, Debug)]
pub struct RegionSpec {
    /// Region name.
    pub name: String,
    /// Region archetype (determines topology, default distribution, wiring).
    pub archetype: RegionArchetype,
    /// Target neuron count.
    pub neuron_count: u32,
    /// Neighbor region names (for association tract wiring).
    pub neighbors: Vec<String>,
    /// Trophic gradient points within this region.
    pub gradients: Vec<GradientSpec>,
    /// Differentiation disc specifications.
    pub discs: Vec<DiscSpec>,
    /// Developmental phase durations.
    pub phase_durations: PhaseDurations,
}

impl RegionSpec {
    /// Create a new region spec with defaults.
    pub fn new(name: String) -> Self {
        Self {
            name,
            archetype: RegionArchetype::Cortical,
            neuron_count: 100,
            neighbors: Vec::new(),
            gradients: Vec::new(),
            discs: Vec::new(),
            phase_durations: PhaseDurations::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// NeurogenBuilder — the host state
// ---------------------------------------------------------------------------

/// Accumulator state for neurogen program evaluation.
///
/// This is the host type that all neurogen module verbs and constructs
/// operate on. The cold-boot execution harness creates it, sets it as the
/// Runes engine host, evaluates neurogen programs, then reads the
/// accumulated specs to drive neuropool incubation.
pub struct NeurogenBuilder {
    /// Completed region specifications.
    pub regions: Vec<RegionSpec>,
    /// Whitematter tract declarations.
    pub tracts: Vec<TractSpec>,

    // --- Construct scope stack ---

    /// Index into `regions` for the currently-being-defined region.
    /// Set by `region do...end` construct enter, cleared by exit.
    active_region: Option<usize>,
    /// Disc being configured inside a `differentiate do...end` block.
    /// Committed to the active region on construct exit.
    active_disc: Option<DiscSpec>,

    // --- Budget tracking ---

    /// Total metabolic budget for the organism.
    total_budget: i64,
    /// Budget consumed so far (each neuron/synapse costs budget).
    spent_budget: i64,
}

impl NeurogenBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            tracts: Vec::new(),
            active_region: None,
            active_disc: None,
            total_budget: 100_000,
            spent_budget: 0,
        }
    }

    /// Create with a custom metabolic budget.
    pub fn with_budget(budget: i64) -> Self {
        let mut b = Self::new();
        b.total_budget = budget;
        b
    }

    // --- Region scope ---

    /// Begin a new region. Returns the region index.
    pub fn begin_region(&mut self, name: String) -> usize {
        let idx = self.regions.len();
        self.regions.push(RegionSpec::new(name));
        self.active_region = Some(idx);
        idx
    }

    /// Finish the current region. Returns the index.
    pub fn end_region(&mut self) -> Option<usize> {
        let idx = self.active_region.take();
        idx
    }

    /// Get the active region mutably, or None if not inside a region block.
    pub fn active_region_mut(&mut self) -> Option<&mut RegionSpec> {
        self.active_region
            .and_then(|idx| self.regions.get_mut(idx))
    }

    /// Whether we're inside a `region do...end` block.
    pub fn in_region(&self) -> bool {
        self.active_region.is_some()
    }

    // --- Disc scope ---

    /// Begin a new differentiation disc.
    pub fn begin_disc(&mut self, name: String) {
        self.active_disc = Some(DiscSpec::new(name));
    }

    /// Finish the current disc, committing it to the active region.
    /// Returns false if no active region.
    pub fn end_disc(&mut self) -> bool {
        if let Some(disc) = self.active_disc.take() {
            if let Some(region) = self.active_region_mut() {
                region.discs.push(disc);
                return true;
            }
        }
        false
    }

    /// Get the active disc mutably.
    pub fn active_disc_mut(&mut self) -> Option<&mut DiscSpec> {
        self.active_disc.as_mut()
    }

    /// Whether we're inside a `differentiate do...end` block.
    pub fn in_disc(&self) -> bool {
        self.active_disc.is_some()
    }

    // --- Budget ---

    /// Total metabolic budget.
    pub fn total_budget(&self) -> i64 {
        self.total_budget
    }

    /// Budget spent so far.
    pub fn spent_budget(&self) -> i64 {
        self.spent_budget
    }

    /// Remaining budget.
    pub fn remaining_budget(&self) -> i64 {
        self.total_budget - self.spent_budget
    }

    /// Set the total budget.
    pub fn set_budget(&mut self, budget: i64) {
        self.total_budget = budget;
    }

    /// Spend budget. Returns false if insufficient.
    pub fn spend(&mut self, amount: i64) -> bool {
        if self.spent_budget + amount > self.total_budget {
            return false;
        }
        self.spent_budget += amount;
        true
    }
}

impl Default for NeurogenBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_region_lifecycle() {
        let mut b = NeurogenBuilder::new();
        assert!(!b.in_region());

        let idx = b.begin_region("brainstem".to_string());
        assert_eq!(idx, 0);
        assert!(b.in_region());

        b.active_region_mut().unwrap().archetype = RegionArchetype::Brainstem;
        b.active_region_mut().unwrap().neuron_count = 200;

        b.end_region();
        assert!(!b.in_region());
        assert_eq!(b.regions.len(), 1);
        assert_eq!(b.regions[0].name, "brainstem");
        assert_eq!(b.regions[0].archetype, RegionArchetype::Brainstem);
        assert_eq!(b.regions[0].neuron_count, 200);
    }

    #[test]
    fn builder_disc_lifecycle() {
        let mut b = NeurogenBuilder::new();
        b.begin_region("brainstem".to_string());

        b.begin_disc("respiratory_osc".to_string());
        assert!(b.in_disc());

        b.active_disc_mut().unwrap().target = DiscTarget::Oscillator;
        b.active_disc_mut().unwrap().threshold = 25;
        b.active_disc_mut().unwrap().population_cap = 15;

        assert!(b.end_disc());
        assert!(!b.in_disc());

        let region = b.active_region_mut().unwrap();
        assert_eq!(region.discs.len(), 1);
        assert_eq!(region.discs[0].name, "respiratory_osc");
        assert_eq!(region.discs[0].target, DiscTarget::Oscillator);

        b.end_region();
    }

    #[test]
    fn builder_disc_without_region_fails() {
        let mut b = NeurogenBuilder::new();
        b.begin_disc("orphan".to_string());
        assert!(!b.end_disc()); // no active region
    }

    #[test]
    fn builder_budget_tracking() {
        let mut b = NeurogenBuilder::with_budget(1000);
        assert_eq!(b.remaining_budget(), 1000);

        assert!(b.spend(400));
        assert_eq!(b.remaining_budget(), 600);

        assert!(b.spend(600));
        assert_eq!(b.remaining_budget(), 0);

        assert!(!b.spend(1)); // over budget
    }

    #[test]
    fn disc_target_from_symbol() {
        assert_eq!(DiscTarget::from_symbol("oscillator"), Some(DiscTarget::Oscillator));
        assert_eq!(DiscTarget::from_symbol("relay"), Some(DiscTarget::Relay));
        assert_eq!(DiscTarget::from_symbol("motor"), Some(DiscTarget::Motor));
        assert_eq!(DiscTarget::from_symbol("nonsense"), None);
    }

    #[test]
    fn tract_type_from_symbol() {
        assert_eq!(TractType::from_symbol("association"), Some(TractType::Association));
        assert_eq!(TractType::from_symbol("projection"), Some(TractType::Projection));
        assert_eq!(TractType::from_symbol("commissural"), Some(TractType::Commissural));
        assert_eq!(TractType::from_symbol("unknown"), None);
    }

    #[test]
    fn multiple_regions() {
        let mut b = NeurogenBuilder::new();

        b.begin_region("brainstem".to_string());
        b.active_region_mut().unwrap().archetype = RegionArchetype::Brainstem;
        b.end_region();

        b.begin_region("thalamus".to_string());
        b.active_region_mut().unwrap().archetype = RegionArchetype::Thalamic;
        b.end_region();

        assert_eq!(b.regions.len(), 2);
        assert_eq!(b.regions[0].name, "brainstem");
        assert_eq!(b.regions[1].name, "thalamus");
    }

    #[test]
    fn multiple_discs_in_region() {
        let mut b = NeurogenBuilder::new();
        b.begin_region("brainstem".to_string());

        for (name, target) in &[
            ("osc1", DiscTarget::Oscillator),
            ("relay1", DiscTarget::Relay),
            ("motor1", DiscTarget::Motor),
        ] {
            b.begin_disc(name.to_string());
            b.active_disc_mut().unwrap().target = *target;
            b.end_disc();
        }

        let region = b.active_region_mut().unwrap();
        assert_eq!(region.discs.len(), 3);
        b.end_region();
    }
}
