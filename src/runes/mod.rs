//! Runes modules shipped with neuropool.
//!
//! ## Nuclei Namespace Modules (pure functions, no host access)
//!
//! - [`SignalModule`] — ternary signal math verbs
//! - [`LearningModule`] — mastery learning verbs
//! - [`CascadeModule`] — neuron firing context verbs
//!
//! ## Neurogen Namespace Modules (host-backed, require NeurogenBuilder)
//!
//! - [`TrophicModule`] — growth factor / trophic gradient verbs
//! - [`DevelopModule`] — tissue generation constructs + config verbs
//! - [`BudgetModule`] — metabolic allocation tracking
//!
//! Requires the `runes` feature flag.

mod signal;
mod learning;
mod cascade;
pub mod neurogen;

pub use signal::SignalModule;
pub use learning::LearningModule;
pub use cascade::CascadeModule;
pub use neurogen::{
    NeurogenBuilder, TrophicModule, DevelopModule, BudgetModule,
    RegionSpec, GradientSpec, DiscSpec, DiscTarget,
    TractSpec, TractType, PhaseDurations,
    execute, hash_string, incubate_region,
    HarnessConfig, IncubatedRegion, NeurogenResult,
};
