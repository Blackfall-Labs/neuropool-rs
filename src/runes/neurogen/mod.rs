//! Neurogen namespace modules — grow brain tissue via Runes programs.
//!
//! Unlike the nuclei modules (`:signal`, `:learning`, `:cascade`) which are
//! pure functions, neurogen modules are host-backed. They operate on a shared
//! [`NeurogenBuilder`] that accumulates region specifications during program
//! evaluation. The cold-boot execution harness creates the builder, evaluates
//! neurogen `.rune` programs against it, then converts the accumulated specs
//! into neuropool imaginal disc incubation calls.
//!
//! ## Modules
//!
//! - `:trophic` — Growth factor verbs (gradient establishment)
//! - `:develop` — Tissue generation (region/differentiate constructs + config verbs)
//! - `:budget` — Metabolic allocation tracking
//!
//! ## Host Requirement
//!
//! All neurogen verbs and constructs downcast `EvalContext.host` to
//! `&mut NeurogenBuilder`. The execution harness must provide this as the host.
//!
//! Requires the `runes` feature flag.

mod builder;
mod trophic;
mod develop;
mod budget;
mod harness;

pub use builder::{
    NeurogenBuilder, RegionSpec, GradientSpec, DiscSpec, DiscTarget,
    TractSpec, TractType, PhaseDurations,
};
pub use trophic::TrophicModule;
pub use develop::DevelopModule;
pub use budget::BudgetModule;
pub use harness::{execute, hash_string, incubate_region, HarnessConfig, IncubatedRegion, NeurogenResult};
