//! # neuropool
//!
//! Neuromorphic neuron pool substrate with spatial biology and dendritic zone processing.
//!
//! ## Three Neuron Systems
//!
//! ### Unified System (Current — v2.0)
//!
//! Zone-aware neurons with dendritic compartments, event-driven cascade execution,
//! imaginal disc incubation, and binary persistence:
//!
//! - [`UnifiedNeuron`] — integer voxel position + spatial biology + dendritic zones
//! - [`CascadeEngine`] — zone-aware event-driven cascade with predicted/burst firing
//! - [`ImaginalDisc`] — developmental blueprints for region archetypes
//! - [`UnifiedPool`] — save/load persistence
//! - [`UnifiedNeuronIO`] — external IO trait for memory/motor/sensory dispatch
//!
//! ### Voxel System (Deprecated — v1.0)
//!
//! The original tick-based SNN with `NeuronType` enum. Still functional but
//! superseded by the unified system. Will be removed in a future major version.
//!
//! ### Spatial System (Deprecated — v1.0)
//!
//! The intermediate event-driven system with float positions. Anatomy types
//! (`Nuclei`, `Dendrite`, `Axon`, `Interface`) are reused by the unified system.
//! The spatial execution engine (`SpatialRuntime`) is deprecated.

// === Voxel System (Deprecated — use unified:: instead) ===
#[deprecated(since = "2.0.0", note = "Use neuropool::unified or neuropool::pool directly. Voxel binding system superseded by unified::CascadeEngine with UnifiedNeuronIO.")]
pub mod binding;
#[deprecated(since = "2.0.0", note = "Use neuropool::unified::CascadeEngine instead.")]
pub mod cascade;
#[deprecated(since = "2.0.0", note = "Use neuropool::unified::persist (UnifiedPool::save/load) instead.")]
pub mod codec;
#[deprecated(since = "2.0.0", note = "Density field superseded by unified voxel grid.")]
pub mod density;
#[deprecated(since = "2.0.0", note = "Use neuropool::unified::UnifiedNeuronIO instead.")]
pub mod io;
#[deprecated(since = "2.0.0", note = "Use neuropool::unified::UnifiedNeuron instead.")]
pub mod neuron;
#[deprecated(since = "2.0.0", note = "Plasticity integrated into unified cascade engine.")]
pub mod plasticity;
pub mod pool;
#[deprecated(since = "2.0.0", note = "Pool stats superseded by CascadeEngine counters.")]
pub mod stats;
#[deprecated(since = "2.0.0", note = "Use neuropool::unified::UnifiedSynapse instead.")]
pub mod synapse;
#[deprecated(since = "2.0.0", note = "Use neuropool::unified::ImaginalDisc instead.")]
pub mod template;

// === Spatial System (Deprecated — anatomy types reused by unified) ===
#[deprecated(since = "2.0.0", note = "Anatomy types (Nuclei, Dendrite, Axon, Interface) reused by unified system. SpatialRuntime superseded by CascadeEngine.")]
pub mod spatial;

// === Unified System (Current) ===
pub mod unified;

// === Voxel Re-exports (deprecated) ===
#[allow(deprecated)]
pub use binding::{BindingConfig, BindingTable};
#[allow(deprecated)]
pub use cascade::{CascadeConfig, CascadePool, SpikeArrival};
#[allow(deprecated)]
pub use density::{Conductivity, DensityField, TissueType};
#[allow(deprecated)]
pub use io::{NeuronIO, NullIO};
#[allow(deprecated)]
pub use neuron::{NeuronArrays, NeuronProfile, NeuronType};
#[allow(deprecated)]
pub use plasticity::GrowthResult;
pub use pool::{BindingSpec, EvolutionConfig, EvolutionResult, FitnessInput, GrowthConfig, MutationEntry, MutationJournal, MutationType, NeuronPool, PoolCheckpoint, PoolConfig, SpatialDims, TypeDistributionSpec};
#[allow(deprecated)]
pub use stats::{PoolStats, ThermalDistribution, TypeDistribution};
#[allow(deprecated)]
pub use synapse::{Synapse, SynapseStore, ThermalState};
#[allow(deprecated)]
pub use template::{SignalType, TemplateType, TemplateRequest, TemplateInstance, TemplateRegistry, SpatialArrangement};

// === Spatial Re-exports (deprecated) ===
#[allow(deprecated)]
pub use spatial::{
    Axon, Dendrite, EnergyGates, Interface, InterfaceAction,
    MasteryConfig, MasteryState, Nuclei, PolarityChange,
    Polarity, Signal, SpatialNeuron, SpatialSynapse, Soma,
    SpatialRuntime, SpatialRuntimeConfig, WiringConfig, wire_by_proximity,
};

// === Unified Re-exports (current) ===
// Note: CascadeConfig, SpikeArrival exist in both voxel cascade and unified cascade.
// The unified versions are accessed via `unified::CascadeConfig`, `unified::SpikeArrival`.
// Note: VoxelGrid, GridDims, VoxelNeighborhood, UnifiedWiringConfig, ZoneStrategy,
// and unified::wire_by_proximity are accessed via `unified::` prefix to avoid
// name conflicts with the spatial system's wire_by_proximity and WiringConfig.
pub use unified::{
    DendriticZone, ZoneWeights,
    UnifiedNeuron, VoxelPosition,
    UnifiedSynapse, UnifiedSynapseStore,
    CascadeEngine, TernsigTrigger,
    VoxelGrid, VoxelNeighborhood, GridDims,
    ImaginalDisc, RegionArchetype, NucleiDistribution, WiringRules, ZoneBias,
    IncubatedPool, IncubateConfig,
    UnifiedNeuronIO, NullUnifiedIO, UnifiedPool,
};
