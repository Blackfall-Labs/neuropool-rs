//! # neuropool
//!
//! Biological neuron pool substrate for neuromorphic computing.
//!
//! Provides LIF (Leaky Integrate-and-Fire) neurons with CSR-format synaptic
//! connections, eligibility traces for STDP, three-factor plasticity gated by
//! neuromodulators, and a thermal maturity lifecycle for synapses.
//!
//! This crate replaces both the SNN dynamics engine and the thermogram weight
//! store with a single unified substrate where neurons spike AND persist, and
//! synapses conduct AND have thermal lifecycle.
//!
//! ## Two Neuron Systems
//!
//! ### Voxel Neurons (Legacy SNN)
//!
//! The original tick-based system using `NeuronType` enum:
//! - `pool::NeuronPool` — synchronous tick() execution
//! - `neuron::NeuronArrays` — SoA storage with type flags
//! - `synapse::SynapseStore` — CSR format with i8 weights
//!
//! ### Spatial Neurons (Biological)
//!
//! The new event-driven system with real anatomy:
//! - `spatial::SpatialNeuron` — soma, dendrite, axon, nuclei
//! - `spatial::Nuclei` — capability machines, not enum categories
//! - `spatial::SpatialSynapse` — ternary signals (polarity × magnitude)
//! - `cascade::CascadePool` — event-driven execution
//!
//! See `spatial` module for the biological neuron architecture.

// === Voxel System (Legacy SNN) ===
pub mod binding;
pub mod cascade;
pub mod codec;
pub mod density;
pub mod io;
pub mod neuron;
pub mod plasticity;
pub mod pool;
pub mod stats;
pub mod synapse;
pub mod template;

// === Spatial System (Biological Neurons) ===
pub mod spatial;

// === Voxel Re-exports ===
pub use binding::{BindingConfig, BindingTable};
pub use cascade::{CascadeConfig, CascadePool, SpikeArrival};
pub use density::{Conductivity, DensityField, TissueType};
pub use io::{NeuronIO, NullIO};
pub use neuron::{NeuronArrays, NeuronProfile, NeuronType};
pub use plasticity::GrowthResult;
pub use pool::{BindingSpec, EvolutionConfig, EvolutionResult, FitnessInput, GrowthConfig, MutationEntry, MutationJournal, MutationType, NeuronPool, PoolCheckpoint, PoolConfig, SpatialDims, TypeDistributionSpec};
pub use stats::{PoolStats, ThermalDistribution, TypeDistribution};
pub use synapse::{Synapse, SynapseStore, ThermalState};
pub use template::{SignalType, TemplateType, TemplateRequest, TemplateInstance, TemplateRegistry, SpatialArrangement};

// === Spatial Re-exports ===
pub use spatial::{
    Axon, Dendrite, EnergyGates, Interface, InterfaceAction,
    MasteryConfig, MasteryState, Nuclei, PolarityChange,
    Polarity, Signal, SpatialNeuron, SpatialSynapse, Soma,
    SpatialRuntime, SpatialRuntimeConfig, WiringConfig, wire_by_proximity,
};
