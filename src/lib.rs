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
//! ## v4: Spatial Brain
//!
//! Neurons exist in continuous 3D space with physical properties:
//! - `soma_position`: where the cell body is
//! - `axon_terminal`: where the axon ends
//! - `dendrite_radius`: local reception range
//! - `axon_health`: survival pressure (0=dead, 255=myelinated)
//!
//! Connections form dynamically based on proximity and activity correlation.
//! Regions emerge from neuron clustering, not predefined boxes.

pub mod binding;
pub mod io;
pub mod neuron;
pub mod synapse;
pub mod pool;
pub mod plasticity;
pub mod codec;
pub mod stats;
pub mod density;
pub mod template;

pub use binding::{BindingConfig, BindingTable};
pub use io::{NeuronIO, NullIO};
pub use neuron::{NeuronArrays, NeuronProfile, NeuronType};
pub use synapse::{Synapse, SynapseStore, ThermalState};
pub use pool::{BindingSpec, EvolutionConfig, EvolutionResult, FitnessInput, GrowthConfig, MutationEntry, MutationJournal, MutationType, NeuronPool, PoolCheckpoint, PoolConfig, SpatialDims, TypeDistributionSpec};
pub use plasticity::GrowthResult;
pub use stats::{PoolStats, ThermalDistribution, TypeDistribution};
pub use density::{Conductivity, DensityField, TissueType};
pub use template::{SignalType, TemplateType, TemplateRequest, TemplateInstance, TemplateRegistry, SpatialArrangement};
