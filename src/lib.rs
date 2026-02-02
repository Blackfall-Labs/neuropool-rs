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

pub mod binding;
pub mod io;
pub mod neuron;
pub mod synapse;
pub mod pool;
pub mod plasticity;
pub mod codec;
pub mod stats;

pub use binding::{BindingConfig, BindingTable};
pub use io::{NeuronIO, NullIO};
pub use neuron::{NeuronArrays, NeuronProfile, NeuronType};
pub use synapse::{Synapse, SynapseStore, ThermalState};
pub use pool::{BindingSpec, EvolutionConfig, EvolutionResult, FitnessInput, GrowthConfig, MutationEntry, MutationJournal, MutationType, NeuronPool, PoolCheckpoint, PoolConfig, SpatialDims, TypeDistributionSpec};
pub use plasticity::GrowthResult;
pub use stats::{PoolStats, ThermalDistribution, TypeDistribution};
