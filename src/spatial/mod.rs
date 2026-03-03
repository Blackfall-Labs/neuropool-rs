#![allow(deprecated)]
//! # Spatial Neurons — Biologically Accurate Point-Cloud Neurons
//!
//! This module implements the "new" neuron system with real anatomy and physics.
//! Unlike voxel neurons (tick-based, NeuronType enum), spatial neurons have:
//!
//! - **Physical structure**: Soma, dendrite, axon with real 3D positions
//! - **Nuclei as capability machines**: Properties determine behavior, not enum categories
//! - **Event-driven execution**: Cascade model, local causality only
//! - **Tissue physics**: Gray/white matter from density, signal delay/attenuation
//! - **Synaptic pruning**: Connections die, neurons persist
//! - **Ternary signals**: polarity × magnitude for genuine inhibition/excitation
//!
//! ## Architecture
//!
//! ```text
//! SpatialNeuron
//! ├── Soma { position }           — where the cell body lives
//! ├── Dendrite { radius, spines } — reception apparatus
//! ├── Axon { terminal, myelin, health } — transmission apparatus
//! ├── Nuclei                      — capability machine (physical properties)
//! │   ├── soma_size, axon_affinity, myelin_affinity, metabolic_rate
//! │   ├── leak, refractory, oscillation_period
//! │   ├── interface (kind: u8 + handlers)
//! │   └── polarity (default ternary)
//! └── Electrical state (membrane, threshold, trace)
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use neuropool::spatial::{SpatialNeuron, Nuclei, Soma, Dendrite, Axon};
//!
//! // Create a pyramidal neuron
//! let neuron = SpatialNeuron::new(
//!     Soma::at([5.0, 5.0, 5.0]),
//!     Dendrite::default(),
//!     Axon::toward([8.0, 5.0, 5.0]),
//!     Nuclei::pyramidal(),
//! );
//! ```

// === Core Data Structures ===
mod anatomy;
mod interface;
mod neuron;
mod nuclei;
mod plasticity;
mod synapse;

// === Execution & Dynamics ===
mod cascade;
mod migration;
mod pruning;
mod runtime;
mod tissue;
mod wiring;

// === Integration Tests ===
#[cfg(test)]
mod integration_test;
#[cfg(test)]
mod load_test;
#[cfg(test)]
mod snapshot;

// === Core Exports ===
pub use anatomy::{Axon, Dendrite, Soma};
pub use interface::{EnergyGates, Interface, InterfaceAction};
pub use neuron::SpatialNeuron;
pub use nuclei::Nuclei;
pub use plasticity::{
    is_participant, learning_direction, modification_cost, FlipCooldown, HubTracker, MasteryConfig,
    MasteryState, PolarityChange,
};
pub use synapse::{SpatialSynapse, SpatialSynapseStore};

// === Execution Exports ===
pub use cascade::{SpatialCascade, SpatialCascadeConfig, SpikeArrival};
pub use migration::{
    apply_migration, compute_migration_forces, migrate_step, CorrelationEntry, CorrelationTracker,
    MigrationConfig,
};
pub use pruning::{hard_prune, pruning_cycle, DormancyTracker, PruningConfig, PruningResult};
pub use runtime::{LearningCounters, SpatialRuntime, SpatialRuntimeConfig, StructuralCounters};
pub use tissue::{
    detect_regions, AxonSegment, EmergentRegion, GridKey, NucleiSignature, RegionConfig,
    SpatialHash, TissueConfig, TissueField, TissueType,
};
pub use wiring::{wire_by_proximity, WiringConfig};

// Re-export ternary_signal types for convenience
pub use ternary_signal::{Polarity, Signal};
