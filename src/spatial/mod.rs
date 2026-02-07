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
mod nuclei;
mod neuron;
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
pub use nuclei::Nuclei;
pub use neuron::SpatialNeuron;
pub use plasticity::{
    MasteryConfig, MasteryState, PolarityChange,
    HubTracker, FlipCooldown,
    learning_direction, modification_cost, is_participant,
};
pub use synapse::{SpatialSynapse, SpatialSynapseStore};

// === Execution Exports ===
pub use cascade::{SpatialCascade, SpatialCascadeConfig, SpikeArrival};
pub use migration::{MigrationConfig, CorrelationTracker, CorrelationEntry, compute_migration_forces, apply_migration, migrate_step};
pub use pruning::{PruningConfig, PruningResult, DormancyTracker, pruning_cycle, hard_prune};
pub use runtime::{SpatialRuntime, SpatialRuntimeConfig, LearningCounters, StructuralCounters};
pub use wiring::{WiringConfig, wire_by_proximity};
pub use tissue::{
    TissueField, TissueConfig, TissueType,
    SpatialHash, GridKey, AxonSegment,
    EmergentRegion, RegionConfig, NucleiSignature,
    detect_regions,
};

// Re-export ternary_signal types for convenience
pub use ternary_signal::{Polarity, Signal};
