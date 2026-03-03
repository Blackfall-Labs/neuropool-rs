//! # Unified Neurons — Voxel Structure + Spatial Biology + Dendritic Zones
//!
//! Merges the voxel grid structure with the spatial neuron biology system.
//! Adds three dendritic zone compartments (feedforward, context, feedback)
//! for HTM-inspired predictive processing.
//!
//! ## What's New vs Spatial
//!
//! - **Integer positions**: `VoxelPosition` (voxel + local coords) replaces `Soma { [f32; 3] }`
//! - **Dendritic zones**: Three compartments per neuron for zone-specific integration
//! - **Zone weights**: Per-nuclei-type weights determine how zones combine into membrane
//! - **Predicted firing**: Context priming enables predictive vs burst firing modes
//! - **Zone-aware synapses**: Each synapse targets a specific dendritic zone
//!
//! ## What's Reused from Spatial
//!
//! - `Dendrite`, `Axon` — anatomy structs unchanged
//! - `Nuclei` — capability machine, extended with `ternsig()` preset and `mutate_to()`
//! - `Interface` — external interface system, extended with ternsig kind
//! - All nuclei factory presets (pyramidal, interneuron, gate, relay, etc.)
//! - Learning pressure, maturity, eligibility traces — same mechanics

mod cascade;
mod disc;
mod grid;
mod incubate;
mod io;
mod neuron;
mod persist;
mod synapse;
mod wiring;
mod zone;

pub use cascade::{CascadeConfig, CascadeEngine, SpikeArrival, TernsigTrigger};
pub use disc::{ImaginalDisc, NucleiDistribution, RegionArchetype, WiringRules, ZoneBias};
pub use grid::{GridDims, VoxelGrid, VoxelNeighborhood};
pub use incubate::{incubate, IncubateConfig, IncubatedPool};
pub use io::{NullUnifiedIO, UnifiedNeuronIO};
pub use neuron::{
    UnifiedNeuron, VoxelPosition, CONTEXT_PRIMING_THRESHOLD, DEFAULT_THRESHOLD, RESET_POTENTIAL,
    RESTING_POTENTIAL,
};
pub use persist::UnifiedPool;
pub use synapse::{UnifiedSynapse, UnifiedSynapseStore};
pub use wiring::{wire_by_proximity, UnifiedWiringConfig, ZoneStrategy};
pub use zone::{DendriticZone, ZoneWeights};
