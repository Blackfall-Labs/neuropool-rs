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

mod zone;
mod neuron;
mod synapse;
mod cascade;
mod grid;
mod wiring;
mod disc;
mod incubate;
mod io;
mod persist;

pub use zone::{DendriticZone, ZoneWeights};
pub use neuron::{UnifiedNeuron, VoxelPosition, DEFAULT_THRESHOLD, RESTING_POTENTIAL, RESET_POTENTIAL, CONTEXT_PRIMING_THRESHOLD};
pub use synapse::{UnifiedSynapse, UnifiedSynapseStore};
pub use cascade::{CascadeEngine, CascadeConfig, SpikeArrival, TernsigTrigger};
pub use grid::{VoxelGrid, VoxelNeighborhood, GridDims};
pub use wiring::{UnifiedWiringConfig, ZoneStrategy, wire_by_proximity};
pub use disc::{ImaginalDisc, RegionArchetype, NucleiDistribution, WiringRules, ZoneBias};
pub use incubate::{IncubatedPool, IncubateConfig, incubate};
pub use io::{UnifiedNeuronIO, NullUnifiedIO};
pub use persist::UnifiedPool;
