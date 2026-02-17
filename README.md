# neuropool-rs

Neuromorphic neuron pool substrate for ternary signal processing. Zone-aware cascade engine with dendritic compartments, imaginal disc incubation, and binary persistence.

## What This Is

A spatial neuron pool where thousands of Leaky Integrate-and-Fire neurons live in a 3D voxel grid, connected by zone-targeted synapses. Each neuron has three dendritic compartments (feedforward, context, feedback) that enable predictive processing — context-primed neurons fire cleanly while unprimed neurons burst and alert their neighbors.

Pools are grown from **imaginal discs** — developmental blueprints that specify region archetype (cortical, thalamic, hippocampal, etc.), nuclei distribution, and wiring topology. After wiring, oscillator neurons settle the network to dynamic equilibrium before external input arrives.

## Architecture (Unified System — v2.0)

```
UnifiedNeuron
├── VoxelPosition { voxel: (u16,u16,u16), local: (u8,u8,u8) }
├── Dendrite { radius, spine_count }
├── Axon { terminal, myelin, health }
├── Nuclei (capability machine — physical properties, not enum categories)
│   ├── soma_size, axon_affinity, myelin_affinity, metabolic_rate
│   ├── leak, refractory, oscillation_period
│   ├── interface (kind: u8, target, modality, energy gates)
│   └── polarity (ternary: excitatory/inhibitory/silent)
├── Zone Potentials
│   ├── feedforward_potential: i16  (bottom-up input)
│   ├── context_potential: i16      (lateral/same-region)
│   └── feedback_potential: i16     (top-down modulatory)
├── ZoneWeights (per-nuclei-type weighted average)
├── Electrical State (membrane, threshold, trace, predicted flag)
└── Metabolic State (stamina, timing)

UnifiedSynapse
├── source, target: u32
├── zone: DendriticZone (Feedforward | Context | Feedback)
├── signal: Signal (polarity × magnitude)
├── delay_us, maturity, pressure
└── CSR storage (UnifiedSynapseStore)

CascadeEngine
├── Event-driven spike propagation (min-heap by arrival time)
├── Zone-aware integration (per-zone → weighted combine → membrane)
├── Predicted vs burst firing (context priming → clean spike / lateral alerts)
├── Oscillator entrainment (periodic context injection)
├── Ternsig triggering (collective activation → program dispatch)
├── Threshold jitter (deterministic per-event variation)
├── Spontaneous depolarization (breaks synchrony)
└── External IO dispatch (UnifiedNeuronIO trait)
```

## Key Properties

- **Integer positions**: Voxel grid + sub-voxel local offsets (no floats in spatial math)
- **Dendritic zones**: Three compartments per neuron with nuclei-specific zone weights
- **Predicted firing**: Context priming enables predictive processing vs burst alerting
- **Zone-aware synapses**: Each synapse targets a specific dendritic compartment
- **Imaginal disc incubation**: Grow pools from archetype blueprints (cortical, thalamic, hippocampal, basal ganglia, cerebellar, brainstem)
- **Threshold jitter**: Deterministic per-event threshold variation breaks artificial synchrony
- **Spontaneous depolarization**: Low-probability random membrane bumps ensure neurons are never silent
- **Binary persistence**: `.upool` format with CRC32 integrity, full neuron + synapse + disc round-trip
- **External IO**: `UnifiedNeuronIO` trait for memory bank query/write, motor output, sensory input

## Region Archetypes

| Archetype | Pyramidal | Interneuron | Gate | Relay | Oscillator | Memory | Topology |
|-----------|-----------|-------------|------|-------|------------|--------|----------|
| Cortical | 50% | 20% | 5% | 5% | 10% | 10% | 5 z-layers, columnar |
| Thalamic | 10% | 10% | 30% | 30% | 15% | 5% | 2 z-layers, gate-heavy |
| Hippocampal | 30% | 15% | 5% | 5% | 10% | 35% | 4 z-layers, CA3 recurrence |
| BasalGanglia | 20% | 30% | 25% | 10% | 10% | 5% | 2 z-layers, pathway competition |
| Cerebellar | 25% | 15% | 5% | 10% | 35% | 10% | 1 z-layer, oscillator-heavy |
| Brainstem | 15% | 10% | 5% | 10% | 50% | 10% | 1 z-layer, oscillator-dominant |

## Usage

```rust
use neuropool::unified::{
    ImaginalDisc, RegionArchetype, IncubateConfig, incubate,
    CascadeEngine, CascadeConfig, DendriticZone,
    UnifiedPool,
};

// Grow a cortical region from an imaginal disc
let disc = ImaginalDisc::new(RegionArchetype::Cortical, 4, 4);
let config = IncubateConfig::default();
let pool = incubate(&disc, 42, 256, &config);

// Build a cascade engine from the incubated pool
let mut engine = CascadeEngine::with_network(
    pool.neurons,
    pool.synapses,
    CascadeConfig::default(),
);

// Inject feedforward input
engine.inject_ff(0, 3000, 0);

// Run cascade
let spikes = engine.run_until(10_000);

// Check oscillators and spontaneous activity
engine.check_oscillators();
engine.check_spontaneous();

// Persistence
let save_pool = UnifiedPool {
    neurons: engine.neurons.clone(),
    synapses: engine.synapses.clone(),
    grid: pool.grid,
    disc: pool.disc,
};
save_pool.save(std::path::Path::new("region.upool")).unwrap();
let restored = UnifiedPool::load(std::path::Path::new("region.upool")).unwrap();
```

## Module Structure

```
src/
  lib.rs                re-exports (unified = current, voxel/spatial = deprecated)
  unified/
    neuron.rs           UnifiedNeuron, VoxelPosition, zone integration
    synapse.rs          UnifiedSynapse, UnifiedSynapseStore (CSR)
    zone.rs             DendriticZone, ZoneWeights (per-nuclei-type)
    cascade.rs          CascadeEngine — event-driven cascade with zones
    grid.rs             VoxelGrid, VoxelNeighborhood, GridDims
    wiring.rs           wire_by_proximity, ZoneStrategy
    disc.rs             ImaginalDisc, RegionArchetype, NucleiDistribution
    incubate.rs         incubate() — grow pool from disc blueprint
    io.rs               UnifiedNeuronIO trait, NullUnifiedIO
    persist.rs          UnifiedPool save/load (.upool binary format)
  pool.rs               NeuronPool, PoolConfig (v1 — still used by sentinel)
  spatial/              Anatomy types (Nuclei, Dendrite, Axon, Interface)
  (deprecated modules)  binding, cascade, codec, density, io, neuron,
                        plasticity, stats, synapse, template
```

## Legacy System (v1.0 — Deprecated)

The original voxel-based `NeuronPool` with tick-based execution, NeuronType enum, and i8 synaptic weights is still available under `neuropool::pool` for backward compatibility. It will be removed in a future major version. New code should use the unified system.

```rust
// Legacy usage (deprecated)
use neuropool::{NeuronPool, PoolConfig};
let mut pool = NeuronPool::with_random_connectivity("v1_pool", 4096, 0.02, PoolConfig::default());
```

## License

MIT OR Apache-2.0
