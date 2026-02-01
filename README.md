# neuropool-rs

Biological neuron pool substrate for neuromorphic computing. Integer-only LIF neurons with CSR synapses, eligibility traces, and three-factor plasticity.

## What This Is

A compact, cache-friendly neuron pool that unifies spiking dynamics and synaptic persistence into a single substrate. Each pool contains thousands of Leaky Integrate-and-Fire neurons connected by 8-byte synapses stored in Compressed Sparse Row format.

The core innovation is **eligibility-trace credit assignment**: only synapses that recently participated in spike propagation get modified when neuromodulators arrive. This solves the credit assignment problem that flat weight stores cannot address.

## Architecture

```
NeuronPool
├── NeuronArrays (SoA layout, 8 bytes/neuron)
│   ├── membrane:    i16  Q8.8 fixed-point potential
│   ├── threshold:   i16  adaptive (homeostatic)
│   ├── leak:        u8   profile-dependent decay rate
│   ├── refract:     u8   refractory countdown
│   ├── flags:       u8   excitatory/inhibitory + profile
│   └── trace:       i8   post-synaptic eligibility trace
│
├── SynapseStore (CSR format)
│   ├── row_ptr:     [u32; N+1]    index into synapses
│   └── synapses:    [Synapse; S]  8 bytes each
│       ├── target:      u16   post-synaptic neuron
│       ├── weight:      i8    Dale's Law constrained
│       ├── delay:       u8    axonal delay (1-8 ticks)
│       ├── eligibility: i8    STDP trace (decaying)
│       └── maturity:    u8    thermal state + counter
│
└── DelayBuffer (ring buffer per delay slot)
```

## Key Properties

- **Integer-only**: Q8.8 fixed-point membrane dynamics, no floats in the tick hot path
- **Dale's Law**: 80/20 excitatory/inhibitory split, enforced at construction
- **STDP**: Spike-timing dependent eligibility traces mark causal synapses
- **Three-factor plasticity**: `eligibility * neuromodulator = weight_change`
  - Dopamine above baseline reinforces eligible synapses
  - Cortisol above baseline weakens eligible synapses
  - Acetylcholine gates synaptogenesis (new connection formation)
- **Thermal maturity lifecycle**: Synapses promote HOT -> WARM -> COOL -> COLD as they accumulate reinforcement. Cold synapses are frozen. Dead synapses (HOT with zero counter) get pruned.
- **Homeostatic plasticity**: Per-neuron threshold adjustment targets ~5% spike rate, preventing seizure and silence
- **Binary persistence**: `.pool` format with CRC32 integrity, contiguous array serialization

## Learning Demo Results

The `learning_demo` example proves the plasticity mechanism works by repeatedly presenting stimulus to input neurons and rewarding output activity with dopamine:

```
Pool: 32 neurons, 517 synapses (50% density)

Structure:
  Synapses: 517 -> 62 (88% pruned)
  Thermal:  H:517 -> F:48 (full maturity lifecycle)
  Weights:  |w|=39.6 -> 125.3 (3x strengthening)

Activity:
  Baseline: 104 spikes / 40 ticks
  Final:    103 spikes / 40 ticks (homeostatic stability)

Performance: 0.32ms/epoch
```

**What this proves:**
- STDP traces correctly mark causal pathways
- DA modulation strengthens traced synapses
- Thermal lifecycle works: HOT -> WARM -> COOL -> FROZEN
- Structural pruning removes dead synapses
- Homeostatic plasticity maintains stable output

**What requires multiple pools:**
Spatial discrimination (routing different inputs to different outputs) needs separate pools or local modulation. A single pool with global neuromodulation learns pathway strength but not spatial routing. This is by design — in the brain, different cortical columns handle different stimuli. The TVMR integration layer orchestrates multiple pools for discrimination tasks.

Run the demo:
```bash
cargo run --example learning_demo
```

## Memory Budget

| Pool Size | Neurons | Synapses (~20 avg) | Total |
|-----------|---------|---------------------|-------|
| 4,096     | 32 KB   | 640 KB              | ~688 KB |
| 16,384    | 128 KB  | 2.5 MB              | ~2.7 MB |
| 65,536    | 512 KB  | 10 MB               | ~10.7 MB |

## Usage

```rust
use neuropool::{NeuronPool, PoolConfig};
use ternary_signal::Signal;

// Create a pool with random connectivity
let mut pool = NeuronPool::with_random_connectivity(
    "cortex_v1", 4096, 0.02, PoolConfig::default()
);

// Tick loop
let input = vec![0i16; 4096];
pool.tick(&input);

// Inject external signal
pool.inject(0..100, Signal::positive(200));

// Read output spikes
let output = pool.read_output(3800..4096);

// Three-factor plasticity (DA=reward, Cortisol=punishment, ACh=attention)
let (reinforced, weakened) = pool.apply_modulation(200, 30, 100);

// Synaptogenesis (ACh-gated)
let new_synapses = pool.synaptogenesis(180);

// Prune dead synapses
let pruned = pool.prune_dead();

// Persistence
pool.save(std::path::Path::new("pool.pool")).unwrap();
let restored = NeuronPool::load(std::path::Path::new("pool.pool")).unwrap();

// Inspection
let stats = pool.stats();
println!("{}", stats);
```

## Module Structure

```
src/
  lib.rs          re-exports
  neuron.rs       NeuronArrays, NeuronProfile, flags encoding
  synapse.rs      Synapse, SynapseStore (CSR), maturity lifecycle
  pool.rs         NeuronPool, PoolConfig, tick loop, inject/read
  plasticity.rs   three-factor modulation, synaptogenesis, pruning
  codec.rs        .pool binary save/load
  stats.rs        PoolStats, ThermalDistribution
examples/
  learning_demo.rs  pathway strengthening proof
```

## License

MIT OR Apache-2.0
