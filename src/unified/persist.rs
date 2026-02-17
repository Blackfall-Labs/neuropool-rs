#![allow(deprecated)]
//! Save/Load persistence for unified pools.
//!
//! Binary format with CRC32 checksum. Persists:
//! - Neurons (positions, nuclei, anatomy, electrical state)
//! - Synapses (CSR with zone tags)
//! - Disc metadata (archetype, distribution, wiring rules)

use std::io::{self, Read};
use std::path::Path;

use ternary_signal::Polarity;

use super::disc::{ImaginalDisc, NucleiDistribution, RegionArchetype, WiringRules, ZoneBias};
use super::grid::{GridDims, VoxelGrid};
use super::neuron::{UnifiedNeuron, VoxelPosition};
use super::synapse::{UnifiedSynapse, UnifiedSynapseStore};
use super::zone::DendriticZone;
use crate::spatial::{Axon, Dendrite, EnergyGates, Interface, Nuclei};

/// Magic bytes for unified pool files.
const MAGIC: &[u8; 4] = b"UNPL";
/// Format version.
const VERSION: u16 = 1;

/// A complete unified pool ready for save/load.
pub struct UnifiedPool {
    /// All neurons (sorted by voxel position).
    pub neurons: Vec<UnifiedNeuron>,
    /// All synapses (zone-aware, CSR indexed).
    pub synapses: UnifiedSynapseStore,
    /// Spatial grid index.
    pub grid: VoxelGrid,
    /// The disc that produced this pool (or a reconstructed one on load).
    pub disc: ImaginalDisc,
}

impl UnifiedPool {
    /// Save pool state to a binary file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut body = Vec::with_capacity(self.neurons.len() * 64);

        // Disc metadata
        write_disc(&mut body, &self.disc);

        // Neuron count
        write_u32(&mut body, self.neurons.len() as u32);

        // Neurons
        for n in &self.neurons {
            write_neuron(&mut body, n);
        }

        // Synapse count
        let syn_count = self.synapses.len() as u32;
        write_u32(&mut body, syn_count);

        // Synapses
        for s in self.synapses.iter() {
            write_synapse(&mut body, s);
        }

        // CRC32
        let checksum = crc32(&body);

        // Write file: header + body
        let mut file_data = Vec::with_capacity(12 + body.len());
        file_data.extend_from_slice(MAGIC);
        file_data.extend_from_slice(&VERSION.to_le_bytes());
        file_data.extend_from_slice(&(self.neurons.len() as u32).to_le_bytes());
        file_data.extend_from_slice(&checksum.to_le_bytes());
        file_data.extend_from_slice(&body);

        std::fs::write(path, &file_data)
    }

    /// Load pool state from a binary file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file_data = std::fs::read(path)?;
        if file_data.len() < 14 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        // Validate header
        if &file_data[0..4] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = u16::from_le_bytes([file_data[4], file_data[5]]);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {}", version),
            ));
        }
        let _n_neurons_header = u32::from_le_bytes([file_data[6], file_data[7], file_data[8], file_data[9]]);
        let expected_crc = u32::from_le_bytes([file_data[10], file_data[11], file_data[12], file_data[13]]);

        let body = &file_data[14..];
        let actual_crc = crc32(body);
        if actual_crc != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CRC mismatch: expected {:08X}, got {:08X}", expected_crc, actual_crc),
            ));
        }

        let mut r: &[u8] = body;

        // Disc metadata
        let disc = read_disc(&mut r)?;

        // Neurons
        let n_neurons = read_u32(&mut r)? as usize;
        let mut neurons = Vec::with_capacity(n_neurons);
        for _ in 0..n_neurons {
            neurons.push(read_neuron(&mut r)?);
        }

        // Synapses
        let n_synapses = read_u32(&mut r)? as usize;
        let mut synapses = UnifiedSynapseStore::with_capacity(n_synapses, n_neurons);
        for _ in 0..n_synapses {
            synapses.add(read_synapse(&mut r)?);
        }
        synapses.rebuild_index(n_neurons as u32);

        // Rebuild grid from disc dims
        let grid_dims = GridDims {
            x: disc.grid_dims.0,
            y: disc.grid_dims.1,
            z: disc.grid_dims.2,
        };
        let grid = VoxelGrid::build_with_dims(&mut neurons, grid_dims);

        Ok(Self {
            neurons,
            synapses,
            grid,
            disc,
        })
    }
}

// === Serialization Helpers ===

fn write_u8(w: &mut Vec<u8>, v: u8) { w.push(v); }
fn write_i8(w: &mut Vec<u8>, v: i8) { w.push(v as u8); }
fn write_u16(w: &mut Vec<u8>, v: u16) { w.extend_from_slice(&v.to_le_bytes()); }
fn write_i16(w: &mut Vec<u8>, v: i16) { w.extend_from_slice(&v.to_le_bytes()); }
fn write_u32(w: &mut Vec<u8>, v: u32) { w.extend_from_slice(&v.to_le_bytes()); }
fn write_u64(w: &mut Vec<u8>, v: u64) { w.extend_from_slice(&v.to_le_bytes()); }
fn write_f32(w: &mut Vec<u8>, v: f32) { w.extend_from_slice(&v.to_le_bytes()); }

fn read_u8(r: &mut &[u8]) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut &[u8]) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut &[u8]) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut &[u8]) -> io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(r: &mut &[u8]) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut &[u8]) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(r: &mut &[u8]) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

// === Neuron Serialization ===

fn write_neuron(w: &mut Vec<u8>, n: &UnifiedNeuron) {
    // Position: voxel (3 x u16) + local (3 x u8)
    write_u16(w, n.position.voxel.0);
    write_u16(w, n.position.voxel.1);
    write_u16(w, n.position.voxel.2);
    write_u8(w, n.position.local.0);
    write_u8(w, n.position.local.1);
    write_u8(w, n.position.local.2);

    // Dendrite
    write_f32(w, n.dendrite.radius);
    write_u16(w, n.dendrite.spine_count);

    // Axon
    write_f32(w, n.axon.terminal[0]);
    write_f32(w, n.axon.terminal[1]);
    write_f32(w, n.axon.terminal[2]);
    write_u8(w, n.axon.myelin);
    write_u8(w, n.axon.health);

    // Nuclei
    write_u8(w, n.nuclei.soma_size);
    write_u8(w, n.nuclei.axon_affinity);
    write_u8(w, n.nuclei.myelin_affinity);
    write_u8(w, n.nuclei.metabolic_rate);
    write_u8(w, n.nuclei.leak);
    write_u32(w, n.nuclei.refractory);
    write_u32(w, n.nuclei.oscillation_period);
    // Interface
    write_u8(w, n.nuclei.interface.kind);
    write_u16(w, n.nuclei.interface.target);
    write_u8(w, n.nuclei.interface.modality);
    write_i16(w, n.nuclei.interface.gates.low_ceiling);
    write_i16(w, n.nuclei.interface.gates.high_floor);
    // Polarity
    write_i8(w, n.nuclei.polarity.as_i8());

    // Electrical state
    write_i16(w, n.feedforward_potential);
    write_i16(w, n.context_potential);
    write_i16(w, n.feedback_potential);
    write_i16(w, n.membrane);
    write_i16(w, n.threshold);
    write_i8(w, n.trace);
    write_u8(w, if n.predicted { 1 } else { 0 });
    write_u8(w, n.stamina);
    write_u64(w, n.last_spike_us);
    write_u64(w, n.last_arrival_us);
}

fn read_neuron(r: &mut &[u8]) -> io::Result<UnifiedNeuron> {
    // Position
    let vx = read_u16(r)?;
    let vy = read_u16(r)?;
    let vz = read_u16(r)?;
    let lx = read_u8(r)?;
    let ly = read_u8(r)?;
    let lz = read_u8(r)?;
    let position = VoxelPosition::new((vx, vy, vz), (lx, ly, lz));

    // Dendrite
    let radius = read_f32(r)?;
    let spine_count = read_u16(r)?;
    let dendrite = Dendrite::new(radius, spine_count);

    // Axon
    let tx = read_f32(r)?;
    let ty = read_f32(r)?;
    let tz = read_f32(r)?;
    let myelin = read_u8(r)?;
    let health = read_u8(r)?;
    let axon = Axon { terminal: [tx, ty, tz], myelin, health };

    // Nuclei
    let soma_size = read_u8(r)?;
    let axon_affinity = read_u8(r)?;
    let myelin_affinity = read_u8(r)?;
    let metabolic_rate = read_u8(r)?;
    let leak = read_u8(r)?;
    let refractory = read_u32(r)?;
    let oscillation_period = read_u32(r)?;
    let kind = read_u8(r)?;
    let target = read_u16(r)?;
    let modality = read_u8(r)?;
    let low_ceiling = read_i16(r)?;
    let high_floor = read_i16(r)?;
    let polarity_i8 = read_i8(r)?;

    let interface = Interface {
        kind,
        target,
        modality,
        gates: EnergyGates { low_ceiling, high_floor },
    };
    let polarity = match polarity_i8 {
        1 => Polarity::Positive,
        -1 => Polarity::Negative,
        _ => Polarity::Zero,
    };
    let nuclei = Nuclei::new(
        soma_size, axon_affinity, myelin_affinity, metabolic_rate,
        leak, refractory, oscillation_period, interface, polarity,
    );

    // Create neuron (sets zone_weights from nuclei)
    let mut n = UnifiedNeuron::new(position, dendrite, axon, nuclei);

    // Restore electrical state
    n.feedforward_potential = read_i16(r)?;
    n.context_potential = read_i16(r)?;
    n.feedback_potential = read_i16(r)?;
    n.membrane = read_i16(r)?;
    n.threshold = read_i16(r)?;
    n.trace = read_i8(r)?;
    n.predicted = read_u8(r)? != 0;
    n.stamina = read_u8(r)?;
    n.last_spike_us = read_u64(r)?;
    n.last_arrival_us = read_u64(r)?;

    Ok(n)
}

// === Synapse Serialization ===

fn write_synapse(w: &mut Vec<u8>, s: &UnifiedSynapse) {
    write_u32(w, s.source);
    write_u32(w, s.target);
    write_u8(w, s.zone.index() as u8);
    write_i8(w, s.signal.polarity);
    write_u8(w, s.signal.magnitude);
    write_u32(w, s.delay_us);
    write_u8(w, s.maturity);
    write_i16(w, s.pressure);
}

fn read_synapse(r: &mut &[u8]) -> io::Result<UnifiedSynapse> {
    let source = read_u32(r)?;
    let target = read_u32(r)?;
    let zone_idx = read_u8(r)?;
    let polarity = read_i8(r)?;
    let magnitude = read_u8(r)?;
    let delay_us = read_u32(r)?;
    let maturity = read_u8(r)?;
    let pressure = read_i16(r)?;

    let zone = match zone_idx {
        0 => DendriticZone::Feedforward,
        1 => DendriticZone::Context,
        _ => DendriticZone::Feedback,
    };

    let signal = ternary_signal::Signal { polarity, magnitude };

    Ok(UnifiedSynapse {
        source,
        target,
        zone,
        signal,
        delay_us,
        maturity,
        pressure,
    })
}

// === Disc Serialization ===

fn write_disc(w: &mut Vec<u8>, disc: &ImaginalDisc) {
    // Archetype
    let arch_id = match disc.archetype {
        RegionArchetype::Cortical => 0u8,
        RegionArchetype::Thalamic => 1,
        RegionArchetype::Hippocampal => 2,
        RegionArchetype::BasalGanglia => 3,
        RegionArchetype::Cerebellar => 4,
        RegionArchetype::Brainstem => 5,
    };
    write_u8(w, arch_id);

    // Distribution
    let d = &disc.distribution;
    write_u8(w, d.pyramidal);
    write_u8(w, d.interneuron);
    write_u8(w, d.gate);
    write_u8(w, d.relay);
    write_u8(w, d.oscillator);
    write_u8(w, d.memory);

    // Grid dims
    write_u16(w, disc.grid_dims.0);
    write_u16(w, disc.grid_dims.1);
    write_u16(w, disc.grid_dims.2);
    write_u16(w, disc.z_layers);

    // Wiring rules
    let wr = &disc.wiring;
    write_u64(w, wr.max_distance_sq);
    write_u16(w, wr.max_fanout);
    write_u16(w, wr.max_fanin);
    for zb in &wr.zone_biases {
        write_u8(w, zb.zone.index() as u8);
        write_u8(w, zb.probability);
        write_u8(w, zb.magnitude);
    }
    write_u8(w, if wr.dense_lateral { 1 } else { 0 });
    write_u32(w, wr.oscillator_period_range.0);
    write_u32(w, wr.oscillator_period_range.1);
}

fn read_disc(r: &mut &[u8]) -> io::Result<ImaginalDisc> {
    let arch_id = read_u8(r)?;
    let archetype = match arch_id {
        0 => RegionArchetype::Cortical,
        1 => RegionArchetype::Thalamic,
        2 => RegionArchetype::Hippocampal,
        3 => RegionArchetype::BasalGanglia,
        4 => RegionArchetype::Cerebellar,
        5 => RegionArchetype::Brainstem,
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown archetype")),
    };

    let distribution = NucleiDistribution {
        pyramidal: read_u8(r)?,
        interneuron: read_u8(r)?,
        gate: read_u8(r)?,
        relay: read_u8(r)?,
        oscillator: read_u8(r)?,
        memory: read_u8(r)?,
    };

    let gx = read_u16(r)?;
    let gy = read_u16(r)?;
    let gz = read_u16(r)?;
    let z_layers = read_u16(r)?;

    let max_distance_sq = read_u64(r)?;
    let max_fanout = read_u16(r)?;
    let max_fanin = read_u16(r)?;

    let mut zone_biases = [ZoneBias { zone: DendriticZone::Feedforward, probability: 0, magnitude: 0 }; 3];
    for i in 0..3 {
        let zone_idx = read_u8(r)?;
        let probability = read_u8(r)?;
        let magnitude = read_u8(r)?;
        zone_biases[i] = ZoneBias {
            zone: match zone_idx {
                0 => DendriticZone::Feedforward,
                1 => DendriticZone::Context,
                _ => DendriticZone::Feedback,
            },
            probability,
            magnitude,
        };
    }

    let dense_lateral = read_u8(r)? != 0;
    let osc_min = read_u32(r)?;
    let osc_max = read_u32(r)?;

    let wiring = WiringRules {
        max_distance_sq,
        max_fanout,
        max_fanin,
        zone_biases,
        dense_lateral,
        oscillator_period_range: (osc_min, osc_max),
    };

    Ok(ImaginalDisc {
        archetype,
        distribution,
        wiring,
        grid_dims: (gx, gy, gz),
        z_layers,
    })
}

// === CRC32 ===

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::disc::RegionArchetype;
    use super::super::incubate::{incubate, IncubateConfig};

    #[test]
    fn save_load_round_trip() {
        let disc = ImaginalDisc::new(RegionArchetype::Cortical, 3, 3);
        let config = IncubateConfig {
            settling_steps: 5,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };
        let incubated = incubate(&disc, 42, 64, &config);

        let pool = UnifiedPool {
            neurons: incubated.neurons,
            synapses: incubated.synapses,
            grid: incubated.grid,
            disc: incubated.disc,
        };

        // Save
        let dir = std::env::temp_dir().join("neuropool_test_unified");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_cortical.upool");
        pool.save(&path).expect("save failed");

        // Load
        let loaded = UnifiedPool::load(&path).expect("load failed");

        // Verify
        assert_eq!(loaded.neurons.len(), pool.neurons.len());
        assert_eq!(loaded.synapses.len(), pool.synapses.len());
        assert_eq!(loaded.disc.archetype, RegionArchetype::Cortical);
        assert_eq!(loaded.disc.z_layers, 5);

        // Verify neuron state preserved
        for (orig, loaded) in pool.neurons.iter().zip(loaded.neurons.iter()) {
            assert_eq!(orig.position, loaded.position);
            assert_eq!(orig.nuclei, loaded.nuclei);
            assert_eq!(orig.threshold, loaded.threshold);
            assert_eq!(orig.stamina, loaded.stamina);
            assert_eq!(orig.trace, loaded.trace);
            assert_eq!(orig.membrane, loaded.membrane);
        }

        // Verify synapse state preserved
        for (orig, loaded) in pool.synapses.iter().zip(loaded.synapses.iter()) {
            assert_eq!(orig.source, loaded.source);
            assert_eq!(orig.target, loaded.target);
            assert_eq!(orig.zone, loaded.zone);
            assert_eq!(orig.signal.polarity, loaded.signal.polarity);
            assert_eq!(orig.signal.magnitude, loaded.signal.magnitude);
            assert_eq!(orig.delay_us, loaded.delay_us);
            assert_eq!(orig.maturity, loaded.maturity);
            assert_eq!(orig.pressure, loaded.pressure);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_load_hippocampal() {
        let disc = ImaginalDisc::new(RegionArchetype::Hippocampal, 2, 2);
        let config = IncubateConfig {
            settling_steps: 3,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };
        let incubated = incubate(&disc, 99, 48, &config);

        let pool = UnifiedPool {
            neurons: incubated.neurons,
            synapses: incubated.synapses,
            grid: incubated.grid,
            disc: incubated.disc,
        };

        let dir = std::env::temp_dir().join("neuropool_test_unified");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_hippocampal.upool");
        pool.save(&path).expect("save failed");

        let loaded = UnifiedPool::load(&path).expect("load failed");
        assert_eq!(loaded.disc.archetype, RegionArchetype::Hippocampal);
        assert_eq!(loaded.disc.z_layers, 4);
        assert_eq!(loaded.neurons.len(), 48);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bad_magic_rejected() {
        let dir = std::env::temp_dir().join("neuropool_test_unified");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_bad_magic.upool");

        std::fs::write(&path, b"BAD_MAGIC_DATA_HERE_LONG_ENOUGH").unwrap();
        let result = UnifiedPool::load(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupted_crc_rejected() {
        let disc = ImaginalDisc::new(RegionArchetype::Brainstem, 2, 2);
        let config = IncubateConfig {
            settling_steps: 2,
            step_duration_us: 5_000,
            prune_after_settling: false,
        };
        let incubated = incubate(&disc, 7, 16, &config);

        let pool = UnifiedPool {
            neurons: incubated.neurons,
            synapses: incubated.synapses,
            grid: incubated.grid,
            disc: incubated.disc,
        };

        let dir = std::env::temp_dir().join("neuropool_test_unified");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_corrupt.upool");
        pool.save(&path).expect("save failed");

        // Corrupt a byte in the body
        let mut data = std::fs::read(&path).unwrap();
        if data.len() > 20 {
            data[20] ^= 0xFF;
        }
        std::fs::write(&path, &data).unwrap();

        let result = UnifiedPool::load(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }
}
