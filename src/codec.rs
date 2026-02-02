//! Binary .pool persistence format.
//!
//! Header (16 bytes):
//!   [0..4]   Magic: "POOL"
//!   [4..6]   Version: u16 (LE)
//!   [6..8]   Flags: u16 (LE)
//!   [8..12]  Neuron count: u32 (LE)
//!   [12..16] CRC32 of body
//!
//! Body:
//!   PoolConfig (fixed-size)
//!   NeuronArrays (contiguous arrays)
//!   SynapseStore (row_ptr + synapses)

use std::io::{self, Read};
use std::path::Path;

use crate::neuron::NeuronArrays;
use crate::pool::{NeuronPool, PoolConfig};
use crate::synapse::{Synapse, SynapseStore};

const MAGIC: &[u8; 4] = b"POOL";
const VERSION: u16 = 1;

/// Simple CRC32 (same polynomial as thermogram-rs for consistency)
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// ---- Write primitives ----

fn write_u8(w: &mut Vec<u8>, val: u8) {
    w.push(val);
}

fn write_i8(w: &mut Vec<u8>, val: i8) {
    w.push(val as u8);
}

fn write_u16(w: &mut Vec<u8>, val: u16) {
    w.extend_from_slice(&val.to_le_bytes());
}

fn write_i16(w: &mut Vec<u8>, val: i16) {
    w.extend_from_slice(&val.to_le_bytes());
}

fn write_u32(w: &mut Vec<u8>, val: u32) {
    w.extend_from_slice(&val.to_le_bytes());
}

fn write_u64(w: &mut Vec<u8>, val: u64) {
    w.extend_from_slice(&val.to_le_bytes());
}

fn write_string(w: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    write_u16(w, bytes.len() as u16);
    w.extend_from_slice(bytes);
}

// ---- Read primitives ----

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

fn read_string(r: &mut &[u8]) -> io::Result<String> {
    let len = read_u16(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ---- Serialization ----

fn serialize_config(w: &mut Vec<u8>, config: &PoolConfig) {
    write_i16(w, config.resting_potential);
    write_i16(w, config.spike_threshold);
    write_i16(w, config.reset_potential);
    write_u8(w, config.refractory_ticks);
    write_u8(w, config.trace_decay);
    write_u8(w, config.homeostatic_rate);
    write_u16(w, config.max_synapses_per_neuron);
    write_i8(w, config.stdp_positive);
    write_i8(w, config.stdp_negative);
    write_u8(w, config.max_delay);
}

fn deserialize_config(r: &mut &[u8]) -> io::Result<PoolConfig> {
    Ok(PoolConfig {
        resting_potential: read_i16(r)?,
        spike_threshold: read_i16(r)?,
        reset_potential: read_i16(r)?,
        refractory_ticks: read_u8(r)?,
        trace_decay: read_u8(r)?,
        homeostatic_rate: read_u8(r)?,
        max_synapses_per_neuron: read_u16(r)?,
        stdp_positive: read_i8(r)?,
        stdp_negative: read_i8(r)?,
        max_delay: read_u8(r)?,
    })
}

fn serialize_neurons(w: &mut Vec<u8>, neurons: &NeuronArrays, n: usize) {
    for i in 0..n { write_i16(w, neurons.membrane[i]); }
    for i in 0..n { write_i16(w, neurons.threshold[i]); }
    for i in 0..n { write_u8(w, neurons.leak[i]); }
    for i in 0..n { write_u8(w, neurons.refract_remaining[i]); }
    for i in 0..n { write_u8(w, neurons.flags[i]); }
    for i in 0..n { write_i8(w, neurons.trace[i]); }
    for i in 0..n { write_u8(w, if neurons.spike_out[i] { 1 } else { 0 }); }
}

fn deserialize_neurons(r: &mut &[u8], n: usize) -> io::Result<NeuronArrays> {
    let mut membrane = vec![0i16; n];
    let mut threshold = vec![0i16; n];
    let mut leak = vec![0u8; n];
    let mut refract_remaining = vec![0u8; n];
    let mut flags = vec![0u8; n];
    let mut trace = vec![0i8; n];
    let mut spike_out = vec![false; n];

    for i in 0..n { membrane[i] = read_i16(r)?; }
    for i in 0..n { threshold[i] = read_i16(r)?; }
    for i in 0..n { leak[i] = read_u8(r)?; }
    for i in 0..n { refract_remaining[i] = read_u8(r)?; }
    for i in 0..n { flags[i] = read_u8(r)?; }
    for i in 0..n { trace[i] = read_i8(r)?; }
    for i in 0..n { spike_out[i] = read_u8(r)? != 0; }

    Ok(NeuronArrays {
        membrane,
        threshold,
        leak,
        refract_remaining,
        flags,
        trace,
        spike_out,
    })
}

fn serialize_synapses(w: &mut Vec<u8>, store: &SynapseStore) {
    let n_neurons = store.n_neurons();
    write_u32(w, n_neurons);
    for &ptr in &store.row_ptr {
        write_u32(w, ptr);
    }

    let total = store.total_synapses() as u32;
    write_u32(w, total);
    for syn in &store.synapses {
        write_u16(w, syn.target);
        write_i8(w, syn.weight);
        write_u8(w, syn.delay);
        write_i8(w, syn.eligibility);
        write_u8(w, syn.maturity);
        write_u8(w, syn._reserved[0]);
        write_u8(w, syn._reserved[1]);
    }
}

fn deserialize_synapses(r: &mut &[u8]) -> io::Result<SynapseStore> {
    let n_neurons = read_u32(r)?;
    let mut row_ptr = vec![0u32; (n_neurons + 1) as usize];
    for i in 0..(n_neurons + 1) as usize {
        row_ptr[i] = read_u32(r)?;
    }

    let total = read_u32(r)? as usize;
    let mut synapses = Vec::with_capacity(total);
    for _ in 0..total {
        synapses.push(Synapse {
            target: read_u16(r)?,
            weight: read_i8(r)?,
            delay: read_u8(r)?,
            eligibility: read_i8(r)?,
            maturity: read_u8(r)?,
            _reserved: [read_u8(r)?, read_u8(r)?],
        });
    }

    Ok(SynapseStore { row_ptr, synapses })
}

impl NeuronPool {
    /// Save pool state to a binary .pool file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut body = Vec::with_capacity(self.n_neurons as usize * 16);

        // Body: name, counts, tick_count, config, neurons, synapses
        write_string(&mut body, &self.name);
        write_u32(&mut body, self.n_neurons);
        write_u32(&mut body, self.n_excitatory);
        write_u32(&mut body, self.n_inhibitory);
        write_u64(&mut body, self.tick_count);

        serialize_config(&mut body, &self.config);
        serialize_neurons(&mut body, &self.neurons, self.n_neurons as usize);
        serialize_synapses(&mut body, &self.synapses);

        let checksum = crc32(&body);

        // Header
        let mut file_data = Vec::with_capacity(16 + body.len());
        file_data.extend_from_slice(MAGIC);
        file_data.extend_from_slice(&VERSION.to_le_bytes());
        file_data.extend_from_slice(&0u16.to_le_bytes()); // flags
        file_data.extend_from_slice(&self.n_neurons.to_le_bytes());
        file_data.extend_from_slice(&checksum.to_le_bytes());
        file_data.extend_from_slice(&body);

        std::fs::write(path, &file_data)
    }

    /// Load pool state from a binary .pool file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file_data = std::fs::read(path)?;
        if file_data.len() < 16 {
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
        let _flags = u16::from_le_bytes([file_data[6], file_data[7]]);
        let n_neurons_header = u32::from_le_bytes([file_data[8], file_data[9], file_data[10], file_data[11]]);
        let expected_crc = u32::from_le_bytes([file_data[12], file_data[13], file_data[14], file_data[15]]);

        let body = &file_data[16..];
        let actual_crc = crc32(body);
        if actual_crc != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CRC mismatch: expected {expected_crc:08X}, got {actual_crc:08X}"),
            ));
        }

        let mut r: &[u8] = body;

        let name = read_string(&mut r)?;
        let n_neurons = read_u32(&mut r)?;
        let n_excitatory = read_u32(&mut r)?;
        let n_inhibitory = read_u32(&mut r)?;
        let tick_count = read_u64(&mut r)?;

        if n_neurons != n_neurons_header {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "neuron count mismatch between header and body",
            ));
        }

        let config = deserialize_config(&mut r)?;
        let neurons = deserialize_neurons(&mut r, n_neurons as usize)?;
        let synapses = deserialize_synapses(&mut r)?;

        let delay_buf_n = n_neurons as usize;

        Ok(Self {
            name,
            neurons,
            synapses,
            n_neurons,
            n_excitatory,
            n_inhibitory,
            tick_count,
            delay_buf: super::pool::DelayBuffer::new(delay_buf_n, config.max_delay),
            synaptic_current: vec![0i16; delay_buf_n],
            last_spike_count: 0,
            spike_rate: vec![0u16; delay_buf_n],
            spike_window: vec![false; delay_buf_n],
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_pool() {
        let pool = NeuronPool::new("test_pool", 100, PoolConfig::default());
        let path = std::env::temp_dir().join("neuropool_test_empty.pool");

        pool.save(&path).expect("save failed");
        let loaded = NeuronPool::load(&path).expect("load failed");

        assert_eq!(loaded.name, "test_pool");
        assert_eq!(loaded.n_neurons, 100);
        assert_eq!(loaded.n_excitatory, pool.n_excitatory);
        assert_eq!(loaded.n_inhibitory, pool.n_inhibitory);
        assert_eq!(loaded.tick_count, 0);
        assert_eq!(loaded.synapse_count(), 0);
        assert_eq!(loaded.config.resting_potential, pool.config.resting_potential);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn round_trip_with_connectivity() {
        let mut pool = NeuronPool::with_random_connectivity("connected", 50, 0.05, PoolConfig::default());

        // Run some ticks to create state
        for _ in 0..10 {
            let input: Vec<i16> = (0..50).map(|i| if i < 10 { 5000 } else { 0 }).collect();
            pool.tick(&input);
        }

        let path = std::env::temp_dir().join("neuropool_test_connected.pool");
        pool.save(&path).expect("save failed");
        let loaded = NeuronPool::load(&path).expect("load failed");

        assert_eq!(loaded.n_neurons, 50);
        assert_eq!(loaded.tick_count, pool.tick_count);
        assert_eq!(loaded.synapse_count(), pool.synapse_count());

        // Verify neuron state preserved
        for i in 0..50 {
            assert_eq!(loaded.neurons.membrane[i], pool.neurons.membrane[i]);
            assert_eq!(loaded.neurons.threshold[i], pool.neurons.threshold[i]);
            assert_eq!(loaded.neurons.flags[i], pool.neurons.flags[i]);
        }

        // Verify synapse state preserved
        for i in 0..pool.synapse_count() {
            assert_eq!(loaded.synapses.synapses[i].target, pool.synapses.synapses[i].target);
            assert_eq!(loaded.synapses.synapses[i].weight, pool.synapses.synapses[i].weight);
            assert_eq!(loaded.synapses.synapses[i].maturity, pool.synapses.synapses[i].maturity);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn corrupted_file_detected() {
        let pool = NeuronPool::new("test", 10, PoolConfig::default());
        let path = std::env::temp_dir().join("neuropool_test_corrupt.pool");

        pool.save(&path).expect("save failed");

        // Corrupt the file
        let mut data = std::fs::read(&path).unwrap();
        if data.len() > 20 {
            data[20] ^= 0xFF;
        }
        std::fs::write(&path, &data).unwrap();

        let result = NeuronPool::load(&path);
        assert!(result.is_err(), "corrupted file should fail to load");

        std::fs::remove_file(&path).ok();
    }
}
