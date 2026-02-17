#![allow(deprecated)]
//! Neuron data structures — SoA (Structure of Arrays) layout for cache performance.
//!
//! Each neuron field is stored as a separate contiguous array so that the tick
//! hot path iterates over dense memory. A single neuron consumes 8 bytes across
//! all arrays combined.

/// Neuron functional type, encoded in bits 3-5 of the flags byte.
///
/// Determines the neuron's behavior during tick():
/// - Pre-spike: What happens before membrane comparison (Sensory reads input, Gate modulates threshold, Oscillator ramps)
/// - Post-spike: What happens after firing (Motor writes output, MemoryReader queries bank)
/// - Non-spike: Special checks on synaptic input (MemoryMatcher compares patterns)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NeuronType {
    /// Standard LIF neuron. The default. No special behavior.
    Computational = 0b000,
    /// Pre-spike: reads from external field via NeuronIO, adds to membrane.
    Sensory = 0b001,
    /// Post-spike: writes magnitude to output channel via NeuronIO.
    Motor = 0b010,
    /// Post-spike: queries databank, injects result as current next tick.
    MemoryReader = 0b011,
    /// On synaptic input: pattern comparison, boosts membrane on match.
    MemoryMatcher = 0b100,
    /// Pre-spike: chemical level modulates firing threshold.
    Gate = 0b101,
    /// Standard LIF with low threshold + high fanout (set at creation).
    Relay = 0b110,
    /// Pre-spike: autonomous depolarization ramp on phase.
    Oscillator = 0b111,
}

impl NeuronType {
    /// Decode from flags byte (bits 3-5).
    #[inline]
    pub fn from_flags(flags: u8) -> Self {
        match (flags >> 3) & 0b111 {
            0b000 => Self::Computational,
            0b001 => Self::Sensory,
            0b010 => Self::Motor,
            0b011 => Self::MemoryReader,
            0b100 => Self::MemoryMatcher,
            0b101 => Self::Gate,
            0b110 => Self::Relay,
            0b111 => Self::Oscillator,
            _ => unreachable!(),
        }
    }
}

/// Neuron firing profile, encoded in bits 1-2 of the flags byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NeuronProfile {
    /// Standard excitatory cortical neuron — moderate leak, regular firing
    RegularSpiking = 0b00,
    /// Fast inhibitory interneuron — high leak, rapid response
    FastSpiking = 0b01,
    /// Burst-capable neuron — low leak, fires in clusters
    IntrinsicBursting = 0b10,
    /// Reserved for Izhikevich upgrade
    Reserved = 0b11,
}

impl NeuronProfile {
    /// Decode from flags byte (bits 1-2)
    #[inline]
    pub fn from_flags(flags: u8) -> Self {
        match (flags >> 1) & 0b11 {
            0b00 => Self::RegularSpiking,
            0b01 => Self::FastSpiking,
            0b10 => Self::IntrinsicBursting,
            _ => Self::Reserved,
        }
    }

    /// Default leak rate for this profile (higher = faster decay toward resting)
    #[inline]
    pub fn default_leak(self) -> u8 {
        match self {
            Self::RegularSpiking => 4,    // moderate leak
            Self::FastSpiking => 8,       // fast leak — quick return to rest
            Self::IntrinsicBursting => 2, // slow leak — sustains depolarization
            Self::Reserved => 4,
        }
    }

    /// Default refractory period for this profile (ticks after spike before firing again).
    ///
    /// FastSpiking interneurons recover in 1 tick (high-frequency bursting role).
    /// IntrinsicBursting neurons have a longer refractory to space out burst clusters.
    #[inline]
    pub fn default_refractory(self) -> u8 {
        match self {
            Self::RegularSpiking => 2,
            Self::FastSpiking => 1,       // rapid recovery — sustains high firing rate
            Self::IntrinsicBursting => 3, // longer gap — burst then pause
            Self::Reserved => 2,
        }
    }
}

/// Flags byte encoding:
/// - Bit 0: 0 = excitatory, 1 = inhibitory
/// - Bits 1-2: NeuronProfile
/// - Bits 3-5: NeuronType
/// - Bits 6-7: reserved
pub mod flags {
    pub const INHIBITORY_BIT: u8 = 0x01;

    #[inline]
    pub fn is_inhibitory(f: u8) -> bool {
        f & INHIBITORY_BIT != 0
    }

    #[inline]
    pub fn is_excitatory(f: u8) -> bool {
        !is_inhibitory(f)
    }

    /// Decode neuron type from flags byte (bits 3-5).
    #[inline]
    pub fn neuron_type(f: u8) -> super::NeuronType {
        super::NeuronType::from_flags(f)
    }

    /// Encode flags byte with inhibitory, profile, and type (default Computational).
    #[inline]
    pub fn encode(inhibitory: bool, profile: super::NeuronProfile) -> u8 {
        encode_full(inhibitory, profile, super::NeuronType::Computational)
    }

    /// Encode flags byte with all fields: inhibitory, profile, and neuron type.
    #[inline]
    pub fn encode_full(inhibitory: bool, profile: super::NeuronProfile, ntype: super::NeuronType) -> u8 {
        let mut f = 0u8;
        if inhibitory {
            f |= INHIBITORY_BIT;
        }
        f |= (profile as u8) << 1;
        f |= (ntype as u8) << 3;
        f
    }
}

/// SoA neuron storage — each field is a separate contiguous array.
///
/// For N neurons, each Vec has exactly N elements.
///
/// Layout per neuron across arrays:
/// - membrane:          2 bytes (i16, Q8.8 fixed-point)
/// - threshold:         2 bytes (i16, Q8.8 adaptive threshold)
/// - leak:              1 byte  (u8, leak rate)
/// - refract_remaining: 1 byte  (u8, refractory countdown)
/// - flags:             1 byte  (u8, excitatory/inhibitory + profile + type)
/// - trace:             1 byte  (i8, post-synaptic eligibility trace)
/// - binding_slot:      1 byte  (u8, index into BindingTable; 0 = no binding)
///
/// === v4: Physical State (Spatial Brain) ===
/// - soma_position:     12 bytes (3 × f32, continuous 3D position)
/// - axon_terminal:     12 bytes (3 × f32, where axon ends)
/// - dendrite_radius:   4 bytes  (f32, local reception range)
/// - axon_health:       1 byte   (u8, 0=dead, 255=fully myelinated)
pub struct NeuronArrays {
    /// Q8.8 fixed-point membrane potential. Resting ~ -17920 (-70 * 256).
    pub membrane: Vec<i16>,
    /// Q8.8 adaptive threshold. Baseline ~ -14080 (-55 * 256).
    /// Adjusted by homeostatic plasticity.
    pub threshold: Vec<i16>,
    /// Leak rate — higher values mean faster return to resting potential.
    /// Applied as: membrane += (resting - membrane) >> (8 - leak.min(7))
    pub leak: Vec<u8>,
    /// Refractory ticks remaining. Neuron cannot fire while > 0.
    pub refract_remaining: Vec<u8>,
    /// Bit flags: bit 0 = inhibitory, bits 1-2 = NeuronProfile.
    pub flags: Vec<u8>,
    /// Post-synaptic eligibility trace. Bumped on spike, decays toward 0.
    /// Used for STDP credit assignment.
    pub trace: Vec<i8>,
    /// Did this neuron spike this tick? Cleared at start of each tick.
    pub spike_out: Vec<bool>,
    /// Index into the pool's BindingTable. 0 = no binding (Computational neuron).
    /// Values 1-255 map to BindingConfig entries for specialized neurons.
    pub binding_slot: Vec<u8>,

    // === v4: Physical State (Spatial Brain) ===

    /// Soma (cell body) position in continuous 3D space.
    /// Role emerges from geometry: if axon stays local → gray matter processing,
    /// if axon leaves dense volume → "output" neuron.
    pub soma_position: Vec<[f32; 3]>,

    /// Axon terminal position — where the axon ends.
    /// Distance from soma determines local vs long-range connectivity.
    pub axon_terminal: Vec<[f32; 3]>,

    /// Dendrite reception radius — neurons within this distance can receive signals.
    pub dendrite_radius: Vec<f32>,

    /// Axon health: 0 = dead/pruned, 255 = fully myelinated.
    /// Decays without correlated activity. Below threshold → retraction → death.
    pub axon_health: Vec<u8>,
}

impl NeuronArrays {
    /// Allocate arrays for `n` neurons, all initialized to resting state.
    ///
    /// Dale's Law: first `n_excitatory` neurons are excitatory (flag bit 0 = 0),
    /// remaining are inhibitory (flag bit 0 = 1).
    ///
    /// Physical state (v4): All neurons start at origin with zero axon length
    /// and default dendrite radius. Use `init_spatial()` to assign positions.
    pub fn new(n: u32, n_excitatory: u32, resting: i16, threshold: i16) -> Self {
        let n = n as usize;
        let n_exc = n_excitatory as usize;

        let mut flags_vec = vec![0u8; n];
        for i in 0..n {
            let inhibitory = i >= n_exc;
            let profile = if inhibitory {
                NeuronProfile::FastSpiking
            } else {
                NeuronProfile::RegularSpiking
            };
            flags_vec[i] = flags::encode(inhibitory, profile);
        }

        let mut leak_vec = vec![0u8; n];
        for i in 0..n {
            leak_vec[i] = NeuronProfile::from_flags(flags_vec[i]).default_leak();
        }

        Self {
            membrane: vec![resting; n],
            threshold: vec![threshold; n],
            leak: leak_vec,
            refract_remaining: vec![0; n],
            flags: flags_vec,
            trace: vec![0i8; n],
            spike_out: vec![false; n],
            binding_slot: vec![0u8; n],
            // v4: Physical state — default to origin, init_spatial() assigns positions
            soma_position: vec![[0.0, 0.0, 0.0]; n],
            axon_terminal: vec![[0.0, 0.0, 0.0]; n],
            dendrite_radius: vec![1.0; n], // Default 1.0 unit reception radius
            axon_health: vec![128; n],     // Start healthy (128 = baseline)
        }
    }

    /// Initialize spatial positions from grid dimensions.
    ///
    /// Distributes neurons evenly in a 3D grid within the given bounds.
    /// Axon terminals start at soma position (no extension yet).
    pub fn init_spatial_from_grid(&mut self, bounds: [f32; 3], dims: (u16, u16, u16)) {
        let (w, h, d) = dims;
        let n = self.len();

        for i in 0..n {
            let x = (i % w as usize) as f32;
            let y = ((i / w as usize) % h as usize) as f32;
            let z = (i / (w as usize * h as usize)) as f32;

            // Scale to bounds
            let sx = if w > 1 { bounds[0] * x / (w as f32 - 1.0) } else { bounds[0] * 0.5 };
            let sy = if h > 1 { bounds[1] * y / (h as f32 - 1.0) } else { bounds[1] * 0.5 };
            let sz = if d > 1 { bounds[2] * z / (d as f32 - 1.0) } else { bounds[2] * 0.5 };

            self.soma_position[i] = [sx, sy, sz];
            self.axon_terminal[i] = [sx, sy, sz]; // Starts at soma
        }
    }

    /// Initialize spatial positions with random jitter within bounds.
    ///
    /// Uses a simple LCG PRNG seeded by the given seed.
    pub fn init_spatial_random(&mut self, bounds: [f32; 3], seed: u64) {
        let n = self.len();
        let mut rng = seed;

        for i in 0..n {
            // Simple LCG
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rx = ((rng >> 32) as u32) as f32 / u32::MAX as f32;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let ry = ((rng >> 32) as u32) as f32 / u32::MAX as f32;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rz = ((rng >> 32) as u32) as f32 / u32::MAX as f32;

            self.soma_position[i] = [bounds[0] * rx, bounds[1] * ry, bounds[2] * rz];
            self.axon_terminal[i] = self.soma_position[i]; // Starts at soma
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.membrane.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.membrane.is_empty()
    }

    /// Extend all SoA arrays with `additional` neurons at resting state.
    ///
    /// New neurons follow Dale's Law: first `n_new_excitatory` are excitatory,
    /// remaining are inhibitory. All start at resting potential with zero trace
    /// and no binding.
    ///
    /// Physical state: New neurons are placed at origin. Caller should assign
    /// positions via soma_position after extension.
    pub fn extend(&mut self, additional: u32, n_new_excitatory: u32, resting: i16, threshold: i16) {
        let n = additional as usize;
        let n_exc = n_new_excitatory as usize;

        for i in 0..n {
            let inhibitory = i >= n_exc;
            let profile = if inhibitory {
                NeuronProfile::FastSpiking
            } else {
                NeuronProfile::RegularSpiking
            };
            let f = flags::encode(inhibitory, profile);
            self.membrane.push(resting);
            self.threshold.push(threshold);
            self.leak.push(NeuronProfile::from_flags(f).default_leak());
            self.refract_remaining.push(0);
            self.flags.push(f);
            self.trace.push(0);
            self.spike_out.push(false);
            self.binding_slot.push(0);
            // v4: Physical state
            self.soma_position.push([0.0, 0.0, 0.0]);
            self.axon_terminal.push([0.0, 0.0, 0.0]);
            self.dendrite_radius.push(1.0);
            self.axon_health.push(128);
        }
    }

    /// Remove neurons at the given sorted-descending indices from all SoA arrays.
    ///
    /// `indices` MUST be sorted in descending order for correct swap-remove behavior.
    /// Returns the number of neurons actually removed.
    pub fn remove_descending(&mut self, indices: &[usize]) -> usize {
        let mut removed = 0;
        for &idx in indices {
            if idx >= self.membrane.len() { continue; }
            self.membrane.swap_remove(idx);
            self.threshold.swap_remove(idx);
            self.leak.swap_remove(idx);
            self.refract_remaining.swap_remove(idx);
            self.flags.swap_remove(idx);
            self.trace.swap_remove(idx);
            self.spike_out.swap_remove(idx);
            self.binding_slot.swap_remove(idx);
            // v4: Physical state
            self.soma_position.swap_remove(idx);
            self.axon_terminal.swap_remove(idx);
            self.dendrite_radius.swap_remove(idx);
            self.axon_health.swap_remove(idx);
            removed += 1;
        }
        removed
    }

    // === v4: Spatial Query Methods ===

    /// Find all neuron indices within `radius` of `pos`.
    ///
    /// Uses brute-force O(n) scan. For large pools, consider spatial indexing.
    pub fn neurons_in_radius(&self, pos: [f32; 3], radius: f32) -> Vec<usize> {
        let r2 = radius * radius;
        let mut result = Vec::new();

        for i in 0..self.len() {
            let dx = self.soma_position[i][0] - pos[0];
            let dy = self.soma_position[i][1] - pos[1];
            let dz = self.soma_position[i][2] - pos[2];
            let d2 = dx * dx + dy * dy + dz * dz;

            if d2 <= r2 {
                result.push(i);
            }
        }

        result
    }

    /// Compute local soma density at `pos` using a Gaussian kernel.
    ///
    /// Returns estimated density (neurons per unit volume).
    /// `sigma` controls the kernel width (default: 1.0).
    pub fn density_at(&self, pos: [f32; 3], sigma: f32) -> f32 {
        let sigma2 = sigma * sigma;
        let mut density = 0.0f32;

        for i in 0..self.len() {
            let dx = self.soma_position[i][0] - pos[0];
            let dy = self.soma_position[i][1] - pos[1];
            let dz = self.soma_position[i][2] - pos[2];
            let d2 = dx * dx + dy * dy + dz * dz;

            // Gaussian contribution: exp(-d² / 2σ²)
            density += (-d2 / (2.0 * sigma2)).exp();
        }

        density
    }

    /// Find the center of mass of neurons near `pos`.
    ///
    /// Returns None if no neurons within radius.
    pub fn cluster_center(&self, pos: [f32; 3], radius: f32) -> Option<[f32; 3]> {
        let nearby = self.neurons_in_radius(pos, radius);
        if nearby.is_empty() {
            return None;
        }

        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;

        for &i in &nearby {
            cx += self.soma_position[i][0];
            cy += self.soma_position[i][1];
            cz += self.soma_position[i][2];
        }

        let n = nearby.len() as f32;
        Some([cx / n, cy / n, cz / n])
    }

    /// Grow axon one step toward direction, returning true if successful.
    ///
    /// Growth fails probabilistically based on local density (resistance).
    pub fn grow_axon_step(&mut self, neuron_idx: usize, direction: [f32; 3], density_sigma: f32, resistance_factor: f32, seed: u64) -> bool {
        if neuron_idx >= self.len() {
            return false;
        }

        let terminal = self.axon_terminal[neuron_idx];
        let new_pos = [
            terminal[0] + direction[0],
            terminal[1] + direction[1],
            terminal[2] + direction[2],
        ];

        // Check density at new position
        let density = self.density_at(new_pos, density_sigma);
        let resistance = (density * resistance_factor).min(1.0);

        // Probabilistic growth based on resistance
        let rng = seed.wrapping_mul(6364136223846793005).wrapping_add(neuron_idx as u64);
        let rand_val = ((rng >> 32) as u32) as f32 / u32::MAX as f32;

        if rand_val > resistance {
            self.axon_terminal[neuron_idx] = new_pos;
            true
        } else {
            false
        }
    }

    /// Retract axon one step toward soma.
    pub fn retract_axon_step(&mut self, neuron_idx: usize, step_size: f32) {
        if neuron_idx >= self.len() {
            return;
        }

        let soma = self.soma_position[neuron_idx];
        let terminal = self.axon_terminal[neuron_idx];

        let dx = soma[0] - terminal[0];
        let dy = soma[1] - terminal[1];
        let dz = soma[2] - terminal[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < step_size {
            // Already at soma
            self.axon_terminal[neuron_idx] = soma;
        } else {
            // Move toward soma
            let scale = step_size / dist;
            self.axon_terminal[neuron_idx] = [
                terminal[0] + dx * scale,
                terminal[1] + dy * scale,
                terminal[2] + dz * scale,
            ];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_arrays_dale_law() {
        let arr = NeuronArrays::new(100, 80, -17920, -14080);
        assert_eq!(arr.len(), 100);

        // First 80 are excitatory
        for i in 0..80 {
            assert!(flags::is_excitatory(arr.flags[i]), "neuron {i} should be excitatory");
        }
        // Last 20 are inhibitory
        for i in 80..100 {
            assert!(flags::is_inhibitory(arr.flags[i]), "neuron {i} should be inhibitory");
        }
    }

    #[test]
    fn profile_encoding() {
        let f = flags::encode(false, NeuronProfile::IntrinsicBursting);
        assert!(flags::is_excitatory(f));
        assert_eq!(NeuronProfile::from_flags(f), NeuronProfile::IntrinsicBursting);

        let f2 = flags::encode(true, NeuronProfile::FastSpiking);
        assert!(flags::is_inhibitory(f2));
        assert_eq!(NeuronProfile::from_flags(f2), NeuronProfile::FastSpiking);
    }

    #[test]
    fn resting_state() {
        let arr = NeuronArrays::new(10, 8, -17920, -14080);
        for &m in &arr.membrane {
            assert_eq!(m, -17920);
        }
        for &t in &arr.threshold {
            assert_eq!(t, -14080);
        }
        for &r in &arr.refract_remaining {
            assert_eq!(r, 0);
        }
        for &tr in &arr.trace {
            assert_eq!(tr, 0);
        }
    }
}
