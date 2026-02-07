//! Anatomical structures of a spatial neuron.
//!
//! Every spatial neuron has the same anatomical components:
//! - **Soma**: The cell body, where the nucleus lives
//! - **Dendrite**: Reception apparatus, collects incoming signals
//! - **Axon**: Transmission apparatus, carries output to targets

/// The cell body — where the neuron lives in space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Soma {
    /// 3D position in continuous space
    pub position: [f32; 3],
}

impl Soma {
    /// Create a soma at the given position.
    #[inline]
    pub const fn at(position: [f32; 3]) -> Self {
        Self { position }
    }

    /// Create a soma at the origin.
    #[inline]
    pub const fn origin() -> Self {
        Self { position: [0.0, 0.0, 0.0] }
    }

    /// Euclidean distance to another soma.
    #[inline]
    pub fn distance_to(&self, other: &Soma) -> f32 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Move the soma by a delta.
    #[inline]
    pub fn translate(&mut self, delta: [f32; 3]) {
        self.position[0] += delta[0];
        self.position[1] += delta[1];
        self.position[2] += delta[2];
    }
}

impl Default for Soma {
    fn default() -> Self {
        Self::origin()
    }
}

/// Reception apparatus — collects incoming signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dendrite {
    /// Reception range (how far can synapses connect from?)
    pub radius: f32,
    /// Number of dendritic spines (synaptic capacity)
    pub spine_count: u16,
}

impl Dendrite {
    /// Create a dendrite with given radius and spine count.
    #[inline]
    pub const fn new(radius: f32, spine_count: u16) -> Self {
        Self { radius, spine_count }
    }

    /// Default dendrite with 1.0 radius and 100 spines.
    #[inline]
    pub const fn standard() -> Self {
        Self { radius: 1.0, spine_count: 100 }
    }

    /// Can this dendrite accept more synapses?
    #[inline]
    pub const fn has_capacity(&self, current_synapses: u16) -> bool {
        current_synapses < self.spine_count
    }
}

impl Default for Dendrite {
    fn default() -> Self {
        Self::standard()
    }
}

/// Transmission apparatus — carries signal to targets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axon {
    /// Where the axon ends (target region)
    pub terminal: [f32; 3],
    /// Myelination level (0=unmyelinated, 255=fully myelinated)
    /// Higher myelin = faster conduction, better protection from decay
    pub myelin: u8,
    /// Survival pressure (0=dead/pruned, 255=maximally healthy)
    /// Decays when inactive, boosted when carrying spikes
    pub health: u8,
}

impl Axon {
    /// Create an axon pointing toward a target.
    #[inline]
    pub const fn toward(terminal: [f32; 3]) -> Self {
        Self {
            terminal,
            myelin: 0,
            health: 128,
        }
    }

    /// Create a healthy, myelinated axon.
    #[inline]
    pub const fn myelinated(terminal: [f32; 3], myelin: u8) -> Self {
        Self {
            terminal,
            myelin,
            health: 255,
        }
    }

    /// Compute axon length from soma position.
    #[inline]
    pub fn length(&self, soma_pos: [f32; 3]) -> f32 {
        let dx = self.terminal[0] - soma_pos[0];
        let dy = self.terminal[1] - soma_pos[1];
        let dz = self.terminal[2] - soma_pos[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Is this axon alive (health > 0)?
    #[inline]
    pub const fn is_alive(&self) -> bool {
        self.health > 0
    }

    /// Is this axon myelinated?
    #[inline]
    pub const fn is_myelinated(&self) -> bool {
        self.myelin > 0
    }

    /// Apply health decay (when inactive).
    #[inline]
    pub fn decay(&mut self, amount: u8) {
        self.health = self.health.saturating_sub(amount);
    }

    /// Boost health (when carrying spikes).
    #[inline]
    pub fn boost(&mut self, amount: u8) {
        self.health = self.health.saturating_add(amount);
    }

    /// Apply myelination (gradual process based on myelin_affinity).
    #[inline]
    pub fn myelinate(&mut self, amount: u8) {
        self.myelin = self.myelin.saturating_add(amount);
    }

    /// Retract axon toward soma by a fraction.
    #[inline]
    pub fn retract_toward(&mut self, soma_pos: [f32; 3], fraction: f32) {
        let dx = soma_pos[0] - self.terminal[0];
        let dy = soma_pos[1] - self.terminal[1];
        let dz = soma_pos[2] - self.terminal[2];
        self.terminal[0] += dx * fraction;
        self.terminal[1] += dy * fraction;
        self.terminal[2] += dz * fraction;
    }

    /// Extend axon away from soma toward target.
    #[inline]
    pub fn extend_toward(&mut self, target: [f32; 3], step: f32) {
        let dx = target[0] - self.terminal[0];
        let dy = target[1] - self.terminal[1];
        let dz = target[2] - self.terminal[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > 0.0 {
            let factor = step / dist;
            self.terminal[0] += dx * factor;
            self.terminal[1] += dy * factor;
            self.terminal[2] += dz * factor;
        }
    }
}

impl Default for Axon {
    fn default() -> Self {
        Self::toward([0.0, 0.0, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soma_distance() {
        let a = Soma::at([0.0, 0.0, 0.0]);
        let b = Soma::at([3.0, 4.0, 0.0]);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_axon_length() {
        let axon = Axon::toward([3.0, 4.0, 0.0]);
        let len = axon.length([0.0, 0.0, 0.0]);
        assert!((len - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_axon_health() {
        let mut axon = Axon::toward([1.0, 0.0, 0.0]);
        assert!(axon.is_alive());
        axon.decay(200);
        assert!(!axon.is_alive());
    }

    #[test]
    fn test_dendrite_capacity() {
        let d = Dendrite::new(1.0, 10);
        assert!(d.has_capacity(5));
        assert!(!d.has_capacity(10));
    }
}
