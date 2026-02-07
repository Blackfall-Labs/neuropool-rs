//! Computational Templates (Circuit Motifs)
//!
//! Templates are structural recipes that the spatial brain instantiates.
//! Programs request templates by expressing computational NEED, not wiring.
//! The spatial brain spawns neurons according to template recipes, and
//! evolution (growth/pruning) fills in the connections.
//!
//! ## Base Circuit Motifs
//!
//! | Motif | What It Solves |
//! |-------|----------------|
//! | Lateral Inhibition | Edge detection, contrast, competition |
//! | Recurrent Attractor | Pattern completion, associative recall |
//! | Temporal Chain | Sequence prediction, motor plans |
//! | Oscillator Network | Timing, rhythm, binding |
//! | Disinhibition Gate | Selective routing, attention |
//! | Winner-Take-All | Classification, decision |

// NeuronType is used indirectly via type_distribution() which returns counts

/// Signal types for need/offer matching.
///
/// Templates declare what signals they need as input and what they offer as output.
/// Axons grow toward need gradients; connections form when need matches offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SignalType {
    /// Raw sensory input (unprocessed)
    SensoryRaw = 0,
    /// Visual: retinal activity
    VisualRetinal = 1,
    /// Visual: edge signals (V1 output)
    VisualEdge = 2,
    /// Visual: shape patterns (V2 output)
    VisualShape = 3,
    /// Visual: bound objects (V4 output)
    VisualObject = 4,
    /// Auditory: raw cochlear
    AuditoryRaw = 10,
    /// Auditory: onset/offset events
    AuditoryOnset = 11,
    /// Auditory: pitch patterns
    AuditoryPitch = 12,
    /// Auditory: rhythm/beat
    AuditoryRhythm = 13,
    /// Motor: movement commands
    MotorCommand = 20,
    /// Motor: position sense
    MotorProprioception = 21,
    /// Motor: efference copy
    MotorEfference = 22,
    /// Cognitive: attention signal
    CognitiveAttention = 30,
    /// Cognitive: working memory
    CognitiveWorkingMem = 31,
    /// Cognitive: decision output
    CognitiveDecision = 32,
    /// Chemical: neuromodulator levels
    ChemicalModulator = 40,
    /// Language: raw text/UTF-8 bytes
    TextRaw = 50,
    /// Language: word-level pattern (VWFA output)
    WordPattern = 51,
    /// Language: phoneme sequence
    PhonemeSequence = 52,
    /// Language: semantic embedding
    SemanticEmbedding = 53,
    /// Memory: compressed hippocampal address (EC encoder output)
    MemoryAddress = 60,
    /// Memory: recalled pattern (EC decoder output)
    MemoryRecall = 61,
    /// Memory: episodic context
    EpisodicContext = 62,
    /// Memory: semantic association
    SemanticAssociation = 63,
    /// Generic: user-defined
    Custom(u8) = 255,
}

impl SignalType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => SignalType::SensoryRaw,
            1 => SignalType::VisualRetinal,
            2 => SignalType::VisualEdge,
            3 => SignalType::VisualShape,
            4 => SignalType::VisualObject,
            10 => SignalType::AuditoryRaw,
            11 => SignalType::AuditoryOnset,
            12 => SignalType::AuditoryPitch,
            13 => SignalType::AuditoryRhythm,
            20 => SignalType::MotorCommand,
            21 => SignalType::MotorProprioception,
            22 => SignalType::MotorEfference,
            30 => SignalType::CognitiveAttention,
            31 => SignalType::CognitiveWorkingMem,
            32 => SignalType::CognitiveDecision,
            40 => SignalType::ChemicalModulator,
            50 => SignalType::TextRaw,
            51 => SignalType::WordPattern,
            52 => SignalType::PhonemeSequence,
            53 => SignalType::SemanticEmbedding,
            60 => SignalType::MemoryAddress,
            61 => SignalType::MemoryRecall,
            62 => SignalType::EpisodicContext,
            63 => SignalType::SemanticAssociation,
            _ => SignalType::Custom(v),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            SignalType::SensoryRaw => 0,
            SignalType::VisualRetinal => 1,
            SignalType::VisualEdge => 2,
            SignalType::VisualShape => 3,
            SignalType::VisualObject => 4,
            SignalType::AuditoryRaw => 10,
            SignalType::AuditoryOnset => 11,
            SignalType::AuditoryPitch => 12,
            SignalType::AuditoryRhythm => 13,
            SignalType::MotorCommand => 20,
            SignalType::MotorProprioception => 21,
            SignalType::MotorEfference => 22,
            SignalType::CognitiveAttention => 30,
            SignalType::CognitiveWorkingMem => 31,
            SignalType::CognitiveDecision => 32,
            SignalType::ChemicalModulator => 40,
            SignalType::TextRaw => 50,
            SignalType::WordPattern => 51,
            SignalType::PhonemeSequence => 52,
            SignalType::SemanticEmbedding => 53,
            SignalType::MemoryAddress => 60,
            SignalType::MemoryRecall => 61,
            SignalType::EpisodicContext => 62,
            SignalType::SemanticAssociation => 63,
            SignalType::Custom(v) => v,
        }
    }
}

/// Template types — structural recipes for circuit motifs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemplateType {
    /// Center-surround inhibition for edge detection and competition.
    ///
    /// Structure: N computational neurons, each surrounded by `surround_ratio` gate neurons.
    /// Equation: out[i] = relu(in[i] - k * Σ(gate[j].output for j in neighbors))
    LateralInhibition {
        /// Number of center (computational) neurons
        scale: u16,
        /// Gate neurons per center neuron (typically 4)
        surround_ratio: u8,
    },

    /// Recurrent attractor network for pattern completion.
    ///
    /// Structure: K memory neurons with strong self-recurrence.
    /// Equation: state[t+1] = f(W * state[t] + input) — settles to fixed points.
    AttractorMemory {
        /// Number of storable patterns (memory neurons)
        capacity: u16,
    },

    /// Sequential activation chain for sequences and motor plans.
    ///
    /// Structure: L computational neurons in chain, 1 gate for control.
    /// Equation: active[i,t+1] = active[i-1,t] * eligibility[i] * gate_open
    TemporalChain {
        /// Chain length (number of steps)
        length: u16,
    },

    /// Oscillator-driven network for timing and binding.
    ///
    /// Structure: 1 oscillator neuron paces M followers via phase coupling.
    /// Equation: phase[i,t+1] = phase[i,t] + freq[i] + coupling * sin(phase[j] - phase[i])
    OscillatorNetwork {
        /// Target frequency in Hz (e.g., 40 for gamma)
        pacemaker_hz: u8,
        /// Number of follower neurons
        follower_count: u16,
    },

    /// Gate-inhibits-gate for selective routing.
    ///
    /// Structure: 2 gate neurons where one inhibits the other.
    /// Use case: attention gating, conditional release.
    DisinhibitionGate,

    /// Strong lateral inhibition for classification.
    ///
    /// Structure: N computational neurons with mutual inhibition via gates.
    /// Equation: out[i] = 1 if in[i] > max(in[j≠i]) else 0
    WinnerTakeAll {
        /// Number of competing options
        competitors: u16,
    },

    /// Sensory input array — fixed topology like a cochlea or retina.
    ///
    /// Structure: N sensory neurons in a linear array. Each neuron corresponds
    /// to one input dimension. Input value X activates neuron X.
    /// Use case: byte → neuron mapping for text, frequency → position for audio.
    SensoryArray {
        /// Number of input dimensions (e.g., 128 for ASCII)
        dimensions: u16,
    },
}

impl TemplateType {
    /// Total neuron count this template requires.
    pub fn neuron_count(&self) -> usize {
        match self {
            TemplateType::LateralInhibition { scale, surround_ratio } => {
                *scale as usize * (1 + *surround_ratio as usize)
            }
            TemplateType::AttractorMemory { capacity } => *capacity as usize,
            TemplateType::TemporalChain { length } => *length as usize + 1, // +1 for gate
            TemplateType::OscillatorNetwork { follower_count, .. } => {
                1 + *follower_count as usize // 1 pacemaker + followers
            }
            TemplateType::DisinhibitionGate => 2,
            TemplateType::WinnerTakeAll { competitors } => {
                // Each competitor + mutual inhibition gates
                *competitors as usize * 2
            }
            TemplateType::SensoryArray { dimensions } => *dimensions as usize,
        }
    }

    /// Neuron type distribution for this template.
    ///
    /// Returns (computational, gate, oscillator, memory, sensory, motor).
    pub fn type_distribution(&self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            TemplateType::LateralInhibition { scale, surround_ratio } => {
                let comp = *scale as usize;
                let gate = *scale as usize * *surround_ratio as usize;
                (comp, gate, 0, 0, 0, 0)
            }
            TemplateType::AttractorMemory { capacity } => {
                (0, 0, 0, *capacity as usize, 0, 0)
            }
            TemplateType::TemporalChain { length } => {
                (*length as usize, 1, 0, 0, 0, 0) // chain + 1 gate
            }
            TemplateType::OscillatorNetwork { follower_count, .. } => {
                (*follower_count as usize, 0, 1, 0, 0, 0) // followers + 1 oscillator
            }
            TemplateType::DisinhibitionGate => (0, 2, 0, 0, 0, 0),
            TemplateType::WinnerTakeAll { competitors } => {
                (*competitors as usize, *competitors as usize, 0, 0, 0, 0)
            }
            TemplateType::SensoryArray { dimensions } => {
                // All sensory neurons
                (0, 0, 0, 0, *dimensions as usize, 0)
            }
        }
    }
}

/// A request to instantiate a template.
#[derive(Debug, Clone)]
pub struct TemplateRequest {
    /// The type of circuit motif to create
    pub template_type: TemplateType,
    /// What signal type this template needs as input
    pub input_signal: SignalType,
    /// What signal type this template produces as output
    pub output_signal: SignalType,
    /// Preferred spatial location (None = auto-place in low-density area)
    pub position_hint: Option<[f32; 3]>,
}

/// A live template instance in the pool.
#[derive(Debug, Clone)]
pub struct TemplateInstance {
    /// Unique identifier for this instance
    pub id: u32,
    /// The template type that was instantiated
    pub template_type: TemplateType,
    /// Indices of neurons belonging to this template
    pub neuron_indices: Vec<usize>,
    /// Spatial center of mass
    pub centroid: [f32; 3],
    /// Input signal type this template needs
    pub input_signal: SignalType,
    /// Output signal type this template offers
    pub output_signal: SignalType,
    /// Fitness score: correlation between input satisfaction and output quality
    /// Range: 0.0 (useless) to 1.0 (perfectly solving the need)
    pub fitness: f32,
    /// How many ticks this template has existed
    pub age_ticks: u64,
    /// Cumulative activity (sum of spikes from template neurons)
    pub cumulative_activity: u64,
}

impl TemplateInstance {
    /// Create a new template instance.
    pub fn new(
        id: u32,
        template_type: TemplateType,
        neuron_indices: Vec<usize>,
        centroid: [f32; 3],
        input_signal: SignalType,
        output_signal: SignalType,
    ) -> Self {
        Self {
            id,
            template_type,
            neuron_indices,
            centroid,
            input_signal,
            output_signal,
            fitness: 0.5, // Start neutral
            age_ticks: 0,
            cumulative_activity: 0,
        }
    }

    /// Update fitness based on recent correlation measurement.
    ///
    /// Uses exponential moving average to smooth fitness over time.
    pub fn update_fitness(&mut self, measured_correlation: f32, alpha: f32) {
        self.fitness = self.fitness * (1.0 - alpha) + measured_correlation * alpha;
    }

    /// Check if this template should be pruned due to low fitness.
    ///
    /// Templates are protected during a grace period (first N ticks).
    pub fn should_prune(&self, fitness_threshold: f32, grace_period: u64) -> bool {
        self.age_ticks > grace_period && self.fitness < fitness_threshold
    }

    /// Tick the template (increment age, called each simulation tick).
    pub fn tick(&mut self) {
        self.age_ticks += 1;
    }
}

/// Registry of active templates in a pool.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    /// All active template instances
    instances: Vec<TemplateInstance>,
    /// Next ID to assign
    next_id: u32,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new template instance.
    pub fn register(&mut self, mut instance: TemplateInstance) -> u32 {
        let id = self.next_id;
        instance.id = id;
        self.next_id += 1;
        self.instances.push(instance);
        id
    }

    /// Get a template by ID.
    pub fn get(&self, id: u32) -> Option<&TemplateInstance> {
        self.instances.iter().find(|t| t.id == id)
    }

    /// Get a mutable template by ID.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut TemplateInstance> {
        self.instances.iter_mut().find(|t| t.id == id)
    }

    /// Remove a template by ID. Returns the removed instance if found.
    pub fn remove(&mut self, id: u32) -> Option<TemplateInstance> {
        if let Some(pos) = self.instances.iter().position(|t| t.id == id) {
            Some(self.instances.swap_remove(pos))
        } else {
            None
        }
    }

    /// Get all templates.
    pub fn all(&self) -> &[TemplateInstance] {
        &self.instances
    }

    /// Get all templates mutably.
    pub fn all_mut(&mut self) -> &mut [TemplateInstance] {
        &mut self.instances
    }

    /// Find templates that need a specific signal type.
    pub fn templates_needing(&self, signal: SignalType) -> Vec<u32> {
        self.instances
            .iter()
            .filter(|t| t.input_signal == signal)
            .map(|t| t.id)
            .collect()
    }

    /// Find templates that offer a specific signal type.
    pub fn templates_offering(&self, signal: SignalType) -> Vec<u32> {
        self.instances
            .iter()
            .filter(|t| t.output_signal == signal)
            .map(|t| t.id)
            .collect()
    }

    /// Prune templates below fitness threshold.
    ///
    /// Returns IDs of pruned templates.
    pub fn prune_unfit(&mut self, fitness_threshold: f32, grace_period: u64) -> Vec<u32> {
        let mut pruned = Vec::new();
        self.instances.retain(|t| {
            if t.should_prune(fitness_threshold, grace_period) {
                pruned.push(t.id);
                false
            } else {
                true
            }
        });
        pruned
    }

    /// Tick all templates.
    pub fn tick_all(&mut self) {
        for t in &mut self.instances {
            t.tick();
        }
    }

    /// Number of active templates.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// Spatial arrangement for template neurons.
#[derive(Debug, Clone, Copy)]
pub enum SpatialArrangement {
    /// Random positions within radius of centroid
    Random { radius: f32 },
    /// Grid layout centered at position
    Grid { spacing: f32 },
    /// Ring layout around center (for surround inhibition)
    Ring { radius: f32 },
    /// Chain layout extending in direction
    Chain { step: f32, direction: [f32; 3] },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lateral_inhibition_neuron_count() {
        let t = TemplateType::LateralInhibition {
            scale: 16,
            surround_ratio: 4,
        };
        assert_eq!(t.neuron_count(), 16 * 5); // 16 centers + 64 gates
    }

    #[test]
    fn test_attractor_memory_count() {
        let t = TemplateType::AttractorMemory { capacity: 32 };
        assert_eq!(t.neuron_count(), 32);
    }

    #[test]
    fn test_temporal_chain_count() {
        let t = TemplateType::TemporalChain { length: 10 };
        assert_eq!(t.neuron_count(), 11); // 10 chain + 1 gate
    }

    #[test]
    fn test_oscillator_network_count() {
        let t = TemplateType::OscillatorNetwork {
            pacemaker_hz: 40,
            follower_count: 8,
        };
        assert_eq!(t.neuron_count(), 9); // 1 pacemaker + 8 followers
    }

    #[test]
    fn test_template_registry() {
        let mut reg = TemplateRegistry::new();

        let instance = TemplateInstance::new(
            0,
            TemplateType::LateralInhibition {
                scale: 4,
                surround_ratio: 4,
            },
            vec![0, 1, 2, 3],
            [1.0, 2.0, 3.0],
            SignalType::VisualRetinal,
            SignalType::VisualEdge,
        );

        let id = reg.register(instance);
        assert_eq!(id, 1);
        assert_eq!(reg.len(), 1);

        let t = reg.get(id).unwrap();
        assert_eq!(t.input_signal, SignalType::VisualRetinal);
        assert_eq!(t.output_signal, SignalType::VisualEdge);
    }

    #[test]
    fn test_fitness_pruning() {
        let mut reg = TemplateRegistry::new();

        let mut instance = TemplateInstance::new(
            0,
            TemplateType::DisinhibitionGate,
            vec![0, 1],
            [0.0, 0.0, 0.0],
            SignalType::CognitiveAttention,
            SignalType::CognitiveDecision,
        );

        // Simulate aging past grace period
        instance.age_ticks = 1000;
        instance.fitness = 0.1; // Low fitness

        let id = reg.register(instance);
        let pruned = reg.prune_unfit(0.2, 500);

        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0], id);
        assert!(reg.is_empty());
    }

    #[test]
    fn test_signal_type_roundtrip() {
        for v in [0u8, 1, 2, 10, 20, 30, 40, 100] {
            let s = SignalType::from_u8(v);
            let back = s.to_u8();
            // Custom types preserve the value
            if v > 40 {
                assert_eq!(back, v);
            }
        }
    }
}
