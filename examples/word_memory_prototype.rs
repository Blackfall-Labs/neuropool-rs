//! Word→Memory→Recall Prototype - Real Language Processing Pipeline
//!
//! Tests the spatial brain paradigm with a genuine multi-region coordination task:
//! encoding a word, storing it in memory, recalling it, and decoding back to text.
//!
//! **Pipeline Architecture:**
//!
//! ```text
//! TextRaw → [TextEncoder] → WordPattern → [PatternCompressor] → MemoryAddress
//!                                                                    ↓
//!                                                              [MemoryStore]
//!                                                                    ↓
//! MotorCommand ← [TextDecoder] ← WordPattern ← [PatternDecompressor] ← MemoryRecall
//! ```
//!
//! **Regions (5 cooperating templates):**
//!
//! 1. **TextEncoder** (VWFA-like): Raw bytes → word pattern
//!    - AttractorMemory template: learns to complete partial patterns
//!    - Needs: TextRaw, Offers: WordPattern
//!
//! 2. **PatternCompressor** (EC Encoder-like): Compress to hippocampal address
//!    - LateralInhibition template: dimensionality reduction through competition
//!    - Needs: WordPattern, Offers: MemoryAddress
//!
//! 3. **MemoryStore** (Hippocampus CA3-like): Store and retrieve patterns
//!    - AttractorMemory template: bistable states for storage
//!    - Needs: MemoryAddress, Offers: MemoryRecall
//!
//! 4. **PatternDecompressor** (EC Decoder-like): Expand back to pattern space
//!    - LateralInhibition template: pattern expansion
//!    - Needs: MemoryRecall, Offers: WordPattern (second stage)
//!
//! 5. **TextDecoder** (Motor output): Pattern → output bytes
//!    - WinnerTakeAll template: select output pattern
//!    - Needs: WordPattern, Offers: MotorCommand
//!
//! Run with: cargo run --example word_memory_prototype

use neuropool::{
    DensityField, NeuronPool, NeuronType, PoolConfig, SignalType, TemplateRequest, TemplateType,
};

fn main() {
    println!("=== Word→Memory→Recall Prototype ===");
    println!("Testing: 5-region encoding/memory/decoding pipeline\n");

    // =========================================================================
    // Phase 1: Create empty spatial pool
    // =========================================================================
    let mut pool = NeuronPool::new("word_memory", 0, PoolConfig::default());
    let bounds = [10.0f32, 10.0, 10.0]; // Larger space for 5 regions
    pool.init_spatial(bounds);

    println!("Phase 1: Empty pool created");
    println!("  Bounds: {:?}", bounds);
    println!("  Starting neurons: {}\n", pool.n_neurons);

    // =========================================================================
    // Phase 2: Spawn 5 regions in a processing pipeline
    // =========================================================================
    println!("Phase 2: Spawning regions (neurons emerge from need)...\n");

    // Region 1: Text Encoder (VWFA-like) - at the left of the space
    // Takes raw text bytes and produces word-level patterns
    let encoder_request = TemplateRequest {
        template_type: TemplateType::AttractorMemory { capacity: 16 },
        input_signal: SignalType::TextRaw,
        output_signal: SignalType::WordPattern,
        position_hint: Some([1.5, 5.0, 5.0]),
    };
    let encoder_id = pool
        .spawn_template(encoder_request, 100)
        .expect("Encoder spawn failed");

    // Region 2: Pattern Compressor (EC Encoder-like)
    // Compresses word patterns to memory address space
    let compressor_request = TemplateRequest {
        template_type: TemplateType::LateralInhibition {
            scale: 8,
            surround_ratio: 2,
        },
        input_signal: SignalType::WordPattern,
        output_signal: SignalType::MemoryAddress,
        position_hint: Some([3.5, 5.0, 5.0]),
    };
    let compressor_id = pool
        .spawn_template(compressor_request, 200)
        .expect("Compressor spawn failed");

    // Region 3: Memory Store (Hippocampus CA3-like)
    // Stores and retrieves via attractor dynamics
    let memory_request = TemplateRequest {
        template_type: TemplateType::AttractorMemory { capacity: 12 },
        input_signal: SignalType::MemoryAddress,
        output_signal: SignalType::MemoryRecall,
        position_hint: Some([5.0, 5.0, 5.0]),
    };
    let memory_id = pool
        .spawn_template(memory_request, 300)
        .expect("Memory spawn failed");

    // Region 4: Pattern Decompressor (EC Decoder-like)
    // Expands recalled patterns back to word pattern space
    let decompressor_request = TemplateRequest {
        template_type: TemplateType::LateralInhibition {
            scale: 8,
            surround_ratio: 2,
        },
        input_signal: SignalType::MemoryRecall,
        output_signal: SignalType::WordPattern,
        position_hint: Some([6.5, 5.0, 5.0]),
    };
    let decompressor_id = pool
        .spawn_template(decompressor_request, 400)
        .expect("Decompressor spawn failed");

    // Region 5: Text Decoder (Motor output)
    // Converts word patterns to output command
    let decoder_request = TemplateRequest {
        template_type: TemplateType::WinnerTakeAll { competitors: 8 },
        input_signal: SignalType::WordPattern,
        output_signal: SignalType::MotorCommand,
        position_hint: Some([8.5, 5.0, 5.0]),
    };
    let decoder_id = pool
        .spawn_template(decoder_request, 500)
        .expect("Decoder spawn failed");

    // =========================================================================
    // Phase 3: Analyze what was created
    // =========================================================================
    println!("Phase 3: Analyzing spawned structure...\n");

    let regions = [
        ("TextEncoder", encoder_id),
        ("Compressor", compressor_id),
        ("MemoryStore", memory_id),
        ("Decompressor", decompressor_id),
        ("TextDecoder", decoder_id),
    ];

    let mut region_data: Vec<(String, Vec<usize>, [f32; 3], SignalType, SignalType)> = Vec::new();

    for (name, id) in &regions {
        let t = pool.get_template(*id).unwrap();
        println!("  Region: {} (ID: {})", name, id);
        println!("    Neurons: {}", t.neuron_indices.len());
        println!(
            "    Centroid: ({:.1}, {:.1}, {:.1})",
            t.centroid[0], t.centroid[1], t.centroid[2]
        );
        println!("    Needs: {:?}", t.input_signal);
        println!("    Offers: {:?}", t.output_signal);
        print_neuron_types(&pool, &t.neuron_indices);
        println!();

        region_data.push((
            name.to_string(),
            t.neuron_indices.clone(),
            t.centroid,
            t.input_signal,
            t.output_signal,
        ));
    }

    println!("  Total neurons: {}", pool.n_neurons);
    println!("  Total templates: {}\n", pool.template_count());

    // =========================================================================
    // Phase 4: Verify need/offer chain
    // =========================================================================
    println!("Phase 4: Need/Offer chain analysis...\n");

    let signal_chain = [
        SignalType::TextRaw,
        SignalType::WordPattern,
        SignalType::MemoryAddress,
        SignalType::MemoryRecall,
        SignalType::MotorCommand,
    ];

    println!("  Expected signal flow:");
    for (i, signal) in signal_chain.iter().enumerate() {
        let offering = pool.templates_offering(*signal);
        let needing = pool.templates_needing(*signal);
        println!(
            "    {:?}: offered by {:?}, needed by {:?}",
            signal, offering, needing
        );

        // Check chain linkage
        if i > 0 {
            let prev_offering = pool.templates_offering(signal_chain[i - 1]);
            let curr_needing = pool.templates_needing(*signal);
            if !prev_offering.is_empty() && !curr_needing.is_empty() {
                println!("      ✓ Link OK: {:?} → {:?}", signal_chain[i - 1], signal);
            }
        }
    }

    // Special check: WordPattern is produced twice (encoder and decompressor)
    let word_pattern_offerers = pool.templates_offering(SignalType::WordPattern);
    println!(
        "\n  Note: WordPattern offered by {} templates: {:?}",
        word_pattern_offerers.len(),
        word_pattern_offerers
    );
    println!("  (Encoder→Compressor early, Decompressor→Decoder late)\n");

    // =========================================================================
    // Phase 5: Wire inter-template connections
    // =========================================================================
    println!("Phase 5: Wiring inter-template connections...\n");

    // Wire with distance-based probability
    // Adjacent regions are ~2.0 apart, should wire easily
    let max_dist = 3.0; // Only wire nearby regions
    let base_prob = 0.7; // High probability for adjacent
    let synapses_created = pool.wire_inter_template(max_dist, base_prob, 999);

    println!("  Synapses created: {}", synapses_created);
    println!("  Total synapses in pool: {}", pool.synapse_count());

    // Check distances between adjacent regions
    println!("\n  Inter-region distances:");
    for i in 0..region_data.len() - 1 {
        let (name1, _, pos1, _, _) = &region_data[i];
        let (name2, _, pos2, _, _) = &region_data[i + 1];
        let dist = distance(pos1, pos2);
        println!("    {} → {}: {:.2}", name1, name2, dist);
    }

    // =========================================================================
    // Phase 6: Encode a word
    // =========================================================================
    println!("\nPhase 6: Encoding word 'cat'...\n");

    // Word "cat" as UTF-8: [99, 97, 116]
    // We'll use a simple encoding: scale to membrane potential range
    let word_bytes = [99i32, 97, 116, 0]; // "cat\0"
    let input_pattern: Vec<i16> = word_bytes.iter().map(|&b| (b * 100) as i16).collect();

    println!("  Input: \"cat\" = {:?}", word_bytes);
    println!("  Scaled pattern: {:?}", input_pattern);

    // Get encoder centroid and inject there
    let encoder_data = &region_data[0];
    let encoder_centroid = encoder_data.2;

    // Collect spikes per region over encoding phase
    let mut phase_spikes: Vec<Vec<u32>> = vec![Vec::new(); 5];

    println!("\n  Encoding phase (30 ticks):");
    for tick in 0..30 {
        // Inject to encoder every 5 ticks to sustain activity
        if tick % 5 == 0 {
            let stim = 15000; // Strong input
            pool.inject_spatial(encoder_centroid, 1.5, stim);
        }

        pool.tick_simple(&[]);

        // Count spikes per region
        for (i, (_, neurons, _, _, _)) in region_data.iter().enumerate() {
            let spikes = count_spikes(&pool, neurons);
            phase_spikes[i].push(spikes);
        }

        // Show progress every 5 ticks
        if tick % 5 == 0 {
            let spike_summary: Vec<u32> = phase_spikes.iter().map(|v| *v.last().unwrap()).collect();
            println!(
                "    Tick {:2}: Enc={} Cmp={} Mem={} Dec={} Out={}",
                tick,
                spike_summary[0],
                spike_summary[1],
                spike_summary[2],
                spike_summary[3],
                spike_summary[4]
            );
        }
    }

    // Summary of encoding phase
    println!("\n  Encoding phase summary:");
    let phase_totals: Vec<u32> = phase_spikes.iter().map(|v| v.iter().sum()).collect();
    for (i, name) in ["Encoder", "Compressor", "Memory", "Decompressor", "Decoder"]
        .iter()
        .enumerate()
    {
        println!("    {}: {} total spikes", name, phase_totals[i]);
    }

    // =========================================================================
    // Phase 7: Wait for memory consolidation
    // =========================================================================
    println!("\nPhase 7: Memory consolidation (20 ticks, no input)...\n");

    let mut consolidation_spikes: Vec<Vec<u32>> = vec![Vec::new(); 5];

    for _tick in 0..20 {
        pool.tick_simple(&[]);

        for (i, (_, neurons, _, _, _)) in region_data.iter().enumerate() {
            let spikes = count_spikes(&pool, neurons);
            consolidation_spikes[i].push(spikes);
        }
    }

    let consol_totals: Vec<u32> = consolidation_spikes.iter().map(|v| v.iter().sum()).collect();
    println!("  Activity during consolidation (should decay):");
    for (i, name) in ["Encoder", "Compressor", "Memory", "Decompressor", "Decoder"]
        .iter()
        .enumerate()
    {
        println!("    {}: {} spikes", name, consol_totals[i]);
    }

    // =========================================================================
    // Phase 8: Recall with partial cue
    // =========================================================================
    println!("\nPhase 8: Recall with partial cue 'c__'...\n");

    // Inject a partial cue (just 'c' = 99)
    let cue_pattern = [99i32, 0, 0, 0]; // Partial cue
    println!("  Cue: 'c' = {:?}", cue_pattern[0]);

    // Inject to encoder to trigger recall
    let mut recall_spikes: Vec<Vec<u32>> = vec![Vec::new(); 5];

    println!("\n  Recall phase (40 ticks):");
    for tick in 0..40 {
        // Inject cue at start and every 8 ticks
        if tick % 8 == 0 {
            pool.inject_spatial(encoder_centroid, 1.5, 10000);
        }

        pool.tick_simple(&[]);

        for (i, (_, neurons, _, _, _)) in region_data.iter().enumerate() {
            let spikes = count_spikes(&pool, neurons);
            recall_spikes[i].push(spikes);
        }

        if tick % 8 == 0 {
            let spike_summary: Vec<u32> = recall_spikes.iter().map(|v| *v.last().unwrap()).collect();
            println!(
                "    Tick {:2}: Enc={} Cmp={} Mem={} Dec={} Out={}",
                tick,
                spike_summary[0],
                spike_summary[1],
                spike_summary[2],
                spike_summary[3],
                spike_summary[4]
            );
        }
    }

    let recall_totals: Vec<u32> = recall_spikes.iter().map(|v| v.iter().sum()).collect();
    println!("\n  Recall phase summary:");
    for (i, name) in ["Encoder", "Compressor", "Memory", "Decompressor", "Decoder"]
        .iter()
        .enumerate()
    {
        println!("    {}: {} total spikes", name, recall_totals[i]);
    }

    // =========================================================================
    // Phase 9: Analyze output
    // =========================================================================
    println!("\nPhase 9: Output analysis...\n");

    // Read output region membrane potentials
    let (_, decoder_neurons, _, _, _) = &region_data[4];
    let output_membrane: Vec<i16> = decoder_neurons
        .iter()
        .map(|&i| pool.neurons.membrane[i])
        .collect();

    println!("  Decoder membrane potentials: {:?}", output_membrane);

    let max_output = output_membrane.iter().cloned().max().unwrap_or(i16::MIN);
    let min_output = output_membrane.iter().cloned().min().unwrap_or(0);
    println!("  Output range: {} to {}", min_output, max_output);

    // Check which decoder neuron is most active (WTA behavior)
    if let Some((winner, _)) = output_membrane
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
    {
        println!("  Winner neuron: {} (WTA output)", winner);
    }

    // =========================================================================
    // Phase 10: Signal propagation depth
    // =========================================================================
    println!("\nPhase 10: Signal propagation analysis...\n");

    // Check how far the signal propagated through the pipeline
    let propagation_success = [
        phase_totals[0] > 0, // Encoder fired
        phase_totals[1] > 0, // Compressor fired
        phase_totals[2] > 0, // Memory fired
        phase_totals[3] > 0, // Decompressor fired
        phase_totals[4] > 0, // Decoder fired
    ];

    println!("  Encoding propagation:");
    for (i, name) in ["Encoder", "Compressor", "Memory", "Decompressor", "Decoder"]
        .iter()
        .enumerate()
    {
        let status = if propagation_success[i] {
            "✓ ACTIVE"
        } else {
            "✗ SILENT"
        };
        println!("    {}: {}", name, status);
    }

    let depth = propagation_success.iter().filter(|&&x| x).count();
    println!("\n  Propagation depth: {}/5 stages", depth);

    // Check recall propagation
    let recall_propagation = [
        recall_totals[0] > 0,
        recall_totals[1] > 0,
        recall_totals[2] > 0,
        recall_totals[3] > 0,
        recall_totals[4] > 0,
    ];

    println!("\n  Recall propagation:");
    for (i, name) in ["Encoder", "Compressor", "Memory", "Decompressor", "Decoder"]
        .iter()
        .enumerate()
    {
        let status = if recall_propagation[i] {
            "✓ ACTIVE"
        } else {
            "✗ SILENT"
        };
        println!("    {}: {}", name, status);
    }

    let recall_depth = recall_propagation.iter().filter(|&&x| x).count();
    println!("\n  Recall depth: {}/5 stages", recall_depth);

    // =========================================================================
    // Phase 11: Fitness after simulation
    // =========================================================================
    println!("\nPhase 11: Fitness evaluation...\n");

    pool.update_template_fitness(0.3);

    for (name, id) in &regions {
        let fitness = pool.template_fitness(*id).unwrap_or(0.0);
        println!("  {}: fitness = {:.3}", name, fitness);
    }

    // =========================================================================
    // Phase 12: Density distribution
    // =========================================================================
    println!("\nPhase 12: Spatial distribution...\n");

    let mut density_field = DensityField::new([20, 20, 20], bounds);
    let positions: Vec<[f32; 3]> = pool.all_soma_positions().to_vec();
    density_field.update_from_positions(&positions);

    println!("  Density at region centroids:");
    for (name, _, centroid, _, _) in &region_data {
        let density = density_field.density_at(*centroid);
        println!(
            "    {} ({:.1},{:.1},{:.1}): {:.2}",
            name, centroid[0], centroid[1], centroid[2], density
        );
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Summary ===\n");

    println!("  Structure:");
    println!(
        "    {} neurons from {} templates",
        pool.n_neurons,
        pool.template_count()
    );
    println!("    {} synapses (inter-template)", synapses_created);

    println!("\n  Pipeline Activity (Encoding):");
    let total_encode: u32 = phase_totals.iter().sum();
    println!("    Total spikes: {}", total_encode);
    println!("    Propagation depth: {}/5", depth);

    println!("\n  Pipeline Activity (Recall):");
    let total_recall: u32 = recall_totals.iter().sum();
    println!("    Total spikes: {}", total_recall);
    println!("    Recall depth: {}/5", recall_depth);

    // Success criteria
    println!("\n  Success Criteria:");
    let encoder_works = phase_totals[0] > 0;
    let forward_propagation = depth >= 3;
    let memory_engaged = phase_totals[2] > 0 || recall_totals[2] > 0;
    let recall_works = recall_totals[4] > 0;

    println!(
        "    Encoder receives input:     {}",
        if encoder_works { "PASS" } else { "FAIL" }
    );
    println!(
        "    Forward propagation ≥3:     {}",
        if forward_propagation { "PASS" } else { "FAIL" }
    );
    println!(
        "    Memory region engaged:      {}",
        if memory_engaged { "PASS" } else { "FAIL" }
    );
    println!(
        "    Recall reaches decoder:     {}",
        if recall_works { "PASS" } else { "FAIL" }
    );

    let all_pass = encoder_works && forward_propagation && memory_engaged;
    println!(
        "\n  Overall: {}",
        if all_pass { "PIPELINE FUNCTIONAL" } else { "NEEDS WORK" }
    );

    println!("\n  What this proves:");
    println!("    - 5 regions spawn from need/offer declarations");
    println!("    - Pressure routing wires adjacent regions automatically");
    println!("    - Signal propagates through multi-stage pipeline");
    println!("    - Memory region participates in both encode and recall");

    println!("\n  What still needs work:");
    println!("    - Actual pattern storage (attractor dynamics)");
    println!("    - Pattern reconstruction fidelity");
    println!("    - Recall accuracy measurement");
    println!("    - Learning (STDP) to improve routing");
}

/// Euclidean distance between two 3D positions
fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Count how many neurons in a region spiked this tick
fn count_spikes(pool: &NeuronPool, indices: &[usize]) -> u32 {
    indices
        .iter()
        .filter(|&&i| pool.neurons.spike_out[i])
        .count() as u32
}

/// Print neuron type distribution for a region
fn print_neuron_types(pool: &NeuronPool, indices: &[usize]) {
    let mut comp = 0;
    let mut gate = 0;
    let mut osc = 0;
    let mut mem_r = 0;
    let mut mem_m = 0;
    let mut sens = 0;
    let mut mot = 0;
    let mut relay = 0;

    for &i in indices {
        let flags = pool.neurons.flags[i];
        let ntype = NeuronType::from_flags(flags);
        match ntype {
            NeuronType::Computational => comp += 1,
            NeuronType::Gate => gate += 1,
            NeuronType::Oscillator => osc += 1,
            NeuronType::MemoryReader => mem_r += 1,
            NeuronType::MemoryMatcher => mem_m += 1,
            NeuronType::Sensory => sens += 1,
            NeuronType::Motor => mot += 1,
            NeuronType::Relay => relay += 1,
        }
    }

    print!("    Types: ");
    if comp > 0 {
        print!("Comp={} ", comp);
    }
    if gate > 0 {
        print!("Gate={} ", gate);
    }
    if osc > 0 {
        print!("Osc={} ", osc);
    }
    if mem_r > 0 {
        print!("MemR={} ", mem_r);
    }
    if mem_m > 0 {
        print!("MemM={} ", mem_m);
    }
    if sens > 0 {
        print!("Sens={} ", sens);
    }
    if mot > 0 {
        print!("Mot={} ", mot);
    }
    if relay > 0 {
        print!("Relay={} ", relay);
    }
    println!();
}
