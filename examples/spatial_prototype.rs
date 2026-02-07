//! Spatial Brain Prototype - Real Problem Solving
//!
//! Tests the spatial brain paradigm with actual computational problems:
//!
//! **Problem 1: Edge Detection (LI template)**
//!   - Input: raw "visual" pattern
//!   - Output: edge-enhanced signal
//!   - Cooperation: feeds into classifier
//!
//! **Problem 2: Pattern Classification (WTA template)**
//!   - Input: edges from LI
//!   - Output: winner neuron indicates class
//!   - Cooperation: depends on LI output
//!
//! **Problem 3: Rhythm Generation (Oscillator)**
//!   - Input: none (intrinsic)
//!   - Output: periodic activity
//!   - Independent: runs regardless of vision pipeline
//!
//! Run with: cargo run --example spatial_prototype

use neuropool::{
    NeuronPool, PoolConfig, DensityField, NeuronType,
    SignalType, TemplateType, TemplateRequest,
};

fn main() {
    println!("=== Spatial Brain Prototype ===");
    println!("Testing: 3 regions, 2 cooperating, 1 independent\n");

    // =========================================================================
    // Phase 1: Create empty spatial pool
    // =========================================================================
    let mut pool = NeuronPool::new("spatial_test", 0, PoolConfig::default());
    let bounds = [6.0f32, 6.0, 6.0]; // Larger space for 3 regions
    pool.init_spatial(bounds);

    println!("Phase 1: Empty pool created");
    println!("  Bounds: {:?}\n", bounds);

    // =========================================================================
    // Phase 2: Spawn 3 regions with distinct problems
    // =========================================================================
    println!("Phase 2: Spawning regions (neurons emerge from need)...\n");

    // Region 1: Edge Detection (Lateral Inhibition)
    let li_request = TemplateRequest {
        template_type: TemplateType::LateralInhibition { scale: 8, surround_ratio: 4 },
        input_signal: SignalType::VisualRetinal,
        output_signal: SignalType::VisualEdge,
        position_hint: Some([1.5, 3.0, 3.0]),
    };
    let li_id = pool.spawn_template(li_request, 42).expect("LI spawn failed");

    // Region 2: Pattern Classification (Winner-Take-All)
    let wta_request = TemplateRequest {
        template_type: TemplateType::WinnerTakeAll { competitors: 4 },
        input_signal: SignalType::VisualEdge, // Needs what LI offers!
        output_signal: SignalType::MotorCommand,
        position_hint: Some([4.5, 3.0, 3.0]),
    };
    let wta_id = pool.spawn_template(wta_request, 123).expect("WTA spawn failed");

    // Region 3: Rhythm Generator (Oscillator) - Independent
    let osc_request = TemplateRequest {
        template_type: TemplateType::OscillatorNetwork { pacemaker_hz: 20, follower_count: 6 },
        input_signal: SignalType::SensoryRaw,
        output_signal: SignalType::AuditoryRhythm,
        position_hint: Some([3.0, 1.0, 3.0]),
    };
    let osc_id = pool.spawn_template(osc_request, 456).expect("Osc spawn failed");

    // =========================================================================
    // Phase 3: Analyze what was created - copy data to avoid borrow issues
    // =========================================================================
    println!("Phase 3: Analyzing spawned structure...\n");

    // Copy template data to avoid holding borrows
    let (li_neurons, li_centroid, li_input, li_output) = {
        let t = pool.get_template(li_id).unwrap();
        (t.neuron_indices.clone(), t.centroid, t.input_signal, t.output_signal)
    };
    let (wta_neurons, wta_centroid, wta_input, wta_output) = {
        let t = pool.get_template(wta_id).unwrap();
        (t.neuron_indices.clone(), t.centroid, t.input_signal, t.output_signal)
    };
    let (osc_neurons, osc_centroid, osc_input, osc_output) = {
        let t = pool.get_template(osc_id).unwrap();
        (t.neuron_indices.clone(), t.centroid, t.input_signal, t.output_signal)
    };

    println!("  Region 1: Edge Detection (LI)");
    println!("    Neurons: {}", li_neurons.len());
    println!("    Centroid: ({:.1}, {:.1}, {:.1})", li_centroid[0], li_centroid[1], li_centroid[2]);
    println!("    Needs: {:?}", li_input);
    println!("    Offers: {:?}", li_output);
    print_neuron_types(&pool, &li_neurons);

    println!("\n  Region 2: Classification (WTA)");
    println!("    Neurons: {}", wta_neurons.len());
    println!("    Centroid: ({:.1}, {:.1}, {:.1})", wta_centroid[0], wta_centroid[1], wta_centroid[2]);
    println!("    Needs: {:?} <- LI offers this!", wta_input);
    println!("    Offers: {:?}", wta_output);
    print_neuron_types(&pool, &wta_neurons);

    println!("\n  Region 3: Rhythm (Oscillator) - Independent");
    println!("    Neurons: {}", osc_neurons.len());
    println!("    Centroid: ({:.1}, {:.1}, {:.1})", osc_centroid[0], osc_centroid[1], osc_centroid[2]);
    println!("    Needs: {:?}", osc_input);
    println!("    Offers: {:?}", osc_output);
    print_neuron_types(&pool, &osc_neurons);

    println!("\n  Total neurons: {}", pool.n_neurons);
    println!("  Total templates: {}", pool.template_count());

    // =========================================================================
    // Phase 4: Need/Offer connectivity analysis
    // =========================================================================
    println!("\nPhase 4: Need/Offer connectivity...\n");

    let needing_retinal = pool.templates_needing(SignalType::VisualRetinal);
    let offering_edge = pool.templates_offering(SignalType::VisualEdge);
    let needing_edge = pool.templates_needing(SignalType::VisualEdge);
    let offering_motor = pool.templates_offering(SignalType::MotorCommand);

    println!("  Signal flow analysis:");
    println!("    VisualRetinal -> {:?} (LI receives raw input)", needing_retinal);
    println!("    VisualEdge offered by {:?}", offering_edge);
    println!("    VisualEdge needed by {:?} (WTA receives from LI)", needing_edge);
    println!("    MotorCommand offered by {:?} (WTA outputs decision)", offering_motor);

    // Check cooperation link
    let li_offers_edge = offering_edge.contains(&li_id);
    let wta_needs_edge = needing_edge.contains(&wta_id);
    println!("\n  Cooperation link: LI({}) -> WTA({}): {}",
             li_id, wta_id,
             if li_offers_edge && wta_needs_edge { "NEED/OFFER MATCHED" } else { "BROKEN" });

    // =========================================================================
    // Phase 5: Wire inter-template connections (axon growth)
    // =========================================================================
    println!("\nPhase 5: Wiring inter-template connections (pressure routing)...\n");

    // Wire based on need/offer matching and spatial proximity
    // Only templates whose output matches another's input will connect
    // Probability based purely on distance - no explicit region knowledge
    let max_dist = 5.0; // Max distance for connection (LI-WTA is 3.0)
    let base_prob = 0.6; // Higher probability - more axon growth
    let synapses_created = pool.wire_inter_template(max_dist, base_prob, 789);

    println!("  Synapses created: {}", synapses_created);
    println!("  Total synapses in pool: {}", pool.synapse_count());

    // Check what connected
    // LI (offers VisualEdge) -> WTA (needs VisualEdge): should wire
    // Osc (offers AuditoryRhythm) -> nothing needs it: should NOT wire
    let li_to_wta_dist = {
        let dx = li_centroid[0] - wta_centroid[0];
        let dy = li_centroid[1] - wta_centroid[1];
        let dz = li_centroid[2] - wta_centroid[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    println!("  LI -> WTA distance: {:.2} (max_dist: {})", li_to_wta_dist, max_dist);

    // =========================================================================
    // Phase 6: Simulate problem solving (with connections)
    // =========================================================================
    println!("\nPhase 6: Simulating problem solving...\n");

    // Strong stimulation to LI - repeat every few ticks to see propagation
    println!("  Test 1: Stimulate LI region (repeated pulses)");

    let mut li_spikes = vec![];
    let mut wta_spikes = vec![];
    let mut osc_spikes = vec![];

    for tick in 0..20 {
        // Re-inject every 4 ticks to sustain activity
        if tick % 4 == 0 {
            pool.inject_spatial(li_centroid, 1.0, 12000);
        }

        pool.tick_simple(&[]);

        // Count spikes per region using spike_out (public)
        let li_s = count_spikes(&pool, &li_neurons);
        let wta_s = count_spikes(&pool, &wta_neurons);
        let osc_s = count_spikes(&pool, &osc_neurons);

        li_spikes.push(li_s);
        wta_spikes.push(wta_s);
        osc_spikes.push(osc_s);

        if tick < 5 || li_s > 0 || wta_s > 0 || osc_s > 0 {
            println!("    Tick {:2}: LI={}, WTA={}, Osc={}", tick, li_s, wta_s, osc_s);
        }
    }

    // Test 2: Check if WTA received signal from LI
    println!("\n  Test 2: Did LI output reach WTA?");
    let wta_membrane: Vec<i16> = wta_neurons.iter()
        .map(|&i| pool.neurons.membrane[i])
        .collect();
    let wta_max_membrane = wta_membrane.iter().cloned().max().unwrap_or(i16::MIN);
    let wta_min_membrane = wta_membrane.iter().cloned().min().unwrap_or(0);
    println!("    WTA membrane range: {} to {} (threshold: -14080)", wta_min_membrane, wta_max_membrane);

    // Test 3: Check oscillator ISOLATION (control)
    println!("\n  Test 3: Oscillator isolation (control case)");
    let osc_total: u32 = osc_spikes.iter().sum();
    let li_total: u32 = li_spikes.iter().sum();
    let wta_total: u32 = wta_spikes.iter().sum();
    println!("    Oscillator fired {} times", osc_total);
    println!("    LI->WTA pressure routing: LI={} -> WTA={}", li_total, wta_total);

    // Check if oscillator membrane was affected by LI/WTA activity
    let osc_membrane: Vec<i16> = osc_neurons.iter()
        .map(|&i| pool.neurons.membrane[i])
        .collect();
    let osc_max_membrane = osc_membrane.iter().cloned().max().unwrap_or(i16::MIN);
    let osc_resting = pool.config.resting_potential;
    let osc_drift = (osc_max_membrane - osc_resting).abs();
    println!("    Osc membrane drift from resting: {} (should be ~0 if isolated)", osc_drift);

    // =========================================================================
    // Phase 7: Performance metrics
    // =========================================================================
    println!("\nPhase 7: Performance metrics...\n");

    let total_spikes: u32 = li_spikes.iter().chain(wta_spikes.iter()).chain(osc_spikes.iter()).sum();
    let li_rate = li_spikes.iter().sum::<u32>() as f32 / 20.0;
    let wta_rate = wta_spikes.iter().sum::<u32>() as f32 / 20.0;
    let osc_rate = osc_spikes.iter().sum::<u32>() as f32 / 20.0;

    println!("  Spike rates (per tick):");
    println!("    LI:  {:.2}", li_rate);
    println!("    WTA: {:.2}", wta_rate);
    println!("    Osc: {:.2}", osc_rate);
    println!("    Total spikes: {}", total_spikes);

    // Fitness after simulation
    pool.update_template_fitness(0.3);
    println!("\n  Fitness after simulation:");
    println!("    LI:  {:.3}", pool.template_fitness(li_id).unwrap_or(0.0));
    println!("    WTA: {:.3}", pool.template_fitness(wta_id).unwrap_or(0.0));
    println!("    Osc: {:.3}", pool.template_fitness(osc_id).unwrap_or(0.0));

    // =========================================================================
    // Phase 8: Density distribution
    // =========================================================================
    println!("\nPhase 8: Spatial distribution...\n");

    let mut density_field = DensityField::new([12, 12, 12], bounds);
    let positions: Vec<[f32; 3]> = pool.all_soma_positions().to_vec();
    density_field.update_from_positions(&positions);

    println!("  Density at region centroids:");
    println!("    LI  ({:.1},{:.1},{:.1}): {:.1}",
             li_centroid[0], li_centroid[1], li_centroid[2],
             density_field.density_at(li_centroid));
    println!("    WTA ({:.1},{:.1},{:.1}): {:.1}",
             wta_centroid[0], wta_centroid[1], wta_centroid[2],
             density_field.density_at(wta_centroid));
    println!("    Osc ({:.1},{:.1},{:.1}): {:.1}",
             osc_centroid[0], osc_centroid[1], osc_centroid[2],
             density_field.density_at(osc_centroid));

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Summary ===\n");

    println!("  Structure:");
    println!("    {} neurons from {} templates", pool.n_neurons, pool.template_count());

    println!("\n  Cooperation (LI -> WTA):");
    let cooperation_works = li_offers_edge && wta_needs_edge;
    println!("    Need/Offer link: {}", if cooperation_works { "ESTABLISHED" } else { "BROKEN" });

    println!("\n  Independence (Oscillator):");
    let osc_independent = osc_total > 0;
    println!("    Self-sustaining rhythm: {}", if osc_independent { "YES" } else { "NO" });

    println!("\n  Activity:");
    println!("    LI fired: {} spikes", li_spikes.iter().sum::<u32>());
    println!("    WTA fired: {} spikes", wta_spikes.iter().sum::<u32>());
    println!("    Osc fired: {} spikes", osc_total);

    // What's missing
    println!("\n  What's NOT proven yet:");
    println!("    - Actual inter-region synaptic connections (need axon growth)");
    println!("    - WTA receiving LI spikes through synapses");
    println!("    - Oscillator maintaining stable frequency");
    println!("    - Fitness correlating with problem-solving quality");
}

/// Count how many neurons in a region spiked this tick
fn count_spikes(pool: &NeuronPool, indices: &[usize]) -> u32 {
    indices.iter()
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
    if comp > 0 { print!("Comp={} ", comp); }
    if gate > 0 { print!("Gate={} ", gate); }
    if osc > 0 { print!("Osc={} ", osc); }
    if mem_r > 0 { print!("MemR={} ", mem_r); }
    if mem_m > 0 { print!("MemM={} ", mem_m); }
    if sens > 0 { print!("Sens={} ", sens); }
    if mot > 0 { print!("Mot={} ", mot); }
    if relay > 0 { print!("Relay={} ", relay); }
    println!();
}
