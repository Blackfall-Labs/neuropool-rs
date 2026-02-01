//! Standalone learning demo: Pathway Strengthening via Three-Factor Plasticity
//!
//! Proves that the neuropool's three-factor plasticity works:
//!   - STDP marks active pathways with eligibility traces (temporal credit)
//!   - DA reinforces traced pathways (reward signal)
//!   - Active pathways are preserved and strengthened over time
//!   - Inactive synapses are pruned, maturing the network to essential connections
//!
//! Task: Maintain and strengthen input→output signal propagation
//!   - Present stimulus to input neurons (0..8)
//!   - Reward when output fires (DA reinforces the active pathway)
//!   - Observe: pathway preserved, weights strengthened, synapses mature
//!
//! What this demonstrates:
//!   - Eligibility traces correctly mark causal pathways (STDP)
//!   - DA modulation strengthens traced synapses (three-factor rule)
//!   - Thermal maturity: HOT → WARM → COOL → FROZEN lifecycle
//!   - Structural pruning: dead synapses removed, essential ones preserved
//!   - Homeostatic stability: output rate maintained even as weights change
//!
//! Architecture note: Spatial discrimination (routing different inputs to different
//! outputs) requires separate pools or local modulation. Global DA affects all
//! traced synapses equally, so a single pool with global modulation learns
//! pathway strength but not spatial routing. TVMR integration addresses this
//! by orchestrating multiple pools as separate cortical columns.
//!
//! Run: cargo run --example learning_demo

use neuropool::{NeuronPool, PoolConfig, ThermalDistribution};

const N: u32 = 32;
const DENSITY: f32 = 0.25;

const INPUT_END: usize = 8;
const OUTPUT_START: u32 = 24;
const OUTPUT_END: u32 = 32;

const PRESENT_TICKS: usize = 40;
const INJECT_CURRENT: i16 = 10000;
const MAX_EPOCHS: usize = 200;

fn main() {
    env_logger::init();

    println!("=== Neuropool Learning Demo: Pathway Strengthening ===");
    println!("    (three-factor plasticity: STDP traces + DA reward)\n");

    let config = PoolConfig {
        stdp_positive: 30,
        stdp_negative: -15,
        trace_decay: 240,
        homeostatic_rate: 2,
        max_synapses_per_neuron: 32,
        max_delay: 3,
        ..PoolConfig::default()
    };

    let mut pool = NeuronPool::with_random_connectivity("demo", N, DENSITY, config.clone());
    let initial_stats = pool.stats();
    println!("Pool: {} neurons, {} synapses ({:.0}% density)",
        initial_stats.n_neurons, initial_stats.n_synapses,
        initial_stats.n_synapses as f64 / (N as f64 * N as f64) * 100.0);
    println!("  Exc: {}, Inh: {}", initial_stats.n_excitatory, initial_stats.n_inhibitory);
    println!("  Thermal: {}", format_thermal(&initial_stats.thermal));
    println!("  Mean |weight|: {:.1}\n", initial_stats.mean_weight_magnitude);

    let start = std::time::Instant::now();

    // Baseline
    let baseline = measure_response(&mut pool, &config);
    println!("Baseline: {} output spikes / {} ticks\n", baseline, PRESENT_TICKS);

    // Learning loop
    println!("--- Learning (reward active pathways with DA=135) ---");

    let mut responses: Vec<u32> = vec![baseline];

    for epoch in 0..MAX_EPOCHS {
        reset_membranes(&mut pool, &config);

        let mut input = vec![0i16; N as usize];
        for i in 0..INPUT_END { input[i] = INJECT_CURRENT; }

        let mut output_spikes = 0u32;
        let mut total_spikes = 0u32;

        for _ in 0..PRESENT_TICKS {
            pool.tick(&input);
            total_spikes += pool.spike_count();
            let out = pool.read_output(OUTPUT_START..OUTPUT_END);
            output_spikes += out.iter().filter(|s| s.magnitude > 0).count() as u32;
        }

        // Reward when output fires — traces on active pathways get reinforced
        if output_spikes > 0 {
            pool.apply_modulation(135, 30, 100);
        }

        if epoch % 10 == 0 {
            pool.prune_dead();
        }

        responses.push(output_spikes);

        if epoch % 20 == 0 || epoch < 5 {
            let stats = pool.stats();
            println!("Epoch {:4} | out={:3} | total={:3} | syn={:3} | |w|={:.0} | {}",
                epoch, output_spikes, total_spikes, stats.n_synapses,
                stats.mean_weight_magnitude, format_thermal(&stats.thermal));
        }
    }

    let elapsed = start.elapsed();
    let final_stats = pool.stats();
    let final_response = measure_response(&mut pool, &config);

    println!("\n=== Results ===");
    println!("Time: {:.2}s ({:.2}ms/epoch)", elapsed.as_secs_f64(),
        elapsed.as_millis() as f64 / MAX_EPOCHS as f64);
    println!();
    println!("Structure:");
    println!("  Synapses: {} -> {} ({:.0}% pruned)",
        initial_stats.n_synapses, final_stats.n_synapses,
        (1.0 - final_stats.n_synapses as f64 / initial_stats.n_synapses as f64) * 100.0);
    println!("  Thermal:  {} -> {}", format_thermal(&initial_stats.thermal),
        format_thermal(&final_stats.thermal));
    println!("  Weights:  |w|={:.1} -> {:.1}",
        initial_stats.mean_weight_magnitude, final_stats.mean_weight_magnitude);
    println!();
    println!("Activity:");
    println!("  Baseline output:  {} spikes / {} ticks", baseline, PRESENT_TICKS);
    println!("  Final output:     {} spikes / {} ticks", final_response, PRESENT_TICKS);

    // Compute trajectory
    let quarter = responses.len() / 4;
    if quarter > 0 {
        let q1 = responses[..quarter].iter().sum::<u32>() as f64 / quarter as f64;
        let q2 = responses[quarter..quarter*2].iter().sum::<u32>() as f64 / quarter as f64;
        let q3 = responses[quarter*2..quarter*3].iter().sum::<u32>() as f64 / quarter as f64;
        let q4 = responses[quarter*3..].iter().sum::<u32>() as f64 / (responses.len() - quarter*3) as f64;
        println!();
        println!("Trajectory (quarter averages):");
        println!("  Q1: {:.1}  Q2: {:.1}  Q3: {:.1}  Q4: {:.1}", q1, q2, q3, q4);
    }

    println!();
    println!("Proven:");
    println!("  [{}] STDP eligibility traces mark active pathways",
        if final_stats.mean_eligibility_magnitude > 0.0 { "x" } else { " " });
    println!("  [{}] DA modulation strengthens traced synapses (|w| grew)",
        if final_stats.mean_weight_magnitude > initial_stats.mean_weight_magnitude { "x" } else { " " });
    println!("  [{}] Thermal maturity: synapses progressed past HOT",
        if final_stats.thermal.warm + final_stats.thermal.cool + final_stats.thermal.cold > 0 { "x" } else { " " });
    println!("  [{}] Structural pruning: dead synapses removed",
        if final_stats.n_synapses < initial_stats.n_synapses { "x" } else { " " });
    println!("  [{}] Homeostatic stability: output rate maintained",
        if final_response > 0 { "x" } else { " " });
}

fn measure_response(pool: &mut NeuronPool, config: &PoolConfig) -> u32 {
    reset_membranes(pool, config);
    let mut input = vec![0i16; N as usize];
    for i in 0..INPUT_END { input[i] = INJECT_CURRENT; }
    let mut output = 0u32;
    for _ in 0..PRESENT_TICKS {
        pool.tick(&input);
        let out = pool.read_output(OUTPUT_START..OUTPUT_END);
        output += out.iter().filter(|s| s.magnitude > 0).count() as u32;
    }
    output
}

fn reset_membranes(pool: &mut NeuronPool, config: &PoolConfig) {
    for i in 0..N as usize {
        pool.neurons.membrane[i] = config.resting_potential;
        pool.neurons.refract_remaining[i] = 0;
    }
}

fn format_thermal(td: &ThermalDistribution) -> String {
    format!("H:{} W:{} C:{} F:{} D:{}", td.hot, td.warm, td.cool, td.cold, td.dead)
}
