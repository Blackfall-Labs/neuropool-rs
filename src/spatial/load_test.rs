#![allow(deprecated)]
//! Load Testing & Break Tests for Spatial Neurons
//!
//! This module provides:
//! 1. **Break Tests** — Deliberately disable mechanisms to see what collapses
//! 2. **Hush-Adjacent Load Test** — MFCC-like audio processing at scale
//! 3. **Metrics Collection** — Detailed performance and behavior metrics
//!
//! ## Break Tests
//!
//! - Remove neighborhood overlap → test convergence necessity
//! - Remove 8x scaling → test coincidence detection
//! - Flatten tissue velocity → test distance penalty
//!
//! ## Hush-Adjacent Test
//!
//! Simulates audio transcription pipeline using a spatial point cloud:
//! - 26 MFCC sensory neurons (external input interface)
//! - Interneurons in the cloud (emerge into functional regions)
//! - 40 motor neurons (phoneme output interface)
//! - Multiple "utterances" processed
//! - Real timing constraints

#[cfg(test)]
mod tests {
    use crate::spatial::{
        Axon, CorrelationTracker, HubTracker, MigrationConfig, migrate_step,
        SpatialCascade, SpatialCascadeConfig, SpatialNeuron, SpatialSynapse,
        SpatialSynapseStore, TissueConfig, TissueField, detect_regions, RegionConfig,
        SpatialRuntime, SpatialRuntimeConfig, WiringConfig, wire_by_proximity,
    };
    use crate::spatial::snapshot::{SnapshotWriter, SnapshotRequest, SnapshotMetrics, snapshot_output_dir};
    use std::time::Instant;

    // ========================================================================
    // HUSH-ADJACENT NETWORK BUILDER
    // ========================================================================

    /// Configuration for the Hush-like network.
    #[allow(dead_code)]
    struct HushConfig {
        /// Number of MFCC coefficients (sensory input channels)
        mfcc_bins: usize,
        /// Number of interneurons in the point cloud
        interneuron_count: usize,
        /// Number of motor output neurons (phoneme channels)
        motor_count: usize,
        /// Use neighborhood activation (coincidence detection)
        use_neighborhood: bool,
        /// Signal scaling factor (normally 8)
        signal_scale: i32,
        /// Use tissue physics
        use_tissue: bool,
        /// Synapse magnitude
        synapse_magnitude: u8,
    }

    impl Default for HushConfig {
        fn default() -> Self {
            Self {
                mfcc_bins: 26,
                interneuron_count: 64,
                motor_count: 40,
                use_neighborhood: true,
                signal_scale: 8,
                use_tissue: true,
                synapse_magnitude: 100,
            }
        }
    }

    /// Deterministic pseudo-random float in [-1, 1] from a seed.
    /// Simple hash-based scatter — no external deps needed.
    fn scatter(seed: u64) -> f32 {
        // xorshift-inspired hash
        let mut x = seed.wrapping_mul(0x517cc1b727220a95);
        x ^= x >> 17;
        x = x.wrapping_mul(0x6c62272e07bb0142);
        x ^= x >> 11;
        // Map to [-1, 1]
        ((x & 0xFFFF) as f32 / 32768.0) - 1.0
    }

    /// Build a Hush-like spatial network for audio processing.
    ///
    /// Neurons are dispersed through continuous 3D space with random jitter:
    /// - Sensory neurons: anchored near x=0..13, y~0 (receive MFCC input)
    /// - Interneurons: scattered through x=5..18, y=0..6 (point cloud)
    /// - Motor neurons: scattered through x=18..28, y=0..3
    ///
    /// Synapses are created based on convergent wiring patterns,
    /// NOT based on layer assignments.
    fn build_hush_network(config: &HushConfig) -> (Vec<SpatialNeuron>, SpatialSynapseStore) {
        let mut neurons = Vec::new();

        // Sensory neurons: anchored along x-axis with mild y-jitter
        // These need stable x-positions (channel identity) but can scatter in y/z
        for i in 0..config.mfcc_bins {
            let x = i as f32 * 0.5;
            let jy = scatter(i as u64 * 3 + 1) * 0.3;
            let jz = scatter(i as u64 * 3 + 2) * 0.2;
            let mut n = SpatialNeuron::sensory_at([x, jy, jz], i as u16, 1);
            n.axon = Axon::myelinated([x + 10.0, 1.0 + jy, jz], 150);
            neurons.push(n);
        }
        // Interneurons: dispersed through a wide volume
        // No grid — scattered through the processing region
        for i in 0..config.interneuron_count {
            let jx = scatter(i as u64 * 7 + 100);
            let jy = scatter(i as u64 * 7 + 101);
            let jz = scatter(i as u64 * 7 + 102);
            let x = 5.0 + (i as f32 / config.interneuron_count as f32) * 13.0 + jx * 2.0;
            let y = 0.5 + jy.abs() * 5.5; // spread vertically (0.5 to 6.0)
            let z = jz * 1.0;
            let axon_jx = scatter(i as u64 * 7 + 103) * 3.0;
            let axon_jy = scatter(i as u64 * 7 + 104) * 2.0;
            let mut n = SpatialNeuron::pyramidal_at([x, y, z]);
            n.axon = Axon::myelinated([x + 8.0 + axon_jx, y + axon_jy, z], 120);
            neurons.push(n);
        }
        // Motor neurons: scattered through the output region
        for i in 0..config.motor_count {
            let jx = scatter(i as u64 * 5 + 200) * 2.5;
            let jy = scatter(i as u64 * 5 + 201) * 1.5;
            let jz = scatter(i as u64 * 5 + 202) * 0.5;
            let x = 20.0 + (i as f32 / config.motor_count as f32) * 6.0 + jx;
            let y = 0.5 + jy.abs() * 2.0;
            neurons.push(SpatialNeuron::motor_at([x, y, jz], i as u16, 1));
        }

        // Wire by spatial proximity: axon terminals connect to nearby somas
        let wiring = WiringConfig {
            max_distance: 8.0,
            max_fanout: 8,
            max_fanin: 20,
            default_magnitude: config.synapse_magnitude,
        };
        let store = wire_by_proximity(&neurons, &wiring);
        (neurons, store)
    }

    /// Simulate MFCC frame injection.
    fn inject_mfcc_frame(
        cascade: &mut SpatialCascade,
        coefficients: &[f32],
        use_neighborhood: bool,
        time_us: u64,
    ) {
        let neighbor_threshold = if use_neighborhood { 0.3 } else { f32::MAX };
        cascade.inject_sensory_scaled(coefficients, 2000.0, 1500.0, neighbor_threshold, 0.1, time_us);
    }

    /// Generate a synthetic MFCC frame (simulates audio feature extraction).
    fn generate_mfcc_frame(frame_idx: usize, phoneme: usize) -> Vec<f32> {
        let mut coeffs = vec![0.0f32; 26];

        // Each phoneme has a characteristic MFCC pattern
        // This is simplified - real MFCC would come from audio
        let base = (phoneme * 3) % 26;
        for i in 0..5 {
            let idx = (base + i) % 26;
            // Add some temporal variation
            let phase = (frame_idx as f32 * 0.1 + i as f32 * 0.5).sin();
            coeffs[idx] = 0.5 + 0.4 * phase;
        }

        // Add harmonic structure
        for i in 0..26 {
            coeffs[i] += 0.1 * ((i as f32 * 0.2 + frame_idx as f32 * 0.05).sin());
        }

        coeffs
    }

    /// Classify neurons by role (sensory/interneuron/motor) and compute per-role utilization.
    fn role_utilization(cascade: &SpatialCascade) -> (f32, f32, f32) {
        let mut sensory_fired = 0u32;
        let mut sensory_total = 0u32;
        let mut inter_fired = 0u32;
        let mut inter_total = 0u32;
        let mut motor_fired = 0u32;
        let mut motor_total = 0u32;

        for n in &cascade.neurons {
            let fired = n.last_spike_us > 0;
            if n.nuclei.is_sensory() {
                sensory_total += 1;
                if fired { sensory_fired += 1; }
            } else if n.nuclei.is_motor() {
                motor_total += 1;
                if fired { motor_fired += 1; }
            } else {
                inter_total += 1;
                if fired { inter_fired += 1; }
            }
        }

        let s = if sensory_total > 0 { sensory_fired as f32 / sensory_total as f32 } else { 0.0 };
        let i = if inter_total > 0 { inter_fired as f32 / inter_total as f32 } else { 0.0 };
        let m = if motor_total > 0 { motor_fired as f32 / motor_total as f32 } else { 0.0 };
        (s, i, m)
    }

    // ========================================================================
    // BREAK TESTS — What happens when we disable each mechanism?
    // ========================================================================

    #[test]
    fn break_test_no_neighborhood() {
        println!("\n=== BREAK TEST: No Neighborhood Activation ===");
        println!("Testing: What happens without coincidence detection?\n");

        let config = HushConfig {
            use_neighborhood: false, // DISABLED
            ..Default::default()
        };

        let (neurons, synapses) = build_hush_network(&config);
        let neuron_count = neurons.len();

        let cascade_config = SpatialCascadeConfig::default();
        let mut cascade = SpatialCascade::with_network(neurons, synapses, cascade_config);
        let mut correlations = CorrelationTracker::new(neuron_count, 50, 5_000);

        let start = Instant::now();

        // Run 100 frames (simulating ~1 second of audio at 100Hz)
        let mut time = 0u64;
        for frame in 0..100 {
            let phoneme = (frame / 10) % 10; // Change phoneme every 10 frames
            let mfcc = generate_mfcc_frame(frame, phoneme);

            inject_mfcc_frame(&mut cascade, &mfcc, false, time);
            cascade.run_until(time + 10_000);

            // Record spikes for correlation
            for (idx, neuron) in cascade.neurons.iter().enumerate() {
                if neuron.last_spike_us > time.saturating_sub(10_000)
                    && neuron.last_spike_us <= time + 10_000
                {
                    correlations.record_spike(idx, neuron.last_spike_us);
                }
            }

            time += 10_000;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let (sensory_util, inter_util, motor_util) = role_utilization(&cascade);

        println!("Results WITHOUT neighborhood activation:");
        println!("  Spikes:            {}", cascade.total_spikes());
        println!("  Sensory util:      {:.1}%", sensory_util * 100.0);
        println!("  Interneuron util:  {:.1}%", inter_util * 100.0);
        println!("  Motor util:        {:.1}%", motor_util * 100.0);
        println!("  Time:              {:.2}ms", elapsed);

        // EXPECTED: Interneurons should have LOW utilization
        // Without neighborhood, single inputs can't push interneurons over threshold
        println!("\n  DIAGNOSIS: Interneuron utilization should be LOW without neighborhood");
        println!("  (Single synapse magnitude 100 -> 800 current, need 1500 for threshold)");

        // Don't assert - this is diagnostic
        if inter_util < 0.3 {
            println!("  Confirmed: Coincidence detection IS load-bearing");
        } else {
            println!("  ? Unexpected: Interneurons still firing without coincidence");
        }
    }

    #[test]
    fn break_test_no_scaling() {
        println!("\n=== BREAK TEST: No Signal Scaling ===");
        println!("Testing: What happens without 8x synapse scaling?\n");

        // We can't easily change the scaling factor, but we can use very low magnitude
        // Magnitude 12 with 8x = 96 current (way below 1500 threshold)
        let config = HushConfig {
            synapse_magnitude: 12, // 12 * 8 = 96, way below threshold
            use_neighborhood: true, // Keep neighborhood on
            ..Default::default()
        };

        let (neurons, synapses) = build_hush_network(&config);

        let cascade_config = SpatialCascadeConfig::default();
        let mut cascade = SpatialCascade::with_network(neurons, synapses, cascade_config);

        let start = Instant::now();

        let mut time = 0u64;
        for frame in 0..100 {
            let phoneme = (frame / 10) % 10;
            let mfcc = generate_mfcc_frame(frame, phoneme);
            inject_mfcc_frame(&mut cascade, &mfcc, true, time);
            cascade.run_until(time + 10_000);
            time += 10_000;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let (_sensory_util, inter_util, motor_util) = role_utilization(&cascade);

        println!("Results WITH weak synapse magnitude (simulating no scaling):");
        println!("  Spikes:            {}", cascade.total_spikes());
        println!("  Interneuron util:  {:.1}%", inter_util * 100.0);
        println!("  Motor util:        {:.1}%", motor_util * 100.0);
        println!("  Time:              {:.2}ms", elapsed);

        println!("\n  DIAGNOSIS: Even with neighborhood, weak synapses can't fire interneurons");

        if inter_util < 0.1 {
            println!("  Confirmed: Signal scaling IS load-bearing");
        } else {
            println!("  ? Unexpected: Interneurons firing with weak synapses");
        }
    }

    #[test]
    fn break_test_all_mechanisms_working() {
        println!("\n=== CONTROL TEST: All Mechanisms Working ===");
        println!("Testing: Baseline with everything enabled\n");

        let config = HushConfig::default(); // Everything on

        let (neurons, synapses) = build_hush_network(&config);
        let neuron_count = neurons.len();

        let cascade_config = SpatialCascadeConfig::default();
        let mut cascade = SpatialCascade::with_network(neurons, synapses, cascade_config);
        let mut correlations = CorrelationTracker::new(neuron_count, 50, 5_000);

        let start = Instant::now();

        let mut time = 0u64;
        for frame in 0..100 {
            let phoneme = (frame / 10) % 10;
            let mfcc = generate_mfcc_frame(frame, phoneme);
            inject_mfcc_frame(&mut cascade, &mfcc, true, time);
            cascade.run_until(time + 10_000);

            for (idx, neuron) in cascade.neurons.iter().enumerate() {
                if neuron.last_spike_us > time.saturating_sub(10_000) {
                    correlations.record_spike(idx, neuron.last_spike_us);
                }
            }

            time += 10_000;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let (sensory_util, inter_util, motor_util) = role_utilization(&cascade);

        // Detect emergent regions
        let regions = detect_regions(
            &cascade.neurons,
            Some(&correlations),
            time,
            &RegionConfig::default(),
        );

        println!("Results WITH all mechanisms:");
        println!("  Spikes:            {}", cascade.total_spikes());
        println!("  Sensory util:      {:.1}%", sensory_util * 100.0);
        println!("  Interneuron util:  {:.1}%", inter_util * 100.0);
        println!("  Motor util:        {:.1}%", motor_util * 100.0);
        println!("  Emergent regions:  {}", regions.len());
        println!("  Time:              {:.2}ms", elapsed);

        // With everything working, interneurons should be selectively active
        // 30%+ indicates they're participating, not just noise
        assert!(inter_util > 0.3,
            "Interneurons should be >30% utilized with all mechanisms, got {:.1}%",
            inter_util * 100.0);
        assert!(motor_util > 0.3,
            "Motor neurons should be >30% utilized with all mechanisms, got {:.1}%",
            motor_util * 100.0);

        println!("\n  BASELINE CONFIRMED: System works with all mechanisms");
    }

    // ========================================================================
    // HUSH-ADJACENT LOAD TEST
    // ========================================================================

    #[test]
    fn hush_load_test_1k_frames() {
        println!("\n{}", "=".repeat(70));
        println!("  HUSH-ADJACENT LOAD TEST: 1000 Frames (~10 seconds of audio)");
        println!("{}", "=".repeat(70));

        let config = HushConfig::default();

        let setup_start = Instant::now();
        let (neurons, synapses) = build_hush_network(&config);
        let neuron_count = neurons.len();
        let synapse_count = synapses.len();
        let setup_time = setup_start.elapsed().as_secs_f64() * 1000.0;

        let cascade_config = SpatialCascadeConfig::default();
        let mut cascade = SpatialCascade::with_network(neurons, synapses, cascade_config);
        let mut correlations = CorrelationTracker::new(neuron_count, 100, 10_000);
        let mut hub_tracker = HubTracker::new(neuron_count);

        // Track hub connections from actual wiring
        for syn in cascade.synapses.iter() {
            hub_tracker.record_connection(syn.target);
        }

        // Snapshot initial positions for migration displacement measurement
        let initial_positions: Vec<[f32; 3]> = cascade.neurons.iter()
            .map(|n| n.soma.position)
            .collect();

        let migration_config = MigrationConfig {
            migration_rate: 0.05,
            correlation_threshold: 0.2,
            attraction_strength: 0.5,
            repulsion_strength: 0.3,
            min_distance: 0.3,
            max_step: 0.2,
            axon_elasticity: 0.8,
            exclusion_radius: 0.3,
            exclusion_strength: 2.0,
            origin_spring: 0.05,
        };
        let mut migration_steps = 0usize;

        // Tissue substrate — persistent resistance/conductivity field
        let mut tissue = TissueField::with_config(TissueConfig::default());
        tissue.rebuild(&cascade.neurons);

        let run_start = Instant::now();

        let mut time = 0u64;
        let frame_interval = 10_000u64; // 10ms per frame (100 Hz)

        // Process 1000 frames (10 seconds of simulated audio)
        for frame in 0..1000 {
            // Cycle through 10 different "phonemes"
            let phoneme = (frame / 25) % 10;
            let mfcc = generate_mfcc_frame(frame, phoneme);

            inject_mfcc_frame(&mut cascade, &mfcc, true, time);
            cascade.run_until_with_tissue(time + frame_interval, &tissue);

            // Record spikes for correlation
            for (idx, neuron) in cascade.neurons.iter().enumerate() {
                if neuron.last_spike_us > time.saturating_sub(frame_interval) {
                    correlations.record_spike(idx, neuron.last_spike_us);
                }
            }

            // Migration: let neurons physically move toward correlated partners
            // Run every 100 frames (~1 second of simulated time)
            if frame % 100 == 99 {
                // Tissue plasticity: active neurons soften local tissue
                let active_mask: Vec<bool> = cascade.neurons.iter()
                    .map(|n| n.last_spike_us > time.saturating_sub(1_000_000))
                    .collect();
                tissue.update_plasticity(&active_mask);
                tissue.rebuild(&cascade.neurons);

                migrate_step(
                    &mut cascade.neurons,
                    &correlations,
                    &migration_config,
                    time,
                    Some(&initial_positions),
                    Some(&tissue),
                );
                migration_steps += 1;
            }

            time += frame_interval;
        }

        let run_time = run_start.elapsed().as_secs_f64() * 1000.0;
        let total_time = setup_start.elapsed().as_secs_f64() * 1000.0;

        // Calculate metrics
        let simulated_duration_s = time as f64 / 1_000_000.0;
        let spikes_per_second = cascade.total_spikes() as f64 / simulated_duration_s;
        let events_per_second = cascade.total_events() as f64 / simulated_duration_s;

        let (sensory_util, inter_util, motor_util) = role_utilization(&cascade);

        // Compute inter-region correlations using emergent regions
        let region_config = RegionConfig::default();
        let regions = detect_regions(&cascade.neurons, Some(&correlations), time, &region_config);

        // Sample cross-region correlations
        let mut cross_region_corrs: Vec<(u32, u32, f32)> = Vec::new();
        for i in 0..regions.len() {
            for j in (i+1)..regions.len() {
                let sample_a: Vec<usize> = regions[i].neurons.iter()
                    .take(20).map(|&n| n as usize).collect();
                let sample_b: Vec<usize> = regions[j].neurons.iter()
                    .take(20).map(|&n| n as usize).collect();
                let corr = mean_correlation(&correlations, &sample_a, &sample_b, time);
                cross_region_corrs.push((regions[i].id, regions[j].id, corr));
            }
        }

        // Migration displacement analysis
        let mut total_displacement = 0.0f32;
        let mut max_displacement = 0.0f32;
        for (i, neuron) in cascade.neurons.iter().enumerate() {
            let dx = neuron.soma.position[0] - initial_positions[i][0];
            let dy = neuron.soma.position[1] - initial_positions[i][1];
            let dz = neuron.soma.position[2] - initial_positions[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            total_displacement += dist;
            if dist > max_displacement {
                max_displacement = dist;
            }
        }
        let mean_displacement = total_displacement / neuron_count as f32;

        // Hub analysis
        let mut max_fan_in = 0u16;
        let mut hub_count = 0;
        for i in 0..neuron_count {
            let fan_in = hub_tracker.fan_in(i as u32);
            if fan_in > max_fan_in {
                max_fan_in = fan_in;
            }
            if fan_in > 20 {
                hub_count += 1;
            }
        }

        // Tissue field
        let mut tissue = TissueField::with_config(TissueConfig::default());
        tissue.rebuild(&cascade.neurons);

        // Estimate memory
        let neuron_size = std::mem::size_of::<SpatialNeuron>();
        let synapse_size = std::mem::size_of::<SpatialSynapse>();
        let approx_memory = neuron_count * neuron_size + synapse_count * synapse_size;

        // Print comprehensive metrics
        println!("\n  TIMING");
        println!("  -------");
        println!("    Setup:          {:>8.2} ms", setup_time);
        println!("    Run:            {:>8.2} ms", run_time);
        println!("    Total:          {:>8.2} ms", total_time);
        println!("    Simulated:      {:>8.2} s", simulated_duration_s);
        println!("    Real-time ratio:{:>8.1}x", simulated_duration_s * 1000.0 / run_time);

        println!("\n  NETWORK");
        println!("  -------");
        println!("    Neurons:        {:>8}", neuron_count);
        println!("    Synapses:       {:>8}", synapse_count);
        println!("    Memory:         {:>8} KB", approx_memory / 1024);

        println!("\n  ACTIVITY");
        println!("  --------");
        println!("    Total spikes:   {:>8}", cascade.total_spikes());
        println!("    Total events:   {:>8}", cascade.total_events());
        println!("    Spikes/sec:     {:>8.0}", spikes_per_second);
        println!("    Events/sec:     {:>8.0}", events_per_second);
        println!("    Coincidence:    {:>8}", cascade.coincidence_events);
        println!("    Tissue atten:   {:>8}", cascade.tissue_attenuated);

        println!("\n  SPATIAL UTILIZATION (by neuron role)");
        println!("  ------------------------------------");
        println!("    Sensory:        {:>7.1}%", sensory_util * 100.0);
        println!("    Interneurons:   {:>7.1}%", inter_util * 100.0);
        println!("    Motor:          {:>7.1}%", motor_util * 100.0);

        println!("\n  EMERGENT REGIONS");
        println!("  -----------------");
        println!("    Detected:       {:>8}", regions.len());
        for region in &regions {
            println!("    Region {}: {} neurons at ({:.1}, {:.1}, {:.1}) [sensory={}, motor={}, osc={}]",
                region.id, region.neurons.len(),
                region.centroid[0], region.centroid[1], region.centroid[2],
                region.signature.has_sensory, region.signature.has_motor,
                region.signature.has_oscillators);
        }

        println!("\n  CROSS-REGION CORRELATIONS");
        println!("  --------------------------");
        for (a, b, corr) in &cross_region_corrs {
            println!("    Region {} <-> Region {}: {:.4}", a, b, corr);
        }

        println!("\n  NEURON MIGRATION (Activity-Dependent)");
        println!("  --------------------------------------");
        println!("    Migration steps:{:>8}", migration_steps);
        println!("    Mean displacement:{:>6.3} units", mean_displacement);
        println!("    Max displacement: {:>6.3} units", max_displacement);

        println!("\n  HUB ANALYSIS (Anti-Hebbian Target)");
        println!("  -----------------------------------");
        println!("    Hub count (>20):{:>8}", hub_count);
        println!("    Max fan-in:     {:>8}", max_fan_in);

        println!("\n  VERDICT");
        println!("  -------");

        let realtime_capable = run_time < simulated_duration_s * 1000.0;
        let good_utilization = inter_util > 0.3 && motor_util > 0.3;
        let structure_emerged = regions.len() >= 2;
        let neurons_migrated = mean_displacement > 0.001;

        if realtime_capable && good_utilization && structure_emerged {
            println!("    REAL-TIME CAPABLE: {}x faster than real-time",
                (simulated_duration_s * 1000.0 / run_time) as i32);
            println!("    GOOD UTILIZATION: Interneurons {:.0}%, Motor {:.0}%",
                inter_util * 100.0, motor_util * 100.0);
            println!("    STRUCTURE EMERGED: {} regions detected", regions.len());
            if neurons_migrated {
                println!("    NEURONS ADAPTED: mean {:.3} units migration", mean_displacement);
            }
            println!("\n    ==> WORTH SCALING UP");
        } else {
            if !realtime_capable {
                println!("    NOT REAL-TIME: Too slow");
            }
            if !good_utilization {
                println!("    LOW UTILIZATION: Check network topology");
            }
            if !structure_emerged {
                println!("    NO STRUCTURE: Check spatial clustering");
            }
            if !neurons_migrated {
                println!("    NO MIGRATION: Neurons didn't adapt spatially");
            }
            println!("\n    ==> NEEDS WORK BEFORE SCALING");
        }

        println!("\n{}\n", "=".repeat(70));

        // Assertions for CI
        assert!(realtime_capable, "Must be real-time capable");
        assert!(inter_util > 0.3,
            "Interneurons must have >30% utilization, got {:.1}%",
            inter_util * 100.0);
    }

    /// Calculate mean correlation between two groups.
    fn mean_correlation(
        correlations: &CorrelationTracker,
        group_a: &[usize],
        group_b: &[usize],
        current_time: u64,
    ) -> f32 {
        if group_a.is_empty() || group_b.is_empty() {
            return 0.0;
        }

        let mut total = 0.0;
        let mut count = 0;

        // Sample correlation (not all pairs for efficiency)
        let sample_size = 20.min(group_a.len());
        for i in 0..sample_size {
            let a = group_a[i * group_a.len() / sample_size];
            for j in 0..sample_size {
                let b = group_b[j * group_b.len() / sample_size];
                if a != b {
                    total += correlations.correlation(a, b, current_time);
                    count += 1;
                }
            }
        }

        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }

    /// Render the spatial neuron field as compact ASCII art.
    /// Sensory at x=0-13, interneurons at x=10-14 y=1-5, motor at x=20-25.
    fn render_ascii_field(neurons: &[SpatialNeuron], label: &str) {
        let w = 52usize;
        let h = 12usize;
        let mut grid = vec![vec![(' ', 0u8); w]; h]; // (char, priority)

        for n in neurons {
            let cx = (n.soma.position[0] * 2.0).round() as usize;
            let cy = (n.soma.position[1] * 2.0).round() as usize;
            if cx >= w || cy >= h { continue; }
            let fired = n.last_spike_us > 0;
            let (ch, pri) = match (n.nuclei.is_sensory(), n.nuclei.is_motor(), fired) {
                (true, _, true)  => ('S', 2),
                (true, _, false) => ('s', 1),
                (_, true, true)  => ('M', 2),
                (_, true, false) => ('m', 1),
                (_, _, true)     => ('#', 2),
                (_, _, false)    => ('.', 1),
            };
            if pri > grid[cy][cx].1 { grid[cy][cx] = (ch, pri); }
        }

        println!("\n    {} [S=sensory ./#=inter m/M=motor  CAPS=fired]", label);
        for y in (0..h).rev() {
            let row: String = grid[y].iter().map(|&(c, _)| c).collect();
            if y % 2 == 0 {
                println!("    {:>2}|{}", y / 2, row.trim_end());
            } else {
                println!("      |{}", row.trim_end());
            }
        }
        println!("    --+{}", "-".repeat(w));
        println!("      0         5        10        15        20        25");
    }

    // ========================================================================
    // REAL AUDIO — LibriSpeech via Hush's dev-clean.spool
    // ========================================================================

    /// Minimal spool reader — avoids pulling in dataspool-rs (and its rusqlite dep).
    /// Format: [SP01][version:1][card_count:u32][index_offset:u64][cards...][index]
    mod spool_reader {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};
        use std::path::Path;

        pub struct MiniSpoolReader {
            file: File,
            entries: Vec<(u64, u32)>, // (offset, length)
        }

        impl MiniSpoolReader {
            pub fn open(path: &Path) -> std::io::Result<Self> {
                let mut file = File::open(path)?;

                let mut magic = [0u8; 4];
                file.read_exact(&mut magic)?;
                if &magic != b"SP01" {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData, "not a spool file"));
                }

                let mut ver = [0u8; 1];
                file.read_exact(&mut ver)?;

                let mut buf4 = [0u8; 4];
                file.read_exact(&mut buf4)?;
                let card_count = u32::from_le_bytes(buf4) as usize;

                let mut buf8 = [0u8; 8];
                file.read_exact(&mut buf8)?;
                let index_offset = u64::from_le_bytes(buf8);

                // Read index
                file.seek(SeekFrom::Start(index_offset))?;
                let mut entries = Vec::with_capacity(card_count);
                for _ in 0..card_count {
                    file.read_exact(&mut buf8)?;
                    let offset = u64::from_le_bytes(buf8);
                    file.read_exact(&mut buf4)?;
                    let length = u32::from_le_bytes(buf4);
                    entries.push((offset, length));
                }

                Ok(Self { file, entries })
            }

            pub fn card_count(&self) -> usize { self.entries.len() }

            pub fn read_card(&mut self, index: usize) -> std::io::Result<Vec<u8>> {
                let (offset, length) = self.entries[index];
                self.file.seek(SeekFrom::Start(offset))?;
                let mut buf = vec![0u8; length as usize];
                self.file.read_exact(&mut buf)?;
                Ok(buf)
            }
        }

        /// Parse a Hush audio spool card.
        /// Format: [n_frames:u32 LE][frame0: N_MFCC×i8]...[transcript_len:u16 LE][transcript]
        pub struct SpoolSample {
            pub n_frames: usize,
            pub transcript: String,
            card_data: Vec<u8>,
        }

        impl SpoolSample {
            const N_MFCC: usize = 26;

            pub fn from_card(data: Vec<u8>) -> Option<Self> {
                if data.len() < 6 { return None; }
                let n_frames = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
                let mfcc_end = 4 + n_frames * Self::N_MFCC;
                if data.len() < mfcc_end + 2 { return None; }

                let tlen = u16::from_le_bytes(
                    data[mfcc_end..mfcc_end + 2].try_into().ok()?) as usize;
                let tstart = mfcc_end + 2;
                let transcript = if tstart + tlen <= data.len() {
                    String::from_utf8_lossy(&data[tstart..tstart + tlen]).to_string()
                } else {
                    String::new()
                };

                Some(Self { n_frames, transcript, card_data: data })
            }

            /// Get frame as f32 slice normalized to [-1.0, 1.0].
            pub fn frame_f32(&self, idx: usize, buf: &mut [f32; 26]) {
                let start = 4 + idx * Self::N_MFCC;
                for i in 0..Self::N_MFCC {
                    buf[i] = (self.card_data[start + i] as i8) as f32 / 127.0;
                }
            }
        }
    }

    #[test]
    fn hush_load_test_real_audio() {
        use spool_reader::{MiniSpoolReader, SpoolSample};

        let spool_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("hush").join("data").join("dev-clean.spool");

        if !spool_path.exists() {
            println!("SKIPPED: dev-clean.spool not found at {}", spool_path.display());
            println!("  (Run hush-prepare on LibriSpeech dev-clean to generate it)");
            return;
        }

        println!("\n{}", "=".repeat(70));
        println!("  REAL AUDIO LOAD TEST: LibriSpeech dev-clean via Hush spool");
        println!("{}", "=".repeat(70));

        // Open spool
        let spool_start = Instant::now();
        let mut spool = MiniSpoolReader::open(&spool_path)
            .expect("failed to open spool");
        let total_utterances = spool.card_count();
        let spool_open_ms = spool_start.elapsed().as_secs_f64() * 1000.0;

        // Use first 1000 utterances — at 400x+ real-time we have massive headroom
        let n_utterances = 1000.min(total_utterances);

        println!("\n  SPOOL");
        println!("  ------");
        println!("    File:            {}", spool_path.display());
        println!("    Total utterances:{:>8}", total_utterances);
        println!("    Testing:         {:>8}", n_utterances);
        println!("    Open time:       {:>8.2} ms", spool_open_ms);

        // Build network
        let config = HushConfig::default();
        let setup_start = Instant::now();
        let (neurons, synapses) = build_hush_network(&config);
        let neuron_count = neurons.len();
        let synapse_count = synapses.len();
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

        let runtime_config = SpatialRuntimeConfig {
            cascade: SpatialCascadeConfig {
                max_events_per_call: 50_000,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut runtime = SpatialRuntime::new(neurons, synapses, runtime_config);

        // Snapshot initial mean magnitude for comparison
        let initial_mean_mag: f32 = runtime.cascade.synapses.iter()
            .map(|s| s.signal.magnitude as f32).sum::<f32>() / synapse_count as f32;

        // === PNG SNAPSHOT WRITER (10 snapshots: before + 8 during + after) ===
        let snapshot_writer = SnapshotWriter::spawn(snapshot_output_dir());
        let mut snapshot_seq = 0u32;
        let total_snapshots = 10usize;
        // Utterance indices at which to take snapshots (evenly spaced)
        let snapshot_at: Vec<usize> = (1..total_snapshots - 1)
            .map(|i| (i * n_utterances) / (total_snapshots - 1))
            .collect();

        // === ASCII + PNG SNAPSHOT: BEFORE ===
        render_ascii_field(&runtime.cascade.neurons, "BEFORE — Initial Positions (all silent)");
        snapshot_writer.queue(SnapshotRequest {
            label: "BEFORE".into(),
            seq: snapshot_seq,
            neurons: runtime.cascade.neurons.clone(),
            metrics: SnapshotMetrics {
                sim_time_us: 0,
                wall_clock_ms: setup_start.elapsed().as_secs_f64() * 1000.0,
                neuron_count,
                synapse_count,
                total_spikes: 0,
                total_events: 0,
                sensory_util: 0.0,
                inter_util: 0.0,
                motor_util: 0.0,
                learning_cycles: 0,
                total_strengthened: 0,
                total_weakened: 0,
                total_dormant: 0,
                mean_displacement: 0.0,
                region_count: 0,
            },
        });
        snapshot_seq += 1;

        // Process real utterances via SpatialRuntime
        let run_start = Instant::now();
        let frame_interval = 10_000u64; // 10ms per MFCC frame
        let mut total_frames = 0usize;
        let mut total_audio_seconds = 0.0f64;
        let mut mfcc_buf = [0.0f32; 26];

        for utt_idx in 0..n_utterances {
            let card = spool.read_card(utt_idx).expect("failed to read card");
            let sample = match SpoolSample::from_card(card) {
                Some(s) => s,
                None => { println!("    Skipping malformed card {}", utt_idx); continue; }
            };

            let utt_duration_s = sample.n_frames as f64 * 0.01; // 10ms per frame
            total_audio_seconds += utt_duration_s;

            // Print first 3, then every 100th, to keep output manageable
            if utt_idx < 3 || utt_idx % 100 == 0 {
                println!("    Utterance {:>4}: {:>5} frames ({:.1}s) \"{}\"",
                    utt_idx, sample.n_frames, utt_duration_s,
                    if sample.transcript.len() > 60 {
                        format!("{}...", &sample.transcript[..57])
                    } else {
                        sample.transcript.clone()
                    });
            } else if utt_idx == 3 {
                println!("    ... (printing every 100th utterance)");
            }

            for frame_idx in 0..sample.n_frames {
                sample.frame_f32(frame_idx, &mut mfcc_buf);
                runtime.inject_sensory_scaled(&mfcc_buf, 2000.0, 1500.0, 0.3, 0.1);
                runtime.step(frame_interval);
                total_frames += 1;
            }

            // === PERIODIC PNG SNAPSHOTS (8 evenly spaced during run) ===
            if snapshot_at.contains(&(utt_idx + 1)) {
                let snap_label = format!("UTT_{}", utt_idx + 1);
                let (su, iu, mu) = role_utilization(&runtime.cascade);
                let mut snap_disp = 0.0f32;
                for (i, neuron) in runtime.cascade.neurons.iter().enumerate() {
                    let dx = neuron.soma.position[0] - runtime.initial_positions()[i][0];
                    let dy = neuron.soma.position[1] - runtime.initial_positions()[i][1];
                    let dz = neuron.soma.position[2] - runtime.initial_positions()[i][2];
                    snap_disp += (dx * dx + dy * dy + dz * dz).sqrt();
                }
                snap_disp /= neuron_count as f32;
                snapshot_writer.queue(SnapshotRequest {
                    label: snap_label,
                    seq: snapshot_seq,
                    neurons: runtime.cascade.neurons.clone(),
                    metrics: SnapshotMetrics {
                        sim_time_us: runtime.time_us(),
                        wall_clock_ms: run_start.elapsed().as_secs_f64() * 1000.0,
                        neuron_count: runtime.cascade.neurons.len(),
                        synapse_count: runtime.cascade.synapses.len(),
                        total_spikes: runtime.cascade.total_spikes(),
                        total_events: runtime.cascade.total_events(),
                        sensory_util: su,
                        inter_util: iu,
                        motor_util: mu,
                        learning_cycles: runtime.learning.cycles,
                        total_strengthened: runtime.learning.strengthened,
                        total_weakened: runtime.learning.weakened,
                        total_dormant: runtime.learning.dormant,
                        mean_displacement: snap_disp,
                        region_count: 0,
                    },
                });
                snapshot_seq += 1;
            }
        }

        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = setup_start.elapsed().as_secs_f64() * 1000.0;

        // Aliases for diagnostic access (avoids changing every reference below)
        let time = runtime.time_us();
        let correlations = runtime.correlations();
        let initial_positions = runtime.initial_positions();
        let cascade = &runtime.cascade;
        let tissue = runtime.tissue();
        let learning_cycles = runtime.learning.cycles;
        let total_strengthened = runtime.learning.strengthened;
        let total_weakened = runtime.learning.weakened;
        let total_dormant = runtime.learning.dormant;
        let total_awakened = runtime.learning.awakened;
        let total_flipped = runtime.learning.flipped;
        let migration_steps = runtime.structural.migration_steps;
        let tissue_updates = runtime.structural.tissue_updates;
        let total_pruning_cycles = runtime.structural.pruning_cycles;
        let total_hard_pruned = runtime.structural.hard_pruned;

        // === ASCII SNAPSHOT: AFTER ===
        let after_label = format!("AFTER — {} utterances, {} learning cycles",
            n_utterances, learning_cycles);
        render_ascii_field(&cascade.neurons, &after_label);

        // Metrics
        let simulated_s = time as f64 / 1_000_000.0;
        let (sensory_util, inter_util, motor_util) = role_utilization(cascade);

        // Regions
        let regions = detect_regions(
            &cascade.neurons, Some(correlations), time, &RegionConfig::default());

        // Cross-region correlations
        let mut cross_region_corrs: Vec<(u32, u32, f32)> = Vec::new();
        for i in 0..regions.len() {
            for j in (i+1)..regions.len() {
                let sa: Vec<usize> = regions[i].neurons.iter().take(20).map(|&n| n as usize).collect();
                let sb: Vec<usize> = regions[j].neurons.iter().take(20).map(|&n| n as usize).collect();
                let corr = mean_correlation(correlations, &sa, &sb, time);
                cross_region_corrs.push((regions[i].id, regions[j].id, corr));
            }
        }

        // Migration displacement
        let mut total_displacement = 0.0f32;
        let mut max_displacement = 0.0f32;
        for (i, neuron) in cascade.neurons.iter().enumerate() {
            let dx = neuron.soma.position[0] - initial_positions[i][0];
            let dy = neuron.soma.position[1] - initial_positions[i][1];
            let dz = neuron.soma.position[2] - initial_positions[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            total_displacement += dist;
            if dist > max_displacement { max_displacement = dist; }
        }
        let mean_displacement = total_displacement / neuron_count as f32;

        // === PNG SNAPSHOT: AFTER ===
        snapshot_writer.queue(SnapshotRequest {
            label: "AFTER".into(),
            seq: snapshot_seq,
            neurons: cascade.neurons.clone(),
            metrics: SnapshotMetrics {
                sim_time_us: time,
                wall_clock_ms: run_start.elapsed().as_secs_f64() * 1000.0,
                neuron_count: cascade.neurons.len(),
                synapse_count: cascade.synapses.len(),
                total_spikes: cascade.total_spikes(),
                total_events: cascade.total_events(),
                sensory_util,
                inter_util,
                motor_util,
                learning_cycles,
                total_strengthened,
                total_weakened,
                total_dormant,
                mean_displacement,
                region_count: regions.len(),
            },
        });
        snapshot_writer.finish(); // Wait for all PNGs to be written

        // Memory
        let neuron_size = std::mem::size_of::<SpatialNeuron>();
        let synapse_size = std::mem::size_of::<SpatialSynapse>();
        let approx_memory = neuron_count * neuron_size + synapse_count * synapse_size;

        // Print
        println!("\n  TIMING");
        println!("  -------");
        println!("    Setup:           {:>8.2} ms", setup_ms);
        println!("    Run:             {:>8.2} ms", run_ms);
        println!("    Total:           {:>8.2} ms", total_ms);
        println!("    Audio processed: {:>8.1} s  ({} frames from {} utterances)",
            total_audio_seconds, total_frames, n_utterances);
        println!("    Simulated:       {:>8.2} s", simulated_s);
        println!("    Real-time ratio: {:>8.1}x", simulated_s * 1000.0 / run_ms);

        println!("\n  NETWORK");
        println!("  -------");
        println!("    Neurons:         {:>8}", neuron_count);
        println!("    Synapses:        {:>8}", synapse_count);
        println!("    Memory:          {:>8} KB", approx_memory / 1024);

        println!("\n  ACTIVITY");
        println!("  --------");
        println!("    Total spikes:    {:>8}", cascade.total_spikes());
        println!("    Total events:    {:>8}", cascade.total_events());
        println!("    Spikes/sec:      {:>8.0}", cascade.total_spikes() as f64 / simulated_s);
        println!("    Events/sec:      {:>8.0}", cascade.total_events() as f64 / simulated_s);
        println!("    Coincidence:     {:>8}", cascade.coincidence_events);
        println!("    Tissue atten:    {:>8}", cascade.tissue_attenuated);

        println!("\n  SPATIAL UTILIZATION (by neuron role)");
        println!("  ------------------------------------");
        println!("    Sensory:         {:>7.1}%", sensory_util * 100.0);
        println!("    Interneurons:    {:>7.1}%", inter_util * 100.0);
        println!("    Motor:           {:>7.1}%", motor_util * 100.0);

        println!("\n  EMERGENT REGIONS");
        println!("  -----------------");
        println!("    Detected:        {:>8}", regions.len());
        for region in &regions {
            println!("    Region {}: {} neurons at ({:.1}, {:.1}, {:.1}) [sensory={}, motor={}, osc={}]",
                region.id, region.neurons.len(),
                region.centroid[0], region.centroid[1], region.centroid[2],
                region.signature.has_sensory, region.signature.has_motor,
                region.signature.has_oscillators);
        }

        println!("\n  CROSS-REGION CORRELATIONS");
        println!("  --------------------------");
        for (a, b, corr) in &cross_region_corrs {
            println!("    Region {} <-> Region {}: {:.4}", a, b, corr);
        }

        println!("\n  NEURON MIGRATION (Activity-Dependent)");
        println!("  --------------------------------------");
        println!("    Migration steps: {:>8}", migration_steps);
        println!("    Mean displacement:{:>6.3} units", mean_displacement);
        println!("    Max displacement: {:>6.3} units", max_displacement);

        // Tissue substrate diagnostics
        println!("\n  TISSUE SUBSTRATE");
        println!("  -----------------");
        println!("    Plasticity updates: {:>5}", tissue_updates);
        let res = tissue.resistance_values();
        let cond = tissue.conductivity_values();
        if !res.is_empty() {
            let mean_r: f32 = res.iter().sum::<f32>() / res.len() as f32;
            let min_r = res.iter().copied().fold(f32::MAX, f32::min);
            let max_r = res.iter().copied().fold(f32::MIN, f32::max);
            let mean_c: f32 = cond.iter().sum::<f32>() / cond.len() as f32;
            let min_c = cond.iter().copied().fold(f32::MAX, f32::min);
            let max_c = cond.iter().copied().fold(f32::MIN, f32::max);
            println!("    Resistance:    mean={:.3}  min={:.3}  max={:.3}  (baseline={})",
                mean_r, min_r, max_r, tissue.config().baseline_resistance);
            println!("    Conductivity:  mean={:.3}  min={:.3}  max={:.3}  (baseline={})",
                mean_c, min_c, max_c, tissue.config().baseline_conductivity);

            // Per-role tissue stats
            let sensory_end = config.mfcc_bins;
            let inter_end = sensory_end + config.interneuron_count;
            let (mut sr, mut sc) = (0.0f32, 0.0f32);
            let (mut ir, mut ic) = (0.0f32, 0.0f32);
            let (mut mr, mut mc) = (0.0f32, 0.0f32);
            for i in 0..res.len() {
                if i < sensory_end { sr += res[i]; sc += cond[i]; }
                else if i < inter_end { ir += res[i]; ic += cond[i]; }
                else { mr += res[i]; mc += cond[i]; }
            }
            let sn = config.mfcc_bins as f32;
            let inn = config.interneuron_count as f32;
            let mn = config.motor_count as f32;
            println!("    Sensory:  R={:.3}  C={:.3}", sr / sn, sc / sn);
            println!("    Inter:    R={:.3}  C={:.3}", ir / inn, ic / inn);
            println!("    Motor:    R={:.3}  C={:.3}", mr / mn, mc / mn);
        }

        // Pruning diagnostics
        println!("\n  STRUCTURAL PRUNING");
        println!("  -------------------");
        println!("    Pruning cycles:  {:>8}", total_pruning_cycles);
        println!("    Hard pruned:     {:>8} (removed from store)", total_hard_pruned);
        println!("    Active synapses: {:>8}", cascade.synapses.len());
        println!("    Dormant synapses:{:>8}", cascade.synapses.count_dormant());
        println!("    Started with:    {:>8} synapses", synapse_count);

        // Motor neuron outlier diagnostic — find motors that migrated far from their origin
        println!("\n  MOTOR NEURON OUTLIER ANALYSIS");
        println!("  ------------------------------");
        let sensory_end = config.mfcc_bins;
        let inter_end = sensory_end + config.interneuron_count;
        let motor_start = inter_end;

        // Collect motor displacement + stats, sort by displacement descending
        let mut motor_stats: Vec<(usize, f32, [f32; 3], [f32; 3], u64, i16)> = Vec::new();
        for i in motor_start..neuron_count {
            let n = &cascade.neurons[i];
            let dx = n.soma.position[0] - initial_positions[i][0];
            let dy = n.soma.position[1] - initial_positions[i][1];
            let dz = n.soma.position[2] - initial_positions[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            motor_stats.push((i, dist, initial_positions[i], n.soma.position, n.last_spike_us, n.membrane));
        }
        motor_stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Print top 5 most displaced motors
        for (rank, &(idx, dist, orig, pos, last_spike, membrane)) in motor_stats.iter().take(5).enumerate() {
            let fired = last_spike > 0;
            // Count incoming synapses and their magnitudes
            let mut in_count = 0u32;
            let mut in_mag_sum = 0u32;
            let mut in_at_cap = 0u32;
            let mut presynaptic_roles: [u32; 3] = [0; 3]; // [sensory, inter, motor]
            for syn in cascade.synapses.iter() {
                if syn.target == idx as u32 {
                    in_count += 1;
                    in_mag_sum += syn.signal.magnitude as u32;
                    if syn.signal.magnitude == 255 { in_at_cap += 1; }
                    let src = syn.source as usize;
                    if src < sensory_end {
                        presynaptic_roles[0] += 1;
                    } else if src < inter_end {
                        presynaptic_roles[1] += 1;
                    } else {
                        presynaptic_roles[2] += 1;
                    }
                }
            }
            let in_mean_mag = if in_count > 0 { in_mag_sum as f32 / in_count as f32 } else { 0.0 };

            // Correlation with top partners
            let top_corr_partners = correlations.correlated_partners(idx, 0.1, time);
            let top3: Vec<(usize, f32)> = {
                let mut sorted = top_corr_partners;
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                sorted.into_iter().take(3).collect()
            };

            println!("    #{} Motor[{}]: displacement={:.2} units  {}",
                rank + 1, idx, dist, if fired { "FIRED" } else { "silent" });
            println!("        Origin: ({:.1}, {:.1}, {:.1}) → Now: ({:.1}, {:.1}, {:.1})",
                orig[0], orig[1], orig[2], pos[0], pos[1], pos[2]);
            println!("        Membrane: {}  Last spike: {}us",
                membrane, last_spike);
            println!("        Incoming: {} synapses (mean mag {:.1}, {} at cap)",
                in_count, in_mean_mag, in_at_cap);
            println!("        Presynaptic: {} sensory, {} inter, {} motor",
                presynaptic_roles[0], presynaptic_roles[1], presynaptic_roles[2]);
            if !top3.is_empty() {
                let partner_strs: Vec<String> = top3.iter().map(|(pid, c)| {
                    let role = if *pid < sensory_end { "S" }
                        else if *pid < inter_end { "I" }
                        else { "M" };
                    format!("{}[{}]={:.3}", role, pid, c)
                }).collect();
                println!("        Top correlated: {}", partner_strs.join(", "));
            }
        }

        // Mastery learning metrics
        let final_mean_mag: f32 = cascade.synapses.iter()
            .map(|s| s.signal.magnitude as f32).sum::<f32>() / synapse_count as f32;
        let active_synapses = cascade.synapses.count_active();
        let dormant_synapses = cascade.synapses.count_dormant();
        let total_changes = total_strengthened + total_weakened + total_dormant + total_awakened + total_flipped;

        println!("\n  MASTERY LEARNING (Hebbian Co-Firing)");
        println!("  -------------------------------------");
        println!("    Learning cycles: {:>8}", learning_cycles);
        println!("    Total changes:   {:>8}", total_changes);
        println!("    Strengthened:    {:>8}", total_strengthened);
        println!("    Weakened:        {:>8}", total_weakened);
        println!("    Gone dormant:    {:>8}", total_dormant);
        println!("    Awakened:        {:>8}", total_awakened);
        println!("    Flipped:         {:>8}", total_flipped);
        println!("    Mean magnitude:  {:>8.1} → {:.1} ({:+.1})",
            initial_mean_mag, final_mean_mag, final_mean_mag - initial_mean_mag);
        println!("    Active synapses: {:>8}/{}", active_synapses, synapse_count);
        println!("    Dormant synapses:{:>8}", dormant_synapses);

        // Magnitude distribution — reveals u8 cap saturation
        let mut mag_buckets = [0u32; 6]; // 0-42, 43-84, 85-127, 128-170, 171-212, 213-255
        let mut at_cap = 0u32;
        for syn in cascade.synapses.iter() {
            let bucket = (syn.signal.magnitude as usize / 43).min(5);
            mag_buckets[bucket] += 1;
            if syn.signal.magnitude == 255 { at_cap += 1; }
        }
        let bar = |n: u32| "#".repeat(((n as usize) + 4) / 5);
        println!("\n  MAGNITUDE DISTRIBUTION (each # = ~5 synapses)");
        println!("  -----------------------------------------------");
        println!("    0-42:    {:>4}  {}", mag_buckets[0], bar(mag_buckets[0]));
        println!("    43-84:   {:>4}  {}", mag_buckets[1], bar(mag_buckets[1]));
        println!("    85-127:  {:>4}  {}", mag_buckets[2], bar(mag_buckets[2]));
        println!("    128-170: {:>4}  {}", mag_buckets[3], bar(mag_buckets[3]));
        println!("    171-212: {:>4}  {}", mag_buckets[4], bar(mag_buckets[4]));
        println!("    213-255: {:>4}  {}", mag_buckets[5], bar(mag_buckets[5]));
        println!("    At cap (255): {}", at_cap);

        // Per-pathway magnitude breakdown
        let sensory_end = config.mfcc_bins;
        let inter_end = sensory_end + config.interneuron_count;
        let mut sen_inter_sum = 0u32;
        let mut sen_inter_n = 0u32;
        let mut sen_inter_cap = 0u32;
        let mut inter_motor_sum = 0u32;
        let mut inter_motor_n = 0u32;
        let mut inter_motor_cap = 0u32;
        for syn in cascade.synapses.iter() {
            let src = syn.source as usize;
            if src < sensory_end {
                sen_inter_sum += syn.signal.magnitude as u32;
                sen_inter_n += 1;
                if syn.signal.magnitude == 255 { sen_inter_cap += 1; }
            } else if src < inter_end {
                inter_motor_sum += syn.signal.magnitude as u32;
                inter_motor_n += 1;
                if syn.signal.magnitude == 255 { inter_motor_cap += 1; }
            }
        }
        let sen_inter_mean = if sen_inter_n > 0 { sen_inter_sum as f32 / sen_inter_n as f32 } else { 0.0 };
        let inter_motor_mean = if inter_motor_n > 0 { inter_motor_sum as f32 / inter_motor_n as f32 } else { 0.0 };
        println!("\n  PER-PATHWAY MAGNITUDE");
        println!("  ----------------------");
        println!("    Sensory->Inter:  mean {:>6.1}, at cap: {:>3}/{} (started at 100)",
            sen_inter_mean, sen_inter_cap, sen_inter_n);
        println!("    Inter->Motor:    mean {:>6.1}, at cap: {:>3}/{} (started at 80)",
            inter_motor_mean, inter_motor_cap, inter_motor_n);

        // ================================================================
        // REFLEX ARC DIAGNOSTIC — Can signals physically traverse the gap?
        // ================================================================
        // Wire direct sensory→motor synapses (bypassing interneurons) and
        // confirm the physics (leak, delay, attenuation) allows long-range
        // propagation. This disambiguates "no learned path" from "physics
        // is too suppressive for signals to ever reach motor."

        println!("\n  REFLEX ARC DIAGNOSTIC");
        println!("  ----------------------");

        // Fresh cascade — same network but with direct reflex synapses added
        let (reflex_neurons, mut reflex_synapses) = build_hush_network(&config);
        let reflex_total = reflex_neurons.len();

        // Identify sensory and motor neuron indices
        let sensory_indices: Vec<usize> = reflex_neurons.iter().enumerate()
            .filter(|(_, n)| n.nuclei.is_sensory()).map(|(i, _)| i).collect();
        let motor_indices: Vec<usize> = reflex_neurons.iter().enumerate()
            .filter(|(_, n)| n.nuclei.is_motor()).map(|(i, _)| i).collect();

        // Measure the physical gap
        let avg_sensory_x: f32 = sensory_indices.iter()
            .map(|&i| reflex_neurons[i].soma.position[0]).sum::<f32>() / sensory_indices.len() as f32;
        let avg_motor_x: f32 = motor_indices.iter()
            .map(|&i| reflex_neurons[i].soma.position[0]).sum::<f32>() / motor_indices.len() as f32;
        let gap = (avg_motor_x - avg_sensory_x).abs();
        println!("    Sensory→Motor gap: {:.1} units (avg x: {:.1} → {:.1})", gap, avg_sensory_x, avg_motor_x);

        // Wire ALL sensory → a motor neuron each (whichever sensory fires, there's a path)
        for k in 0..sensory_indices.len() {
            let motor_target = motor_indices[k % motor_indices.len()];
            reflex_synapses.add(SpatialSynapse::excitatory(
                sensory_indices[k] as u32,
                motor_target as u32,
                255, // max magnitude → 2040 current (threshold needs +1500 from rest)
                0,   // delay=0 → compute from distance + myelin
            ));
        }
        reflex_synapses.rebuild_index(reflex_total);
        println!("    Reflex synapses:   {} direct sensory→motor (magnitude=255, current=2040)",
            sensory_indices.len());
        println!("    Threshold gap:     1500 from resting (-7000 → -5500)");

        let mut reflex_cascade = SpatialCascade::with_network(
            reflex_neurons,
            reflex_synapses,
            SpatialCascadeConfig {
                max_events_per_call: 50_000,
                ..Default::default()
            },
        );

        // Feed first real utterance through the reflex network
        let card = spool.read_card(0).expect("failed to re-read card 0");
        let sample = SpoolSample::from_card(card).expect("card 0 malformed");
        let test_frames = 200.min(sample.n_frames); // 2 seconds of audio
        let mut reflex_time = 0u64;
        let mut mfcc_reflex = [0.0f32; 26];

        // Track peak membrane and max MFCC values per sensory neuron
        let mut peak_membrane = vec![i16::MIN; sensory_indices.len()];
        let mut peak_mfcc = vec![0.0f32; sensory_indices.len()];

        for f in 0..test_frames {
            sample.frame_f32(f, &mut mfcc_reflex);
            inject_mfcc_frame(&mut reflex_cascade, &mfcc_reflex, true, reflex_time);
            reflex_cascade.run_until(reflex_time + frame_interval);

            // Track peak values
            for (k, &si) in sensory_indices.iter().enumerate() {
                let m = reflex_cascade.neurons[si].membrane;
                if m > peak_membrane[k] { peak_membrane[k] = m; }
                if mfcc_reflex[k].abs() > peak_mfcc[k] { peak_mfcc[k] = mfcc_reflex[k].abs(); }
            }

            reflex_time += frame_interval;
        }

        // Check which sensory neurons fired
        let mut reflex_sensory_fired = 0;
        for &si in &sensory_indices {
            if reflex_cascade.neurons[si].last_spike_us > 0 {
                reflex_sensory_fired += 1;
            }
        }

        // Check which motor neurons fired (all of them — any reflex target)
        let mut reflex_motor_fired = 0;
        let mut reflex_motor_samples: Vec<(usize, i16, u64)> = Vec::new();
        for &mi in &motor_indices {
            let n = &reflex_cascade.neurons[mi];
            if n.last_spike_us > 0 {
                reflex_motor_fired += 1;
            }
            reflex_motor_samples.push((mi, n.membrane, n.last_spike_us));
        }

        let physics_ok = reflex_motor_fired > 0;
        println!("    Frames processed:  {} ({:.0}ms)", test_frames, test_frames as f64 * 10.0);
        println!("    Sensory fired:     {}/{}", reflex_sensory_fired, sensory_indices.len());
        println!("    Motor fired:       {}/{}", reflex_motor_fired, motor_indices.len());
        // Show first 8 motor neurons for diagnostics
        for &(mi, membrane, last_spike) in reflex_motor_samples.iter().take(8) {
            let spiked = if last_spike > 0 { "SPIKED" } else { "silent" };
            println!("      Motor[{}]: membrane={:>6}, last_spike={}us [{}]",
                mi, membrane, last_spike, spiked);
        }

        // Sensory neuron diagnostics — what's happening at the injection site?
        let global_peak = peak_membrane.iter().copied().max().unwrap_or(i16::MIN);
        let global_peak_mfcc = peak_mfcc.iter().copied().fold(0.0f32, f32::max);
        println!("    Peak sensory membrane: {} (threshold={})", global_peak, SpatialNeuron::DEFAULT_THRESHOLD);
        println!("    Peak MFCC |value|:     {:.3} (inject current: {:.0})",
            global_peak_mfcc, global_peak_mfcc * 2000.0);
        // Show per-bin peaks for first 10
        for k in 0..10.min(sensory_indices.len()) {
            println!("      Bin[{}]: peak_membrane={:>6}, peak_|mfcc|={:.3} (current={:.0})",
                k, peak_membrane[k], peak_mfcc[k], peak_mfcc[k] * 2000.0);
        }

        if physics_ok {
            println!("    ==> PHYSICS OK: Signals traverse sensory→motor gap");
        } else if reflex_sensory_fired == 0 {
            println!("    ==> SENSORY SILENT: Injection current too weak to cross threshold");
            println!("        Peak membrane {} vs threshold {}, gap = {}",
                global_peak, SpatialNeuron::DEFAULT_THRESHOLD,
                SpatialNeuron::DEFAULT_THRESHOLD - global_peak);
        } else {
            println!("    ==> PHYSICS BLOCKED: Sensory fires but signal dies before reaching motor");
        }

        println!("\n  VERDICT");
        println!("  -------");

        let realtime_capable = run_ms < simulated_s * 1000.0;
        println!("    Real-time:       {} ({}x)",
            if realtime_capable { "YES" } else { "NO" },
            (simulated_s * 1000.0 / run_ms) as i32);
        println!("    Utilization:     Sensory {:.0}% / Inter {:.0}% / Motor {:.0}%",
            sensory_util * 100.0, inter_util * 100.0, motor_util * 100.0);
        println!("    Regions:         {}", regions.len());
        println!("    Migration:       {:.3} mean / {:.3} max units",
            mean_displacement, max_displacement);
        println!("    Reflex arc:      {}", if physics_ok { "PHYSICS OK" } else { "BLOCKED" });

        if realtime_capable && physics_ok {
            println!("\n    ==> REAL AUDIO: PASSES");
        } else {
            if !realtime_capable { println!("\n    ==> NEEDS OPTIMIZATION (too slow)"); }
            if !physics_ok { println!("\n    ==> NEEDS TUNING (signals can't reach motor)"); }
        }

        println!("\n{}\n", "=".repeat(70));

        // Assertions
        assert!(realtime_capable, "Must process real LibriSpeech faster than real-time");
        assert!(total_frames > 0, "Must have processed at least some frames");
        assert!(physics_ok, "Reflex arc: signals must be able to traverse sensory→motor gap");
    }

    // ========================================================================
    // STRESS TEST — Push the limits
    // ========================================================================

    #[test]
    fn stress_test_10k_frames() {
        println!("\n=== STRESS TEST: 10,000 Frames (~100 seconds of audio) ===\n");

        let config = HushConfig {
            mfcc_bins: 26,
            interneuron_count: 128,
            motor_count: 64,
            ..Default::default()
        };

        let (neurons, synapses) = build_hush_network(&config);
        let neuron_count = neurons.len();

        let cascade_config = SpatialCascadeConfig {
            max_events_per_call: 100_000, // Higher limit for stress test
            ..Default::default()
        };
        let mut cascade = SpatialCascade::with_network(neurons, synapses, cascade_config);

        let start = Instant::now();

        let mut time = 0u64;
        let frame_interval = 10_000u64;

        for frame in 0..10_000 {
            let phoneme = (frame / 25) % 20;
            let mfcc = generate_mfcc_frame(frame, phoneme);
            inject_mfcc_frame(&mut cascade, &mfcc, true, time);
            cascade.run_until(time + frame_interval);
            time += frame_interval;

            // Progress indicator
            if frame % 1000 == 999 {
                let elapsed = start.elapsed().as_secs_f64();
                let simulated = (frame + 1) as f64 * 0.01; // seconds
                println!("  Frame {}: {:.1}s simulated in {:.1}s real ({:.1}x)",
                    frame + 1, simulated, elapsed, simulated / elapsed);
            }
        }

        let total_time = start.elapsed().as_secs_f64();
        let simulated_time = time as f64 / 1_000_000.0;

        let (sensory_util, inter_util, motor_util) = role_utilization(&cascade);

        println!("\nSTRESS TEST RESULTS:");
        println!("  Neurons:           {}", neuron_count);
        println!("  Frames:            10,000");
        println!("  Simulated:         {:.1}s", simulated_time);
        println!("  Real time:         {:.1}s", total_time);
        println!("  Speed ratio:       {:.1}x real-time", simulated_time / total_time);
        println!("  Total spikes:      {}", cascade.total_spikes());
        println!("  Spikes/sec:        {:.0}", cascade.total_spikes() as f64 / simulated_time);
        println!("  Sensory util:      {:.1}%", sensory_util * 100.0);
        println!("  Interneuron util:  {:.1}%", inter_util * 100.0);
        println!("  Motor util:        {:.1}%", motor_util * 100.0);

        assert!(
            simulated_time / total_time > 1.0,
            "Must be at least real-time capable"
        );
    }
}
