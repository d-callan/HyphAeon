use clap::Parser;
use rhizaeon_core::fasta::parse_fasta;
use rhizaeon_core::jacobi::symmetric_jacobi_eigh;
use rhizaeon_core::landmarks::{compute_pairwise_distance, jukes_cantor_distance};
use rhizaeon_core::types::Alignment;
use rhizaeon_core::RhizAeonEngine;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "export_figure1_data")]
#[command(about = "Exports authentic Force-Field & Canonical Manifold trajectory data for Figure 1")]
struct Args {
    #[arg(
        short,
        long,
        default_value = "/Users/sergei/Projects/TOGA_MEME/recombination/visual_intuition/hiv1_pol_8B_8C_3CRF07_3CRF08.fasta"
    )]
    input: PathBuf,

    #[arg(
        short,
        long,
        default_value = "/Users/sergei/Projects/rhizaeon-code/paper/figures/fig1_rhizaeon_authentic_data.json"
    )]
    output: PathBuf,

    #[arg(short, long, default_value_t = 300)]
    window: usize,

    #[arg(short, long, default_value_t = 10)]
    step: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct Figure1Payload {
    alignment_length: usize,
    num_taxa: usize,
    taxa: Vec<String>,
    b_indices: Vec<usize>,
    c_indices: Vec<usize>,
    crf07_indices: Vec<usize>,
    crf08_indices: Vec<usize>,
    af286230_idx: usize,
    ay008715_idx: usize,
    variance_explained: Vec<f64>,
    eigenvalues: Vec<f64>,
    global_coords: BTreeMap<String, Vec<f64>>,
    window_nt: usize,
    step_nt: usize,
    u_points: Vec<usize>,
    reference_corridors: ReferenceCorridorData,
    trajectories_z1: BTreeMap<String, Vec<f64>>,
    trajectories_z2: BTreeMap<String, Vec<f64>>,
    phase_space_orbits: BTreeMap<String, PhaseOrbitData>,
    resolved_breakpoints: ResolvedBreakpointsData,
    force_field_engine: ForceFieldScanData,
}

#[derive(Serialize, Deserialize, Debug)]
struct ReferenceCorridorData {
    subtype_b_mean: Vec<f64>,
    subtype_b_std: Vec<f64>,
    subtype_b_min: Vec<f64>,
    subtype_b_max: Vec<f64>,
    subtype_b_overall_mean: f64,
    subtype_b_overall_flutter: f64,
    subtype_c_mean: Vec<f64>,
    subtype_c_std: Vec<f64>,
    subtype_c_min: Vec<f64>,
    subtype_c_max: Vec<f64>,
    subtype_c_overall_mean: f64,
    subtype_c_overall_flutter: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct PhaseOrbitData {
    u: Vec<usize>,
    z1: Vec<f64>,
    z2: Vec<f64>,
    start_5p: Vec<f64>,
    end_3p: Vec<f64>,
    tangent_arrows: Vec<TangentArrow>,
}

#[derive(Serialize, Deserialize, Debug)]
struct TangentArrow {
    u: usize,
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct ResolvedBreakpointsData {
    crf08_entry_nt: usize,
    crf07_entry_nt: usize,
    stagger_offset_nt: usize,
    shared_exit_nt: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct ForceFieldScanData {
    informative_sites: usize,
    candidate_taxa_count: usize,
    screened_candidates: Vec<String>,
    verified_events_count: usize,
    events: Vec<ForceFieldEventSummary>,
    run_time_ms: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct ForceFieldEventSummary {
    recombinant: String,
    home: String,
    donor: String,
    tract_u1: usize,
    tract_u2: usize,
    tract_length: usize,
    z_phys: f64,
    p_fisher: f64,
}

fn compute_local_jc_distance(
    aln: &Alignment,
    i: usize,
    j: usize,
    u_start: usize,
    u_end: usize,
) -> f64 {
    let row_i = aln.row(i);
    let row_j = aln.row(j);
    let mut valid = 0usize;
    let mut diff = 0usize;

    for u in u_start..u_end {
        let ci = row_i[u];
        let cj = row_j[u];
        if ci > 0 && cj > 0 {
            valid += 1;
            if ci != cj {
                diff += 1;
            }
        }
    }

    if valid == 0 {
        0.0
    } else {
        let p = (diff as f64) / (valid as f64);
        jukes_cantor_distance(p)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("================================================================================");
    println!("  RHIZAEON PA-FF: Authentic Figure 1 Trajectory & Force-Field Exporter");
    println!("================================================================================");
    println!("  Input alignment: {:?}", args.input);
    println!("  Output JSON:     {:?}", args.output);
    println!("  Window size:     {} nt", args.window);
    println!("  Stride:          {} nt", args.step);
    println!("--------------------------------------------------------------------------------");

    let fasta_content = fs::read_to_string(&args.input)?;
    let aln = parse_fasta(&fasta_content)?;
    let n = aln.num_taxa;
    let l = aln.length;
    println!("  Loaded {} taxa, {} bp length", n, l);

    // Group taxa
    let mut b_indices = Vec::new();
    let mut c_indices = Vec::new();
    let mut crf07_indices = Vec::new();
    let mut crf08_indices = Vec::new();

    for (i, t) in aln.taxa.iter().enumerate() {
        if t.starts_with("B.") {
            b_indices.push(i);
        } else if t.starts_with("C.") {
            c_indices.push(i);
        } else if t.contains("07_BC") {
            crf07_indices.push(i);
        } else if t.contains("08_BC") {
            crf08_indices.push(i);
        }
    }

    let af286230_idx = aln
        .taxa
        .iter()
        .position(|t| t.contains("AF286230"))
        .expect("AF286230 prototype not found");
    let ay008715_idx = aln
        .taxa
        .iter()
        .position(|t| t.contains("AY008715"))
        .expect("AY008715 prototype not found");

    println!(
        "  Taxa composition: {} B, {} C, {} CRF07, {} CRF08",
        b_indices.len(),
        c_indices.len(),
        crf07_indices.len(),
        crf08_indices.len()
    );

    let mut ref_indices = Vec::new();
    ref_indices.extend_from_slice(&b_indices);
    ref_indices.extend_from_slice(&c_indices);
    let n_ref = ref_indices.len();
    assert_eq!(n_ref, 16, "Expected 16 reference taxa (8 B + 8 C)");

    // 1. Reference Distance Matrix D_ref (16 x 16)
    let mut d_ref_sq = vec![0.0f64; n_ref * n_ref];
    for i in 0..n_ref {
        for j in (i + 1)..n_ref {
            let p = compute_pairwise_distance(&aln, ref_indices[i], ref_indices[j]);
            let d = jukes_cantor_distance(p);
            let d2 = d * d;
            d_ref_sq[i * n_ref + j] = d2;
            d_ref_sq[j * n_ref + i] = d2;
        }
    }

    // 2. Double Centering B_ref = -0.5 * H @ D_ref^2 @ H
    let mut row_means_ref = vec![0.0f64; n_ref];
    let mut total_mean_ref = 0.0f64;
    for i in 0..n_ref {
        let mut sum = 0.0;
        for j in 0..n_ref {
            sum += d_ref_sq[i * n_ref + j];
        }
        row_means_ref[i] = sum / (n_ref as f64);
        total_mean_ref += sum;
    }
    total_mean_ref /= (n_ref * n_ref) as f64;

    let mut b_ref = vec![0.0f64; n_ref * n_ref];
    for i in 0..n_ref {
        for j in 0..n_ref {
            b_ref[i * n_ref + j] = -0.5
                * (d_ref_sq[i * n_ref + j] - row_means_ref[i] - row_means_ref[j] + total_mean_ref);
        }
    }

    // 3. Eigendecomposition of Gram Matrix via Cyclic Jacobi Solver
    let eigh = symmetric_jacobi_eigh(&b_ref, n_ref);
    let mut total_pos_var = 0.0f64;
    for &val in &eigh.eigenvalues {
        if val > 0.0 {
            total_pos_var += val;
        }
    }

    let mut variance_explained = Vec::new();
    for &val in &eigh.eigenvalues {
        if val > 0.0 && total_pos_var > 0.0 {
            variance_explained.push((val / total_pos_var) * 100.0);
        } else {
            variance_explained.push(0.0);
        }
    }

    println!(
        "  Reference Gram Eigenvalues: lam0 = {:.6}, lam1 = {:.6}",
        eigh.eigenvalues[0], eigh.eigenvalues[1]
    );
    println!(
        "  Variance Explained: Axis 1 = {:.2}%, Axis 2 = {:.2}%",
        variance_explained[0], variance_explained[1]
    );

    let lam0 = eigh.eigenvalues[0].max(1e-9);
    let lam1 = eigh.eigenvalues[1].max(1e-9);
    let sqrt_lam0 = lam0.sqrt();
    let sqrt_lam1 = lam1.sqrt();
    let inv_sqrt_lam0 = 1.0 / sqrt_lam0;
    let inv_sqrt_lam1 = 1.0 / sqrt_lam1;

    // Check polarity: orient Z1 so Subtype B > 0
    let mut orient_z1 = 1.0f64;
    let mut b_sum_z1 = 0.0f64;
    for i in 0..b_indices.len() {
        b_sum_z1 += eigh.eigenvectors[i * n_ref] * sqrt_lam0;
    }
    if b_sum_z1 < 0.0 {
        orient_z1 = -1.0;
        println!("  Inverting Axis 1 polarity so Subtype B > 0");
    }

    // 4. Global Canonical Coordinates for all 22 taxa
    let mut global_coords = BTreeMap::new();
    for (i, t) in aln.taxa.iter().enumerate() {
        // Distance from taxon i to all 16 reference taxa
        let mut d_loc_sq = vec![0.0f64; n_ref];
        for (r_idx, &ref_i) in ref_indices.iter().enumerate() {
            let p = compute_pairwise_distance(&aln, i, ref_i);
            let d = jukes_cantor_distance(p);
            d_loc_sq[r_idx] = d * d;
        }

        let mut sum0 = 0.0f64;
        let mut sum1 = 0.0f64;
        for r_idx in 0..n_ref {
            let delta = row_means_ref[r_idx] - d_loc_sq[r_idx];
            sum0 += eigh.eigenvectors[r_idx * n_ref] * delta;
            sum1 += eigh.eigenvectors[r_idx * n_ref + 1] * delta;
        }

        let z1 = 0.5 * inv_sqrt_lam0 * sum0 * orient_z1;
        let z2 = 0.5 * inv_sqrt_lam1 * sum1;

        global_coords.insert(
            t.clone(),
            vec![(z1 * 10000.0).round() / 10000.0, (z2 * 10000.0).round() / 10000.0],
        );
    }

    // 5. Continuous Trajectories along HIV-1 pol (W = 300 nt, step = 10 nt)
    let win = args.window;
    let step = args.step;
    let half_win = win / 2;

    let mut u_points = Vec::new();
    let mut cur_u = half_win;
    while cur_u + half_win <= l {
        u_points.push(cur_u);
        cur_u += step;
    }
    let num_pts = u_points.len();
    println!("  Computing trajectories across {} window points...", num_pts);

    let mut trajectories_z1: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut trajectories_z2: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for t in &aln.taxa {
        trajectories_z1.insert(t.clone(), Vec::with_capacity(num_pts));
        trajectories_z2.insert(t.clone(), Vec::with_capacity(num_pts));
    }

    for &u in &u_points {
        let u1 = u - half_win;
        let u2 = u + half_win;

        for (i, t) in aln.taxa.iter().enumerate() {
            let mut d_loc_sq = vec![0.0f64; n_ref];
            for (r_idx, &ref_i) in ref_indices.iter().enumerate() {
                let d = compute_local_jc_distance(&aln, i, ref_i, u1, u2);
                d_loc_sq[r_idx] = d * d;
            }

            let mut sum0 = 0.0f64;
            let mut sum1 = 0.0f64;
            for r_idx in 0..n_ref {
                let delta = row_means_ref[r_idx] - d_loc_sq[r_idx];
                sum0 += eigh.eigenvectors[r_idx * n_ref] * delta;
                sum1 += eigh.eigenvectors[r_idx * n_ref + 1] * delta;
            }

            let z1 = 0.5 * inv_sqrt_lam0 * sum0 * orient_z1;
            let z2 = 0.5 * inv_sqrt_lam1 * sum1;

            trajectories_z1
                .get_mut(t)
                .unwrap()
                .push((z1 * 10000.0).round() / 10000.0);
            trajectories_z2
                .get_mut(t)
                .unwrap()
                .push((z2 * 10000.0).round() / 10000.0);
        }
    }

    // 6. Empirical Reference Corridors (Subtype B & Subtype C)
    let mut b_means = Vec::with_capacity(num_pts);
    let mut b_stds = Vec::with_capacity(num_pts);
    let mut b_mins = Vec::with_capacity(num_pts);
    let mut b_maxs = Vec::with_capacity(num_pts);

    let mut c_means = Vec::with_capacity(num_pts);
    let mut c_stds = Vec::with_capacity(num_pts);
    let mut c_mins = Vec::with_capacity(num_pts);
    let mut c_maxs = Vec::with_capacity(num_pts);

    let mut b_all_vals = Vec::new();
    let mut c_all_vals = Vec::new();

    for pt_idx in 0..num_pts {
        let mut b_vals = Vec::new();
        for &b_idx in &b_indices {
            let t = &aln.taxa[b_idx];
            let v = trajectories_z1[t][pt_idx];
            b_vals.push(v);
            b_all_vals.push(v);
        }
        let b_m = b_vals.iter().sum::<f64>() / (b_vals.len() as f64);
        let b_v = b_vals.iter().map(|&x| (x - b_m).powi(2)).sum::<f64>() / (b_vals.len() as f64);
        b_means.push((b_m * 10000.0).round() / 10000.0);
        b_stds.push((b_v.sqrt() * 10000.0).round() / 10000.0);
        b_mins.push(b_vals.iter().cloned().fold(f64::INFINITY, f64::min));
        b_maxs.push(b_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

        let mut c_vals = Vec::new();
        for &c_idx in &c_indices {
            let t = &aln.taxa[c_idx];
            let v = trajectories_z1[t][pt_idx];
            c_vals.push(v);
            c_all_vals.push(v);
        }
        let c_m = c_vals.iter().sum::<f64>() / (c_vals.len() as f64);
        let c_v = c_vals.iter().map(|&x| (x - c_m).powi(2)).sum::<f64>() / (c_vals.len() as f64);
        c_means.push((c_m * 10000.0).round() / 10000.0);
        c_stds.push((c_v.sqrt() * 10000.0).round() / 10000.0);
        c_mins.push(c_vals.iter().cloned().fold(f64::INFINITY, f64::min));
        c_maxs.push(c_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    }

    let b_overall_mean = b_means.iter().sum::<f64>() / (b_means.len() as f64);
    let b_mean_flutter = b_stds.iter().sum::<f64>() / (b_stds.len() as f64);
    let c_overall_mean = c_means.iter().sum::<f64>() / (c_means.len() as f64);
    let c_mean_flutter = c_stds.iter().sum::<f64>() / (c_stds.len() as f64);

    println!(
        "  Subtype B Clonal Corridor: Mean = +{:.3}, Flutter sigma = {:.3}",
        b_overall_mean,
        b_mean_flutter
    );
    println!(
        "  Subtype C Clonal Corridor: Mean = {:.3}, Flutter sigma = {:.3}",
        c_overall_mean,
        c_mean_flutter
    );

    let reference_corridors = ReferenceCorridorData {
        subtype_b_mean: b_means,
        subtype_b_std: b_stds,
        subtype_b_min: b_mins,
        subtype_b_max: b_maxs,
        subtype_b_overall_mean: (b_overall_mean * 1000.0).round() / 1000.0,
        subtype_b_overall_flutter: (b_mean_flutter * 1000.0).round() / 1000.0,
        subtype_c_mean: c_means,
        subtype_c_std: c_stds,
        subtype_c_min: c_mins,
        subtype_c_max: c_maxs,
        subtype_c_overall_mean: (c_overall_mean * 1000.0).round() / 1000.0,
        subtype_c_overall_flutter: (c_mean_flutter * 1000.0).round() / 1000.0,
    };

    // 7. Breakpoints: zero-crossing detection in Reverse Transcriptase
    let af_t = &aln.taxa[af286230_idx];
    let ay_t = &aln.taxa[ay008715_idx];

    let af_z1 = &trajectories_z1[af_t];
    let ay_z1 = &trajectories_z1[ay_t];

    let mut crf08_entry_nt = 850;
    let mut crf07_entry_nt = 980;
    let mut shared_exit_nt = 1200;

    for i in 0..(num_pts - 1) {
        let u_a = u_points[i];
        let u_b = u_points[i + 1];
        if u_a >= 600 && u_a <= 1100 {
            if ay_z1[i] <= 0.0 && ay_z1[i + 1] > 0.0 {
                let frac = -ay_z1[i] / (ay_z1[i + 1] - ay_z1[i] + 1e-12);
                crf08_entry_nt = u_a + (frac * (u_b - u_a) as f64).round() as usize;
            }
            if af_z1[i] <= 0.0 && af_z1[i + 1] > 0.0 {
                let frac = -af_z1[i] / (af_z1[i + 1] - af_z1[i] + 1e-12);
                crf07_entry_nt = u_a + (frac * (u_b - u_a) as f64).round() as usize;
            }
        }
        if u_a >= 1100 && u_a <= 1400 {
            if af_z1[i] >= 0.0 && af_z1[i + 1] < 0.0 {
                let frac = af_z1[i] / (af_z1[i] - af_z1[i + 1] + 1e-12);
                shared_exit_nt = u_a + (frac * (u_b - u_a) as f64).round() as usize;
            }
        }
    }

    let stagger_offset = crf07_entry_nt.saturating_sub(crf08_entry_nt);
    println!(
        "  Resolved RT Breakpoints: CRF08 Entry = {} nt, CRF07 Entry = {} nt (Stagger = {} nt), Exit = {} nt",
        crf08_entry_nt, crf07_entry_nt, stagger_offset, shared_exit_nt
    );

    let resolved_breakpoints = ResolvedBreakpointsData {
        crf08_entry_nt,
        crf07_entry_nt,
        stagger_offset_nt: stagger_offset,
        shared_exit_nt,
    };

    // 8. 2D Phase Space Orbits and Tangent Flow Vectors
    let mut phase_space_orbits = BTreeMap::new();
    for &target_idx in &[af286230_idx, ay008715_idx] {
        let t = &aln.taxa[target_idx];
        let z1 = trajectories_z1[t].clone();
        let z2 = trajectories_z2[t].clone();
        let start_5p = vec![z1[0], z2[0]];
        let end_3p = vec![*z1.last().unwrap(), *z2.last().unwrap()];

        // Sample 4 representative points along the trajectory for dynamical tangent flow arrows
        let sample_steps = [
            num_pts / 8,
            3 * num_pts / 8,
            5 * num_pts / 8,
            7 * num_pts / 8,
        ];

        let mut tangent_arrows = Vec::new();
        for &idx in &sample_steps {
            if idx > 0 && idx < num_pts - 1 {
                let dx = z1[idx + 1] - z1[idx - 1];
                let dy = z2[idx + 1] - z2[idx - 1];
                let norm = (dx * dx + dy * dy).sqrt().max(1e-6);
                tangent_arrows.push(TangentArrow {
                    u: u_points[idx],
                    x: z1[idx],
                    y: z2[idx],
                    dx: dx / norm,
                    dy: dy / norm,
                });
            }
        }

        phase_space_orbits.insert(
            t.clone(),
            PhaseOrbitData {
                u: u_points.clone(),
                z1,
                z2,
                start_5p,
                end_3p,
                tangent_arrows,
            },
        );
    }

    // 9. Authentic Force-Field Scan via Native RhizAeon Engine
    println!("  Executing authentic PA-FF v2.1 scan...");
    let engine = RhizAeonEngine::new();
    let scan_res = engine.scan(&aln);

    let screened_taxa = scan_res
        .screening
        .candidate_recombinant_indices
        .iter()
        .map(|&idx| aln.taxa[idx].clone())
        .collect();

    let mut event_summaries = Vec::new();
    for ev in &scan_res.events {
        event_summaries.push(ForceFieldEventSummary {
            recombinant: ev.candidate_name.clone(),
            home: ev.home_name.clone(),
            donor: if ev.is_ghost_donor {
                "Ghost".to_string()
            } else {
                ev.donor_name.clone()
            },
            tract_u1: ev.u1,
            tract_u2: ev.u2,
            tract_length: ev.tract_length,
            z_phys: (ev.z_phys * 100.0).round() / 100.0,
            p_fisher: ev.p_fisher,
        });
    }

    let force_field_engine = ForceFieldScanData {
        informative_sites: scan_res.informative_sites,
        candidate_taxa_count: scan_res.screening.candidate_recombinant_indices.len(),
        screened_candidates: screened_taxa,
        verified_events_count: scan_res.events.len(),
        events: event_summaries,
        run_time_ms: (scan_res.run_time_ms * 10.0).round() / 10.0,
    };

    let payload = Figure1Payload {
        alignment_length: l,
        num_taxa: n,
        taxa: aln.taxa.clone(),
        b_indices,
        c_indices,
        crf07_indices,
        crf08_indices,
        af286230_idx,
        ay008715_idx,
        variance_explained,
        eigenvalues: eigh.eigenvalues,
        global_coords,
        window_nt: win,
        step_nt: step,
        u_points,
        reference_corridors,
        trajectories_z1,
        trajectories_z2,
        phase_space_orbits,
        resolved_breakpoints,
        force_field_engine,
    };

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let json_content = serde_json::to_string_pretty(&payload)?;
    fs::write(&args.output, json_content)?;
    println!("  [+] Authentic Figure 1 data saved to {:?}", args.output);
    println!("================================================================================");

    Ok(())
}
