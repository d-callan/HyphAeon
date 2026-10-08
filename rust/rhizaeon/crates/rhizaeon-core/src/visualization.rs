use crate::jacobi::symmetric_jacobi_eigh;
use crate::types::{
    Alignment, CanonicalManifold, GowerPoint, InformativeSnpRecord, MacroSegment,
    MosaicSegment, ScanResult, TaxonMeta, VisualizationBreakpoint, VisualizationDossier,
    VisualizationMetadata, VisualizationSignals, VisualizationSubspaces,
};
use std::collections::{BTreeMap, HashSet};

/// Embed the dashboard HTML template at compile time
pub const DASHBOARD_HTML_TEMPLATE: &str = include_str!("../assets/dashboard_template.html");

/// Standard categorical color palette for parents and clades
const PALETTE: &[&str] = &[
    "#d55e00", // Vermilion / Orange
    "#0072b2", // Blue
    "#10b981", // Emerald
    "#ec4899", // Pink
    "#8b5cf6", // Purple-indigo
    "#f59e0b", // Amber
    "#06b6d4", // Cyan
    "#059669", // Dark emerald
    "#2563eb", // Royal blue
];

/// Generates a rich, structured, organism-agnostic VisualizationDossier from an alignment and scan results.
pub fn generate_visualization_dossier(
    aln: &Alignment,
    scan_res: &ScanResult,
    title_opt: Option<&str>,
) -> VisualizationDossier {
    let n = aln.num_taxa;
    let l = aln.length;

    let default_title = format!("RhizAeon Scan Analysis ({} taxa, {} bp)", n, l);
    let title = title_opt.unwrap_or(&default_title).to_string();

    // 1. Reconstruct decoded nucleotide sequences
    let mut sequences = BTreeMap::new();
    for i in 0..n {
        let mut seq_str = String::with_capacity(l);
        for u in 0..l {
            let ch = match aln.get(i, u) {
                1 => 'A',
                2 => 'C',
                3 => 'G',
                4 => 'T',
                _ => '-',
            };
            seq_str.push(ch);
        }
        sequences.insert(aln.taxa[i].clone(), seq_str);
    }

    // 2. Identify recombinant, parental, and reference taxa
    let mut rec_indices = Vec::new();
    let mut rec_taxa_set = HashSet::new();
    let mut parent_taxa_set = HashSet::new();

    for ev in &scan_res.events {
        if !rec_taxa_set.contains(&ev.candidate_name) {
            rec_indices.push(ev.candidate_idx);
            rec_taxa_set.insert(ev.candidate_name.clone());
        }
        if !ev.home_name.is_empty() && ev.home_name != "Ghost" {
            parent_taxa_set.insert(ev.home_name.clone());
        }
        if !ev.donor_name.is_empty() && ev.donor_name != "Ghost" && !ev.is_ghost_donor {
            parent_taxa_set.insert(ev.donor_name.clone());
        }
    }

    // Ordering: Recombinants first, then Parents, then References
    let mut sorted_taxa = Vec::with_capacity(n);
    for t in &aln.taxa {
        if rec_taxa_set.contains(t) {
            sorted_taxa.push(t.clone());
        }
    }
    for t in &aln.taxa {
        if !rec_taxa_set.contains(t) && parent_taxa_set.contains(t) {
            sorted_taxa.push(t.clone());
        }
    }
    for t in &aln.taxa {
        if !rec_taxa_set.contains(t) && !parent_taxa_set.contains(t) {
            sorted_taxa.push(t.clone());
        }
    }

    // 3. Taxa metadata and color mapping
    let mut taxa_meta = BTreeMap::new();
    let mut parent_color_idx = 0;
    let mut parent_colors: BTreeMap<String, String> = BTreeMap::new();

    for t in &sorted_taxa {
        if rec_taxa_set.contains(t) {
            taxa_meta.insert(
                t.clone(),
                TaxonMeta {
                    label: format!("{} [Recombinant]", t),
                    taxon_type: "recombinant".to_string(),
                    color: "#a855f7".to_string(), // Purple
                },
            );
        } else if parent_taxa_set.contains(t) {
            let color = PALETTE[parent_color_idx % PALETTE.len()].to_string();
            parent_colors.insert(t.clone(), color.clone());
            parent_color_idx += 1;
            taxa_meta.insert(
                t.clone(),
                TaxonMeta {
                    label: format!("{} [Parent Donor]", t),
                    taxon_type: "parent".to_string(),
                    color,
                },
            );
        } else {
            taxa_meta.insert(
                t.clone(),
                TaxonMeta {
                    label: format!("{} [Reference]", t),
                    taxon_type: "reference".to_string(),
                    color: "#64748b".to_string(), // Slate
                },
            );
        }
    }

    // 4. Detailed Breakpoint Records and Uncertainty Plateaus
    let mut breakpoints = Vec::with_capacity(scan_res.events.len());
    let mut total_plateau_width = 0usize;

    for (bp_i, ev) in scan_res.events.iter().enumerate() {
        let cand_idx = ev.candidate_idx;
        let home_idx = ev.home_idx.unwrap_or(0);
        let donor_idx_opt = ev.donor_idx;

        // Find flanking informative sites
        let mut left_flank = None;
        let mut right_flank = None;

        if let Some(donor_idx) = donor_idx_opt {
            // Scan left for home match (u < ev.u1)
            let search_start = ev.u1.saturating_sub(1);
            for u in (0..search_start).rev() {
                let rc = aln.get(cand_idx, u);
                let rh = aln.get(home_idx, u);
                let rd = aln.get(donor_idx, u);
                if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rh {
                    left_flank = Some(u + 1); // 1-indexed
                    break;
                }
            }
            // Scan right for donor match (u >= ev.u1 - 1)
            for u in search_start..l {
                let rc = aln.get(cand_idx, u);
                let rh = aln.get(home_idx, u);
                let rd = aln.get(donor_idx, u);
                if rc > 0 && rh > 0 && rd > 0 && rh != rd && rc == rd {
                    right_flank = Some(u + 1); // 1-indexed
                    break;
                }
            }
        }

        let ci_l = left_flank.unwrap_or_else(|| ev.u1.saturating_sub(5).max(1));
        let ci_r = right_flank.unwrap_or_else(|| (ev.u1 + 5).min(l));
        let plat_w = if ci_r >= ci_l { ci_r - ci_l } else { 0 };
        total_plateau_width += plat_w;

        let donor_display_name = if ev.is_ghost_donor {
            "Ghost (Unsampled)".to_string()
        } else {
            ev.donor_name.clone()
        };

        breakpoints.push(VisualizationBreakpoint {
            breakpoint_id: format!("BP_{}", bp_i + 1),
            recombinant: ev.candidate_name.clone(),
            parent_1: ev.home_name.clone(),
            parent_2: donor_display_name,
            breakpoint_nt: ev.u1,
            coarse_bp: ev.u1_continuous.round() as usize,
            ci_left: ci_l,
            ci_right: ci_r,
            plateau_width: plat_w,
            flanking_p1_site: left_flank,
            flanking_p2_site: right_flank,
            log_likelihood_gain: (ev.z_phys * 2.5).max(0.0),
            kinetic_z: ev.z_phys,
            l_pir: ev.d_tract_donor,
            p_fisher: ev.p_fisher,
            is_crossover: ev.is_crossover,
            is_ghost_donor: ev.is_ghost_donor,
        });
    }

    // 5. Build Mosaic Tracks for each taxon
    let mut mosaic_taxa = BTreeMap::new();
    for t in &sorted_taxa {
        let t_bps: Vec<&VisualizationBreakpoint> =
            breakpoints.iter().filter(|b| &b.recombinant == t).collect();

        if t_bps.is_empty() {
            let meta = taxa_meta.get(t).unwrap();
            let lineage = if meta.taxon_type == "parent" {
                format!("{} (Parental Reference)", t)
            } else {
                "Homologous".to_string()
            };
            mosaic_taxa.insert(
                t.clone(),
                vec![MosaicSegment {
                    start: 1,
                    end: l,
                    lineage,
                    color: meta.color.clone(),
                    is_plateau: None,
                }],
            );
        } else {
            let mut segments = Vec::new();
            let mut curr = 1usize;

            for b in &t_bps {
                let p1_col = parent_colors
                    .get(&b.parent_1)
                    .cloned()
                    .unwrap_or_else(|| "#0072b2".to_string());

                let cl = b.ci_left.max(curr);
                let cr = b.ci_right.min(l);

                if cl > curr {
                    segments.push(MosaicSegment {
                        start: curr,
                        end: cl - 1,
                        lineage: format!("Donor: {}", b.parent_1),
                        color: p1_col,
                        is_plateau: None,
                    });
                }

                segments.push(MosaicSegment {
                    start: cl,
                    end: cr,
                    lineage: format!("Plateau {} [Δ={} nt]", b.breakpoint_id, b.plateau_width),
                    color: "#fbbf24".to_string(), // Yellow plateau
                    is_plateau: Some(true),
                });
                curr = cr + 1;
            }

            if curr <= l {
                let last_b = t_bps.last().unwrap();
                let trailing_col = parent_colors
                    .get(&last_b.parent_2)
                    .cloned()
                    .unwrap_or_else(|| "#d55e00".to_string());
                segments.push(MosaicSegment {
                    start: curr,
                    end: l,
                    lineage: format!("Donor: {}", last_b.parent_2),
                    color: trailing_col,
                    is_plateau: None,
                });
            }

            mosaic_taxa.insert(t.clone(), segments);
        }
    }

    // 6. Macro genomic segments
    let mut macro_segments = Vec::new();
    if breakpoints.is_empty() {
        macro_segments.push(MacroSegment {
            name: "Segment 1".to_string(),
            start: 1,
            end: l,
            color: "#3b82f6".to_string(),
        });
    } else {
        let mut cutpoints: Vec<usize> = breakpoints.iter().map(|b| b.breakpoint_nt).collect();
        cutpoints.sort_unstable();
        cutpoints.dedup();

        let mut seg_starts = vec![1];
        for &cp in &cutpoints {
            seg_starts.push(cp + 1);
        }
        let mut seg_ends = cutpoints;
        seg_ends.push(l);

        let seg_colors = [
            "#3b82f6", "#0ea5e9", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#06b6d4",
        ];
        for (s_i, (&s_st, &s_en)) in seg_starts.iter().zip(seg_ends.iter()).enumerate() {
            if s_en >= s_st {
                macro_segments.push(MacroSegment {
                    name: format!("Segment {}", s_i + 1),
                    start: s_st,
                    end: s_en,
                    color: seg_colors[s_i % seg_colors.len()].to_string(),
                });
            }
        }
    }

    // 7. Focal Recombinant & Continuous Signals
    let (focal_cand_idx, focal_cand_name, focal_home_idx, focal_home_name, focal_donor_idx_opt, focal_donor_name) =
        if let Some(ev) = scan_res.events.first() {
            (
                ev.candidate_idx,
                ev.candidate_name.clone(),
                ev.home_idx.unwrap_or(0),
                ev.home_name.clone(),
                ev.donor_idx,
                ev.donor_name.clone(),
            )
        } else {
            (
                0,
                aln.taxa[0].clone(),
                if n > 1 { 1 } else { 0 },
                if n > 1 { aln.taxa[1].clone() } else { aln.taxa[0].clone() },
                if n > 2 { Some(2) } else { None },
                if n > 2 { aln.taxa[2].clone() } else { "None".to_string() },
            )
        };

    let focal_donor_idx = focal_donor_idx_opt.unwrap_or_else(|| if n > 2 { 2 } else { 0 });

    // Compute site differences for focal candidate vs home and donor
    let mut diff_p1 = vec![0i32; l];
    let mut diff_p2 = vec![0i32; l];
    let mut informative_snps = Vec::new();
    let mut walk_x = Vec::new();
    let mut walk_s = Vec::new();
    let mut current_walk = 0i32;

    for u in 0..l {
        let cr = aln.get(focal_cand_idx, u);
        let c1 = aln.get(focal_home_idx, u);
        let c2 = aln.get(focal_donor_idx, u);

        if cr > 0 && c1 > 0 && c2 > 0 {
            if cr != c1 {
                diff_p1[u] = 1;
            }
            if cr != c2 {
                diff_p2[u] = 1;
            }
            if c1 != c2 {
                let (match_type, walk_step) = if cr == c1 {
                    ("match_p1", -1)
                } else if cr == c2 {
                    ("match_p2", 1)
                } else {
                    ("divergent", 0)
                };

                let base_char = |b: u8| match b {
                    1 => 'A',
                    2 => 'C',
                    3 => 'G',
                    4 => 'T',
                    _ => '-',
                };

                informative_snps.push(InformativeSnpRecord {
                    pos: u + 1,
                    match_type: match_type.to_string(),
                    base_r: base_char(cr),
                    base_p1: base_char(c1),
                    base_p2: base_char(c2),
                });

                current_walk += walk_step;
                walk_x.push(u + 1);
                walk_s.push(current_walk);
            }
        }
    }

    // Cumulative sums
    let mut cum_p1 = vec![0i32; l + 1];
    let mut cum_p2 = vec![0i32; l + 1];
    for u in 0..l {
        cum_p1[u + 1] = cum_p1[u] + diff_p1[u];
        cum_p2[u + 1] = cum_p2[u] + diff_p2[u];
    }

    // Grid sampling (smooth 300-500 points for canvas rendering)
    let grid_step = (l / 400).max(5);
    let mut u_grid = Vec::new();
    let mut cur_u = 0usize;
    while cur_u <= l {
        u_grid.push(cur_u);
        cur_u += grid_step;
    }
    if *u_grid.last().unwrap() != l {
        u_grid.push(l);
    }

    // Ensure all breakpoint coordinates are explicitly represented in u_grid
    for b in &breakpoints {
        if !u_grid.contains(&b.breakpoint_nt) {
            u_grid.push(b.breakpoint_nt);
        }
    }
    u_grid.sort_unstable();
    u_grid.dedup();

    let mut trajectory_x = Vec::with_capacity(u_grid.len());
    let mut trajectory_y = Vec::with_capacity(u_grid.len());
    let mut running_max = 0.0f64;
    let mut drawdown = Vec::with_capacity(u_grid.len());
    let mut brownian_bridge = Vec::with_capacity(u_grid.len());

    let y_final = (cum_p2[l] - cum_p1[l]) as f64;

    for &u in &u_grid {
        let y = if u == 0 {
            0.0
        } else {
            (cum_p2[u] - cum_p1[u]) as f64
        };
        trajectory_x.push(u);
        trajectory_y.push(y);

        if y > running_max {
            running_max = y;
        }
        drawdown.push(running_max - y);

        let bridge = y - (u as f64 / l.max(1) as f64) * y_final;
        brownian_bridge.push(bridge);
    }

    // Instantaneous numerical velocity dy
    let mut velocity_dy = Vec::with_capacity(trajectory_y.len());
    let m = trajectory_y.len();
    for i in 0..m {
        if m < 2 {
            velocity_dy.push(0.0);
        } else if i == 0 {
            let du = (trajectory_x[1] - trajectory_x[0]).max(1) as f64;
            velocity_dy.push((trajectory_y[1] - trajectory_y[0]) / du);
        } else if i == m - 1 {
            let du = (trajectory_x[m - 1] - trajectory_x[m - 2]).max(1) as f64;
            velocity_dy.push((trajectory_y[m - 1] - trajectory_y[m - 2]) / du);
        } else {
            let du = (trajectory_x[i + 1] - trajectory_x[i - 1]).max(1) as f64;
            velocity_dy.push((trajectory_y[i + 1] - trajectory_y[i - 1]) / du);
        }
    }

    let signals = VisualizationSignals {
        trajectory_x,
        trajectory_y,
        velocity_dy,
        informative_snps,
        walk_x,
        walk_s,
        drawdown,
        brownian_bridge,
    };

    // 8. 2D Classical MDS Metric Subspaces
    let focal_bp_nt = breakpoints
        .first()
        .map(|b| b.breakpoint_nt)
        .unwrap_or(l / 2);

    let p1_end = focal_bp_nt.saturating_sub(1).max(1);
    let p2_start = focal_bp_nt.min(l);

    let coords_p1 = compute_mds_2d(aln, 0, p1_end);
    let coords_p2 = compute_mds_2d(aln, p2_start.saturating_sub(1), l);

    let subspaces = VisualizationSubspaces {
        partition_1: coords_p1,
        partition_2: coords_p2,
        p1_title: format!("Partition 1 (nt 1 – {})", p1_end),
        p2_title: format!("Partition 2 (nt {} – {})", p2_start, l),
    };

    // 9. Time-invariant Canonical Metric Manifold & Chord Colinearity
    let canonical_manifold = compute_canonical_manifold(
        aln,
        focal_cand_idx,
        focal_home_idx,
        focal_donor_idx,
    );

    // 10. Summary Metadata
    let mean_plat_w = if !breakpoints.is_empty() {
        (total_plateau_width as f64 / breakpoints.len() as f64 * 10.0).round() / 10.0
    } else {
        0.0
    };

    let max_support = scan_res
        .events
        .iter()
        .map(|e| e.z_phys)
        .fold(0.0f64, |a, b| a.max(b));

    let metadata = VisualizationMetadata {
        title,
        alignment_length: l,
        taxa_count: n,
        query_id: focal_cand_name,
        p1_id: focal_home_name,
        p2_id: focal_donor_name,
        recombinants_count: rec_taxa_set.len(),
        breakpoints_count: breakpoints.len(),
        informative_snps_count: scan_res.informative_sites,
        mean_plateau_width: mean_plat_w,
        max_support: (max_support * 100.0).round() / 100.0,
        run_time_ms: scan_res.run_time_ms,
        is_non_recombinant_h0: scan_res.is_non_recombinant_h0,
    };

    VisualizationDossier {
        metadata,
        taxa: sorted_taxa,
        taxa_meta,
        macro_segments,
        breakpoints,
        mosaic_taxa,
        signals,
        subspaces,
        canonical_manifold: Some(canonical_manifold),
        sequences,
    }
}

/// Computes 2D Classical MDS coordinates for all taxa over a sliced partition [start_site, end_site)
fn compute_mds_2d(
    aln: &Alignment,
    start_site: usize,
    end_site: usize,
) -> BTreeMap<String, Vec<f64>> {
    let n = aln.num_taxa;
    let s_end = end_site.min(aln.length);
    let s_len = s_end.saturating_sub(start_site);

    let mut res = BTreeMap::new();
    if n < 2 || s_len == 0 {
        for t in &aln.taxa {
            res.insert(t.clone(), vec![0.0, 0.0]);
        }
        return res;
    }

    // Pairwise Hamming distance matrix
    let mut d_sq = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = 0usize;
            let mut valid = 0usize;
            for u in start_site..s_end {
                let ci = aln.get(i, u);
                let cj = aln.get(j, u);
                if ci > 0 && cj > 0 {
                    valid += 1;
                    if ci != cj {
                        diff += 1;
                    }
                }
            }
            let dist = if valid > 0 {
                diff as f64 / valid as f64
            } else {
                0.0
            };
            let d2 = dist * dist;
            d_sq[i * n + j] = d2;
            d_sq[j * n + i] = d2;
        }
    }

    // Double centering: B = -0.5 * H @ D_sq @ H
    let mut row_means = vec![0.0f64; n];
    let mut total_mean = 0.0f64;
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += d_sq[i * n + j];
        }
        row_means[i] = sum / (n as f64);
        total_mean += sum;
    }
    total_mean /= (n * n) as f64;

    let mut b = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (d_sq[i * n + j] - row_means[i] - row_means[j] + total_mean);
        }
    }

    let eigh = symmetric_jacobi_eigh(&b, n);
    let lam0 = eigh.eigenvalues[0].max(0.0).sqrt();
    let lam1 = if n > 1 {
        eigh.eigenvalues[1].max(0.0).sqrt()
    } else {
        0.0
    };

    for i in 0..n {
        let x = eigh.eigenvectors[i * n] * lam0;
        let y = if n > 1 {
            eigh.eigenvectors[i * n + 1] * lam1
        } else {
            0.0
        };
        res.insert(
            aln.taxa[i].clone(),
            vec![(x * 10000.0).round() / 10000.0, (y * 10000.0).round() / 10000.0],
        );
    }

    res
}

/// Computes whole-gene time-invariant Canonical Metric Manifold (R^2),
/// out-of-sample continuous Gower trajectory, and chord colinearity cos(theta).
fn compute_canonical_manifold(
    aln: &Alignment,
    cand_idx: usize,
    home_idx: usize,
    donor_idx: usize,
) -> CanonicalManifold {
    let n = aln.num_taxa;
    let l = aln.length;

    // 1. Whole-alignment pairwise distance matrix D_glob
    let mut d_glob_sq = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = 0usize;
            let mut valid = 0usize;
            for u in 0..l {
                let ci = aln.get(i, u);
                let cj = aln.get(j, u);
                if ci > 0 && cj > 0 {
                    valid += 1;
                    if ci != cj {
                        diff += 1;
                    }
                }
            }
            let dist = if valid > 0 {
                diff as f64 / valid as f64
            } else {
                0.0
            };
            let d2 = dist * dist;
            d_glob_sq[i * n + j] = d2;
            d_glob_sq[j * n + i] = d2;
        }
    }

    // Double centering: B = -0.5 * H @ D_glob_sq @ H
    let mut row_means = vec![0.0f64; n];
    let mut total_mean = 0.0f64;
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += d_glob_sq[i * n + j];
        }
        row_means[i] = sum / (n as f64);
        total_mean += sum;
    }
    total_mean /= (n * n) as f64;

    let mut b = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -0.5 * (d_glob_sq[i * n + j] - row_means[i] - row_means[j] + total_mean);
        }
    }

    let eigh = symmetric_jacobi_eigh(&b, n);
    let k_eff = n.min(2);
    let lam0 = eigh.eigenvalues[0].max(1e-9);
    let lam1 = if k_eff > 1 {
        eigh.eigenvalues[1].max(1e-9)
    } else {
        1e-9
    };

    let sqrt_lam0 = lam0.sqrt();
    let sqrt_lam1 = lam1.sqrt();

    let mut ref_coords = BTreeMap::new();
    for i in 0..n {
        let x = eigh.eigenvectors[i * n] * sqrt_lam0;
        let y = if k_eff > 1 {
            eigh.eigenvectors[i * n + 1] * sqrt_lam1
        } else {
            0.0
        };
        ref_coords.insert(
            aln.taxa[i].clone(),
            vec![(x * 10000.0).round() / 10000.0, (y * 10000.0).round() / 10000.0],
        );
    }

    let home_coord = ref_coords
        .get(&aln.taxa[home_idx])
        .cloned()
        .unwrap_or_else(|| vec![0.0, 0.0]);
    let donor_coord = ref_coords
        .get(&aln.taxa[donor_idx])
        .cloned()
        .unwrap_or_else(|| vec![0.1, 0.0]);

    let chord_dx = donor_coord[0] - home_coord[0];
    let chord_dy = donor_coord[1] - home_coord[1];
    let chord_vector = vec![chord_dx, chord_dy];

    // 2. Continuous Out-of-sample Gower Trajectory
    // Slide a local window of length w = min(300, max(50, L / 20)) across alignment
    let win = (l / 20).clamp(50, 300);
    let step = (win / 4).max(5);
    let inv_sqrt_lam0 = 1.0 / sqrt_lam0;
    let inv_sqrt_lam1 = 1.0 / sqrt_lam1;

    let mut gower_trajectory = Vec::new();

    let mut p = 0usize;
    while p + win <= l {
        let u_mid = p + win / 2;

        // Compute local distance vector d_loc^2 from cand_idx to all taxa
        let mut d_loc_sq = vec![0.0f64; n];
        for j in 0..n {
            let mut diff = 0usize;
            let mut valid = 0usize;
            for u in p..(p + win) {
                let cr = aln.get(cand_idx, u);
                let cj = aln.get(j, u);
                if cr > 0 && cj > 0 {
                    valid += 1;
                    if cr != cj {
                        diff += 1;
                    }
                }
            }
            let d = if valid > 0 {
                diff as f64 / valid as f64
            } else {
                0.0
            };
            d_loc_sq[j] = d * d;
        }

        // Exact Gower projection: z = 0.5 * Lambda^{-1/2} * V^T * (row_means - d_loc_sq)
        let mut sum0 = 0.0f64;
        let mut sum1 = 0.0f64;
        for j in 0..n {
            let delta = row_means[j] - d_loc_sq[j];
            sum0 += eigh.eigenvectors[j * n] * delta;
            if k_eff > 1 {
                sum1 += eigh.eigenvectors[j * n + 1] * delta;
            }
        }
        let z_x = 0.5 * inv_sqrt_lam0 * sum0;
        let z_y = 0.5 * inv_sqrt_lam1 * sum1;

        gower_trajectory.push(GowerPoint {
            x: (z_x * 10000.0).round() / 10000.0,
            y: (z_y * 10000.0).round() / 10000.0,
            u: u_mid,
        });

        p += step;
    }

    // 3. Directional Chord Colinearity cos(theta)
    let (cos_theta, theta_deg, is_colinear) = if gower_trajectory.len() >= 2 {
        let first_pt = gower_trajectory.first().unwrap();
        let last_pt = gower_trajectory.last().unwrap();
        let traj_dx = last_pt.x - first_pt.x;
        let traj_dy = last_pt.y - first_pt.y;

        let dot = traj_dx * chord_dx + traj_dy * chord_dy;
        let norm_traj = (traj_dx * traj_dx + traj_dy * traj_dy).sqrt();
        let norm_chord = (chord_dx * chord_dx + chord_dy * chord_dy).sqrt();

        if norm_traj > 1e-6 && norm_chord > 1e-6 {
            let cos_th = (dot / (norm_traj * norm_chord)).clamp(-1.0, 1.0);
            let th_rad = cos_th.acos();
            let th_deg = th_rad * 180.0 / std::f64::consts::PI;
            (cos_th, th_deg, cos_th >= 0.40)
        } else {
            (0.0, 90.0, false)
        }
    } else {
        (1.0, 0.0, true)
    };

    CanonicalManifold {
        ref_coords,
        gower_trajectory,
        home_coord,
        donor_coord,
        chord_vector,
        cos_theta: (cos_theta * 1000.0).round() / 1000.0,
        theta_deg: (theta_deg * 10.0).round() / 10.0,
        is_colinear,
    }
}

/// Renders a standalone, fully self-contained HTML interactive dashboard with embedded JSON data.
pub fn generate_standalone_html(dossier: &VisualizationDossier) -> Result<String, String> {
    let json_str = serde_json::to_string(dossier)
        .map_err(|e| format!("Failed to serialize VisualizationDossier to JSON: {}", e))?;

    let html = DASHBOARD_HTML_TEMPLATE
        .replace("__RHIZAEON_TITLE__", &dossier.metadata.title)
        .replace("__RHIZAEON_DATA_JSON__", &json_str);

    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RhizAeonEngine;
    use crate::fasta::parse_fasta;

    #[test]
    fn test_visualization_dossier_generation() {
        let fasta = "\
>R
AAAAAAAAGGGGGGGG
>P1
AAAAAAAAAAAAAAAA
>P2
GGGGGGGGGGGGGGGG
>O
TTTTTTTTTTTTTTTT
";
        let aln = parse_fasta(fasta).unwrap();
        let engine = RhizAeonEngine::new();
        let scan_res = engine.scan(&aln);

        let dossier = generate_visualization_dossier(&aln, &scan_res, Some("Test Recombination"));

        assert_eq!(dossier.metadata.taxa_count, 4);
        assert_eq!(dossier.metadata.alignment_length, 16);
        assert_eq!(dossier.taxa.len(), 4);
        assert_eq!(dossier.sequences.len(), 4);
        assert!(dossier.canonical_manifold.is_some());

        let html = generate_standalone_html(&dossier).unwrap();
        assert!(html.contains("<title>Test Recombination</title>"));
        assert!(html.contains("<script id=\"rhizaeon-data\" type=\"application/json\">"));
    }
}

