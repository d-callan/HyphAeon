use crate::jacobi::symmetric_jacobi_eigh;
use crate::types::{Alignment, LandmarkSet};

/// Computes normalized Hamming distance between two sequence rows, excluding non-informative / gap sites
pub fn compute_pairwise_distance(aln: &Alignment, i: usize, j: usize) -> f64 {
    let row_i = aln.row(i);
    let row_j = aln.row(j);
    let mut valid_count = 0usize;
    let mut diff_count = 0usize;

    for u in 0..aln.length {
        let ci = row_i[u];
        let cj = row_j[u];
        if ci > 0 && cj > 0 {
            valid_count += 1;
            if ci != cj {
                diff_count += 1;
            }
        }
    }

    if valid_count == 0 {
        0.0
    } else {
        (diff_count as f64) / (valid_count as f64)
    }
}

/// Jukes-Cantor distance correction: d = -3/4 * ln(1 - 4/3 * p)
#[inline(always)]
pub fn jukes_cantor_distance(p: f64) -> f64 {
    let clamped_p = p.min(0.7499).max(0.0);
    -0.75 * (1.0 - (4.0 / 3.0) * clamped_p).ln()
}

/// Computes empirical scale-invariant sister and clade distance thresholds from the alignment.
/// - d_sister: Poisson transmission drift floor, bounded below by 3 / L
/// - d_clade: 10th percentile of positive pairwise distances, bounded below by 5 / L
pub fn compute_scale_invariant_distance_thresholds(aln: &Alignment) -> (f64, f64) {
    let l_f64 = aln.length.max(1) as f64;
    let sister_floor = 0.008f64.max(3.0 / l_f64);
    let clade_floor = 0.015f64.max(5.0 / l_f64);

    let n = aln.num_taxa;
    if n <= 1 {
        return (sister_floor, clade_floor);
    }

    let total_pairs = (n * (n - 1)) / 2;
    let max_samples = 2048;
    let stride = (total_pairs / max_samples).max(1);

    let mut dists = Vec::with_capacity(total_pairs.min(max_samples) + 16);
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            count += 1;
            if count % stride == 0 {
                let d = compute_pairwise_distance(aln, i, j);
                if d > 0.0 {
                    dists.push(d);
                }
            }
        }
    }

    if dists.is_empty() {
        return (sister_floor, clade_floor);
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx_01 = (0.01 * (dists.len() as f64)).floor() as usize;
    let d_sister = dists[idx_01.min(dists.len() - 1)].max(sister_floor);

    let idx_10 = (0.10 * (dists.len() as f64)).floor() as usize;
    let d_clade = dists[idx_10.min(dists.len() - 1)]
        .max(clade_floor)
        .max(1.5 * d_sister);

    (d_sister, d_clade)
}

/// Builds Full-Cohort LandmarkSet when N <= 64 (zero-subsampling manifold tracking)
pub fn build_full_cohort_landmarks(alignment: &Alignment) -> LandmarkSet {
    let n = alignment.num_taxa;
    let selected_indices: Vec<usize> = (0..n).collect();
    let landmark_taxa: Vec<String> = alignment.taxa.clone();

    let mut d_sq = vec![0.0f64; n * n];
    for p in 0..n {
        for q in 0..n {
            let dist = if p == q {
                0.0
            } else {
                compute_pairwise_distance(alignment, p, q)
            };
            let jc = jukes_cantor_distance(dist);
            d_sq[p * n + q] = jc * jc;
        }
    }

    let mut row_means = vec![0.0f64; n];
    let mut total_sum = 0.0f64;
    for p in 0..n {
        let mut s = 0.0;
        for q in 0..n {
            s += d_sq[p * n + q];
        }
        row_means[p] = s / (n as f64);
        total_sum += s;
    }
    let grand_mean = total_sum / ((n * n) as f64);

    let mut b = vec![0.0f64; n * n];
    for p in 0..n {
        for q in 0..n {
            let val = -0.5 * (d_sq[p * n + q] - row_means[p] - row_means[q] + grand_mean);
            b[p * n + q] = val;
        }
    }

    let eigh = symmetric_jacobi_eigh(&b, n);
    let lam0 = eigh
        .eigenvalues
        .get(0)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();
    let lam1 = eigh
        .eigenvalues
        .get(1)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();

    let mut coords = vec![0.0f64; (n + 1) * 2];
    let mut root_to_tip = Vec::with_capacity(n);

    for p in 0..n {
        let x = eigh.eigenvectors[p * n + 0] * lam0;
        let y = if n > 1 {
            eigh.eigenvectors[p * n + 1] * lam1
        } else {
            0.0
        };

        let radius = (x * x + y * y).sqrt().max(1e-6);
        let mut angle = y.atan2(x);
        if angle < 0.0 {
            angle += 2.0 * std::f64::consts::PI;
        }

        root_to_tip.push(radius);

        let out_idx = (p + 1) * 2;
        coords[out_idx] = radius;
        coords[out_idx + 1] = angle;
    }

    LandmarkSet {
        indices: selected_indices,
        taxa: landmark_taxa,
        num_landmarks: n,
        coords,
        root_to_tip,
    }
}

/// Selects K landmarks and builds exact LandmarkSet with uncorrupted tree coordinates
pub fn select_landmarks(alignment: &Alignment, target_k: usize) -> LandmarkSet {
    select_landmarks_from_pool(alignment, target_k, None)
}

/// Selects K landmarks strictly from an allowed pool of indices (e.g. verified clonal backbone)
pub fn select_landmarks_from_pool(
    alignment: &Alignment,
    target_k: usize,
    pool_opt: Option<&[usize]>,
) -> LandmarkSet {
    let n = alignment.num_taxa;

    // Full-cohort resolution for small/medium cohorts when unconstrained
    if n <= 64 && pool_opt.is_none() {
        return build_full_cohort_landmarks(alignment);
    }

    let pool: Vec<usize> = match pool_opt {
        Some(p) if !p.is_empty() => p.to_vec(),
        _ => (0..n).collect(),
    };

    let pool_len = pool.len();
    let k = target_k.min(pool_len).max(1);

    // 1. Sequence 1: sequence in pool with fewest gaps/missing data
    let mut best_first_idx = pool[0];
    let mut max_valid = 0;
    for &i in &pool {
        let row = aln_row_valid_count(alignment, i);
        if row > max_valid {
            max_valid = row;
            best_first_idx = i;
        }
    }

    let mut selected_indices = Vec::with_capacity(k);
    selected_indices.push(best_first_idx);

    if k > 1 {
        let mut min_dists: Vec<f64> = pool
            .iter()
            .map(|&i| compute_pairwise_distance(alignment, i, best_first_idx))
            .collect();

        for _ in 1..k {
            let mut best_cand = pool[0];
            let mut max_min_d = -1.0;
            for (idx, &i) in pool.iter().enumerate() {
                if !selected_indices.contains(&i) && min_dists[idx] > max_min_d {
                    max_min_d = min_dists[idx];
                    best_cand = i;
                }
            }
            selected_indices.push(best_cand);

            for (idx, &i) in pool.iter().enumerate() {
                let d = compute_pairwise_distance(alignment, i, best_cand);
                if d < min_dists[idx] {
                    min_dists[idx] = d;
                }
            }
        }
    }

    let landmark_taxa: Vec<String> = selected_indices
        .iter()
        .map(|&i| alignment.taxa[i].clone())
        .collect();

    // 2. Compute Classical MDS on the K landmarks to get 2D polar tree coordinates
    // Pairwise distance matrix D_k
    let mut d_sq = vec![0.0f64; k * k];
    for p in 0..k {
        for q in 0..k {
            let p_idx = selected_indices[p];
            let q_idx = selected_indices[q];
            let dist = if p == q {
                0.0
            } else {
                compute_pairwise_distance(alignment, p_idx, q_idx)
            };
            let jc = jukes_cantor_distance(dist);
            d_sq[p * k + q] = jc * jc;
        }
    }

    // Double centering: B = -0.5 * H * D^2 * H
    // Row means and col means
    let mut row_means = vec![0.0f64; k];
    let mut total_sum = 0.0f64;
    for p in 0..k {
        let mut s = 0.0;
        for q in 0..k {
            s += d_sq[p * k + q];
        }
        row_means[p] = s / (k as f64);
        total_sum += s;
    }
    let grand_mean = total_sum / ((k * k) as f64);

    let mut b = vec![0.0f64; k * k];
    for p in 0..k {
        for q in 0..k {
            let val = -0.5 * (d_sq[p * k + q] - row_means[p] - row_means[q] + grand_mean);
            b[p * k + q] = val;
        }
    }

    let eigh = symmetric_jacobi_eigh(&b, k);

    // Embed into 2D: coord[p] = [ V[p, 0] * sqrt(max(lambda_0, 0)), V[p, 1] * sqrt(max(lambda_1, 0)) ]
    let lam0 = eigh
        .eigenvalues
        .get(0)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();
    let lam1 = eigh
        .eigenvalues
        .get(1)
        .copied()
        .unwrap_or(0.0)
        .max(1e-9)
        .sqrt();

    // Root is at index 0 with coords [0.0, 0.0]
    // Landmarks are at indices 1..=K with polar coords: [t (radius), theta (angle)]
    let mut coords = vec![0.0f64; (k + 1) * 2];
    let mut root_to_tip = Vec::with_capacity(k);

    for p in 0..k {
        let x = eigh.eigenvectors[p * k + 0] * lam0;
        let y = if k > 1 {
            eigh.eigenvectors[p * k + 1] * lam1
        } else {
            0.0
        };

        let radius = (x * x + y * y).sqrt().max(1e-6);
        let mut angle = y.atan2(x);
        if angle < 0.0 {
            angle += 2.0 * std::f64::consts::PI;
        }

        root_to_tip.push(radius);

        let out_idx = (p + 1) * 2;
        coords[out_idx] = radius;
        coords[out_idx + 1] = angle;
    }

    LandmarkSet {
        indices: selected_indices,
        taxa: landmark_taxa,
        num_landmarks: k,
        coords,
        root_to_tip,
    }
}

fn aln_row_valid_count(aln: &Alignment, i: usize) -> usize {
    let row = aln.row(i);
    row.iter().filter(|&&c| c > 0).count()
}
