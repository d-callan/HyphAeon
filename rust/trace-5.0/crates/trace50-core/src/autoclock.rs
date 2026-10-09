//! STEVE: Spectral Temporal Evolutionary Velocity Engine & Recursive AutoClock.
//!
//! Implements:
//! - Local adaptive bandwidth Gaussian affinity graphs.
//! - Normalized Graph Laplacian spectral partitioning via Jacobi diagonalization.
//! - Newman-Girvan modularity optimization across Fiedler Cheeger cuts.
//! - Bartlett effective sample size (N_eff) phylogenetic covariance penalty.
//! - Fieller's theorem exact confidence set inversion for evolutionary clock velocity and t_MRCA.

use serde::{Deserialize, Serialize};
use crate::math::{jacobi_eigensystem, linregress, student_t_ppf};

/// Evolutionary and epidemiological parameters of a resolved transmission community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteveCommunity {
    pub community_id: usize,
    pub path: String,
    pub members: Vec<usize>,
    pub size: usize,
    pub n_eff: f64,
    pub mu: f64,
    pub se_mu: f64,
    pub ci_mu: [f64; 2],
    pub r_squared: f64,
    pub tmrca: f64,
    pub ci_mrca: [f64; 2],
    pub fieller_status: String,
    pub mean_pairwise_dist: f64,
    pub time_span_years: f64,
    pub is_active_outbreak: bool,
    pub operational_tier: String,
    pub stop_reason: String,
}

/// Computes exact Fieller theorem confidence intervals for t_MRCA = t_ref - (d0 / mu).
pub fn compute_fieller_mrca(
    mu: f64,
    d0: f64,
    se_mu: f64,
    se_d0: f64,
    t_ref: f64,
    df: f64,
    alpha: f64,
) -> ([f64; 2], String) {
    if df < 1.0 || mu.abs() < 1e-12 {
        return ([f64::NAN, f64::NAN], "UNIDENTIFIABLE_RATE".to_string());
    }

    let t_crit = student_t_ppf(1.0 - alpha / 2.0, df);
    let g = ((t_crit * se_mu) / mu).powi(2);

    if g >= 1.0 {
        return ([f64::NEG_INFINITY, f64::INFINITY], "RATE_UNIDENTIFIABLE_G_GE_1".to_string());
    }

    let denom = 1.0 - g;
    let center = d0 / mu;
    let discriminant = center * center - denom * ((d0 * d0 - (t_crit * se_d0).powi(2)) / (mu * mu));

    if discriminant < 0.0 {
        return ([f64::NEG_INFINITY, f64::INFINITY], "COMPLEX_ROOTS".to_string());
    }

    let delta = discriminant.sqrt();
    let theta_lo = (center - delta) / denom;
    let theta_hi = (center + delta) / denom;

    let ci_lo = t_ref - theta_hi;
    let ci_hi = t_ref - theta_lo;

    ([ci_lo, ci_hi], "OK".to_string())
}

/// Computes Bartlett effective sample size N_eff = N^2 / (1^T C(lambda) 1).
pub fn compute_bartlett_neff(d_sub: &[f64], n: usize, decay_param: f64) -> f64 {
    if n <= 2 {
        return n as f64;
    }

    let mut sum_c = 0.0;
    for i in 0..n {
        for j in 0..n {
            let dist = d_sub[i * n + j];
            sum_c += (-decay_param * dist).exp();
        }
    }

    if sum_c <= 0.0 {
        return n as f64;
    }

    let n_f = n as f64;
    let n_eff = (n_f * n_f) / sum_c;
    n_eff.max(1.0).min(n_f)
}

/// Fits root-to-tip linear regression with Bartlett covariance and Fieller bounds.
pub fn fit_clock_community(
    sub_idx: &[usize],
    dates: &[f64],
    d_matrix: &[f64],
    n_total: usize,
    alpha: f64,
) -> SteveCommunity {
    let n = sub_idx.len();
    let sub_dates: Vec<f64> = sub_idx.iter().map(|&i| dates[i]).collect();

    // Extract submatrix
    let mut d_sub = vec![0.0; n * n];
    let mut nonzero_dists = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let d = d_matrix[sub_idx[i] * n_total + sub_idx[j]];
            d_sub[i * n + j] = d;
            if i < j && d > 0.0 {
                nonzero_dists.push(d);
            }
        }
    }

    let mean_d = if !nonzero_dists.is_empty() {
        nonzero_dists.iter().sum::<f64>() / (nonzero_dists.len() as f64)
    } else {
        0.0
    };

    let min_date = sub_dates.iter().copied().fold(f64::INFINITY, f64::min);
    let max_date = sub_dates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = if max_date >= min_date { max_date - min_date } else { 0.0 };

    if n < 3 {
        return SteveCommunity {
            community_id: 0,
            path: "".to_string(),
            members: sub_idx.to_vec(),
            size: n,
            n_eff: n as f64,
            mu: 0.0,
            se_mu: 0.0,
            ci_mu: [0.0, 0.0],
            r_squared: 0.0,
            tmrca: min_date,
            ci_mrca: [min_date, min_date],
            fieller_status: "TOO_SMALL".to_string(),
            mean_pairwise_dist: mean_d,
            time_span_years: span,
            is_active_outbreak: false,
            operational_tier: if n == 2 {
                "Contemporaneous Acute Dyad".to_string()
            } else {
                "Singleton Node".to_string()
            },
            stop_reason: "size_floor".to_string(),
        };
    }

    // Root-to-tip divergence measured from earliest sampled isolate.
    let earliest_local_idx = sub_dates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let root_dists: Vec<f64> = (0..n).map(|i| d_sub[earliest_local_idx * n + i]).collect();

    let lr = match linregress(&sub_dates, &root_dists) {
        Some(res) => res,
        None => return SteveCommunity {
            community_id: 0,
            path: "".to_string(),
            members: sub_idx.to_vec(),
            size: n,
            n_eff: n as f64,
            mu: 0.0,
            se_mu: 0.0,
            ci_mu: [0.0, 0.0],
            r_squared: 0.0,
            tmrca: min_date,
            ci_mrca: [min_date, min_date],
            fieller_status: "REGRESSION_FAILED".to_string(),
            mean_pairwise_dist: mean_d,
            time_span_years: span,
            is_active_outbreak: false,
            operational_tier: "Indeterminate".to_string(),
            stop_reason: "reg_failure".to_string(),
        },
    };

    let n_eff = compute_bartlett_neff(&d_sub, n, 100.0);
    let df_eff = (n_eff - 2.0).max(1.0);

    let t_mean = sub_dates.iter().sum::<f64>() / (n as f64);
    let d_mean = root_dists.iter().sum::<f64>() / (n as f64);

    let s_tt = sub_dates.iter().map(|&t| (t - t_mean).powi(2)).sum::<f64>();
    let rss = (root_dists.iter().map(|&d| (d - d_mean).powi(2)).sum::<f64>() - lr.slope * lr.slope * s_tt).max(0.0);
    let sig2 = rss / df_eff;
    let se_mu = if s_tt > 1e-12 { (sig2 / s_tt).sqrt() } else { 0.0 };
    let se_d0 = (sig2 / n_eff).sqrt();

    let t_crit = student_t_ppf(1.0 - alpha / 2.0, df_eff);
    let ci_mu = [lr.slope - t_crit * se_mu, lr.slope + t_crit * se_mu];

    let (ci_mrca, fieller_status) = compute_fieller_mrca(
        lr.slope,
        d_mean,
        se_mu,
        se_d0,
        t_mean,
        df_eff,
        alpha,
    );

    let tmrca = if lr.slope.abs() > 1e-12 {
        t_mean - d_mean / lr.slope
    } else {
        min_date
    };

    let is_recent = tmrca >= (max_date - 6.0) || (ci_mrca[1].is_finite() && ci_mrca[1] >= (max_date - 4.0));
    let is_active = lr.slope >= 1.0e-3
        && fieller_status == "OK"
        && lr.r_squared >= 0.15
        && is_recent;

    let operational_tier = if is_active {
        "Active Outbreak Chain".to_string()
    } else if lr.slope >= 5.0e-4 && fieller_status == "OK" && is_recent {
        "Emergent Seed Cluster".to_string()
    } else if lr.slope <= 0.0 || (span > 5.0 && lr.r_squared < 0.02) {
        "Endemic Extinct Lineage".to_string()
    } else {
        "Stationary Chronic Compartment".to_string()
    };

    SteveCommunity {
        community_id: 0,
        path: "".to_string(),
        members: sub_idx.to_vec(),
        size: n,
        n_eff,
        mu: lr.slope,
        se_mu,
        ci_mu,
        r_squared: lr.r_squared,
        tmrca,
        ci_mrca,
        fieller_status,
        mean_pairwise_dist: mean_d,
        time_span_years: span,
        is_active_outbreak: is_active,
        operational_tier,
        stop_reason: "cohesive".to_string(),
    }
}

/// Optimizes Newman-Girvan modularity along sorted Fiedler vector cut points.
pub fn optimize_fiedler_modularity(
    w: &[f64],
    fiedler: &[f64],
    n: usize,
    min_size: usize,
) -> (Vec<usize>, Vec<usize>, f64, f64) {
    if n < 2 * min_size {
        return ((0..n).collect(), Vec::new(), 0.0, 0.0);
    }

    let mut degrees = vec![0.0; n];
    let mut two_m = 0.0;
    for i in 0..n {
        for j in 0..n {
            let val = w[i * n + j];
            degrees[i] += val;
            two_m += val;
        }
    }

    if two_m <= 1e-12 {
        return ((0..n).collect(), Vec::new(), 0.0, 0.0);
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| fiedler[a].partial_cmp(&fiedler[b]).unwrap());

    let mut best_q = -1e9;
    let mut best_k = 0;

    let min_k = 1;
    let max_k = if n > 1 { n - 1 } else { 0 };

    for k in min_k..=max_k {
        let mut k0_deg = 0.0;
        let mut k1_deg = 0.0;
        for i in 0..k {
            k0_deg += degrees[sorted_indices[i]];
        }
        for i in k..n {
            k1_deg += degrees[sorted_indices[i]];
        }

        let mut a00 = 0.0;
        for i in 0..k {
            for j in 0..k {
                a00 += w[sorted_indices[i] * n + sorted_indices[j]];
            }
        }

        let mut a11 = 0.0;
        for i in k..n {
            for j in k..n {
                a11 += w[sorted_indices[i] * n + sorted_indices[j]];
            }
        }

        let q = (1.0 / two_m) * ((a00 - (k0_deg * k0_deg) / two_m) + (a11 - (k1_deg * k1_deg) / two_m));
        if q > best_q {
            best_q = q;
            best_k = k;
        }
    }

    if best_k == 0 {
        return ((0..n).collect(), Vec::new(), 0.0, 0.0);
    }

    let c0 = sorted_indices[0..best_k].to_vec();
    let c1 = sorted_indices[best_k..n].to_vec();
    let delta_q = best_q; // Q_base for single community is 0.0

    (c0, c1, best_q, delta_q)
}

/// Recursive AutoClock / STEVE spectral deconvolution engine.
pub fn recursive_autoclock_deconvolution(
    sub_idx: &[usize],
    dates: &[f64],
    d_matrix: &[f64],
    n_total: usize,
    min_size: usize,
    max_depth: usize,
    depth: usize,
    path: &str,
    conductance_cut: f64,
) -> Vec<SteveCommunity> {
    let n = sub_idx.len();

    let mut fit = fit_clock_community(sub_idx, dates, d_matrix, n_total, 0.05);
    fit.path = path.to_string();

    if n < 3 {
        fit.stop_reason = if n == 2 {
            "dyad_node".to_string()
        } else {
            "singleton_node".to_string()
        };
        return vec![fit];
    }

    if depth >= max_depth {
        fit.stop_reason = "max_depth".to_string();
        return vec![fit];
    }

    // Extract nonzero distances to compute local adaptive bandwidth (15th percentile)
    let mut nonzero = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = d_matrix[sub_idx[i] * n_total + sub_idx[j]];
            if d > 0.0 {
                nonzero.push(d);
            }
        }
    }

    if nonzero.is_empty() {
        fit.stop_reason = "identical_submanifold".to_string();
        return vec![fit];
    }

    nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p15_idx = ((nonzero.len() as f64) * 0.15) as usize;
    let sigma = nonzero[p15_idx.min(nonzero.len() - 1)].max(0.005);

    // Build affinity graph W
    let mut w = vec![0.0; n * n];
    let mut degrees = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let d = d_matrix[sub_idx[i] * n_total + sub_idx[j]];
                let aff = (-d * d / (2.0 * sigma * sigma)).exp();
                w[i * n + j] = aff;
                degrees[i] += aff;
            }
        }
    }

    for &deg in &degrees {
        if deg <= 1e-12 {
            fit.stop_reason = "disconnected_graph".to_string();
            return vec![fit];
        }
    }

    // Normalized symmetric Laplacian L_sym = I - D^-1/2 W D^-1/2
    let mut l_sym = vec![0.0; n * n];
    for i in 0..n {
        l_sym[i * n + i] = 1.0;
        let d_i_inv = 1.0 / degrees[i].sqrt();
        for j in 0..n {
            if i != j {
                let d_j_inv = 1.0 / degrees[j].sqrt();
                l_sym[i * n + j] = -d_i_inv * w[i * n + j] * d_j_inv;
            }
        }
    }

    let (evals, evecs) = jacobi_eigensystem(&l_sym, n, 200, 1e-10);
    if evals.len() < 2 {
        fit.stop_reason = "trivial_spectrum".to_string();
        return vec![fit];
    }

    // Fiedler vector is the 2nd column of evecs (corresponding to evals[1])
    let mut fiedler = vec![0.0; n];
    for i in 0..n {
        fiedler[i] = evecs[i * n + 1];
    }

    let (c0, c1, _best_q, delta_q) = optimize_fiedler_modularity(&w, &fiedler, n, min_size);
    if c0.is_empty() || c1.is_empty() {
        fit.stop_reason = "modularity_size_floor".to_string();
        return vec![fit];
    }

    // Calculate cut weight and Cheeger conductance
    let mut cut_weight = 0.0;
    for &i in &c0 {
        for &j in &c1 {
            cut_weight += w[i * n + j];
        }
    }

    let vol_c0: f64 = c0.iter().map(|&i| degrees[i]).sum();
    let vol_c1: f64 = c1.iter().map(|&i| degrees[i]).sum();
    let conductance = cut_weight / vol_c0.min(vol_c1).max(1e-12);

    // Stopping criteria: delta_q <= 0 or conductance > cut
    if delta_q <= 0.0 || conductance > conductance_cut {
        fit.stop_reason = format!("cohesive_bottleneck_h{:.2}", conductance);
        return vec![fit];
    }

    let left_sub: Vec<usize> = c0.iter().map(|&i| sub_idx[i]).collect();
    let right_sub: Vec<usize> = c1.iter().map(|&i| sub_idx[i]).collect();

    let mut left_leaves = recursive_autoclock_deconvolution(
        &left_sub, dates, d_matrix, n_total, min_size, max_depth, depth + 1, &format!("{}.0", path), conductance_cut,
    );
    let right_leaves = recursive_autoclock_deconvolution(
        &right_sub, dates, d_matrix, n_total, min_size, max_depth, depth + 1, &format!("{}.1", path), conductance_cut,
    );

    left_leaves.extend(right_leaves);
    left_leaves
}
