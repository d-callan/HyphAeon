//! STEVE: Spectral Temporal Evolutionary Velocity Engine & Recursive AutoClock.
//!
//! Canonical implementation for both `trace50` (native pipeline) and
//! `chronaeon` (Python, via `aeon_core._native` bindings). Per the
//! canonical-semantics decision (2026-10-13, Sergei): where the two engines
//! historically diverged — Bartlett n_eff df penalty, operational tier
//! taxonomy, n<3 leaf fields, and the cohesive stopping criterion —
//! THIS implementation is the reference. Python's fallback code paths
//! remain for sdist installs without a Rust toolchain and may differ.
//!
//! Implements:
//! - Local adaptive bandwidth Gaussian affinity graphs.
//! - Normalized Graph Laplacian spectral partitioning via Jacobi diagonalization.
//! - Newman-Girvan modularity optimization across Fiedler Cheeger cuts.
//! - Bartlett effective sample size (N_eff) phylogenetic covariance penalty.
//! - Fieller's theorem exact confidence set inversion for evolutionary clock velocity and t_MRCA.

use aeon_manifold::{linregress, student_t_ppf, symmetric_jacobi_eigh};
use serde::{Deserialize, Serialize};

/// Sentinel AICc for unidentifiable fits (n < 3, zero time variance,
/// regression failure): astronomically penalized so downstream model
/// selection always rejects them. Matches the Python fit-dict contract
/// (`chronaeon.autoclock.fit_clock` emits 1e6 for these cases).
pub const AICC_UNIDENTIFIABLE: f64 = 1.0e6;

/// Evolutionary and epidemiological parameters of a resolved transmission
/// community. Doubles as the ClockFit record bound into Python: the dict
/// keys `n, mu, se, r2, rss, tot_var, aicc, tmrca, ci_mrca, fieller_status,
/// mean_dist, max_dist, span, d0` map to `size, mu, se_mu, r_squared, rss,
/// tot_var, aicc, tmrca, ci_mrca, fieller_status, mean_pairwise_dist,
/// max_pairwise_dist, time_span_years, d0` respectively.
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
    /// Residual sum of squares of the root-to-tip regression.
    pub rss: f64,
    /// Total variance of root-to-tip distances, sum((d - d_mean)^2).
    pub tot_var: f64,
    /// Hurvich-Tsai corrected AIC on the regression likelihood (p = 3),
    /// using the Bartlett-penalized sigma^2. `AICC_UNIDENTIFIABLE` when the
    /// fit cannot be identified.
    pub aicc: f64,
    pub tmrca: f64,
    pub ci_mrca: [f64; 2],
    pub fieller_status: String,
    pub mean_pairwise_dist: f64,
    /// Maximum nonzero intra-community pairwise distance (0 for empty).
    pub max_pairwise_dist: f64,
    /// Mean root-to-tip distance — the OLS intercept d0 at t_mean.
    pub d0: f64,
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
        return (
            [f64::NEG_INFINITY, f64::INFINITY],
            "RATE_UNIDENTIFIABLE_G_GE_1".to_string(),
        );
    }

    let denom = 1.0 - g;
    let center = d0 / mu;
    // discriminant = (d0^2 * g + denom * (t_crit * se_d0)^2) / mu^2, which is
    // non-negative whenever g < 1 and inputs are finite. COMPLEX_ROOTS is
    // therefore a defensive guard for non-finite inputs, not a reachable
    // geometry of the estimator.
    let discriminant = center * center - denom * ((d0 * d0 - (t_crit * se_d0).powi(2)) / (mu * mu));

    if discriminant < 0.0 {
        return (
            [f64::NEG_INFINITY, f64::INFINITY],
            "COMPLEX_ROOTS".to_string(),
        );
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
            // A pair without a distance (NaN: insufficient overlap, or absent
            // from a supplied matrix) contributes no covariance.
            if dist.is_finite() {
                sum_c += (-decay_param * dist).exp();
            }
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
    let max_d = nonzero_dists.iter().copied().fold(0.0, f64::max);

    let min_date = sub_dates.iter().copied().fold(f64::INFINITY, f64::min);
    let max_date = sub_dates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = if max_date >= min_date {
        max_date - min_date
    } else {
        0.0
    };

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
            rss: 0.0,
            tot_var: 0.0,
            aicc: AICC_UNIDENTIFIABLE,
            tmrca: min_date,
            ci_mrca: [min_date, min_date],
            fieller_status: "TOO_SMALL".to_string(),
            mean_pairwise_dist: mean_d,
            max_pairwise_dist: max_d,
            d0: 0.0,
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
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Root-to-tip regression over the members that have a distance to the
    // root isolate; pairs without one (NaN) carry no information here.
    let (reg_dates, root_dists): (Vec<f64>, Vec<f64>) = (0..n)
        .map(|i| (sub_dates[i], d_sub[earliest_local_idx * n + i]))
        .filter(|&(_, d)| d.is_finite())
        .unzip();
    let n_reg = reg_dates.len();

    // No member carries a finite distance to the root isolate (e.g. every
    // pair was below tn93's minimum overlap): the regression has no data.
    if n_reg == 0 {
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
            rss: 0.0,
            tot_var: 0.0,
            aicc: AICC_UNIDENTIFIABLE,
            tmrca: min_date,
            ci_mrca: [min_date, min_date],
            fieller_status: "REGRESSION_FAILED".to_string(),
            mean_pairwise_dist: mean_d,
            max_pairwise_dist: max_d,
            d0: 0.0,
            time_span_years: span,
            is_active_outbreak: false,
            operational_tier: "Indeterminate".to_string(),
            stop_reason: "reg_failure".to_string(),
        };
    }

    // d_mean (OLS intercept d0 at t_mean) and tot_var are computed before
    // the degenerate-time bail so the ZERO_TIME_VAR leaf still reports the
    // real residual mass rather than zeroed placeholders.
    let d_mean = root_dists.iter().sum::<f64>() / (n_reg as f64);
    let tot_var = root_dists
        .iter()
        .map(|&d| (d - d_mean).powi(2))
        .sum::<f64>();

    let t_mean = reg_dates.iter().sum::<f64>() / (n_reg as f64);
    let s_tt = reg_dates.iter().map(|&t| (t - t_mean).powi(2)).sum::<f64>();

    // Python contract (fit_clock): a degenerate time axis (var_x < 1e-12)
    // cannot identify a rate — bail before regression with ZERO_TIME_VAR
    // rather than extrapolating an unstable slope.
    if s_tt < 1e-12 {
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
            rss: tot_var,
            tot_var,
            aicc: AICC_UNIDENTIFIABLE,
            tmrca: min_date,
            ci_mrca: [min_date, min_date],
            fieller_status: "ZERO_TIME_VAR".to_string(),
            mean_pairwise_dist: mean_d,
            max_pairwise_dist: max_d,
            d0: d_mean,
            time_span_years: span,
            is_active_outbreak: false,
            operational_tier: "Indeterminate".to_string(),
            stop_reason: "zero_time_var".to_string(),
        };
    }

    let lr = match linregress(&reg_dates, &root_dists) {
        Some(res) => res,
        None => {
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
                rss: tot_var,
                tot_var,
                aicc: AICC_UNIDENTIFIABLE,
                tmrca: min_date,
                ci_mrca: [min_date, min_date],
                fieller_status: "REGRESSION_FAILED".to_string(),
                mean_pairwise_dist: mean_d,
                max_pairwise_dist: max_d,
                d0: d_mean,
                time_span_years: span,
                is_active_outbreak: false,
                operational_tier: "Indeterminate".to_string(),
                stop_reason: "reg_failure".to_string(),
            }
        }
    };

    let n_eff = compute_bartlett_neff(&d_sub, n, 100.0);
    let df_eff = (n_eff - 2.0).max(1.0);

    // OLS through the mean: RSS = TSS - slope^2 * S_tt.
    let rss = (tot_var - lr.slope * lr.slope * s_tt).max(0.0);
    let sig2 = rss / df_eff;
    let se_mu = if s_tt > 1e-12 {
        (sig2 / s_tt).sqrt()
    } else {
        0.0
    };
    let se_d0 = (sig2 / n_eff).sqrt();

    let t_crit = student_t_ppf(1.0 - alpha / 2.0, df_eff);
    let ci_mu = [lr.slope - t_crit * se_mu, lr.slope + t_crit * se_mu];

    let (mut ci_mrca, fieller_status) =
        compute_fieller_mrca(lr.slope, d_mean, se_mu, se_d0, t_mean, df_eff, alpha);
    // The MRCA cannot postdate the earliest sample: clamp the upper bound.
    if ci_mrca[1].is_finite() {
        ci_mrca[1] = ci_mrca[1].min(min_date);
    }

    // Gaussian-profile log-likelihood at the OLS optimum; p = 3
    // (intercept, slope, sigma^2). Uses the Bartlett-penalized sigma^2 —
    // the canonical n_eff convention for this crate.
    let loglik = -0.5 * (n as f64) * ((2.0 * std::f64::consts::PI * sig2).ln() + 1.0);
    let aicc = -2.0 * loglik + 2.0 * 3.0 + (2.0 * 3.0 * 4.0) / ((n as f64 - 4.0).max(1.0));

    // Python contract: only a strictly positive rate yields a tMRCA
    // estimate; a non-positive slope falls back to the earliest sample
    // date rather than extrapolating a future origin.
    let tmrca = if lr.slope > 1e-12 {
        (t_mean - d_mean / lr.slope).min(min_date)
    } else {
        min_date
    };

    let is_recent =
        tmrca >= (max_date - 6.0) || (ci_mrca[1].is_finite() && ci_mrca[1] >= (max_date - 4.0));
    let is_active =
        lr.slope >= 1.0e-3 && fieller_status == "OK" && lr.r_squared >= 0.15 && is_recent;

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
        rss,
        tot_var,
        aicc,
        tmrca,
        ci_mrca,
        fieller_status,
        mean_pairwise_dist: mean_d,
        max_pairwise_dist: max_d,
        d0: d_mean,
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
    sorted_indices.sort_by(|&a, &b| fiedler[a].total_cmp(&fiedler[b]));

    let mut best_q = -1e9;
    let mut best_k = 0;

    // Children must each have >= min_size members, matching the Python
    // contract (chronaeon.autoclock requires len(c0), len(c1) >= min_size
    // or it refuses the bisection).
    let min_k = min_size;
    let max_k = n.saturating_sub(min_size);

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

        let q =
            (1.0 / two_m) * ((a00 - (k0_deg * k0_deg) / two_m) + (a11 - (k1_deg * k1_deg) / two_m));
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

    // Adaptive bandwidth: 15th percentile of local subcommunity distances.
    // Uses the canonical aeon-manifold percentile (numpy-linear
    // interpolation), matching np.percentile(nonzero_d, 15) exactly.
    nonzero.sort_by(|a, b| a.total_cmp(b));
    let sigma = aeon_manifold::percentile(&nonzero, 15.0).max(0.005);

    // Build affinity graph W
    let mut w = vec![0.0; n * n];
    let mut degrees = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let d = d_matrix[sub_idx[i] * n_total + sub_idx[j]];
                // No distance (NaN) means no affinity, not an undefined one.
                let aff = if d.is_finite() {
                    (-d * d / (2.0 * sigma * sigma)).exp()
                } else {
                    0.0
                };
                w[i * n + j] = aff;
                degrees[i] += aff;
            }
        }
    }

    // Python contract: only abort when EVERY node is disconnected;
    // isolated taxa get d_inv_sqrt = 0 and stay in the spectrum as
    // eigenvalue-1 singletons instead of killing the recursion.
    if degrees.iter().all(|&d| d <= 1e-12) {
        fit.stop_reason = "disconnected_graph".to_string();
        return vec![fit];
    }

    // Normalized symmetric Laplacian L_sym = I - D^-1/2 W D^-1/2
    let d_inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let mut l_sym = vec![0.0; n * n];
    for i in 0..n {
        l_sym[i * n + i] = 1.0;
        for j in 0..n {
            if i != j {
                l_sym[i * n + j] = -d_inv_sqrt[i] * w[i * n + j] * d_inv_sqrt[j];
            }
        }
    }

    // aeon_manifold::symmetric_jacobi_eigh returns eigenvalues in DESCENDING
    // order, so the Fiedler pair (2nd-smallest eigenvalue) is column n - 2.
    let eigh = symmetric_jacobi_eigh(&l_sym, n);
    if eigh.eigenvalues.len() < 2 {
        fit.stop_reason = "trivial_spectrum".to_string();
        return vec![fit];
    }

    // Fiedler vector is the column of evecs matching the 2nd-smallest eigenvalue
    let fiedler_col = eigh.eigenvalues.len() - 2;

    // Homogeneity gate (Python: evals[1] > 0.999): a uniformly connected
    // blob has lambda_2 ~ n/(n-1) and carries no community structure.
    if eigh.eigenvalues[fiedler_col] > 0.999 {
        fit.stop_reason = "homogeneous_spectral_manifold".to_string();
        return vec![fit];
    }

    let mut fiedler = vec![0.0; n];
    for i in 0..n {
        fiedler[i] = eigh.eigenvectors[i * n + fiedler_col];
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
        &left_sub,
        dates,
        d_matrix,
        n_total,
        min_size,
        max_depth,
        depth + 1,
        &format!("{}.0", path),
        conductance_cut,
    );
    let right_leaves = recursive_autoclock_deconvolution(
        &right_sub,
        dates,
        d_matrix,
        n_total,
        min_size,
        max_depth,
        depth + 1,
        &format!("{}.1", path),
        conductance_cut,
    );

    left_leaves.extend(right_leaves);
    left_leaves
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Near-perfect linear clock: root-to-tip distance grows linearly with
    /// sampling date, plus a small constant off-diagonal offset so pairwise
    /// distances are nonzero.
    fn clock_fixture(n: usize, mu: f64, t0: f64, span: f64) -> (Vec<f64>, Vec<f64>) {
        let dates: Vec<f64> = (0..n)
            .map(|i| t0 + span * (i as f64) / (n - 1) as f64)
            .collect();
        let dists: Vec<f64> = dates.iter().map(|&t| mu * (t - t0) + 0.001).collect();
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = (dists[i] - dists[j]).abs() + 0.0005;
            }
            d[i * n + i] = 0.0;
        }
        (dates, d)
    }

    #[test]
    fn clock_fit_reports_full_field_contract() {
        let n = 8;
        let (dates, d) = clock_fixture(n, 0.004, 2020.0, 3.0);
        let idx: Vec<usize> = (0..n).collect();
        let c = fit_clock_community(&idx, &dates, &d, n, 0.05);
        assert_eq!(c.size, n);
        assert!(c.mu > 0.0, "positive clock rate expected");
        assert!(c.r_squared > 0.95);
        assert!(c.d0 > 0.0);
        // OLS partition identity: tot_var = rss + explained variance.
        assert!(c.tot_var >= c.rss);
        assert!(c.aicc.is_finite() && c.aicc != AICC_UNIDENTIFIABLE);
        let expected_max: f64 = d.iter().copied().filter(|&x| x > 0.0).fold(0.0, f64::max);
        assert!((c.max_pairwise_dist - expected_max).abs() < 1e-12);
        assert_eq!(c.stop_reason, "cohesive");
    }

    #[test]
    fn dyad_leaf_uses_sentinel_aicc_and_real_geometry() {
        let n = 2;
        let dates = vec![2020.0, 2021.0];
        let d = vec![0.0, 0.01, 0.01, 0.0];
        let idx: Vec<usize> = (0..n).collect();
        let c = fit_clock_community(&idx, &dates, &d, n, 0.05);
        assert_eq!(c.fieller_status, "TOO_SMALL");
        assert_eq!(c.aicc, AICC_UNIDENTIFIABLE);
        // Canonical (Rust) semantics: dyads report real geometry, not zeros.
        assert!(c.mean_pairwise_dist > 0.0);
        assert!(c.time_span_years > 0.0);
        assert_eq!(c.operational_tier, "Contemporaneous Acute Dyad");
    }

    #[test]
    fn degenerate_time_axis_reports_residual_mass() {
        let n = 4;
        let dates = vec![2020.0; n];
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in i + 1..n {
                d[i * n + j] = 0.01;
                d[j * n + i] = 0.01;
            }
        }
        let idx: Vec<usize> = (0..n).collect();
        let c = fit_clock_community(&idx, &dates, &d, n, 0.05);
        assert_eq!(c.fieller_status, "ZERO_TIME_VAR");
        assert_eq!(c.aicc, AICC_UNIDENTIFIABLE);
        // rss == tot_var: with no temporal signal nothing is explained.
        assert!(c.tot_var > 0.0);
        assert!((c.rss - c.tot_var).abs() < 1e-12);
    }

    #[test]
    fn clock_fit_recovers_known_tmrca() {
        // Perfectly additive clock: pairwise distance = mu * |t_i - t_j|,
        // so the root-to-tip row is exactly mu * (t_i - t0) and OLS must
        // recover slope = mu, t_mrca = t0, r^2 = 1.
        let n = 10;
        let mu = 0.004;
        let t0 = 2020.0;
        let dates: Vec<f64> = (0..n)
            .map(|i| t0 + 4.0 * (i as f64) / (n - 1) as f64)
            .collect();
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = mu * (dates[i] - dates[j]).abs();
            }
        }
        let idx: Vec<usize> = (0..n).collect();
        let c = fit_clock_community(&idx, &dates, &d, n, 0.05);
        assert_eq!(c.fieller_status, "OK");
        assert!((c.mu - mu).abs() < 1e-9);
        assert!((c.r_squared - 1.0).abs() < 1e-9);
        assert!((c.tmrca - t0).abs() < 1e-9);
        // The MRCA must predate the earliest sample, and the CI upper
        // bound is clamped to it.
        assert!(c.tmrca <= 2020.0);
        assert!(c.ci_mrca[1] <= 2020.0 + 1e-9);
        // A strong recent clock should tier as an active outbreak.
        assert_eq!(c.operational_tier, "Active Outbreak Chain");
        assert!(c.is_active_outbreak);
    }

    // ---- Fieller confidence set ---------------------------------------

    #[test]
    fn fieller_rejects_unidentifiable_rate() {
        let (ci, status) = compute_fieller_mrca(0.0, 0.03, 0.001, 0.001, 2021.5, 10.0, 0.05);
        assert_eq!(status, "UNIDENTIFIABLE_RATE");
        assert!(ci.iter().all(|b| b.is_nan()));

        let (_, status) = compute_fieller_mrca(0.01, 0.03, 0.001, 0.001, 2021.5, 0.5, 0.05);
        assert_eq!(status, "UNIDENTIFIABLE_RATE");
    }

    #[test]
    fn fieller_unbounded_when_rate_swamped_by_noise() {
        // g = (t_crit * se_mu / mu)^2 >= 1 means the rate is
        // statistically indistinguishable from zero; the Fieller set is
        // the whole real line.
        let (ci, status) = compute_fieller_mrca(0.001, 0.03, 0.01, 0.001, 2021.5, 10.0, 0.05);
        assert_eq!(status, "RATE_UNIDENTIFIABLE_G_GE_1");
        assert_eq!(ci[0], f64::NEG_INFINITY);
        assert_eq!(ci[1], f64::INFINITY);
    }

    #[test]
    fn fieller_tight_interval_brackets_ratio_estimate() {
        // mu = 0.01/yr, d0 = 0.03 at t_ref = 2021.5 -> t_mrca = 2018.5.
        // Tiny standard errors collapse the Fieller set onto the
        // ratio-of-means estimate.
        let (ci, status) = compute_fieller_mrca(0.01, 0.03, 1e-4, 1e-4, 2021.5, 10.0, 0.05);
        assert_eq!(status, "OK");
        assert!(ci[0] <= 2018.5 && 2018.5 <= ci[1]);
        assert!(ci[1] - ci[0] < 0.5, "CI width {} too wide", ci[1] - ci[0]);
    }

    // ---- Bartlett effective sample size --------------------------------

    #[test]
    fn bartlett_neff_boundary_cases() {
        // n <= 2 returns n unconditionally.
        assert_eq!(compute_bartlett_neff(&[0.0; 4], 2, 1.0), 2.0);

        // A fully covarying clique (all distances 0, including the
        // diagonal) gives sum_c = n^2, so n_eff collapses to 1.
        assert_eq!(compute_bartlett_neff(&[0.0; 9], 3, 1.0), 1.0);

        // Distant pairs contribute ~zero covariance; n_eff saturates at n.
        // off-diag 1.0, decay 100 -> off-diag covariance e^-100 ~ 0,
        // sum_c ~ n (diagonal only) -> n_eff = n.
        let far = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        assert!((compute_bartlett_neff(&far, 3, 100.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn bartlett_neff_known_value_and_nan_exclusion() {
        // n = 3, off-diagonal distances 0.5, decay 1:
        // sum_c = 3 * e^0 + 6 * e^-0.5 -> n_eff = 9 / (3 + 6e^-0.5).
        let mut d = vec![0.5; 9];
        d[0] = 0.0;
        d[4] = 0.0;
        d[8] = 0.0;
        let expected = 9.0 / (3.0 + 6.0 * (-0.5f64).exp());
        assert!((compute_bartlett_neff(&d, 3, 1.0) - expected).abs() < 1e-12);

        // A NaN pair (insufficient overlap) contributes nothing: dropping
        // the (0,1) pair removes two terms from sum_c and raises n_eff.
        d[1] = f64::NAN;
        d[3] = f64::NAN;
        let expected_nan = 9.0 / (3.0 + 4.0 * (-0.5f64).exp());
        assert!((compute_bartlett_neff(&d, 3, 1.0) - expected_nan).abs() < 1e-12);
    }

    // ---- Fiedler modularity --------------------------------------------

    /// Two equally sized cliques (strong within weights, weak between)
    /// plus a step Fiedler vector separating them.
    fn two_block_graph(n_half: usize, w_in: f64, w_out: f64) -> (Vec<f64>, Vec<f64>, usize) {
        let n = 2 * n_half;
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w[i * n + j] = if (i < n_half) == (j < n_half) {
                        w_in
                    } else {
                        w_out
                    };
                }
            }
        }
        let fiedler: Vec<f64> = (0..n)
            .map(|i| if i < n_half { -0.5 } else { 0.5 })
            .collect();
        (w, fiedler, n)
    }

    #[test]
    fn fiedler_cut_recovers_two_blocks() {
        let (w, fiedler, n) = two_block_graph(4, 1.0, 0.01);
        let (mut c0, mut c1, q, delta_q) = optimize_fiedler_modularity(&w, &fiedler, n, 2);
        c0.sort_unstable();
        c1.sort_unstable();
        assert_eq!(c0, vec![0, 1, 2, 3]);
        assert_eq!(c1, vec![4, 5, 6, 7]);
        assert!(q > 0.0 && delta_q > 0.0);
    }

    #[test]
    fn fiedler_cut_respects_min_size_floor() {
        // n = 4 < 2 * min_size = 6: no legal bisection, so the community
        // is returned whole with Q = 0.
        let (w, fiedler, n) = two_block_graph(2, 1.0, 0.01);
        let (c0, c1, q, delta_q) = optimize_fiedler_modularity(&w, &fiedler, n, 3);
        assert_eq!(c0.len(), n);
        assert!(c1.is_empty());
        assert_eq!(q, 0.0);
        assert_eq!(delta_q, 0.0);
    }

    // ---- Recursive deconvolution ---------------------------------------

    /// Two cliques of `n_half` members each: intra-cluster pairwise
    /// distance `d_in`, cross-cluster `d_out`, spread-out dates.
    fn two_cluster_fixture(n_half: usize, d_in: f64, d_out: f64) -> (Vec<f64>, Vec<f64>) {
        let n = 2 * n_half;
        let dates: Vec<f64> = (0..n).map(|i| 2020.0 + 0.1 * (i % n_half) as f64).collect();
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    d[i * n + j] = if (i < n_half) == (j < n_half) {
                        d_in
                    } else {
                        d_out
                    };
                }
            }
        }
        (dates, d)
    }

    #[test]
    fn deconvolution_splits_two_separated_clusters() {
        let nh = 8;
        let n = 2 * nh;
        let (dates, d) = two_cluster_fixture(nh, 0.002, 0.4);
        let idx: Vec<usize> = (0..n).collect();
        // trace50 pipeline call signature: min_size 3, depth 5, cut 0.65.
        let leaves = recursive_autoclock_deconvolution(&idx, &dates, &d, n, 3, 5, 0, "root", 0.65);
        // Each child is internally uniform -> both stop at the
        // homogeneity gate, so exactly two leaves.
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].path, "root.0");
        assert_eq!(leaves[1].path, "root.1");
        let mut left = leaves[0].members.clone();
        let mut right = leaves[1].members.clone();
        left.sort_unstable();
        right.sort_unstable();
        let mut merged: Vec<usize> = left.iter().chain(right.iter()).copied().collect();
        merged.sort_unstable();
        assert_eq!(merged, (0..n).collect::<Vec<_>>());
        let first: Vec<usize> = (0..nh).collect();
        assert!(
            left == first || right == first,
            "partition did not recover the planted clusters"
        );
    }

    #[test]
    fn deconvolution_homogeneous_blob_stops() {
        // Uniform pairwise distances -> uniform affinities -> lambda_2 =
        // n/(n-1) > 0.999 -> the homogeneity gate fires before any cut.
        let (dates, d) = two_cluster_fixture(4, 0.005, 0.005);
        let idx: Vec<usize> = (0..8).collect();
        let leaves = recursive_autoclock_deconvolution(&idx, &dates, &d, 8, 3, 5, 0, "root", 0.65);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].stop_reason, "homogeneous_spectral_manifold");
    }

    #[test]
    fn deconvolution_stop_reasons() {
        // All-zero distances: no nonzero entries -> identical_submanifold.
        let (dates, d_zero) = two_cluster_fixture(4, 0.0, 0.0);
        let idx: Vec<usize> = (0..8).collect();
        let leaves =
            recursive_autoclock_deconvolution(&idx, &dates, &d_zero, 8, 3, 5, 0, "root", 0.65);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].stop_reason, "identical_submanifold");

        // max_depth 0 returns immediately, before any spectral work.
        let (dates2, d) = two_cluster_fixture(8, 0.002, 0.4);
        let idx2: Vec<usize> = (0..16).collect();
        let leaves2 =
            recursive_autoclock_deconvolution(&idx2, &dates2, &d, 16, 3, 0, 0, "root", 0.65);
        assert_eq!(leaves2.len(), 1);
        assert_eq!(leaves2[0].stop_reason, "max_depth");
    }
}
