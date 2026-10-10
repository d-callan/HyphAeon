//! CHIN: Cluster-to-Host Incidence and Network Estimator.
//!
//! Implements:
//! - Quadratic sampling law inversion: E_obs ≈ (1/2) * n * rho * R0 = (1/2) * rho^2 * N_act * R0.
//! - Closed-form MLE for composite transmission parameter theta = rho * R0 = 2 * E_obs / n.
//! - Inferred active transmitting population scale: N_act = R0 * n^2 / (2 * E_obs) with Poisson CIs.
//! - Inferred longitudinal surveillance sampling fraction: rho = 2 * E_obs / (n * R0).
//! - Borel branching process cluster size spectrum decay test.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::math::{linregress, norm_ppf};

/// Active transmitting population and surveillance coverage estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivePopulationEstimate {
    pub n_act: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub relative_error: f64,
    pub r0: f64,
}

/// Borel cluster size spectrum decay test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorelDecayResult {
    pub slope: f64,
    pub r_squared: f64,
    pub p_value: f64,
    pub n_clusters: usize,
    pub n_megaclusters: usize,
    pub max_cluster_size: usize,
    pub rejects_uniform_sampling: bool,
}

/// CHIN Macro-Epidemic Parameter Inference Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChinEstimator {
    pub r0_default: f64,
}

impl ChinEstimator {
    pub fn new(r0_default: f64) -> Self {
        Self {
            r0_default: r0_default.max(0.1),
        }
    }

    /// Computes MLE of composite parameter theta = rho * R0 = 2 * E_obs / n.
    pub fn estimate_composite_parameter(&self, n_samples: usize, n_edges: usize) -> f64 {
        if n_samples == 0 {
            return 0.0;
        }
        (2.0 * (n_edges as f64)) / (n_samples as f64)
    }

    /// Inverts quadratic law to estimate surveillance sampling fraction rho in [0, 1].
    pub fn estimate_sampling_fraction(&self, n_samples: usize, n_edges: usize, r0_opt: Option<f64>) -> f64 {
        let r0 = r0_opt.unwrap_or(self.r0_default);
        if r0 <= 0.0 {
            return 0.0;
        }
        let theta = self.estimate_composite_parameter(n_samples, n_edges);
        (theta / r0).max(0.0).min(1.0)
    }

    /// Inverts quadratic scaling law to infer active transmitting population scale N_act:
    /// N_act = (R0 * n^2) / (2 * E_obs).
    pub fn estimate_active_population(
        &self,
        n_samples: usize,
        n_edges: usize,
        r0_opt: Option<f64>,
        confidence_level: f64,
    ) -> ActivePopulationEstimate {
        let r0 = r0_opt.unwrap_or(self.r0_default);
        if n_edges == 0 || n_samples == 0 {
            return ActivePopulationEstimate {
                n_act: f64::INFINITY,
                ci_lower: f64::INFINITY,
                ci_upper: f64::INFINITY,
                relative_error: f64::INFINITY,
                r0,
            };
        }

        let n = n_samples as f64;
        let e = n_edges as f64;
        let n_act = (r0 * n * n) / (2.0 * e);

        let alpha = 1.0 - confidence_level;
        let z = norm_ppf(1.0 - alpha / 2.0);
        let rel_err = z / e.sqrt();

        let ci_lower = n_act * (-rel_err).exp();
        let ci_upper = n_act * rel_err.exp();

        ActivePopulationEstimate {
            n_act,
            ci_lower,
            ci_upper,
            relative_error: rel_err,
            r0,
        }
    }

    /// Tests whether cluster frequencies follow subcritical Borel branching process decay.
    pub fn test_borel_branching_decay(
        &self,
        cluster_sizes: &[usize],
        max_size_fit: usize,
    ) -> BorelDecayResult {
        let valid_sizes: Vec<usize> = cluster_sizes.iter().copied().filter(|&s| s >= 2).collect();
        if valid_sizes.is_empty() {
            return BorelDecayResult {
                slope: 0.0,
                r_squared: 0.0,
                p_value: 1.0,
                n_clusters: 0,
                n_megaclusters: 0,
                max_cluster_size: 0,
                rejects_uniform_sampling: false,
            };
        }

        let max_size = valid_sizes.iter().copied().max().unwrap_or(0);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &s in &valid_sizes {
            *counts.entry(s).or_insert(0) += 1;
        }

        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut n_megaclusters = 0usize;

        for (&size, &cnt) in &counts {
            if size > max_size_fit {
                n_megaclusters += 1;
            }
            if size <= max_size_fit {
                x_vals.push(size as f64);
                y_vals.push((cnt as f64).ln());
            }
        }

        let (slope, r_squared, p_value) = if x_vals.len() >= 3 {
            match linregress(&x_vals, &y_vals) {
                Some(lr) => (lr.slope, lr.r_squared, lr.p_value),
                None => (0.0, 0.0, 1.0),
            }
        } else {
            (0.0, 0.0, 1.0)
        };

        // Rejection if massive single-linkage percolation mega-clusters exist (size >= 50)
        let rejects = max_size >= 50 || (n_megaclusters >= 5 && r_squared < 0.85);

        BorelDecayResult {
            slope,
            r_squared,
            p_value,
            n_clusters: valid_sizes.len(),
            n_megaclusters,
            max_cluster_size: max_size,
            rejects_uniform_sampling: rejects,
        }
    }

    /// Executes full joint Bayesian Monte Carlo uncertainty integration (S draws)
    /// propagating parameter uncertainty across R0, superspreading overdispersion k,
    /// and partner-tracing enrichment phi under degree-inflated counting variance.
    pub fn run_joint_bayesian_monte_carlo(
        &self,
        n_samples: usize,
        n_edges: usize,
        n_draws: usize,
        r0_prior: (f64, f64),
        k_prior: (f64, f64),
        phi_prior: (f64, f64),
        seed: u64,
    ) -> Result<ChinBayesianResult, String> {
        if n_samples == 0 || n_edges == 0 {
            return Err("n_samples and n_edges must be positive".to_string());
        }

        let s = n_draws.max(100);
        let n_total = n_samples as f64;
        let e_obs = n_edges as f64;
        let theta = (2.0 * e_obs) / n_total;

        let mut rng = SimpleRng::new(seed);

        let mut rho_draws = Vec::with_capacity(s);
        let mut n_act_draws = Vec::with_capacity(s);
        let mut cycles_draws = Vec::with_capacity(s);
        let mut fano_draws = Vec::with_capacity(s);

        for _ in 0..s {
            let r0 = rng.next_uniform(r0_prior.0, r0_prior.1);
            let k = rng.next_uniform(k_prior.0, k_prior.1);
            let phi = rng.next_uniform(phi_prior.0, phi_prior.1);

            let fano = 1.0 + theta * (1.0 + 1.0 / k);
            fano_draws.push(fano);

            let e_std = (e_obs * fano).sqrt();
            let e_draw = rng.next_normal(e_obs, e_std).max(1.0);

            let rho = (2.0 * e_draw) / (n_total * r0 * (1.0 + phi));
            let n_act = (n_total * n_total * r0 * (1.0 + phi)) / (2.0 * e_draw);
            let cycles = 1.0 / rho.max(1e-12);

            rho_draws.push(rho * 100.0);
            n_act_draws.push(n_act);
            cycles_draws.push(cycles);
        }

        let summarize = |mut v: Vec<f64>| -> ChinPosteriorSummary {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ChinPosteriorSummary {
                median: percentile(&v, 50.0),
                iqr_low: percentile(&v, 25.0),
                iqr_high: percentile(&v, 75.0),
                ci_low: percentile(&v, 2.5),
                ci_high: percentile(&v, 97.5),
            }
        };

        fano_draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let fano_median = percentile(&fano_draws, 50.0);

        Ok(ChinBayesianResult {
            n_samples,
            n_edges,
            n_draws: s,
            theta_mle: theta,
            rho_pct: summarize(rho_draws),
            n_act: summarize(n_act_draws),
            cycles_g: summarize(cycles_draws),
            fano_factor_median: fano_median,
        })
    }

    /// Audits reporting quantization artifacts (e.g. integer or half-year reporting).
    pub fn detect_date_quantization(&self, dates: &[f64]) -> DateQuantizationResult {
        let valid: Vec<f64> = dates.iter().copied().filter(|d| d.is_finite()).collect();
        if valid.is_empty() {
            return DateQuantizationResult {
                fraction_quantized: 0.0,
                is_severely_quantized: false,
            };
        }

        let mut quantized_count = 0usize;
        for &d in &valid {
            let rem = (d - d.round()).abs();
            let mid_rem = (d - (d.floor() + 0.5)).abs();
            if rem < 0.005 || mid_rem < 0.005 {
                quantized_count += 1;
            }
        }

        let fraction_quantized = (quantized_count as f64) / (valid.len() as f64);
        DateQuantizationResult {
            fraction_quantized,
            is_severely_quantized: fraction_quantized > 0.80,
        }
    }
}

/// Quantile summary from Monte Carlo draws.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChinPosteriorSummary {
    pub median: f64,
    pub iqr_low: f64,
    pub iqr_high: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// Full joint Bayesian Monte Carlo inference result for CHIN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChinBayesianResult {
    pub n_samples: usize,
    pub n_edges: usize,
    pub n_draws: usize,
    pub theta_mle: f64,
    pub rho_pct: ChinPosteriorSummary,
    pub n_act: ChinPosteriorSummary,
    pub cycles_g: ChinPosteriorSummary,
    pub fano_factor_median: f64,
}

/// Date quantization detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateQuantizationResult {
    pub fraction_quantized: f64,
    pub is_severely_quantized: bool,
}

/// High-speed zero-dependency PCG32 pseudorandom generator.
pub struct SimpleRng {
    state: u64,
    inc: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: seed.wrapping_add(0x853c49e6748fea9b),
            inc: 1442695040888963407,
        };
        rng.next_u64();
        rng
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let oldstate = self.state;
        self.state = oldstate.wrapping_mul(6364136223846793005).wrapping_add(self.inc | 1);
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        let p = (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
        p as u64
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u32::MAX as f64 + 1.0)
    }

    #[inline]
    pub fn next_uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    #[inline]
    pub fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }
}

pub fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * ((sorted.len() - 1) as f64);
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    let frac = rank - (low as f64);
    if low >= sorted.len() {
        sorted[sorted.len() - 1]
    } else if high >= sorted.len() {
        sorted[low]
    } else {
        sorted[low] * (1.0 - frac) + sorted[high] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chin_bayesian_mcmc() {
        let chin = ChinEstimator::new(1.5);
        let res = chin.run_joint_bayesian_monte_carlo(
            1658,
            101,
            5000,
            (1.2, 2.0),
            (0.1, 0.5),
            (0.0, 0.40),
            42,
        ).expect("MCMC should succeed");

        assert_eq!(res.n_samples, 1658);
        assert_eq!(res.n_edges, 101);
        assert!(res.theta_mle > 0.0);
        assert!(res.n_act.median > 1000.0 && res.n_act.median < 100000.0);
        assert!(res.rho_pct.median > 0.0 && res.rho_pct.median <= 100.0);
        assert!(res.n_act.ci_low < res.n_act.median);
        assert!(res.n_act.median < res.n_act.ci_high);
    }

    #[test]
    fn test_date_quantization() {
        let chin = ChinEstimator::new(1.5);
        let quantized_dates = vec![2012.0, 2013.0, 2014.5, 2015.0, 2016.5];
        let res = chin.detect_date_quantization(&quantized_dates);
        assert_eq!(res.fraction_quantized, 1.0);
        assert!(res.is_severely_quantized);

        let continuous_dates = vec![2012.123, 2013.456, 2014.789, 2015.234];
        let res2 = chin.detect_date_quantization(&continuous_dates);
        assert_eq!(res2.fraction_quantized, 0.0);
        assert!(!res2.is_severely_quantized);
    }
}
