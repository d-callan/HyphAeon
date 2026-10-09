//! DANNO: Dynamic Ancestral Negative-binomial Network Odds.
//!
//! Evaluates candidate transmission dyads under Kingman coalescent and molecular clock
//! models versus unlinked circulating background and temporal data pathologies.
//!
//! Implements:
//! - Exact closed-form continuous Poisson-coalescent marginal via the upper incomplete
//!   gamma function Q(K + 1, .) without discrete loops or numerical quadrature.
//! - Continuous Negative Binomial background divergence density supporting fractional K.
//! - Exact physical molecular clock adequacy testing handling fractional K to eliminate stagnant
//!   "Wo Fat" data pathologies (specimen duplicates, laboratory carryover contamination).
//! - Min-max robust Bayes Factor evaluation across clock rate and intra-host coalescent depth domains.
//! - Benjamini-Hochberg False Discovery Rate (FDR) control for acute transmission edge inference.

use serde::{Deserialize, Serialize};
use crate::math::{gammaincc, ln_gamma, poisson_cdf};

/// Candidate transmission dyad with Bayesian transmission odds and FDR calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmissionDyad {
    pub idx1: usize,
    pub idx2: usize,
    pub dist: f64,
    pub delta_t: f64,
    pub k_substitutions: f64,
    pub bayes_factor: f64,
    pub pep: f64,
    pub q_value: f64,
    pub p_adequacy: f64,
    pub is_clock_violation: bool,
    pub is_supported: bool,
    #[serde(default)]
    pub is_certified: bool,
}

/// DANNO Statistical Inference Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DannoEstimator {
    pub seq_len: usize,
    pub mu: f64,
    pub tau_bar: f64,
    pub omega: f64,
    pub t_span: f64,
    pub alpha_adequacy: f64,
    pub r_trans: f64,
    pub p_trans: f64,
    pub r_bg: f64,
    pub p_bg: f64,
}

impl DannoEstimator {
    /// Creates a new DANNO estimator with standard parameters.
    pub fn new(
        seq_len: usize,
        mu: f64,
        tau_bar: f64,
        omega: f64,
        t_span: f64,
        alpha_adequacy: f64,
        bg_mean_dist: f64,
        bg_var_dist: f64,
    ) -> Self {
        let seq_len = seq_len.max(1);
        let mu = mu.max(1e-6);
        let tau_bar = tau_bar.max(1e-3);
        let omega = omega.max(1e-3);
        let t_span = t_span.max(1.0);
        let alpha_adequacy = alpha_adequacy.max(1e-4);

        let r_trans = 1.0;
        let l_f = seq_len as f64;
        let p_trans = (2.0 * mu * l_f * tau_bar) / (2.0 * mu * l_f * tau_bar + 1.0);

        let mut estimator = Self {
            seq_len,
            mu,
            tau_bar,
            omega,
            t_span,
            alpha_adequacy,
            r_trans,
            p_trans,
            r_bg: 5.0,
            p_bg: 0.95,
        };

        estimator.fit_background(bg_mean_dist, bg_var_dist);
        estimator
    }

    /// Fits Negative Binomial background divergence parameters via moment-matching.
    pub fn fit_background(&mut self, bg_mean_dist: f64, bg_var_dist: f64) {
        let l_f = self.seq_len as f64;
        let bg_mean_k = (bg_mean_dist * l_f).max(1.0);
        let bg_var_k = (bg_var_dist * l_f * l_f).max(bg_mean_k + 1.0);

        let p_bg_val = 1.0 - (bg_mean_k / bg_var_k);
        if p_bg_val > 0.0 && p_bg_val < 1.0 {
            self.p_bg = p_bg_val;
            self.r_bg = (bg_mean_k * (1.0 - self.p_bg)) / self.p_bg;
        } else {
            self.p_bg = 0.95;
            self.r_bg = 5.0;
        }
    }

    /// Automatically self-calibrates empirical background divergence and observation span from data.
    pub fn calibrate_from_data(&mut self, d_matrix: &[f64], n: usize, dates: &[f64]) {
        // Calibrate observation duration span
        let valid_dates: Vec<f64> = dates.iter().copied().filter(|d| d.is_finite()).collect();
        if valid_dates.len() > 1 {
            let mut min_t = f64::INFINITY;
            let mut max_t = f64::NEG_INFINITY;
            for &t in &valid_dates {
                if t < min_t { min_t = t; }
                if t > max_t { max_t = t; }
            }
            let span = max_t - min_t;
            if span > 0.5 {
                self.t_span = span.max(1.0).min(30.0);
            }
        }

        // Calibrate background genetic distance distribution
        if n > 2 && d_matrix.len() >= n * n {
            let mut d_vals = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = d_matrix[i * n + j];
                    if d.is_finite() {
                        d_vals.push(d);
                    }
                }
            }

            if d_vals.len() >= 10 {
                let m = d_vals.len();
                let bg_d = if m < 500 {
                    d_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = d_vals[m / 2];
                    d_vals.into_iter().filter(|&d| d >= median).collect()
                } else {
                    d_vals
                };

                let mean_d = bg_d.iter().sum::<f64>() / (bg_d.len() as f64);
                let var_d = bg_d.iter().map(|&d| (d - mean_d).powi(2)).sum::<f64>() / (bg_d.len() as f64);
                self.fit_background(mean_d, var_d.max(1e-6));
            }
        }
    }

    /// Evaluates physical molecular clock adequacy:
    /// Under active viral replication at rate mu, expected minimum mutations along
    /// the unshared lineage over interval delta_t is lambda_min = mu * L * delta_t.
    #[inline]
    pub fn check_clock_adequacy(&self, k: f64, delta_t: f64, mu_opt: Option<f64>) -> (f64, bool) {
        let mu_val = mu_opt.unwrap_or(self.mu);
        let k_val = k.max(0.0);
        let delta_t = delta_t.max(0.0);
        let lam_min = mu_val * (self.seq_len as f64) * delta_t;
        let p_adequacy = poisson_cdf(k_val.floor() as usize, lam_min);
        (p_adequacy, p_adequacy >= self.alpha_adequacy)
    }

    /// Closed-form continuous Poisson-coalescent marginal P(K | Delta T, H_trans)
    /// via regularized upper incomplete gamma Q(s, x).
    #[inline]
    pub fn evaluate_trans_marginal(&self, k: f64, delta_t: f64, mu_val: f64, tau_val: f64) -> f64 {
        let k = k.max(0.0);
        let delta_t = delta_t.max(0.0);
        let l_f = self.seq_len as f64;

        let b = 2.0 * mu_val * l_f;
        let c = b + 1.0 / tau_val;
        let p_tr = (b * tau_val) / (b * tau_val + 1.0);

        if delta_t <= 1e-15 {
            return (1.0 - p_tr) * p_tr.powf(k);
        }

        let x = c * delta_t / 2.0;
        let s = k + 1.0;
        let q_val = gammaincc(s, x);

        if q_val > 0.0 {
            let log_core = delta_t / (2.0 * tau_val) + k * b.ln() - s * c.ln() - tau_val.ln();
            if log_core < 700.0 {
                (log_core.exp() * q_val).max(1e-300)
            } else {
                (log_core + q_val.ln()).exp().max(1e-300)
            }
        } else {
            1e-300
        }
    }

    /// Evaluates joint transmission likelihood P(K, Delta T | H_trans).
    #[inline]
    pub fn compute_transmission_likelihood(
        &self,
        k: f64,
        delta_t: f64,
        mu: Option<f64>,
        tau_bar: Option<f64>,
        omega: Option<f64>,
    ) -> f64 {
        let mu_val = mu.unwrap_or(self.mu);
        let tau_val = tau_bar.unwrap_or(self.tau_bar);
        let omega_val = omega.unwrap_or(self.omega);
        let delta_t = delta_t.max(0.0);

        let p_delta_t = (1.0 / omega_val) * (-delta_t / omega_val).exp();
        let p_k_given_dt = self.evaluate_trans_marginal(k, delta_t, mu_val, tau_val);
        p_k_given_dt * p_delta_t
    }

    /// Evaluates joint unlinked background likelihood P(K, Delta T | H_null).
    #[inline]
    pub fn compute_background_likelihood(&self, k: f64, delta_t: f64) -> f64 {
        let k = k.max(0.0);
        let delta_t = delta_t.max(0.0);

        // 1. Triangular temporal density for difference of two uniform arrivals on [0, T_span]
        let p_delta_t = if delta_t < self.t_span {
            ((2.0 / self.t_span) * (1.0 - delta_t / self.t_span)).max(1e-6 / self.t_span)
        } else {
            1e-6 / self.t_span
        };

        // 2. Continuous Negative Binomial background genetic likelihood
        let p_scipy = 1.0 - self.p_bg;
        let log_p_k = ln_gamma(k + self.r_bg)
            - ln_gamma(self.r_bg)
            - ln_gamma(k + 1.0)
            + self.r_bg * p_scipy.ln()
            + k * (1.0 - p_scipy).ln();

        let p_k = log_p_k.exp().max(1e-300);
        p_k * p_delta_t
    }

    /// Computes continuous Bayes Factor:
    /// BF = P(K, Delta T | H_trans) / P(K, Delta T | H_null).
    /// If the pair physically violates the molecular clock adequacy test, returns 0.0.
    pub fn compute_bayes_factor(&self, dist: f64, delta_t: f64) -> f64 {
        let k = (dist * (self.seq_len as f64)).max(0.0);
        let delta_t = delta_t.max(0.0);

        let (_, is_adequate) = self.check_clock_adequacy(k, delta_t, None);
        if !is_adequate {
            return 0.0;
        }

        let lik_trans = self.compute_transmission_likelihood(k, delta_t, None, None, None);
        let lik_bg = self.compute_background_likelihood(k, delta_t);

        if lik_bg > 0.0 {
            lik_trans / lik_bg
        } else {
            0.0
        }
    }

    /// Computes min-max robust Bayes Factor over biological uncertainty domain:
    /// BF_robust = inf_{(mu, tau_bar) in Theta} BF(d, Delta T | mu, tau_bar).
    pub fn compute_robust_bayes_factor(
        &self,
        dist: f64,
        delta_t: f64,
        mu_range: (f64, f64),
        tau_bar_range: (f64, f64),
        omega_opt: Option<f64>,
    ) -> f64 {
        let k = (dist * (self.seq_len as f64)).max(0.0);
        let delta_t = delta_t.max(0.0);
        let omega_val = omega_opt.unwrap_or(self.omega);

        let vertices = [
            (mu_range.0, tau_bar_range.0),
            (mu_range.0, tau_bar_range.1),
            (mu_range.1, tau_bar_range.0),
            (mu_range.1, tau_bar_range.1),
        ];

        let mut min_bf = f64::INFINITY;
        for &(mu_val, tau_val) in &vertices {
            let (_, is_adequate) = self.check_clock_adequacy(k, delta_t, Some(mu_val));
            if !is_adequate {
                return 0.0;
            }

            let lik_t = self.compute_transmission_likelihood(k, delta_t, Some(mu_val), Some(tau_val), Some(omega_val));
            let lik_bg = self.compute_background_likelihood(k, delta_t);
            let bf = if lik_bg > 0.0 { lik_t / lik_bg } else { 0.0 };
            if bf < min_bf {
                min_bf = bf;
            }
        }

        min_bf
    }

    /// "Book 'em, Danno!"
    /// Screens candidate dyads, evaluates continuous Bayes Factors, enforces molecular clock adequacy,
    /// and controls Bayesian FDR via monotonic Benjamini-Hochberg ranking.
    pub fn screen_dyads(
        &mut self,
        indices: &[usize],
        dates: &[f64],
        d_matrix: &[f64],
        n_total: usize,
        max_delta_t: f64,
        prior_odds: f64,
        fdr_threshold: f64,
        use_robust: bool,
    ) -> Vec<TransmissionDyad> {
        self.calibrate_from_data(d_matrix, n_total, dates);

        let n = indices.len();
        let mut candidates = Vec::new();

        // Dynamic transmission envelope: d_crit = (d0 + 2 * mu * dt) * 1.5
        // Approximation of 99th percentile of NegBin(r=1, p_trans)
        let d0 = (3.0 / (self.seq_len as f64)).max(0.005);

        for i in 0..n {
            let idx_i = indices[i];
            let t_i = dates[idx_i];

            for j in (i + 1)..n {
                let idx_j = indices[j];
                let t_j = dates[idx_j];

                let dt = (t_i - t_j).abs();
                if dt <= max_delta_t {
                    let d_val = d_matrix[idx_i * n_total + idx_j];
                    let d_crit = (d0 + 2.0 * self.mu * dt) * 1.5;

                    if d_val <= d_crit {
                        let k = (d_val * (self.seq_len as f64)).max(0.0);
                        let (p_adeq, is_adequate) = self.check_clock_adequacy(k, dt, None);

                        let bf = if is_adequate {
                            if use_robust {
                                self.compute_robust_bayes_factor(d_val, dt, (1.5e-3, 3.5e-3), (0.05, 3.0), None)
                            } else {
                                self.compute_bayes_factor(d_val, dt)
                            }
                        } else {
                            0.0
                        };

                        let pep = if bf > 0.0 {
                            let post_odds = prior_odds * bf;
                            1.0 / (1.0 + post_odds)
                        } else {
                            1.0
                        };

                        candidates.push(TransmissionDyad {
                            idx1: idx_i,
                            idx2: idx_j,
                            dist: d_val,
                            delta_t: dt,
                            k_substitutions: k,
                            bayes_factor: bf,
                            pep,
                            q_value: 1.0,
                            p_adequacy: p_adeq,
                            is_clock_violation: !is_adequate,
                            is_supported: false,
                            is_certified: false,
                        });
                    }
                }
            }
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        // Bayesian False Discovery Rate control: sort by PEP ascending
        candidates.sort_by(|a, b| a.pep.partial_cmp(&b.pep).unwrap_or(std::cmp::Ordering::Equal));

        let m_tot = candidates.len();
        let mut cum_pep = 0.0;
        for (rank, cand) in candidates.iter_mut().enumerate() {
            cum_pep += cand.pep;
            cand.q_value = (cum_pep / ((rank + 1) as f64)).min(1.0);
        }

        // Enforce monotonicity of q-values from back to front
        for i in (0..(m_tot - 1)).rev() {
            if candidates[i].q_value > candidates[i + 1].q_value {
                candidates[i].q_value = candidates[i + 1].q_value;
            }
        }

        // Mark supported transmission links
        for cand in &mut candidates {
            if cand.q_value <= fdr_threshold && !cand.is_clock_violation {
                cand.is_supported = true;
                cand.is_certified = true;
            }
        }

        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_danno_clock_adequacy() {
        let danno = DannoEstimator::new(1000, 2e-3, 1.0, 2.0, 20.0, 0.05, 0.065, 0.0004);
        // delta_t = 0 => lam_min = 0 => 0 mutations is adequate
        let (p, ok) = danno.check_clock_adequacy(0.0, 0.0, None);
        assert!(ok);
        assert_eq!(p, 1.0);

        // delta_t = 5 years => lam_min = 2e-3 * 1000 * 5 = 10 mutations
        // 0 mutations when 10 expected => Poisson CDF(0 | 10) = exp(-10) ≈ 4.5e-5 < 0.05 => NOT adequate
        let (p5, ok5) = danno.check_clock_adequacy(0.0, 5.0, None);
        assert!(!ok5);
        assert!(p5 < 1e-4);
    }

    #[test]
    fn test_danno_bayes_factor() {
        let danno = DannoEstimator::new(1000, 2e-3, 1.0, 2.0, 20.0, 0.05, 0.065, 0.0004);
        // Very close pair: d = 0.001 (1 substitution), delta_t = 0.2 years
        let bf = danno.compute_bayes_factor(0.001, 0.2);
        assert!(bf > 1000.0);

        // Clock violation: d = 0.0, delta_t = 5.0 years => BF = 0.0
        let bf_viol = danno.compute_bayes_factor(0.0, 5.0);
        assert_eq!(bf_viol, 0.0);
    }
}
