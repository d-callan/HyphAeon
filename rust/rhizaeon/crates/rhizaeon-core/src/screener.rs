use crate::types::{Alignment, LandmarkSet, ScreeningResult};

pub struct LocalFluxScreener {
    pub root_ratio_threshold: f64,
    pub spatial_coherence_threshold: f64,
    pub coherence_window: usize,
    pub variance_multiplier: f64,
    pub epsilon: f64,
}

impl Default for LocalFluxScreener {
    fn default() -> Self {
        Self {
            root_ratio_threshold: 0.50,
            spatial_coherence_threshold: 0.30,
            coherence_window: 50,
            variance_multiplier: 1.80,
            epsilon: 1e-12,
        }
    }
}

impl LocalFluxScreener {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates spatial coherence of the Root flux stream
    fn compute_spatial_coherence(&self, f_tensor: &[f64], i: usize, l: usize, n: usize, num_channels: usize) -> f64 {
        let mut total_abs = 0.0f64;
        let w = self.coherence_window.min(l);
        if w == 0 {
            return 0.0;
        }

        // Circular buffer avoids allocating full L-length vector on heap
        let mut window_buf = vec![0.0f64; w];
        let mut current_sum = 0.0f64;
        let mut max_window_sum = 0.0f64;

        for u in 0..l {
            let val = f_tensor[u * n * num_channels + i * num_channels + 0];
            total_abs += val.abs();
            let pos_val = val.max(0.0);

            if u < w {
                window_buf[u] = pos_val;
                current_sum += pos_val;
                if current_sum > max_window_sum {
                    max_window_sum = current_sum;
                }
            } else {
                let slot = u % w;
                let old_val = window_buf[slot];
                window_buf[slot] = pos_val;
                current_sum += pos_val - old_val;
                if current_sum > max_window_sum {
                    max_window_sum = current_sum;
                }
            }
        }

        if total_abs < self.epsilon {
            0.0
        } else {
            max_window_sum / total_abs
        }
    }

    /// Computes Spectral Participation Ratio (PR) of contemporary channels for a candidate taxon.
    /// Measures effective dimensional complexity of contemporary flux to distinguish
    /// multidirectional heterotachy bursts (high PR) from low-rank reticulation dipoles (PR ~ 1).
    #[allow(dead_code)]
    fn compute_participation_ratio(
        &self,
        c_tensor: &[f64],
        i: usize,
        l: usize,
        n: usize,
        num_channels: usize,
        home_ch: Option<usize>,
        inv_l: f64,
    ) -> f64 {
        let mut active_channels = Vec::with_capacity(num_channels);
        for ch in 1..num_channels {
            if Some(ch) != home_ch {
                active_channels.push(ch);
            }
        }
        let m = active_channels.len();
        if m < 2 {
            return 1.0;
        }

        // 1. Single pass to compute channel means
        let mut sum_a = vec![0.0f64; m];
        for u in 0..l {
            let base_idx = u * n * num_channels + i * num_channels;
            for (a_idx, &cha) in active_channels.iter().enumerate() {
                sum_a[a_idx] += c_tensor[base_idx + cha];
            }
        }
        let mut means = vec![0.0f64; m];
        for a in 0..m {
            means[a] = sum_a[a] * inv_l;
        }

        // 2. Single pass to compute symmetric covariance matrix
        let mut cov = vec![0.0f64; m * m];
        let mut diff = vec![0.0f64; m];
        for u in 0..l {
            let base_idx = u * n * num_channels + i * num_channels;
            for (a_idx, &cha) in active_channels.iter().enumerate() {
                diff[a_idx] = c_tensor[base_idx + cha] - means[a_idx];
            }
            for a in 0..m {
                let da = diff[a];
                for b in a..m {
                    cov[a * m + b] += da * diff[b];
                }
            }
        }

        // Symmetrize and scale by 1/L
        for a in 0..m {
            cov[a * m + a] *= inv_l;
            for b in (a + 1)..m {
                let val = cov[a * m + b] * inv_l;
                cov[a * m + b] = val;
                cov[b * m + a] = val;
            }
        }

        let mut tr1 = 0.0f64;
        for a in 0..m {
            tr1 += cov[a * m + a];
        }

        let mut tr2 = 0.0f64;
        for a in 0..m {
            for b in 0..m {
                tr2 += cov[a * m + b] * cov[b * m + a];
            }
        }

        if tr2 > 1e-12 {
            (tr1 * tr1) / tr2
        } else {
            1.0
        }
    }

    /// Executes Pass 1 O(NL) Screening
    pub fn screen(
        &self,
        f_tensor: &[f64],
        c_tensor: &[f64],
        aln: &Alignment,
        landmarks: &LandmarkSet,
    ) -> ScreeningResult {
        let n = aln.num_taxa;
        let l = aln.length;
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;

        let mut v_root = vec![0.0f64; n];
        let mut v_cont = vec![0.0f64; n];
        let mut cont_pulse = vec![0.0f64; n];
        let mut r_root = vec![0.0f64; n];
        let mut r_dim = vec![0.0f64; n];
        let mut gamma_root = vec![0.0f64; n];

        let inv_l = 1.0 / (l as f64);

        // 1. Compute variances and peak-to-trough pulses in a single cache-friendly pass per taxon
        for i in 0..n {
            let mut sum_c = vec![0.0f64; num_channels];
            let mut sum_sq_c = vec![0.0f64; num_channels];
            let mut min_c = vec![f64::INFINITY; num_channels];
            let mut max_c = vec![f64::NEG_INFINITY; num_channels];

            for u in 0..l {
                let base_idx = u * n * num_channels + i * num_channels;
                for ch in 0..num_channels {
                    let c = c_tensor[base_idx + ch];
                    sum_c[ch] += c;
                    sum_sq_c[ch] += c * c;
                    if c < min_c[ch] { min_c[ch] = c; }
                    if c > max_c[ch] { max_c[ch] = c; }
                }
            }

            // Root channel 0
            let mean_c0 = sum_c[0] * inv_l;
            let var_c0 = (sum_sq_c[0] * inv_l - mean_c0 * mean_c0).max(0.0);
            v_root[i] = var_c0;

            // Contemporary channels 1..=K
            let mut sum_var_cont = 0.0;
            let mut max_pulse = 0.0f64;

            for ch in 1..num_channels {
                let mean_c = sum_c[ch] * inv_l;
                let var_c = (sum_sq_c[ch] * inv_l - mean_c * mean_c).max(0.0);
                sum_var_cont += var_c;

                let pulse = (max_c[ch] - min_c[ch]).max(0.0);
                if pulse > max_pulse {
                    max_pulse = pulse;
                }
            }

            v_cont[i] = sum_var_cont;
            cont_pulse[i] = max_pulse;

            let v_cont_mean = sum_var_cont / (k.max(1) as f64);
            r_root[i] = var_c0 / (sum_var_cont + var_c0 + self.epsilon);
            r_dim[i] = var_c0 / (v_cont_mean + var_c0 + self.epsilon);

            // Spatial coherence
            gamma_root[i] = self.compute_spatial_coherence(f_tensor, i, l, n, num_channels);
        }

        // 2. Heterotachy logit outlier screening
        let mut logit_r = Vec::with_capacity(n);
        let eps_logit = 1e-4;
        for &r in &r_root {
            let clamped = r.clamp(eps_logit, 1.0 - eps_logit);
            logit_r.push((clamped / (1.0 - clamped)).ln());
        }

        let med_logit = median(&logit_r);
        let mad_logit = median(&logit_r.iter().map(|&x| (x - med_logit).abs()).collect::<Vec<_>>());
        let scale_logit = (1.4826 * mad_logit).max(1.0 / (l as f64).sqrt());

        let z_alpha_r = 1.645f64; // norm.ppf(0.95)

        let mut rate_burst_indices = Vec::new();
        let mut ghost_candidate_indices = Vec::new();
        let mut is_rate_burst = vec![false; n];
        let mut is_ghost_candidate = vec![false; n];

        for i in 0..n {
            let z_r = (logit_r[i] - med_logit) / scale_logit;

            if z_r > z_alpha_r && gamma_root[i] <= self.spatial_coherence_threshold {
                is_rate_burst[i] = true;
                rate_burst_indices.push(i);
            } else if (z_r > z_alpha_r || r_dim[i] > self.root_ratio_threshold)
                && gamma_root[i] > self.spatial_coherence_threshold
            {
                is_ghost_candidate[i] = true;
                ghost_candidate_indices.push(i);
            }
        }

        // 3. Pool of non-burst taxa
        let mut pool_indices: Vec<usize> = (0..n).filter(|&i| !is_rate_burst[i]).collect();
        if pool_indices.is_empty() {
            pool_indices = (0..n).collect();
        }

        let pool_v: Vec<f64> = pool_indices.iter().map(|&i| v_cont[i]).collect();
        let pool_pulse: Vec<f64> = pool_indices.iter().map(|&i| cont_pulse[i]).collect();

        let med_v = median(&pool_v);
        let med_pulse = median(&pool_pulse);

        let lower_v: Vec<f64> = pool_v.iter().cloned().filter(|&x| x <= med_v).collect();
        let lower_pulse: Vec<f64> = pool_pulse.iter().cloned().filter(|&x| x <= med_pulse).collect();

        let mad_v = if !lower_v.is_empty() {
            let med_low = median(&lower_v);
            median(&lower_v.iter().map(|&x| (x - med_low).abs()).collect::<Vec<_>>())
        } else {
            0.0
        };
        let mad_pulse = if !lower_pulse.is_empty() {
            let med_low = median(&lower_pulse);
            median(&lower_pulse.iter().map(|&x| (x - med_low).abs()).collect::<Vec<_>>())
        } else {
            0.0
        };

        // Gumbel extreme value critical threshold
        let alpha_fwer = 0.05;
        let z_crit_ev = normal_ppf(1.0 - alpha_fwer / (n as f64));

        let floor_v = med_v / (l as f64).sqrt();
        let floor_pulse = med_pulse / (l as f64).sqrt();

        let scale_v = (1.4826 * mad_v).max(floor_v);
        let scale_pulse = (1.4826 * mad_pulse).max(floor_pulse);

        let thresh_v = med_v + z_crit_ev * scale_v;
        let thresh_pulse = med_pulse + z_crit_ev * scale_pulse;

        let mut candidate_mask = vec![false; n];
        for &i in &pool_indices {
            if v_cont[i] > thresh_v || cont_pulse[i] > thresh_pulse || is_ghost_candidate[i] {
                candidate_mask[i] = true;
            }
        }

        // Requirement 1 (DIR-RHIZ-PAFF-012.7): Patristic-Distance Gated Screener Admission
        // Small-cohort high-sensitivity admission in Pass 1 must ONLY activate
        // when the alignment is genuinely in a low-divergence regime (d <= 0.035).
        let mut is_low_div = false;
        if k >= 2 {
            let mut l_dists = Vec::new();
            for p in 0..k {
                for q in (p + 1)..k {
                    l_dists.push(crate::landmarks::compute_pairwise_distance(
                        aln,
                        landmarks.indices[p],
                        landmarks.indices[q],
                    ));
                }
            }
            if !l_dists.is_empty() {
                is_low_div = median(&l_dists) <= 0.035;
            }
        }

        // Small-cohort dispersion admission (N <= 8, DIR-RHIZ-PAFF-012.6 / DIR-RHIZ-PAFF-012.7)
        if n <= 8 && is_low_div && !pool_indices.is_empty() {
            let max_v = pool_v.iter().cloned().fold(0.0f64, f64::max);
            let disp_ratio = max_v / med_v.max(1e-6);
            if disp_ratio <= 4.0 || pool_v.iter().all(|&x| x < 3.0) {
                for &i in &pool_indices {
                    if !is_rate_burst[i] {
                        candidate_mask[i] = true;
                    }
                }
            }
        }

        // Pervasive Recombination Admission Sieve (DIR-RHIZ-PAFF-013.3 Requirement 2)
        // In cohorts with pervasive recombination (e.g. viral swarms, potyviruses, CRFs),
        // the cohort-wide variance is orders of magnitude above the clonal floor (mean_v >= 50.0).
        // In this regime, median-based outlier thresholds suffer from breakdown point masking.
        // We admit all non-burst taxa whose local variance or pulse exceeds the clonal floor.
        if !pool_indices.is_empty() {
            let mean_v_cohort: f64 = pool_v.iter().sum::<f64>() / (pool_v.len().max(1) as f64);
            if mean_v_cohort >= 50.0 || med_v >= 50.0 {
                for &i in &pool_indices {
                    if !is_rate_burst[i] && (v_cont[i] >= 10.0 || cont_pulse[i] >= 15.0) {
                        candidate_mask[i] = true;
                    }
                }
            }
        }

        let mut candidate_recombinant_indices = Vec::new();
        let mut clonal_backbone_indices = Vec::new();

        for i in 0..n {
            if candidate_mask[i] {
                candidate_recombinant_indices.push(i);
            } else if !is_rate_burst[i] {
                clonal_backbone_indices.push(i);
            }
        }

        let is_non_recombinant_h0 = candidate_recombinant_indices.is_empty();

        ScreeningResult {
            v_cont,
            v_root,
            r_root,
            gamma_root,
            rate_burst_indices,
            ghost_candidate_indices,
            candidate_recombinant_indices,
            clonal_backbone_indices,
            median_v_cont: med_v,
            is_non_recombinant_h0,
        }
    }
}

fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

/// Machine-precision Acklam rational approximation of the standard normal percent point function (probit)
pub fn normal_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798529320e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}
