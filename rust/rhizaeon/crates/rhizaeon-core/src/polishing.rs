use crate::attribution::TrajectoryParentalPair;
use crate::screener::normal_ppf;
use crate::types::PhysicalTract;

pub struct CumulativeTrajectoryPolisher {
    pub min_excursion_floor: f64,
    pub poisson_floor: f64,
    pub min_tract_length: usize,
    pub dyadic_scales: Vec<usize>,
    pub alpha: f64,
}

impl Default for CumulativeTrajectoryPolisher {
    fn default() -> Self {
        Self {
            min_excursion_floor: 0.30,
            poisson_floor: 0.25,
            min_tract_length: 30,
            dyadic_scales: vec![30, 60, 120, 240, 300, 360, 480, 600, 800],
            alpha: 0.05,
        }
    }
}

impl CumulativeTrajectoryPolisher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recursive Binary Segmentation on differential flux X in interval [s, e] (1-indexed inclusive)
    fn recursive_binary_segmentation(
        &self,
        x: &[f64],
        s: usize,
        e: usize,
        eff_alpha: f64,
        sigma_x: f64,
    ) -> Vec<usize> {
        let n = e.saturating_sub(s) + 1;
        if n < 2 * self.min_tract_length {
            return Vec::new();
        }

        // Sub-slice X[s-1..e]
        let x_sub = &x[(s - 1)..e];
        let mut c_sub = Vec::with_capacity(n);
        let mut run_sum = 0.0;
        let mut sum_sq_sub = 0.0;
        for &val in x_sub {
            run_sum += val;
            c_sub.push(run_sum);
            sum_sq_sub += val * val;
        }
        let total_c = run_sum;

        let mean_sub = total_c / (n as f64);
        let sigma_local = ((sum_sq_sub / (n as f64)) - mean_sub * mean_sub).max(1e-12).sqrt();
        let sigma_eff = sigma_local.min(sigma_x);

        let mut b_sub = Vec::with_capacity(n);
        for u in 0..n {
            let bridge_val = c_sub[u] - ((u + 1) as f64 / (n as f64)) * total_c;
            b_sub.push(bridge_val);
        }

        let idx_start = self.min_tract_length - 1;
        let idx_end = n - self.min_tract_length;
        if idx_start >= idx_end {
            return Vec::new();
        }

        let mut max_abs = -1.0;
        let mut local_apex = idx_start;
        for u in idx_start..=idx_end {
            let abs_val = b_sub[u].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                local_apex = u;
            }
        }

        let z_local = max_abs / (sigma_eff * (n as f64).sqrt());
        let z_crit = (-0.5 * (eff_alpha / 2.0).max(1e-15).ln()).sqrt();

        if z_local < z_crit {
            return Vec::new();
        }

        let u_star = s + local_apex; // 1-indexed changepoint

        let mut left_cps = self.recursive_binary_segmentation(x, s, u_star, eff_alpha, sigma_x);
        let mut right_cps = self.recursive_binary_segmentation(x, u_star + 1, e, eff_alpha, sigma_x);

        let mut result = Vec::new();
        result.append(&mut left_cps);
        result.push(u_star);
        result.append(&mut right_cps);
        result
    }

    /// Multiscale Dyadic Matched Filter for localized cassettes (DIR-RHIZ-PAFF-012.5)
    fn dyadic_matched_filter(
        &self,
        _x: &[f64],
        cum_x: &[f64],
        l: usize,
        num_taxa: usize,
        sigma_x: f64,
    ) -> Vec<(usize, usize)> {
        let m_scales = self.dyadic_scales.len().max(1);
        let mut candidates = Vec::new();

        for &tau in &self.dyadic_scales {
            if tau >= l {
                continue;
            }

            // Spatial extreme-value family-wise correction: L/tau independent spatial blocks
            let n_blocks = ((l as f64) / (tau as f64)).max(1.0);
            let eff_alpha_tau = (self.alpha / (2.0 * (num_taxa as f64) * (m_scales as f64) * n_blocks)).max(1e-15);
            let z_crit_tau = normal_ppf(1.0 - eff_alpha_tau);

            let se_tau = sigma_x * (tau as f64).sqrt();
            let theta_tau = self.poisson_floor.max(z_crit_tau * se_tau);

            let mut max_boxcar = f64::NEG_INFINITY;
            let mut best_k = 0;

            for k in 0..=(l - tau) {
                // Integral over [k, k + tau]
                let sum_val = cum_x[k + tau] - cum_x[k];
                if sum_val > max_boxcar {
                    max_boxcar = sum_val;
                    best_k = k;
                }
            }

            if max_boxcar >= theta_tau {
                let s_disc = best_k + 1;
                let e_disc = best_k + tau;
                candidates.push((s_disc, e_disc));
            }
        }

        candidates
    }

    /// Deconvolves continuous recombination breakpoints for a candidate taxon
    pub fn deconvolve_candidate(
        &self,
        candidate_idx: usize,
        parental: &TrajectoryParentalPair,
        c_tensor: &[f64],
        l: usize,
        n: usize,
        num_channels: usize,
    ) -> Vec<PhysicalTract> {
        let home_ch = parental.home_channel;
        let donor_ch = parental.donor_channel;

        // 1. Instantaneous and cumulative differential flux streams: X(u) and Y(u)
        let mut x = Vec::with_capacity(l);
        let mut y = Vec::with_capacity(l);
        let mut cum_x = Vec::with_capacity(l + 1);
        cum_x.push(0.0);

        let mut prev_c = 0.0;
        let mut run_sum = 0.0;

        for u in 0..l {
            let offset = u * n * num_channels + candidate_idx * num_channels;
            let cd = c_tensor[offset + donor_ch];
            let ch = c_tensor[offset + home_ch];
            let diff_c = cd - ch;

            let inst_x = diff_c - prev_c;
            prev_c = diff_c;

            x.push(inst_x);
            y.push(diff_c);

            run_sum += inst_x;
            cum_x.push(run_sum);
        }

        let mut sum_sq = 0.0;
        let mut sum_val = 0.0;
        for &val in &x {
            sum_val += val;
            sum_sq += val * val;
        }
        let mean_x = sum_val / (l as f64);
        let sigma_x = ((sum_sq / (l as f64)) - mean_x * mean_x).max(1e-12).sqrt();

        // 2. Centered Brownian Bridge: B(u) = Y(u) - ((u+1)/L) * Y[-1]
        let y_end = y[l - 1];
        let mut b = Vec::with_capacity(l);
        let mut min_b = f64::INFINITY;
        let mut max_b = f64::NEG_INFINITY;

        for u in 0..l {
            let bridge_val = y[u] - ((u + 1) as f64 / (l as f64)) * y_end;
            if bridge_val < min_b { min_b = bridge_val; }
            if bridge_val > max_b { max_b = bridge_val; }
            b.push(bridge_val);
        }

        let delta_c_max = (max_b - min_b).max(0.0);
        let eff_alpha = (self.alpha / (2.0 * (n as f64))).max(1e-15);
        let z_crit = (-0.5 * (eff_alpha / 2.0).max(1e-15).ln()).sqrt();

        let floor_adaptive = self.poisson_floor.max(z_crit * sigma_x * (self.min_tract_length as f64).sqrt());
        let dyadic_hits = self.dyadic_matched_filter(&x, &cum_x, l, n, sigma_x);

        if delta_c_max < floor_adaptive && dyadic_hits.is_empty() {
            return Vec::new();
        }

        // 3. Recursive Binary Segmentation
        let mut changepoints = self.recursive_binary_segmentation(&x, 1, l, eff_alpha, sigma_x);
        changepoints.sort();
        changepoints.dedup();

        if changepoints.is_empty() && !dyadic_hits.is_empty() {
            for (s_d, e_d) in dyadic_hits {
                if s_d > 1 {
                    changepoints.push(s_d - 1);
                }
                if e_d < l {
                    changepoints.push(e_d);
                }
            }
            changepoints.sort();
            changepoints.dedup();
        }

        if changepoints.is_empty() {
            return Vec::new();
        }

        // 4. Form partition segments between changepoints and evaluate donor polarity
        let mut tau_bounds = Vec::new();
        tau_bounds.push(0);
        tau_bounds.extend_from_slice(&changepoints);
        tau_bounds.push(l);

        let mut segments = Vec::new();
        for k in 0..(tau_bounds.len() - 1) {
            let s_k = tau_bounds[k] + 1;
            let e_k = tau_bounds[k + 1];
            let seg_len = (e_k - s_k + 1) as f64;

            let sum_flux: f64 = x[(s_k - 1)..e_k].iter().sum();
            let mean_flux = sum_flux / seg_len;

            // One-sided z-test for donor polarity beyond background noise
            let is_donor_polarity = mean_flux > (1.645 * sigma_x / seg_len.sqrt());
            segments.push((s_k, e_k, is_donor_polarity));
        }

        // 5. Merge contiguous same-polarity segments
        let mut merged_segments: Vec<(usize, usize, bool)> = Vec::new();
        for (s_k, e_k, is_d) in segments {
            if let Some(last) = merged_segments.last_mut() {
                if last.2 == is_d {
                    last.1 = e_k;
                } else {
                    merged_segments.push((s_k, e_k, is_d));
                }
            } else {
                merged_segments.push((s_k, e_k, is_d));
            }
        }

        // Determine which polarity represents the localized donor introgression:
        // An introgression tract is the minority excursion departing from the clonal backbone.
        let mut total_len_true = 0;
        let mut total_len_false = 0;
        for (s, e, is_d) in &merged_segments {
            let len = e - s + 1;
            if *is_d {
                total_len_true += len;
            } else {
                total_len_false += len;
            }
        }

        let (target_polarity, actual_home, actual_donor) = if total_len_true <= total_len_false {
            (true, home_ch, donor_ch)
        } else {
            (false, donor_ch, home_ch)
        };

        let donor_segments: Vec<(usize, usize)> = merged_segments
            .into_iter()
            .filter(|&(_, _, is_d)| is_d == target_polarity)
            .map(|(s, e, _)| (s, e))
            .collect();

        if donor_segments.is_empty() {
            return Vec::new();
        }

        // 6. Build PhysicalTracts for surviving donor segments
        let mut tracts = Vec::new();

        for (s_disc, e_disc) in donor_segments {
            let tract_len = e_disc - s_disc + 1;
            if tract_len < self.min_tract_length {
                continue;
            }

            let is_crossover = (s_disc == 1) || (e_disc == l);

            let max_b_seg = b[(s_disc - 1)..e_disc].iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
            let b_peak = b[(s_disc - 1)..e_disc].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let b_trough = b[(s_disc - 1)..e_disc].iter().cloned().fold(f64::INFINITY, f64::min);
            let delta_c_seg = (b_peak - b_trough).max(0.0);

            let (z_phys, z_eff_crit, floor_req) = if is_crossover {
                let z = max_b_seg / (sigma_x * (l as f64).sqrt());
                (z, z_crit, floor_adaptive)
            } else {
                let w_span = (tract_len as f64).max(self.min_tract_length as f64);
                let delta_c_tract: f64 = x[(s_disc - 1)..e_disc].iter().sum();
                let z = (max_b_seg / (sigma_x * w_span.sqrt())).max(delta_c_tract / (sigma_x * w_span.sqrt()));

                let m_scales = self.dyadic_scales.len().max(1);
                let eff_alpha_dyadic = (self.alpha / (2.0 * (n as f64) * (m_scales as f64))).max(1e-15);
                let z_cr = normal_ppf(1.0 - eff_alpha_dyadic);
                (z, z_cr, self.poisson_floor)
            };

            // Newtonian Velocity Reversal Symmetry Breaking (DIR-RHIZ-PAFF-012.0)
            let (v_l, v_r) = if is_crossover {
                let u_turn = if s_disc == 1 && e_disc < l { e_disc } else { s_disc };
                let turn_idx = u_turn.clamp(1, l);
                let off_turn = (turn_idx - 1) * n * num_channels + candidate_idx * num_channels;
                let off_end = (l - 1) * n * num_channels + candidate_idx * num_channels;

                let l_span = turn_idx.max(1) as f64;
                let r_span = (l - turn_idx).max(1) as f64;

                let mut vl = Vec::with_capacity(num_channels.saturating_sub(1));
                let mut vr = Vec::with_capacity(num_channels.saturating_sub(1));
                for ch in 1..num_channels {
                    let c_turn = c_tensor[off_turn + ch];
                    let c_end = c_tensor[off_end + ch];
                    vl.push(c_turn / l_span);
                    vr.push((c_end - c_turn) / r_span);
                }
                (vl, vr)
            } else {
                let s_prev = s_disc.saturating_sub(2);
                let e_prev = e_disc - 1;

                let off_s = s_prev * n * num_channels + candidate_idx * num_channels;
                let off_e = e_prev * n * num_channels + candidate_idx * num_channels;
                let off_end = (l - 1) * n * num_channels + candidate_idx * num_channels;

                let tr_span = (e_disc - s_disc + 1).max(1) as f64;
                let fl_span = ((s_disc - 1) + (l - e_disc)).max(1) as f64;

                let mut vl = Vec::with_capacity(num_channels.saturating_sub(1));
                let mut vr = Vec::with_capacity(num_channels.saturating_sub(1));
                for ch in 1..num_channels {
                    let c_s = if s_disc > 1 { c_tensor[off_s + ch] } else { 0.0 };
                    let c_e = c_tensor[off_e + ch];
                    let c_end = c_tensor[off_end + ch];

                    let v_tr = (c_e - c_s) / tr_span;
                    let v_fl = (c_s + (c_end - c_e)) / fl_span;
                    vl.push(v_tr);
                    vr.push(v_fl);
                }
                (vl, vr)
            };

            let mut norm_l = 0.0;
            let mut norm_r = 0.0;
            let mut dot_lr = 0.0;
            for i in 0..v_l.len() {
                norm_l += v_l[i] * v_l[i];
                norm_r += v_r[i] * v_r[i];
                dot_lr += v_l[i] * v_r[i];
            }
            norm_l = norm_l.sqrt();
            norm_r = norm_r.sqrt();

            let cos_theta = if norm_l > 1e-12 && norm_r > 1e-12 {
                dot_lr / (norm_l * norm_r)
            } else {
                0.0
            };

            let is_phys_sig = z_phys >= z_eff_crit
                && (delta_c_seg >= floor_req || delta_c_max >= floor_adaptive)
                && cos_theta <= 0.20;

            if !is_phys_sig {
                continue;
            }

            tracts.push(PhysicalTract {
                candidate_idx,
                home_channel: actual_home,
                donor_channel: actual_donor,
                is_crossover,
                u1_discrete: s_disc,
                u2_discrete: e_disc,
                u1_continuous: s_disc as f64,
                u2_continuous: e_disc as f64,
                delta_c_max,
                z_phys,
                z_crit: z_eff_crit,
                is_physically_significant: true,
            });
        }

        tracts
    }
}

