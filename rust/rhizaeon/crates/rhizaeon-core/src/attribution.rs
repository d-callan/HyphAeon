use crate::types::LandmarkSet;

#[derive(Debug, Clone)]
pub struct TrajectoryParentalPair {
    pub candidate_idx: usize,
    pub home_channel: usize,
    pub donor_channel: usize,
    pub home_landmark_idx: Option<usize>,
    pub donor_landmark_idx: Option<usize>,
    pub is_ghost_donor: bool,
    pub home_name: String,
    pub donor_name: String,
    pub home_global_idx: Option<usize>,
    pub donor_global_idx: Option<usize>,
    pub swing_home: f64,
    pub swing_donor: f64,
}

pub struct TrajectoryParentalResolver {
    pub min_root_swing: f64,
}

impl Default for TrajectoryParentalResolver {
    fn default() -> Self {
        Self { min_root_swing: 3.0 }
    }
}

impl TrajectoryParentalResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves active reticulation dipole (P_Home, P_Donor) for candidate R (backward-compatible)
    pub fn resolve_reticulation_channels(
        &self,
        candidate_idx: usize,
        c_tensor: &[f64],
        a_tensor: &[f64],
        l: usize,
        n: usize,
        landmarks: &LandmarkSet,
    ) -> TrajectoryParentalPair {
        let all_pairs = self.resolve_all_reticulation_channels(
            candidate_idx,
            c_tensor,
            a_tensor,
            l,
            n,
            landmarks,
        );
        all_pairs.into_iter().next().unwrap()
    }

    /// Resolves all significant reticulation dipoles (P_Home, P_Donor_k) for candidate R
    /// enabling Multi-Way Mosaicism resolution across distinct parental channels.
    pub fn resolve_all_reticulation_channels(
        &self,
        candidate_idx: usize,
        c_tensor: &[f64],
        a_tensor: &[f64],
        l: usize,
        n: usize,
        landmarks: &LandmarkSet,
    ) -> Vec<TrajectoryParentalPair> {
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;
        let inv_l = 1.0 / (l as f64);

        // 1. Sequence-wide mean attention for candidate
        let mut mean_a = vec![0.0f64; num_channels];
        for u in 0..l {
            let offset = u * n * num_channels + candidate_idx * num_channels;
            for ch in 0..num_channels {
                mean_a[ch] += a_tensor[offset + ch];
            }
        }
        for ch in 0..num_channels {
            mean_a[ch] *= inv_l;
        }

        // 2. Swings for each channel
        let mut swings = vec![0.0f64; num_channels];
        for ch in 0..num_channels {
            let mut min_c = f64::INFINITY;
            let mut max_c = f64::NEG_INFINITY;
            for u in 0..l {
                let c = c_tensor[u * n * num_channels + candidate_idx * num_channels + ch];
                if c < min_c { min_c = c; }
                if c > max_c { max_c = c; }
            }
            swings[ch] = (max_c - min_c).max(0.0);
        }

        // 3. Covariance matrix of C_R in shape (num_channels, num_channels)
        let mut cov_c = vec![0.0f64; num_channels * num_channels];
        for a in 0..num_channels {
            for b in 0..num_channels {
                let mut sum_ab = 0.0;
                let mut sum_a = 0.0;
                let mut sum_b = 0.0;
                for u in 0..l {
                    let ca = c_tensor[u * n * num_channels + candidate_idx * num_channels + a];
                    let cb = c_tensor[u * n * num_channels + candidate_idx * num_channels + b];
                    sum_a += ca;
                    sum_b += cb;
                    sum_ab += ca * cb;
                }
                cov_c[a * num_channels + b] = sum_ab * inv_l - (sum_a * inv_l) * (sum_b * inv_l);
            }
        }

        // 4. Eligible contemporary landmark channels
        let mut contemporary_channels = Vec::new();
        for p in 0..k {
            let l_global = landmarks.indices[p];
            if l_global != candidate_idx {
                contemporary_channels.push(p + 1);
            }
        }
        if contemporary_channels.is_empty() {
            contemporary_channels.push(1);
        }

        // 5. Find optimal primary pair across all contemporary pairs
        let mut best_score = -1.0;
        let mut best_pair = (contemporary_channels[0], contemporary_channels[0]);

        for i in 0..contemporary_channels.len() {
            let ch_a = contemporary_channels[i];
            let var_a = cov_c[ch_a * num_channels + ch_a].max(1e-12);

            for j in (i + 1)..contemporary_channels.len() {
                let ch_b = contemporary_channels[j];
                let var_b = cov_c[ch_b * num_channels + ch_b].max(1e-12);
                let cov_ab = cov_c[ch_a * num_channels + ch_b];

                let diff_var = (var_a + var_b - 2.0 * cov_ab).max(0.0);
                let rho_ab = (cov_ab / (var_a * var_b).sqrt()).clamp(-1.0, 1.0);
                let opposition = 1.0 - rho_ab;

                let max_mean_a = mean_a[ch_a].max(mean_a[ch_b]);
                let attention_support = max_mean_a / (max_mean_a + 0.1);

                let score = diff_var * opposition * attention_support;
                if score > best_score {
                    best_score = score;
                    best_pair = (ch_a, ch_b);
                }
            }
        }

        let (mut home_ch, primary_donor_ch) = if mean_a[best_pair.0] >= mean_a[best_pair.1] {
            (best_pair.0, best_pair.1)
        } else {
            (best_pair.1, best_pair.0)
        };

        // If there's an undisputed sequence-wide dominant backbone channel among contemporary channels:
        for &ch in &contemporary_channels {
            if mean_a[ch] > 1.3 * mean_a[home_ch] && swings[ch] < swings[home_ch] {
                home_ch = ch;
            }
        }

        // 6. Score candidate donors against home_ch
        struct CandidateDonor {
            channel: usize,
            is_ghost: bool,
            score: f64,
        }

        let mut candidate_donors = Vec::new();

        // Always include primary donor
        candidate_donors.push(CandidateDonor {
            channel: primary_donor_ch,
            is_ghost: false,
            score: best_score.max(1e-6),
        });

        // Test other contemporary channels against home_ch
        for &d in &contemporary_channels {
            if d == home_ch || d == primary_donor_ch {
                continue;
            }

            let var_h = cov_c[home_ch * num_channels + home_ch].max(1e-12);
            let var_d = cov_c[d * num_channels + d].max(1e-12);
            let cov_hd = cov_c[home_ch * num_channels + d];

            let diff_var = (var_h + var_d - 2.0 * cov_hd).max(0.0);
            let rho_hd = (cov_hd / (var_h * var_d).sqrt()).clamp(-1.0, 1.0);
            let opposition = 1.0 - rho_hd;

            let max_mean = mean_a[home_ch].max(mean_a[d]);
            let attention_support = max_mean / (max_mean + 0.1);

            let score_d = diff_var * opposition * attention_support;

            // Donor selection criterion:
            // Needs meaningful contrast with Home and non-trivial attention/swing
            if score_d >= 0.05 * best_score.max(1e-6)
                && (mean_a[d] >= 0.010 || swings[d] >= 1.5)
                && swings[d] >= 0.80
            {
                candidate_donors.push(CandidateDonor {
                    channel: d,
                    is_ghost: false,
                    score: score_d,
                });
            }
        }

        // Channel 0 is reserved as the continuous autapomorphy sink (DIR-RHIZ-PAFF-012.3)
        // Autapomorphic rate bursts naturally pool into [ROOT], preventing false donor attribution.

        // Sort candidate donors by score descending
        candidate_donors.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        // Limit to top 4 active donors to preserve efficiency and eliminate spurious noise
        candidate_donors.truncate(4);

        // 7. Construct TrajectoryParentalPairs
        let home_k = home_ch.saturating_sub(1);
        let home_name = landmarks.taxa.get(home_k).cloned().unwrap_or_else(|| "Home".to_string());
        let home_global = landmarks.indices.get(home_k).copied();

        let mut pairs = Vec::with_capacity(candidate_donors.len());
        for cd in candidate_donors {
            let (donor_k, donor_name, donor_global) = if cd.is_ghost {
                (None, "Ghost".to_string(), None)
            } else {
                let dk = cd.channel.saturating_sub(1);
                (
                    Some(dk),
                    landmarks.taxa.get(dk).cloned().unwrap_or_else(|| "Donor".to_string()),
                    landmarks.indices.get(dk).copied(),
                )
            };

            pairs.push(TrajectoryParentalPair {
                candidate_idx,
                home_channel: home_ch,
                donor_channel: cd.channel,
                home_landmark_idx: Some(home_k),
                donor_landmark_idx: donor_k,
                is_ghost_donor: cd.is_ghost,
                home_name: home_name.clone(),
                donor_name,
                home_global_idx: home_global,
                donor_global_idx: donor_global,
                swing_home: swings[home_ch],
                swing_donor: swings[cd.channel],
            });
        }

        if pairs.is_empty() {
            pairs.push(TrajectoryParentalPair {
                candidate_idx,
                home_channel: home_ch,
                donor_channel: primary_donor_ch,
                home_landmark_idx: Some(home_k),
                donor_landmark_idx: Some(primary_donor_ch.saturating_sub(1)),
                is_ghost_donor: false,
                home_name,
                donor_name: landmarks.taxa.get(primary_donor_ch.saturating_sub(1)).cloned().unwrap_or_else(|| "Donor".to_string()),
                home_global_idx: home_global,
                donor_global_idx: landmarks.indices.get(primary_donor_ch.saturating_sub(1)).copied(),
                swing_home: swings[home_ch],
                swing_donor: swings[primary_donor_ch],
            });
        }

        pairs
    }

    /// Localized Attractive Work Integration (DIR-RHIZ-PAFF-012.0)
    pub fn resolve_reticulation_channels_localized(
        &self,
        candidate_idx: usize,
        u1: usize,
        u2: usize,
        c_tensor: &[f64],
        l: usize,
        n: usize,
        landmarks: &LandmarkSet,
        aln: &crate::types::Alignment,
        preferred_home_ch: Option<usize>,
        preferred_donor_ch: Option<usize>,
    ) -> TrajectoryParentalPair {
        let k = landmarks.num_landmarks;
        let num_channels = k + 1;

        let u1_idx = (u1.saturating_sub(1)).min(l - 1);
        let u2_idx = (u2.saturating_sub(1)).min(l - 1);
        let (s_idx, e_idx) = if u1_idx <= u2_idx { (u1_idx, u2_idx) } else { (u2_idx, u1_idx) };

        let mut w_tract = vec![0.0f64; num_channels];
        let mut w_flank = vec![0.0f64; num_channels];

        for ch in 0..num_channels {
            if ch > 0 {
                let k_idx = ch - 1;
                if landmarks.indices[k_idx] == candidate_idx {
                    continue;
                }
            }

            let c_u1 = if s_idx > 0 {
                c_tensor[(s_idx - 1) * n * num_channels + candidate_idx * num_channels + ch]
            } else {
                0.0
            };
            let c_u2 = c_tensor[e_idx * n * num_channels + candidate_idx * num_channels + ch];
            w_tract[ch] = c_u2 - c_u1;

            let c_0 = 0.0;
            let c_l = c_tensor[(l - 1) * n * num_channels + candidate_idx * num_channels + ch];
            w_flank[ch] = (c_u1 - c_0) + (c_l - c_u2);
        }

        let mut contemporary_channels = Vec::new();
        for p in 0..k {
            let l_global = landmarks.indices[p];
            if l_global != candidate_idx {
                contemporary_channels.push(p + 1);
            }
        }
        if contemporary_channels.is_empty() {
            contemporary_channels.push(1);
        }

        // Test for Ghost Donor
        let mut is_ghost = false;
        let mut max_w_tr_cont = f64::NEG_INFINITY;
        let mut best_cont_donor = contemporary_channels[0];
        for &ch in &contemporary_channels {
            if w_tract[ch] > max_w_tr_cont {
                max_w_tr_cont = w_tract[ch];
                best_cont_donor = ch;
            }
        }

        if preferred_donor_ch == Some(0) || (w_tract[0] >= self.min_root_swing && w_tract[0] > max_w_tr_cont) {
            is_ghost = true;
        }

        if is_ghost {
            let home_candidates = contemporary_channels.clone();
            let mut best_home_ch = home_candidates[0];
            let mut max_w_fl = f64::NEG_INFINITY;
            for &ch in &home_candidates {
                if w_flank[ch] > max_w_fl {
                    max_w_fl = w_flank[ch];
                    best_home_ch = ch;
                }
            }

            let mut final_home_ch = best_home_ch;
            if home_candidates.len() > 1 {
                let best_w = w_flank[best_home_ch];
                let mut runner_ups = Vec::new();
                for &ch in &home_candidates {
                    if (best_w - w_flank[ch]) < 0.25 * best_w.abs().max(1.0) {
                        runner_ups.push(ch);
                    }
                }
                if runner_ups.len() > 1 {
                    let cand_row = aln.row(candidate_idx);
                    let mut best_dist = f64::INFINITY;
                    for ch in runner_ups {
                        let k_idx = ch - 1;
                        let t_idx = landmarks.indices[k_idx];
                        let t_row = aln.row(t_idx);
                        let mut mismatches = 0usize;
                        let mut valid = 0usize;
                        for u in 0..l {
                            if u < s_idx || u > e_idx {
                                let rc = cand_row[u];
                                let rt = t_row[u];
                                if rc > 0 && rt > 0 {
                                    valid += 1;
                                    if rc != rt { mismatches += 1; }
                                }
                            }
                        }
                        let d = if valid > 0 { (mismatches as f64) / (valid as f64) } else { 1.0 };
                        if d < best_dist {
                            best_dist = d;
                            final_home_ch = ch;
                        }
                    }
                }
            }

            let home_k = final_home_ch.saturating_sub(1);
            let home_name = landmarks.taxa.get(home_k).cloned().unwrap_or_else(|| "Home".to_string());
            let home_global = landmarks.indices.get(home_k).copied();

            return TrajectoryParentalPair {
                candidate_idx,
                home_channel: final_home_ch,
                donor_channel: 0,
                home_landmark_idx: Some(home_k),
                donor_landmark_idx: None,
                is_ghost_donor: true,
                home_name,
                donor_name: "Ghost".to_string(),
                home_global_idx: home_global,
                donor_global_idx: None,
                swing_home: w_flank[final_home_ch],
                swing_donor: w_tract[0],
            };
        }

        // Contemporary Donor Resolution
        let best_donor_ch = if let Some(pref_d) = preferred_donor_ch {
            if pref_d > 0 && contemporary_channels.contains(&pref_d) && w_tract[pref_d] >= 0.5 * max_w_tr_cont {
                pref_d
            } else {
                best_cont_donor
            }
        } else {
            best_cont_donor
        };

        // Intra-Clade Sister Disambiguation via normalized Hamming distance inside tract
        let mut final_donor_ch = best_donor_ch;
        if contemporary_channels.len() > 1 {
            let best_w = w_tract[best_donor_ch];
            let mut runner_ups = Vec::new();
            for &ch in &contemporary_channels {
                if (best_w - w_tract[ch]) < 0.25 * best_w.abs().max(1.0) {
                    runner_ups.push(ch);
                }
            }
            if runner_ups.len() > 1 {
                let cand_row = aln.row(candidate_idx);
                let mut best_dist = f64::INFINITY;
                for ch in runner_ups {
                    let k_idx = ch - 1;
                    let t_idx = landmarks.indices[k_idx];
                    let t_row = aln.row(t_idx);
                    let mut mismatches = 0usize;
                    let mut valid = 0usize;
                    for u in s_idx..=e_idx {
                        let rc = cand_row[u];
                        let rt = t_row[u];
                        if rc > 0 && rt > 0 {
                            valid += 1;
                            if rc != rt { mismatches += 1; }
                        }
                    }
                    let d = if valid > 0 { (mismatches as f64) / (valid as f64) } else { 1.0 };
                    if d < best_dist {
                        best_dist = d;
                        final_donor_ch = ch;
                    }
                }
            }
        }

        // Home Resolution in Flanks
        let mut home_candidates: Vec<usize> = contemporary_channels
            .iter()
            .cloned()
            .filter(|&ch| ch != final_donor_ch)
            .collect();
        if home_candidates.is_empty() {
            home_candidates = contemporary_channels.clone();
        }

        let mut max_w_fl = f64::NEG_INFINITY;
        let mut best_home_ch = home_candidates[0];
        for &ch in &home_candidates {
            if w_flank[ch] > max_w_fl {
                max_w_fl = w_flank[ch];
                best_home_ch = ch;
            }
        }

        if let Some(pref_h) = preferred_home_ch {
            if home_candidates.contains(&pref_h) && w_flank[pref_h] >= 0.5 * max_w_fl {
                best_home_ch = pref_h;
            }
        }

        let mut final_home_ch = best_home_ch;
        if home_candidates.len() > 1 {
            let best_w = w_flank[best_home_ch];
            let mut runner_ups = Vec::new();
            for &ch in &home_candidates {
                if (best_w - w_flank[ch]) < 0.25 * best_w.abs().max(1.0) {
                    runner_ups.push(ch);
                }
            }
            if runner_ups.len() > 1 {
                let cand_row = aln.row(candidate_idx);
                let mut best_dist = f64::INFINITY;
                for ch in runner_ups {
                    let k_idx = ch - 1;
                    let t_idx = landmarks.indices[k_idx];
                    let t_row = aln.row(t_idx);
                    let mut mismatches = 0usize;
                    let mut valid = 0usize;
                    for u in 0..l {
                        if u < s_idx || u > e_idx {
                            let rc = cand_row[u];
                            let rt = t_row[u];
                            if rc > 0 && rt > 0 {
                                valid += 1;
                                if rc != rt { mismatches += 1; }
                            }
                        }
                    }
                    let d = if valid > 0 { (mismatches as f64) / (valid as f64) } else { 1.0 };
                    if d < best_dist {
                        best_dist = d;
                        final_home_ch = ch;
                    }
                }
            }
        }

        // Polarity Verification: In tract, candidate must be closer to Donor than in flank
        let cand_row = aln.row(candidate_idx);
        let donor_k_temp = final_donor_ch - 1;
        let home_k_temp = final_home_ch - 1;
        let d_row = aln.row(landmarks.indices[donor_k_temp]);
        let h_row = aln.row(landmarks.indices[home_k_temp]);

        let mut d_tr_d = 0usize;
        let mut v_tr = 0usize;
        let mut d_fl_d = 0usize;
        let mut v_fl = 0usize;

        let mut d_tr_h = 0usize;
        let mut d_fl_h = 0usize;

        for u in 0..l {
            let rc = cand_row[u];
            let rd = d_row[u];
            let rh = h_row[u];
            if rc > 0 && rd > 0 && rh > 0 {
                if u >= s_idx && u <= e_idx {
                    v_tr += 1;
                    if rc != rd { d_tr_d += 1; }
                    if rc != rh { d_tr_h += 1; }
                } else {
                    v_fl += 1;
                    if rc != rd { d_fl_d += 1; }
                    if rc != rh { d_fl_h += 1; }
                }
            }
        }

        let dist_tr_d = if v_tr > 0 { (d_tr_d as f64) / (v_tr as f64) } else { 0.0 };
        let dist_fl_d = if v_fl > 0 { (d_fl_d as f64) / (v_fl as f64) } else { 0.0 };
        let dist_tr_h = if v_tr > 0 { (d_tr_h as f64) / (v_tr as f64) } else { 0.0 };
        let dist_fl_h = if v_fl > 0 { (d_fl_h as f64) / (v_fl as f64) } else { 0.0 };

        // If inverted polarity (closer to Donor in flank and closer to Home in tract), swap them
        if (dist_fl_d < dist_tr_d) && (dist_tr_h < dist_fl_h) {
            std::mem::swap(&mut final_home_ch, &mut final_donor_ch);
        }

        let home_k = final_home_ch - 1;
        let home_name = landmarks.taxa.get(home_k).cloned().unwrap_or_else(|| "Home".to_string());
        let home_global = landmarks.indices.get(home_k).copied();

        let donor_k = final_donor_ch - 1;
        let donor_name = landmarks.taxa.get(donor_k).cloned().unwrap_or_else(|| "Donor".to_string());
        let donor_global = landmarks.indices.get(donor_k).copied();

        TrajectoryParentalPair {
            candidate_idx,
            home_channel: final_home_ch,
            donor_channel: final_donor_ch,
            home_landmark_idx: Some(home_k),
            donor_landmark_idx: Some(donor_k),
            is_ghost_donor: false,
            home_name,
            donor_name,
            home_global_idx: home_global,
            donor_global_idx: donor_global,
            swing_home: w_flank[final_home_ch],
            swing_donor: w_tract[final_donor_ch],
        }
    }
}

