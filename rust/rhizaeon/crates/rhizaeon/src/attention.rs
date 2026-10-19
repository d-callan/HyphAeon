use crate::types::{Alignment, LandmarkSet};

pub struct PhyloAttentionEngine {
    pub dim: usize,
    pub sqrt_d: f64,
    pub root_sink_factor: f64,
}

impl PhyloAttentionEngine {
    pub fn new(dim: usize, root_sink_factor: f64) -> Self {
        Self {
            dim,
            sqrt_d: (dim as f64).sqrt(),
            root_sink_factor,
        }
    }

    /// Computes A_tensor and C_tensor of shape (L, N, K+1).
    /// Flat indexing: `u * (N * num_channels) + i * num_channels + ch`
    pub fn compute_attention_and_force_tensors(
        &self,
        aln: &Alignment,
        landmarks: &LandmarkSet,
        m_matrix: &[f64],
        b_prior: &[f64],
        root_sequence: &[u8],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = aln.num_taxa;
        let k = landmarks.num_landmarks;
        let l = aln.length;
        let num_channels = k + 1;
        let total_size = l * n * num_channels;

        let mut a_tensor = vec![0.0f64; total_size];
        let mut f_tensor = vec![0.0f64; total_size];
        let mut c_tensor = vec![0.0f64; total_size];

        let mut sum_a = vec![0.0f64; n * num_channels];
        let mut valid_pair_counts = vec![0usize; n * num_channels];
        let shift_d = (self.dim as f64) / 2.0;

        // Precompute self- and twin-clone masking matrix (n x k):
        // Mask landmark p for query i if i == l_global OR if l_global is an identical/sister clone
        // of i (d < d_sister) AND there is at least one other landmark in i's clade (d <= d_clade)
        // AND the sister sub-cluster is not the strict majority consensus of the clade.
        // This prevents minority co-recombinant twins (e.g. pop1.21 and pop1.23, or Rec_0..K) from
        // absorbing 100% of each other's attention while preserving majority parental clades.
        let (d_sister, d_clade) =
            crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
        let mut mask_landmark = vec![false; n * k];
        let mut is_non_sister_landmark = vec![false; n * k];
        for i in 0..n {
            let mut sister_landmarks = Vec::new();
            let mut ext_landmarks = Vec::new();
            let mut non_sister_clade_landmarks = 0usize;
            for p in 0..k {
                let l_global = landmarks.indices[p];
                if l_global == i {
                    mask_landmark[i * k + p] = true;
                } else {
                    let d = crate::landmarks::compute_pairwise_distance(aln, i, l_global);
                    if crate::landmarks::is_true_sister_pair(aln, i, l_global, d_sister) {
                        sister_landmarks.push(p);
                    } else {
                        ext_landmarks.push((p, l_global, d));
                        is_non_sister_landmark[i * k + p] = true;
                        if d <= d_clade {
                            non_sister_clade_landmarks += 1;
                        }
                    }
                }
            }
            if !sister_landmarks.is_empty() && !ext_landmarks.is_empty() {
                let mut my_size = 1usize;
                for j in 0..n {
                    if j != i && crate::landmarks::compute_pairwise_distance(aln, i, j) < d_sister {
                        my_size += 1;
                    }
                }

                // 1. Metric Geodesic Interior Condition (Contemporary / Surrogate Donor):
                // A mosaic recombinant R = (1 - a)*P1 + a*P2 lies strictly inside the metric interval
                // between its parental landmarks (p1, p2): d(p1, p2) > max(d(i, p1), d(i, p2)) + 0.5*d_sister
                // and d(i, p1) + d(i, p2) <= 1.15 * d(p1, p2).
                let mut is_metric_interior = false;
                for a_idx in 0..ext_landmarks.len() {
                    let (_, l1, di1) = ext_landmarks[a_idx];
                    let mut l1_size = 1usize;
                    for j in 0..n {
                        if j != l1
                            && crate::landmarks::compute_pairwise_distance(aln, l1, j) < d_sister
                        {
                            l1_size += 1;
                        }
                    }
                    for b_idx in (a_idx + 1)..ext_landmarks.len() {
                        let (_, l2, di2) = ext_landmarks[b_idx];
                        let d12 = crate::landmarks::compute_pairwise_distance(aln, l1, l2);
                        let is_geodesic_interior = (di1 + di2) <= 1.15 * d12
                            || (di1.max(di2) <= 1.35 * di1.min(di2) && (di1 + di2) <= 1.65 * d12);
                        if d12 > di1.max(di2) + 0.5 * d_sister && is_geodesic_interior {
                            let mut l2_size = 1usize;
                            for j in 0..n {
                                if j != l2
                                    && crate::landmarks::compute_pairwise_distance(aln, l2, j)
                                        < d_sister
                                {
                                    l2_size += 1;
                                }
                            }
                            let is_asymmetric_outgroup_mirror =
                                my_size > l1_size.min(l2_size) && di1.max(di2) > 1.4 * di1.min(di2);
                            if !is_asymmetric_outgroup_mirror {
                                is_metric_interior = true;
                                break;
                            }
                        }
                    }
                    if is_metric_interior {
                        break;
                    }
                }

                // 2. Minority Intra-Clade Sub-Cluster Fallback:
                let mut non_sister_clade_aln_count = 0usize;
                for j in 0..n {
                    if j != i {
                        let d = crate::landmarks::compute_pairwise_distance(aln, i, j);
                        if d >= d_sister && d <= d_clade {
                            non_sister_clade_aln_count += 1;
                        }
                    }
                }
                let is_minority_subclade =
                    non_sister_clade_landmarks > 0 && (my_size - 1) <= non_sister_clade_aln_count;

                if is_metric_interior || is_minority_subclade {
                    for &p in &sister_landmarks {
                        mask_landmark[i * k + p] = true;
                    }
                }
            }
        }

        // Pass 1: Site-by-site softmax attention
        for u in 0..l {
            let u_offset = u * n * num_channels;
            let root_c = root_sequence.get(u).copied().unwrap_or(0);

            for i in 0..n {
                let qi_c = aln.get(i, u);
                let valid_q = qi_c > 0;
                let in_offset = i * num_channels;
                let out_offset = u_offset + in_offset;

                if !valid_q {
                    continue;
                }

                let mut max_s = f64::NEG_INFINITY;
                let mut logits = [0.0f64; 128]; // Max K+1 <= 128
                assert!(num_channels <= 128, "K+1 exceeds stack buffer limit");

                // Landmark channels 1..=K
                let mut any_non_sister_match = false;
                for p in 0..k {
                    let ch = p + 1;
                    let l_global = landmarks.indices[p];
                    let l_c = aln.get(l_global, u);

                    if mask_landmark[i * k + p] {
                        // Mask self-attention and twin-clone attention
                        logits[ch] = f64::NEG_INFINITY;
                        continue;
                    }

                    let m_val = m_matrix[in_offset + ch];
                    let match_aff = m_val + shift_d;
                    let is_match = valid_q && qi_c == l_c;
                    if is_match && is_non_sister_landmark[i * k + p] {
                        any_non_sister_match = true;
                    }
                    let dot = if is_match { match_aff } else { 0.0 };

                    let s = b_prior[in_offset + ch] + (dot / self.sqrt_d);
                    logits[ch] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }

                // Root channel 0:
                // Matches the ancestral consensus allele (`qi_c == root_c`), AND acts as the
                // autapomorphy / unsampled-outgroup sink (`root_affinity * root_sink_factor`)
                // when a valid query base mismatches all non-sister contemporary landmarks.
                let m_root = m_matrix[in_offset + 0];
                let root_affinity = m_root + shift_d;
                let root_dot = if root_c > 0 && qi_c == root_c {
                    root_affinity
                } else if !any_non_sister_match {
                    root_affinity * self.root_sink_factor
                } else {
                    0.0
                };
                let s_root = b_prior[in_offset + 0] + (root_dot / self.sqrt_d);
                logits[0] = s_root;
                if s_root > max_s {
                    max_s = s_root;
                }

                // Softmax
                let mut sum_exp = 0.0;
                for ch in 0..num_channels {
                    let s = logits[ch];
                    let exp_s = if s.is_finite() {
                        (s - max_s).exp()
                    } else {
                        0.0
                    };
                    logits[ch] = exp_s;
                    sum_exp += exp_s;
                }

                let inv_denom = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                let a_root = logits[0] * inv_denom;
                a_tensor[out_offset + 0] = a_root;
                sum_a[in_offset + 0] += a_root;
                valid_pair_counts[in_offset + 0] += 1;

                for p in 0..k {
                    let ch = p + 1;
                    if !mask_landmark[i * k + p] {
                        let a_val = logits[ch] * inv_denom;
                        a_tensor[out_offset + ch] = a_val;
                        sum_a[in_offset + ch] += a_val;
                        valid_pair_counts[in_offset + ch] += 1;
                    }
                }
            }
        }

        // Empirical sequence-wide mean attention over valid (un-scrubbed) sites: mean_A in (N, K+1)
        let mut mean_a = vec![0.0f64; n * num_channels];
        for idx in 0..(n * num_channels) {
            let cnt = valid_pair_counts[idx];
            if cnt > 0 {
                mean_a[idx] = sum_a[idx] / (cnt as f64);
            }
        }

        // Pass 2: Zero-drift Force Field F = A - mean_A (with F = 0 on scrubbed/gapped sites) and cumulative momentum C
        for i in 0..n {
            let in_offset = i * num_channels;
            let mut running_c = vec![0.0f64; num_channels];

            for u in 0..l {
                let valid_q = aln.get(i, u) > 0;
                let cell_offset = u * n * num_channels + in_offset;

                // Channel 0 (Root)
                let f_root = if valid_q {
                    a_tensor[cell_offset + 0] - mean_a[in_offset + 0]
                } else {
                    a_tensor[cell_offset + 0] = mean_a[in_offset + 0];
                    0.0
                };
                f_tensor[cell_offset + 0] = f_root;
                running_c[0] += f_root;
                c_tensor[cell_offset + 0] = running_c[0];

                // Landmark channels 1..=K
                for p in 0..k {
                    let ch = p + 1;
                    let valid_pair = valid_q && !mask_landmark[i * k + p];
                    let f_val = if valid_pair {
                        a_tensor[cell_offset + ch] - mean_a[in_offset + ch]
                    } else {
                        a_tensor[cell_offset + ch] = mean_a[in_offset + ch];
                        0.0
                    };
                    f_tensor[cell_offset + ch] = f_val;
                    running_c[ch] += f_val;
                    c_tensor[cell_offset + ch] = running_c[ch];
                }
            }
        }

        (a_tensor, f_tensor, c_tensor)
    }
}
