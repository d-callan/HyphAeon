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
        let shift_d = (self.dim as f64) / 2.0;

        // Pass 1: Site-by-site softmax attention
        for u in 0..l {
            let u_offset = u * n * num_channels;
            let root_c = root_sequence.get(u).copied().unwrap_or(0);

            for i in 0..n {
                let qi_c = aln.get(i, u);
                let valid_q = qi_c > 0;
                let in_offset = i * num_channels;
                let out_offset = u_offset + in_offset;

                let mut max_s = f64::NEG_INFINITY;
                let mut logits = [0.0f64; 64]; // Max K+1 <= 64
                assert!(num_channels <= 64, "K+1 exceeds stack buffer limit");

                // Root channel 0
                let m_root = m_matrix[in_offset + 0];
                let root_affinity = m_root + shift_d;
                let root_dot = if root_c > 0 {
                    if valid_q && qi_c == root_c {
                        root_affinity
                    } else {
                        0.0
                    }
                } else if valid_q {
                    root_affinity * self.root_sink_factor
                } else {
                    0.0
                };
                let s_root = b_prior[in_offset + 0] + (root_dot / self.sqrt_d);
                logits[0] = s_root;
                if s_root > max_s {
                    max_s = s_root;
                }

                // Landmark channels 1..=K
                for p in 0..k {
                    let ch = p + 1;
                    let l_global = landmarks.indices[p];
                    if i == l_global {
                        // Mask self-attention
                        logits[ch] = f64::NEG_INFINITY;
                        continue;
                    }

                    let l_c = aln.get(l_global, u);
                    let m_val = m_matrix[in_offset + ch];
                    let match_aff = m_val + shift_d;
                    let dot = if valid_q && l_c > 0 && qi_c == l_c {
                        match_aff
                    } else {
                        0.0
                    };

                    let s = b_prior[in_offset + ch] + (dot / self.sqrt_d);
                    logits[ch] = s;
                    if s > max_s {
                        max_s = s;
                    }
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
                for ch in 0..num_channels {
                    let a_val = logits[ch] * inv_denom;
                    a_tensor[out_offset + ch] = a_val;
                    sum_a[in_offset + ch] += a_val;
                }
            }
        }

        // Empirical sequence-wide mean attention: mean_A in (N, K+1)
        let inv_l = 1.0 / (l as f64);
        let mut mean_a = vec![0.0f64; n * num_channels];
        for idx in 0..(n * num_channels) {
            mean_a[idx] = sum_a[idx] * inv_l;
        }

        // Pass 2: Zero-drift Force Field F = A - mean_A and cumulative momentum C
        for i in 0..n {
            let in_offset = i * num_channels;
            let mut running_c = vec![0.0f64; num_channels];

            for u in 0..l {
                let cell_offset = u * n * num_channels + in_offset;
                for ch in 0..num_channels {
                    let a_val = a_tensor[cell_offset + ch];
                    let f_val = a_val - mean_a[in_offset + ch];
                    f_tensor[cell_offset + ch] = f_val;

                    running_c[ch] += f_val;
                    c_tensor[cell_offset + ch] = running_c[ch];
                }
            }
        }

        (a_tensor, f_tensor, c_tensor)
    }
}
