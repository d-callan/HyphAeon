//! Info-DANNO: Information-Theoretic Transmission Odds and Forensic Allele Frequency Imputation.
//!
//! Extends DANNO with site-specific background mutation frequency weighting and convergent
//! drug-resistance mutation (DRM) homoplasy downweighting (Tzou, Kosakovsky Pond et al. 2020).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use crate::danno::DannoEstimator;
use crate::tn93::Alignment;

/// Site-specific empirical allele frequency table across alignment columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlleleFrequencyTable {
    pub seq_len: usize,
    /// Frequencies for A, C, G, T at each site l in [0, seq_len): [f_A, f_C, f_G, f_T]
    pub frequencies: Vec<[f64; 4]>,
    /// Consensus base index (0=A, 1=C, 2=G, 3=T) at each site
    pub consensus: Vec<i8>,
}

impl AlleleFrequencyTable {
    /// Computes empirical base frequencies directly from an alignment.
    pub fn from_alignment(alignment: &Alignment) -> Self {
        let n = alignment.num_sequences();
        let l = alignment.seq_len;
        let mut frequencies = vec![[0.25, 0.25, 0.25, 0.25]; l];
        let mut consensus = vec![-1i8; l];

        if n == 0 || l == 0 {
            return Self { seq_len: l, frequencies, consensus };
        }

        for col in 0..l {
            let mut counts: [f64; 4] = [0.0; 4];
            let mut valid_total: f64 = 0.0;

            for seq in &alignment.encoded_ints {
                let base = seq[col];
                if base >= 0 && base < 4 {
                    counts[base as usize] += 1.0;
                    valid_total += 1.0;
                }
            }

            if valid_total > 0.0 {
                let mut best_base = 0;
                let mut best_count: f64 = -1.0;
                for b in 0..4 {
                    let freq: f64 = (counts[b] / valid_total).max(1e-4);
                    frequencies[col][b] = freq;
                    if counts[b] > best_count {
                        best_count = counts[b];
                        best_base = b;
                    }
                }
                consensus[col] = best_base as i8;
            }
        }

        Self {
            seq_len: l,
            frequencies,
            consensus,
        }
    }
}

/// Info-DANNO Evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoDannoEstimator {
    pub danno: DannoEstimator,
    pub freq_table: AlleleFrequencyTable,
    pub drm_sites: HashSet<usize>,
    pub drm_penalty_nats: f64,
}

impl InfoDannoEstimator {
    pub fn new(
        danno: DannoEstimator,
        freq_table: AlleleFrequencyTable,
        drm_sites: Option<HashSet<usize>>,
        drm_penalty_nats: Option<f64>,
    ) -> Self {
        Self {
            danno,
            freq_table,
            drm_sites: drm_sites.unwrap_or_default(),
            drm_penalty_nats: drm_penalty_nats.unwrap_or(3.0),
        }
    }

    /// Evaluates information-theoretic transmission evidence (in nats) for two sequences.
    ///
    /// Net Info Score (nats) = ln(BF_DANNO) + sum_{l in rare shared} ln(1 / f_l) - sum_{l in DRM} penalty.
    pub fn evaluate_pair_info(
        &self,
        seq1: &[i8],
        seq2: &[i8],
        dist: f64,
        delta_t: f64,
    ) -> (f64, f64, usize) {
        let base_bf = self.danno.compute_bayes_factor(dist, delta_t);
        if base_bf <= 0.0 {
            return (0.0, -1000.0, 0);
        }

        let mut delta_info = 0.0;
        let mut rare_shared_count = 0usize;
        let l = seq1.len().min(seq2.len()).min(self.freq_table.seq_len);

        for col in 0..l {
            let b1 = seq1[col];
            let b2 = seq2[col];

            // Both resolved and identical
            if b1 >= 0 && b1 < 4 && b1 == b2 {
                let base_idx = b1 as usize;
                let cons_idx = self.freq_table.consensus[col];

                // If differing from consensus base (a true mutation)
                if cons_idx >= 0 && b1 != cons_idx {
                    let freq = self.freq_table.frequencies[col][base_idx];
                    if freq < 0.01 {
                        // Rare neutral polymorphism (< 1% prevalence)
                        delta_info += (1.0 / freq).ln();
                        rare_shared_count += 1;
                    }

                    // Check if this site is a known convergent DRM
                    if self.drm_sites.contains(&col) {
                        delta_info -= self.drm_penalty_nats;
                    }
                }
            }
        }

        let total_nats = base_bf.ln() + delta_info;
        let info_bf = total_nats.exp().max(1e-300);

        (info_bf, total_nats, rare_shared_count)
    }
}
