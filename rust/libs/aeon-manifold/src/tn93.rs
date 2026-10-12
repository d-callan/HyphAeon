//! Tamura-Nei 93 (TN93) pairwise genetic distance kernel.
//!
//! Operates on IUPAC bitmasks (see [`crate::encoding`]). Ambiguities are
//! resolved to minimize distance: sites whose masks share at least one bit
//! contribute zero divergence ("resolve" mode). Nucleotide frequencies are
//! counted over the union of both sequences' masks per valid site — an
//! ambiguity-aware approximation of HyPhy's marginal frequencies, not a
//! bit-exact port of the tn93 binary. This module is the canonical Rust
//! implementation; `trace50` re-exports it and the `aeon_core._native`
//! extension (aeon-core/native) binds it.

use crate::encoding::{PURINE_MASK, PYRIMIDINE_MASK};

/// Computes analytical Tamura-Nei 93 (TN93) pairwise genetic distance.
/// Resolves ambiguities to minimize distance (matching characters share bits).
///
/// Pairs with no valid comparison sites return `0.0` — the historical
/// trace50 kernel contract. Consumers that need to distinguish "identical"
/// from "unresolvable" (e.g. `aeon_core._native` -> `dataset.py` max-distance
/// imputation) should use the `_nan` matrix variants instead.
pub fn compute_pairwise_tn93(seq_a_bits: &[u8], seq_b_bits: &[u8]) -> f64 {
    pairwise_inner(seq_a_bits, seq_b_bits).unwrap_or(0.0)
}

/// Inner TN93 computation. Returns `None` when the pair has zero valid
/// comparison sites (empty sequences, or every site masked by N/gap/`*`).
fn pairwise_inner(seq_a_bits: &[u8], seq_b_bits: &[u8]) -> Option<f64> {
    let n = seq_a_bits.len().min(seq_b_bits.len());
    if n == 0 {
        return None;
    }

    let mut l_valid = 0usize;
    let mut p1_count = 0usize;
    let mut p2_count = 0usize;
    let mut q_count = 0usize;

    let mut cnt_a = 0usize;
    let mut cnt_c = 0usize;
    let mut cnt_g = 0usize;
    let mut cnt_t = 0usize;

    for i in 0..n {
        let a = seq_a_bits[i];
        let b = seq_b_bits[i];

        if a > 0 && b > 0 {
            l_valid += 1;

            let a_or_b = a | b;
            if (a_or_b & 0b0001) > 0 {
                cnt_a += 1;
            }
            if (a_or_b & 0b0010) > 0 {
                cnt_c += 1;
            }
            if (a_or_b & 0b0100) > 0 {
                cnt_g += 1;
            }
            if (a_or_b & 0b1000) > 0 {
                cnt_t += 1;
            }

            // If bits overlap, zero divergence at this site
            if (a & b) == 0 {
                let p1 = ((a & PURINE_MASK) > 0) && ((b & PURINE_MASK) > 0);
                let p2 = ((a & PYRIMIDINE_MASK) > 0) && ((b & PYRIMIDINE_MASK) > 0);

                if p1 {
                    p1_count += 1;
                } else if p2 {
                    p2_count += 1;
                } else {
                    q_count += 1;
                }
            }
        }
    }

    if l_valid == 0 {
        return None;
    }

    let p1 = (p1_count as f64) / (l_valid as f64);
    let p2 = (p2_count as f64) / (l_valid as f64);
    let q = (q_count as f64) / (l_valid as f64);
    let p_ham = p1 + p2 + q;

    if p_ham <= 0.0 {
        return Some(0.0);
    }

    let tot = (cnt_a + cnt_c + cnt_g + cnt_t).max(1) as f64;
    let f_a = ((cnt_a as f64) / tot).max(1e-4);
    let f_c = ((cnt_c as f64) / tot).max(1e-4);
    let f_g = ((cnt_g as f64) / tot).max(1e-4);
    let f_t = ((cnt_t as f64) / tot).max(1e-4);

    let f_r = f_a + f_g;
    let f_y = f_c + f_t;

    let k1 = 2.0 * f_a * f_g / f_r;
    let k2 = 2.0 * f_c * f_t / f_y;
    let k3 = 2.0 * (f_r * f_y - (f_a * f_g * f_y / f_r) - (f_c * f_t * f_r / f_y));

    let arg1 = 1.0 - (f_r / (2.0 * f_a * f_g)) * p1 - (q / (2.0 * f_r));
    let arg2 = 1.0 - (f_y / (2.0 * f_c * f_t)) * p2 - (q / (2.0 * f_y));
    let arg3 = 1.0 - (q / (2.0 * f_r * f_y));

    if arg1 <= 1e-6 || arg2 <= 1e-6 || arg3 <= 1e-6 {
        return Some(p_ham);
    }

    let d = -k1 * arg1.ln() - k2 * arg2.ln() - k3 * arg3.ln();
    if !d.is_finite() || d < 0.0 {
        return Some(p_ham);
    }
    Some(d)
}

/// Flat row-major `N x N` TN93 distance matrix over mask-encoded sequences.
/// Diagonal is 0; every off-diagonal pair is computed (no threshold gating).
/// Unresolvable pairs (no valid sites) are 0.0.
pub fn distance_matrix(encoded: &[Vec<u8>]) -> Vec<f64> {
    distance_matrix_impl(encoded, 0.0)
}

/// Same as [`distance_matrix`] but unresolvable pairs are `NaN`, so callers
/// can impute them (e.g. to the maximum observed distance) instead of
/// silently treating incomparable taxa as identical.
pub fn distance_matrix_nan(encoded: &[Vec<u8>]) -> Vec<f64> {
    distance_matrix_impl(encoded, f64::NAN)
}

fn distance_matrix_impl(encoded: &[Vec<u8>], unresolvable: f64) -> Vec<f64> {
    let n = encoded.len();
    let mut matrix = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pairwise_inner(&encoded[i], &encoded[j]).unwrap_or(unresolvable);
            matrix[i * n + j] = d;
            matrix[j * n + i] = d;
        }
    }
    matrix
}

/// Flat row-major `N x M` TN93 cross-distance matrix: every row sequence in
/// `a` against every column sequence in `b` (landmark/Nystroem-style use).
pub fn cross_distance_matrix(a: &[Vec<u8>], b: &[Vec<u8>]) -> Vec<f64> {
    cross_distance_matrix_impl(a, b, 0.0)
}

/// Same as [`cross_distance_matrix`] but unresolvable pairs are `NaN`.
pub fn cross_distance_matrix_nan(a: &[Vec<u8>], b: &[Vec<u8>]) -> Vec<f64> {
    cross_distance_matrix_impl(a, b, f64::NAN)
}

fn cross_distance_matrix_impl(a: &[Vec<u8>], b: &[Vec<u8>], unresolvable: f64) -> Vec<f64> {
    let (n, m) = (a.len(), b.len());
    let mut matrix = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            matrix[i * m + j] = pairwise_inner(&a[i], &b[j]).unwrap_or(unresolvable);
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::char_to_iupac_mask;

    fn enc(s: &[u8]) -> Vec<u8> {
        s.iter().map(|&c| char_to_iupac_mask(c)).collect()
    }

    #[test]
    fn test_tn93_identical() {
        let bits = enc(b"ACGTACGT");
        assert_eq!(compute_pairwise_tn93(&bits, &bits), 0.0);
    }

    #[test]
    fn test_tn93_transition() {
        let bits1 = enc(b"ACGTACGTACGTACGTACGTACGT");
        let bits2 = enc(b"GCGTACGTACGTACGTACGTACGT"); // 1 transition in 24 bp
        let dist = compute_pairwise_tn93(&bits1, &bits2);
        assert!(dist > 0.0 && dist < 0.5);
    }

    #[test]
    fn test_tn93_n_ambiguity_skipped() {
        let bits1 = enc(b"ACGTACGTNN");
        let bits2 = enc(b"ACGTTCGTNN"); // 1 transition in 8 valid bp, Ns skipped
        let dist = compute_pairwise_tn93(&bits1, &bits2);
        assert!(dist > 0.12 && dist < 0.16);
    }

    #[test]
    fn test_distance_matrix_symmetric() {
        let seqs = vec![enc(b"ACGTACGT"), enc(b"ACGTTCGT"), enc(b"NNNNNNNN")];
        let m = distance_matrix(&seqs);
        assert_eq!(m.len(), 9);
        assert_eq!(m[0], 0.0);
        assert_eq!(m[1], m[3]);
        assert!(m[1] > 0.0);
        // All-N pair: no valid sites -> 0.0 (kernel contract)
        assert_eq!(m[2], 0.0);
    }

    #[test]
    fn test_distance_matrix_nan_unresolvable() {
        let seqs = vec![enc(b"ACGTACGT"), enc(b"NNNNNNNN")];
        let m = distance_matrix_nan(&seqs);
        assert!(m[1].is_nan());
        assert!(m[2].is_nan());
        assert_eq!(m[0], 0.0);
        assert_eq!(m[3], 0.0);
    }

    #[test]
    fn test_cross_distance_matrix_nan_unresolvable() {
        let a = vec![enc(b"ACGT")];
        let b = vec![enc(b"ACGT"), enc(b"----")];
        let m = cross_distance_matrix_nan(&a, &b);
        assert_eq!(m[0], 0.0);
        assert!(m[1].is_nan());
    }

    #[test]
    fn test_cross_distance_matrix_rectangular() {
        let a = vec![enc(b"ACGT"), enc(b"ACGA")];
        let b = vec![enc(b"ACGT"), enc(b"TCGT"), enc(b"NNNN")];
        let m = cross_distance_matrix(&a, &b);
        assert_eq!(m.len(), 6);
        assert_eq!(m[0], 0.0); // identical pair
        assert!(m[1] > 0.0); // transversion pair
    }
}
