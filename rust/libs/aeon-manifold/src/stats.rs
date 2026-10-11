//! Canonical LRT→p-value and FDR kernels (ported from `aeon_core.stats`).
//!
//! chi2 survival function: `chi2.sf(x, k) = gammaincc(k/2, x/2)`.

use crate::gammaincc;
use std::f64::consts::PI;

#[inline]
fn chi2_sf(x: f64, df: f64) -> f64 {
    gammaincc(df / 2.0, x / 2.0)
}

/// MEME asymptotic mixture null: 1/3·δ(0) + 2/3·(0.45·χ²₁ + 0.55·χ²₂).
///
/// LRT ≤ 0 → p = 2/3 (point mass at zero); LRT > 0 → continuous mixture
/// tail. This is the HyPhy MEME null and the distribution HyphAeon's
/// neural model is trained to mimic.
pub fn pvals_from_lrt_meme(lrts: &[f64]) -> Vec<f64> {
    lrts.iter()
        .map(|&l| {
            if l <= 0.0 {
                2.0 / 3.0
            } else {
                (2.0 / 3.0) * (0.45 * chi2_sf(l, 1.0) + 0.55 * chi2_sf(l, 2.0))
            }
        })
        .collect()
}

/// Self & Liang (1987) mixture null: 0.5·δ(0) + 0.5·χ²₁.
///
/// Used by BUSTED (whole-gene omnibus test) and attribution p-values.
pub fn pvals_from_lrt_self_liang(lrts: &[f64]) -> Vec<f64> {
    lrts.iter()
        .map(|&l| if l <= 0.0 { 1.0 } else { 0.5 * chi2_sf(l, 1.0) })
        .collect()
}

/// Benjamini-Hochberg FDR q-values, clipped to [0, 1].
///
/// Identical order semantics to the Python impl: argsort ascending,
/// backward cumulative min of p·n/(i+1), scatter back to original order.
/// NaN inputs sort to the end and produce NaN q-values (argsort total order
/// matches np.argsort's NaN-last convention only for finite inputs).
pub fn benjamini_hochberg(pvals: &[f64]) -> Vec<f64> {
    let n = pvals.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pvals[a].partial_cmp(&pvals[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut q_sorted = vec![0.0; n];
    let mut min_q = 1.0;
    for i in (0..n).rev() {
        let q = pvals[order[i]] * n as f64 / (i as f64 + 1.0);
        if q < min_q {
            min_q = q;
        }
        q_sorted[i] = min_q;
    }

    let mut res = vec![0.0; n];
    for (i, &idx) in order.iter().enumerate() {
        res[idx] = q_sorted[i].clamp(0.0, 1.0);
    }
    res
}

/// Cauchy Combination Test omnibus p-value.
///
/// t = mean(tan((0.5 - p)·π)) over p clipped to [1e-15, 1-1e-15];
/// p_cct = 0.5 - atan(t)/π clipped to [1e-15, 1.0]. Empty input → 1.0.
/// Robust to dependency among the input p-values.
pub fn cauchy_combination_p(pvals: &[f64]) -> f64 {
    if pvals.is_empty() {
        return 1.0;
    }
    let n = pvals.len() as f64;
    let t: f64 = pvals
        .iter()
        .map(|&p| ((0.5 - p.clamp(1e-15, 1.0 - 1e-15)) * PI).tan())
        .sum::<f64>()
        / n;
    (0.5 - t.atan() / PI).clamp(1e-15, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meme_null_point_mass() {
        let p = pvals_from_lrt_meme(&[0.0, -1.0]);
        assert_eq!(p, vec![2.0 / 3.0, 2.0 / 3.0]);
    }

    #[test]
    fn meme_positive_tail() {
        // chi2.sf(1.0, df=1) = 0.3173105078629142, df=2 = 0.6065306597126334
        let p = pvals_from_lrt_meme(&[1.0]);
        let expected =
            (2.0 / 3.0) * (0.45 * 0.3173105078629142 + 0.55 * 0.6065306597126334);
        assert!((p[0] - expected).abs() < 1e-12, "got {}", p[0]);
    }

    #[test]
    fn self_liang_values() {
        let p = pvals_from_lrt_self_liang(&[0.0, 1.0]);
        assert_eq!(p[0], 1.0);
        assert!((p[1] - 0.5 * 0.3173105078629142).abs() < 1e-12);
    }

    #[test]
    fn bh_simple() {
        // p = [0.01, 0.04, 0.03]: sorted [0.01,0.03,0.04],
        // q = min-backward of [0.03, 0.045, 0.04] -> [0.03, 0.04, 0.04]
        let q = benjamini_hochberg(&[0.01, 0.04, 0.03]);
        assert!((q[0] - 0.03).abs() < 1e-12);
        assert!((q[1] - 0.04).abs() < 1e-12);
        assert!((q[2] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn bh_empty() {
        assert!(benjamini_hochberg(&[]).is_empty());
    }

    #[test]
    fn cauchy_empty_and_uniform() {
        assert_eq!(cauchy_combination_p(&[]), 1.0);
        // p=0.5 → tan(0)=0 → t=0 → p_cct = 0.5
        assert!((cauchy_combination_p(&[0.5]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn cauchy_tiny_pvals_gives_small_p() {
        assert!(cauchy_combination_p(&[1e-10, 1e-10]) < 1e-6);
    }
}
