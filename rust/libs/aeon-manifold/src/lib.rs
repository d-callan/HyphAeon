//! aeon-manifold: shared math/stats kernel for the Aeon engine family.
//!
//! High-precision, zero-dependency, pure-Rust implementations of:
//! - Lanczos log-gamma function `ln_gamma`.
//! - Regularized upper incomplete gamma function `gammaincc`.
//! - Cumulative Poisson probability `poisson_cdf`.
//! - Normal and Student-t quantile functions (`norm_cdf`, `norm_ppf`, `student_t_ppf`).
//! - Ordinary least squares linear regression `linregress`.
//! - Order statistics: `percentile` (linear interpolation) and `median`.
//! - Canonical IUPAC alphabet + encoding projections (`encoding` module,
//!   re-exported at crate root).
//! - Shared FASTA tokenizer with strict/pad length policy (`fasta` module).
//! - Header date extraction (`temporal` module).
//! - Exact cyclic Jacobi eigensolver `symmetric_jacobi_eigh` for real symmetric
//!   matrices (normalized graph Laplacians, Gram matrices).

use std::f64::consts::{PI, SQRT_2};

pub mod encoding;
pub mod fasta;
pub mod stats;
pub mod temporal;
pub mod tn93;
pub use encoding::*;

/// 15-coefficient Lanczos approximation for ln(Gamma(z)) (g = 4.7421875).
/// Machine-precision accuracy across z > 0.
const LANCZOS_G: f64 = 607.0 / 128.0;
const LANCZOS_C: [f64; 15] = [
    0.99999999999999709182,
    57.156235665862923517,
    -59.597960355475491248,
    14.136097974741747174,
    -0.49191381609762019978,
    0.33994649984811888699e-4,
    0.46523628927048575665e-4,
    -0.98374475304879564677e-4,
    0.15808870322437011488e-3,
    -0.21026444172410488319e-3,
    0.21743961811521264320e-3,
    -0.16431810653676389022e-3,
    0.84418223983852743293e-4,
    -0.26190838401581408670e-4,
    0.36899182659531622704e-5,
];

/// Computes ln(Gamma(z)) for z > 0.
pub fn ln_gamma(z: f64) -> f64 {
    if z <= 0.0 {
        // Reflection formula for non-positive values
        if z.floor() == z {
            return f64::INFINITY;
        }
        let sin_pi_z = (PI * z).sin().abs();
        return (PI / sin_pi_z).ln() - ln_gamma(1.0 - z);
    }

    let x = z - 1.0;
    let mut a = LANCZOS_C[0];
    let t = x + LANCZOS_G + 0.5;
    for (k, &c) in LANCZOS_C[1..].iter().enumerate() {
        a += c / (x + (k as f64) + 1.0);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized lower incomplete gamma P(s, x) = gamma(s, x) / Gamma(s)
/// via Taylor series expansion for x < s + 1.
fn gamma_p_series(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut sum = 1.0 / s;
    let mut term = 1.0 / s;
    let mut n = 1.0;
    while n < 200.0 {
        term *= x / (s + n);
        sum += term;
        if term.abs() < sum.abs() * 1e-15 {
            break;
        }
        n += 1.0;
    }
    let ln_prefix = -x + s * x.ln() - ln_gamma(s);
    (ln_prefix.exp() * sum).min(1.0).max(0.0)
}

/// Regularized upper incomplete gamma Q(s, x) = Gamma(s, x) / Gamma(s)
/// via Legendre's continued fraction (modified Lentz's method) for x >= s + 1.
fn gamma_q_cf(s: f64, x: f64) -> f64 {
    let tiny = 1e-30;
    let mut f = x + 1.0 - s;
    if f.abs() < tiny {
        f = tiny;
    }
    let mut c = f;
    let mut d = 0.0;

    for i in 1..200 {
        let an = -(i as f64) * ((i as f64) - s);
        let bn = x + ((2 * i + 1) as f64) - s;

        d = bn + an * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = bn + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }

    let ln_prefix = -x + s * x.ln() - ln_gamma(s);
    (ln_prefix.exp() / f).min(1.0).max(0.0)
}

/// Regularized upper incomplete gamma function:
/// Q(s, x) = Gamma(s, x) / Gamma(s) = 1 - P(s, x).
/// Identical to SciPy's `scipy.special.gammaincc(s, x)`.
pub fn gammaincc(s: f64, x: f64) -> f64 {
    if s <= 0.0 || x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < s + 1.0 {
        (1.0 - gamma_p_series(s, x)).max(0.0).min(1.0)
    } else {
        gamma_q_cf(s, x).max(0.0).min(1.0)
    }
}

/// Cumulative Poisson probability: P(X <= k | lambda).
/// Exactly equal to Q(k + 1, lambda) = gammaincc(k + 1, lambda).
pub fn poisson_cdf(k: usize, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    gammaincc((k + 1) as f64, lambda)
}

/// Standard normal cumulative distribution function Phi(x).
/// Phi(x) = erfc(-x / sqrt(2)) / 2, evaluated through `erfc` so the lower tail
/// keeps relative precision.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

/// Error function erf(x) = 1 - erfc(x).
pub fn erf(x: f64) -> f64 {
    1.0 - erfc(x)
}

/// Complementary error function erfc(x) = Q(1/2, x^2) for x >= 0, 2 - Q(1/2, x^2) for x < 0.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let q = gammaincc(0.5, x * x);
    if x >= 0.0 {
        q
    } else {
        2.0 - q
    }
}

/// Standard normal percent-point function (inverse CDF) Phi^-1(p).
/// Uses Peter J. Acklam's rational approximation (error < 1.15e-9).
/// Returns `-inf` for p <= 0 and `+inf` for p >= 1.
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients in rational approximations
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley step on e = Phi(x) - p. In the upper half, e is formed as
    // (1 - p) - (1 - Phi(x)) so the residual does not cancel.
    let e = if p > 0.5 {
        (1.0 - p) - 0.5 * erfc(x / SQRT_2)
    } else {
        0.5 * erfc(-x / SQRT_2) - p
    };
    let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

/// Inverse CDF for Student's t-distribution with `df` degrees of freedom.
/// Cornish-Fisher asymptotic expansion around normal quantile.
pub fn student_t_ppf(p: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let z = norm_ppf(p);
    if df >= 100.0 {
        return z;
    }
    // Cornish-Fisher expansion in powers of 1/df
    let z2 = z * z;
    let z3 = z2 * z;
    let z5 = z3 * z2;
    let z7 = z5 * z2;

    let term1 = (z3 + z) / (4.0 * df);
    let term2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df * df);
    let term3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df * df * df);

    z + term1 + term2 + term3
}

/// Continued fraction for the regularized incomplete beta function
/// (modified Lentz's method; Numerical Recipes `betacf`).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b), given both x and xc = 1 - x so callers
/// can pass a complement computed without cancellation.
fn betainc_with_complement(a: f64, b: f64, x: f64, xc: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if xc <= 0.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * xc.ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        (ln_front.exp() * beta_cf(a, b, x) / a).max(0.0).min(1.0)
    } else {
        (1.0 - ln_front.exp() * beta_cf(b, a, xc) / b)
            .max(0.0)
            .min(1.0)
    }
}

/// Regularized incomplete beta function I_x(a, b) (SciPy's `scipy.special.betainc(a, b, x)`).
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    betainc_with_complement(a, b, x, 1.0 - x)
}

/// Two-sided Student-t tail probability P(|T| >= |t|) with `df` degrees of freedom:
/// I_{df/(df+t^2)}(df/2, 1/2), i.e. SciPy's `2 * stats.t.sf(abs(t), df)`.
pub fn student_t_two_sided_p(t: f64, df: f64) -> f64 {
    if t.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    let t2 = t * t;
    let denom = df + t2;
    betainc_with_complement(0.5 * df, 0.5, df / denom, t2 / denom)
}

/// Ordinary Least Squares Linear Regression Result.
#[derive(Debug, Clone, Copy)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_value: f64,
    pub r_squared: f64,
    pub p_value: f64,
    pub std_err: f64,
}

/// Computes ordinary least squares linear regression y = slope * x + intercept.
pub fn linregress(x: &[f64], y: &[f64]) -> Option<LinearRegression> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }

    let n_f = n as f64;
    let mean_x = x.iter().sum::<f64>() / n_f;
    let mean_y = y.iter().sum::<f64>() / n_f;

    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;
    let mut ss_xy = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
        ss_xy += dx * dy;
    }

    if ss_xx <= 1e-15 {
        return Some(LinearRegression {
            slope: 0.0,
            intercept: mean_y,
            r_value: 0.0,
            r_squared: 0.0,
            p_value: 1.0,
            std_err: 0.0,
        });
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    let r_value = if ss_yy > 1e-15 {
        (ss_xy / (ss_xx * ss_yy).sqrt()).max(-1.0).min(1.0)
    } else {
        0.0
    };
    let r_squared = r_value * r_value;

    let df = (n - 2) as f64;
    let rss = (ss_yy - slope * ss_xy).max(0.0);
    let std_err = if df > 0.0 {
        (rss / (df * ss_xx)).sqrt()
    } else {
        0.0
    };

    // Two-sided p-value for slope = 0 exactly as scipy.stats.linregress: Student t
    // with n - 2 df, t = r * sqrt(df / ((1 - r + TINY) * (1 + r + TINY))); with two
    // points scipy reports 1 if y[0] == y[1] and 0 otherwise.
    let p_value = if n == 2 {
        if y[0] == y[1] {
            1.0
        } else {
            0.0
        }
    } else {
        const TINY: f64 = 1.0e-20;
        let t_stat = r_value * (df / ((1.0 - r_value + TINY) * (1.0 + r_value + TINY))).sqrt();
        student_t_two_sided_p(t_stat, df)
    };

    Some(LinearRegression {
        slope,
        intercept,
        r_value,
        r_squared,
        p_value,
        std_err,
    })
}

/// Eigendecomposition of a real symmetric matrix: A * V = V * diag(eigenvalues).
pub struct Eigendecomposition {
    /// Eigenvalues sorted in descending order.
    pub eigenvalues: Vec<f64>,
    /// Row-major flattened k x k matrix of orthonormal eigenvectors:
    /// `eigenvectors[row * k + col]` is the `row`-th component of the `col`-th
    /// eigenvector (columns sorted to match `eigenvalues`).
    pub eigenvectors: Vec<f64>,
}

/// Exact cyclic Jacobi eigensolver for real symmetric k x k matrices.
///
/// Input `a_flat` is a row-major slice of length k * k representing a symmetric
/// matrix. Returns eigenvalues sorted descending with matching orthonormal
/// eigenvectors.
///
/// This is the single canonical Jacobi implementation for the Aeon engine
/// family: thresholded cyclic sweeps with the Numerical-Recipes stabilized
/// rotation (tau-form). Callers needing ascending order (e.g. Fiedler vectors
/// of a normalized Laplacian) index from the end of `eigenvalues`.
pub fn symmetric_jacobi_eigh(a_flat: &[f64], k: usize) -> Eigendecomposition {
    assert_eq!(a_flat.len(), k * k);

    let mut a = a_flat.to_vec();
    // Initialize V as identity matrix
    let mut v = vec![0.0f64; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }

    if k <= 1 {
        return Eigendecomposition {
            eigenvalues: vec![if k == 1 { a[0] } else { 0.0 }],
            eigenvectors: v,
        };
    }

    let max_sweeps = 50;
    let eps = 1e-15;

    for _sweep in 0..max_sweeps {
        // Measure sum of squares of off-diagonal elements
        let mut off_diag_sum = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                off_diag_sum += a[p * k + q].abs();
            }
        }

        if off_diag_sum < eps {
            break;
        }

        let thresh = if _sweep < 3 {
            0.2 * off_diag_sum / ((k * (k - 1)) as f64)
        } else {
            0.0
        };

        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                let g = 100.0 * apq.abs();

                if _sweep > 4
                    && (a[p * k + p].abs() + g == a[p * k + p].abs())
                    && (a[q * k + q].abs() + g == a[q * k + q].abs())
                {
                    a[p * k + q] = 0.0;
                } else if apq.abs() > thresh {
                    let h = a[q * k + q] - a[p * k + p];
                    let t = if h.abs() + g == h.abs() {
                        apq / h
                    } else {
                        let theta = 0.5 * h / apq;
                        let mut t_val = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                        if theta < 0.0 {
                            t_val = -t_val;
                        }
                        t_val
                    };

                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    let tau = s / (1.0 + c);
                    let h_prime = t * apq;

                    a[p * k + p] -= h_prime;
                    a[q * k + q] += h_prime;
                    a[p * k + q] = 0.0;

                    for j in 0..p {
                        let g_pj = a[j * k + p];
                        let h_qj = a[j * k + q];
                        a[j * k + p] = g_pj - s * (h_qj + g_pj * tau);
                        a[j * k + q] = h_qj + s * (g_pj - h_qj * tau);
                    }
                    for j in (p + 1)..q {
                        let g_jp = a[p * k + j];
                        let h_qj = a[j * k + q];
                        a[p * k + j] = g_jp - s * (h_qj + g_jp * tau);
                        a[j * k + q] = h_qj + s * (g_jp - h_qj * tau);
                    }
                    for j in (q + 1)..k {
                        let g_jp = a[p * k + j];
                        let h_jq = a[q * k + j];
                        a[p * k + j] = g_jp - s * (h_jq + g_jp * tau);
                        a[q * k + j] = h_jq + s * (g_jp - h_jq * tau);
                    }

                    for j in 0..k {
                        let g_v = v[j * k + p];
                        let h_v = v[j * k + q];
                        v[j * k + p] = g_v - s * (h_v + g_v * tau);
                        v[j * k + q] = h_v + s * (g_v - h_v * tau);
                    }
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let mut evals: Vec<(f64, usize)> = (0..k).map(|i| (a[i * k + i], i)).collect();
    // Sort descending
    evals.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_evals = Vec::with_capacity(k);
    let mut sorted_v = vec![0.0f64; k * k];

    for (new_col, (val, old_col)) in evals.into_iter().enumerate() {
        sorted_evals.push(val);
        for row in 0..k {
            sorted_v[row * k + new_col] = v[row * k + old_col];
        }
    }

    Eigendecomposition {
        eigenvalues: sorted_evals,
        eigenvectors: sorted_v,
    }
}

/// Linear-interpolation percentile on a PRE-SORTED slice.
///
/// Rank convention matches `numpy.percentile`: rank = pct/100 * (n - 1),
/// linear interpolation between the floor and ceil order statistics.
/// Returns NaN for empty input. Callers must sort first — this is the
/// contract historically held by `trace50::chin::percentile`.
pub fn percentile(sorted: &[f64], pct: f64) -> f64 {
    debug_assert!(
        sorted.windows(2).all(|w| w[0] <= w[1]),
        "percentile requires non-decreasing pre-sorted input"
    );
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

/// Median of an unordered slice; sorts a copy internally.
///
/// Equivalent to `percentile(sorted, 50)` — identical interpolation
/// semantics. Returns NaN for empty input; callers wanting the historical
/// `rhizaeon` empty -> 0.0 contract must handle it at the call site.
pub fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&sorted, 50.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_endpoints_and_interp() {
        let s = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&s, 0.0), 1.0);
        assert_eq!(percentile(&s, 100.0), 4.0);
        assert_eq!(percentile(&s, 50.0), 2.5); // rank 1.5 -> midpoint
        assert_eq!(percentile(&s, 25.0), 1.75); // rank 0.75 -> lerp
        assert_eq!(percentile(&[7.0], 63.0), 7.0);
    }

    #[test]
    fn test_percentile_empty_is_nan() {
        assert!(percentile(&[], 50.0).is_nan());
    }

    #[test]
    fn test_median_odd_even_empty() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn test_ln_gamma() {
        // Gamma(5) = 24 => ln(24) ≈ 3.1780538303479458
        let val = ln_gamma(5.0);
        assert!((val - 24.0_f64.ln()).abs() < 1e-12);
        // Gamma(0.5) = sqrt(pi)
        let val_half = ln_gamma(0.5);
        assert!((val_half - PI.sqrt().ln()).abs() < 1e-12);
    }

    #[test]
    fn test_gammaincc() {
        // gammaincc(1.0, 1.0) = exp(-1) ≈ 0.36787944117144233
        let q = gammaincc(1.0, 1.0);
        assert!((q - (-1.0_f64).exp()).abs() < 1e-10);
        // gammaincc(s, 0) == 1
        assert_eq!(gammaincc(2.5, 0.0), 1.0);
    }

    #[test]
    fn test_poisson_cdf() {
        // P(X <= 0 | lambda=2) = exp(-2) ≈ 0.1353352832366127
        let p0 = poisson_cdf(0, 2.0);
        assert!((p0 - (-2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_eigh_2x2() {
        // [2.0, 1.0]
        // [1.0, 2.0]
        // Eigenvalues sorted descending: 3.0, 1.0
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let res = symmetric_jacobi_eigh(&a, 2);
        assert!((res.eigenvalues[0] - 3.0).abs() < 1e-12);
        assert!((res.eigenvalues[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_norm_ppf_scipy_goldens() {
        // Reference values: scipy.stats.norm.ppf (1.17.1)
        // Acklam's max error is ~1e-9; use 1e-8 so the test is a regression
        // net rather than a false-precision oracle.
        let cases: &[(f64, f64)] = &[
            (0.5, 0.0),
            (0.025, -1.9599639845400545),
            (0.975, 1.959963984540054),
            (0.01, -2.3263478740408408),
            (0.99, 2.3263478740408408),
            (0.001, -3.090232306167813),
            (0.999, 3.090232306167813),
            (0.1, -1.2815515655446004),
            (0.9, 1.2815515655446004),
        ];
        for &(p, expected) in cases {
            assert!(
                (norm_ppf(p) - expected).abs() < 1e-8,
                "norm_ppf({p}) = {} expected {expected}",
                norm_ppf(p)
            );
        }
    }

    #[test]
    fn test_norm_ppf_extremes() {
        // Acklam approximation degrades in far tails; check against scipy
        // goldens with relaxed tolerance.
        assert!((norm_ppf(1e-10) - -6.361340902404056).abs() < 1e-4);
        assert!((norm_ppf(1.0 - 1e-10) - 6.361340889697422).abs() < 1e-4);
        // Contract: exact 0/1 return infinities (callers that clamp handle this)
        assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_ppf(1.0), f64::INFINITY);
    }

    #[test]
    fn test_student_t_ppf_scipy_goldens() {
        // Cornish-Fisher expansion: accurate for moderate df, approximates
        // norm_ppf for df >= 100. scipy.stats.t.ppf references.
        let cases: &[(f64, f64, f64, f64)] = &[
            // (p, df, expected, tolerance)
            (0.025, 5.0, -2.5705818356363155, 5e-3),
            (0.975, 5.0, 2.5705818356363146, 5e-3),
            (0.025, 30.0, -2.042272456301238, 1e-4),
            (0.975, 30.0, 2.0422724563012378, 1e-4),
            (0.5, 2.0, 0.0, 1e-12),
            // df >= 100: implementation returns norm_ppf exactly (documented);
            // tolerance tracks Acklam's own ~1.6e-9 error, not the true t value
            (0.025, 100.0, -1.9599639845400545, 1e-8),
        ];
        for &(p, df, expected, tol) in cases {
            let got = student_t_ppf(p, df);
            assert!(
                (got - expected).abs() < tol,
                "student_t_ppf({p}, {df}) = {got} expected {expected} (tol {tol})"
            );
        }
        assert!(student_t_ppf(0.5, 0.0).is_nan());
        assert!(student_t_ppf(0.5, -1.0).is_nan());
    }

    #[test]
    fn test_linregress_scipy_golden() {
        // scipy.stats.linregress([1,2,3,4,5], [2.1,3.9,6.2,7.8,10.1])
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.2, 7.8, 10.1];
        let r = linregress(&x, &y).unwrap();
        assert!((r.slope - 1.99).abs() < 1e-12);
        assert!((r.intercept - 0.05000000000000071).abs() < 1e-12);
        assert!((r.r_value - 0.9986517555689656).abs() < 1e-12);
        assert!((r.std_err - 0.059721576223897795).abs() < 1e-12);
        // scipy's p_value is 5.941539111755929e-05 (Student-t, df = 3).
        assert!((r.p_value - 5.941539111755929e-05).abs() < 1e-12);
    }

    #[test]
    fn test_linregress_degenerate() {
        assert!(linregress(&[1.0], &[1.0]).is_none());
        assert!(linregress(&[1.0, 2.0], &[1.0]).is_none());
        // Constant x: returns zero-slope sentinel, not None
        let r = linregress(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(r.slope, 0.0);
        assert_eq!(r.intercept, 2.0);
    }

    #[test]
    fn test_jacobi_eigh_normalized_laplacian() {
        // Path graph P3 (degrees [1,2,1]): normalized Laplacian has
        // eigenvalues {0, 1, 2}. The Fiedler vector (2nd smallest, i.e.
        // index n-2 in descending order) must put opposite signs on the
        // two leaves.
        let n = 3;
        let deg = [1.0f64, 2.0, 1.0];
        let w = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut l = vec![0.0f64; n * n];
        for i in 0..n {
            l[i * n + i] = 1.0;
            for j in 0..n {
                if i != j {
                    l[i * n + j] = -w[i * n + j] / (deg[i] * deg[j]).sqrt();
                }
            }
        }
        let res = symmetric_jacobi_eigh(&l, n);
        // Smallest eigenvalue ~ 0
        assert!(res.eigenvalues[n - 1].abs() < 1e-12);
        // Fiedler eigenvalue of normalized P3 Laplacian = 1
        assert!((res.eigenvalues[n - 2] - 1.0).abs() < 1e-12);
        // Fiedler vector gives opposite signs on the two leaves
        let f0 = res.eigenvectors[0 * n + (n - 2)];
        let f2 = res.eigenvectors[2 * n + (n - 2)];
        assert!(f0 * f2 < 0.0);
    }

    #[test]
    fn test_jacobi_eigh_large_laplacian() {
        // 120x120 normalized graph Laplacian with two weakly coupled 60-node
        // blocks (upstream trace50-core test_jacobi_eigensystem_large_laplacian,
        // c195a16). Eigenvalues are descending here: the zero eigenvalue is at
        // index n-1 and the Fiedler pair at n-2.
        let n = 120;
        let mut w: Vec<f64> = vec![0.0; n * n];
        let mut deg: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let same_block = (i < 60 && j < 60) || (i >= 60 && j >= 60);
                let weight = if same_block { 1.0 } else { 0.02 };
                w[i * n + j] = weight;
                w[j * n + i] = weight;
                deg[i] += weight;
                deg[j] += weight;
            }
        }
        let mut l_sym = vec![0.0; n * n];
        for i in 0..n {
            l_sym[i * n + i] = 1.0;
            for j in 0..n {
                if i != j {
                    l_sym[i * n + j] = -w[i * n + j] / (deg[i] * deg[j]).sqrt();
                }
            }
        }
        let res = symmetric_jacobi_eigh(&l_sym, n);
        // Smallest eigenvalue of normalized Laplacian is 0.0
        assert!(
            res.eigenvalues[n - 1].abs() < 1e-9,
            "lambda_1 = {}",
            res.eigenvalues[n - 1]
        );
        // Second eigenvalue (Fiedler value) for two weakly coupled blocks is small positive
        let lam2 = res.eigenvalues[n - 2];
        assert!(lam2 > 0.01 && lam2 < 0.10, "lambda_2 = {}", lam2);
        // Fiedler vector (column n-2) must cleanly separate block 0..60 from 60..120 by sign
        let fc = n - 2;
        let sign_0 = res.eigenvectors[fc].signum();
        for i in 0..60 {
            assert_eq!(res.eigenvectors[i * n + fc].signum(), sign_0);
        }
        for i in 60..120 {
            assert_eq!(res.eigenvectors[i * n + fc].signum(), -sign_0);
        }
    }
}
