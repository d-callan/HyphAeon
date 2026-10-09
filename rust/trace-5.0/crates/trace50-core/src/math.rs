//! High-precision mathematical and statistical functions for TRACE-5.0.
//!
//! Provides zero-dependency, pure-Rust implementations of:
//! - Lanczos log-gamma function `ln_gamma`.
//! - Regularized upper incomplete gamma function `gammaincc` / `q_gamma`.
//! - Cumulative Poisson probability `poisson_cdf`.
//! - Normal and Student-t quantile functions (`norm_ppf`, `student_t_ppf`).
//! - Ordinary least squares linear regression `linregress`.
//! - Symmetric matrix Jacobi eigensolver for normalized Graph Laplacians.

use std::f64::consts::{PI, SQRT_2};

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
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Error function erf(x) approximation (accurate to 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal percent-point function (inverse CDF) Phi^-1(p).
/// Uses Peter J. Acklam's rational approximation (error < 1.15e-9).
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

    if p < p_low {
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
    }
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

    let p_value = if std_err > 1e-15 && df > 0.0 {
        let t_stat = (slope / std_err).abs();
        // Approximate two-tailed p-value via normal distribution if df is sufficient
        2.0 * (1.0 - norm_cdf(t_stat))
    } else {
        1.0
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

/// Jacobi Eigenvalue Algorithm for real symmetric n x n matrices.
/// Guarantees unconditional convergence and produces sorted eigenvalues and orthogonal eigenvectors.
///
/// Output: `(eigenvalues, eigenvectors)`
/// - `eigenvalues`: sorted in ascending order.
/// - `eigenvectors`: flattened n x n matrix in column-major format (column `i` is eigenvector `i`).
pub fn jacobi_eigensystem(matrix: &[f64], n: usize, max_iter: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_offdiag = 0.0;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_offdiag {
                    max_offdiag = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_offdiag < tol {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = phi.cos();
        let s = phi.sin();

        // Perform rotation on matrix A
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                a[i * n + p] = c * aip - s * aiq;
                a[p * n + i] = a[i * n + p];
                a[i * n + q] = s * aip + c * aiq;
                a[q * n + i] = a[i * n + q];
            }
        }

        a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Accumulate transformation into V
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }

    let mut evals: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
    evals.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_evals: Vec<f64> = evals.iter().map(|&(val, _)| val).collect();
    let mut sorted_evecs = vec![0.0; n * n];

    for (new_col, &(_, old_col)) in evals.iter().enumerate() {
        for row in 0..n {
            sorted_evecs[row * n + new_col] = v[row * n + old_col];
        }
    }

    (sorted_evals, sorted_evecs)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_jacobi_eigensystem() {
        // 2x2 symmetric matrix: [[2, 1], [1, 2]] -> evals: 1.0, 3.0
        let m = vec![2.0, 1.0, 1.0, 2.0];
        let (evals, _) = jacobi_eigensystem(&m, 2, 100, 1e-12);
        assert!((evals[0] - 1.0).abs() < 1e-6);
        assert!((evals[1] - 3.0).abs() < 1e-6);
    }
}
