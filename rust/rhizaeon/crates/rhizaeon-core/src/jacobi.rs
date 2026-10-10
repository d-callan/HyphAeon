/// Exact Cyclic Jacobi Eigensolver for real symmetric K x K matrices.
/// Computes eigenvalues (sorted descending) and orthogonal eigenvectors:
/// A * V = V * diag(evals)

pub struct Eigendecomposition {
    pub eigenvalues: Vec<f64>,
    /// Column-major or row-major eigenvectors: V[i * K + col] is i-th component of col-th eigenvector
    pub eigenvectors: Vec<f64>,
}

/// Solves symmetric eigensystem for matrix of dimension k x k.
/// Input `a_flat` is a row-major slice of size k * k, representing a symmetric matrix.
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

                if _sweep > 4 && (a[p * k + p].abs() + g == a[p * k + p].abs())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_eigh_2x2() {
        // [2.0, 1.0]
        // [1.0, 2.0]
        // Eigenvalues: 3.0, 1.0
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let res = symmetric_jacobi_eigh(&a, 2);
        assert!((res.eigenvalues[0] - 3.0).abs() < 1e-12);
        assert!((res.eigenvalues[1] - 1.0).abs() < 1e-12);
    }
}
