//! CHIN Borel statistics and N_act interval vs the Python reference (C6).
//!
//! Oracle: tests/fixtures/chin/make_fixture.py runs the unmodified
//! trace50.chin.ChinEstimator, scipy.stats.linregress and scipy.stats.norm.ppf
//! on fixtures/chin/cases.json and writes fixtures/chin/oracle.json. The cases
//! are the DC DANNO cluster sizes (151 and 163 edges), the DC static sizes and
//! edge cases, from parity_2026-10-12/chin/cases.json.
//!
//! Tolerance: 1e-12 relative on every real value (the brief asked for 1e-9).
//! Python and Rust run the same closed forms; what remains is summation order
//! and the special functions (Lanczos ln_gamma, continued-fraction incomplete
//! beta/gamma vs cephes). The largest difference observed on this fixture is
//! 1.8e-15, so 1e-12 leaves a wide margin for other platforms while catching
//! the old normal-approximation p-value (off by 3-30x) and Acklam's unpolished
//! norm_ppf (1.3e-10 on the CI).

use serde_json::Value;
use trace50::chin::ChinEstimator;
use trace50::{linregress, norm_ppf};

const REL_TOL: f64 = 1e-12;

fn oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/chin/oracle.json")).expect("oracle.json parses")
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let scale = want.abs().max(f64::MIN_POSITIVE);
    eprintln!(
        "{}: rust {:e} python {:e} rel diff {:e}",
        what,
        got,
        want,
        (got - want).abs() / scale
    );
    assert!(
        (got - want).abs() <= tol * scale,
        "{}: rust {} vs python {} (rel diff {:e})",
        what,
        got,
        want,
        (got - want).abs() / scale
    );
}

#[test]
fn test_borel_decay_matches_python() {
    let o = oracle();
    let chin = ChinEstimator::new(1.5);
    for case in o["chin"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let sizes: Vec<usize> = case["sizes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let b = chin.test_borel_branching_decay(&sizes, 12);
        let want = &case["borel"];
        assert_rel(
            b.slope,
            want["slope"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} slope", name),
        );
        assert_rel(
            b.r_squared,
            want["r_squared"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} r_squared", name),
        );
        assert_rel(
            b.p_value,
            want["p_value"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} p_value", name),
        );
        assert_eq!(
            b.n_clusters as u64,
            want["n_clusters"].as_u64().unwrap(),
            "{} n_clusters",
            name
        );
        assert_eq!(
            b.n_megaclusters as u64,
            want["n_megaclusters"].as_u64().unwrap(),
            "{} n_megaclusters",
            name
        );
        assert_eq!(
            b.max_cluster_size as u64,
            want["max_cluster_size"].as_u64().unwrap(),
            "{} max_cluster_size",
            name
        );
        assert_eq!(
            b.rejects_uniform_sampling,
            want["rejects_uniform_sampling"].as_bool().unwrap(),
            "{} rejects_uniform_sampling",
            name
        );
    }
}

#[test]
fn test_active_population_ci_matches_python() {
    let o = oracle();
    let chin = ChinEstimator::new(1.5);
    let mut checked = 0;
    for case in o["chin"].as_array().unwrap() {
        let Some(want) = case.get("active_population") else {
            continue;
        };
        let name = case["name"].as_str().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;
        let e = case["e"].as_u64().unwrap() as usize;
        let r0 = case["r0"].as_f64().unwrap();
        let est = chin
            .estimate_active_population(n, e, Some(r0), 0.95)
            .expect("E_obs > 0 is estimable");
        assert_rel(
            est.n_act,
            want["N_act"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} N_act", name),
        );
        assert_rel(
            est.ci_lower,
            want["ci_lower"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} ci_lower", name),
        );
        assert_rel(
            est.ci_upper,
            want["ci_upper"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} ci_upper", name),
        );
        assert_rel(
            est.relative_error,
            want["relative_error"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} relative_error", name),
        );
        checked += 1;
    }
    assert_eq!(checked, 8);
}

#[test]
fn test_linregress_matches_scipy() {
    let o = oracle();
    for case in o["linregress"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let x: Vec<f64> = case["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let y: Vec<f64> = case["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let lr = linregress(&x, &y).expect("regression defined");
        let want_p = case["p_value"].as_f64().unwrap();
        assert_rel(
            lr.slope,
            case["slope"].as_f64().unwrap(),
            REL_TOL,
            &format!("{} slope", name),
        );
        if name == "perfect_fit" {
            // r = 1 exactly: scipy's p is set by its TINY = 1e-20 guard (p = 1e-20);
            // one ulp in r moves it by orders of magnitude, so only "effectively zero"
            // is a meaningful comparison.
            assert!(
                lr.p_value < 1e-12 && want_p < 1e-12,
                "{}: rust {} python {}",
                name,
                lr.p_value,
                want_p
            );
        } else {
            assert_rel(lr.p_value, want_p, REL_TOL, &format!("{} p_value", name));
        }
    }
}

#[test]
fn test_norm_ppf_matches_scipy() {
    let o = oracle();
    for case in o["norm_ppf"].as_array().unwrap() {
        let p = case["p"].as_f64().unwrap();
        assert_rel(
            norm_ppf(p),
            case["x"].as_f64().unwrap(),
            1e-12,
            &format!("norm_ppf({})", p),
        );
    }
}
