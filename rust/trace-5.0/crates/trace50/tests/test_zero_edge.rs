//! E_obs = 0: an alignment with static TN93 edges but no supported DANNO edge.
//!
//! Fixture and oracle: tests/fixtures/zero_edge/ (make_fixture.py, oracle.sh).
//! `tn93 -t 1.0 -a RMYSKB` + validation/run_danno.py on static_only.fasta give
//! (static_only.py_danno.json): DANNO edges 0, clusters 0, max_cluster 0;
//! static 0.015 edges 3, clusters 3, max_cluster 2.

use trace50::{run_trace50, Alignment, Trace50Config, N_ACT_NOT_ESTIMABLE_NO_EDGES};

/// `ChinEstimator(1.5).estimate_sampling_fraction(6, 3)`, printed by
/// phase2/verifier-fixes/c_rho/chin_rho_oracle.py (chin_rho_oracle.json).
/// For (6, 0) the same script gives rho 0.0 with N_act inf: rho = n / N_act.
const CONTROL_RHO_PY: f64 = 0.6666666666666666;

fn fixture_dossier() -> trace50::VisualizationDossier {
    let fasta = include_str!("fixtures/zero_edge/static_only.fasta");
    let alignment = Alignment::from_fasta_str(fasta).expect("fixture parses");
    run_trace50(&alignment, &Trace50Config::default())
        .expect("pipeline runs with zero supported edges")
}

#[test]
fn test_zero_edge_networks_match_python() {
    let d = fixture_dossier();
    assert_eq!(d.kpi.supported_transmission_edges, 0);
    assert_eq!(d.kpi.danno_network.total_clusters, 0);
    assert_eq!(d.kpi.danno_network.clustered_individuals, 0);
    // No singleton "giant component": max cluster counts only size >= 2.
    assert_eq!(d.kpi.danno_network.max_cluster_size, 0);
    assert_eq!(d.macro_scaling.borel_decay.max_cluster_size, 0);
    assert_eq!(d.kpi.static_network.total_clusters, 3);
    assert_eq!(d.kpi.static_network.clustered_individuals, 6);
    assert_eq!(d.kpi.static_network.max_cluster_size, 2);
}

#[test]
fn test_zero_edge_refuses_n_act_and_reduction() {
    let d = fixture_dossier();
    assert_eq!(d.kpi.giant_component_reduction_pct, None);
    assert_eq!(d.kpi.inferred_active_transmitting_pool, None);
    assert_eq!(d.kpi.inferred_active_pool_ci, None);
    assert_eq!(
        d.kpi.inferred_active_pool_status,
        N_ACT_NOT_ESTIMABLE_NO_EDGES
    );
    assert!(d.macro_scaling.active_population.is_none());
    assert_eq!(
        d.macro_scaling.active_population_status,
        N_ACT_NOT_ESTIMABLE_NO_EDGES
    );
    assert!(d.macro_scaling.bayesian_mcmc.is_none());
    // rho = n / N_act is refused with N_act, not reported as 0%
    assert_eq!(d.kpi.inferred_surveillance_sampling_fraction_pct, None);
    assert_eq!(d.macro_scaling.surveillance_coverage_rho, None);

    let json: serde_json::Value = serde_json::to_value(&d).unwrap();
    assert!(json["kpi"]["giant_component_reduction_pct"].is_null());
    assert!(json["kpi"]["inferred_active_transmitting_pool"].is_null());
    assert!(json["kpi"]["inferred_active_pool_ci"].is_null());
    assert!(json["macro_scaling"]["active_population"].is_null());
    assert!(json["kpi"]["inferred_surveillance_sampling_fraction_pct"].is_null());
    assert!(json["macro_scaling"]["surveillance_coverage_rho"].is_null());
}

/// Control (control.fasta, same sequences with each identical pair sampled half
/// a year apart): run_danno gives DANNO edges 3, clusters 3, max_cluster 2 and
/// static max_cluster 2 (control.py_danno.json). N_act at n = 6, E_obs = 3 from
/// the Python ChinEstimator (phase2/c5-zero-edge/chin_oracle.json): 9.0
/// [2.902692757565392, 27.905123540507965]. The inversion is unchanged when E_obs > 0.
#[test]
fn test_supported_edges_keep_estimates() {
    let fasta = include_str!("fixtures/zero_edge/control.fasta");
    let alignment = Alignment::from_fasta_str(fasta).unwrap();
    let d = run_trace50(&alignment, &Trace50Config::default()).unwrap();
    assert_eq!(d.kpi.supported_transmission_edges, 3);
    assert_eq!(d.kpi.danno_network.total_clusters, 3);
    assert_eq!(d.kpi.danno_network.max_cluster_size, 2);
    assert_eq!(d.kpi.static_network.max_cluster_size, 2);
    assert_eq!(d.kpi.giant_component_reduction_pct, Some(0.0));
    assert_eq!(d.kpi.inferred_active_pool_status, "estimated");
    let n_act = d.kpi.inferred_active_transmitting_pool.unwrap();
    let ci = d.kpi.inferred_active_pool_ci.unwrap();
    for (got, want) in [
        (n_act, 9.0),
        (ci[0], 2.902692757565392),
        (ci[1], 27.905123540507965),
    ] {
        assert!(
            (got - want).abs() <= 1e-8 * want,
            "{} vs python {}",
            got,
            want
        );
    }
    assert!(d.macro_scaling.active_population.is_some());
    // rho from the Python ChinEstimator(1.5).estimate_sampling_fraction(6, 3)
    // (phase2/verifier-fixes/c_rho/chin_rho_oracle.json)
    let rho = d.macro_scaling.surveillance_coverage_rho.unwrap();
    assert!(
        (rho - CONTROL_RHO_PY).abs() <= 1e-12,
        "rho {} vs python {}",
        rho,
        CONTROL_RHO_PY
    );
    assert_eq!(
        d.kpi.inferred_surveillance_sampling_fraction_pct,
        Some(rho * 100.0)
    );
}

/// The control dates span 0.5 yr, so run_danno.py keeps t_span = max(span, 1.0)
/// = 1.0 (calibration only overrides it when span > 0.5). Its per-edge Bayes
/// factors (control.py_danno.json.edges.csv) are reproduced only if the
/// pipeline starts from the data span rather than a fixed 20-year window,
/// which inflated every BF 20-fold on single-year cohorts. Tolerance 1e-4 as
/// in the parity harness's internal-TN93 path: the oracle calibrates on tn93's
/// 6-significant-digit distances (measured deviation here 8.4e-6).
#[test]
fn test_short_span_bayes_factors_match_python() {
    let fasta = include_str!("fixtures/zero_edge/control.fasta");
    let oracle = include_str!("fixtures/zero_edge/control.py_danno.json.edges.csv");
    let alignment = Alignment::from_fasta_str(fasta).unwrap();
    let d = run_trace50(&alignment, &Trace50Config::default()).unwrap();
    let mut n = 0;
    for line in oracle.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let want: f64 = f[4].parse().unwrap();
        let e = d
            .edges
            .iter()
            .find(|e| {
                (e.source == f[0] && e.target == f[1]) || (e.source == f[1] && e.target == f[0])
            })
            .expect("oracle edge present in dossier");
        assert!(
            (e.bayes_factor - want).abs() <= 1e-4 * want,
            "BF {} vs python {}",
            e.bayes_factor,
            want
        );
        n += 1;
    }
    assert_eq!(n, 3);
}
