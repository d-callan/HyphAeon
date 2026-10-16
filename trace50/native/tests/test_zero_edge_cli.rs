//! CLI scorecard with zero supported edges (fixture: trace50's
//! `tests/fixtures/zero_edge`). N_act must read "not estimable", the giant
//! component reduction "n/a" — the dossier JSON carries nulls and status
//! strings, never inf/NaN or a fake percentage. Exercises `cli_main`
//! in-process (the Python console script is a thin forwarder).

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    // <repo>/trace50/native -> <repo>/rust/trace-5.0/crates/trace50/tests/fixtures/zero_edge
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust/trace-5.0/crates/trace50/tests/fixtures/zero_edge")
}

fn run_cli(fixture: &str, out_dir: &Path) -> serde_json::Value {
    let fasta = fixture_dir().join(fixture);
    let argv: Vec<String> = vec![
        "trace50".into(),
        "-f".into(),
        fasta.to_str().unwrap().into(),
        "-o".into(),
        out_dir.to_str().unwrap().into(),
    ];
    let code = trace50_native::cli_main(argv);
    assert_eq!(code, 0, "trace50 exited {code} on {fixture}");
    let dossier_path = out_dir.join("trace50_visualization_dossier.json");
    let text = std::fs::read_to_string(&dossier_path)
        .unwrap_or_else(|e| panic!("dossier not written to {dossier_path:?}: {e}"));
    serde_json::from_str(&text).expect("dossier is valid JSON")
}

#[test]
fn test_cli_zero_edge_scorecard() {
    let out_dir = std::env::temp_dir().join("trace50_zero_edge_cli_static_only");
    let dossier = run_cli("static_only.fasta", &out_dir);
    let kpi = &dossier["kpi"];

    // CHIN refusal: N_act and its CI are null with an explicit status.
    assert!(
        kpi["inferred_active_transmitting_pool"].is_null(),
        "N_act must be null with zero supported edges: {kpi}"
    );
    assert_eq!(
        kpi["inferred_active_pool_status"],
        trace50::N_ACT_NOT_ESTIMABLE_NO_EDGES
    );
    assert!(kpi["inferred_active_pool_ci"].is_null());
    assert!(kpi["inferred_surveillance_sampling_fraction_pct"].is_null());

    // No supported edge -> no cluster to reduce: reduction is null.
    assert!(kpi["giant_component_reduction_pct"].is_null(), "{kpi}");

    // The DANNO giant component is 0, not a singleton node.
    assert_eq!(kpi["danno_network"]["max_cluster_size"], 0);
    assert_eq!(kpi["supported_transmission_edges"], 0);
}

#[test]
fn test_cli_control_scorecard_reports_n_act() {
    // Control: 3 supported edges (run_danno oracle); N_act = 9 [2.9 - 27.9]
    // (Python ChinEstimator).
    let out_dir = std::env::temp_dir().join("trace50_zero_edge_cli_control");
    let dossier = run_cli("control.fasta", &out_dir);
    let kpi = &dossier["kpi"];

    let n_act = kpi["inferred_active_transmitting_pool"]
        .as_f64()
        .expect("N_act estimable on control");
    assert!((n_act - 9.0).abs() < 0.5, "N_act = {n_act}");
    let ci = kpi["inferred_active_pool_ci"].as_array().unwrap();
    assert!(ci[0].as_f64().unwrap() > 2.0 && ci[0].as_f64().unwrap() < 4.0);
    assert!(ci[1].as_f64().unwrap() > 20.0 && ci[1].as_f64().unwrap() < 35.0);
    assert_eq!(kpi["inferred_active_pool_status"], "estimated");
    assert_eq!(kpi["giant_component_reduction_pct"], 0.0);
}
