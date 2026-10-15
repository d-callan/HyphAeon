//! Rust-vs-Python parity harness on a fixed 200-sequence subset of the
//! Washington, DC cohort (golden files in `tests/fixtures/parity/`).
//!
//! The golden values were produced by external tools, not by this crate:
//! `tn93 -t 1.0 -a RMYSKB` (distances), `validation/run_danno.py` fed those
//! distances (Python path (b): supported edges with BF / PEP / q, calibrated
//! background, cluster partition, static 1.5% partition), `hivnetworkcsv -t
//! 0.015` (static partition, checked equal at generation time) and
//! `trace50.chin.ChinEstimator` (N_act, CI, Borel statistics). The fixture
//! README lists the commands and tool versions; regenerate them with
//! `python3 scripts/regen_parity_golden.py`. Neither Python nor tn93 is
//! needed to run these tests.
//!
//! Edge sets are compared as sets of unordered node-ID pairs and partitions
//! as sets of sets of node IDs, so labels and output order do not matter.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use trace50::{
    run_trace50, run_trace50_with_matrix, Alignment, DannoEstimator, Trace50Config,
    VisualizationDossier,
};

const FASTA: &str = include_str!("fixtures/parity/subset.fasta");
const META: &str = include_str!("fixtures/parity/meta.csv");
const DISTANCES: &str = include_str!("fixtures/parity/golden_distances.csv");
const GOLDEN: &str = include_str!("fixtures/parity/golden.json");

/// The fixture's distances are tn93's 6-significant-digit text, so the
/// internal (full-precision) TN93 can differ from them by rounding only.
const TN93_ABS_TOL: f64 = 1e-6;
/// Relative tolerance for values computed from identical inputs (the golden
/// distances): only floating-point evaluation order differs.
const EXACT_REL_TOL: f64 = 1e-9;
/// Relative tolerance for BF / PEP / q / calibration when the inputs are the
/// internal distances, which differ from tn93's printed ones by rounding.
const INTERNAL_REL_TOL: f64 = 1e-4;

type Pair = (String, String);
type Partition = BTreeSet<BTreeSet<String>>;

fn pair(a: &str, b: &str) -> Pair {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn golden() -> Value {
    serde_json::from_str(GOLDEN).unwrap()
}

fn alignment() -> Alignment {
    Alignment::from_fasta_str(FASTA).unwrap()
}

fn rel_close(got: f64, want: f64, tol: f64) -> bool {
    if want == 0.0 {
        got.abs() <= tol
    } else {
        ((got - want) / want).abs() <= tol
    }
}

/// Golden tn93 distances keyed by unordered ID pair.
fn golden_pairs(aln: &Alignment) -> BTreeMap<Pair, f64> {
    let mut out = BTreeMap::new();
    for line in DISTANCES.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let (i, j): (usize, usize) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        out.insert(
            pair(&aln.headers[i], &aln.headers[j]),
            f[2].parse::<f64>().unwrap(),
        );
    }
    out
}

/// The golden distances as an n x n matrix, NaN where tn93 reported no pair.
fn golden_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![f64::NAN; n * n];
    for i in 0..n {
        m[i * n + i] = 0.0;
    }
    for line in DISTANCES.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let (i, j): (usize, usize) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        let d: f64 = f[2].parse().unwrap();
        m[i * n + j] = d;
        m[j * n + i] = d;
    }
    m
}

fn golden_partition(g: &Value, key: &str) -> Partition {
    g[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            c.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect()
        })
        .collect()
}

/// Partition of the nodes with a nonzero cluster label, as sets of IDs.
fn dossier_partition(
    d: &VisualizationDossier,
    label: impl Fn(&trace50::visualization::DossierNode) -> usize,
) -> Partition {
    let mut by: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    for node in &d.nodes {
        let c = label(node);
        if c > 0 {
            by.entry(c).or_default().insert(node.id.clone());
        }
    }
    by.into_values().collect()
}

/// Config matching run_danno.py's defaults as recorded in golden.json; L is
/// left to the pipeline default (median non-gap length).
fn config(g: &Value) -> Trace50Config {
    let p = &g["params"];
    let f = |k: &str| p[k].as_f64().unwrap();
    Trace50Config {
        mu: f("mu"),
        tau_bar: f("tau_bar"),
        omega: f("omega"),
        fdr_threshold: f("fdr"),
        max_delta_t: f("max_delta_t"),
        prior_odds: f("prior_odds"),
        static_distance_threshold: g["static_threshold"].as_f64().unwrap(),
        r0: g["chin"]["r0"].as_f64().unwrap(),
        ..Trace50Config::default()
    }
}

#[test]
fn parity_alignment_length_and_dates() {
    let g = golden();
    let aln = alignment();
    assert_eq!(
        aln.num_sequences() as u64,
        g["n_sequences"].as_u64().unwrap()
    );
    assert_eq!(
        aln.seq_len as u64,
        g["alignment_width"].as_u64().unwrap(),
        "alignment width"
    );
    assert_eq!(
        aln.effective_len as u64,
        g["seq_len"].as_u64().unwrap(),
        "default L (median non-gap)"
    );
    assert_ne!(
        aln.effective_len, aln.seq_len,
        "fixture must distinguish L from the width"
    );

    // meta.csv holds run_danno.parse_date of each header
    let meta: BTreeMap<&str, f64> = META
        .lines()
        .skip(1)
        .map(|l| {
            let (id, d) = l.split_once(',').unwrap();
            (id, d.parse().unwrap())
        })
        .collect();
    assert_eq!(meta.len(), aln.num_sequences());
    for (h, &d) in aln.headers.iter().zip(&aln.dates) {
        let want = meta[h.as_str()];
        assert!(
            (d - want).abs() <= 1e-12,
            "date of {}: rust {} python {}",
            h,
            d,
            want
        );
    }
}

#[test]
fn parity_internal_tn93_matches_tn93_binary() {
    let g = golden();
    let aln = alignment();
    let want = golden_pairs(&aln);
    assert_eq!(want.len() as u64, g["tn93_pairs"].as_u64().unwrap());

    let n = aln.num_sequences();
    let m = aln.compute_distance_matrix();
    let mut got = BTreeMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if m[i * n + j].is_finite() {
                got.insert(pair(&aln.headers[i], &aln.headers[j]), m[i * n + j]);
            }
        }
    }
    let got_keys: BTreeSet<&Pair> = got.keys().collect();
    let want_keys: BTreeSet<&Pair> = want.keys().collect();
    let extra: Vec<_> = got_keys.difference(&want_keys).take(5).collect();
    let missing: Vec<_> = want_keys.difference(&got_keys).take(5).collect();
    assert!(
        extra.is_empty() && missing.is_empty(),
        "pair set differs from tn93 -t 1.0 -a RMYSKB: {} pairs vs {}; extra (first 5) {:?}; missing (first 5) {:?}",
        got.len(),
        want.len(),
        extra,
        missing
    );
    let mut worst = (0.0f64, None);
    for (k, &w) in &want {
        let diff = (got[k] - w).abs();
        if diff > worst.0 {
            worst = (diff, Some(k.clone()));
        }
    }
    assert!(
        worst.0 <= TN93_ABS_TOL,
        "max |d_rust - d_tn93| = {} at {:?}",
        worst.0,
        worst.1
    );
    eprintln!(
        "[parity] internal TN93: {} pairs = tn93 pair set; max |d_rust - d_tn93| {:.3e}",
        got.len(),
        worst.0
    );
}

#[test]
fn parity_background_calibration() {
    let g = golden();
    let aln = alignment();
    let n = aln.num_sequences();
    let want_p = g["calibrated"]["p_bg"].as_f64().unwrap();
    let want_r = g["calibrated"]["r_bg"].as_f64().unwrap();
    let want_t = g["calibrated"]["t_span"].as_f64().unwrap();
    let cfg = config(&g);

    for (label, matrix, tol) in [
        ("golden distances", golden_matrix(n), EXACT_REL_TOL),
        (
            "internal TN93",
            aln.compute_distance_matrix(),
            INTERNAL_REL_TOL,
        ),
    ] {
        let mut est = DannoEstimator::new(
            aln.effective_len,
            cfg.mu,
            cfg.tau_bar,
            cfg.omega,
            cfg.t_span,
            cfg.alpha_adequacy,
            0.065,
            0.0004,
        );
        est.calibrate_from_data(&matrix, n, &aln.dates);
        assert!(
            rel_close(est.p_bg, want_p, tol),
            "{}: p_bg rust {} python {}",
            label,
            est.p_bg,
            want_p
        );
        assert!(
            rel_close(est.r_bg, want_r, tol),
            "{}: r_bg rust {} python {}",
            label,
            est.r_bg,
            want_r
        );
        assert!(
            rel_close(est.t_span, want_t, EXACT_REL_TOL),
            "{}: t_span rust {} python {}",
            label,
            est.t_span,
            want_t
        );
        eprintln!(
            "[parity] calibration ({}): p_bg rel dev {:.3e}, r_bg rel dev {:.3e}",
            label,
            ((est.p_bg - want_p) / want_p).abs(),
            ((est.r_bg - want_r) / want_r).abs()
        );
    }
}

/// Every downstream comparison for one pipeline run.
fn assert_pipeline_matches_golden(label: &str, d: &VisualizationDossier, tol: f64) {
    let g = golden();

    assert_eq!(
        d.metadata.seq_len as u64,
        g["seq_len"].as_u64().unwrap(),
        "{}: L used",
        label
    );
    assert_eq!(
        d.metadata.alignment_width as u64,
        g["alignment_width"].as_u64().unwrap(),
        "{}: width",
        label
    );

    // supported edge set, then BF / PEP / q per edge
    let want: BTreeMap<Pair, &Value> = g["supported_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                pair(e["id1"].as_str().unwrap(), e["id2"].as_str().unwrap()),
                e,
            )
        })
        .collect();
    let got: BTreeMap<Pair, _> = d
        .edges
        .iter()
        .filter(|e| e.status == "supported_transmission")
        .map(|e| (pair(&e.source, &e.target), e))
        .collect();
    let got_keys: BTreeSet<&Pair> = got.keys().collect();
    let want_keys: BTreeSet<&Pair> = want.keys().collect();
    assert!(
        got_keys == want_keys,
        "{}: supported edges {} vs Python {}; Rust-only {:?}; Python-only {:?}",
        label,
        got.len(),
        want.len(),
        got_keys.difference(&want_keys).collect::<Vec<_>>(),
        want_keys.difference(&got_keys).collect::<Vec<_>>()
    );
    assert_eq!(
        d.kpi.supported_transmission_edges,
        want.len(),
        "{}: KPI edge count",
        label
    );
    let mut worst_rel = 0.0f64;
    for (k, w) in &want {
        let e = got[k];
        let wdt = w["delta_t"].as_f64().unwrap();
        assert!(
            (e.delta_t_years - wdt).abs() <= 1e-9,
            "{}: {:?} delta_t rust {} python {}",
            label,
            k,
            e.delta_t_years,
            wdt
        );
        for (name, gv) in [
            ("bayes_factor", e.bayes_factor),
            ("pep", e.pep),
            ("q_value", e.q_value),
        ] {
            let wv = w[name].as_f64().unwrap();
            assert!(
                rel_close(gv, wv, tol),
                "{}: {:?} {} rust {} python {}",
                label,
                k,
                name,
                gv,
                wv
            );
            worst_rel = worst_rel.max(((gv - wv) / wv).abs());
        }
    }
    eprintln!(
        "[parity] {}: {} supported edges match; max relative BF/PEP/q deviation {:.3e}",
        label,
        want.len(),
        worst_rel
    );

    // partitions
    let danno = dossier_partition(d, |n| n.cluster_id_supported);
    assert_eq!(
        danno,
        golden_partition(&g, "danno_partition"),
        "{}: DANNO cluster partition",
        label
    );
    let stat = dossier_partition(d, |n| n.cluster_id_static);
    assert_eq!(
        stat,
        golden_partition(&g, "static_partition"),
        "{}: static 1.5% partition",
        label
    );

    // CHIN
    let c = &g["chin"];
    let ap = d
        .macro_scaling
        .active_population
        .as_ref()
        .expect("N_act estimable");
    let wa = &c["active_population"];
    for (name, gv, key) in [
        ("N_act", ap.n_act, "N_act"),
        ("ci_lower", ap.ci_lower, "ci_lower"),
        ("ci_upper", ap.ci_upper, "ci_upper"),
        ("relative_error", ap.relative_error, "relative_error"),
    ] {
        let wv = wa[key].as_f64().unwrap();
        assert!(
            rel_close(gv, wv, EXACT_REL_TOL),
            "{}: CHIN {} rust {} python {}",
            label,
            name,
            gv,
            wv
        );
    }
    let b = &d.macro_scaling.borel_decay;
    let wb = &c["borel"];
    for (name, gv) in [
        ("slope", b.slope),
        ("r_squared", b.r_squared),
        ("p_value", b.p_value),
    ] {
        let wv = wb[name].as_f64().unwrap();
        assert!(
            rel_close(gv, wv, EXACT_REL_TOL),
            "{}: Borel {} rust {} python {}",
            label,
            name,
            gv,
            wv
        );
    }
    assert_eq!(
        b.n_clusters as u64,
        wb["n_clusters"].as_u64().unwrap(),
        "{}: Borel n_clusters",
        label
    );
    assert_eq!(
        b.n_megaclusters as u64,
        wb["n_megaclusters"].as_u64().unwrap(),
        "{}: Borel n_megaclusters",
        label
    );
    assert_eq!(
        b.rejects_uniform_sampling,
        wb["rejects_uniform_sampling"].as_bool().unwrap(),
        "{}: Borel rejects",
        label
    );
}

#[test]
fn parity_pipeline_internal_tn93() {
    let g = golden();
    let d = run_trace50(&alignment(), &config(&g)).unwrap();
    assert_pipeline_matches_golden("internal TN93", &d, INTERNAL_REL_TOL);
}

#[test]
fn parity_pipeline_golden_distance_matrix() {
    let g = golden();
    let aln = alignment();
    let n = aln.num_sequences();
    let d = run_trace50_with_matrix(&aln, &config(&g), Some(golden_matrix(n))).unwrap();
    assert_pipeline_matches_golden("--distances (golden tn93)", &d, EXACT_REL_TOL);
}
