//! Sequence length L parity with the Python reference (`validation/run_danno.py`).
//!
//! L is the median per-sequence non-gap length (numpy median, `int()`
//! truncation), not the alignment width. Expected values are read from
//! `fixtures/seq_len/oracle.json`, written by `fixtures/seq_len/gen_oracle.py`
//! (run_danno.py's L logic and `trace50.bayes_factor.DannoEstimator`).

use serde_json::Value;
use trace50::{run_trace50, run_trace50_with_matrix, Alignment, DannoEstimator, Trace50Config};

const FASTA: &str = include_str!("fixtures/seq_len/ragged_toy.fasta");

fn oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/seq_len/oracle.json")).unwrap()
}

/// Ragged rows with an even count of sequences whose middle non-gap counts
/// average to x.5, so numpy's median has to be truncated.
#[test]
fn test_effective_len_matches_run_danno() {
    let o = oracle();
    let aln = Alignment::from_fasta_str(FASTA).unwrap();
    assert_eq!(
        aln.effective_len as u64,
        o["effective_len"].as_u64().unwrap()
    );
    assert_eq!(aln.seq_len as u64, o["alignment_width"].as_u64().unwrap());
    assert_ne!(aln.effective_len, aln.seq_len);
}

#[test]
fn test_bayes_factor_at_effective_len_matches_python() {
    let o = oracle();
    let aln = Alignment::from_fasta_str(FASTA).unwrap();
    let p = &o["params"];
    let f = |k: &str| p[k].as_f64().unwrap();
    let est = DannoEstimator::new(
        aln.effective_len,
        f("mu"),
        f("tau_bar"),
        f("omega"),
        f("t_span"),
        f("alpha_adequacy"),
        f("bg_mean_dist"),
        f("bg_var_dist"),
    );
    for (name, got) in [
        ("p_trans", est.p_trans),
        ("p_bg", est.p_bg),
        ("r_bg", est.r_bg),
    ] {
        let want = o[name].as_f64().unwrap();
        assert!(
            (got - want).abs() <= 1e-9 * want.abs(),
            "{}: rust {} python {}",
            name,
            got,
            want
        );
    }
    let cases = o["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    for c in cases {
        let (d, dt) = (c["dist"].as_f64().unwrap(), c["delta_t"].as_f64().unwrap());
        let want = c["bf"].as_f64().unwrap();
        let got = est.compute_bayes_factor(d, dt);
        assert!(
            (got - want).abs() <= 1e-9 * want.abs(),
            "BF(d={}, dt={}): rust {} python {}",
            d,
            dt,
            got,
            want
        );
    }
}

/// The pipeline defaults to the median non-gap length, reports the width
/// separately, and honours an explicit `seq_len` override.
#[test]
fn test_pipeline_uses_effective_len_and_override() {
    let o = oracle();
    let l = o["effective_len"].as_u64().unwrap() as usize;
    let width = o["alignment_width"].as_u64().unwrap() as usize;
    let mut aln = Alignment::from_fasta_str(FASTA).unwrap();
    let n = aln.num_sequences();
    // The toy headers carry no dates, and undated sequences are not imputed
    // (the pipeline refuses to run without them); L does not depend on dates.
    aln.dates = vec![2020.0; n];

    let dossier = run_trace50(&aln, &Trace50Config::default()).unwrap();
    assert_eq!(dossier.metadata.seq_len, l);
    assert_eq!(dossier.metadata.alignment_width, width);

    // One static-range pair, so the dossier reports k = d * L for it.
    let d = 0.01;
    let mut m = vec![f64::NAN; n * n];
    for i in 0..n {
        m[i * n + i] = 0.0;
    }
    m[1] = d;
    m[n] = d;
    for (cfg_len, want_l) in [(None, l), (Some(1000usize), 1000usize)] {
        let cfg = Trace50Config {
            seq_len: cfg_len,
            ..Trace50Config::default()
        };
        let dossier = run_trace50_with_matrix(&aln, &cfg, Some(m.clone())).unwrap();
        assert_eq!(dossier.metadata.seq_len, want_l);
        let e = dossier
            .edges
            .iter()
            .find(|e| {
                (e.source_idx, e.target_idx) == (0, 1) || (e.source_idx, e.target_idx) == (1, 0)
            })
            .expect("static pair missing from dossier");
        assert!((e.k_substitutions - d * want_l as f64).abs() < 1e-12);
    }
}
