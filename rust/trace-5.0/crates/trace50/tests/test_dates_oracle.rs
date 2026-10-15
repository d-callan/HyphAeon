//! Collection-date parity with the Python reference (C4).
//!
//! Expected values are read from `fixtures/dates/oracle.json` and
//! `fixtures/dates/dc_headers.csv`, both written by
//! `fixtures/dates/gen_oracle.py`, which calls `validation/run_danno.py`
//! (`parse_date`, `parse_dc`, `ymd_to_decimal`) and
//! `validation/fetch_dates.py` (`parse_collection_date`). Values compare
//! exactly: the Rust arithmetic is the same sequence of f64 operations.

use serde_json::Value;
use trace50::dates::{parse_date_from_header, parse_date_value};
use trace50::{run_trace50_with_matrix, Alignment, Trace50Config};

fn oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/dates/oracle.json")).unwrap()
}

fn expected(row: &Value) -> Option<f64> {
    row["expected"].as_str().map(|s| s.parse::<f64>().unwrap())
}

#[test]
fn test_header_dates_match_run_danno() {
    let o = oracle();
    for row in o["headers"].as_array().unwrap() {
        let h = row["input"].as_str().unwrap();
        let got = parse_date_from_header(h);
        match expected(row) {
            Some(e) => assert_eq!(got, e, "header {:?} ({})", h, row["rule"]),
            None => assert!(
                got.is_nan(),
                "header {:?}: expected undated, got {}",
                h,
                got
            ),
        }
    }
}

#[test]
fn test_metadata_dates_match_fetch_dates() {
    let o = oracle();
    for row in o["meta"].as_array().unwrap() {
        let s = row["input"].as_str().unwrap();
        assert_eq!(
            parse_date_value(s),
            expected(row),
            "value {:?} ({})",
            s,
            row["rule"]
        );
    }
}

/// All 1,879 DC headers (Perez-Losada 2017) against run_danno.parse_date.
#[test]
fn test_all_dc_headers_match_run_danno() {
    let csv = include_str!("fixtures/dates/dc_headers.csv");
    let mut n = 0;
    for line in csv.lines().skip(1) {
        let (h, e) = line.rsplit_once(',').unwrap();
        let e: f64 = e.parse().unwrap();
        assert_eq!(parse_date_from_header(h), e, "{}", h);
        n += 1;
    }
    assert_eq!(n, 1879);
}

/// Dates come from the whole defline, while the sequence ID is its first word.
#[test]
fn test_alignment_reads_date_from_full_defline() {
    let seq = "ACGT".repeat(40);
    let fasta = format!(
        ">MH074765.1 HIV-1 isolate p6s6_2013 from Germany\n{}\n>seq2\n{}\n",
        seq, seq
    );
    let aln = Alignment::from_fasta_str(&fasta).unwrap();
    assert_eq!(aln.headers[0], "MH074765.1");
    assert_eq!(aln.dates[0], 2013.5);
    assert!(aln.dates[1].is_nan());
}

/// Undated sequences are excluded from DANNO and from the static network, as
/// run_danno.py drops them before both arms: the dated sequences give the same
/// scored dyads (same background calibration, BF, q), the same static and
/// DANNO networks and the same CHIN values whether or not undated sequences
/// are in the alignment, and no edge touches an undated sequence.
#[test]
fn test_undated_sequences_excluded_not_imputed() {
    // distinct sequences so the background calibration has spread
    let base: Vec<u8> = b"ACGTTGCAAGCTTCGAGATC".repeat(10);
    let mut seqs = Vec::new();
    for k in 0..8u8 {
        let mut s = base.clone();
        for p in 0..(k as usize * 3) {
            let pos = (p * 37 + k as usize * 11) % s.len();
            s[pos] = match s[pos] {
                b'A' => b'G',
                b'G' => b'A',
                b'C' => b'T',
                _ => b'C',
            };
        }
        seqs.push(String::from_utf8(s).unwrap());
    }
    let dated_headers = [
        "s0_2010.1",
        "s1_2010.4",
        "s2_2011.2",
        "s3_2011.9",
        "s4_2012.3",
        "s5_2012.8",
    ];
    let mut with_undated = String::new();
    let mut only_dated = String::new();
    for (i, h) in dated_headers.iter().enumerate() {
        with_undated += &format!(">{}\n{}\n", h, seqs[i]);
        only_dated += &format!(">{}\n{}\n", h, seqs[i]);
        if i == 2 {
            with_undated += &format!(">u0\n{}\n", seqs[6]);
        }
    }
    with_undated += &format!(">u1\n{}\n", seqs[7]);

    let cfg = Trace50Config {
        max_delta_t: 5.0,
        seq_len: Some(200),
        ..Trace50Config::default()
    };
    let a = Alignment::from_fasta_str(&with_undated).unwrap();
    let b = Alignment::from_fasta_str(&only_dated).unwrap();
    assert_eq!(a.dates.iter().filter(|d| d.is_nan()).count(), 2);

    let da = run_trace50_with_matrix(&a, &cfg, None).unwrap();
    let db = run_trace50_with_matrix(&b, &cfg, None).unwrap();
    assert_eq!(da.metadata.num_undated, 2);
    assert_eq!(db.metadata.num_undated, 0);

    let scored = |d: &trace50::VisualizationDossier| {
        let mut v: Vec<(String, String, String, String, String)> = d
            .edges
            .iter()
            .map(|e| {
                (
                    e.source.clone(),
                    e.target.clone(),
                    format!("{:e}", e.bayes_factor),
                    format!("{:e}", e.q_value),
                    e.status.clone(),
                )
            })
            .collect();
        v.sort();
        v
    };
    assert!(!scored(&db).is_empty());
    assert_eq!(scored(&da), scored(&db));
    for e in &da.edges {
        assert!(
            !e.source.starts_with('u') && !e.target.starts_with('u'),
            "{:?}",
            (&e.source, &e.target, &e.status)
        );
    }
    assert_eq!(
        serde_json::to_value(&da.kpi.static_network).unwrap(),
        serde_json::to_value(&db.kpi.static_network).unwrap()
    );
    assert_eq!(
        da.kpi.giant_component_reduction_pct,
        db.kpi.giant_component_reduction_pct
    );
    assert_eq!(
        da.kpi.inferred_active_transmitting_pool,
        db.kpi.inferred_active_transmitting_pool
    );
    assert_eq!(
        da.macro_scaling.surveillance_coverage_rho,
        db.macro_scaling.surveillance_coverage_rho
    );
    for node in &da.nodes {
        assert_eq!(node.id.starts_with('u'), node.date.is_nan());
    }
}

#[test]
fn test_no_dated_sequences_is_refused() {
    let seq = "ACGT".repeat(40);
    let fasta = format!(">a\n{}\n>b\n{}\n>c\n{}\n", seq, seq, seq);
    let aln = Alignment::from_fasta_str(&fasta).unwrap();
    let err = run_trace50_with_matrix(&aln, &Trace50Config::default(), None).unwrap_err();
    assert!(err.contains("collection dates"), "{}", err);
}
