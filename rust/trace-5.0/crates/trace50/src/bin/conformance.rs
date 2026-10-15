//! Conformance driver: dumps the Rust pipeline's intermediate quantities as
//! JSON so the Python test suite can compare them against the Python
//! implementations (aeon-core / chronaeon). Test infrastructure only — not a
//! user-facing CLI.
//!
//! Usage: conformance <fasta_path> <out.json>

use std::fs;

use serde::Serialize;
use trace50::autoclock::recursive_autoclock_deconvolution;
use trace50::tn93::Alignment;

#[derive(Serialize)]
struct CommunityOut {
    path: String,
    members: Vec<usize>,
    size: usize,
    mu: f64,
    r_squared: f64,
    tmrca: f64,
    mean_pairwise_dist: f64,
    time_span_years: f64,
    stop_reason: String,
    operational_tier: String,
}

#[derive(Serialize)]
struct ConformanceDump {
    names: Vec<String>,
    // JSON cannot represent NaN; unparseable dates serialize as null.
    dates: Vec<Option<f64>>,
    n: usize,
    // JSON cannot represent NaN; unscored pairs serialize as null.
    distance_matrix_flat: Vec<Option<f64>>,
    num_undated: usize,
    communities: Vec<CommunityOut>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: conformance <fasta_path> <out.json>");
        std::process::exit(2);
    }
    let fasta_path = &args[1];
    let out_path = &args[2];

    let content = fs::read_to_string(fasta_path)
        .unwrap_or_else(|e| panic!("failed to read {fasta_path}: {e}"));
    let aln = Alignment::from_fasta_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse {fasta_path}: {e}"));

    let n = aln.num_sequences();
    let dates: Vec<Option<f64>> = aln
        .dates
        .iter()
        .map(|&d| if d.is_nan() { None } else { Some(d) })
        .collect();

    let d_matrix = aln.compute_distance_matrix();

    // Mirror the pipeline invocation: min_cluster_size and max_recursion_depth
    // use the pipeline defaults; conductance cut 0.65 as in pipeline.rs.
    // AutoClock runs on the dated subset only (undated sequences are excluded,
    // not imputed); member indices map back to the alignment.
    let dates_raw: Vec<f64> = dates.iter().map(|d| d.unwrap_or(f64::NAN)).collect();
    let dated: Vec<usize> = (0..n).filter(|&i| dates_raw[i].is_finite()).collect();
    let communities = recursive_autoclock_deconvolution(
        &dated, &dates_raw, &d_matrix, n, 3,  // min_cluster_size (pipeline default)
        10, // max_recursion_depth (pipeline default)
        0,  // depth
        "root", 0.65, // conductance_cut (pipeline value)
    );

    let dump = ConformanceDump {
        names: aln.headers.clone(),
        dates,
        n,
        distance_matrix_flat: d_matrix
            .iter()
            .map(|&d| if d.is_nan() { None } else { Some(d) })
            .collect(),
        num_undated: n - dated.len(),
        communities: communities
            .iter()
            .map(|c| CommunityOut {
                path: c.path.clone(),
                members: c.members.clone(),
                size: c.size,
                mu: c.mu,
                r_squared: c.r_squared,
                tmrca: c.tmrca,
                mean_pairwise_dist: c.mean_pairwise_dist,
                time_span_years: c.time_span_years,
                stop_reason: c.stop_reason.clone(),
                operational_tier: c.operational_tier.clone(),
            })
            .collect(),
    };

    let json = serde_json::to_string_pretty(&dump).expect("serialize");
    fs::write(out_path, json).expect("write out.json");
}
