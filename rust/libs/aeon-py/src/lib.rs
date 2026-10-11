//! aeon-py: PyO3 bindings for aeon-manifold kernels.
//!
//! Scope: thin pass-throughs only — every function delegates 1:1 to
//! `aeon_manifold` so Rust remains the single source of truth. Python-side
//! NaN contracts are preserved by returning f64::NAN unchanged. Sequences
//! cross the boundary as `bytes` (lossless for non-UTF-8 payloads).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use aeon_manifold::fasta::{normalize_lengths, tokenize_fasta, FastaRecord, ParseMode};

/// Numpy-linear percentile on an already-sorted input (NaN on empty).
#[pyfunction]
fn percentile(sorted: Vec<f64>, p: f64) -> f64 {
    aeon_manifold::percentile(&sorted, p)
}

/// Median of arbitrary input (sorts internally; NaN on empty).
#[pyfunction]
fn median(values: Vec<f64>) -> f64 {
    aeon_manifold::median(&values)
}

/// Fractional-year date parsed from a FASTA header (first-token-wins
/// semantics; NaN when no date token is found).
#[pyfunction]
fn parse_date_from_header(header: &str) -> f64 {
    aeon_manifold::temporal::parse_date_from_header(header)
}

/// IUPAC ambiguity bitmask for a nucleotide byte (0 for non-IUPAC input).
#[pyfunction]
fn char_to_iupac_mask(c: u8) -> u8 {
    aeon_manifold::char_to_iupac_mask(c)
}

/// Degenerate-base probability vector [A, C, G, T] for an IUPAC code byte
/// (uniform [0.25; 4] for mask-0 inputs, matching the trace50 contract).
#[pyfunction]
fn iupac_base_probabilities(c: u8) -> [f64; 4] {
    aeon_manifold::iupac_base_probabilities(c)
}

/// Raw FASTA records as (first-token name, uppercase sequence bytes).
#[pyfunction]
fn fasta_tokenize(text: &str) -> PyResult<Vec<(String, Vec<u8>)>> {
    tokenize_fasta(text)
        .map(|recs| recs.into_iter().map(|r| (r.name, r.seq)).collect())
        .map_err(PyValueError::new_err)
}

/// Length-normalize (name, seq-bytes) records; returns
/// (normalized_records, aligned_length). mode: "strict" or "pad".
#[pyfunction]
fn fasta_normalize_lengths(
    records: Vec<(String, Vec<u8>)>,
    mode: &str,
) -> PyResult<(Vec<(String, Vec<u8>)>, usize)> {
    let parsed = match mode {
        "strict" => ParseMode::Strict,
        "pad" => ParseMode::PadToMax,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown mode {other:?}; expected \"strict\" or \"pad\""
            )))
        }
    };
    let mut recs: Vec<FastaRecord> = records
        .into_iter()
        .map(|(name, seq)| FastaRecord { name, seq })
        .collect();
    let len = normalize_lengths(&mut recs, parsed).map_err(PyValueError::new_err)?;
    Ok((
        recs.into_iter().map(|r| (r.name, r.seq)).collect(),
        len,
    ))
}

/// Flat row-major `N x N` TN93 distance matrix from raw ASCII sequences.
/// Sequences may be ragged (pairs truncate to the shorter length, matching
/// the kernel contract). Non-IUPAC bytes encode as mask 0 and are skipped.
#[pyfunction]
fn tn93_distance_matrix(seqs: Vec<Vec<u8>>) -> Vec<f64> {
    let enc: Vec<Vec<u8>> = seqs
        .iter()
        .map(|s| s.iter().map(|&c| aeon_manifold::char_to_iupac_mask(c)).collect())
        .collect();
    aeon_manifold::tn93::distance_matrix(&enc)
}

/// Flat row-major `N x M` TN93 cross-distance matrix: each sequence in
/// `seqs_a` (rows) against each in `seqs_b` (columns, e.g. landmarks).
#[pyfunction]
fn tn93_cross_distance_matrix(seqs_a: Vec<Vec<u8>>, seqs_b: Vec<Vec<u8>>) -> Vec<f64> {
    let encode = |v: &[Vec<u8>]| -> Vec<Vec<u8>> {
        v.iter()
            .map(|s| s.iter().map(|&c| aeon_manifold::char_to_iupac_mask(c)).collect())
            .collect()
    };
    aeon_manifold::tn93::cross_distance_matrix(&encode(&seqs_a), &encode(&seqs_b))
}

/// MEME mixture-null p-values: 1/3·δ(0) + 2/3·(0.45·χ²₁ + 0.55·χ²₂).
#[pyfunction]
fn pvals_from_lrt_meme(lrts: Vec<f64>) -> Vec<f64> {
    aeon_manifold::stats::pvals_from_lrt_meme(&lrts)
}

/// Self & Liang (1987) mixture-null p-values: 0.5·δ(0) + 0.5·χ²₁.
#[pyfunction]
fn pvals_from_lrt_self_liang(lrts: Vec<f64>) -> Vec<f64> {
    aeon_manifold::stats::pvals_from_lrt_self_liang(&lrts)
}

/// Benjamini-Hochberg FDR q-values, clipped to [0, 1].
#[pyfunction]
fn benjamini_hochberg(pvals: Vec<f64>) -> Vec<f64> {
    aeon_manifold::stats::benjamini_hochberg(&pvals)
}

/// Cauchy Combination Test omnibus p-value (empty input → 1.0).
#[pyfunction]
fn cauchy_combination_p(pvals: Vec<f64>) -> f64 {
    aeon_manifold::stats::cauchy_combination_p(&pvals)
}

#[pymodule]
fn aeon_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(parse_date_from_header, m)?)?;
    m.add_function(wrap_pyfunction!(char_to_iupac_mask, m)?)?;
    m.add_function(wrap_pyfunction!(iupac_base_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(fasta_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(fasta_normalize_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(tn93_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(tn93_cross_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(pvals_from_lrt_meme, m)?)?;
    m.add_function(wrap_pyfunction!(pvals_from_lrt_self_liang, m)?)?;
    m.add_function(wrap_pyfunction!(benjamini_hochberg, m)?)?;
    m.add_function(wrap_pyfunction!(cauchy_combination_p, m)?)?;
    Ok(())
}
