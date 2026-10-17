//! Tamura-Nei 93 (TN93) pairwise genetic distance and IUPAC ambiguity resolution.
//!
//! The pairwise kernel lives in [`aeon_manifold::tn93`] (single source of
//! truth): a semantic port of the `tn93` binary (`tn93_shared.cc`), which
//! this module re-exports under the binary's reporting contract
//! (`tn93 -t 1.0 -l 100 -a <mode>`): pairs sharing `TN93_MIN_OVERLAP` or
//! fewer non-gap sites, or a distance above `TN93_MAX_REPORTED`, are
//! reported as `NaN` ("distance unavailable"). Trace50 always uses the
//! faithful counting scheme; chronaeon's pre-unification counting survives
//! only behind `aeon_manifold::tn93::Counting::Legacy`, which this crate
//! never selects.
//!
//! What remains here is trace50-specific glue: the [`Alignment`] parser
//! and distance-matrix driver, [`compute_fractional_substitutions`], and
//! [`median_nongap_len`].

use serde::{Deserialize, Serialize};

// Canonical IUPAC alphabet (aeon-manifold), re-exported so existing
// `crate::tn93::*` call sites are unchanged.
pub use aeon_manifold::encoding::{
    char_to_iupac_mask, char_to_nuc_int, iupac_base_probabilities, PURINE_MASK, PYRIMIDINE_MASK,
};
// Collection-date extraction from FASTA deflines (run_danno path-b rules).
pub use crate::dates::parse_date_from_header;

// The unified TN93 kernel (aeon-manifold), re-exported so every
// `trace50::tn93::*` and crate-level re-export is unchanged.
pub use aeon_manifold::tn93::{
    char_to_tn93_code, compute_pairwise_tn93, tn93_distance, AmbiguityMode, TN93_ALPHABET,
    TN93_GAP, TN93_MAX_REPORTED, TN93_MIN_OVERLAP, TN93_SATURATED,
};

/// Computes observed fractional substitution count K and genetic distance d
/// between two nucleotide sequences incorporating IUPAC ambiguities.
pub fn compute_fractional_substitutions(seq1: &[u8], seq2: &[u8]) -> (f64, usize, f64) {
    let n = seq1.len().min(seq2.len());
    if n == 0 {
        return (0.0, 0, 0.0);
    }

    let mut k_tot = 0.0;
    let mut valid_count = 0usize;

    for i in 0..n {
        let c1 = seq1[i];
        let c2 = seq2[i];

        if (c1 == b'-' || c1 == b'?') && (c2 == b'-' || c2 == b'?') {
            continue;
        }

        let m1 = iupac_base_probabilities(c1);
        let m2 = iupac_base_probabilities(c2);

        let shared = m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2] + m1[3] * m2[3];
        k_tot += 1.0 - shared;
        valid_count += 1;
    }

    let d_frac = if valid_count > 0 {
        k_tot / (valid_count as f64)
    } else {
        0.0
    };

    (k_tot, valid_count, d_frac)
}

/// Median number of non-gap characters per sequence, as `validation/run_danno.py`
/// computes the sequence length L: count the characters other than `-` in each
/// sequence, take the numpy median (mean of the two middle values when the
/// count is even), and truncate to an integer.
pub fn median_nongap_len(seqs: &[Vec<u8>]) -> usize {
    let mut counts: Vec<usize> = seqs
        .iter()
        .map(|s| s.iter().filter(|&&c| c != b'-').count())
        .collect();
    if counts.is_empty() {
        return 0;
    }
    counts.sort_unstable();
    let m = counts.len();
    if m % 2 == 1 {
        counts[m / 2]
    } else {
        // int((a + b) / 2) for non-negative a, b is floor division
        (counts[m / 2 - 1] + counts[m / 2]) / 2
    }
}

/// A parsed nucleotide FASTA alignment ready for TRACE-5.0 processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub headers: Vec<String>,
    pub dates: Vec<f64>,
    /// Alignment width (number of columns); ragged rows are padded to it.
    /// Used by site-wise computations such as InfoDANNO.
    pub seq_len: usize,
    /// Median per-sequence non-gap length, the default sequence length L for
    /// DANNO (see [`median_nongap_len`]).
    #[serde(default)]
    pub effective_len: usize,
    /// Per-sequence tn93 character codes (see [`char_to_tn93_code`]).
    pub encoded_tn93: Vec<Vec<u8>>,
    pub encoded_ints: Vec<Vec<i8>>,
    pub raw_seqs: Vec<Vec<u8>>,
}

impl Alignment {
    /// Parses an aligned FASTA string and extracts dates from the deflines.
    ///
    /// Tokenization and ragged-sequence padding delegate to
    /// `aeon_manifold::fasta` (`PadToMax` mode); each sequence is then
    /// encoded to tn93 codes and integer states. Dates are read from the
    /// whole defline, as run_danno.py does; headers without one are `NaN`
    /// ("undated") and are never imputed.
    pub fn from_fasta_str(fasta_content: &str) -> Result<Self, String> {
        use aeon_manifold::fasta::{normalize_lengths, tokenize_fasta, ParseMode};

        let mut records = tokenize_fasta(fasta_content)
            .map_err(|_| "No sequences found in FASTA content".to_string())?;
        let seq_len = normalize_lengths(&mut records, ParseMode::PadToMax)?;

        let n = records.len();
        let mut headers = Vec::with_capacity(n);
        let mut encoded_tn93 = Vec::with_capacity(n);
        let mut encoded_ints = Vec::with_capacity(n);
        let mut dates = Vec::with_capacity(n);
        let mut raw_seqs = Vec::with_capacity(n);
        let mut non_gap_counts = Vec::with_capacity(n);

        for rec in records {
            non_gap_counts.push(rec.seq.iter().filter(|&&c| c != b'-').count());
            encoded_tn93.push(rec.seq.iter().map(|&c| char_to_tn93_code(c)).collect());
            encoded_ints.push(rec.seq.iter().map(|&c| char_to_nuc_int(c)).collect());
            // Collection date from the full defline; NaN when it carries none.
            dates.push(parse_date_from_header(&rec.defline));
            headers.push(rec.name);
            raw_seqs.push(rec.seq);
        }

        // Median per-sequence non-gap count; equivalent to
        // median_nongap_len(&raw_seqs) without a second pass.
        non_gap_counts.sort_unstable();
        let m = non_gap_counts.len();
        let effective_len = if m == 0 {
            0
        } else if m % 2 == 1 {
            non_gap_counts[m / 2]
        } else {
            (non_gap_counts[m / 2 - 1] + non_gap_counts[m / 2]) / 2
        };

        Ok(Self {
            headers,
            dates,
            seq_len,
            effective_len,
            encoded_tn93,
            encoded_ints,
            raw_seqs,
        })
    }

    /// Number of sequences in the alignment.
    #[inline(always)]
    pub fn num_sequences(&self) -> usize {
        self.headers.len()
    }

    /// Computes full symmetric pairwise TN93 distance matrix.
    /// Returns a flattened `N x N` matrix in row-major order; pairs without
    /// sufficient overlap are `NaN` (see [`compute_pairwise_tn93`]).
    /// Ambiguities are handled as `tn93 -a RMYSKB`.
    #[inline]
    pub fn compute_distance_matrix(&self) -> Vec<f64> {
        self.compute_distance_matrix_with(&AmbiguityMode::default())
    }

    /// As [`Alignment::compute_distance_matrix`], with an explicit ambiguity mode.
    #[inline]
    pub fn compute_distance_matrix_with(&self, mode: &AmbiguityMode) -> Vec<f64> {
        self.compute_distance_matrix_with_mode_and_progress(mode, |_, _, _| {})
    }

    /// As [`Alignment::compute_distance_matrix`], with a progress callback.
    /// `on_progress(done_pairs, total_pairs, fraction)`
    #[inline]
    pub fn compute_distance_matrix_with_progress<F>(&self, on_progress: F) -> Vec<f64>
    where
        F: FnMut(usize, usize, f64),
    {
        self.compute_distance_matrix_with_mode_and_progress(&AmbiguityMode::default(), on_progress)
    }

    /// Full symmetric pairwise TN93 matrix with an explicit ambiguity mode and
    /// a progress callback `on_progress(done_pairs, total_pairs, fraction)`.
    pub fn compute_distance_matrix_with_mode_and_progress<F>(
        &self,
        mode: &AmbiguityMode,
        mut on_progress: F,
    ) -> Vec<f64>
    where
        F: FnMut(usize, usize, f64),
    {
        let n = self.num_sequences();
        let mut matrix = vec![0.0; n * n];
        let total_pairs = n * (n.saturating_sub(1)) / 2;
        let mut done_pairs = 0usize;
        let report_interval = (total_pairs / 100).clamp(200, 10000);

        for i in 0..n {
            for j in (i + 1)..n {
                let dist =
                    compute_pairwise_tn93(&self.encoded_tn93[i], &self.encoded_tn93[j], mode);
                matrix[i * n + j] = dist;
                matrix[j * n + i] = dist;
                done_pairs += 1;
                if done_pairs % report_interval == 0 || done_pairs == total_pairs {
                    let frac = done_pairs as f64 / total_pairs.max(1) as f64;
                    on_progress(done_pairs, total_pairs, frac);
                }
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tn93_identical() {
        let s1 = b"ACGTACGT".repeat(16); // 128 sites > TN93_MIN_OVERLAP
        let c1: Vec<u8> = s1.iter().map(|&c| char_to_tn93_code(c)).collect();
        let dist = compute_pairwise_tn93(&c1, &c1, &AmbiguityMode::default());
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_tn93_transition() {
        let s1 = b"ACGTACGTACGTACGTACGTACGT".repeat(5);
        let mut s2 = s1.clone();
        s2[0] = b'G'; // 1 transition in 120 bp
        let c1: Vec<u8> = s1.iter().map(|&c| char_to_tn93_code(c)).collect();
        let c2: Vec<u8> = s2.iter().map(|&c| char_to_tn93_code(c)).collect();
        let dist = compute_pairwise_tn93(&c1, &c2, &AmbiguityMode::default());
        println!("TN93 dist for 1/120 transition: {}", dist);
        assert!(dist > 0.0 && dist < 0.5);
    }

    #[test]
    fn test_ambiguity_mode_parse() {
        assert_eq!(
            AmbiguityMode::parse("resolve").unwrap(),
            AmbiguityMode::Resolve
        );
        assert_eq!(
            AmbiguityMode::parse("average").unwrap(),
            AmbiguityMode::Average
        );
        assert!(AmbiguityMode::parse("").is_err());
        let m = AmbiguityMode::default();
        assert_eq!(m, AmbiguityMode::parse("rmyskb").unwrap());
        for (code, &ch) in TN93_ALPHABET.iter().enumerate() {
            assert_eq!(char_to_tn93_code(ch) as usize, code);
            assert_eq!(
                m.resolves(code as u8),
                b"RMYSKB".contains(&ch),
                "{}",
                ch as char
            );
        }
    }

    #[test]
    fn test_fractional_substitutions_ambiguity() {
        let s1 = b"ACGT";
        let s2 = b"RCGT"; // R = A or G (0.5 shared with A)
        let (k, valid, d) = compute_fractional_substitutions(s1, s2);
        assert_eq!(valid, 4);
        assert!((k - 0.5).abs() < 1e-10);
        assert!((d - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_tn93_n_handling_matches_binary() {
        // 128 bp + 20 trailing N (or gap) per sequence, one difference at site 4.
        // Expected values from `tn93 -t 1.0 -a RMYSKB` v1.1.0 on these three
        // sequences: under RMYSKB, N is averaged rather than skipped, so the
        // shared N run counts as expected mismatch (a,b); against gaps it is
        // excluded (b,c).
        let base = b"ACGTACGT".repeat(16);
        let a = [base.as_slice(), &[b'N'; 20]].concat();
        let mut b = a.clone();
        b[4] = b'G';
        let c = [base.as_slice(), &[b'-'; 20]].concat();
        let enc = |s: &[u8]| -> Vec<u8> { s.iter().map(|&x| char_to_tn93_code(x)).collect() };
        let m = AmbiguityMode::default();
        for (x, y, want) in [(&a, &b, 0.116824), (&a, &c, 0.0), (&b, &c, 0.00793721)] {
            let d = compute_pairwise_tn93(&enc(x), &enc(y), &m);
            assert!((d - want).abs() <= 5e-7, "{} vs tn93 {}", d, want);
        }
    }
}
