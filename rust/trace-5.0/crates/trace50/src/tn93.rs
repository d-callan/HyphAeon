//! Tamura-Nei 93 (TN93) pairwise genetic distance and IUPAC ambiguity resolution.
//!
//! The pairwise kernel is a port of the `tn93` binary (`tn93_shared.cc`),
//! matching `tn93 -t 1.0 -l 100 -a <mode>`: pairs sharing `TN93_MIN_OVERLAP`
//! or fewer non-gap sites, or a distance above `TN93_MAX_REPORTED`, are
//! reported as `NaN` ("distance unavailable"). The canonical IUPAC alphabet
//! lives in aeon-manifold (single source of truth); the tn93-specific
//! 18-code alphabet used by the kernel is defined below.

use serde::{Deserialize, Serialize};

// Canonical IUPAC alphabet (aeon-manifold), re-exported so existing
// `crate::tn93::*` call sites are unchanged.
pub use aeon_manifold::encoding::{
    char_to_iupac_mask, char_to_nuc_int, iupac_base_probabilities, PURINE_MASK, PYRIMIDINE_MASK,
};
// Collection-date extraction from FASTA deflines (run_danno path-b rules).
pub use crate::dates::parse_date_from_header;

// ---------------------------------------------------------------------------
// Port of the `tn93` binary's pairwise distance (tn93 v1.0.15 source,
// src/tn93_shared.cc: alphabet tables and computeTN93; validated against the
// v1.1.0 binary). The binary, run as `tn93 -t 1.0 -a RMYSKB`, produced the
// distances the published TRACE-5.0 analyses consumed, so it is the oracle.
// ---------------------------------------------------------------------------

/// tn93 nucleotide alphabet; a character's code is its index here
/// (`ValidChars` in tn93_shared.cc).
pub const TN93_ALPHABET: &[u8; 18] = b"ACGTURYSWKMBDHVN?-";
/// Code of the gap character `-`.
pub const TN93_GAP: u8 = 17;
/// Code of `N`, used for the `gapmm` mode.
const TN93_N: u8 = 15;

/// Minimum overlap of the `tn93` binary (`-l`, default 100): a pair is only
/// scored when it shares MORE than this many sites at which neither sequence
/// has a gap (`tn93_shared.cc`, computeTN93: `if (totalNonGap <= min_overlap)
/// return -1.;`, and the -1 is never written out).
pub const TN93_MIN_OVERLAP: usize = 100;

/// tn93's saturation sentinel (`TN93_MAX_DIST`), returned when a log argument
/// of the TN93 (or K2P) formula is not positive.
pub const TN93_SATURATED: f64 = 1000.0;

/// Largest distance kept by [`compute_pairwise_tn93`]: the `-t 1.0` threshold
/// used to produce the reference distances (`thisD <= args.distance`). The
/// saturation sentinel is above it, so saturated pairs are missing as well.
pub const TN93_MAX_REPORTED: f64 = 1.0;

/// `resolutions` in tn93_shared.cc: which of A, C, G, T each code can be.
const TN93_RESOLUTIONS: [[bool; 4]; 18] = [
    [true, false, false, false],  // A
    [false, true, false, false],  // C
    [false, false, true, false],  // G
    [false, false, false, true],  // T
    [false, false, false, true],  // U
    [true, false, true, false],   // R
    [false, true, false, true],   // Y
    [false, true, true, false],   // S
    [true, false, false, true],   // W
    [false, false, true, true],   // K
    [true, true, false, false],   // M
    [false, true, true, true],    // B
    [true, false, true, true],    // D
    [true, true, false, true],    // H
    [true, true, true, false],    // V
    [true, true, true, true],     // N
    [true, true, true, true],     // ?
    [false, false, false, false], // -
];

/// `resolutionsCount` in tn93_shared.cc: weight given to each resolution.
const TN93_RESOLUTION_COUNT: [f64; 18] = [
    1.0, 1.0, 1.0, 1.0, 1.0,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
    0.25, 0.25,
    0.0,
];

/// Maps an ASCII character to its tn93 code (case-insensitive).
///
/// Characters outside the tn93 alphabet are mapped to a gap. (The tn93 binary
/// drops them from the sequence instead, which shifts the rest of the row and
/// usually ends in an alignment-length error; a gap keeps columns aligned.)
#[inline(always)]
pub fn char_to_tn93_code(c: u8) -> u8 {
    match c.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        b'U' => 4,
        b'R' => 5,
        b'Y' => 6,
        b'S' => 7,
        b'W' => 8,
        b'K' => 9,
        b'M' => 10,
        b'B' => 11,
        b'D' => 12,
        b'H' => 13,
        b'V' => 14,
        b'N' => 15,
        b'?' => 16,
        _ => TN93_GAP,
    }
}

/// How ambiguous nucleotides enter the distance (`tn93 -a`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AmbiguityMode {
    /// `resolve`: an ambiguity compatible with the other base counts as a match.
    Resolve,
    /// `average`: an ambiguity is spread over its resolutions (R vs A is
    /// 0.5 A:A + 0.5 G:A).
    Average,
    /// `skip`: sites with any non-ACGT character are ignored.
    Skip,
    /// `gapmm`: a gap opposite a non-gap is treated as N.
    GapMismatch,
    /// A list such as `RMYSKB`: listed ambiguities are resolved, all others
    /// averaged. Indexed by tn93 code.
    Subset([bool; 18]),
}

impl Default for AmbiguityMode {
    /// `RMYSKB`, the HIV-TRACE convention behind the reference distances.
    fn default() -> Self {
        AmbiguityMode::parse("RMYSKB").expect("valid default")
    }
}

impl AmbiguityMode {
    /// Parses a `tn93 -a` argument: `resolve`, `average`, `skip`, `gapmm`, or
    /// a list of ambiguity codes to resolve (case-insensitive; as in tn93,
    /// characters that are not ambiguity codes are ignored).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "resolve" => Ok(AmbiguityMode::Resolve),
            "average" => Ok(AmbiguityMode::Average),
            "skip" => Ok(AmbiguityMode::Skip),
            "gapmm" => Ok(AmbiguityMode::GapMismatch),
            "" => Err("empty TN93 ambiguity mode".to_string()),
            list => {
                let mut set = [false; 18];
                for c in list.bytes() {
                    let code = char_to_tn93_code(c);
                    // tn93 initAlphabets: only codes > 3 (non-ACGT) are marked;
                    // the gap check here also excludes characters outside the
                    // alphabet, which tn93 ignores.
                    if code > 3 && (code != TN93_GAP || c == b'-') {
                        set[code as usize] = true;
                    }
                }
                Ok(AmbiguityMode::Subset(set))
            }
        }
    }

    #[inline(always)]
    fn resolves(&self, code: u8) -> bool {
        match self {
            AmbiguityMode::Resolve => true,
            AmbiguityMode::Subset(set) => set[code as usize],
            _ => false,
        }
    }
}

/// Raw tn93 `computeTN93` for two code sequences over sites `lo..hi`.
///
/// Returns `None` when the pair shares `min_overlap` or fewer non-gap sites
/// (tn93 returns -1 and omits the pair), otherwise the distance, which is
/// [`TN93_SATURATED`] when the formula saturates. Base frequencies are
/// per pair: the row plus column sums of the pair's 4x4 table of (possibly
/// fractional, for averaged ambiguities) site counts. If any base frequency
/// is zero, the K2P formula is used, as in tn93. Floating-point operations
/// follow the order of the C++ source so values agree to rounding.
fn tn93_distance_range(
    s1: &[u8],
    s2: &[u8],
    lo: usize,
    hi: usize,
    mode: &AmbiguityMode,
    min_overlap: usize,
) -> Option<f64> {
    let mut float_counts = [[0.0f64; 4]; 4];
    let mut integer_counts = [[0u64; 4]; 4];
    let gapmm = matches!(mode, AmbiguityMode::GapMismatch);
    let skip = matches!(mode, AmbiguityMode::Skip);

    for p in lo..hi {
        let mut c1 = s1[p];
        let mut c2 = s2[p];
        if c1 < 4 && c2 < 4 {
            integer_counts[c1 as usize][c2 as usize] += 1;
            continue;
        }
        if c1 == TN93_GAP || c2 == TN93_GAP {
            if !gapmm || (c1 == TN93_GAP && c2 == TN93_GAP) {
                continue;
            }
            if c1 == TN93_GAP {
                c1 = TN93_N;
            } else {
                c2 = TN93_N;
            }
        }
        if skip {
            continue;
        }

        // ambiguityHandler
        let (u1, u2) = (c1 as usize, c2 as usize);
        if c1 < 4 {
            // c1 resolved, c2 not
            let w = TN93_RESOLUTION_COUNT[u2];
            if w > 0.0 {
                if mode.resolves(c2) && TN93_RESOLUTIONS[u2][u1] {
                    integer_counts[u1][u1] += 1;
                    continue;
                }
                for j in 0..4 {
                    if TN93_RESOLUTIONS[u2][j] {
                        float_counts[u1][j] += w;
                    }
                }
            }
        } else if c2 < 4 {
            // c2 resolved, c1 not
            let w = TN93_RESOLUTION_COUNT[u1];
            if w > 0.0 {
                if mode.resolves(c1) && TN93_RESOLUTIONS[u1][u2] {
                    integer_counts[u2][u2] += 1;
                    continue;
                }
                for i in 0..4 {
                    if TN93_RESOLUTIONS[u1][i] {
                        float_counts[i][u2] += w;
                    }
                }
            }
        } else {
            // both ambiguous
            let norm = TN93_RESOLUTION_COUNT[u1] * TN93_RESOLUTION_COUNT[u2];
            if norm > 0.0 {
                if mode.resolves(c1) && mode.resolves(c2) {
                    let mut positive = [false; 4];
                    let mut matched = 0usize;
                    for i in 0..4 {
                        if TN93_RESOLUTIONS[u1][i] && TN93_RESOLUTIONS[u2][i] {
                            positive[i] = true;
                            matched += 1;
                        }
                    }
                    if matched > 0 {
                        let norm2 = 1.0 / matched as f64;
                        for i in 0..4 {
                            if positive[i] {
                                float_counts[i][i] += norm2;
                            }
                        }
                        continue;
                    }
                }
                for i in 0..4 {
                    if TN93_RESOLUTIONS[u1][i] {
                        for j in 0..4 {
                            if TN93_RESOLUTIONS[u2][j] {
                                float_counts[i][j] += norm;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut nuc_freq = [0.0f64; 4];
    let mut total_non_gap = 0.0f64;
    for c1 in 0..4 {
        for c2 in 0..4 {
            float_counts[c1][c2] += integer_counts[c1][c2] as f64;
            let pc = float_counts[c1][c2];
            total_non_gap += pc;
            nuc_freq[c1] += pc;
            nuc_freq[c2] += pc;
        }
    }

    if total_non_gap <= min_overlap as f64 {
        return None;
    }

    let freq_sum = nuc_freq[0] + nuc_freq[1] + nuc_freq[2] + nuc_freq[3];
    let scale = 2.0 / freq_sum;
    let auxd = 1.0 / freq_sum;
    let nuc_f = [nuc_freq[0] * auxd, nuc_freq[1] * auxd, nuc_freq[2] * auxd, nuc_freq[3] * auxd];
    let f_r = nuc_f[0] + nuc_f[2];
    let f_y = nuc_f[1] + nuc_f[3];
    let use_k2p = nuc_freq.iter().any(|&f| f == 0.0);

    let ag = (float_counts[0][2] + float_counts[2][0]) * scale;
    let ct = (float_counts[1][3] + float_counts[3][1]) * scale;
    let tv = 1.0
        - ((float_counts[0][0] + float_counts[1][1] + float_counts[2][2] + float_counts[3][3]) * scale
            + ag
            + ct);

    let dist = if use_k2p {
        let ti = ag + ct;
        let a = 1.0 - 2.0 * ti - tv;
        let c = 1.0 - 2.0 * tv;
        if a > 0.0 && c > 0.0 {
            -0.5 * a.ln() - 0.25 * c.ln()
        } else {
            TN93_SATURATED
        }
    } else {
        let k1 = 2.0 * nuc_f[0] * nuc_f[2] / f_r;
        let k2 = 2.0 * nuc_f[1] * nuc_f[3] / f_y;
        let k3 = 2.0 * (f_r * f_y - nuc_f[0] * nuc_f[2] * f_y / f_r - nuc_f[1] * nuc_f[3] * f_r / f_y);
        let a = 1.0 - ag / k1 - 0.5 * tv / f_r;
        let c = 1.0 - ct / k2 - 0.5 * tv / f_y;
        let t = 1.0 - 0.5 * tv / f_y / f_r;
        if a > 0.0 && c > 0.0 && t > 0.0 {
            -k1 * a.ln() - k2 * c.ln() - k3 * t.ln()
        } else {
            TN93_SATURATED
        }
    };

    Some(if dist <= 0.0 { 0.0 } else { dist })
}

/// First and last non-gap positions of a code sequence (`None` if all gaps).
fn non_gap_span(s: &[u8]) -> Option<(usize, usize)> {
    let first = s.iter().position(|&c| c != TN93_GAP)?;
    let last = s.iter().rposition(|&c| c != TN93_GAP)?;
    Some((first, last))
}

/// Raw tn93 distance between two tn93-code sequences (see
/// [`char_to_tn93_code`]): `None` if they share `min_overlap` or fewer
/// non-gap sites, else the distance, possibly [`TN93_SATURATED`].
pub fn tn93_distance(s1: &[u8], s2: &[u8], mode: &AmbiguityMode, min_overlap: usize) -> Option<f64> {
    let n = s1.len().min(s2.len());
    let (lo, hi) = if matches!(mode, AmbiguityMode::GapMismatch) {
        // a gap opposite a base counts here, so every column is visited
        (0, n)
    } else {
        // columns outside either sequence's non-gap span cannot contribute
        match (non_gap_span(&s1[..n]), non_gap_span(&s2[..n])) {
            (Some((f1, l1)), Some((f2, l2))) => (f1.max(f2), (l1.min(l2) + 1).max(f1.max(f2))),
            _ => (0, 0),
        }
    };
    tn93_distance_range(s1, s2, lo, hi, mode, min_overlap)
}

/// TN93 distance as reported by `tn93 -t 1.0 -l 100 -a <mode>`.
///
/// Returns `f64::NAN` ("distance unavailable") for the pairs the binary does
/// not write: fewer than `TN93_MIN_OVERLAP + 1` shared non-gap sites, or a
/// distance above [`TN93_MAX_REPORTED`] (which includes saturation).
/// Downstream consumers treat NaN the same way as a pair missing from a
/// `--distances` CSV.
pub fn compute_pairwise_tn93(seq_a: &[u8], seq_b: &[u8], mode: &AmbiguityMode) -> f64 {
    match tn93_distance(seq_a, seq_b, mode, TN93_MIN_OVERLAP) {
        Some(d) if d <= TN93_MAX_REPORTED => d,
        _ => f64::NAN,
    }
}

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
    pub fn compute_distance_matrix_with_mode_and_progress<F>(&self, mode: &AmbiguityMode, mut on_progress: F) -> Vec<f64>
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
                let dist = compute_pairwise_tn93(&self.encoded_tn93[i], &self.encoded_tn93[j], mode);
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
        assert_eq!(AmbiguityMode::parse("resolve").unwrap(), AmbiguityMode::Resolve);
        assert_eq!(AmbiguityMode::parse("average").unwrap(), AmbiguityMode::Average);
        assert!(AmbiguityMode::parse("").is_err());
        let m = AmbiguityMode::default();
        assert_eq!(m, AmbiguityMode::parse("rmyskb").unwrap());
        for (code, &ch) in TN93_ALPHABET.iter().enumerate() {
            assert_eq!(char_to_tn93_code(ch) as usize, code);
            assert_eq!(m.resolves(code as u8), b"RMYSKB".contains(&ch), "{}", ch as char);
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
