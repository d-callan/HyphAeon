//! Tamura-Nei 93 (TN93) pairwise genetic distance and IUPAC ambiguity resolution.

use serde::{Deserialize, Serialize};

// Canonical IUPAC alphabet and header date extraction live in aeon-manifold
// (single source of truth); re-exported here so existing `crate::tn93::*`
// call sites are unchanged.
pub use aeon_manifold::encoding::{
    char_to_iupac_mask, char_to_nuc_int, iupac_base_probabilities, PURINE_MASK, PYRIMIDINE_MASK,
};
pub use aeon_manifold::temporal::parse_date_from_header;
pub use aeon_manifold::tn93::compute_pairwise_tn93;

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

/// A parsed nucleotide FASTA alignment ready for TRACE-5.0 processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub headers: Vec<String>,
    pub dates: Vec<f64>,
    pub seq_len: usize,
    pub encoded_bits: Vec<Vec<u8>>,
    pub encoded_ints: Vec<Vec<i8>>,
    pub raw_seqs: Vec<Vec<u8>>,
}

impl Alignment {
    /// Parses an aligned FASTA string and extracts dates from headers.
    ///
    /// Tokenization and ragged-sequence padding delegate to
    /// `aeon_manifold::fasta` (`PadToMax` mode); each sequence is then
    /// dual-encoded to IUPAC bitmasks and integer states.
    pub fn from_fasta_str(fasta_content: &str) -> Result<Self, String> {
        use aeon_manifold::fasta::{normalize_lengths, tokenize_fasta, ParseMode};

        let mut records = tokenize_fasta(fasta_content)
            .map_err(|_| "No sequences found in FASTA content".to_string())?;
        let seq_len = normalize_lengths(&mut records, ParseMode::PadToMax)?;

        let n = records.len();
        let mut headers = Vec::with_capacity(n);
        let mut encoded_bits = Vec::with_capacity(n);
        let mut encoded_ints = Vec::with_capacity(n);
        let mut dates = Vec::with_capacity(n);
        let mut raw_seqs = Vec::with_capacity(n);

        for rec in records {
            encoded_bits.push(rec.seq.iter().map(|&c| char_to_iupac_mask(c)).collect());
            encoded_ints.push(rec.seq.iter().map(|&c| char_to_nuc_int(c)).collect());
            dates.push(parse_date_from_header(&rec.name));
            headers.push(rec.name);
            raw_seqs.push(rec.seq);
        }

        Ok(Self {
            headers,
            dates,
            seq_len,
            encoded_bits,
            encoded_ints,
            raw_seqs,
        })
    }

    /// Number of sequences in the alignment.
    #[inline(always)]
    pub fn num_sequences(&self) -> usize {
        self.headers.len()
    }

    /// Computes full symmetric pairwise TN93 distance matrix with a progress callback.
    /// `on_progress(done_pairs, total_pairs, fraction)`
    pub fn compute_distance_matrix_with_progress<F>(&self, mut on_progress: F) -> Vec<f64>
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
                let dist = compute_pairwise_tn93(&self.encoded_bits[i], &self.encoded_bits[j]);
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

    /// Computes full symmetric pairwise TN93 distance matrix.
    /// Returns a flattened `N x N` matrix in row-major order.
    #[inline]
    pub fn compute_distance_matrix(&self) -> Vec<f64> {
        self.compute_distance_matrix_with_progress(|_, _, _| {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_tn93_n_ambiguity_skipped() {
        let s1 = b"ACGTACGTNN";
        let s2 = b"ACGTTCGTNN"; // 1 transition in 8 valid bp, Ns skipped
        let bits1: Vec<u8> = s1.iter().map(|&c| char_to_iupac_mask(c)).collect();
        let bits2: Vec<u8> = s2.iter().map(|&c| char_to_iupac_mask(c)).collect();
        let dist = compute_pairwise_tn93(&bits1, &bits2);
        assert!(dist > 0.12 && dist < 0.16);
    }

    #[test]
    fn test_char_to_iupac_mask_canonical() {
        assert_eq!(char_to_iupac_mask(b'A'), 0b0001);
        assert_eq!(char_to_iupac_mask(b'C'), 0b0010);
        assert_eq!(char_to_iupac_mask(b'G'), 0b0100);
        assert_eq!(char_to_iupac_mask(b'T'), 0b1000);
        assert_eq!(char_to_iupac_mask(b'U'), 0b1000); // RNA folds to T
    }

    #[test]
    fn test_char_to_iupac_mask_ambiguity_codes() {
        assert_eq!(char_to_iupac_mask(b'R'), 0b0101); // A|G
        assert_eq!(char_to_iupac_mask(b'Y'), 0b1010); // C|T
        assert_eq!(char_to_iupac_mask(b'S'), 0b0110); // C|G
        assert_eq!(char_to_iupac_mask(b'W'), 0b1001); // A|T
        assert_eq!(char_to_iupac_mask(b'K'), 0b1100); // G|T
        assert_eq!(char_to_iupac_mask(b'M'), 0b0011); // A|C
        assert_eq!(char_to_iupac_mask(b'B'), 0b1110); // C|G|T
        assert_eq!(char_to_iupac_mask(b'D'), 0b1101); // A|G|T
        assert_eq!(char_to_iupac_mask(b'H'), 0b1011); // A|C|T
        assert_eq!(char_to_iupac_mask(b'V'), 0b0111); // A|C|G
                                                      // N and unknowns/gaps map to 0 (skipped in valid overlap)
        assert_eq!(char_to_iupac_mask(b'N'), 0);
        assert_eq!(char_to_iupac_mask(b'-'), 0);
        assert_eq!(char_to_iupac_mask(b'?'), 0);
        assert_eq!(char_to_iupac_mask(b'X'), 0);
    }

    #[test]
    fn test_char_to_iupac_mask_case_insensitive() {
        assert_eq!(char_to_iupac_mask(b'a'), char_to_iupac_mask(b'A'));
        assert_eq!(char_to_iupac_mask(b'r'), char_to_iupac_mask(b'R'));
        assert_eq!(char_to_iupac_mask(b'n'), 0);
    }

    #[test]
    fn test_char_to_nuc_int() {
        assert_eq!(char_to_nuc_int(b'A'), 0);
        assert_eq!(char_to_nuc_int(b'C'), 1);
        assert_eq!(char_to_nuc_int(b'G'), 2);
        assert_eq!(char_to_nuc_int(b'T'), 3);
        assert_eq!(char_to_nuc_int(b'U'), 3);
        assert_eq!(char_to_nuc_int(b'a'), 0);
        assert_eq!(char_to_nuc_int(b'N'), -1);
        assert_eq!(char_to_nuc_int(b'R'), -1); // ambiguity not representable
        assert_eq!(char_to_nuc_int(b'-'), -1);
    }

    #[test]
    fn test_iupac_base_probabilities_sum_to_one() {
        for &c in b"ACGTURYSWKMBDHVNacgturyswkmbdhvn".iter() {
            let p = iupac_base_probabilities(c);
            let s: f64 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "{c} sums to {s}");
        }
    }

    #[test]
    fn test_iupac_base_probabilities_distribute_evenly() {
        assert_eq!(iupac_base_probabilities(b'R'), [0.5, 0.0, 0.5, 0.0]);
        let b = iupac_base_probabilities(b'B');
        for i in [1, 2, 3] {
            assert!((b[i] - 1.0 / 3.0).abs() < 1e-12);
        }
        assert_eq!(b[0], 0.0);
    }

    #[test]
    fn test_from_fasta_str_basic() {
        let aln = Alignment::from_fasta_str(">a|2001\nACGT\n>b|2002\nTGCA\n").unwrap();
        assert_eq!(aln.num_sequences(), 2);
        assert_eq!(
            aln.headers,
            vec!["a|2001".to_string(), "b|2002".to_string()]
        );
        assert_eq!(aln.dates, vec![2001.0, 2002.0]);
        assert_eq!(aln.seq_len, 4);
    }

    #[test]
    fn test_from_fasta_str_ragged_pads_to_max() {
        // trace50 contract: ragged sequences are PADDED to max length with '-',
        // not rejected (opposite of rhizaeon's parse_fasta).
        let aln = Alignment::from_fasta_str(">a\nACGT\n>b\nAC\n").unwrap();
        assert_eq!(aln.seq_len, 4);
        assert_eq!(aln.raw_seqs[1], b"AC--");
        // Padded positions encode as 0 in both encodings
        assert_eq!(aln.encoded_bits[1][3], 0);
        assert_eq!(aln.encoded_ints[1][3], -1);
    }

    #[test]
    fn test_from_fasta_str_dual_encoding_consistent() {
        // For canonical bases, mask bit position and nuc_int must agree:
        // A: mask bit0 -> int 0; C: bit1 -> 1; G: bit2 -> 2; T: bit3 -> 3.
        let aln = Alignment::from_fasta_str(">a\nACGT\n").unwrap();
        let canonical = [0, 1, 2, 3]; // expected nuc_int per position
        for u in 0..4 {
            let mask = aln.encoded_bits[0][u];
            let int = aln.encoded_ints[0][u];
            assert_eq!(mask, 1 << u);
            assert_eq!(int, canonical[u]);
            // mask has exactly one bit set, at position `int`
            assert_eq!(mask.trailing_zeros() as i8, int);
        }
    }

    #[test]
    fn test_from_fasta_str_preserves_raw_and_uppercases() {
        let aln = Alignment::from_fasta_str(">a\nacgt\n").unwrap();
        assert_eq!(aln.raw_seqs[0], b"ACGT");
    }

    #[test]
    fn test_from_fasta_str_empty_is_error() {
        assert!(Alignment::from_fasta_str("").is_err());
        assert!(Alignment::from_fasta_str("   \n").is_err());
    }

    #[test]
    fn test_from_fasta_str_missing_date_is_nan() {
        let aln = Alignment::from_fasta_str(">nodateshere\nACGT\n").unwrap();
        assert!(aln.dates[0].is_nan());
    }

    #[test]
    fn test_parse_date_from_header_decimal() {
        assert_eq!(parse_date_from_header("seq|2014"), 2014.0);
        assert!((parse_date_from_header("seq_2014.25_x") - 2014.25).abs() < 1e-9);
        // pipe/underscore/slash/space all delimit tokens
        assert_eq!(parse_date_from_header("MH352627|10001|2001|USA"), 2001.0);
    }

    #[test]
    fn test_parse_date_from_header_iso() {
        // 2021-04-15: month (4-1)/12=.25, day (15-1)/365.25≈.0383
        let d = parse_date_from_header("seq|2021-04-15");
        let expected = 2021.0 + 3.0 / 12.0 + 14.0 / 365.25;
        assert!((d - expected).abs() < 1e-9);
        // YYYY-MM defaults day to 15
        let d2 = parse_date_from_header("seq|2021-04");
        assert!((d2 - expected).abs() < 1e-9);
    }

    #[test]
    fn test_parse_date_from_header_out_of_range_skipped() {
        // Years outside [1950,2035] are skipped; NaN if nothing valid.
        assert!(parse_date_from_header("seq|1900").is_nan());
        assert!(parse_date_from_header("seq|2050").is_nan());
        // ...but a valid later token wins
        assert_eq!(parse_date_from_header("seq|1900|2005"), 2005.0);
    }

    #[test]
    fn test_parse_date_from_header_no_date_is_nan() {
        assert!(parse_date_from_header("HXB2").is_nan());
        assert!(parse_date_from_header("").is_nan());
    }
}
