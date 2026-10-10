//! Tamura-Nei 93 (TN93) pairwise genetic distance and IUPAC ambiguity resolution.

use serde::{Deserialize, Serialize};

pub const PURINE_MASK: u8 = 0b0101;     // A (1) or G (4)
pub const PYRIMIDINE_MASK: u8 = 0b1010; // C (2) or T (8)

/// Converts an ASCII character to a 4-bit IUPAC bitmask.
/// A=1, C=2, G=4, T=8.
#[inline(always)]
pub fn char_to_iupac_mask(c: u8) -> u8 {
    match c {
        b'A' | b'a' => 0b0001,
        b'C' | b'c' => 0b0010,
        b'G' | b'g' => 0b0100,
        b'T' | b't' | b'U' | b'u' => 0b1000,
        b'R' | b'r' => 0b0101, // A or G
        b'Y' | b'y' => 0b1010, // C or T
        b'S' | b's' => 0b0110, // C or G
        b'W' | b'w' => 0b1001, // A or T
        b'K' | b'k' => 0b1100, // G or T
        b'M' | b'm' => 0b0011, // A or C
        b'B' | b'b' => 0b1110, // C, G, or T
        b'D' | b'd' => 0b1101, // A, G, or T
        b'H' | b'h' => 0b1011, // A, C, or T
        b'V' | b'v' => 0b0111, // A, C, or G
        b'N' | b'n' => 0b0000, // Fully ambiguous / missing data (skipped in valid overlap)
        _ => 0b0000,           // Gap, unknown
    }
}

/// Converts an ASCII character to an integer representation:
/// 0=A, 1=C, 2=G, 3=T, -1=other.
#[inline(always)]
pub fn char_to_nuc_int(c: u8) -> i8 {
    match c {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' | b'U' | b'u' => 3,
        _ => -1,
    }
}

/// Probabilistic component base vector [f_A, f_C, f_G, f_T] for an IUPAC code.
#[inline(always)]
pub fn iupac_base_probabilities(c: u8) -> [f64; 4] {
    match c {
        b'A' | b'a' => [1.0, 0.0, 0.0, 0.0],
        b'C' | b'c' => [0.0, 1.0, 0.0, 0.0],
        b'G' | b'g' => [0.0, 0.0, 1.0, 0.0],
        b'T' | b't' | b'U' | b'u' => [0.0, 0.0, 0.0, 1.0],
        b'R' | b'r' => [0.5, 0.0, 0.5, 0.0],
        b'Y' | b'y' => [0.0, 0.5, 0.0, 0.5],
        b'S' | b's' => [0.0, 0.5, 0.5, 0.0],
        b'W' | b'w' => [0.5, 0.0, 0.0, 0.5],
        b'K' | b'k' => [0.0, 0.0, 0.5, 0.5],
        b'M' | b'm' => [0.5, 0.5, 0.0, 0.0],
        b'B' | b'b' => [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        b'D' | b'd' => [1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0],
        b'H' | b'h' => [1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0],
        b'V' | b'v' => [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        _ => [0.25, 0.25, 0.25, 0.25],
    }
}

/// Computes analytical Tamura-Nei 93 (TN93) pairwise genetic distance.
/// Resolves ambiguities to minimize distance (matching characters share bits).
pub fn compute_pairwise_tn93(seq_a_bits: &[u8], seq_b_bits: &[u8]) -> f64 {
    let n = seq_a_bits.len().min(seq_b_bits.len());
    if n == 0 {
        return 0.0;
    }

    let mut l_valid = 0usize;
    let mut p1_count = 0usize;
    let mut p2_count = 0usize;
    let mut q_count = 0usize;

    let mut cnt_a = 0usize;
    let mut cnt_c = 0usize;
    let mut cnt_g = 0usize;
    let mut cnt_t = 0usize;

    for i in 0..n {
        let a = seq_a_bits[i];
        let b = seq_b_bits[i];

        if a > 0 && b > 0 {
            l_valid += 1;

            let a_or_b = a | b;
            if (a_or_b & 0b0001) > 0 { cnt_a += 1; }
            if (a_or_b & 0b0010) > 0 { cnt_c += 1; }
            if (a_or_b & 0b0100) > 0 { cnt_g += 1; }
            if (a_or_b & 0b1000) > 0 { cnt_t += 1; }

            // If bits overlap, zero divergence at this site
            if (a & b) == 0 {
                let p1 = ((a & PURINE_MASK) > 0) && ((b & PURINE_MASK) > 0);
                let p2 = ((a & PYRIMIDINE_MASK) > 0) && ((b & PYRIMIDINE_MASK) > 0);

                if p1 {
                    p1_count += 1;
                } else if p2 {
                    p2_count += 1;
                } else {
                    q_count += 1;
                }
            }
        }
    }

    if l_valid == 0 {
        return 0.0;
    }

    let p1 = (p1_count as f64) / (l_valid as f64);
    let p2 = (p2_count as f64) / (l_valid as f64);
    let q = (q_count as f64) / (l_valid as f64);
    let p_ham = p1 + p2 + q;

    if p_ham <= 0.0 {
        return 0.0;
    }

    let tot = (cnt_a + cnt_c + cnt_g + cnt_t).max(1) as f64;
    let f_a = ((cnt_a as f64) / tot).max(1e-4);
    let f_c = ((cnt_c as f64) / tot).max(1e-4);
    let f_g = ((cnt_g as f64) / tot).max(1e-4);
    let f_t = ((cnt_t as f64) / tot).max(1e-4);

    let f_r = f_a + f_g;
    let f_y = f_c + f_t;

    let k1 = 2.0 * f_a * f_g / f_r;
    let k2 = 2.0 * f_c * f_t / f_y;
    let k3 = 2.0 * (f_r * f_y - (f_a * f_g * f_y / f_r) - (f_c * f_t * f_r / f_y));

    let arg1 = 1.0 - (f_r / (2.0 * f_a * f_g)) * p1 - (q / (2.0 * f_r));
    let arg2 = 1.0 - (f_y / (2.0 * f_c * f_t)) * p2 - (q / (2.0 * f_y));
    let arg3 = 1.0 - (q / (2.0 * f_r * f_y));

    if arg1 <= 1e-6 || arg2 <= 1e-6 || arg3 <= 1e-6 {
        return p_ham;
    }

    let d = -k1 * arg1.ln() - k2 * arg2.ln() - k3 * arg3.ln();
    if !d.is_finite() || d < 0.0 {
        return p_ham;
    }
    d
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
    /// Parses an aligned FASTA string and optional date extraction regex.
    pub fn from_fasta_str(fasta_content: &str) -> Result<Self, String> {
        let mut headers = Vec::new();
        let mut raw_seqs = Vec::new();

        let mut current_header = None;
        let mut current_seq = Vec::new();

        for line in fasta_content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with('>') {
                if let Some(h) = current_header.take() {
                    headers.push(h);
                    raw_seqs.push(current_seq);
                    current_seq = Vec::new();
                }
                let h = trimmed[1..].split_whitespace().next().unwrap_or("").to_string();
                current_header = Some(h);
            } else {
                current_seq.extend_from_slice(trimmed.to_ascii_uppercase().as_bytes());
            }
        }

        if let Some(h) = current_header {
            headers.push(h);
            raw_seqs.push(current_seq);
        }

        if headers.is_empty() {
            return Err("No sequences found in FASTA content".to_string());
        }

        let seq_len = raw_seqs.iter().map(|s| s.len()).max().unwrap_or(0);

        let mut encoded_bits = Vec::with_capacity(headers.len());
        let mut encoded_ints = Vec::with_capacity(headers.len());
        let mut dates = Vec::with_capacity(headers.len());

        for (i, seq) in raw_seqs.iter_mut().enumerate() {
            // Pad sequences to maximum length if ragged
            if seq.len() < seq_len {
                seq.resize(seq_len, b'-');
            }

            let bits: Vec<u8> = seq.iter().map(|&c| char_to_iupac_mask(c)).collect();
            let ints: Vec<i8> = seq.iter().map(|&c| char_to_nuc_int(c)).collect();

            encoded_bits.push(bits);
            encoded_ints.push(ints);

            // Attempt to parse date from header string
            let d = parse_date_from_header(&headers[i]);
            dates.push(d);
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

/// Helper function to parse a calendar year or decimal date from sequence header.
pub fn parse_date_from_header(header: &str) -> f64 {
    for token in header.split(|c| c == '|' || c == '_' || c == '/' || c == ' ') {
        // Direct float/integer parse (e.g. 2014.25 or 2014)
        if let Ok(val) = token.parse::<f64>() {
            if (1950.0..=2035.0).contains(&val) {
                return val;
            }
        }
        // ISO 8601 calendar date parse (e.g. 2021-04-15 or 2021-04)
        let parts: Vec<&str> = token.split('-').collect();
        if parts.len() == 2 || parts.len() == 3 {
            if let Ok(year) = parts[0].parse::<f64>() {
                if (1950.0..=2035.0).contains(&year) {
                    if let Ok(month) = parts[1].parse::<f64>() {
                        if (1.0..=12.0).contains(&month) {
                            let day = if parts.len() == 3 {
                                parts[2].parse::<f64>().unwrap_or(15.0)
                            } else {
                                15.0
                            };
                            return year + (month - 1.0) / 12.0 + (day - 1.0) / 365.25;
                        }
                    }
                }
            }
        }
    }
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tn93_identical() {
        let s1 = b"ACGTACGT";
        let bits1: Vec<u8> = s1.iter().map(|&c| char_to_iupac_mask(c)).collect();
        let dist = compute_pairwise_tn93(&bits1, &bits1);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_tn93_transition() {
        let s1 = b"ACGTACGTACGTACGTACGTACGT";
        let s2 = b"GCGTACGTACGTACGTACGTACGT"; // 1 transition in 24 bp
        let bits1: Vec<u8> = s1.iter().map(|&c| char_to_iupac_mask(c)).collect();
        let bits2: Vec<u8> = s2.iter().map(|&c| char_to_iupac_mask(c)).collect();
        let dist = compute_pairwise_tn93(&bits1, &bits2);
        println!("TN93 dist for 1/24 transition: {}", dist);
        assert!(dist > 0.0 && dist < 0.5);
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
    fn test_tn93_n_ambiguity_skipped() {
        let s1 = b"ACGTACGTNN";
        let s2 = b"ACGTTCGTNN"; // 1 transition in 8 valid bp, Ns skipped
        let bits1: Vec<u8> = s1.iter().map(|&c| char_to_iupac_mask(c)).collect();
        let bits2: Vec<u8> = s2.iter().map(|&c| char_to_iupac_mask(c)).collect();
        let dist = compute_pairwise_tn93(&bits1, &bits2);
        assert!(dist > 0.12 && dist < 0.16);
    }
}
