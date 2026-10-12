//! Canonical IUPAC nucleotide alphabet and its encoding projections.
//!
//! One authoritative table (`char_to_iupac_mask`) maps ASCII bases to 4-bit
//! IUPAC sets (A=1, C=2, G=4, T=8). Every other representation used across
//! the engine family is derived from it, so encodings can never drift apart:
//!
//! - `char_to_nuc_int`        — 0=A,1=C,2=G,3=T,-1=other   (trace50 TN93 counts)
//! - `encode_char`            — 1=A,2=C,3=G,4=T/U,0=other  (rhizaeon matrix)
//! - `iupac_base_probabilities` — uniform component freqs  (trace50 likelihood)
//!
//! The ambiguity-collapse asymmetry is intentional: `char_to_nuc_int` and
//! `encode_char` flatten ambiguous codes to -1/0, while `char_to_iupac_mask`
//! preserves the full set. Port projections, never unify inputs.

/// A (0b0001) or G (0b0100).
pub const PURINE_MASK: u8 = 0b0101;
/// C (0b0010) or T (0b1000).
pub const PYRIMIDINE_MASK: u8 = 0b1010;

/// Converts an ASCII character to a 4-bit IUPAC bitmask.
/// A=1, C=2, G=4, T=8. Ambiguous codes retain all possible bases; N, gaps,
/// and unknown characters map to 0b0000.
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
        _ => 0b0000,           // N, gap, unknown
    }
}

/// Converts an ASCII character to an integer representation:
/// 0=A, 1=C, 2=G, 3=T, -1=other (ambiguous included).
///
/// Derived from `char_to_iupac_mask`: singleton masks map to the set bit's
/// index; any non-singleton collapses to -1.
#[inline(always)]
pub fn char_to_nuc_int(c: u8) -> i8 {
    match char_to_iupac_mask(c) {
        0b0001 => 0,
        0b0010 => 1,
        0b0100 => 2,
        0b1000 => 3,
        _ => -1,
    }
}

/// Encodes a nucleotide ASCII character into integer state:
/// 1: A, 2: C, 3: G, 4: T/U, 0: gap/ambiguity.
///
/// Derived from `char_to_iupac_mask`: bit index + 1 for singletons.
#[inline(always)]
pub fn encode_char(c: u8) -> u8 {
    match char_to_iupac_mask(c) {
        0b0001 => 1,
        0b0010 => 2,
        0b0100 => 3,
        0b1000 => 4,
        _ => 0,
    }
}

/// Probabilistic component base vector [f_A, f_C, f_G, f_T] for an IUPAC
/// code — 1/k per possible base. Mask-0 characters (N, gap, unknown) map to
/// the uniform distribution [0.25; 4], matching the historical trace50
/// contract.
#[inline(always)]
pub fn iupac_base_probabilities(c: u8) -> [f64; 4] {
    let mask = char_to_iupac_mask(c);
    if mask == 0 {
        return [0.25; 4];
    }
    let k = mask.count_ones() as f64;
    let f = 1.0 / k;
    [
        if mask & 0b0001 != 0 { f } else { 0.0 },
        if mask & 0b0010 != 0 { f } else { 0.0 },
        if mask & 0b0100 != 0 { f } else { 0.0 },
        if mask & 0b1000 != 0 { f } else { 0.0 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaustive consistency: every projection must agree with the mask
    /// table for all 256 byte values.
    #[test]
    fn test_projections_consistent_with_mask() {
        for c in 0u16..=255 {
            let b = c as u8;
            let m = char_to_iupac_mask(b);
            let n = char_to_nuc_int(b);
            let e = encode_char(b);
            let p = iupac_base_probabilities(b);

            match m {
                0b0001 => {
                    assert_eq!(n, 0, "nuc_int for {b}");
                    assert_eq!(e, 1, "encode_char for {b}");
                }
                0b0010 => {
                    assert_eq!(n, 1, "nuc_int for {b}");
                    assert_eq!(e, 2, "encode_char for {b}");
                }
                0b0100 => {
                    assert_eq!(n, 2, "nuc_int for {b}");
                    assert_eq!(e, 3, "encode_char for {b}");
                }
                0b1000 => {
                    assert_eq!(n, 3, "nuc_int for {b}");
                    assert_eq!(e, 4, "encode_char for {b}");
                }
                _ => {
                    assert_eq!(n, -1, "nuc_int for {b}");
                    assert_eq!(e, 0, "encode_char for {b}");
                }
            }

            if m == 0 {
                assert_eq!(p, [0.25; 4], "uniform probs for {b}");
            } else {
                let k = m.count_ones() as f64;
                for (i, &bit) in [0b0001u8, 0b0010, 0b0100, 0b1000].iter().enumerate() {
                    let want = if m & bit != 0 { 1.0 / k } else { 0.0 };
                    assert_eq!(p[i], want, "prob[{i}] for {b}");
                }
            }
        }
    }
}
