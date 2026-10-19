//! Tamura-Nei 93 (TN93) pairwise genetic distance — one implementation,
//! two contracts.
//!
//! Canonical home of the family TN93 kernel. The site-counting and
//! base-frequency scheme is selected by [`Counting`]:
//!
//! - [`Counting::Faithful`] — a semantic port of the `tn93` binary's
//!   `computeTN93` (tn93_shared.cc; tn93 v1.0.15 source, validated against
//!   the v1.1.0 binary). It reproduces the binary's 4x4 site-count table
//!   with fractional ambiguity contributions, per-pair marginal base
//!   frequencies, the K2P fallback on a zero base frequency, and the
//!   SATURATED sentinel on non-positive log arguments. Floating-point
//!   operations follow the order of the C++ source so values agree to
//!   rounding. The binary, run as `tn93 -t 1.0 -a RMYSKB`, produced the
//!   distances the published TRACE-5.0 analyses consumed, so it is the
//!   oracle for trace50/DANNO.
//! - [`Counting::Legacy`] — the pre-unification manifold kernel: shared-bit
//!   "resolve" ambiguity handling and union-of-masks base-frequency
//!   estimation over IUPAC bitmasks. Kept verbatim because chronaeon's
//!   published outputs were produced with it. **Pending revalidation**:
//!   chronaeon should be re-benchmarked under `Counting::Faithful`. The
//!   union-of-masks frequency estimator has no basis in the TN93
//!   literature and biases base frequencies on ambiguity-rich inputs
//!   (tracked in REVIEW.md's dedup notes).
//!
//! [`Reporting`] selects which pairs produce a distance at all:
//! `Tn93Compat` mirrors `tn93 -l 100 -t 1.0` (pairs the binary would not
//! write are unresolvable); `AllPairs` scores every pair with at least one
//! comparable site — the contract chronaeon's matrix consumers require,
//! since a missing pair breaks downstream algebra rather than denoting
//! "no edge". `Reporting` only applies under `Counting::Faithful`; the
//! legacy scheme predates the tn93 reporting rules and always scores
//! every comparable pair.
//!
//! Sequences enter the kernel as tn93 codes (see [`char_to_tn93_code`]);
//! `trace50` re-exports this module under the binary's reporting contract,
//! and the `aeon_core._native` extension (aeon-core/native) binds the
//! matrix functions under [`Tn93Options::legacy_all_pairs`].

use crate::encoding::{PURINE_MASK, PYRIMIDINE_MASK};

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

/// Largest distance kept by the `tn93` reporting contract: the `-t 1.0`
/// threshold used to produce the reference distances (`thisD <=
/// args.distance`). The saturation sentinel is above it, so saturated pairs
/// are unreported as well.
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
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    0.25,
    0.25,
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

    /// Whether this mode resolves `code` (a tn93 alphabet index) to a
    /// matching base instead of averaging over its resolutions.
    #[inline(always)]
    pub fn resolves(&self, code: u8) -> bool {
        match self {
            AmbiguityMode::Resolve => true,
            AmbiguityMode::Subset(set) => set[code as usize],
            _ => false,
        }
    }
}

/// Which site-counting and base-frequency scheme the kernel uses.
#[derive(Debug, Clone, PartialEq)]
pub enum Counting {
    /// Faithful port of the `tn93` binary's 4x4 count table, fractional
    /// ambiguity handling, and marginal base frequencies, under the given
    /// [`AmbiguityMode`]. This is the scheme every new consumer should use.
    Faithful(AmbiguityMode),
    /// The pre-unification manifold kernel (verbatim): shared-bit resolve,
    /// coarse transition/transversion classes for disjoint masks, and
    /// union-of-masks base-frequency estimation with a 1e-4 clamp and a
    /// p-distance fallback.
    ///
    /// Retained bit-exact so chronaeon keeps producing the distances its
    /// published outputs were computed with. The union-of-masks frequency
    /// estimator has no literature basis; chronaeon should be revalidated
    /// under [`Counting::Faithful`] before this scheme is retired.
    Legacy,
}

/// Which pairs yield a reported distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reporting {
    /// `tn93 -l 100 -t 1.0`: pairs sharing [`TN93_MIN_OVERLAP`] or fewer
    /// non-gap sites, or scoring above [`TN93_MAX_REPORTED`], are
    /// unreported. Only meaningful under [`Counting::Faithful`];
    /// `Counting::Legacy` has no overlap or reporting rules and always
    /// scores every comparable pair.
    Tn93Compat,
    /// Every pair with at least one comparable site is scored; a `None`
    /// result means "no comparable sites" only.
    AllPairs,
}

/// Kernel switches selecting counting scheme and reporting contract.
#[derive(Debug, Clone, PartialEq)]
pub struct Tn93Options {
    pub counting: Counting,
    pub reporting: Reporting,
}

impl Tn93Options {
    /// The `tn93 -t 1.0 -l 100 -a <mode>` contract: faithful counting,
    /// tn93 reporting gates.
    pub fn tn93_compat(mode: AmbiguityMode) -> Self {
        Self {
            counting: Counting::Faithful(mode),
            reporting: Reporting::Tn93Compat,
        }
    }

    /// The chronaeon contract: the pre-unification manifold kernel with
    /// every comparable pair scored. Preserves bit-exact the distances
    /// `aeon_core._native` produced before the kernels were unified.
    pub fn legacy_all_pairs() -> Self {
        Self {
            counting: Counting::Legacy,
            reporting: Reporting::AllPairs,
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
    let nuc_f = [
        nuc_freq[0] * auxd,
        nuc_freq[1] * auxd,
        nuc_freq[2] * auxd,
        nuc_freq[3] * auxd,
    ];
    let f_r = nuc_f[0] + nuc_f[2];
    let f_y = nuc_f[1] + nuc_f[3];
    let use_k2p = nuc_freq.iter().any(|&f| f == 0.0);

    let ag = (float_counts[0][2] + float_counts[2][0]) * scale;
    let ct = (float_counts[1][3] + float_counts[3][1]) * scale;
    let tv = 1.0
        - ((float_counts[0][0] + float_counts[1][1] + float_counts[2][2] + float_counts[3][3])
            * scale
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
        let k3 =
            2.0 * (f_r * f_y - nuc_f[0] * nuc_f[2] * f_y / f_r - nuc_f[1] * nuc_f[3] * f_r / f_y);
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
pub fn tn93_distance(
    s1: &[u8],
    s2: &[u8],
    mode: &AmbiguityMode,
    min_overlap: usize,
) -> Option<f64> {
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

/// Unified pairwise TN93 distance under `opts`. `None` means the pair was
/// not scored: under [`Reporting::Tn93Compat`] because the binary would not
/// report it, under [`Reporting::AllPairs`] only when the pair has zero
/// comparable sites.
pub fn pairwise_distance(s1: &[u8], s2: &[u8], opts: &Tn93Options) -> Option<f64> {
    match &opts.counting {
        Counting::Faithful(mode) => {
            let min_overlap = match opts.reporting {
                Reporting::Tn93Compat => TN93_MIN_OVERLAP,
                Reporting::AllPairs => 0,
            };
            match tn93_distance(s1, s2, mode, min_overlap) {
                Some(d) if opts.reporting == Reporting::Tn93Compat && d > TN93_MAX_REPORTED => None,
                other => other,
            }
        }
        // The legacy scheme predates the tn93 reporting contract; it scores
        // every pair with >= 1 comparable site, so `reporting` does not
        // apply to it.
        Counting::Legacy => legacy_inner(s1, s2),
    }
}

/// TN93 distance as reported by `tn93 -t 1.0 -l 100 -a <mode>`.
///
/// Returns `f64::NAN` ("distance unavailable") for the pairs the binary does
/// not write: fewer than `TN93_MIN_OVERLAP + 1` shared non-gap sites, or a
/// distance above [`TN93_MAX_REPORTED`] (which includes saturation).
/// Downstream consumers treat NaN the same way as a pair missing from a
/// `--distances` CSV.
pub fn compute_pairwise_tn93(seq_a: &[u8], seq_b: &[u8], mode: &AmbiguityMode) -> f64 {
    pairwise_distance(seq_a, seq_b, &Tn93Options::tn93_compat(mode.clone())).unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------------------
// Legacy counting scheme — the pre-unification manifold kernel, verbatim.
// Operates on IUPAC masks derived from the tn93 codes; `LEGACY_MASK` is
// pinned equal to `char_to_iupac_mask` for all 256 input bytes by a test,
// so distances are bit-identical to the old mask-input kernel.
// ---------------------------------------------------------------------------

/// tn93 code -> the mask `char_to_iupac_mask` produced for the original
/// character. N, `?`, and `-` map to 0 (no comparable bases), matching the
/// old mask encoding.
const LEGACY_MASK: [u8; 18] = [
    0b0001, // A
    0b0010, // C
    0b0100, // G
    0b1000, // T
    0b1000, // U -> T
    0b0101, // R = A|G
    0b1010, // Y = C|T
    0b0110, // S = C|G
    0b1001, // W = A|T
    0b1100, // K = G|T
    0b0011, // M = A|C
    0b1110, // B = C|G|T
    0b1101, // D = A|G|T
    0b1011, // H = A|C|T
    0b0111, // V = A|C|G
    0b0000, // N
    0b0000, // ?
    0b0000, // -
];

/// Pre-unification manifold kernel, verbatim. Ambiguities are resolved to
/// minimize distance: sites whose masks share at least one bit contribute
/// zero divergence ("resolve" mode). Nucleotide frequencies are counted
/// over the union of both sequences' masks per valid site. Returns `None`
/// when the pair has zero valid comparison sites (empty sequences, or
/// every site masked by N/gap/`?`).
fn legacy_inner(seq_a: &[u8], seq_b: &[u8]) -> Option<f64> {
    let n = seq_a.len().min(seq_b.len());
    if n == 0 {
        return None;
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
        let a = LEGACY_MASK[seq_a[i] as usize];
        let b = LEGACY_MASK[seq_b[i] as usize];

        if a > 0 && b > 0 {
            l_valid += 1;

            let a_or_b = a | b;
            if (a_or_b & 0b0001) > 0 {
                cnt_a += 1;
            }
            if (a_or_b & 0b0010) > 0 {
                cnt_c += 1;
            }
            if (a_or_b & 0b0100) > 0 {
                cnt_g += 1;
            }
            if (a_or_b & 0b1000) > 0 {
                cnt_t += 1;
            }

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
        return None;
    }

    let p1 = (p1_count as f64) / (l_valid as f64);
    let p2 = (p2_count as f64) / (l_valid as f64);
    let q = (q_count as f64) / (l_valid as f64);
    let p_ham = p1 + p2 + q;

    if p_ham <= 0.0 {
        return Some(0.0);
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
        return Some(p_ham);
    }

    let d = -k1 * arg1.ln() - k2 * arg2.ln() - k3 * arg3.ln();
    if !d.is_finite() || d < 0.0 {
        return Some(p_ham);
    }
    Some(d)
}

/// Flat row-major `N x N` TN93 distance matrix over tn93-code sequences.
/// Diagonal is 0; every off-diagonal pair is computed. Unscored pairs
/// (see [`pairwise_distance`]) are filled with `0.0` — under
/// [`Reporting::Tn93Compat`] use [`distance_matrix_nan`] instead, since
/// an unreported pair is not a zero distance.
pub fn distance_matrix(encoded: &[Vec<u8>], opts: &Tn93Options) -> Vec<f64> {
    distance_matrix_impl(encoded, opts, 0.0)
}

/// Same as [`distance_matrix`] but unscored pairs are `NaN`, so callers
/// can impute them (e.g. to the maximum observed distance) or drop them,
/// instead of silently treating incomparable taxa as identical.
pub fn distance_matrix_nan(encoded: &[Vec<u8>], opts: &Tn93Options) -> Vec<f64> {
    distance_matrix_impl(encoded, opts, f64::NAN)
}

fn distance_matrix_impl(encoded: &[Vec<u8>], opts: &Tn93Options, unscored: f64) -> Vec<f64> {
    let n = encoded.len();
    let mut matrix = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pairwise_distance(&encoded[i], &encoded[j], opts).unwrap_or(unscored);
            matrix[i * n + j] = d;
            matrix[j * n + i] = d;
        }
    }
    matrix
}

/// Flat row-major `N x M` TN93 cross-distance matrix: every row sequence in
/// `a` against every column sequence in `b` (landmark/Nystroem-style use).
pub fn cross_distance_matrix(a: &[Vec<u8>], b: &[Vec<u8>], opts: &Tn93Options) -> Vec<f64> {
    cross_distance_matrix_impl(a, b, opts, 0.0)
}

/// Same as [`cross_distance_matrix`] but unscored pairs are `NaN`.
pub fn cross_distance_matrix_nan(a: &[Vec<u8>], b: &[Vec<u8>], opts: &Tn93Options) -> Vec<f64> {
    cross_distance_matrix_impl(a, b, opts, f64::NAN)
}

fn cross_distance_matrix_impl(
    a: &[Vec<u8>],
    b: &[Vec<u8>],
    opts: &Tn93Options,
    unscored: f64,
) -> Vec<f64> {
    let (n, m) = (a.len(), b.len());
    let mut matrix = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            matrix[i * m + j] = pairwise_distance(&a[i], &b[j], opts).unwrap_or(unscored);
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::char_to_iupac_mask;

    fn enc(s: &[u8]) -> Vec<u8> {
        s.iter().map(|&c| char_to_tn93_code(c)).collect()
    }

    /// `LEGACY_MASK` must reproduce `char_to_iupac_mask` exactly for every
    /// possible input byte; then the tn93-code input path is bit-identical
    /// to the old mask-input kernel for any sequence.
    #[test]
    fn test_legacy_mask_matches_iupac_encoding() {
        for c in 0u8..=255 {
            assert_eq!(
                LEGACY_MASK[char_to_tn93_code(c) as usize],
                char_to_iupac_mask(c),
                "byte {c}"
            );
        }
    }

    // -- Legacy scheme (pre-unification contract) ---------------------------

    #[test]
    fn test_legacy_identical() {
        let bits = enc(b"ACGTACGT");
        let d = pairwise_distance(&bits, &bits, &Tn93Options::legacy_all_pairs());
        assert_eq!(d, Some(0.0));
    }

    #[test]
    fn test_legacy_transition() {
        let bits1 = enc(b"ACGTACGTACGTACGTACGTACGT");
        let bits2 = enc(b"GCGTACGTACGTACGTACGTACGT"); // 1 transition in 24 bp
        let dist = pairwise_distance(&bits1, &bits2, &Tn93Options::legacy_all_pairs()).unwrap();
        assert!(dist > 0.0 && dist < 0.5);
    }

    #[test]
    fn test_legacy_n_ambiguity_skipped() {
        let bits1 = enc(b"ACGTACGTNN");
        let bits2 = enc(b"ACGTTCGTNN"); // 1 transition in 8 valid bp, Ns skipped
        let dist = pairwise_distance(&bits1, &bits2, &Tn93Options::legacy_all_pairs()).unwrap();
        assert!(dist > 0.12 && dist < 0.16);
    }

    #[test]
    fn test_legacy_distance_matrix_symmetric() {
        let seqs = vec![enc(b"ACGTACGT"), enc(b"ACGTTCGT"), enc(b"NNNNNNNN")];
        let m = distance_matrix(&seqs, &Tn93Options::legacy_all_pairs());
        assert_eq!(m.len(), 9);
        assert_eq!(m[0], 0.0);
        assert_eq!(m[1], m[3]);
        assert!(m[1] > 0.0);
        // All-N pair: no valid sites -> 0.0 (legacy sentinel)
        assert_eq!(m[2], 0.0);
    }

    #[test]
    fn test_legacy_distance_matrix_nan_unresolvable() {
        let seqs = vec![enc(b"ACGTACGT"), enc(b"NNNNNNNN")];
        let m = distance_matrix_nan(&seqs, &Tn93Options::legacy_all_pairs());
        assert!(m[1].is_nan());
        assert!(m[2].is_nan());
        assert_eq!(m[0], 0.0);
        assert_eq!(m[3], 0.0);
    }

    #[test]
    fn test_legacy_cross_distance_matrix_nan_unresolvable() {
        let a = vec![enc(b"ACGT")];
        let b = vec![enc(b"ACGT"), enc(b"----")];
        let m = cross_distance_matrix_nan(&a, &b, &Tn93Options::legacy_all_pairs());
        assert_eq!(m[0], 0.0);
        assert!(m[1].is_nan());
    }

    #[test]
    fn test_legacy_cross_distance_matrix_rectangular() {
        let a = vec![enc(b"ACGT"), enc(b"ACGA")];
        let b = vec![enc(b"ACGT"), enc(b"TCGT"), enc(b"NNNN")];
        let m = cross_distance_matrix(&a, &b, &Tn93Options::legacy_all_pairs());
        assert_eq!(m.len(), 6);
        assert_eq!(m[0], 0.0); // identical pair
        assert!(m[1] > 0.0); // transversion pair
    }

    // -- Faithful scheme (tn93 binary port) ----------------------------------

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

    // -- Reporting options ---------------------------------------------------

    /// `AllPairs` scores a short pair that `Tn93Compat` withholds (below the
    /// 100-site overlap gate), and `Legacy` scores it too.
    #[test]
    fn test_reporting_gate() {
        let s = enc(b"ACGTACGT"); // 8 shared sites << 100
        let compat = Tn93Options::tn93_compat(AmbiguityMode::Resolve);
        assert_eq!(pairwise_distance(&s, &s, &compat), None);

        let all_pairs = Tn93Options {
            counting: Counting::Faithful(AmbiguityMode::Resolve),
            reporting: Reporting::AllPairs,
        };
        assert_eq!(pairwise_distance(&s, &s, &all_pairs), Some(0.0));

        assert_eq!(
            pairwise_distance(&s, &s, &Tn93Options::legacy_all_pairs()),
            Some(0.0)
        );
    }

    /// On fully resolvable (ACGT-only) input, Legacy and Faithful implement
    /// the same TN93 math; the frequency estimators coincide on unambiguous
    /// sites, so the residual gap is fp-order only. Pin a tight bound so a
    /// formula-level regression still trips.
    #[test]
    fn test_schemes_agree_on_unambiguous_input() {
        let bases = b"ACGT";
        let mk = |len: usize, f: &dyn Fn(usize, u8) -> u8| -> Vec<u8> {
            (0..len)
                .map(|i| char_to_tn93_code(f(i, bases[i % 4])))
                .collect()
        };
        let a = mk(120, &|_, b| b);
        let b = mk(120, &|i, b| {
            if i % 20 == 7 {
                match b {
                    b'A' => b'G',
                    b'G' => b'A',
                    b'C' => b'T',
                    _ => b'C',
                }
            } else {
                b
            }
        });
        for (x, y) in [(&a, &a), (&a, &b)] {
            let l = pairwise_distance(x, y, &Tn93Options::legacy_all_pairs()).unwrap();
            for mode in [
                AmbiguityMode::Resolve,
                AmbiguityMode::Average,
                AmbiguityMode::Skip,
                AmbiguityMode::GapMismatch,
            ] {
                let f = pairwise_distance(x, y, &Tn93Options::tn93_compat(mode)).unwrap();
                assert!((l - f).abs() < 5e-3, "legacy={l} faithful={f}");
            }
        }
    }
}
