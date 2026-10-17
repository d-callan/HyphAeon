//! Cross-implementation seam tests for the duplicated pieces tracked in
//! REVIEW.md's dedup list.
//!
//! Two TN93 kernels exist with different contracts, and two header-date
//! parsers exist with different precedence rules. Each side is already
//! tested against its own oracle (test_tn93_oracle.rs, test_dates_oracle.rs,
//! aeon-manifold inline tests). What was missing — and what a dedup refactor
//! needs — is a pin on *where the two agree* and *where they intentionally
//! diverge*. If a merge changes any of these, this file fails loudly.
//!
//! - `trace50::tn93`          — bit-faithful port of the `tn93` binary
//!   oracle (`-t 1.0 -l 100 -a <mode>`): NaN for pairs the binary would
//!   not report (<= 100 shared non-gap sites, or distance > 1.0).
//! - `aeon_manifold::tn93`    — generic IUPAC-bitmask kernel used by
//!   chronaeon via `aeon_core._native`: 0.0 on unresolvable pairs.
//! - `trace50::dates`         — DANNO path-b rules: DC headers,
//!   underscore-years, `|` fields via calendar-precise ymd math.
//! - `aeon_manifold::temporal`— generic first-token-wins header parser
//!   (the semantics formerly in `trace50::tn93`).

use aeon_manifold::char_to_iupac_mask;
use aeon_manifold::temporal::parse_date_from_header as manifold_parse_date;
use aeon_manifold::tn93 as manifold_tn93;
use trace50::dates::{self, parse_date_from_header as trace50_parse_date};
use trace50::tn93::{char_to_tn93_code, compute_pairwise_tn93, AmbiguityMode};

// ---- TN93 kernels ---------------------------------------------------------

/// manifold kernel input: IUPAC bitmasks.
fn encode(seq: &str) -> Vec<u8> {
    seq.bytes().map(char_to_iupac_mask).collect()
}

/// trace50 kernel input: the 18-code tn93 alphabet (not raw ASCII — raw
/// bytes index the per-code tables and panic).
fn encode_tn93(seq: &str) -> Vec<u8> {
    seq.bytes().map(char_to_tn93_code).collect()
}

/// 120-nt sequence from a repeating ACGT block with `mutate` applied at
/// chosen positions. ACGT-only, so every AmbiguityMode must agree.
fn make_seq(len: usize, mutate: impl Fn(usize, u8) -> u8) -> String {
    let bases = b"ACGT";
    (0..len).map(|i| mutate(i, bases[i % 4]) as char).collect()
}

/// On fully resolvable (ACGT-only) input the two kernels implement the
/// same TN93 math but are NOT bit-identical: manifold estimates nucleotide
/// frequencies from the union of both sequences' IUPAC masks per valid
/// site while the trace50 port replicates the tn93 binary's marginal
/// frequencies (documented in manifold/tn93.rs as "an ambiguity-aware
/// approximation ... not a bit-exact port"). Observed gap on the fixtures
/// below: <= ~5e-4 absolute (~0.6% relative). The 5e-3 bound tolerates
/// that documented approximation gap while still tripping on a formula-
/// level regression, which shifts distances by the order of the distance
/// itself (>= ~0.05 here). Identical sequences must agree exactly.
#[test]
fn tn93_kernels_agree_on_unambiguous_input() {
    let a = make_seq(120, |_, b| b);
    // ~5% divergence, transitions only (A<->G, C<->T at fixed sites).
    let b = make_seq(120, |i, b| {
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
    // ~10% divergence, mixed transversions.
    let c = make_seq(120, |i, b| {
        if i % 12 == 3 {
            match b {
                b'A' => b'C',
                b'C' => b'A',
                b'G' => b'T',
                _ => b'G',
            }
        } else {
            b
        }
    });

    let modes = [
        AmbiguityMode::Resolve,
        AmbiguityMode::Average,
        AmbiguityMode::Skip,
        AmbiguityMode::GapMismatch,
        AmbiguityMode::Subset([false; 18]),
    ];
    for (x, y, identical) in [
        (&a, &a, true),
        (&a, &b, false),
        (&a, &c, false),
        (&b, &c, false),
    ] {
        let m = manifold_tn93::compute_pairwise_tn93(&encode(x), &encode(y));
        for mode in &modes {
            let t = compute_pairwise_tn93(&encode_tn93(x), &encode_tn93(y), mode);
            assert!(
                t.is_finite(),
                "trace50 kernel reported NaN for a resolvable pair"
            );
            if identical {
                assert_eq!(t, m, "identical sequences must agree exactly");
            } else {
                assert!(
                    (t - m).abs() < 5e-3,
                    "kernel divergence beyond documented approximation gap: \
                     trace50={t} manifold={m} (mode {mode:?})"
                );
            }
        }
    }
}

/// The kernels disagree on *unresolvable* input by design: the oracle port
/// returns NaN ("the binary would not report this pair"), the generic
/// kernel returns 0.0 (the historical trace50 contract, kept for callers
/// that impute missing distances themselves). Pin the divergence so a
/// refactor does not silently unify the sentinels.
#[test]
fn tn93_unresolvable_sentinels_diverge() {
    // Fewer than 101 shared non-gap sites -> below the oracle's
    // min-overlap; the generic kernel still scores the pair.
    let a = make_seq(50, |_, b| b);
    let t = compute_pairwise_tn93(&encode_tn93(&a), &encode_tn93(&a), &AmbiguityMode::Resolve);
    let m = manifold_tn93::compute_pairwise_tn93(&encode(&a), &encode(&a));
    assert!(t.is_nan(), "trace50 should withhold the pair (NaN)");
    assert_eq!(m, 0.0);

    // Zero valid sites at all (gaps in both seqs -> no non-gap span in
    // trace50; mask 0 everywhere for manifold) -> NaN vs 0.0.
    let gaps = make_seq(120, |_, _| b'-');
    let t = compute_pairwise_tn93(
        &encode_tn93(&gaps),
        &encode_tn93(&gaps),
        &AmbiguityMode::Resolve,
    );
    let m = manifold_tn93::compute_pairwise_tn93(&encode(&gaps), &encode(&gaps));
    assert!(t.is_nan());
    assert_eq!(m, 0.0);
}

// ---- Header-date parsers --------------------------------------------------

/// The parsers share one format exactly: a `|`-delimited decimal year in
/// [1950, 2035]. Pin the agreement.
#[test]
fn date_parsers_agree_on_pipe_decimal_year() {
    for h in ["seq|2013.81", "ID|2001.0", "x|1950.0"] {
        let t = trace50_parse_date(h);
        let m = manifold_parse_date(h);
        assert_eq!(t, m, "decimal-year divergence on {h:?}");
        assert!(t.is_finite());
    }
    // Both refuse headers with no parseable date.
    assert!(trace50_parse_date("no date in this").is_nan());
    assert!(manifold_parse_date("no date in this").is_nan());
}

/// Everywhere else the two parsers diverge *by design* — trace50 uses
/// calendar-precise DANNO path-b semantics, manifold uses the simpler
/// first-token rules it inherited from the old trace50 header parser.
/// These pins are the checklist a merge refactor must resolve explicitly.
#[test]
fn date_parsers_documented_divergences() {
    // Bare year in a `|` field: trace50 mid-year (+0.5), manifold the
    // numeric token as-is.
    assert_eq!(trace50_parse_date("ID|2015"), 2015.5);
    assert_eq!(manifold_parse_date("ID|2015"), 2015.0);

    // Underscore year: same split.
    assert_eq!(trace50_parse_date("HCV_2015"), 2015.5);
    assert_eq!(manifold_parse_date("HCV_2015"), 2015.0);

    // ISO calendar date: trace50 converts via day-of-year/year-length;
    // manifold approximates with (month-1)/12 + (day-1)/365.25.
    let t = trace50_parse_date("ID|2021-04-15");
    let m = manifold_parse_date("ID|2021-04-15");
    assert_eq!(t, dates::ymd_to_decimal(2021, 4, 15).unwrap());
    assert!((m - (2021.0 + 3.0 / 12.0 + 14.0 / 365.25)).abs() < 1e-9);
    assert!((t - m).abs() > 1e-3, "ISO formulas must stay distinct");

    // Invalid calendar day (Feb 30): trace50 validates and rejects the
    // field; manifold's formula accepts any numeric day.
    assert!(trace50_parse_date("ID|2020-02-30").is_nan());
    assert!(manifold_parse_date("ID|2020-02-30").is_finite());

    // DC (Perez-Losada) header: DANNO-specific, only trace50 resolves it.
    assert_eq!(trace50_parse_date("B.US.2013.DC9_0121022_PR.X"), 2013.5);
    assert!(manifold_parse_date("B.US.2013.DC9_0121022_PR.X").is_nan());
}
