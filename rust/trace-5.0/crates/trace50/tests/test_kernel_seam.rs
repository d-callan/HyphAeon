//! Cross-implementation seam tests for the pieces tracked in REVIEW.md's
//! dedup list.
//!
//! The TN93 kernel is now single — `aeon_manifold::tn93` — with the two
//! historical behaviors preserved as explicit options. Two header-date
//! parsers still exist with different precedence rules. Each side is
//! already tested against its own oracle (test_tn93_oracle.rs,
//! test_dates_oracle.rs, aeon-manifold inline tests). This file pins *where
//! the paths agree* and *where they intentionally diverge*. If a refactor
//! changes any of these, this file fails loudly.
//!
//! - `aeon_manifold::tn93` (`Counting::Faithful`) — bit-faithful port of
//!   the `tn93` binary; `trace50::tn93` re-exports it verbatim.
//! - `aeon_manifold::tn93` (`Counting::Legacy`) — the pre-unification
//!   manifold counting scheme, kept verbatim for chronaeon pending
//!   revalidation under Faithful.
//! - `trace50::dates`         — DANNO path-b rules: DC headers,
//!   underscore-years, `|` fields via calendar-precise ymd math.
//! - `aeon_manifold::temporal`— generic first-token-wins header parser
//!   (the semantics formerly in `trace50::tn93`).

use aeon_manifold::temporal::parse_date_from_header as manifold_parse_date;
use aeon_manifold::tn93::{self as manifold_tn93, Counting, Reporting, Tn93Options};
use trace50::dates::{self, parse_date_from_header as trace50_parse_date};
use trace50::tn93::{char_to_tn93_code, compute_pairwise_tn93, AmbiguityMode};

// ---- TN93 kernel ----------------------------------------------------------

/// Kernel input: the 18-code tn93 alphabet (not raw ASCII — raw bytes
/// index the per-code tables and panic). `char_to_tn93_code` is the
/// re-export from aeon-manifold; there is only one encoder now.
fn encode_tn93(seq: &str) -> Vec<u8> {
    seq.bytes().map(char_to_tn93_code).collect()
}

/// 120-nt sequence from a repeating ACGT block with `mutate` applied at
/// chosen positions. ACGT-only, so every AmbiguityMode must agree.
fn make_seq(len: usize, mutate: impl Fn(usize, u8) -> u8) -> String {
    let bases = b"ACGT";
    (0..len).map(|i| mutate(i, bases[i % 4]) as char).collect()
}

/// `trace50::tn93` no longer has a kernel of its own: it re-exports
/// `aeon_manifold::tn93`, so the trace50 path and the manifold path under
/// `Tn93Options::tn93_compat` are the same function. Pin bitwise equality
/// on every (pair, mode) so a future refactor cannot reintroduce a second
/// implementation drifting on floating-point order.
#[test]
fn tn93_trace50_path_is_manifold_kernel() {
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
    for (x, y) in [(&a, &a), (&a, &b), (&a, &c), (&b, &c)] {
        for mode in &modes {
            let t = compute_pairwise_tn93(&encode_tn93(x), &encode_tn93(y), mode);
            let m = manifold_tn93::compute_pairwise_tn93(&encode_tn93(x), &encode_tn93(y), mode);
            assert_eq!(
                t.to_bits(),
                m.to_bits(),
                "trace50 re-export diverged from the manifold kernel (mode {mode:?})"
            );
        }
    }
}

/// `Counting::Legacy` (chronaeon's pre-unification scheme, kept verbatim
/// for its published outputs) estimates base frequencies from the union
/// of both sequences' IUPAC masks per valid site, while `Counting::
/// Faithful` replicates the tn93 binary's marginal frequencies. On
/// unambiguous input the estimators coincide, so the residual gap is
/// floating-point order only (observed <= ~5e-4 absolute here). The 5e-3
/// bound tolerates that documented gap while tripping on a formula-level
/// regression (>= ~0.05 here). Identical sequences must agree exactly.
#[test]
fn tn93_legacy_counting_divergence_is_bounded() {
    let a = make_seq(120, |_, b| b);
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

    let legacy = Tn93Options::legacy_all_pairs();
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
        let l = manifold_tn93::pairwise_distance(&encode_tn93(x), &encode_tn93(y), &legacy)
            .expect("legacy scores every comparable pair");
        for mode in &modes {
            let f = manifold_tn93::pairwise_distance(
                &encode_tn93(x),
                &encode_tn93(y),
                &Tn93Options::tn93_compat(mode.clone()),
            )
            .expect("faithful scores a 120-site pair under tn93 reporting");
            if identical {
                assert_eq!(l, f, "identical sequences must agree exactly");
            } else {
                assert!(
                    (l - f).abs() < 5e-3,
                    "legacy divergence beyond documented approximation gap: \
                     legacy={l} faithful={f} (mode {mode:?})"
                );
            }
        }
    }
}

/// Reporting is a separate axis from counting. `Tn93Compat` withholds the
/// pairs the binary would not write (overlap <= 100 sites, or d > 1.0);
/// `AllPairs` scores every pair with at least one comparable site; the
/// legacy scheme predates the reporting contract and also scores every
/// comparable pair. A pair with zero comparable sites is unscored under
/// every scheme.
#[test]
fn tn93_reporting_gates_are_explicit() {
    // 50 shared non-gap sites < TN93_MIN_OVERLAP: withheld under Tn93Compat
    // (and on the trace50 path), scored under AllPairs and Legacy.
    let short = encode_tn93(&make_seq(50, |_, b| b));
    let compat = Tn93Options::tn93_compat(AmbiguityMode::Resolve);
    let all_pairs = Tn93Options {
        counting: Counting::Faithful(AmbiguityMode::Resolve),
        reporting: Reporting::AllPairs,
    };
    assert_eq!(
        manifold_tn93::pairwise_distance(&short, &short, &compat),
        None
    );
    assert!(compute_pairwise_tn93(&short, &short, &AmbiguityMode::Resolve).is_nan());
    assert_eq!(
        manifold_tn93::pairwise_distance(&short, &short, &all_pairs),
        Some(0.0)
    );
    assert_eq!(
        manifold_tn93::pairwise_distance(&short, &short, &Tn93Options::legacy_all_pairs()),
        Some(0.0)
    );

    // Zero comparable sites: unscored under every scheme and reporting.
    let gaps = encode_tn93(&make_seq(120, |_, _| b'-'));
    for opts in [&compat, &all_pairs, &Tn93Options::legacy_all_pairs()] {
        assert_eq!(manifold_tn93::pairwise_distance(&gaps, &gaps, opts), None);
    }
}

// ---- Header-date parsers --------------------------------------------------

/// The parsers share the calendar-correct core exactly: `|`-delimited
/// decimal years in [1950, 2035], ISO calendar dates via the shared
/// ymd_to_decimal/ym_to_decimal helpers (calendar-precise midpoints), and
/// rejection of impossible calendar days. Pin the agreement.
#[test]
fn date_parsers_agree_on_pipe_decimal_year() {
    for h in ["seq|2013.81", "ID|2001.0", "x|1950.0"] {
        let t = trace50_parse_date(h);
        let m = manifold_parse_date(h);
        assert_eq!(t, m, "decimal-year divergence on {h:?}");
        assert!(t.is_finite());
    }
    // ISO calendar dates resolve identically through the shared helpers.
    for h in ["ID|2021-04-15", "seq|2020-12-31", "x|2000-02-29"] {
        let t = trace50_parse_date(h);
        let m = manifold_parse_date(h);
        assert_eq!(t, m, "ISO-date divergence on {h:?}");
        assert!(t.is_finite());
    }
    // Both reject impossible calendar days and dateless headers.
    assert!(trace50_parse_date("ID|2020-02-30").is_nan());
    assert!(manifold_parse_date("ID|2020-02-30").is_nan());
    assert!(trace50_parse_date("no date in this").is_nan());
    assert!(manifold_parse_date("no date in this").is_nan());
}

/// The remaining divergences are semantic, not arithmetic: trace50 uses
/// DANNO path-b midpoint conventions and format-priority scanning, manifold
/// keeps the simpler first-token rules it inherited from the old trace50
/// header parser. These pins are the checklist the chronaeon migration
/// must resolve explicitly.
#[test]
fn date_parsers_documented_divergences() {
    // Bare year in a `|` field: trace50 mid-year (+0.5), manifold the
    // numeric token as-is.
    assert_eq!(trace50_parse_date("ID|2015"), 2015.5);
    assert_eq!(manifold_parse_date("ID|2015"), 2015.0);

    // Underscore year: same split.
    assert_eq!(trace50_parse_date("HCV_2015"), 2015.5);
    assert_eq!(manifold_parse_date("HCV_2015"), 2015.0);

    // Precedence: manifold is first-token-wins over a `|_/` tokenizer, so
    // the leading `_2015` wins; trace50's format-priority scan skips it
    // (boundary rules) and resolves the ISO field instead.
    assert_eq!(manifold_parse_date("X_2015|2020-06-01"), 2015.0);
    assert_eq!(
        trace50_parse_date("X_2015|2020-06-01"),
        dates::ymd_to_decimal(2020, 6, 1).unwrap()
    );

    // DC (Perez-Losada) header: DANNO-specific, only trace50 resolves it.
    assert_eq!(trace50_parse_date("B.US.2013.DC9_0121022_PR.X"), 2013.5);
    assert!(manifold_parse_date("B.US.2013.DC9_0121022_PR.X").is_nan());
}
