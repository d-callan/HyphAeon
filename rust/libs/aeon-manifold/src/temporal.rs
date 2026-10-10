//! Date extraction from sequence headers.
//!
//! Contract (canonical Rust semantics, formerly `trace50_core::tn93`):
//! the header is split on `|`, `_`, `/`, and space; tokens are examined
//! left-to-right and the FIRST resolvable token wins.
//!
//! - Numeric tokens: any `f64` in `[1950, 2035]` — covers bare years and
//!   decimal years of any precision (`2015`, `2015.8`, `2015.85`).
//! - ISO 8601 tokens: `YYYY-MM` or `YYYY-MM-DD` convert to a fractional
//!   year via `year + (month-1)/12 + (day-1)/365.25`; a missing day defaults
//!   to the 15th. An unparseable day also defaults to 15 (the historical
//!   fallback), while an out-of-range month rejects the token.
//! - No resolvable token => `f64::NAN`.
//!
//! Note the first-token-wins asymmetry: an embedded bare year
//! (`HCV_REPEAT_2015|2015-01-10`) resolves to `2015.0` before the later ISO
//! date is considered. This is pinned by the cross-language conformance
//! suite — the Python engine prefers the ISO match instead.

/// Extracts a fractional calendar year from a FASTA header, or NaN.
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
    fn test_decimal_and_bare_year() {
        assert_eq!(parse_date_from_header("seq|2015.8"), 2015.8);
        assert_eq!(parse_date_from_header("seq_2014"), 2014.0);
    }

    #[test]
    fn test_iso_dates() {
        // 2021-04-15 -> 2021 + 3/12 + 14/365.25
        let d = parse_date_from_header("seq|2021-04-15");
        assert!((d - (2021.0 + 3.0 / 12.0 + 14.0 / 365.25)).abs() < 1e-12);
        // YYYY-MM defaults to day 15
        let m = parse_date_from_header("seq|2021-04");
        assert!((m - (2021.0 + 3.0 / 12.0 + 14.0 / 365.25)).abs() < 1e-12);
    }

    #[test]
    fn test_first_token_wins() {
        assert_eq!(parse_date_from_header("X_2015|2020-06-01"), 2015.0);
    }

    #[test]
    fn test_out_of_range_and_nan() {
        assert!(parse_date_from_header("seq|1800").is_nan());
        assert!(parse_date_from_header("no_date_here").is_nan());
    }
}
