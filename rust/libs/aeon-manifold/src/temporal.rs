//! Date extraction from sequence headers.
//!
//! Contract (canonical Rust semantics, formerly `trace50::tn93`):
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
//!
//! The second half of this module holds the calendar-precise decimal-year
//! math (`ymd_to_decimal`, `ym_to_decimal`, `month_from_abbr`) shared with
//! `trace50::dates`: canonical day-of-year midpoints, so `2021-04-15` is
//! `2021 + 104.5/365`, not the approximate `(m-1)/12 + (d-1)/365.25` used
//! by the header parser above. trace50 re-exports them for its
//! DANNO-specific parsers; the two conventions are intentionally distinct
//! (see `trace50/tests/test_kernel_seam.rs`).

/// Days in each month of a non-leap year, indexed `[month - 1]`.
const MONTH_DAYS: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const MONTH_ABBR: [&str; 12] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];

#[inline]
pub fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// Days in calendar month `m` (1-based) of year `y`. Callers guarantee
/// `1 <= m <= 12`; out-of-range `m` is a programming error and panics.
pub fn days_in_month(y: i64, m: u32) -> u32 {
    if m == 2 && is_leap(y) {
        29
    } else {
        MONTH_DAYS[(m - 1) as usize]
    }
}

/// Days in the months before `m` (1-based) of year `y`.
fn days_before_month(y: i64, m: u32) -> u32 {
    let mut cum: u32 = MONTH_DAYS[..(m - 1) as usize].iter().sum();
    if m > 2 && is_leap(y) {
        cum += 1;
    }
    cum
}

/// Calendar length of year `y` in days (365 or 366).
pub fn year_len(y: i64) -> f64 {
    if is_leap(y) {
        366.0
    } else {
        365.0
    }
}

/// Decimal year of the middle of calendar day `y-m-d`, or `None` if the date
/// does not exist (Feb 30, month 13, ...). The midpoint convention matches
/// `run_danno.ymd_to_decimal`: day `d` is `y + (doy - 0.5) / days_in_year`.
pub fn ymd_to_decimal(y: i64, m: u32, d: u32) -> Option<f64> {
    if !(1..=12).contains(&m) {
        return None;
    }
    if !(1..=days_in_month(y, m)).contains(&d) {
        return None;
    }
    let doy = days_before_month(y, m) + d;
    Some(y as f64 + (doy as f64 - 0.5) / year_len(y))
}

/// Decimal year of the middle of month `m` of year `y`, or `None` for an
/// invalid month (`fetch_dates.decimal_year(y, m)`).
pub fn ym_to_decimal(y: i64, m: u32) -> Option<f64> {
    if !(1..=12).contains(&m) {
        return None;
    }
    let mid = days_before_month(y, m) as f64 + days_in_month(y, m) as f64 / 2.0;
    Some(y as f64 + mid / year_len(y))
}

/// Three-letter month abbreviation (`JAN`, `feb`, ...) to its number.
pub fn month_from_abbr(s: &str) -> Option<u32> {
    if s.len() != 3 || !s.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    let up = s.to_ascii_uppercase();
    MONTH_ABBR
        .iter()
        .position(|&a| a == up)
        .map(|i| i as u32 + 1)
}

/// Extracts a fractional calendar year from a FASTA header, or NaN.
pub fn parse_date_from_header(header: &str) -> f64 {
    for token in header.split(|c| c == '|' || c == '_' || c == '/' || c == ' ') {
        // Direct float/integer parse (e.g. 2014.25 or 2014)
        if let Ok(val) = token.parse::<f64>() {
            if (1950.0..=2035.0).contains(&val) {
                return val;
            }
        }
        // ISO 8601 calendar date (e.g. 2021-04-15 or 2021-04): calendar-
        // precise midpoints via the shared helpers; impossible months or
        // days are rejected rather than silently folded into a wrong date.
        let parts: Vec<&str> = token.split('-').collect();
        if parts.len() == 2 || parts.len() == 3 {
            if let Ok(year) = parts[0].parse::<i64>() {
                if (1950..=2035).contains(&year) {
                    if let Ok(month) = parts[1].parse::<u32>() {
                        let d = if parts.len() == 3 {
                            parts[2]
                                .parse::<u32>()
                                .ok()
                                .and_then(|day| ymd_to_decimal(year, month, day))
                        } else {
                            ym_to_decimal(year, month)
                        };
                        if let Some(d) = d {
                            return d;
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
        // Calendar-precise midpoints, identical to ymd_to_decimal /
        // ym_to_decimal (April 15 2021 is day 105 of 365).
        let d = parse_date_from_header("seq|2021-04-15");
        assert_eq!(d, ymd_to_decimal(2021, 4, 15).unwrap());
        let m = parse_date_from_header("seq|2021-04");
        assert_eq!(m, ym_to_decimal(2021, 4).unwrap());
        // Impossible calendar dates are rejected, not approximated.
        assert!(parse_date_from_header("seq|2020-02-30").is_nan());
        assert!(parse_date_from_header("seq|2021-13-01").is_nan());
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

    #[test]
    fn test_ymd_to_decimal_calendar_precise() {
        // 2013-10-22: doy 295, midpoint (295 - 0.5) / 365.
        assert_eq!(ymd_to_decimal(2013, 10, 22), Some(2013.0 + 294.5 / 365.0));
        // Leap-year day 60 (Feb 29) divides by 366.
        assert_eq!(ymd_to_decimal(2020, 2, 29), Some(2020.0 + 59.5 / 366.0));
        // Impossible dates give None rather than an approximate value.
        assert_eq!(ymd_to_decimal(2021, 2, 29), None);
        assert_eq!(ymd_to_decimal(2020, 2, 30), None);
        assert_eq!(ymd_to_decimal(2020, 13, 1), None);
        assert_eq!(ymd_to_decimal(2020, 0, 1), None);
        assert_eq!(ymd_to_decimal(2020, 1, 0), None);
    }

    #[test]
    fn test_ym_to_decimal_mid_month() {
        // April 2021: days before = 90, mid = 90 + 30/2 = 105.
        assert_eq!(ym_to_decimal(2021, 4), Some(2021.0 + 105.0 / 365.0));
        // February midpoint honors leap years (31 + 29/2 = 45.5 in 2020).
        assert_eq!(ym_to_decimal(2020, 2), Some(2020.0 + 45.5 / 366.0));
        assert_eq!(ym_to_decimal(2021, 13), None);
    }

    #[test]
    fn test_month_from_abbr() {
        assert_eq!(month_from_abbr("Jan"), Some(1));
        assert_eq!(month_from_abbr("DEC"), Some(12));
        assert_eq!(month_from_abbr("Foo"), None);
        assert_eq!(month_from_abbr("J1N"), None);
        assert_eq!(month_from_abbr("JANU"), None);
    }
}
