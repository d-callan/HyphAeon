//! Collection-date parsing for FASTA headers and metadata tables.
//!
//! The rules follow the Python reference used for DANNO (path b):
//!
//! * FASTA headers: `validation/run_danno.py` `parse_date` (`parse_dc`, then
//!   `parse_underscore_year`), with day-of-year decimals from
//!   `ymd_to_decimal`.
//! * Metadata date strings: `validation/fetch_dates.py`
//!   `parse_collection_date` (the GenBank `collection_date` forms that feed
//!   `run_danno.py --dates`): `DD-Mon-YYYY`, `Mon-YYYY`, `YYYY-MM-DD`,
//!   `YYYY-MM`, `YYYY`, with `a/b` ranges reduced to `a`. Calendar dates are
//!   validated as in `ymd_to_decimal` (an impossible day gives no date).
//!
//! Every date is the midpoint of the interval it names: a day `d` is
//! `y + (doy - 0.5) / days_in_year`, a month is its middle, and a bare year is
//! `y + 0.5`. A sequence whose date cannot be parsed is undated (`NaN`); it
//! is never assigned a date it does not carry.
//!
//! The calendar math (`ymd_to_decimal`, `ym_to_decimal`, leap-year and
//! month-length helpers) lives in `aeon_manifold::temporal` — this module
//! keeps only the DANNO-specific parsing rules built on it.

use aeon_manifold::temporal::month_from_abbr;
pub use aeon_manifold::temporal::{ym_to_decimal, ymd_to_decimal};

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `(19|20)\d{2}` as a full match.
fn is_19_20_year(s: &str) -> bool {
    is_19_20_year_bytes(s.as_bytes())
}

fn is_19_20_year_bytes(b: &[u8]) -> bool {
    b.len() == 4 && b.iter().all(|c| c.is_ascii_digit()) && (b[..2] == *b"19" || b[..2] == *b"20")
}

/// Perez-Losada / DC header: `B.US.2013.DC1000_0131022_PR_RT.MF455515`.
///
/// The fourth dot-field carries `_0YYMMDD` followed by `_` or the end of the
/// field; it gives a day-precision date when `20YY` equals the explicit year
/// field and the calendar date exists. Otherwise the year field alone gives
/// `year + 0.5` (`run_danno.parse_dc`).
pub fn parse_dc(header: &str) -> Option<f64> {
    let parts: Vec<&str> = header.split('.').collect();
    if parts.len() < 4 {
        return None;
    }
    let year: i64 = if is_19_20_year(parts[2]) {
        parts[2].parse().ok()?
    } else {
        return None;
    };

    // re.search(r'_0(\d{6})(?:_|$)', parts[3]): leftmost match
    let f = parts[3].as_bytes();
    let mut block = None;
    for p in 0..f.len() {
        if p + 8 > f.len() {
            break;
        }
        if f[p] == b'_'
            && f[p + 1] == b'0'
            && f[p + 2..p + 8].iter().all(|b| b.is_ascii_digit())
            && (p + 8 == f.len() || f[p + 8] == b'_')
        {
            block = Some(&parts[3][p + 2..p + 8]);
            break;
        }
    }
    if let Some(b) = block {
        let yy: i64 = b[0..2].parse().ok()?;
        let mm: u32 = b[2..4].parse().ok()?;
        let dd: u32 = b[4..6].parse().ok()?;
        if 2000 + yy == year {
            if let Some(dec) = ymd_to_decimal(year, mm, dd) {
                return Some(dec);
            }
        }
    }
    Some(year as f64 + 0.5)
}

/// `^[A-Za-z0-9_]+\.[A-Z]{2}\.(19|20)\d{2}\.`
fn looks_like_dc(header: &str) -> bool {
    let parts: Vec<&str> = header.splitn(4, '.').collect();
    parts.len() == 4
        && !parts[0].is_empty()
        && parts[0]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && parts[1].len() == 2
        && parts[1].bytes().all(|b| b.is_ascii_uppercase())
        && is_19_20_year(parts[2])
}

/// Whether the character after a match satisfies `(?=[_\s]|$)`.
fn boundary_after(s: &str, pos: usize) -> bool {
    match s[pos..].chars().next() {
        None => true,
        Some(c) => c == '_' || c.is_whitespace(),
    }
}

/// `_YYYY.NN` (decimal year, 1-2 fractional digits) anywhere in the header,
/// else `_YYYY` (year, placed mid-year), each followed by `_`, whitespace or
/// the end, with 1980 <= YYYY <= 2030 (`run_danno.parse_underscore_year`).
pub fn parse_underscore_year(header: &str) -> Option<f64> {
    let b = header.as_bytes();
    let starts: Vec<usize> = (0..b.len())
        .filter(|&p| b[p] == b'_' && p + 5 <= b.len() && is_19_20_year_bytes(&b[p + 1..p + 5]))
        .collect();

    for &p in &starts {
        let y: i64 = header[p + 1..p + 5].parse().ok()?;
        if b.get(p + 5) != Some(&b'.') {
            continue;
        }
        let fs = p + 6;
        let mut fe = fs;
        while fe < b.len() && b[fe].is_ascii_digit() {
            fe += 1;
        }
        // \d{1,2} followed by the boundary: a longer digit run never matches
        if (1..=2).contains(&(fe - fs)) && boundary_after(header, fe) && (1980..=2030).contains(&y)
        {
            let frac: f64 = format!("0.{}", &header[fs..fe]).parse().ok()?;
            return Some(y as f64 + frac);
        }
    }
    for &p in &starts {
        let y: i64 = header[p + 1..p + 5].parse().ok()?;
        if boundary_after(header, p + 5) && (1980..=2030).contains(&y) {
            return Some(y as f64 + 0.5);
        }
    }
    None
}

/// Parses one metadata date value to a decimal year.
///
/// Accepts the `fetch_dates.parse_collection_date` forms (`DD-Mon-YYYY`,
/// `Mon-YYYY`, `YYYY-MM-DD`, `YYYY-MM`, `YYYY`; `a/b` ranges use `a`), and a
/// decimal year such as `2013.806849` in [1950, 2035], which is taken as is.
/// Fields are positional; day-first or month-first numeric forms such as
/// `10/22/2013` are not guessed at and give `None`.
pub fn parse_date_value(raw: &str) -> Option<f64> {
    let mut s = raw.trim().trim_matches('"').trim();
    if s.is_empty() {
        return None;
    }
    if let Some(first) = s.split('/').next() {
        if s.contains('/') {
            s = first.trim();
        }
    }
    let f: Vec<&str> = s.split('-').collect();
    match f.as_slice() {
        // DD-Mon-YYYY
        [d, mon, y]
            if (1..=2).contains(&d.len())
                && all_digits(d)
                && y.len() == 4
                && all_digits(y)
                && mon.len() == 3
                && mon.bytes().all(|b| b.is_ascii_alphabetic()) =>
        {
            ymd_to_decimal(y.parse().ok()?, month_from_abbr(mon)?, d.parse().ok()?)
        }
        // Mon-YYYY
        [mon, y]
            if mon.len() == 3
                && mon.bytes().all(|b| b.is_ascii_alphabetic())
                && y.len() == 4
                && all_digits(y) =>
        {
            ym_to_decimal(y.parse().ok()?, month_from_abbr(mon)?)
        }
        // YYYY-MM-DD
        [y, m, d]
            if y.len() == 4
                && all_digits(y)
                && m.len() == 2
                && all_digits(m)
                && d.len() == 2
                && all_digits(d) =>
        {
            ymd_to_decimal(y.parse().ok()?, m.parse().ok()?, d.parse().ok()?)
        }
        // YYYY-MM
        [y, m] if y.len() == 4 && all_digits(y) && m.len() == 2 && all_digits(m) => {
            ym_to_decimal(y.parse().ok()?, m.parse().ok()?)
        }
        // YYYY
        [y] if y.len() == 4 && all_digits(y) => Some(y.parse::<i64>().ok()? as f64 + 0.5),
        // decimal year
        [v] => {
            let val: f64 = v.parse().ok()?;
            if val.is_finite() && (1950.0..=2035.0).contains(&val) {
                Some(val)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Years accepted from a header `|` field: [1950, 2036), i.e. calendar years
/// 1950-2035, the range of the decimal-year form in [`parse_date_value`].
const PIPE_YEAR_MIN: f64 = 1950.0;
const PIPE_YEAR_END: f64 = 2036.0;

/// Collection date from a FASTA defline, as a decimal year, or `NaN` when the
/// header carries no parseable date.
///
/// Formats, in order: the DC form (`parse_dc`), `_YYYY.NN` / `_YYYY`
/// (`parse_underscore_year`), then any `|`-delimited field holding a
/// [`parse_date_value`] date (`ID|2013-10-22`, `ID|2013.81`) in years
/// 1950-2035. The `|` rule has no counterpart in run_danno.py; the year range
/// (that of the decimal-year form) keeps numeric ID fields such as
/// `PT|1001|clone0` or `PT|0042` from being read as dates.
pub fn parse_date_from_header(header: &str) -> f64 {
    let h = header.trim();
    if looks_like_dc(h) {
        if let Some(d) = parse_dc(h) {
            return d;
        }
    }
    if let Some(d) = parse_underscore_year(h) {
        return d;
    }
    if h.contains('|') {
        for field in h.split('|') {
            if let Some(d) = parse_date_value(field) {
                if (PIPE_YEAR_MIN..PIPE_YEAR_END).contains(&d) {
                    return d;
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
    fn test_dc_header_day_and_fallback() {
        // 2013-10-22: doy 295, (295 - 0.5) / 365
        let d = parse_date_from_header("B.US.2013.DC1000_0131022_PR_RT.MF455515");
        assert_eq!(d, 2013.0 + 294.5 / 365.0);
        // YY disagrees with the year field: year only
        assert_eq!(parse_date_from_header("B.US.2013.DC9_0121022_PR.X"), 2013.5);
    }

    #[test]
    fn test_iso_is_positional() {
        assert_eq!(parse_date_value("2013-10-22"), ymd_to_decimal(2013, 10, 22));
        assert_eq!(parse_date_value("2013-02-30"), None);
        assert_eq!(parse_date_value("10/22/2013"), None);
    }
}
