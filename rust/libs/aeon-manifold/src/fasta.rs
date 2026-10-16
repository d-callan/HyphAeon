//! Shared FASTA tokenizer for the engine family.
//!
//! Produces raw `(name, seq)` records; each engine applies its own encoding
//! projection afterward (`char_to_iupac_mask`, `encode_char`, …). Length
//! policy is explicit via [`ParseMode`]: `Strict` rejects ragged input
//! (rhizaeon contract), `PadToMax` right-pads with `-` (trace50 contract).
//!
//! Tokenizer contract (union of both historical parsers):
//! - `name` = first whitespace-delimited token after `>`; `""` when absent.
//! - Sequence bytes are ASCII-uppercased; interior whitespace is stripped.
//! - Sequence lines before the first `>` are ignored.
//! - A header pairs with its sequence even when the sequence is empty —
//!   under `Strict` that is a length error, under `PadToMax` an all-gap row.
//!   (The historical rhizaeon parser silently dropped empty sequences, which
//!   could silently misalign taxa/sequence vectors; the shared tokenizer
//!   deliberately does not reproduce that quirk.)

/// How ragged sequence lengths are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    /// All sequences must match the first sequence's length exactly.
    Strict,
    /// Right-pad shorter sequences with `-` to the maximum length.
    PadToMax,
}

/// A raw FASTA record before encoding.
#[derive(Debug, Clone)]
pub struct FastaRecord {
    /// First whitespace token of the header; `""` when the header is bare `>`.
    pub name: String,
    /// The full defline (everything after `>`, trimmed). Engines that parse
    /// metadata such as collection dates from the whole header (e.g. trace50's
    /// run_danno-style date rules) need this, not just `name`.
    pub defline: String,
    /// Uppercase ASCII sequence bytes, whitespace stripped.
    pub seq: Vec<u8>,
}

/// Tokenizes FASTA text into raw records.
///
/// Errors only when no `>` header is found; callers should map the error to
/// their own message if they need a crate-specific string.
pub fn tokenize_fasta(input: &str) -> Result<Vec<FastaRecord>, String> {
    let mut records = Vec::new();
    let mut current: Option<FastaRecord> = None;

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('>') {
            if let Some(rec) = current.take() {
                records.push(rec);
            }
            let defline = rest.trim().to_string();
            let name = defline.split_whitespace().next().unwrap_or("").to_string();
            current = Some(FastaRecord {
                name,
                defline,
                seq: Vec::new(),
            });
        } else if let Some(rec) = current.as_mut() {
            rec.seq.extend(
                trimmed
                    .bytes()
                    .filter(|b| !b.is_ascii_whitespace())
                    .map(|b| b.to_ascii_uppercase()),
            );
        }
    }
    if let Some(rec) = current {
        records.push(rec);
    }

    if records.is_empty() {
        return Err("No sequences found in FASTA input".to_string());
    }
    Ok(records)
}

/// Resolves sequence lengths per `mode`; returns the aligned length.
///
/// `Strict` uses the FIRST record's length as the expected value and reports
/// errors in the historical rhizaeon message format.
pub fn normalize_lengths(records: &mut [FastaRecord], mode: ParseMode) -> Result<usize, String> {
    debug_assert!(!records.is_empty());
    match mode {
        ParseMode::Strict => {
            let expected = records[0].seq.len();
            for (i, rec) in records.iter().enumerate() {
                if rec.seq.len() != expected {
                    return Err(format!(
                        "Sequence {} ('{}') has length {} but expected {}",
                        i,
                        rec.name,
                        rec.seq.len(),
                        expected
                    ));
                }
            }
            Ok(expected)
        }
        ParseMode::PadToMax => {
            let max_len = records.iter().map(|r| r.seq.len()).max().unwrap_or(0);
            for rec in records.iter_mut() {
                rec.seq.resize(max_len, b'-');
            }
            Ok(max_len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_names_and_uppercase() {
        let fasta = ">seq1 description words\nacgt\n>seq2\nTT\nAA\n";
        let recs = tokenize_fasta(fasta).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].name, "seq1");
        assert_eq!(recs[0].seq, b"ACGT");
        assert_eq!(recs[1].name, "seq2");
        assert_eq!(recs[1].seq, b"TTAA");
    }

    #[test]
    fn test_tokenize_strips_interior_whitespace() {
        let recs = tokenize_fasta(">a\nAC GT\n").unwrap();
        assert_eq!(recs[0].seq, b"ACGT");
    }

    #[test]
    fn test_tokenize_empty_input_errors() {
        assert!(tokenize_fasta("ACGT\nACGT\n").is_err());
        assert!(tokenize_fasta("").is_err());
    }

    #[test]
    fn test_strict_ragged_error_message() {
        let mut recs = tokenize_fasta(">a\nACGT\n>b\nACG\n").unwrap();
        let err = normalize_lengths(&mut recs, ParseMode::Strict).unwrap_err();
        assert_eq!(err, "Sequence 1 ('b') has length 3 but expected 4");
    }

    #[test]
    fn test_pad_to_max_fills_with_gaps() {
        let mut recs = tokenize_fasta(">a\nACGT\n>b\nAC\n").unwrap();
        let len = normalize_lengths(&mut recs, ParseMode::PadToMax).unwrap();
        assert_eq!(len, 4);
        assert_eq!(recs[1].seq, b"AC--");
    }

    #[test]
    fn test_empty_sequence_pairs_with_header() {
        // Header with no sequence is a record too; PadToMax yields all gaps.
        let mut recs = tokenize_fasta(">a\n>b\nACGT\n").unwrap();
        let len = normalize_lengths(&mut recs, ParseMode::PadToMax).unwrap();
        assert_eq!(len, 4);
        assert_eq!(recs[0].seq, b"----");
    }
}
