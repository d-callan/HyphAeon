use crate::types::Alignment;

/// Encodes a nucleotide ASCII character into integer state:
/// 1: A, 2: C, 3: G, 4: T/U, 0: gap/ambiguity
#[inline(always)]
pub fn encode_char(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 1,
        b'C' | b'c' => 2,
        b'G' | b'g' => 3,
        b'T' | b't' | b'U' | b'u' => 4,
        _ => 0,
    }
}

/// Parses a FASTA format string into an Alignment struct
pub fn parse_fasta(fasta_str: &str) -> Result<Alignment, String> {
    let mut taxa = Vec::new();
    let mut sequences: Vec<Vec<u8>> = Vec::new();
    let mut current_seq: Vec<u8> = Vec::new();

    for line in fasta_str.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('>') {
            if !current_seq.is_empty() {
                sequences.push(current_seq);
                current_seq = Vec::new();
            }
            let header = trimmed[1..].trim().to_string();
            // Take first token up to whitespace as canonical taxon name
            let name = header.split_whitespace().next().unwrap_or("taxon").to_string();
            taxa.push(name);
        } else {
            for b in trimmed.bytes() {
                if !b.is_ascii_whitespace() {
                    current_seq.push(encode_char(b));
                }
            }
        }
    }
    if !current_seq.is_empty() {
        sequences.push(current_seq);
    }

    if taxa.is_empty() {
        return Err("No sequences found in FASTA input".to_string());
    }

    let length = sequences[0].len();
    for (i, seq) in sequences.iter().enumerate() {
        if seq.len() != length {
            return Err(format!(
                "Sequence {} ('{}') has length {} but expected {}",
                i, taxa[i], seq.len(), length
            ));
        }
    }

    let num_taxa = taxa.len();
    let mut matrix = Vec::with_capacity(num_taxa * length);
    for seq in sequences {
        matrix.extend_from_slice(&seq);
    }

    Ok(Alignment::new(taxa, length, matrix))
}

/// Counts parsimony-informative sites in an alignment
/// A site is informative if at least two distinct valid character states (1..4) each appear >= 2 times
pub fn count_parsimony_informative_sites(aln: &Alignment) -> usize {
    if aln.num_taxa < 4 || aln.length == 0 {
        return 0;
    }

    let mut count = 0;
    for u in 0..aln.length {
        let mut char_counts = [0usize; 5];
        for i in 0..aln.num_taxa {
            let c = aln.get(i, u) as usize;
            char_counts[c] += 1;
        }

        // Check valid characters 1..4
        let mut states_with_ge_2 = 0;
        for c in 1..=4 {
            if char_counts[c] >= 2 {
                states_with_ge_2 += 1;
            }
        }

        if states_with_ge_2 >= 2 {
            count += 1;
        }
    }
    count
}

/// Computes divergence-adaptive informative site admission floor:
/// M_min(N) = max(6, ceil(2.5 * ln(N)))
pub fn compute_min_informative_sites(num_taxa: usize) -> usize {
    let n = (num_taxa.max(2)) as f64;
    let min_val = (2.5 * n.ln()).ceil() as usize;
    min_val.max(6)
}

/// Trims ragged boundaries where coverage < min(3, N)
pub fn trim_coverage_envelope(aln: &Alignment) -> (Alignment, usize) {
    let k_min = aln.num_taxa.min(3);
    let mut first_col = None;
    let mut last_col = None;

    for u in 0..aln.length {
        let mut valid_taxa = 0;
        for i in 0..aln.num_taxa {
            if aln.get(i, u) > 0 {
                valid_taxa += 1;
            }
        }
        if valid_taxa >= k_min {
            if first_col.is_none() {
                first_col = Some(u);
            }
            last_col = Some(u);
        }
    }

    match (first_col, last_col) {
        (Some(start), Some(end)) if start > 0 || end + 1 < aln.length => {
            let new_len = end - start + 1;
            let mut new_matrix = Vec::with_capacity(aln.num_taxa * new_len);
            for i in 0..aln.num_taxa {
                let row = aln.row(i);
                new_matrix.extend_from_slice(&row[start..=end]);
            }
            (Alignment::new(aln.taxa.clone(), new_len, new_matrix), start)
        }
        _ => (aln.clone(), 0),
    }
}
