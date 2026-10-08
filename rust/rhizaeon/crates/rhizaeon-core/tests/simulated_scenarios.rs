use rhizaeon_core::{parse_fasta, RhizAeonEngine};

#[test]
fn test_synthetic_crossover() {
    // Generate synthetic alignment with 4 taxa: P1, P2, R (crossover P1->P2 at 500), O (outgroup)
    let length = 1000;
    let mut s_p1 = Vec::with_capacity(length);
    let mut s_p2 = Vec::with_capacity(length);
    let mut s_r = Vec::with_capacity(length);
    let mut s_o = Vec::with_capacity(length);

    for u in 0..length {
        // Base pattern
        let b1 = if u % 7 == 0 { 'A' } else { 'C' };
        let b2 = if u % 7 == 0 { 'G' } else { 'T' };
        let bo = 'G';
        let br = if u < 500 { b1 } else { b2 };

        s_p1.push(b1);
        s_p2.push(b2);
        s_r.push(br);
        s_o.push(bo);
    }

    let fasta = format!(
        ">P1\n{}\n>P2\n{}\n>R\n{}\n>O\n{}\n",
        s_p1.into_iter().collect::<String>(),
        s_p2.into_iter().collect::<String>(),
        s_r.into_iter().collect::<String>(),
        s_o.into_iter().collect::<String>(),
    );

    let aln = parse_fasta(&fasta).expect("Parse fasta failed");
    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&aln);

    assert!(!result.events.is_empty(), "Should detect synthetic crossover");
    let ev = &result.events[0];
    assert_eq!(ev.candidate_name, "R", "R must be identified as recombinant");
    assert!(ev.p_fisher < 1e-5, "Significant p-value required");
}

#[test]
fn test_synthetic_heterotachy_null() {
    // Clonal tree where one taxon has private mutations (heterotachy rate acceleration)
    let length = 1000;
    let mut s1 = vec!['A'; length];
    let mut s2 = vec!['A'; length];
    let mut s3 = vec!['A'; length];
    let s4 = vec!['A'; length];

    // Some shared tree mutations
    for u in (0..length).step_by(50) {
        s1[u] = 'C';
        s2[u] = 'C';
    }
    // Episodic rate acceleration in s3 (private to s3)
    for u in 300..600 {
        if u % 5 == 0 {
            s3[u] = 'T';
        }
    }

    let fasta = format!(
        ">T1\n{}\n>T2\n{}\n>T3\n{}\n>T4\n{}\n",
        s1.into_iter().collect::<String>(),
        s2.into_iter().collect::<String>(),
        s3.into_iter().collect::<String>(),
        s4.into_iter().collect::<String>(),
    );

    let aln = parse_fasta(&fasta).expect("Parse fasta failed");
    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&aln);

    assert!(result.events.is_empty(), "Private heterotachy burst must NOT trigger false positive recombination");
}

#[test]
fn test_strictly_clonal_monomorphic_null() {
    let fasta = ">S1\nACGTACGTACGTACGT\n>S2\nACGTACGTACGTACGT\n>S3\nACGTACGTACGTACGT\n>S4\nACGTACGTACGTACGT\n";
    let aln = parse_fasta(fasta).expect("Parse fasta failed");
    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&aln);

    assert!(result.events.is_empty(), "Monomorphic sequences must yield 0 events");
    assert_eq!(result.informative_sites, 0);
}
