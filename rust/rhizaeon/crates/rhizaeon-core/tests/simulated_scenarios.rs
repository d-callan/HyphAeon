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

#[test]
fn test_bacterial_snp_compression_scenario() {
    use rhizaeon_core::fasta::extract_informative_snp_alignment;

    // Simulate 60,000 bp bacterial chromosome with sparse polymorphic sites (every 100 bp)
    // Clade 1: P1_1, P1_2 carry 'T'
    // Clade 2: P2_1, P2_2 carry 'C'
    // Outgroup: O carries 'G'
    // Recombinant: R carries Clade 1 ('T') on flanks, Clade 2 ('C') in [20,000 - 40,000] bp
    let length = 60_000;
    let mut s_p1_1 = vec!['A'; length];
    let mut s_p1_2 = vec!['A'; length];
    let mut s_p2_1 = vec!['A'; length];
    let mut s_p2_2 = vec!['A'; length];
    let mut s_r = vec!['A'; length];
    let mut s_o = vec!['A'; length];

    for u in (0..length).step_by(100) {
        s_p1_1[u] = 'T';
        s_p1_2[u] = 'T';
        s_p2_1[u] = 'C';
        s_p2_2[u] = 'C';
        s_o[u] = 'G';
        s_r[u] = if u >= 20_000 && u <= 40_000 { 'C' } else { 'T' };
    }

    let fasta = format!(
        ">P1_1\n{}\n>P1_2\n{}\n>P2_1\n{}\n>P2_2\n{}\n>R\n{}\n>O\n{}\n",
        s_p1_1.into_iter().collect::<String>(),
        s_p1_2.into_iter().collect::<String>(),
        s_p2_1.into_iter().collect::<String>(),
        s_p2_2.into_iter().collect::<String>(),
        s_r.into_iter().collect::<String>(),
        s_o.into_iter().collect::<String>(),
    );

    let aln = parse_fasta(&fasta).expect("Parse fasta failed");
    assert_eq!(aln.length, 60_000);

    // Verify extraction
    let (snp_aln, snp_map) = extract_informative_snp_alignment(&aln);
    assert_eq!(snp_aln.length, 600, "Expected 600 informative SNPs out of 60,000 bp");
    assert_eq!(snp_map.len(), 600);
    assert_eq!(snp_map[0], 0);
    assert_eq!(snp_map[1], 100);

    // Scan with auto-compression (L > 50,000 triggers automatically)
    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&aln);

    assert_eq!(result.length, 60_000, "Output length must preserve full chromosomal length");
    assert!(!result.events.is_empty(), "Recombination event must be detected in compressed SNP mode");
    println!("Detected events: {:?}", result.events);
    let ev = &result.events[0];
    assert_eq!(ev.candidate_name, "R");
    assert!(ev.donor_name.starts_with("P2"), "Donor should be from P2 clade, got {}", ev.donor_name);
    // Check that coordinates are restored to physical chromosomal space
    assert!(ev.u1 >= 19_500 && ev.u1 <= 20_500, "Start breakpoint should map near 20,000 bp, got {}", ev.u1);
    assert!(ev.u2 >= 39_500 && ev.u2 <= 40_500, "End breakpoint should map near 40,000 bp, got {}", ev.u2);
    assert!(ev.tract_length >= 19_000 && ev.tract_length <= 21_000, "Tract length should be ~20,000 bp, got {}", ev.tract_length);
    assert!(ev.p_fisher < 1e-20, "Significant p-value required");
}

