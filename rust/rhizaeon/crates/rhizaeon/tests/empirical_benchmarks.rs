use rhizaeon::{parse_fasta, RhizAeonEngine};
use std::path::Path;

#[test]
fn test_kal153_crf02_ag_empirical() {
    let fasta_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/empirical/data/hiv1_kal153_crf02_ag.fasta");

    assert!(
        fasta_path.exists(),
        "KAL153 alignment not found at {:?}",
        fasta_path
    );
    let fasta_content = std::fs::read_to_string(&fasta_path).expect("Failed to read KAL153");
    let alignment = parse_fasta(&fasta_content).expect("Failed to parse KAL153 FASTA");

    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&alignment);

    assert!(
        !result.events.is_empty(),
        "Expected recombination events in KAL153 CRF02_AG"
    );
    assert_eq!(
        result.events[0].candidate_name, "R",
        "Expected R to be top recombinant candidate"
    );
    assert_eq!(
        result.events[0].home_name, "B",
        "Expected B to be home parent"
    );
    assert_eq!(
        result.events[0].donor_name, "A",
        "Expected A to be donor parent"
    );
    assert!(
        result.events[0].p_fisher < 1e-10,
        "Expected highly significant Fisher p-value"
    );
}

#[test]
fn test_human_mtdna_empirical_null() {
    let fasta_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/empirical/data/human_mtdna.fasta");

    assert!(
        fasta_path.exists(),
        "Human mtDNA alignment not found at {:?}",
        fasta_path
    );
    let fasta_content = std::fs::read_to_string(&fasta_path).expect("Failed to read mtDNA");
    let alignment = parse_fasta(&fasta_content).expect("Failed to parse mtDNA FASTA");

    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&alignment);

    assert!(
        result.events.is_empty(),
        "Human mtDNA is strictly clonal; expected 0 false positives, got {}",
        result.events.len()
    );
}

#[test]
fn test_pvy_example1_empirical() {
    let fasta_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/empirical/data/pvy_example1.fasta");

    assert!(
        fasta_path.exists(),
        "PVY example1 alignment not found at {:?}",
        fasta_path
    );
    let fasta_content = std::fs::read_to_string(&fasta_path).expect("Failed to read PVY FASTA");
    let alignment = parse_fasta(&fasta_content).expect("Failed to parse PVY FASTA");

    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&alignment);

    assert_eq!(
        result.events.len(),
        6,
        "Expected 6 distinct ancestral recombination events in PVY example1, got {}: {:#?}",
        result.events.len(),
        result.events
    );

    // Zero self-donation invariant: no event may list its own donor or home in `isolates`
    for ev in &result.events {
        assert!(
            !ev.isolates.contains(&ev.donor_name),
            "Event #{} lists donor {} in recombinant isolates {:?}",
            ev.event_id,
            ev.donor_name,
            ev.isolates
        );
        assert!(
            !ev.isolates.contains(&ev.home_name),
            "Event #{} lists home {} in recombinant isolates {:?}",
            ev.event_id,
            ev.home_name,
            ev.isolates
        );
    }

    // Event #1: 5' crossover [1..2258] shared by all 5 isolates I, J, N, O, W (H=Q, D=T)
    let ev1 = &result.events[0];
    assert_eq!(ev1.u1, 1);
    assert!((2240..=2270).contains(&ev1.u2));
    assert_eq!(ev1.home_name, "Q");
    assert_eq!(ev1.donor_name, "T");
    assert!(ev1.is_crossover);
    assert_eq!(ev1.isolates, vec!["I", "J", "N", "O", "W"]);

    // 3' U-donor events in N (~5709-8453), W (~5705-9099), and J (~5643-6587) must remain distinct
    let ev_n = result
        .events
        .iter()
        .find(|e| e.isolates == vec!["N"])
        .expect("Expected distinct event for N");
    assert_eq!(ev_n.home_name, "Q");
    assert_eq!(ev_n.donor_name, "U");
    assert!((5690..=5730).contains(&ev_n.u1) && (8430..=8470).contains(&ev_n.u2));

    let ev_w = result
        .events
        .iter()
        .find(|e| e.isolates == vec!["W"])
        .expect("Expected distinct extended event for W");
    assert_eq!(ev_w.home_name, "Q");
    assert_eq!(ev_w.donor_name, "U");
    assert!((5690..=5730).contains(&ev_w.u1) && (9070..=9120).contains(&ev_w.u2));

    let ev_j = result
        .events
        .iter()
        .find(|e| e.isolates == vec!["J"])
        .expect("Expected distinct overprinted 5' event for J");
    assert_eq!(ev_j.home_name, "Q");
    assert_eq!(ev_j.donor_name, "U");
    assert!((5620..=5670).contains(&ev_j.u1) && (6560..=6610).contains(&ev_j.u2));

    // H cassette (~7770-8210, H=Y, D=T, is_crossover=false)
    let ev_h = result
        .events
        .iter()
        .find(|e| e.isolates == vec!["H"])
        .expect("Expected independent cassette event for H");
    assert!(
        !ev_h.is_crossover,
        "H must be a cassette, not a single crossover"
    );
    assert!((7750..=7800).contains(&ev_h.u1) && (8190..=8240).contains(&ev_h.u2));

    // K cassette (~7849-8228, H=T, D=L, isolates=["K"] only)
    let ev_k = result
        .events
        .iter()
        .find(|e| e.isolates == vec!["K"])
        .expect("Expected independent cassette event for K alone");
    assert!(!ev_k.is_crossover);
    assert_eq!(ev_k.home_name, "T");
    assert_eq!(ev_k.donor_name, "L");
    assert!((7830..=7870).contains(&ev_k.u1) && (8210..=8250).contains(&ev_k.u2));
}
