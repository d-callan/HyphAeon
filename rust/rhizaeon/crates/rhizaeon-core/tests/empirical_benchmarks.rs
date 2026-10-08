use rhizaeon_core::{parse_fasta, RhizAeonEngine};
use std::path::Path;

#[test]
fn test_kal153_crf02_ag_empirical() {
    let fasta_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/empirical/data/hiv1_kal153_crf02_ag.fasta");

    assert!(fasta_path.exists(), "KAL153 alignment not found at {:?}", fasta_path);
    let fasta_content = std::fs::read_to_string(&fasta_path).expect("Failed to read KAL153");
    let alignment = parse_fasta(&fasta_content).expect("Failed to parse KAL153 FASTA");

    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&alignment);

    assert!(!result.events.is_empty(), "Expected recombination events in KAL153 CRF02_AG");
    assert_eq!(result.events[0].candidate_name, "R", "Expected R to be top recombinant candidate");
    assert_eq!(result.events[0].home_name, "B", "Expected B to be home parent");
    assert_eq!(result.events[0].donor_name, "A", "Expected A to be donor parent");
    assert!(result.events[0].p_fisher < 1e-10, "Expected highly significant Fisher p-value");
}

#[test]
fn test_human_mtdna_empirical_null() {
    let fasta_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/empirical/data/human_mtdna.fasta");

    assert!(fasta_path.exists(), "Human mtDNA alignment not found at {:?}", fasta_path);
    let fasta_content = std::fs::read_to_string(&fasta_path).expect("Failed to read mtDNA");
    let alignment = parse_fasta(&fasta_content).expect("Failed to parse mtDNA FASTA");

    let engine = RhizAeonEngine::new().with_landmarks(16);
    let result = engine.scan(&alignment);

    assert!(result.events.is_empty(), "Human mtDNA is strictly clonal; expected 0 false positives, got {}", result.events.len());
}
