use trace50_core::{run_trace50, Alignment, Trace50Config};

#[test]
fn test_end_to_end_pipeline() {
    let fasta = r#">seq1_2015.0
ACGTACGTACGTACGTACGTACGTACGTACGT
>seq2_2015.1
ACGTACGTACGTACGTACGTACGTACGTACGT
>seq3_2015.2
ACGTACGTACGTACGTACGTACGTACGTACGA
>seq4_2018.0
TGCATGCATGCATGCATGCATGCATGCATGCA
>seq5_2018.5
TGCATGCATGCATGCATGCATGCATGCATGCT
"#;

    let alignment = Alignment::from_fasta_str(fasta).expect("Failed to parse FASTA");
    assert_eq!(alignment.num_sequences(), 5);

    let mut config = Trace50Config::default();
    config.min_cluster_size = 2;
    config.static_distance_threshold = 0.05;

    let dossier = run_trace50(&alignment, &config).expect("Pipeline execution failed");

    assert_eq!(dossier.metadata.num_sequences, 5);
    assert_eq!(dossier.nodes.len(), 5);
    assert!(dossier.kpi.total_pairwise_comparisons == 10);
    assert!(dossier.macro_scaling.composite_parameter_theta >= 0.0);

    assert_eq!(dossier.metadata.parameters.static_distance_threshold, 0.05);
    for e in &dossier.edges {
        if e.distance <= 0.05 {
            assert!(e.is_static);
        }
    }

    // Verify JSON serialization
    let json_str = serde_json::to_string_pretty(&dossier).expect("Serialization failed");
    assert!(json_str.contains("schema_version"));
    assert!(json_str.contains("TRACE-5.0 (Rust Kernel)"));
    assert!(json_str.contains("static_distance_threshold"));
    assert!(json_str.contains("macro_scaling"));
    assert!(json_str.contains("phase_space_points"));
}

#[test]
fn test_end_to_end_pipeline_with_info_danno() {
    let fasta = r#">seq1_2015.0
ACGTACGTACGTACGTACGTACGTACGTACGT
>seq2_2015.1
ACGTACGTACGTACGTACGTACGTACGTACGT
>seq3_2015.2
ACGTACGTACGTACGTACGTACGTACGTACGA
>seq4_2018.0
TGCATGCATGCATGCATGCATGCATGCATGCA
>seq5_2018.5
TGCATGCATGCATGCATGCATGCATGCATGCT
"#;

    let alignment = Alignment::from_fasta_str(fasta).expect("Failed to parse FASTA");
    let mut config = Trace50Config::default();
    config.use_info_danno = true;
    config.min_cluster_size = 2;
    config.static_distance_threshold = 0.05;

    let dossier = run_trace50(&alignment, &config).expect("Pipeline execution with InfoDANNO failed");
    assert_eq!(dossier.metadata.num_sequences, 5);
    assert_eq!(dossier.nodes.len(), 5);
}
