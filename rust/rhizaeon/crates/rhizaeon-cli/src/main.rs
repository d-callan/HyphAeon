use clap::Parser;
use rhizaeon_core::{
    generate_standalone_html, generate_visualization_dossier, parse_fasta, RhizAeonEngine,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "rhizaeon")]
#[command(author = "Sergei L. Kosakovsky Pond & Antigravity AI Engine")]
#[command(version = "0.1.0")]
#[command(about = "Ultra-fast Physical Force Field (PA-FF v2.1) Recombination Engine", long_about = None)]
struct Args {
    /// Path to input alignment FASTA file
    #[arg(short, long)]
    input: PathBuf,

    /// Optional path to write structured scan summary JSON
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optional path to write rich visualization JSON dossier
    #[arg(short, long)]
    viz: Option<PathBuf>,

    /// Optional path to write standalone interactive HTML dashboard
    #[arg(long)]
    html: Option<PathBuf>,

    /// Optional custom title for dashboard and visualization reports
    #[arg(short, long)]
    title: Option<String>,

    /// Target number of landmarks K on Delta^K
    #[arg(short, long, default_value_t = 16)]
    landmarks: usize,

    /// Family-wise significance alpha
    #[arg(short, long, default_value_t = 0.05)]
    alpha: f64,

    /// Minimum supporting informative sites
    #[arg(short, long, default_value_t = 3)]
    poisson_floor: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("================================================================================");
    println!("  RHIZAEON PA-FF v2.1 (Native Rust Engine)");
    println!("  Phylo-Attention Force Field Recombination Scanner");
    println!("================================================================================");
    println!("  Input alignment: {:?}", args.input);
    println!("  Landmarks (K):   {}", args.landmarks);
    println!("  Significance:    alpha = {}", args.alpha);
    println!("--------------------------------------------------------------------------------");

    let fasta_content = fs::read_to_string(&args.input)?;
    let parse_start = Instant::now();
    let aln = parse_fasta(&fasta_content)?;
    let parse_dur = parse_start.elapsed();

    println!("  Alignment loaded: {} taxa, {} bp (parsed in {:.2?})", aln.num_taxa, aln.length, parse_dur);

    let engine = RhizAeonEngine::new()
        .with_landmarks(args.landmarks);

    let scan_start = Instant::now();
    let mut result = engine.scan(&aln);
    let scan_dur = scan_start.elapsed();
    result.run_time_ms = scan_dur.as_secs_f64() * 1000.0;

    println!("  Scan completed in {:.2?} ({:.1} ms)", scan_dur, result.run_time_ms);
    println!("  Informative sites: {}", result.informative_sites);
    println!("  Candidate taxa screened: {}", result.screening.candidate_recombinant_indices.len());
    println!("  Verified events found:   {}", result.events.len());
    println!("--------------------------------------------------------------------------------");

    if result.events.is_empty() {
        println!("  STATUS: Certified Non-Recombinant (H0 Null)");
    } else {
        println!("  STATUS: Verified Recombination Events (|E| = {})", result.events.len());
        println!();
        println!("{:<20} {:<15} {:<15} {:<12} {:<10} {:<12} {:<10}",
                 "Recombinant", "Home Parent", "Donor Parent", "Tract (bp)", "Length", "p-Fisher", "Z-Score");
        println!("{}", "-".repeat(95));
        for ev in &result.events {
            let tract_str = format!("{}-{}", ev.u1, ev.u2);
            let donor_str = if ev.is_ghost_donor { "Ghost (Unsampled)" } else { &ev.donor_name };
            println!("{:<20} {:<15} {:<15} {:<12} {:<10} {:<12.2e} {:<10.2}",
                     ev.candidate_name, ev.home_name, donor_str, tract_str, ev.tract_length, ev.p_fisher, ev.z_phys);
        }
    }
    println!("================================================================================");

    if let Some(out_path) = args.output {
        let json_str = serde_json::to_string_pretty(&result)?;
        fs::write(&out_path, json_str)?;
        println!("  Summary JSON written to: {:?}", out_path);
    }

    if args.viz.is_some() || args.html.is_some() {
        let dossier_title = args.title.as_deref().unwrap_or_else(|| {
            args.input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("RhizAeon Scan")
        });
        let dossier = generate_visualization_dossier(&aln, &result, Some(dossier_title));

        if let Some(viz_path) = args.viz {
            let viz_json = serde_json::to_string_pretty(&dossier)?;
            fs::write(&viz_path, viz_json)?;
            println!("  Visualization Dossier JSON written to: {:?}", viz_path);
        }

        if let Some(html_path) = args.html {
            let html_content = generate_standalone_html(&dossier)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            fs::write(&html_path, html_content)?;
            println!("  Standalone HTML Dashboard written to:  {:?}", html_path);
        }
    }

    Ok(())
}
