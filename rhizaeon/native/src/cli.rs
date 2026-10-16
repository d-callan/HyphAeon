use clap::Parser;
use rhizaeon::{
    generate_standalone_html, generate_visualization_dossier, parse_fasta, RhizAeonEngine,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "rhizaeon")]
#[command(author = "Sergei L. Kosakovsky Pond & Antigravity AI Engine")]
#[command(version)]
#[command(about = "Ultra-fast Physical Force Field (PA-FF v2.1) Recombination Engine", long_about = None)]
struct Args {
    /// Path to input alignment FASTA file
    ///
    /// `-f`/`--fasta` aliases keep parity with `trace50 -f/--fasta` so
    /// muscle memory carries across the Aeon command-line tools.
    #[arg(short, long, visible_short_alias = 'f', visible_alias = "fasta")]
    input: PathBuf,

    /// Optional path to write structured scan summary JSON
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optional path to write rich visualization JSON dossier
    ///
    /// Long flag only: a lowercase `-v` would squat on the conventional
    /// verbose flag and invite misuse (`rhizaeon -v` expecting version).
    #[arg(long)]
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

    /// Tier 2 filter stringency multiplier (trajectory & donor attribution; < 1.0 relaxes, > 1.0 increases stringency)
    #[arg(long, default_value_t = 1.0)]
    tier2_stringency: f64,

    /// Tier 3 filter stringency multiplier (sequence & topological due diligence; < 1.0 relaxes, > 1.0 increases stringency)
    #[arg(long, default_value_t = 1.0)]
    tier3_stringency: f64,

    /// Force compression to parsimony-informative SNPs (auto-triggers for L > 30kb, N >= 48, or N*L > 600k)
    #[arg(long, conflicts_with = "no_compress_snps")]
    compress_snps: bool,

    /// Force full physical sequence scanning (disables automatic SNP compression)
    #[arg(long, conflicts_with = "compress_snps")]
    no_compress_snps: bool,
}

/// Entry point forwarded by the `rhizaeon` Python console script. `argv`
/// includes the program name at index 0 (clap `parse_from` semantics); clap
/// reports --help/--version/usage errors via `try_parse_from` so a
/// programmatic caller never has its process exited underneath it. The
/// console script still sees the same exit codes (0 help, 2 usage) as the
/// standalone binary. Returns a process exit code.
pub fn cli_main(argv: Vec<String>) -> i32 {
    // Intercept --version/-V ahead of clap so the printed version is the
    // PEP 440 form `pip show rhizaeon` reports (Cargo semver writes
    // `0.1.0-beta.1`; PyPI normalizes it to `0.1.0b1`). Same
    // short-circuit semantics as clap's built-in version flag.
    if argv.iter().skip(1).any(|a| a == "--version" || a == "-V") {
        println!("rhizaeon {}", pep440(env!("CARGO_PKG_VERSION")));
        return 0;
    }
    let args = match Args::try_parse_from(&argv) {
        Ok(a) => a,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    match run(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Render a cargo semver string in PEP 440 form (the version `pip show`
/// reports): `0.1.0-beta.1` -> `0.1.0b1`, `0.1.0-alpha.2` -> `0.1.0a2`,
/// `0.1.0-rc.1` -> `0.1.0rc1`, `0.1.0-dev.3` -> `0.1.0.dev3`. Stable
/// releases and unrecognized prerelease tags pass through unchanged.
fn pep440(cargo_version: &str) -> String {
    let (version, suffix) = match cargo_version.split_once('-') {
        Some(pair) => pair,
        None => return cargo_version.to_string(),
    };
    let normalized = match suffix.split_once('.') {
        Some(("alpha", n)) => format!("a{}", n),
        Some(("beta", n)) => format!("b{}", n),
        Some(("rc", n)) => format!("rc{}", n),
        Some(("dev", n)) => format!(".dev{}", n),
        _ => format!("-{}", suffix),
    };
    format!("{}{}", version, normalized)
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  RHIZAEON PA-FF v2.1 (Native Rust Engine)");
    println!("  Phylo-Attention Force Field Recombination Scanner");
    println!("================================================================================");
    println!("  Input alignment: {:?}", args.input);
    println!("  Landmarks (K):   {}", args.landmarks);
    println!(
        "  Significance:    alpha = {} (Tier 2 = {:.2}x, Tier 3 = {:.2}x)",
        args.alpha, args.tier2_stringency, args.tier3_stringency
    );
    if args.compress_snps {
        println!("  SNP Compression: FORCED (Informative SNPs only)");
    } else if args.no_compress_snps {
        println!("  SNP Compression: DISABLED (Full nucleotide sequence)");
    } else {
        println!("  SNP Compression: AUTO (Trigger if L > 30kb, N >= 48, or N*L > 600k)");
    }
    println!("--------------------------------------------------------------------------------");

    let fasta_content = fs::read_to_string(&args.input)?;
    let parse_start = Instant::now();
    let aln = parse_fasta(&fasta_content)?;
    let parse_dur = parse_start.elapsed();

    println!(
        "  Alignment loaded: {} taxa, {} bp (parsed in {:.2?})",
        aln.num_taxa, aln.length, parse_dur
    );

    let mut engine = RhizAeonEngine::new()
        .with_landmarks(args.landmarks)
        .with_alpha(args.alpha)
        .with_poisson_floor(args.poisson_floor)
        .with_tier2_stringency(args.tier2_stringency)
        .with_tier3_stringency(args.tier3_stringency);

    if args.compress_snps {
        engine = engine.with_compress_snps(true);
    } else if args.no_compress_snps {
        engine = engine.with_compress_snps(false);
    }

    let scan_start = Instant::now();
    let mut result = engine.scan(&aln);
    let scan_dur = scan_start.elapsed();
    result.run_time_ms = scan_dur.as_secs_f64() * 1000.0;

    println!(
        "  Scan completed in {:.2?} ({:.1} ms)",
        scan_dur, result.run_time_ms
    );
    println!("  Informative sites: {}", result.informative_sites);
    println!(
        "  Candidate taxa screened: {}",
        result.screening.candidate_recombinant_indices.len()
    );
    println!("  Verified events found:   {}", result.events.len());
    println!("--------------------------------------------------------------------------------");

    if result.events.is_empty() {
        println!("  STATUS: Certified Non-Recombinant (H0 Null)");
    } else {
        println!(
            "  STATUS: Verified Recombination Events (|E| = {})",
            result.events.len()
        );
        println!();
        println!(
            "{:<20} {:<15} {:<15} {:<12} {:<10} {:<12} {:<10}",
            "Recombinant",
            "Home Parent",
            "Donor Parent",
            "Tract (bp)",
            "Length",
            "p-Fisher",
            "Z-Score"
        );
        println!("{}", "-".repeat(95));
        for ev in &result.events {
            let tract_str = format!("{}-{}", ev.u1, ev.u2);
            let donor_str = if ev.is_ghost_donor {
                "Ghost (Unsampled)"
            } else {
                &ev.donor_name
            };
            println!(
                "{:<20} {:<15} {:<15} {:<12} {:<10} {:<12.2e} {:<10.2}",
                ev.candidate_name,
                ev.home_name,
                donor_str,
                tract_str,
                ev.tract_length,
                ev.p_fisher,
                ev.z_phys
            );
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
