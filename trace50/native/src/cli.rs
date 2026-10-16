//! TRACE-5.0 Native Standalone CLI Executable.
//!
//! Provides high-performance command-line molecular transmission network deconvolution,
//! generating analytical CSV summaries and full visualization dossier JSONs.

use clap::Parser;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use trace50::dates::parse_date_value;
use trace50::{run_trace50_with_matrix, Alignment, Trace50Config, VisualizationDossier};

#[derive(Parser, Debug)]
#[command(
    name = "trace50",
    author = "Sergei L. Kosakovsky Pond & Antigravity AI Engine",
    version,
    about = "TRACE-5.0: Dynamic Molecular Surveillance of Viral Transmission Networks"
)]
struct CliArgs {
    /// Path to input nucleotide FASTA alignment
    ///
    /// `-i`/`--input` aliases keep parity with `rhizaeon -i/--input` so
    /// muscle memory carries across the Aeon command-line tools.
    #[arg(short, long, required = true, visible_short_alias = 'i', visible_alias = "input")]
    fasta: PathBuf,

    /// Optional metadata CSV/TSV file with taxon IDs and collection dates
    #[arg(short, long)]
    meta: Option<PathBuf>,

    /// Taxon identifier column in metadata (default: id)
    #[arg(long, default_value = "id")]
    id_col: String,

    /// Collection date column in metadata (default: date)
    #[arg(long, default_value = "date")]
    date_col: String,

    /// Output directory for results
    #[arg(short, long, default_value = "trace50_results")]
    out_dir: PathBuf,

    /// Optional custom path for full visualization dossier JSON
    #[arg(long)]
    out_json: Option<PathBuf>,

    /// Retroviral substitution rate mu (substitutions/site/year)
    #[arg(long, default_value_t = 2.0e-3)]
    mu: f64,

    /// Mean intra-host coalescent depth tau_bar (years)
    #[arg(long, default_value_t = 1.0)]
    tau_bar: f64,

    /// Clinical surveillance delay timescale omega (years)
    #[arg(long, default_value_t = 2.0)]
    omega: f64,

    /// False Discovery Rate (FDR) threshold for acute transmission dyads
    #[arg(long, default_value_t = 0.05)]
    fdr: f64,

    /// Significance threshold for molecular clock adequacy
    #[arg(long, default_value_t = 0.05)]
    alpha_adequacy: f64,

    /// Maximum sampling separation window (years)
    #[arg(long, default_value_t = 2.0)]
    max_delta_t: f64,

    /// Static genetic distance threshold for comparison benchmark (e.g. 0.015)
    #[arg(long, default_value_t = 0.015)]
    threshold: f64,

    /// Basic reproductive ratio R0 within transmission networks
    #[arg(long, default_value_t = 1.5)]
    r0: f64,

    /// Minimum leaf community size for STEVE spectral deconvolution
    #[arg(long, default_value_t = 3)]
    min_size: usize,

    /// Maximum recursion depth for AutoClock
    #[arg(long, default_value_t = 8)]
    max_depth: usize,

    /// Use min-max robust Bayes Factor over biological uncertainty domain
    #[arg(long, default_value_t = false)]
    use_robust: bool,

    /// Use site-specific Information-Theoretic Transmission Odds (InfoDANNO)
    #[arg(long, default_value_t = false)]
    info_danno: bool,

    /// Consume a precomputed pairwise distance CSV instead of computing TN93
    /// internally. Expects `tn93` output: a header line then
    /// ID1,ID2,Distance rows, IDs matching the FASTA headers. Pairs absent
    /// from the file are treated as distance-unavailable, so pass the full
    /// matrix (`tn93 -t 1.0`) if the background is to be calibrated from it.
    #[arg(long)]
    distances: Option<PathBuf>,

    /// Ambiguity handling for the internal TN93 distances, as `tn93 -a`:
    /// resolve, average, skip, gapmm, or a list of ambiguity codes to resolve
    /// (all others averaged). Ignored with --distances.
    #[arg(long, default_value = "RMYSKB")]
    tn93_ambig: String,

    /// Sequence length L used by DANNO (k = d * L). Defaults to the median
    /// number of non-gap characters per sequence (as validation/run_danno.py).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    seq_len: Option<u64>,
}

/// Reads `id_col` -> decimal date from a metadata CSV/TSV. Dates are parsed
/// positionally by [`parse_date_value`]; rows whose date cannot be parsed are
/// counted (second return value) and left out, so those sequences stay undated.
fn parse_metadata_dates(
    meta_path: &Path,
    id_col: &str,
    date_col: &str,
) -> Result<(std::collections::HashMap<String, f64>, usize), String> {
    let file = File::open(meta_path).map_err(|e| format!("Failed to open metadata file: {}", e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header_line = lines
        .next()
        .ok_or_else(|| "Empty metadata file".to_string())?
        .map_err(|e| format!("Error reading metadata header: {}", e))?;

    let is_tsv = meta_path.extension().map_or(false, |ext| ext == "tsv");
    let delimiter = if is_tsv { '\t' } else { ',' };

    let headers: Vec<String> = header_line
        .split(delimiter)
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();

    let id_idx = headers.iter().position(|h| h == id_col).ok_or_else(|| {
        format!(
            "ID column '{}' not found in metadata headers: {:?}",
            id_col, headers
        )
    })?;
    let date_idx = headers.iter().position(|h| h == date_col).ok_or_else(|| {
        format!(
            "Date column '{}' not found in metadata headers: {:?}",
            date_col, headers
        )
    })?;

    let mut map = std::collections::HashMap::new();
    let mut unparsed = 0usize;

    for line_res in lines {
        let line = line_res.map_err(|e| format!("Error reading metadata row: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line
            .split(delimiter)
            .map(|s| s.trim().trim_matches('"'))
            .collect();
        if fields.len() > id_idx && fields.len() > date_idx {
            let id = fields[id_idx].to_string();
            let date_str = fields[date_idx];

            match parse_date_value(date_str) {
                Some(val) => {
                    map.insert(id, val);
                }
                None => unparsed += 1,
            }
        }
    }

    Ok((map, unparsed))
}

/// Loads a `tn93`-style pairwise distance CSV into a row-major n x n matrix
/// ordered to match `headers`.
///
/// Expected format: an optional header line, then `ID1,ID2,Distance` rows.
/// Returns (matrix, pairs_loaded, rows_with_unknown_ids).
///
/// Pairs not present in the file are left as NaN, i.e. "distance
/// unavailable", which the pipeline and `calibrate_from_data` both skip. A
/// thresholded edge list therefore yields a truncated background, so pass the
/// full matrix when the background is to be calibrated from it.
fn load_distance_csv(path: &Path, headers: &[String]) -> Result<(Vec<f64>, usize, usize), String> {
    let n = headers.len();
    let index: std::collections::HashMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();

    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut m = vec![f64::NAN; n * n];
    for i in 0..n {
        m[i * n + i] = 0.0;
    }

    let mut loaded = 0usize;
    let mut unknown = 0usize;
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line
            .split(',')
            .map(|s| s.trim().trim_matches('"'))
            .collect();
        if f.len() < 3 {
            continue;
        }
        // skip a header row
        if lineno == 0 && f[2].parse::<f64>().is_err() {
            continue;
        }
        let d: f64 = match f[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        match (index.get(f[0]), index.get(f[1])) {
            (Some(&i), Some(&j)) => {
                m[i * n + j] = d;
                m[j * n + i] = d;
                loaded += 1;
            }
            _ => unknown += 1,
        }
    }
    if loaded == 0 {
        return Err(format!(
            "no usable rows: none of the ID pairs matched the {} FASTA headers",
            n
        ));
    }
    Ok((m, loaded, unknown))
}

/// Entry point forwarded by the `trace50` Python console script. `argv`
/// includes the program name at index 0 (clap `parse_from` semantics);
/// clap reports --help/--version/usage errors via `try_parse_from` so a
/// programmatic caller never has its process exited underneath it; the
/// console script still sees the same exit codes (0 help, 2 usage) as the
/// standalone binary. Returns a process exit code.
pub fn cli_main(argv: Vec<String>) -> i32 {
    // Intercept --version/-V ahead of clap so the printed version is the
    // PEP 440 form `pip show trace50` reports (Cargo semver writes
    // `5.0.0-beta.1`; PyPI normalizes it to `5.0.0b1`). Same
    // short-circuit semantics as clap's built-in version flag.
    if argv.iter().skip(1).any(|a| a == "--version" || a == "-V") {
        println!("trace50 {}", pep440(env!("CARGO_PKG_VERSION")));
        return 0;
    }
    let args = match CliArgs::try_parse_from(&argv) {
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
/// reports): `5.0.0-beta.1` -> `5.0.0b1`, `5.0.0-rc.1` -> `5.0.0rc1`.
/// Stable releases and unrecognized prerelease tags pass through
/// unchanged.
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

fn run(args: CliArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!(" TRACE-5.0: Molecular Transmission Surveillance Engine (Rust Native)");
    println!("================================================================================");

    // Read input FASTA
    println!("[*] Loading FASTA alignment from: {}", args.fasta.display());
    let fasta_content = fs::read_to_string(&args.fasta)
        .map_err(|e| format!("[!] Error reading FASTA file: {}", e))?;

    let mut alignment = Alignment::from_fasta_str(&fasta_content)
        .map_err(|e| format!("[!] Error parsing alignment: {}", e))?;

    let seq_len = args
        .seq_len
        .map(|v| v as usize)
        .unwrap_or(alignment.effective_len);
    println!(
        "[*] Ingested {} sequences: alignment width {}, median non-gap length {} -> L = {}{}.",
        alignment.num_sequences(),
        alignment.seq_len,
        alignment.effective_len,
        seq_len,
        if args.seq_len.is_some() {
            " (--seq-len)"
        } else {
            ""
        }
    );

    // Merge metadata if provided
    if let Some(meta_path) = &args.meta {
        println!("[*] Reading metadata from: {}", meta_path.display());
        match parse_metadata_dates(meta_path, &args.id_col, &args.date_col) {
            Ok((date_map, unparsed)) => {
                if unparsed > 0 {
                    eprintln!(
                        "[!] WARNING: {} metadata rows have a date that could not be parsed (accepted: YYYY-MM-DD, YYYY-MM, YYYY, DD-Mon-YYYY, Mon-YYYY, decimal year); they are ignored.",
                        unparsed
                    );
                }
                let mut matched = 0usize;
                for (i, h) in alignment.headers.iter().enumerate() {
                    let clean_id = h.trim();
                    let first_word = clean_id.split_whitespace().next().unwrap_or("");
                    let token = first_word
                        .split(|c| c == '|' || c == '_' || c == '.')
                        .next()
                        .unwrap_or("");

                    if let Some(&d) = date_map.get(clean_id) {
                        alignment.dates[i] = d;
                        matched += 1;
                    } else if let Some(&d) = date_map.get(first_word) {
                        alignment.dates[i] = d;
                        matched += 1;
                    } else if let Some(&d) = date_map.get(token) {
                        alignment.dates[i] = d;
                        matched += 1;
                    }
                }
                println!(
                    "[*] Temporal calibration: matched {}/{} sequences to collection dates.",
                    matched,
                    alignment.num_sequences()
                );
            }
            Err(e) => {
                eprintln!(
                    "[!] Warning: metadata parsing failed: {}. Continuing with header dates.",
                    e
                );
            }
        }
    }

    let num_undated = alignment.dates.iter().filter(|d| !d.is_finite()).count();
    if num_undated == alignment.num_sequences() {
        return Err("[!] ERROR: no sequence has a parseable collection date (from --meta or the FASTA headers). DANNO needs a sampling interval for every pair and cannot run.".into());
    } else if num_undated > 0 {
        eprintln!(
            "================================================================================"
        );
        eprintln!(
            "[!] WARNING: {} of {} sequences have NO collection date. They are EXCLUDED from",
            num_undated,
            alignment.num_sequences()
        );
        eprintln!(
            "[!] DANNO, the static network, AutoClock and CHIN (not imputed); as in run_danno.py."
        );
        eprintln!(
            "================================================================================"
        );
    } else {
        println!(
            "[*] Collection dates: all {} sequences dated.",
            alignment.num_sequences()
        );
    }

    let config = Trace50Config {
        mu: args.mu,
        tau_bar: args.tau_bar,
        omega: args.omega,
        t_span: 20.0,
        alpha_adequacy: args.alpha_adequacy,
        fdr_threshold: args.fdr,
        prior_odds: 0.01,
        max_delta_t: args.max_delta_t,
        r0: args.r0,
        min_cluster_size: args.min_size,
        max_recursion_depth: args.max_depth,
        use_robust_bf: args.use_robust,
        static_distance_threshold: args.threshold,
        use_info_danno: args.info_danno,
        tn93_ambiguity: args.tn93_ambig.clone(),
        seq_len: Some(seq_len),
    };

    // Optional precomputed distance matrix (e.g. from the `tn93` binary)
    let supplied_matrix = match &args.distances {
        None => None,
        Some(path) => match load_distance_csv(path, &alignment.headers) {
            Ok((m, found, missing)) => {
                println!(
                    "[*] Loaded {} pairwise distances from {} ({} of {} pairs present{})",
                    found,
                    path.display(),
                    found,
                    alignment.headers.len() * (alignment.headers.len() - 1) / 2,
                    if missing > 0 {
                        format!(", {} unknown IDs skipped", missing)
                    } else {
                        String::new()
                    }
                );
                Some(m)
            }
            Err(e) => {
                return Err(
                    format!("[!] Failed to load --distances {}: {}", path.display(), e).into(),
                );
            }
        },
    };

    if supplied_matrix.is_none() {
        println!(
            "[*] Internal TN93 distances (tn93 -t 1.0 -l {} -a {}).",
            trace50::tn93::TN93_MIN_OVERLAP,
            args.tn93_ambig
        );
    }

    println!("[*] Executing end-to-end TRACE-5.0 pipeline...");
    let dossier = run_trace50_with_matrix(&alignment, &config, supplied_matrix)
        .map_err(|e| format!("[!] Pipeline execution failed: {}", e))?;

    // Ensure output directory exists
    fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("[!] Failed to create output directory: {}", e))?;

    // 1. Write full visualization JSON dossier
    let json_path = args
        .out_json
        .unwrap_or_else(|| args.out_dir.join("trace50_visualization_dossier.json"));
    println!(
        "[+] Serializing full visualization dossier to: {}",
        json_path.display()
    );
    match serde_json::to_string_pretty(&dossier) {
        Ok(json_str) => {
            if let Err(e) = fs::write(&json_path, json_str) {
                eprintln!("[!] Error writing JSON dossier: {}", e);
            }
        }
        Err(e) => {
            eprintln!("[!] Serialization error: {}", e);
        }
    }

    // 2. Write CSV outputs
    write_csv_outputs(&dossier, &args.out_dir)
        .map_err(|e| format!("[!] Failed writing CSV outputs: {}", e))?;

    // Print summary KPIs
    println!("\n================================================================================");
    println!(" OPERATIONAL SURVEILLANCE SCORECARD (KPIs)");
    println!("================================================================================");
    println!(
        " Total Sequences Evaluated:            {}",
        dossier.kpi.total_sequences
    );
    println!(
        " Total Pairwise Comparisons:           {}",
        dossier.kpi.total_pairwise_comparisons
    );
    println!(
        " Candidate Dyads Screened:             {}",
        dossier.kpi.candidate_pairs_screened
    );
    println!(
        " Clock Violations Purged (Wo Fat):     {}",
        dossier.kpi.physical_clock_violations_purged
    );
    println!(
        " Supported Transmission Edges:         {} (FDR q <= {})",
        dossier.kpi.supported_transmission_edges, config.fdr_threshold
    );
    println!("--------------------------------------------------------------------------------");
    println!(
        " Static Distance Benchmark (d <= {:.1}%):",
        config.static_distance_threshold * 100.0
    );
    println!(
        "   - Total Clusters:                   {}",
        dossier.kpi.static_network.total_clusters
    );
    println!(
        "   - Clustered Individuals:            {}",
        dossier.kpi.static_network.clustered_individuals
    );
    println!(
        "   - Giant Component Size:             {} patients",
        dossier.kpi.static_network.max_cluster_size
    );
    println!(" DANNO Calibrated Transmission Network:");
    println!(
        "   - Total Clusters:                   {}",
        dossier.kpi.danno_network.total_clusters
    );
    println!(
        "   - Clustered Individuals:            {}",
        dossier.kpi.danno_network.clustered_individuals
    );
    println!(
        "   - Giant Component Size:             {} patients",
        dossier.kpi.danno_network.max_cluster_size
    );
    match dossier.kpi.giant_component_reduction_pct {
        Some(pct) => println!("   - Giant Component Reduction:        {:.1}%", pct),
        None => println!("   - Giant Component Reduction:        n/a (no supported DANNO cluster)"),
    }
    println!("--------------------------------------------------------------------------------");
    println!(" CHIN Macro-Scale Epidemiological Scaling:");
    match (
        dossier.kpi.inferred_active_transmitting_pool,
        dossier.kpi.inferred_active_pool_ci,
    ) {
        (Some(n_act), Some(ci)) => println!(
            "   - Active Transmitting Pool (N_act): {:.0} [CI: {:.0} - {:.0}]",
            n_act, ci[0], ci[1]
        ),
        _ => println!(
            "   - Active Transmitting Pool (N_act): {}",
            dossier.kpi.inferred_active_pool_status
        ),
    }
    match dossier.kpi.inferred_surveillance_sampling_fraction_pct {
        Some(pct) => println!("   - Surveillance Sampling Fraction:   {:.2}%", pct),
        None => {
            println!("   - Surveillance Sampling Fraction:   not estimable (N_act not estimable)")
        }
    }
    println!(
        "   - Borel Branching Decay (R^2):      {:.4}",
        dossier.macro_scaling.borel_decay.r_squared
    );
    println!("--------------------------------------------------------------------------------");
    println!(
        " Resolved STEVE Communities:           {}",
        dossier.communities.len()
    );
    println!(
        " Execution Time:                       {:.3} seconds",
        dossier.metadata.execution_time_seconds
    );
    println!(
        " Output Dossier:                       {}",
        json_path.display()
    );
    println!("================================================================================\n");
    Ok(())
}

/// Write the three tabular outputs. io errors propagate to `run` — this
/// used to panic via expect()/unwrap(), which under a Python host would
/// unwind into the interpreter rather than exit cleanly.
fn write_csv_outputs(dossier: &VisualizationDossier, out_dir: &Path) -> std::io::Result<()> {
    let dyads_csv = out_dir.join("acute_transmission_dyads.csv");
    let mut dyads_file = File::create(&dyads_csv)?;
    writeln!(dyads_file, "source,target,distance,delta_t_years,k_substitutions,bayes_factor,pep,q_value,p_adequacy,status")?;
    for e in &dossier.edges {
        writeln!(
            dyads_file,
            "{},{},{:.5},{:.3},{:.2},{:.2},{:.6},{:.6},{:.5},{}",
            e.source,
            e.target,
            e.distance,
            e.delta_t_years,
            e.k_substitutions,
            e.bayes_factor,
            e.pep,
            e.q_value,
            e.p_adequacy,
            e.status
        )?;
    }

    let communities_csv = out_dir.join("autoclock_communities.csv");
    let mut comm_file = File::create(&communities_csv)?;
    writeln!(comm_file, "community_id,size,n_eff,clock_velocity_mu,se_mu,r_squared,tmrca,ci_tmrca_low,ci_tmrca_high,fieller_status,is_active,operational_tier")?;
    for c in &dossier.communities {
        writeln!(
            comm_file,
            "{},{},{:.2},{:.6},{:.6},{:.4},{:.2},{:.2},{:.2},{},{},\"{}\"",
            c.community_id,
            c.size,
            c.n_eff,
            c.mu,
            c.se_mu,
            c.r_squared,
            c.tmrca,
            c.ci_mrca[0],
            c.ci_mrca[1],
            c.fieller_status,
            c.is_active_outbreak,
            c.operational_tier
        )?;
    }

    let nodes_csv = out_dir.join("patient_surveillance_assignments.csv");
    let mut nodes_file = File::create(&nodes_csv)?;
    writeln!(nodes_file, "taxon_id,date,degree_static,degree_supported,cluster_static,cluster_supported,community_id,operational_tier")?;
    for n in &dossier.nodes {
        let date = if n.date.is_finite() {
            format!("{:.2}", n.date)
        } else {
            String::new()
        };
        writeln!(
            nodes_file,
            "{},{},{},{},{},{},{},\"{}\"",
            n.id,
            date,
            n.degree_static,
            n.degree_supported,
            n.cluster_id_static,
            n.cluster_id_supported,
            n.community_id,
            n.operational_tier
        )?;
    }
    Ok(())
}
