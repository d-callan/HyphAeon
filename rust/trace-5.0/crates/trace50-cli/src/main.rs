//! TRACE-5.0 Native Standalone CLI Executable.
//!
//! Provides high-performance command-line molecular transmission network deconvolution,
//! generating analytical CSV summaries and full visualization dossier JSONs.

use clap::Parser;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use trace50_core::{run_trace50, Alignment, Trace50Config};

#[derive(Parser, Debug)]
#[command(
    name = "trace50",
    author = "Sergei L. Kosakovsky Pond & Antigravity AI Engine",
    version = "5.0.0",
    about = "TRACE-5.0: Dynamic Molecular Surveillance of Viral Transmission Networks"
)]
struct CliArgs {
    /// Path to input nucleotide FASTA alignment
    #[arg(short, long, required = true)]
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
}

fn parse_date_string(date_str: &str) -> Option<f64> {
    let s = date_str.trim().trim_matches('"');
    if let Ok(val) = s.parse::<f64>() {
        if (1950.0..=2035.0).contains(&val) {
            return Some(val);
        }
    }

    // Split by common date separators: '-', '/', '.', ' '
    let parts: Vec<&str> = s.split(|c| c == '-' || c == '/' || c == '.' || c == ' ').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return None;
    }

    let months = [
        ("jan", 1.0), ("feb", 2.0), ("mar", 3.0), ("apr", 4.0),
        ("may", 5.0), ("jun", 6.0), ("jul", 7.0), ("aug", 8.0),
        ("sep", 9.0), ("oct", 10.0), ("nov", 11.0), ("dec", 12.0)
    ];

    let mut year = None;
    let mut month = None;
    let mut day = None;

    for part in &parts {
        let p_lower = part.to_ascii_lowercase();
        // Check for month name
        if let Some(&(_, m_num)) = months.iter().find(|(name, _)| p_lower.starts_with(name)) {
            month = Some(m_num);
            continue;
        }

        if let Ok(num) = part.parse::<f64>() {
            if num >= 1950.0 && num <= 2035.0 {
                year = Some(num);
            } else if num >= 1.0 && num <= 31.0 {
                if month.is_none() && num <= 12.0 && parts.len() == 3 && parts[0] == *part {
                    // Possible month in MM/DD/YYYY
                    month = Some(num);
                } else if day.is_none() {
                    day = Some(num);
                } else if month.is_none() && num <= 12.0 {
                    month = Some(num);
                }
            }
        }
    }

    if let Some(y) = year {
        let m = month.unwrap_or(6.5);
        let d = day.unwrap_or(15.0);
        let frac = (m - 1.0 + (d - 1.0) / 30.0) / 12.0;
        return Some(y + frac);
    }

    None
}

fn parse_metadata_dates(meta_path: &Path, id_col: &str, date_col: &str) -> Result<std::collections::HashMap<String, f64>, String> {
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

    let id_idx = headers.iter().position(|h| h == id_col)
        .ok_or_else(|| format!("ID column '{}' not found in metadata headers: {:?}", id_col, headers))?;
    let date_idx = headers.iter().position(|h| h == date_col)
        .ok_or_else(|| format!("Date column '{}' not found in metadata headers: {:?}", date_col, headers))?;

    let mut map = std::collections::HashMap::new();

    for line_res in lines {
        let line = line_res.map_err(|e| format!("Error reading metadata row: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(delimiter).map(|s| s.trim().trim_matches('"')).collect();
        if fields.len() > id_idx && fields.len() > date_idx {
            let id = fields[id_idx].to_string();
            let date_str = fields[date_idx];

            if let Some(val) = parse_date_string(date_str) {
                map.insert(id, val);
            }
        }
    }

    Ok(map)
}

fn main() {
    let args = CliArgs::parse();

    println!("================================================================================");
    println!(" TRACE-5.0: Molecular Transmission Surveillance Engine (Rust Native)");
    println!("================================================================================");

    // Read input FASTA
    println!("[*] Loading FASTA alignment from: {}", args.fasta.display());
    let fasta_content = match fs::read_to_string(&args.fasta) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[!] Error reading FASTA file: {}", e);
            std::process::exit(1);
        }
    };

    let mut alignment = match Alignment::from_fasta_str(&fasta_content) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[!] Error parsing alignment: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "[*] Ingested {} sequences across {} nucleotide sites.",
        alignment.num_sequences(),
        alignment.seq_len
    );

    // Merge metadata if provided
    if let Some(meta_path) = &args.meta {
        println!("[*] Reading metadata from: {}", meta_path.display());
        match parse_metadata_dates(meta_path, &args.id_col, &args.date_col) {
            Ok(date_map) => {
                let mut matched = 0usize;
                for (i, h) in alignment.headers.iter().enumerate() {
                    let clean_id = h.trim();
                    let first_word = clean_id.split_whitespace().next().unwrap_or("");
                    let token = first_word.split(|c| c == '|' || c == '_' || c == '.').next().unwrap_or("");

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
                println!("[*] Temporal calibration: matched {}/{} sequences to collection dates.", matched, alignment.num_sequences());
            }
            Err(e) => {
                eprintln!("[!] Warning: metadata parsing failed: {}. Continuing with header dates.", e);
            }
        }
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
    };

    println!("[*] Executing end-to-end TRACE-5.0 pipeline...");
    let dossier = match run_trace50(&alignment, &config) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[!] Pipeline execution failed: {}", e);
            std::process::exit(1);
        }
    };

    // Ensure output directory exists
    if let Err(e) = fs::create_dir_all(&args.out_dir) {
        eprintln!("[!] Failed to create output directory: {}", e);
        std::process::exit(1);
    }

    // 1. Write full visualization JSON dossier
    let json_path = args.out_json.unwrap_or_else(|| args.out_dir.join("trace50_visualization_dossier.json"));
    println!("[+] Serializing full visualization dossier to: {}", json_path.display());
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
    let dyads_csv = args.out_dir.join("acute_transmission_dyads.csv");
    let mut dyads_file = File::create(&dyads_csv).expect("Failed to create dyads CSV");
    writeln!(dyads_file, "source,target,distance,delta_t_years,k_substitutions,bayes_factor,pep,q_value,p_adequacy,status").unwrap();
    for e in &dossier.edges {
        writeln!(
            dyads_file,
            "{},{},{:.5},{:.3},{:.2},{:.2},{:.6},{:.6},{:.5},{}",
            e.source, e.target, e.distance, e.delta_t_years, e.k_substitutions, e.bayes_factor, e.pep, e.q_value, e.p_adequacy, e.status
        ).unwrap();
    }

    let communities_csv = args.out_dir.join("autoclock_communities.csv");
    let mut comm_file = File::create(&communities_csv).expect("Failed to create communities CSV");
    writeln!(comm_file, "community_id,size,n_eff,clock_velocity_mu,se_mu,r_squared,tmrca,ci_tmrca_low,ci_tmrca_high,fieller_status,is_active,operational_tier").unwrap();
    for c in &dossier.communities {
        writeln!(
            comm_file,
            "{},{},{:.2},{:.6},{:.6},{:.4},{:.2},{:.2},{:.2},{},{},\"{}\"",
            c.community_id, c.size, c.n_eff, c.mu, c.se_mu, c.r_squared, c.tmrca, c.ci_mrca[0], c.ci_mrca[1], c.fieller_status, c.is_active_outbreak, c.operational_tier
        ).unwrap();
    }

    let nodes_csv = args.out_dir.join("patient_surveillance_assignments.csv");
    let mut nodes_file = File::create(&nodes_csv).expect("Failed to create nodes CSV");
    writeln!(nodes_file, "taxon_id,date,degree_static,degree_supported,cluster_static,cluster_supported,community_id,operational_tier").unwrap();
    for n in &dossier.nodes {
        writeln!(
            nodes_file,
            "{},{:.2},{},{},{},{},{},\"{}\"",
            n.id, n.date, n.degree_static, n.degree_supported, n.cluster_id_static, n.cluster_id_supported, n.community_id, n.operational_tier
        ).unwrap();
    }

    // Print summary KPIs
    println!("\n================================================================================");
    println!(" OPERATIONAL SURVEILLANCE SCORECARD (KPIs)");
    println!("================================================================================");
    println!(" Total Sequences Evaluated:            {}", dossier.kpi.total_sequences);
    println!(" Total Pairwise Comparisons:           {}", dossier.kpi.total_pairwise_comparisons);
    println!(" Candidate Dyads Screened:             {}", dossier.kpi.candidate_pairs_screened);
    println!(" Clock Violations Purged (Wo Fat):     {}", dossier.kpi.physical_clock_violations_purged);
    println!(" Supported Transmission Edges:         {} (FDR q <= {})", dossier.kpi.supported_transmission_edges, config.fdr_threshold);
    println!("--------------------------------------------------------------------------------");
    println!(" Static Distance Benchmark (d <= {:.1}%):", config.static_distance_threshold * 100.0);
    println!("   - Total Clusters:                   {}", dossier.kpi.static_network.total_clusters);
    println!("   - Clustered Individuals:            {}", dossier.kpi.static_network.clustered_individuals);
    println!("   - Giant Component Size:             {} patients", dossier.kpi.static_network.max_cluster_size);
    println!(" DANNO Calibrated Transmission Network:");
    println!("   - Total Clusters:                   {}", dossier.kpi.danno_network.total_clusters);
    println!("   - Clustered Individuals:            {}", dossier.kpi.danno_network.clustered_individuals);
    println!("   - Giant Component Size:             {} patients", dossier.kpi.danno_network.max_cluster_size);
    println!("   - Giant Component Reduction:        {:.1}%", dossier.kpi.giant_component_reduction_pct);
    println!("--------------------------------------------------------------------------------");
    println!(" CHIN Macro-Scale Epidemiological Scaling:");
    println!("   - Active Transmitting Pool (N_act): {:.0} [CI: {:.0} - {:.0}]",
        dossier.kpi.inferred_active_transmitting_pool,
        dossier.kpi.inferred_active_pool_ci[0],
        dossier.kpi.inferred_active_pool_ci[1]
    );
    println!("   - Surveillance Sampling Fraction:   {:.2}%", dossier.kpi.inferred_surveillance_sampling_fraction_pct);
    println!("   - Borel Branching Decay (R^2):      {:.4}", dossier.macro_scaling.borel_decay.r_squared);
    println!("--------------------------------------------------------------------------------");
    println!(" Resolved STEVE Communities:           {}", dossier.communities.len());
    println!(" Execution Time:                       {:.3} seconds", dossier.metadata.execution_time_seconds);
    println!(" Output Dossier:                       {}", json_path.display());
    println!("================================================================================\n");
}
