//! Full JSON visualization dossier serialization for TRACE-5.0.
//!
//! Generates comprehensive, self-contained JSON documents designed for interactive
//! visualization across Observable Framework, D3.js, Cytoscape, and WebAssembly dashboards.

use crate::autoclock::SteveCommunity;
use crate::chin::{ActivePopulationEstimate, BorelDecayResult, ChinBayesianResult};
use serde::{Deserialize, Serialize};

/// Master visualization dossier containing all network, phylogenetic, and epidemiological outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationDossier {
    pub schema_version: String,
    pub engine: String,
    pub metadata: DossierMetadata,
    pub kpi: DossierKpi,
    pub nodes: Vec<DossierNode>,
    pub edges: Vec<DossierEdge>,
    pub clusters: Vec<DossierCluster>,
    pub communities: Vec<SteveCommunity>,
    pub macro_scaling: DossierMacroScaling,
    pub phase_space_points: Vec<DossierPhasePoint>,
    pub borel_spectrum: Vec<DossierBorelBin>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierMetadata {
    pub num_sequences: usize,
    /// Sequence length L used by DANNO (median non-gap length unless overridden).
    pub seq_len: usize,
    /// Number of alignment columns.
    #[serde(default)]
    pub alignment_width: usize,
    /// Sequences without a parseable collection date. They are excluded from
    /// DANNO, the static network, AutoClock and the CHIN sample size (never
    /// imputed); their `date` is null and they have no edges.
    #[serde(default)]
    pub num_undated: usize,
    pub date_min: f64,
    pub date_max: f64,
    pub time_span_years: f64,
    pub execution_time_seconds: f64,
    pub parameters: DossierParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierParameters {
    pub mu: f64,
    pub tau_bar: f64,
    pub omega: f64,
    pub fdr_threshold: f64,
    pub alpha_adequacy: f64,
    pub r0: f64,
    #[serde(default = "default_static_threshold")]
    pub static_distance_threshold: f64,
}

fn default_static_threshold() -> f64 {
    0.015
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierKpi {
    pub total_sequences: usize,
    pub total_pairwise_comparisons: usize,
    pub candidate_pairs_screened: usize,
    pub physical_clock_violations_purged: usize,
    pub supported_transmission_edges: usize,
    #[serde(default)]
    pub certified_transmission_edges: usize,
    pub static_network: NetworkSummary,
    pub danno_network: NetworkSummary,
    /// Percent reduction of the largest cluster, static -> DANNO. `null` when there
    /// is no supported edge (no DANNO cluster) or no static cluster to reduce.
    pub giant_component_reduction_pct: Option<f64>,
    /// CHIN N_act. `null` when not estimable (E_obs = 0); see
    /// `inferred_active_pool_status`.
    pub inferred_active_transmitting_pool: Option<f64>,
    /// 95% interval for N_act; `null` when N_act is not estimable.
    pub inferred_active_pool_ci: Option<[f64; 2]>,
    /// "estimated", or the refusal reason when N_act is not estimable.
    #[serde(default)]
    pub inferred_active_pool_status: String,
    /// rho = n / N_act, in percent. `null` whenever N_act is not estimable
    /// (E_obs = 0 would give rho = 0, the reciprocal of N_act = inf).
    pub inferred_surveillance_sampling_fraction_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSummary {
    pub total_clusters: usize,
    pub clustered_individuals: usize,
    pub max_cluster_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierNode {
    pub id: String,
    pub index: usize,
    pub date: f64,
    pub degree_static: usize,
    pub degree_supported: usize,
    #[serde(default)]
    pub degree_certified: usize,
    pub cluster_id_static: usize,
    pub cluster_id_supported: usize,
    #[serde(default)]
    pub cluster_id_certified: usize,
    pub community_id: usize,
    pub operational_tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierEdge {
    pub source: String,
    pub target: String,
    pub source_idx: usize,
    pub target_idx: usize,
    pub distance: f64,
    pub delta_t_years: f64,
    pub k_substitutions: f64,
    pub bayes_factor: f64,
    pub pep: f64,
    pub q_value: f64,
    pub p_adequacy: f64,
    pub status: String,
    #[serde(default)]
    pub is_static: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierCluster {
    pub cluster_id: usize,
    pub size: usize,
    pub members: Vec<String>,
    pub time_span_years: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierMacroScaling {
    pub composite_parameter_theta: f64,
    /// Surveillance sampling fraction rho = n / N_act; `null` when N_act is not
    /// estimable (see `active_population_status`).
    pub surveillance_coverage_rho: Option<f64>,
    /// `null` when N_act is not estimable (E_obs = 0); the reason is in
    /// `active_population_status`.
    pub active_population: Option<ActivePopulationEstimate>,
    /// "estimated", or the refusal reason when N_act is not estimable.
    #[serde(default)]
    pub active_population_status: String,
    pub borel_decay: BorelDecayResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bayesian_mcmc: Option<ChinBayesianResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierPhasePoint {
    pub dist: f64,
    pub delta_t: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DossierBorelBin {
    pub cluster_size: usize,
    pub count: usize,
}
