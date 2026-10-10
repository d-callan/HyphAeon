//! Full JSON visualization dossier serialization for TRACE-5.0.
//!
//! Generates comprehensive, self-contained JSON documents designed for interactive
//! visualization across Observable Framework, D3.js, Cytoscape, and WebAssembly dashboards.

use serde::{Deserialize, Serialize};
use crate::autoclock::SteveCommunity;
use crate::chin::{ActivePopulationEstimate, BorelDecayResult, ChinBayesianResult};

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
    pub seq_len: usize,
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
    pub giant_component_reduction_pct: f64,
    pub inferred_active_transmitting_pool: f64,
    pub inferred_active_pool_ci: [f64; 2],
    pub inferred_surveillance_sampling_fraction_pct: f64,
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
    pub surveillance_coverage_rho: f64,
    pub active_population: ActivePopulationEstimate,
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
