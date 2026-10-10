//! TRACE-5.0: Dynamic Molecular Surveillance of Viral Transmission Networks.
//!
//! High-performance, zero-dependency algorithmic kernel implemented in pure Rust
//! for native execution and WebAssembly compilation.

pub mod autoclock;
pub mod chin;
pub mod danno;
pub mod info_danno;
pub mod math;
pub mod network;
pub mod pipeline;
pub mod tn93;
pub mod visualization;

pub use autoclock::{recursive_autoclock_deconvolution, SteveCommunity};
pub use chin::{
    ActivePopulationEstimate, BorelDecayResult, ChinBayesianResult, ChinEstimator,
    ChinPosteriorSummary, DateQuantizationResult,
};
pub use danno::{DannoEstimator, TransmissionDyad};
pub use info_danno::{AlleleFrequencyTable, InfoDannoEstimator};
pub use math::{gammaincc, linregress, ln_gamma, norm_ppf, poisson_cdf, student_t_ppf};
pub use pipeline::{
    run_trace50, run_trace50_json, run_trace50_json_with_progress, run_trace50_with_progress,
    Trace50Config,
};
pub use tn93::{compute_fractional_substitutions, compute_pairwise_tn93, Alignment};
pub use visualization::VisualizationDossier;
