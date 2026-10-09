//! TRACE-5.0: Dynamic Molecular Surveillance of Viral Transmission Networks.
//!
//! High-performance, zero-dependency algorithmic kernel implemented in pure Rust
//! for native execution and WebAssembly compilation.

pub mod math;
pub mod tn93;
pub mod danno;
pub mod info_danno;
pub mod autoclock;
pub mod chin;
pub mod network;
pub mod visualization;
pub mod pipeline;

pub use math::{gammaincc, ln_gamma, poisson_cdf, norm_ppf, student_t_ppf, linregress};
pub use tn93::{Alignment, compute_pairwise_tn93, compute_fractional_substitutions};
pub use danno::{DannoEstimator, TransmissionDyad};
pub use info_danno::{InfoDannoEstimator, AlleleFrequencyTable};
pub use autoclock::{SteveCommunity, recursive_autoclock_deconvolution};
pub use chin::{ChinEstimator, ActivePopulationEstimate, BorelDecayResult, ChinBayesianResult, ChinPosteriorSummary, DateQuantizationResult};
pub use visualization::VisualizationDossier;
pub use pipeline::{Trace50Config, run_trace50, run_trace50_with_progress, run_trace50_json, run_trace50_json_with_progress};
