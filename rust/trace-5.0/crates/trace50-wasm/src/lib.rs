//! WebAssembly client-side engine for TRACE-5.0.
//!
//! Exposes the complete molecular transmission network surveillance engine
//! directly to Web browsers, Observable Framework, D3.js dashboards, and Node.js.

use wasm_bindgen::prelude::*;
use trace50_core::{
    compute_pairwise_tn93, DannoEstimator, Alignment, Trace50Config, run_trace50, run_trace50_json,
    run_trace50_json_with_progress
};
use trace50_core::tn93::char_to_iupac_mask;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Returns the TRACE-5.0 engine version.
#[wasm_bindgen]
pub fn version() -> String {
    "5.0.0".to_string()
}

/// Calculates the analytical Tamura-Nei 93 (TN93) pairwise genetic distance
/// between two nucleotide sequences with IUPAC ambiguity resolution.
#[wasm_bindgen]
pub fn compute_tn93(seq_a: &str, seq_b: &str) -> f64 {
    let bits_a: Vec<u8> = seq_a.bytes().map(char_to_iupac_mask).collect();
    let bits_b: Vec<u8> = seq_b.bytes().map(char_to_iupac_mask).collect();
    compute_pairwise_tn93(&bits_a, &bits_b)
}

/// Evaluates the continuous DANNO Bayes Factor for a candidate transmission dyad.
/// If physical molecular clock adequacy fails (p < 0.05), returns 0.0.
#[wasm_bindgen]
pub fn compute_danno_bf(
    dist: f64,
    delta_t: f64,
    seq_len: usize,
    mu: f64,
    tau_bar: f64,
    omega: f64,
    t_span: f64,
) -> f64 {
    let estimator = DannoEstimator::new(
        seq_len,
        mu,
        tau_bar,
        omega,
        t_span,
        0.05,
        0.065,
        0.0004,
    );
    estimator.compute_bayes_factor(dist, delta_t)
}

/// Evaluates molecular clock physical adequacy cumulative Poisson probability.
#[wasm_bindgen]
pub fn check_clock_adequacy(
    k_substitutions: f64,
    delta_t: f64,
    seq_len: usize,
    mu: f64,
) -> f64 {
    let estimator = DannoEstimator::new(seq_len, mu, 1.0, 2.0, 20.0, 0.05, 0.065, 0.0004);
    let (p_adeq, _) = estimator.check_clock_adequacy(k_substitutions, delta_t, Some(mu));
    p_adeq
}

/// Runs the complete, end-to-end TRACE-5.0 surveillance pipeline from a FASTA string
/// and optional JSON configuration string, providing real-time progress callbacks to JavaScript.
/// Callback signature: `on_progress(stage: string, percent: number, detail: string)`
#[wasm_bindgen]
pub fn run_trace50_pipeline_with_progress(
    fasta_content: &str,
    config_json: Option<String>,
    progress_callback: Option<js_sys::Function>,
) -> Result<String, JsValue> {
    run_trace50_json_with_progress(fasta_content, config_json.as_deref(), |stage, pct, detail| {
        if let Some(ref cb) = progress_callback {
            let this = JsValue::NULL;
            let arg_stage = JsValue::from_str(stage);
            let arg_pct = JsValue::from_f64(pct);
            let arg_detail = JsValue::from_str(detail);
            let _ = cb.call3(&this, &arg_stage, &arg_pct, &arg_detail);
        }
    }).map_err(|e| JsValue::from_str(&e))
}

/// Runs the complete, end-to-end TRACE-5.0 surveillance pipeline from a FASTA string
/// and optional JSON configuration string, returning the full visualization dossier JSON string.
#[wasm_bindgen]
pub fn run_trace50_pipeline(
    fasta_content: &str,
    config_json: Option<String>,
) -> Result<String, JsValue> {
    run_trace50_json(fasta_content, config_json.as_deref())
        .map_err(|e| JsValue::from_str(&e))
}

/// Runs the complete TRACE-5.0 pipeline and returns the visualization dossier directly
/// as a native JavaScript Object.
#[wasm_bindgen]
pub fn run_trace50_object(
    fasta_content: &str,
    config_val: JsValue,
) -> Result<JsValue, JsValue> {
    let alignment = Alignment::from_fasta_str(fasta_content)
        .map_err(|e| JsValue::from_str(&e))?;

    let config: Trace50Config = if !config_val.is_undefined() && !config_val.is_null() {
        serde_wasm_bindgen::from_value(config_val)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse config object: {}", e)))?
    } else {
        Trace50Config::default()
    };

    let dossier = run_trace50(&alignment, &config)
        .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&dossier)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize dossier to JS object: {}", e)))
}

/// Runs the full joint Bayesian Monte Carlo uncertainty sampler for CHIN macro-epidemic sizing.
#[wasm_bindgen]
pub fn run_chin_bayesian_mcmc(
    n_samples: usize,
    n_edges: usize,
    n_draws: usize,
    r0_min: f64,
    r0_max: f64,
    k_min: f64,
    k_max: f64,
    phi_min: f64,
    phi_max: f64,
    seed: u32,
) -> Result<JsValue, JsValue> {
    let chin = trace50_core::ChinEstimator::new(1.5);
    let res = chin.run_joint_bayesian_monte_carlo(
        n_samples,
        n_edges,
        n_draws,
        (r0_min, r0_max),
        (k_min, k_max),
        (phi_min, phi_max),
        seed as u64,
    ).map_err(|e| JsValue::from_str(&e))?;
    serde_wasm_bindgen::to_value(&res)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize CHIN result: {}", e)))
}

/// Evaluates site-specific Information-Theoretic Transmission Odds (in nats)
/// comparing two sequences against an alignment background.
#[wasm_bindgen]
pub fn compute_info_danno_score(
    seq_a: &str,
    seq_b: &str,
    delta_t: f64,
    mu: f64,
) -> f64 {
    let bits_a: Vec<u8> = seq_a.bytes().map(char_to_iupac_mask).collect();
    let bits_b: Vec<u8> = seq_b.bytes().map(char_to_iupac_mask).collect();
    let dist = compute_pairwise_tn93(&bits_a, &bits_b);
    let l = seq_a.len().min(seq_b.len());
    let danno = DannoEstimator::new(l, mu, 1.0, 2.0, 20.0, 0.05, 0.065, 0.0004);
    let base_bf = danno.compute_bayes_factor(dist, delta_t);
    if base_bf <= 0.0 {
        return -1000.0;
    }
    base_bf.ln()
}
