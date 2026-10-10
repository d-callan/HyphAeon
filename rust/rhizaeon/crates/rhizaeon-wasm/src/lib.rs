use wasm_bindgen::prelude::*;
use rhizaeon_core::{parse_fasta, RhizAeonEngine};

#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Scans a FASTA string for recombination events and returns a JS Object payload
#[wasm_bindgen]
pub fn scan_fasta(
    fasta_str: &str,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
) -> Result<JsValue, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new().with_landmarks(k);
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    serde_wasm_bindgen::to_value(&res).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scans a FASTA string and returns a formatted JSON string (optimal for zero-copy UI passing)
#[wasm_bindgen]
pub fn scan_fasta_json(
    fasta_str: &str,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
) -> Result<String, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new().with_landmarks(k);
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    serde_json::to_string_pretty(&res).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Fast pre-screening: checks parsimony-informative sites in < 2 ms
#[wasm_bindgen]
pub fn count_informative_sites(fasta_str: &str) -> Result<usize, JsValue> {
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;
    Ok(rhizaeon_core::fasta::count_parsimony_informative_sites(&aln))
}

/// Scans a FASTA string and returns a rich visualization dossier as a JS Object payload
#[wasm_bindgen]
pub fn scan_fasta_dossier(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
) -> Result<JsValue, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new().with_landmarks(k);
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    let dossier = rhizaeon_core::generate_visualization_dossier(&aln, &res, title.as_deref());
    serde_wasm_bindgen::to_value(&dossier).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scans a FASTA string and returns a rich visualization dossier as a formatted JSON string
#[wasm_bindgen]
pub fn scan_fasta_dossier_json(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
) -> Result<String, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new().with_landmarks(k);
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    let dossier = rhizaeon_core::generate_visualization_dossier(&aln, &res, title.as_deref());
    serde_json::to_string_pretty(&dossier).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Scans a FASTA string and returns a complete, self-contained standalone HTML interactive dashboard
#[wasm_bindgen]
pub fn scan_fasta_html(
    fasta_str: &str,
    title: Option<String>,
    target_landmarks: Option<usize>,
    compress_snps: Option<bool>,
) -> Result<String, JsValue> {
    init_panic_hook();

    let start_time = js_sys::Date::now();
    let aln = parse_fasta(fasta_str).map_err(|e| JsValue::from_str(&e))?;

    let k = target_landmarks.unwrap_or(16);
    let mut engine = RhizAeonEngine::new().with_landmarks(k);
    if let Some(cs) = compress_snps {
        engine = engine.with_compress_snps(cs);
    }
    let mut res = engine.scan(&aln);

    let end_time = js_sys::Date::now();
    res.run_time_ms = end_time - start_time;

    let dossier = rhizaeon_core::generate_visualization_dossier(&aln, &res, title.as_deref());
    rhizaeon_core::generate_standalone_html(&dossier).map_err(|e| JsValue::from_str(&e))
}
