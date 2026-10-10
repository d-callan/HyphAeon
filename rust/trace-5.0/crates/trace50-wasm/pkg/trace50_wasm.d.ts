/* tslint:disable */
/* eslint-disable */

/**
 * Evaluates molecular clock physical adequacy cumulative Poisson probability.
 */
export function check_clock_adequacy(k_substitutions: number, delta_t: number, seq_len: number, mu: number): number;

/**
 * Evaluates the continuous DANNO Bayes Factor for a candidate transmission dyad.
 * If physical molecular clock adequacy fails (p < 0.05), returns 0.0.
 */
export function compute_danno_bf(dist: number, delta_t: number, seq_len: number, mu: number, tau_bar: number, omega: number, t_span: number): number;

/**
 * Evaluates site-specific Information-Theoretic Transmission Odds (in nats)
 * comparing two sequences against an alignment background.
 */
export function compute_info_danno_score(seq_a: string, seq_b: string, delta_t: number, mu: number): number;

/**
 * Calculates the analytical Tamura-Nei 93 (TN93) pairwise genetic distance
 * between two nucleotide sequences with IUPAC ambiguity resolution.
 */
export function compute_tn93(seq_a: string, seq_b: string): number;

export function init(): void;

/**
 * Runs the full joint Bayesian Monte Carlo uncertainty sampler for CHIN macro-epidemic sizing.
 */
export function run_chin_bayesian_mcmc(n_samples: number, n_edges: number, n_draws: number, r0_min: number, r0_max: number, k_min: number, k_max: number, phi_min: number, phi_max: number, seed: number): any;

/**
 * Runs the complete TRACE-5.0 pipeline and returns the visualization dossier directly
 * as a native JavaScript Object.
 */
export function run_trace50_object(fasta_content: string, config_val: any): any;

/**
 * Runs the complete, end-to-end TRACE-5.0 surveillance pipeline from a FASTA string
 * and optional JSON configuration string, returning the full visualization dossier JSON string.
 */
export function run_trace50_pipeline(fasta_content: string, config_json?: string | null): string;

/**
 * Runs the complete, end-to-end TRACE-5.0 surveillance pipeline from a FASTA string
 * and optional JSON configuration string, providing real-time progress callbacks to JavaScript.
 * Callback signature: `on_progress(stage: string, percent: number, detail: string)`
 */
export function run_trace50_pipeline_with_progress(fasta_content: string, config_json?: string | null, progress_callback?: Function | null): string;

/**
 * Returns the TRACE-5.0 engine version.
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly check_clock_adequacy: (a: number, b: number, c: number, d: number) => number;
    readonly compute_danno_bf: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly compute_info_danno_score: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly compute_tn93: (a: number, b: number, c: number, d: number) => number;
    readonly init: () => void;
    readonly run_chin_bayesian_mcmc: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number];
    readonly run_trace50_object: (a: number, b: number, c: any) => [number, number, number];
    readonly run_trace50_pipeline: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly run_trace50_pipeline_with_progress: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly version: () => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
