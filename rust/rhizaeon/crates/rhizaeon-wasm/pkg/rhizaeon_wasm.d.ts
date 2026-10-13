/* tslint:disable */
/* eslint-disable */

/**
 * Fast pre-screening: checks parsimony-informative sites in < 2 ms
 */
export function count_informative_sites(fasta_str: string): number;

export function init_panic_hook(): void;

/**
 * Scans a FASTA string for recombination events and returns a JS Object payload
 */
export function scan_fasta(fasta_str: string, target_landmarks?: number | null, compress_snps?: boolean | null): any;

/**
 * Scans a FASTA string and returns a rich visualization dossier as a JS Object payload
 */
export function scan_fasta_dossier(fasta_str: string, title?: string | null, target_landmarks?: number | null, compress_snps?: boolean | null): any;

/**
 * Scans a FASTA string and returns a rich visualization dossier as a formatted JSON string
 */
export function scan_fasta_dossier_json(fasta_str: string, title?: string | null, target_landmarks?: number | null, compress_snps?: boolean | null): string;

/**
 * Scans a FASTA string and returns a complete, self-contained standalone HTML interactive dashboard
 */
export function scan_fasta_html(fasta_str: string, title?: string | null, target_landmarks?: number | null, compress_snps?: boolean | null): string;

/**
 * Scans a FASTA string and returns a formatted JSON string (optimal for zero-copy UI passing)
 */
export function scan_fasta_json(fasta_str: string, target_landmarks?: number | null, compress_snps?: boolean | null): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly count_informative_sites: (a: number, b: number) => [number, number, number];
    readonly init_panic_hook: () => void;
    readonly scan_fasta: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly scan_fasta_dossier: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly scan_fasta_dossier_json: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly scan_fasta_html: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly scan_fasta_json: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
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
