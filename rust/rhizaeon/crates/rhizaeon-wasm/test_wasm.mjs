import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import wasmPkg from './pkg/rhizaeon_wasm.js';
const { scan_fasta_json, count_informative_sites, scan_fasta_dossier_json, scan_fasta_html } = wasmPkg;

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function run() {
    console.log("================================================================================");
    console.log("  RHIZAEON WASM CLIENT ENGINE VERIFICATION (Node.js & Browser Runtime)");
    console.log("================================================================================");

    const wasmPath = path.join(__dirname, 'pkg', 'rhizaeon_wasm_bg.wasm');
    const wasmBytes = fs.readFileSync(wasmPath);

    console.log("  WASM Binary loaded:", wasmPath);
    console.log("  WASM Binary size:  ", (wasmBytes.length / 1024).toFixed(1), "KB");
    console.log("--------------------------------------------------------------------------------");

    // Test 1: TYLCV IS76 (Navas-Castillo 2000)
    const tylcvPath = path.resolve(__dirname, '../../../../TOGA_MEME/recombination/benchmarks/01_tylcv_is76/data/tylcv_is76_aligned.fasta');
    const tylcvFasta = fs.readFileSync(tylcvPath, 'utf8');

    const infSites = count_informative_sites(tylcvFasta);
    console.log(`[TEST 1] TYLCV IS76 Benchmark`);
    console.log(`  Informative sites check: ${infSites}`);

    const t0 = performance.now();
    const tylcvJsonStr = scan_fasta_json(tylcvFasta, 16);
    const t1 = performance.now();
    const tylcvResult = JSON.parse(tylcvJsonStr);

    console.log(`  Execution time: ${(t1 - t0).toFixed(2)} ms (internal WASM: ${tylcvResult.run_time_ms.toFixed(2)} ms)`);
    console.log(`  Candidate taxa: ${tylcvResult.screening.candidate_recombinant_indices.length}`);
    console.log(`  Verified events: ${tylcvResult.events.length}`);
    for (const ev of tylcvResult.events) {
        console.log(`  -> Event: ${ev.candidate_name} | Home: ${ev.home_name} | Donor: ${ev.donor_name} | Tract: ${ev.u1}-${ev.u2} (${ev.tract_length} bp) | p-Fisher: ${ev.p_fisher.toExponential(2)} | Z: ${ev.z_phys.toFixed(2)}`);
    }

    const is76 = tylcvResult.events.find(e => e.candidate_name === 'AF271234.1');
    if (is76 && is76.home_name === 'Z25751.1' && is76.donor_name === 'X15656.1') {
        console.log("  >>> PASS: TYLCV IS76 exactly replicated ground truth (AF271234.1 mosaic)!");
    } else {
        console.error("  >>> FAIL: Unexpected TYLCV result");
        process.exit(1);
    }

    console.log("--------------------------------------------------------------------------------");

    // Test 2: HIV-1 KAL153 (Salminen 1997)
    const hivPath = path.resolve(__dirname, '../../../../TOGA_MEME/recombination/benchmarks/03_hiv1_kal153/data/hiv1_kal153_aligned.fasta');
    const hivFasta = fs.readFileSync(hivPath, 'utf8');

    console.log(`[TEST 2] HIV-1 KAL153 Full Genome (10 kb) Benchmark`);
    const t2 = performance.now();
    const hivJsonStr = scan_fasta_json(hivFasta, 16);
    const t3 = performance.now();
    const hivResult = JSON.parse(hivJsonStr);

    console.log(`  Execution time: ${(t3 - t2).toFixed(2)} ms (internal WASM: ${hivResult.run_time_ms.toFixed(2)} ms)`);
    console.log(`  Candidate taxa: ${hivResult.screening.candidate_recombinant_indices.length}`);
    console.log(`  Verified events: ${hivResult.events.length}`);
    for (const ev of hivResult.events) {
        console.log(`  -> Event: ${ev.candidate_name} | Home: ${ev.home_name} | Donor: ${ev.donor_name} | Tract: ${ev.u1}-${ev.u2} (${ev.tract_length} bp) | p-Fisher: ${ev.p_fisher.toExponential(2)} | Z: ${ev.z_phys.toFixed(2)}`);
    }

    const kal153 = hivResult.events.find(e => e.candidate_name === 'R');
    if (kal153 && kal153.home_name === 'B' && kal153.donor_name === 'A') {
        console.log("  >>> PASS: HIV-1 KAL153 exactly replicated ground truth (B/A mosaic)!");
    } else {
        console.error("  >>> FAIL: KAL153 mosaic not found or mismatched");
        process.exit(1);
    }

    console.log("--------------------------------------------------------------------------------");

    // Test 3: Adversarial Null Suite (Darren trap, APOBEC, Codon Darwinian, Clock)
    console.log(`[TEST 3] Adversarial Null Calibration Suite (Zero False Positive Mandate)`);
    const nullFiles = [
        't5_null_darren_trap_rep37.fasta',
        't5_null_apobec_burst_rep81.fasta',
        't5_null_codon_darwinian_rep92.fasta',
        't1_null_clock_rep62.fasta',
    ];

    for (const nf of nullFiles) {
        const nPath = path.resolve(__dirname, `../../../../TOGA_MEME/recombination/benchmarks/14_graduated_difficulty_ladder/alignments/${nf}`);
        const nFasta = fs.readFileSync(nPath, 'utf8');
        const nRes = JSON.parse(scan_fasta_json(nFasta, 16));
        console.log(`  ${nf.padEnd(35)} -> Events: ${nRes.events.length} (H0: ${nRes.is_non_recombinant_h0}) in ${nRes.run_time_ms.toFixed(2)} ms`);
        if (nRes.events.length > 0) {
            console.error(`  >>> FAIL: Adversarial null ${nf} leaked ${nRes.events.length} false positives!`);
            process.exit(1);
        }
    }
    // Test 4: 8-Taxon Cassette Swap (t3_rec_cassette_rep0)
    console.log("--------------------------------------------------------------------------------");
    console.log(`[TEST 4] 8-Taxon Cassette Swap (t3_rec_cassette_rep0) Benchmark`);
    const cassPath = path.resolve(__dirname, '../../../../TOGA_MEME/recombination/benchmarks/14_graduated_difficulty_ladder/alignments/t3_rec_cassette_rep0.fasta');
    const cassFasta = fs.readFileSync(cassPath, 'utf8');
    const t4 = performance.now();
    const cassJsonStr = scan_fasta_json(cassFasta, 16);
    const t5 = performance.now();
    const cassResult = JSON.parse(cassJsonStr);

    console.log(`  Execution time: ${(t5 - t4).toFixed(2)} ms (internal WASM: ${cassResult.run_time_ms.toFixed(2)} ms)`);
    console.log(`  Candidate taxa: ${cassResult.screening.candidate_recombinant_indices.length}`);
    console.log(`  Verified events: ${cassResult.events.length}`);
    for (const ev of cassResult.events) {
        console.log(`  -> Event: ${ev.candidate_name} | Home: ${ev.home_name} | Donor: ${ev.donor_name} | Tract: ${ev.u1}-${ev.u2} (${ev.tract_length} bp) | p-Fisher: ${ev.p_fisher.toExponential(2)} | Z: ${ev.z_phys.toFixed(2)}`);
    }

    const cassEv = cassResult.events.find(e => e.candidate_name === 'R');
    if (cassEv && cassEv.home_name === 'P1' && cassEv.donor_name === 'P2') {
        console.log("  >>> PASS: Cassette Swap exactly replicated ground truth (R = P1 + P2, clonal parents pruned)!");
    } else {
        console.error("  >>> FAIL: Cassette swap result mismatched");
        process.exit(1);
    }

    // Test 5: Rich Visualization Dossier and Standalone HTML Export
    console.log("--------------------------------------------------------------------------------");
    console.log(`[TEST 5] Rich Visualization Dossier and Standalone HTML Export`);

    const dossierJsonStr = scan_fasta_dossier_json(hivFasta, "HIV-1 KAL153 WASM Dossier", 16);
    const dossier = JSON.parse(dossierJsonStr);

    console.log(`  Dossier JSON size: ${(dossierJsonStr.length / 1024).toFixed(1)} KB`);
    console.log(`  Dossier keys: ${Object.keys(dossier).join(', ')}`);
    console.log(`  Metadata title: "${dossier.metadata.title}"`);
    console.log(`  Recombinant query: "${dossier.metadata.query_id}"`);
    console.log(`  Taxa meta count: ${Object.keys(dossier.taxa_meta).length}`);
    console.log(`  Breakpoints count: ${dossier.breakpoints.length}`);
    console.log(`  Signals trajectory points: ${dossier.signals.trajectory_x.length}`);
    console.log(`  Informative SNPs count: ${dossier.signals.informative_snps.length}`);
    console.log(`  Subspaces partition 1 taxa: ${Object.keys(dossier.subspaces.partition_1).length}`);
    console.log(`  Canonical manifold colinear: ${dossier.canonical_manifold.is_colinear} (cos theta = ${dossier.canonical_manifold.cos_theta})`);

    if (dossier.metadata.query_id === 'R' && dossier.breakpoints.length === 2 && dossier.signals.trajectory_x.length > 100) {
        console.log("  >>> PASS: WASM generated complete, valid Visualization Dossier JSON!");
    } else {
        console.error("  >>> FAIL: Dossier JSON validation failed");
        process.exit(1);
    }

    const htmlStr = scan_fasta_html(hivFasta, "HIV-1 KAL153 WASM Dashboard", 16);
    console.log(`  Standalone HTML size: ${(htmlStr.length / 1024).toFixed(1)} KB`);
    if (htmlStr.includes('<script id="rhizaeon-data" type="application/json">') && htmlStr.includes('HIV-1 KAL153 WASM Dashboard')) {
        console.log("  >>> PASS: WASM generated complete standalone HTML dashboard!");
    } else {
        console.error("  >>> FAIL: Standalone HTML generation failed");
        process.exit(1);
    }

    console.log("================================================================================");
    console.log("  ALL WASM VERIFICATION CHECKS (SCAN + DOSSIER + HTML) PASSED!");
    console.log("================================================================================");
}

run().catch(err => {
    console.error("Error running WASM test:", err);
    process.exit(1);
});
