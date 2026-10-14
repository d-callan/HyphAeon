import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import wasmPkg, {
    initSync,
    version,
    compute_tn93,
    compute_danno_bf,
    check_clock_adequacy,
    run_trace50_pipeline,
    run_trace50_object,
    run_chin_bayesian_mcmc,
    compute_info_danno_score
} from './pkg/trace50_wasm.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function run() {
    console.log("================================================================================");
    console.log("  TRACE-5.0 WEB-ASSEMBLY CLIENT ENGINE VERIFICATION (Node.js & Browser Runtime)");
    console.log("================================================================================");

    const wasmPath = path.join(__dirname, 'pkg', 'trace50_wasm_bg.wasm');
    const wasmBytes = fs.readFileSync(wasmPath);
    console.log("  [*] WASM Binary loaded:", wasmPath);
    console.log("  [*] WASM Binary size:  ", (wasmBytes.length / 1024).toFixed(1), "KB");

    initSync({ module: wasmBytes });
    console.log("  [+] WebAssembly instantiated successfully!");
    console.log("  [*] Engine Version:", version());
    console.log("--------------------------------------------------------------------------------");

    // Test 1: TN93 Genetic Distance
    console.log("[TEST 1] Tamura-Nei 93 (TN93) Pairwise Distance");
    const s1 = "ACGTACGTACGTACGTACGTACGT";
    const s2 = "GCGTACGTACGTACGTACGTACGT"; // 1 purine transition in 24 bp
    const dist1 = compute_tn93(s1, s1);
    const dist2 = compute_tn93(s1, s2);
    console.log(`  Identical distance: ${dist1}`);
    console.log(`  1 transition / 24 bp distance: ${dist2.toFixed(5)}`);
    if (dist1 === 0.0 && dist2 > 0.0 && dist2 < 0.1) {
        console.log("  >>> PASS: TN93 analytical calculation verified!");
    } else {
        console.error("  >>> FAIL: Unexpected TN93 distance");
        process.exit(1);
    }
    console.log("--------------------------------------------------------------------------------");

    // Test 2: Clock Adequacy Test
    console.log("[TEST 2] Molecular Clock Adequacy ('Wo Fat Zone' Detection)");
    // 0 mutations over 0 years -> adequate
    const pAdeq0 = check_clock_adequacy(0.0, 0.0, 1000, 2e-3);
    // 0 mutations over 5 years (expected min = 10) -> violation (p < 1e-4)
    const pAdeq5 = check_clock_adequacy(0.0, 5.0, 1000, 2e-3);
    console.log(`  p_adequacy (dt=0 yr, k=0): ${pAdeq0.toFixed(4)} (Adequate)`);
    console.log(`  p_adequacy (dt=5 yr, k=0): ${pAdeq5.toExponential(4)} (Violation)`);
    if (pAdeq0 >= 0.05 && pAdeq5 < 0.05) {
        console.log("  >>> PASS: Physical clock adequacy test verified!");
    } else {
        console.error("  >>> FAIL: Unexpected clock adequacy result");
        process.exit(1);
    }
    console.log("--------------------------------------------------------------------------------");

    // Test 3: DANNO Continuous Bayes Factor
    console.log("[TEST 3] DANNO Continuous Bayes Factor");
    const bfClose = compute_danno_bf(0.001, 0.2, 1000, 2e-3, 1.0, 2.0, 20.0);
    const bfViol = compute_danno_bf(0.0, 5.0, 1000, 2e-3, 1.0, 2.0, 20.0);
    console.log(`  BF for acute dyad (d=0.1%, dt=0.2 yr): ${bfClose.toFixed(2)}`);
    console.log(`  BF for clock violation (d=0.0%, dt=5.0 yr): ${bfViol.toFixed(2)}`);
    if (bfClose > 1000.0 && bfViol === 0.0) {
        console.log("  >>> PASS: DANNO Bayes Factor and hard-zero adequacy verified!");
    } else {
        console.error("  >>> FAIL: Unexpected Bayes Factor result");
        process.exit(1);
    }
    console.log("--------------------------------------------------------------------------------");

    // Test 4: End-to-End Pipeline returning full JSON Dossier
    console.log("[TEST 4] End-to-End Pipeline returning Full Visualization Dossier JSON");
    const sampleFasta = `>seqA_2015.0
ACGTACGTACGTACGTACGTACGTACGTACGT
>seqB_2015.2
ACGTACGTACGTACGTACGTACGTACGTACGT
>seqC_2015.4
ACGTACGTACGTACGTACGTACGTACGTACGA
>seqD_2018.0
TGCATGCATGCATGCATGCATGCATGCATGCA
>seqE_2018.2
TGCATGCATGCATGCATGCATGCATGCATGCT
>seqF_2018.4
TGCATGCATGCATGCATGCATGCATGCATGCT
`;

    const config = {
        mu: 2.0e-3,
        tau_bar: 1.0,
        omega: 2.0,
        fdr_threshold: 0.10,
        min_cluster_size: 2,
        static_distance_threshold: 0.05
    };

    const t0 = performance.now();
    const jsonStr = run_trace50_pipeline(sampleFasta, JSON.stringify(config));
    const t1 = performance.now();

    const dossier = JSON.parse(jsonStr);
    console.log(`  Execution time: ${(t1 - t0).toFixed(2)} ms`);
    console.log(`  Dossier schema version: ${dossier.schema_version}`);
    console.log(`  Sequences evaluated: ${dossier.metadata.num_sequences}`);
    console.log(`  Total pairwise comparisons: ${dossier.kpi.total_pairwise_comparisons}`);
    console.log(`  Supported transmission edges: ${dossier.kpi.supported_transmission_edges || dossier.kpi.certified_transmission_edges}`);
    console.log(`  Static network giant component: ${dossier.kpi.static_network.max_cluster_size}`);
    console.log(`  DANNO transmission clusters: ${dossier.kpi.danno_network.total_clusters}`);
    console.log(`  Active transmitting pool N_act: ${dossier.kpi.inferred_active_transmitting_pool.toFixed(1)}`);
    console.log(`  Resolved STEVE communities: ${dossier.communities.length}`);
    console.log(`  Phase space points: ${dossier.phase_space_points.length}`);

    if (dossier.metadata.num_sequences === 6 && dossier.kpi.total_pairwise_comparisons === 15) {
        console.log("  >>> PASS: Full Visualization Dossier JSON successfully verified!");
    } else {
        console.error("  >>> FAIL: Dossier contents mismatch");
        process.exit(1);
    }
    console.log("--------------------------------------------------------------------------------");

    // Test 5: Native JavaScript Object Return
    console.log("[TEST 5] Native JavaScript Object API (run_trace50_object)");
    const jsObj = run_trace50_object(sampleFasta, config);
    console.log(`  Returned object keys: ${Object.keys(jsObj).join(', ')}`);
    console.log(`  Engine: ${jsObj.engine}`);
    console.log(`  Nodes count: ${jsObj.nodes.length}`);
    console.log(`  Edges count: ${jsObj.edges.length}`);
    if (jsObj.nodes.length === 6 && jsObj.engine.includes("Rust")) {
        console.log("  >>> PASS: Native JS Object output verified!");
    } else {
        console.error("  >>> FAIL: Invalid JS Object returned");
        process.exit(1);
    }
    console.log("--------------------------------------------------------------------------------");

    // Test 6: CHIN Joint Bayesian Monte Carlo Sampler
    console.log("[TEST 6] CHIN Macro-Epidemic Joint Bayesian Monte Carlo Sampler");
    const chinRes = run_chin_bayesian_mcmc(
        500,    // n_samples
        35,     // n_edges
        5000,   // n_draws
        1.1, 2.5, // r0 range
        0.1, 1.0, // k range
        0.5, 2.0, // phi range
        42      // seed
    );
    console.log(`  Draws simulated: ${chinRes.n_draws}`);
    console.log(`  Median active pool N_act: ${chinRes.n_act.median.toFixed(1)}`);
    console.log(`  95% CrI N_act: [${chinRes.n_act.ci_low.toFixed(1)}, ${chinRes.n_act.ci_high.toFixed(1)}]`);
    console.log(`  Median sampling fraction rho: ${chinRes.rho_pct.median.toFixed(2)}%`);
    console.log(`  Median Fano factor F: ${chinRes.fano_factor_median.toFixed(3)}`);
    if (chinRes.n_draws === 5000 && chinRes.n_act.median > 0) {
        console.log("  >>> PASS: CHIN Bayesian Monte Carlo sampler verified!");
    } else {
        console.error("  >>> FAIL: Unexpected CHIN MCMC results");
        process.exit(1);
    }
    console.log("--------------------------------------------------------------------------------");

    // Test 7: Info-DANNO Information-Theoretic Transmission Odds
    console.log("[TEST 7] Info-DANNO Information-Theoretic Transmission Odds Score");
    const seqAcute1 = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    const seqAcute2 = "ACGTACGTACGTACGTACGTACGTACGTACGA"; // 1 difference
    const infoScore = compute_info_danno_score(seqAcute1, seqAcute2, 0.1, 2.0e-3);
    console.log(`  Info-DANNO log-odds score (nats): ${infoScore.toFixed(3)}`);
    if (infoScore > 0.0) {
        console.log("  >>> PASS: Info-DANNO log-odds score verified!");
    } else {
        console.error("  >>> FAIL: Unexpected Info-DANNO score");
        process.exit(1);
    }

    console.log("================================================================================");
    console.log("  ALL TRACE-5.0 WASM VERIFICATION TESTS PASSED SUCCESSFULLY! (100% GREEN)");
    console.log("================================================================================\n");
}

run().catch(err => {
    console.error("Unhandled error:", err);
    process.exit(1);
});
