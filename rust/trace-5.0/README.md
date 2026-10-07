# TRACE-5.0: Dynamic Molecular Surveillance of Viral Transmission Networks

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![WASM](https://img.shields.io/badge/wasm-ready-blue.svg)](https://webassembly.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A unified, multi-scale statistical framework for molecular epidemiology that replaces arbitrary distance cutoffs with coalescent Bayes Factors, spectral evolutionary velocity deconvolution, and macro-epidemic quadratic scaling dynamics.**

---

## 1. Scientific & Methodological Framework

Legacy molecular surveillance relies on decoupled, rectangular heuristics---such as demanding pairwise genetic distance $d \le 1.5\%$ alongside sampling within an arbitrary trailing calendar window---while full Bayesian phylodynamics remains computationally intractable for longitudinal municipal registries. This decoupling triggers catastrophic failure: it severs genuine transmission chains delayed by clinical diagnosis lags while clustering unevolved repeat accessions and laboratory cross-contaminations across calendar decades.

**TRACE-5.0** resolves these blind spots across three physically coupled dimensions:

```
+---------------------------------------------------------------------------------------+
|                                    TRACE-5.0                                          |
|                                                                                       |
|  [Dimension 1: Micro-Scale]     [Dimension 2: Meso-Scale]    [Dimension 3: Macro-Scale]|
|  DANNO Bayes Factor Engine     STEVE / AutoClock Engine       CHIN Population Inversion|
|  - Kingman coalescent prior    - Spectral Cheeger cuts       - Quadratic collision law|
|  - Upper incomplete gamma Q    - Jacobi eigensolver          - Active pool N_act      |
|  - Physical clock adequacy     - Bartlett effective N_eff    - Sampling fraction rho  |
|  - Benjamini-Hochberg FDR      - Fieller's theorem (g < 1)   - Borel branching decay  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
               [Methodological Extension: Info-DANNO Forensic Prior]
               - Site-specific empirical allele frequency weighting (nats)
               - Shared rare neutral variant reward (P < 10^-11)
               - Convergent drug-resistance (DRM) homoplasy filtering
```

- **Dimension 1 (Micro-Dyadic Scale: DANNO)**: Dynamic Ancestral Negative-binomial Network Odds. Compounds an exponential Kingman coalescent intra-host depth prior $\tau \sim \text{Exp}(\bar{\tau})$ with forward Poisson mutational accumulation over sampling interval $\Delta T$. Evaluates closed-form marginal likelihoods via the regularized upper incomplete gamma function $Q(K+1, \dots)$ without numerical quadrature. A physical molecular clock adequacy test hard-zeroes links violating active forward replication ($p_{\text{adeq}} < 0.05 \implies \mathbf{BF=0}$), purging stagnant clinic duplicates and contamination. Monotonic Benjamini-Hochberg ranking on Posterior Error Probabilities (PEPs) guarantees formal False Discovery Rate control.
- **Dimension 2 (Meso-Community Scale: STEVE / AutoClock)**: Spectral Temporal Evolutionary Velocity Engine. Deconstructs sequence affinity manifolds along Fiedler vector Cheeger cuts using Newman-Girvan modularity optimization. Resolves structured outbreak communities in seconds without building phylogenetic trees. Computes community-specific evolutionary velocities ($\mu$), Bartlett effective sample sizes ($N_{\text{eff}}$), and exact Fieller theorem confidence intervals ($g < 1$) for $t_{\text{MRCA}}$, separating rapidly expanding young networks from chronic background reservoirs.
- **Dimension 3 (Macro-Population Scale: CHIN)**: Cluster-to-Host Incidence and Network Estimator. Inverts quadratic network collision dynamics ($E_{\text{obs}} \approx \frac{1}{2} n \rho R_0 = \frac{1}{2} \rho^2 N_{\text{act}} R_0$) to estimate the absolute regional active transmitting pool $N_{\text{act}}$ and surveillance sampling completeness $\hat{\rho}$. Unsupervised topological diagnostics evaluate cluster size decay against subcritical Borel branching processes to audit for non-random sampling artifacts.
- **Methodological Extension (Info-DANNO Forensic Imputation)**: Incorporates database-level mutation prevalence catalogs (Stanford HIVDB Group-M facts) or cohort-internal empirical frequencies to evaluate transmission odds per site (Tzou, Kosakovsky Pond et al., *PLoS ONE* 2020). Replaces identical-by-state with identical-by-descent by rewarding rare neutral variants ($f_l < 0.8\%$) while discounting convergent drug-resistance mutations (DRMs).

---

## 2. Implementations: Dual Python & Pure-Rust Architecture

TRACE-5.0 is implemented in two complementary, fully validated engines:
1. **Python Engine (`trace50`)**: Pure Python/NumPy/SciPy module and CLI for data science and analysis pipelines.
2. **Rust & WebAssembly Engine (`trace50-rust`)**: High-performance, zero-C-dependency Rust kernel (`trace50-core`), standalone native CLI binary (`trace50`), and compiled WebAssembly module (`trace50-wasm`) for browser and Observable dashboards.

---

## 3. Quickstart: Rust Standalone CLI

### Installation & Build
```bash
# Clone repository
git clone https://github.com/veg/trace-5.0.git
cd trace-5.0/trace50-rust

# Build optimized release binary
cargo build --release --bin trace50
```
The compiled executable is located at `target/release/trace50`.

### Running Transmission Surveillance
```bash
./target/release/trace50 \
  --fasta /path/to/alignment.fasta \
  --meta /path/to/metadata.csv \
  --id-col id \
  --date-col date \
  --fdr 0.05 \
  --out-dir surveillance_results/
```

### CLI Output Summary Scorecard
```
================================================================================
 TRACE-5.0: Molecular Transmission Surveillance Engine (Rust Native)
================================================================================
[*] Loading FASTA alignment from: aligned.fasta
[*] Ingested 388 sequences across 1092 nucleotide sites.
[*] Reading metadata from: attributes.csv
[*] Temporal calibration: matched 388/388 sequences to collection dates.
[*] Executing end-to-end TRACE-5.0 pipeline...
[+] Serializing full visualization dossier to: surveillance_results/trace50_visualization_dossier.json

================================================================================
 OPERATIONAL SURVEILLANCE SCORECARD (KPIs)
================================================================================
 Total Sequences Evaluated:            388
 Total Pairwise Comparisons:           75078
 Candidate Dyads Screened:             39
 Clock Violations Purged (Wo Fat):     0
 Certified Transmission Edges:         8 (FDR q <= 0.05)
--------------------------------------------------------------------------------
 Static Distance Benchmark (d <= 1.5%):
   - Total Clusters:                   19
   - Clustered Individuals:            151
   - Giant Component Size:             109 patients
 DANNO Calibrated Transmission Network:
   - Total Clusters:                   8
   - Clustered Individuals:            16
   - Giant Component Size:             2 patients
   - Giant Component Reduction:        98.2%
--------------------------------------------------------------------------------
 CHIN Macro-Scale Epidemiological Scaling:
   - Active Transmitting Pool (N_act): 14114 [CI: 7058 - 28221]
   - Surveillance Sampling Fraction:   2.75%
   - Borel Branching Decay (R^2):      0.0000
--------------------------------------------------------------------------------
 Resolved STEVE Communities:           1
 Execution Time:                       0.112 seconds
================================================================================
```

---

## 4. Quickstart: WebAssembly (WASM) & Web Dashboards

The WebAssembly engine enables TRACE-5.0 to run entirely client-side inside Web browsers, Observable Framework, D3.js visualizations, and Node.js without requiring backend server compute.

### Compile WASM Target
```bash
cd trace50-rust/crates/trace50-wasm
wasm-pack build --target web --out-dir pkg
```

### Verification in Node.js
```bash
node test_wasm.mjs
```

### Client-Side Browser / JavaScript API
```javascript
import init, { run_trace50_object, compute_tn93, compute_danno_bf } from './pkg/trace50_wasm.js';

await init();

// 1. Analytical Pairwise Distance with IUPAC ambiguities
const dist = compute_tn93("ACGTACGT", "GCGTACGT");

// 2. Coalescent Bayes Factor evaluation
const bf = compute_danno_bf(0.001, 0.2, 1000, 2e-3, 1.0, 2.0, 20.0);

// 3. Complete End-to-End Pipeline in browser memory:
const fastaContent = ">seq1_2015.1\nACGT...\n>seq2_2015.3\nACGT...";
const config = {
  mu: 2.0e-3,
  tau_bar: 1.0,
  omega: 2.0,
  fdr_threshold: 0.05,
  min_cluster_size: 2
};

const dossier = run_trace50_object(fastaContent, config);
console.log("Certified edges:", dossier.kpi.certified_transmission_edges);
console.log("Giant component reduction:", dossier.kpi.giant_component_reduction_pct, "%");
console.log("Estimated active reservoir N_act:", dossier.kpi.inferred_active_transmitting_pool);
```

---

## 5. Quickstart: Python Package

### Installation
```bash
pip install -e .
```

### Python API Usage
```python
from trace50.tn93 import parse_fasta, encode_alignment, compute_tn93_distance_matrix
from trace50.bayes_factor import DannoEstimator
from trace50.chin import ChinEstimator

# 1. Load and encode alignment
records = parse_fasta("examples/sample_alignment.fasta")
headers, M_int, M_bits = encode_alignment(records)
D = compute_tn93_distance_matrix(M_bits)

# 2. DANNO Bayesian edge inference & FDR control
dates = [2015.1, 2015.2, 2015.3, 2016.1, 2017.0, 2017.4]
danno = DannoEstimator(seq_len=M_bits.shape[1], mu=2e-3, tau_bar=1.0, omega=2.0)
certified_dyads = danno.screen_dyads(
    indices=list(range(len(headers))),
    dates=dates,
    D_matrix=D,
    max_delta_t=2.5,
    fdr_threshold=0.10
)

# 3. Macro-scale population inversion
chin = ChinEstimator(R0_default=1.5)
active_pool = chin.estimate_active_population(len(headers), len(certified_dyads))
print(f"Active transmitting pool N_act: {active_pool['N_act']:.1f}")
```

---

## 6. Visualization Dossier JSON Schema

Both the native CLI and WASM engines output a standardized, self-contained JSON dossier (`trace50_visualization_dossier.json`):

| Section | Contents |
| :--- | :--- |
| `metadata` | Sequence count, alignment length, temporal span, runtime, model parameters. |
| `kpi` | Pairwise comparisons, candidate pairs, clock violations purged, certified transmission links, static vs. DANNO giant component sizes, active pool $N_{\text{act}}$, and sampling coverage $\hat{\rho}$. |
| `nodes` | Sequence identifier, collection date, static degree, certified degree, cluster membership, resolved community ID, and operational triage tier. |
| `edges` | Source, target, TN93 distance, elapsed time $\Delta T$, substitution count $K$, continuous Bayes Factor, PEP, $q$-value, adequacy $p$-value, and status. |
| `clusters` | Certified transmission clusters with member lists, size spectrum, and internal time spans. |
| `communities` | STEVE spectral communities with evolutionary clock velocity $\mu$, $t_{\text{MRCA}}$, Fieller theorem confidence set, and active outbreak flag. |
| `macro_scaling` | Inverted composite transmission parameter $\theta$, sampling fraction $\rho$, active pool $N_{\text{act}}$, and Borel decay statistics. |
| `phase_space_points` | Points for the $(d, \Delta T)$ phase plot categorized by epidemiological status. |
| `borel_spectrum` | Cluster size frequency distribution for audit of single-linkage percolation collapse. |

---

## 7. Empirical Benchmarks on Real-World Cohorts

| Surveillance Cohort | Sequences ($N$) | Comparisons | Legacy Giant Component ($d \le 1.5\%$) | DANNO Certified Network | Clock Violations Purged |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Washington, DC** (Perez et al.) | 1,658 | 1.37 M | 90 patients | 23 patients max | 8 links |
| **Middle Tennessee** (Dennis et al.) | 2,915 | 4.25 M | 332 patients | 27 patients max | 422 links |
| **Japan Nationwide** (Shiino et al.) | 5,232 | 13.68 M | 2,896 patients | 4 patients max | 673 links |
| **China CRF07_BC** (Zhao et al.) | 454 | 102,831 | 414 patients | 17 patients max | 576 links |

---

## 8. Automated Test Suite

Run the full verification suites across Python, Rust, and WebAssembly:

```bash
# Python test suite (33 unit tests)
pytest tests/

# Rust test suite (core + CLI + integration tests)
cd trace50-rust && cargo test --all

# WebAssembly Node.js verification test suite
cd trace50-rust/crates/trace50-wasm && node test_wasm.mjs
```

---

## 9. Citation

If you use TRACE-5.0, DANNO, STEVE / AutoClock, or CHIN in your research or surveillance operations, please cite:

```bibtex
@article{kosakovskypond2026trace50,
  title={Dynamic Molecular Surveillance of Viral Transmission Networks},
  author={Kosakovsky Pond, Sergei L. and colleagues},
  journal={Virus Evolution},
  year={2026}
}

@article{tzou2020analysis,
  title={Analysis of shared mutations reveals extensive transmission linkage in viral epidemics},
  author={Tzou, Philip L. and Kosakovsky Pond, Sergei L. and others},
  journal={PLoS ONE},
  volume={15},
  number={5},
  pages={e0225352},
  year={2020}
}
```

---

## 10. License

MIT License. Developed by the Temple University Institute for Genomics and Evolutionary Medicine (iGEM) and the Viral Evolution Group (VEG).
