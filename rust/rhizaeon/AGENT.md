# AGENT.md: Machine-Readable Architectural Specification & Operational Guidelines for RhizAeon

> **Audience:** Autonomous AI Agents, Pair-Programming Assistants, and Automated Code Reviewers.  
> **Repository:** `https://github.com/veg/rhizaeon-code`  
> **Core Version:** RhizAeon PA-FF v2.1 (Native Rust + WebAssembly)

---

## 1. Non-Negotiable Operational Directives & Mandates

### 1.1. Scientific Integrity & Zero Synthetic Mockery Mandate
- **ABSOLUTE PROHIBITION ON SYNTHETIC PLACEHOLDERS:** Never script parametric approximations (e.g. artificial exponential decay curves, sigmoid heuristics) and label them as test statistics, p-values, or lead times. Every single reported metric must be the direct output of executed computational code operating on authentic genomic sequence data.
- **NO HARDCODED THRESHOLDS OR ORACLES:** Never filter on taxon names (e.g. `taxa[i] == "O"` or `taxa[i] == "Ghost"`), metadata attributes, or post-hoc ground truth labels. All algorithmic decisions must arise strictly from unsupervised mathematical operators applied to sequence alignments.
- **DATA LEAKAGE PREVENTION:** In prospective surveillance evaluations, never sort or filter by retrospective collection dates. All temporal streams must respect real-world availability.

### 1.2. Git & Commit Protection Mandates
- **PERMISSION MANDATE:** In any git repository, obtain explicit imperative permission before performing actions that overwrite uncommitted code (`git checkout`, `git reset`, `git restore`).
- **COMMIT PROTECTION:** When committing changes where some files are unchanged, never delete those files irrevocably.

---

## 2. Mathematical Formulations & Algorithmic Specifications

RhizAeon replaces combinatorial triplet scanning with continuous metric manifold projection and physical attention force field deconvolution.

### 2.1. Buneman Metric Manifold & Spectral Landmarks

1. Given an alignment of $N$ contemporary sequences of length $L$, compute whole-sequence normalized Hamming divergence $D_{\text{glob}} \in \mathbb{R}^{N \times N}$.

2. Apply double-centering:

   $$
   B = -\frac{1}{2} H (D_{\text{glob}}^{\odot 2}) H, \quad H = I_N - \frac{1}{N} \mathbf{1} \mathbf{1}^T
   $$

3. Compute top $k=4$ eigenvectors via analytical Jacobi rotation (`jacobi.rs`):

   $$
   m_i = V_4[i] \odot \sqrt{\Lambda_4} \in \mathbb{R}^4
   $$

4. Anchor the unpolarized `[ROOT]` sink token at the metric origin:

   $$
   m_{\text{root}} = [0.0, 0.0, 0.0, 0.0] \in \mathbb{R}^4
   $$

5. For large cohorts ($N > 32$), select $K \le 32$ spectral landmarks on the simplex $\Delta^K$ using farthest-point sampling (`landmarks.rs`).

### 2.2. Tree-RoPE & PhyloBias Continuous-Time Markov Prior

1. **Tree-RoPE Phase Computation (`tree_rope.rs`):** Rotary embedding angles for sequence $i$ in channel $d \in \{0, \dots, \frac{D}{2}-1\}$:

   $$
   \theta_{i, d} = \sum_{c=1}^4 m_{i, c} \cdot \Theta_{d, c}
   $$

2. **PhyloBias Prior (`prior.rs`):** Log-space continuous-time substitution prior with background pseudocount $\epsilon_0 = 0.05$:

   $$
   P_{ij} = \epsilon_0 + (1 - \epsilon_0) \exp(-\lambda_h \cdot D_{\text{glob}}[i, j])
   $$

   $$
   \text{PhyloBias}_{ij} = \ln(\max(10^{-5}, P_{ij}))
   $$

3. **Continuous Attention Weights (`attention.rs`):** Softmax is computed over all $N+1$ nodes (Index 0 = `[ROOT]`, Indices $1, \dots, N$ = Leaves):

   $$
   A_u(i \to \text{root}) + \sum_{j=1}^N A_u(i \to j) = 1.0
   $$

4. **Cumulative Attention Curves:**

   $$
   C_{i, m}(u) = \sum_{t=0}^u \alpha_{i, m}(t) \in \mathbb{R}^{N \times M \times (U+1)}
   $$

### 2.3. Candidate Outlier Screener (`screener.rs`)

To maintain $O(N \cdot K \cdot L)$ throughput, the screener applies strict zero-heap memory invariants:

- **Spatial Coherence:** Evaluated via an in-place circular buffer of window size $W=50$ (zero heap allocation in inner loops).
- **Lazy Participation Ratio (PR):** Only evaluated for taxa exceeding outlier bounds ($z_r > z_{\alpha_r}$ and $\gamma_{\text{root}} \le \theta$), bypassing $>95\%$ of clonal sequences.
- **Autapomorphic Rate Burst Filter:** Sequences with $\gamma_{\text{root}} \le \theta$ (energy pooling into `[ROOT]`) are tagged as rate bursts/hypermutations, preventing spurious recombinant calls.

### 2.4. Dual-Flank Hookean Displacement & Parental Attribution (`attribution.rs`)

1. Recombination creates localized tension between the background clade ($\text{Home}$) and the introgressed segment ($\text{Donor}$).
2. For candidate child $R$, segment $u \in [u_1, u_2]$ experiences force $f_{\text{tract}}$, while flanking segments experience $f_{\text{flank}}$.
3. Hookean apparent coordinate displacement:

   $$
   x_{\text{apparent}}(u) = m_R + \frac{f(u)}{\lambda}
   $$

4. $\text{Home}$ is chosen as the clade minimizing $\lVert x_{\text{apparent}}(\text{flank}) - m_{\text{clade}} \rVert$.
5. $\text{Donor}$ is chosen as the clade minimizing $\lVert x_{\text{apparent}}(\text{tract}) - m_{\text{clade}} \rVert$.
6. If no sampled clade is within metric tolerance of the tract displacement, the event is certified as a **Ghost Donor Introgression** (`is_ghost_donor = true`).

### 2.5. Rigorous Due Diligence & Cycle Adjudication (`due_diligence.rs`)

1. **Fisher's Exact 2x2 Test:** Contingency table of informative sites supporting Home vs. Donor in tract vs. flanks must satisfy $p < \alpha / M$.
2. **Poisson Informative Site Floor:** Minimum 3 supporting informative sites required.
3. **Reticulation Cycle Adjudication:** If mutual events are reported ($A \to B$ and $B \to A$), construct a directed reticulation graph and resolve the 2-cycle in favor of the lineage exhibiting maximal dynamic metric velocity reversal.

---

## 3. Codebase Navigation & Architecture Map

```
crates/
├── rhizaeon-core/             # High-performance algorithmic engine
│   ├── src/
│   │   ├── lib.rs             # Public API exports: RhizAeonEngine, Alignment, Event
│   │   ├── engine.rs          # Main driver: pipeline orchestration & event deduplication
│   │   ├── attention.rs       # Tree-RoPE cross-attention kernel & Softmax
│   │   ├── tree_rope.rs       # Buneman metric MDS & rotary phase angles
│   │   ├── prior.rs           # PhyloBias CTMC prior matrix
│   │   ├── soft_root.rs       # [ROOT] sink token parameters
│   │   ├── landmarks.rs       # Farthest-point landmark selection on Delta^K
│   │   ├── jacobi.rs          # Pure analytical Jacobi rotation for eigendecomposition
│   │   ├── screener.rs        # O(NKL) circular-buffer candidate outlier screener
│   │   ├── attribution.rs     # Dual-flank Hookean parental attribution
│   │   ├── polishing.rs       # Profile likelihood breakpoint polisher
│   │   ├── due_diligence.rs   # Fisher exact test & reticulation cycle adjudication
│   │   ├── fasta.rs           # Zero-copy FASTA parser with validation
│   │   ├── types.rs           # Core Rust types & JSON serde schemas
│   │   └── visualization.rs   # Visualization dossier & interactive HTML builder
│   ├── assets/
│   │   ├── dashboard_template.html                     # Embedded self-contained HTML/Canvas UI
│   │   └── rhizaeon_visualization_dossier.schema.json  # JSON schema (draft 2020-12)
│   └── tests/
│       ├── empirical_benchmarks.rs                     # Integration test on KAL-153 & mtDNA
│       └── simulated_scenarios.rs                      # Synthetic crossover, cassette, nulls
├── rhizaeon-cli/              # CLI executable
│   └── src/main.rs            # Clap-based CLI interface
└── rhizaeon-wasm/             # WebAssembly client-side library
    └── src/lib.rs             # wasm-bindgen bindings: scan_alignment, generate_dashboard
tests/
├── empirical/                 # Real-world benchmark alignments & validation runner
│   ├── data/                  # KAL-153 CRF02_AG, SARS-CoV-2 XBB, Human mtDNA
│   └── test_empirical_benchmarks.py
└── simulated/                 # Simulation suite & 3SEQ head-to-head evaluation
    ├── evaluate_head_to_head_3seq.py
    └── test_simulated_scenarios.py
```

---

## 4. AST Rules & Invariants for Autonomous Modifications

When editing or extending `crates/rhizaeon-core`:

1. **Inner Loop Allocation Prohibition:**
   - Inner loops over genomic coordinates $u \in [0, L)$ must perform **zero heap allocations** (`Vec::new`, `String`, etc.). All buffers must be pre-allocated or circular (`[f64; 50]`).
2. **Taxon Blindness:**
   - The engine must be 100% blind to sequence names. String comparisons matching `"O"`, `"Outgroup"`, or `"Ghost"` inside algorithmic logic are strictly prohibited. Sequence names are used exclusively as labels in output JSON.
3. **No Magic Floats in Conditionals:**
   - Conditional branches must not compare against unparameterized magic floats. Thresholds must be derived from sample dimensions ($N, L, M$), degrees of freedom, or significance $\alpha$.
4. **Coordinate Integrity:**
   - Sequence coordinates are 1-indexed in CLI/reporting ($u \in [1, L]$) and 0-indexed internally ($u \in [0, L)$). Conversions must be explicit and checked against boundary off-by-one errors.

---

## 5. Verification Commands

Before committing any modification, agents MUST verify that the full test battery passes:

```bash
# 1. Rust Workspace Test Suite
cargo test --workspace

# 2. WebAssembly Build Validation
cd crates/rhizaeon-wasm && wasm-pack build --target web && cd ../..

# 3. Python Empirical Benchmark Runner
python3 tests/empirical/test_empirical_benchmarks.py

# 4. Python Simulated Scenario Runner
python3 tests/simulated/test_simulated_scenarios.py
```
