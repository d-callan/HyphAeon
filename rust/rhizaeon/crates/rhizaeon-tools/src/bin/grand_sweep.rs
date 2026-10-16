use clap::Parser;
use rayon::prelude::*;
use rhizaeon::{Alignment, RhizAeonEngine};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Exact nucleotide frequencies (HKY85)
const PI: [f64; 4] = [0.30, 0.20, 0.20, 0.30];
const KAPPA: f64 = 2.5;

/// High-speed 64-bit pseudo-random number generator (XorShift64Star)
struct FastRng {
    state: u64,
}

impl FastRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545f4914f6cdd1d)
    }

    #[inline(always)]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    #[inline(always)]
    fn gen_range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + self.next_f64() * (high - low)
    }

    #[inline(always)]
    fn gen_range_usize(&mut self, low: usize, high: usize) -> usize {
        if low >= high {
            return low;
        }
        low + (self.next_u64() as usize % (high - low + 1))
    }
}

/// Marsaglia and Tsang (2000) standard normal variate
#[inline(always)]
fn sample_standard_normal(rng: &mut FastRng) -> f64 {
    let u1 = rng.next_f64().max(1e-15);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Marsaglia and Tsang (2000) exact Gamma distribution sampler
fn sample_gamma(alpha: f64, rng: &mut FastRng) -> f64 {
    if alpha < 1.0 {
        let g = sample_gamma(alpha + 1.0, rng);
        let u = rng.next_f64().max(1e-15);
        return g * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = sample_standard_normal(rng);
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * z * z * z * z {
            return d * v;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Exact analytical HKY85 transition matrix for branch length `t`
#[inline(always)]
fn compute_hky85_p_mat(t: f64, kappa: f64, pi: &[f64; 4]) -> [[f64; 4]; 4] {
    let pi_r = pi[0] + pi[2]; // Purines A, G
    let pi_y = pi[1] + pi[3]; // Pyrimidines C, T

    // Scaling coefficient beta
    let beta_coeff = pi_r * pi_y + kappa * (pi[0] * pi[2] + pi[1] * pi[3]);
    let u = t / (2.0 * beta_coeff);

    let exp_u = (-u).exp();
    let exp_u_r = (-(1.0 + (kappa - 1.0) * pi_r) * u).exp();
    let exp_u_y = (-(1.0 + (kappa - 1.0) * pi_y) * u).exp();

    let mut p = [[0.0; 4]; 4];

    for i in 0..4 {
        let is_purine = i == 0 || i == 2;
        let pi_g = if is_purine { pi_r } else { pi_y };
        let exp_u_g = if is_purine { exp_u_r } else { exp_u_y };

        let mut sum_non_diag = 0.0;
        for j in 0..4 {
            if i == j {
                continue;
            }
            let is_ts = (is_purine && (j == 0 || j == 2)) || (!is_purine && (j == 1 || j == 3));
            let p_ij = if !is_ts {
                // Transversion
                pi[j] * (1.0 - exp_u)
            } else {
                // Transition
                pi[j] + pi[j] * (1.0 / pi_g - 1.0) * exp_u - (pi[j] / pi_g) * exp_u_g
            };
            let clamped = p_ij.max(0.0);
            p[i][j] = clamped;
            sum_non_diag += clamped;
        }
        p[i][i] = (1.0 - sum_non_diag).max(0.0);
    }
    p
}

/// Precompute HKY85 position-specific transition matrices for length `L` along branch length `branch_len`
fn precompute_p_mats(
    length: usize,
    branch_len: f64,
    site_rates: Option<&[f64]>,
) -> Vec<[[f64; 4]; 4]> {
    let mut mats = Vec::with_capacity(length);
    let default_mat = if site_rates.is_none() {
        Some(compute_hky85_p_mat(branch_len, KAPPA, &PI))
    } else {
        None
    };

    for u in 0..length {
        let p_mat = match site_rates {
            Some(rates) => compute_hky85_p_mat(branch_len * rates[u], KAPPA, &PI),
            None => default_mat.unwrap(),
        };
        mats.push(p_mat);
    }
    mats
}

/// High-speed nucleotide sequence evolution using precomputed transition matrices (zero transcendental calls)
#[inline(always)]
fn evolve_with_p_mats(src: &[u8], p_mats: &[[[f64; 4]; 4]], rng: &mut FastRng) -> Vec<u8> {
    let mut dest = Vec::with_capacity(src.len());
    for (u, &parent_state) in src.iter().enumerate() {
        let row = &p_mats[u][parent_state as usize];
        let rand_val = rng.next_f64();

        let new_state = if rand_val < row[0] {
            0u8
        } else if rand_val < row[0] + row[1] {
            1u8
        } else if rand_val < row[0] + row[1] + row[2] {
            2u8
        } else {
            3u8
        };
        dest.push(new_state);
    }
    dest
}

#[derive(Debug, Clone)]
struct SimEvent {
    u1: usize,
    u2: usize,
    donor_clade: usize, // 1 for P2, 2 for P3
    is_5prime_crossover: bool,
    is_3prime_crossover: bool,
    mut_in_tract: usize,
}

#[derive(Debug)]
struct TrialResult {
    trial_id: usize,
    is_recombinant_true: u8,
    length: usize,
    num_taxa: usize,
    d_parent: f64,
    tract_len: usize,
    k_true: usize,
    u1_true: usize,
    u2_true: usize,
    mut_in_tract: usize,
    mut_total: usize,
    heterotachy_alpha: f64,
    informative_sites: usize,
    detected: u8,
    unique_events_inferred: usize,
    recall: f64,
    precision: f64,
    bg_fp_count: usize,
    bg_fpr: f64,
    jaccard: f64,
    bp1_error: usize,
    bp2_error: usize,
    run_time_ms: f64,
    // Multi-event expansion columns:
    num_events_true: usize,
    cardinality_error: i64,
    event_recall: f64,
    event_precision: f64,
    event_f1: f64,
    mean_jaccard: f64,
    mean_bp_error: f64,
    has_crossover: u8,
    has_cassette: u8,
    has_multi_donor: u8,
}

impl TrialResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{:.5},{},{},{},{},{},{},{:.4},{},{},{},{:.4},{:.4},{},{:.5},{:.4},{},{},{:.2},{},{},{:.4},{:.4},{:.4},{:.4},{:.1},{},{},{}",
            self.trial_id,
            self.is_recombinant_true,
            self.length,
            self.num_taxa,
            self.d_parent,
            self.tract_len,
            self.k_true,
            self.u1_true,
            self.u2_true,
            self.mut_in_tract,
            self.mut_total,
            self.heterotachy_alpha,
            self.informative_sites,
            self.detected,
            self.unique_events_inferred,
            self.recall,
            self.precision,
            self.bg_fp_count,
            self.bg_fpr,
            self.jaccard,
            self.bp1_error,
            self.bp2_error,
            self.run_time_ms,
            self.num_events_true,
            self.cardinality_error,
            self.event_recall,
            self.event_precision,
            self.event_f1,
            self.mean_jaccard,
            self.mean_bp_error,
            self.has_crossover,
            self.has_cassette,
            self.has_multi_donor
        )
    }
}

fn compute_jaccard(t_u1: usize, t_u2: usize, p_u1: usize, p_u2: usize) -> f64 {
    if p_u1 == 0 || p_u2 == 0 || p_u1 > p_u2 {
        return 0.0;
    }
    let start = t_u1.max(p_u1);
    let end = t_u2.min(p_u2);
    if end < start {
        return 0.0;
    }
    let inter = (end - start + 1) as f64;
    let union = ((t_u2 - t_u1 + 1) + (p_u2 - p_u1 + 1)) as f64 - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Executes a single multi-event simulation trial completely in memory
fn run_single_simulation(trial_id: usize, seed_base: u64, engine: &RhizAeonEngine) -> TrialResult {
    let mut rng = FastRng::new(seed_base + (trial_id as u64).wrapping_mul(7919) + 31);

    // 1. Draw parameters from continuous phase space
    // 33.3% strictly non-recombinant null controls (guaranteeing exact FPR calibration)
    let is_recombinant = trial_id % 3 != 0;

    // Length: log-uniform in [1,000, 30,000]
    let log_l = rng.gen_range_f64((1000.0f64).ln(), (30000.0f64).ln());
    let length = ((log_l.exp() / 50.0).round() * 50.0) as usize;

    // Taxa count: 40% small [4, 30], 40% medium [31, 100], 20% large [101, 250]
    let r_n = rng.next_f64();
    let num_taxa = if r_n < 0.40 {
        rng.gen_range_usize(4, 30)
    } else if r_n < 0.80 {
        rng.gen_range_usize(31, 100)
    } else {
        rng.gen_range_usize(101, 250)
    };

    // Divergence: log-uniform in [0.005, 0.25]
    let log_d = rng.gen_range_f64((0.005f64).ln(), (0.25f64).ln());
    let d_parent = log_d.exp();

    // Heterotachy Gamma alpha: 25% strict clock (alpha = 1000.0), 75% Gamma [0.05, 5.0]
    let heterotachy_alpha = if rng.next_f64() < 0.25 {
        1000.0
    } else {
        let log_a = rng.gen_range_f64((0.05f64).ln(), (5.0f64).ln());
        log_a.exp()
    };

    let transmission_drift = rng.gen_range_f64(0.0005, 0.0035);

    // Site rates
    let site_rates: Option<Vec<f64>> = if heterotachy_alpha < 50.0 {
        let mut rates = Vec::with_capacity(length);
        for _ in 0..length {
            let r = sample_gamma(heterotachy_alpha, &mut rng) / heterotachy_alpha;
            rates.push(r.clamp(0.01, 50.0));
        }
        Some(rates)
    } else {
        None
    };

    // 2. Precompute transition matrices for this trial (10x-50x speedup)
    let p_mats_anc = precompute_p_mats(length, d_parent / 2.0, site_rates.as_deref());
    let p_mats_drift = precompute_p_mats(length, transmission_drift, site_rates.as_deref());

    // 3. Evolve Ancestral Root Sequence
    let mut anc_root = Vec::with_capacity(length);
    for _ in 0..length {
        let p = rng.next_f64();
        let s = if p < PI[0] {
            0u8
        } else if p < PI[0] + PI[1] {
            1u8
        } else if p < PI[0] + PI[1] + PI[2] {
            2u8
        } else {
            3u8
        };
        anc_root.push(s);
    }

    // Parental node ancestors (P1, P2, and optional P3 for cohorts >= 8 taxa)
    let has_p3 = num_taxa >= 8;
    let p1_anc = evolve_with_p_mats(&anc_root, &p_mats_anc, &mut rng);
    let p2_anc = evolve_with_p_mats(&anc_root, &p_mats_anc, &mut rng);
    let p3_anc = if has_p3 {
        evolve_with_p_mats(&anc_root, &p_mats_anc, &mut rng)
    } else {
        Vec::new()
    };

    // Total mutations between P1 and P2
    let mut mut_total = 0;
    for u in 0..length {
        if p1_anc[u] != p2_anc[u] {
            mut_total += 1;
        }
    }

    // 4. Generate events (Multi-event / multi-topology expansion)
    let mut events: Vec<SimEvent> = Vec::new();
    let num_events = if !is_recombinant {
        0
    } else {
        let r = rng.next_f64();
        if r < 0.55 {
            1 // 55% single event
        } else if r < 0.80 {
            2 // 25% dual events
        } else if r < 0.92 {
            3 // 12% triple events
        } else {
            4 // 8% quad events
        }
    };

    if num_events > 0 {
        for e in 0..num_events {
            let seg_start = 1 + (e * length) / num_events;
            let seg_end = ((e + 1) * length) / num_events;
            let seg_len = seg_end.saturating_sub(seg_start);

            // Draw donor clade: P2 (1) or P3 (2)
            let donor_clade = if has_p3 && rng.next_f64() < 0.50 {
                2
            } else {
                1
            };
            let donor_anc = if donor_clade == 1 { &p2_anc } else { &p3_anc };

            // Determine event topology: terminal crossover vs intercalary cassette
            let (u1, u2, is_5prime, is_3prime) = if e == 0 && rng.next_f64() < 0.20 {
                // 5' Terminal Crossover
                let max_u2 = (seg_start + (seg_len * 7 / 10)).min(seg_end.saturating_sub(50));
                let u2 = rng.gen_range_usize(50, max_u2.max(51));
                (1, u2, true, false)
            } else if e == num_events - 1 && rng.next_f64() < 0.20 {
                // 3' Terminal Crossover
                let min_u1 = (seg_start + (seg_len * 3 / 10)).min(length.saturating_sub(60));
                let max_u1 = length.saturating_sub(50);
                let u1 = rng.gen_range_usize(min_u1, max_u1.max(min_u1));
                (u1, length, false, true)
            } else {
                // Intercalary Conversion Cassette
                let min_t = 30;
                let max_t = ((seg_len as f64 * 0.65).round() as usize)
                    .clamp(min_t + 10, seg_len.saturating_sub(40));
                let t_len = if max_t <= min_t {
                    min_t
                } else {
                    let log_t = rng.gen_range_f64((min_t as f64).ln(), (max_t as f64).ln());
                    (log_t.exp().round() as usize).clamp(min_t, max_t)
                };
                let min_u1 = seg_start + 15;
                let max_u1 = seg_end.saturating_sub(t_len + 15);
                let u1 = rng.gen_range_usize(min_u1, max_u1.max(min_u1));
                let u2 = (u1 + t_len - 1).min(seg_end.saturating_sub(5));
                (u1, u2, false, false)
            };

            let mut mut_tr = 0;
            for u in (u1 - 1)..u2 {
                if p1_anc[u] != donor_anc[u] {
                    mut_tr += 1;
                }
            }

            events.push(SimEvent {
                u1,
                u2,
                donor_clade,
                is_5prime_crossover: is_5prime,
                is_3prime_crossover: is_3prime,
                mut_in_tract: mut_tr,
            });
        }
    }

    // Swarm size K
    let (k_true, n_bg_total) = if is_recombinant && !events.is_empty() {
        let max_k = ((num_taxa as f64 * 0.35).floor() as usize).max(1);
        let k = rng.gen_range_usize(1, max_k);
        (k, num_taxa.saturating_sub(k))
    } else {
        (0, num_taxa)
    };

    let mut taxa_names = Vec::with_capacity(num_taxa);
    let mut full_matrix = Vec::with_capacity(num_taxa * length);

    // Split background taxa across clades
    let (n_p1, n_p2, n_p3) = if has_p3 {
        let p1 = (n_bg_total / 3).max(1);
        let p2 = (n_bg_total / 3).max(1);
        let p3 = n_bg_total.saturating_sub(p1 + p2).max(1);
        (p1, p2, p3)
    } else {
        let p1 = (n_bg_total / 2).max(1);
        let p2 = n_bg_total.saturating_sub(p1).max(1);
        (p1, p2, 0)
    };

    // Generate P1 background taxa
    for i in 0..n_p1 {
        taxa_names.push(format!("P1_{}", i));
        let leaf = evolve_with_p_mats(&p1_anc, &p_mats_drift, &mut rng);
        for s in leaf {
            full_matrix.push(s + 1);
        }
    }

    // Generate P2 background taxa
    for i in 0..n_p2 {
        taxa_names.push(format!("P2_{}", i));
        let leaf = evolve_with_p_mats(&p2_anc, &p_mats_drift, &mut rng);
        for s in leaf {
            full_matrix.push(s + 1);
        }
    }

    // Generate P3 background taxa (if applicable)
    for i in 0..n_p3 {
        taxa_names.push(format!("P3_{}", i));
        let leaf = evolve_with_p_mats(&p3_anc, &p_mats_drift, &mut rng);
        for s in leaf {
            full_matrix.push(s + 1);
        }
    }

    // Generate Recombinant taxa if applicable
    if is_recombinant && k_true > 0 && !events.is_empty() {
        let mut rec_anc = p1_anc.clone();
        for ev in &events {
            let donor_anc = if ev.donor_clade == 1 {
                &p2_anc
            } else {
                &p3_anc
            };
            for u in (ev.u1 - 1)..ev.u2 {
                rec_anc[u] = donor_anc[u];
            }
        }

        for k in 0..k_true {
            taxa_names.push(format!("Rec_{}", k));
            let leaf = evolve_with_p_mats(&rec_anc, &p_mats_drift, &mut rng);
            for s in leaf {
                full_matrix.push(s + 1);
            }
        }
    }

    let actual_taxa = taxa_names.len();
    let aln = Alignment::new(taxa_names, length, full_matrix);

    // 5. Run RhizAeon Engine
    let t0 = Instant::now();
    let scan_res = engine.scan(&aln);
    let run_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // 6. Score Taxon Attribution
    let mut rec_taxa_called = std::collections::HashSet::new();
    let mut bg_taxa_called = std::collections::HashSet::new();

    let unique_events_inferred = scan_res.events.len();
    let detected = if unique_events_inferred > 0 { 1u8 } else { 0u8 };

    for ev in &scan_res.events {
        for iso in &ev.isolates {
            if iso.starts_with("Rec_") {
                rec_taxa_called.insert(iso.clone());
            } else {
                bg_taxa_called.insert(iso.clone());
            }
        }
    }

    let recall = if is_recombinant && k_true > 0 {
        rec_taxa_called.len() as f64 / k_true as f64
    } else {
        if detected == 0 {
            1.0
        } else {
            0.0
        }
    };

    let total_called = rec_taxa_called.len() + bg_taxa_called.len();
    let precision = if is_recombinant && k_true > 0 {
        if total_called > 0 {
            rec_taxa_called.len() as f64 / total_called as f64
        } else {
            1.0
        }
    } else {
        if detected == 0 {
            1.0
        } else {
            0.0
        }
    };

    let bg_fp_count = bg_taxa_called.len();
    let bg_fpr = if n_bg_total > 0 {
        bg_fp_count as f64 / n_bg_total as f64
    } else {
        0.0
    };

    // 7. Bipartite Greedy Matching for Event-Level Metrics
    let n_true_events = events.len();
    let (
        event_recall,
        event_precision,
        event_f1,
        mean_jaccard,
        mean_bp_error,
        best_jaccard,
        best_bp1_err,
        best_bp2_err,
    ) = if n_true_events == 0 {
        if unique_events_inferred == 0 {
            (1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0, 0)
        } else {
            (1.0, 0.0, 0.0, 0.0, length as f64, 0.0, length, length)
        }
    } else {
        if unique_events_inferred == 0 {
            (0.0, 1.0, 0.0, 0.0, length as f64, 0.0, length, length)
        } else {
            // Build pair candidates
            struct MatchPair {
                t_idx: usize,
                i_idx: usize,
                jaccard: f64,
                bp_err: f64,
                bp1_err: usize,
                bp2_err: usize,
            }
            let mut pairs = Vec::with_capacity(n_true_events * unique_events_inferred);
            for (t_idx, tev) in events.iter().enumerate() {
                for (i_idx, iev) in scan_res.events.iter().enumerate() {
                    let j = compute_jaccard(tev.u1, tev.u2, iev.u1, iev.u2);
                    let e1 = iev.u1.abs_diff(tev.u1);
                    let e2 = iev.u2.abs_diff(tev.u2);
                    pairs.push(MatchPair {
                        t_idx,
                        i_idx,
                        jaccard: j,
                        bp_err: (e1 + e2) as f64 / 2.0,
                        bp1_err: e1,
                        bp2_err: e2,
                    });
                }
            }
            // Sort descending by Jaccard
            pairs.sort_by(|a, b| {
                b.jaccard
                    .partial_cmp(&a.jaccard)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut matched_t = vec![false; n_true_events];
            let mut matched_i = vec![false; unique_events_inferred];
            let mut sum_jaccard = 0.0;
            let mut sum_bp_err = 0.0;
            let mut n_matched = 0;
            let mut best_j = 0.0;
            let mut b_bp1 = length;
            let mut b_bp2 = length;

            for p in &pairs {
                if !matched_t[p.t_idx] && !matched_i[p.i_idx] && p.jaccard > 0.0 {
                    matched_t[p.t_idx] = true;
                    matched_i[p.i_idx] = true;
                    sum_jaccard += p.jaccard;
                    sum_bp_err += p.bp_err;
                    n_matched += 1;
                    if p.jaccard > best_j {
                        best_j = p.jaccard;
                        b_bp1 = p.bp1_err;
                        b_bp2 = p.bp2_err;
                    }
                }
            }

            let ev_rec = n_matched as f64 / n_true_events as f64;
            let ev_prec = n_matched as f64 / unique_events_inferred as f64;
            let ev_f1 = if ev_rec + ev_prec > 0.0 {
                2.0 * ev_rec * ev_prec / (ev_rec + ev_prec)
            } else {
                0.0
            };
            let m_j = if n_matched > 0 {
                sum_jaccard / n_matched as f64
            } else {
                0.0
            };
            let m_err = if n_matched > 0 {
                sum_bp_err / n_matched as f64
            } else {
                length as f64
            };
            (ev_rec, ev_prec, ev_f1, m_j, m_err, best_j, b_bp1, b_bp2)
        }
    };

    let cardinality_error = unique_events_inferred as i64 - n_true_events as i64;
    let total_tract_len: usize = events.iter().map(|e| e.u2 - e.u1 + 1).sum();
    let total_mut_in_tract: usize = events.iter().map(|e| e.mut_in_tract).sum();

    let u1_primary = events.first().map(|e| e.u1).unwrap_or(0);
    let u2_primary = events.first().map(|e| e.u2).unwrap_or(0);

    let has_crossover = if events
        .iter()
        .any(|e| e.is_5prime_crossover || e.is_3prime_crossover)
    {
        1u8
    } else {
        0u8
    };
    let has_cassette = if events
        .iter()
        .any(|e| !e.is_5prime_crossover && !e.is_3prime_crossover)
    {
        1u8
    } else {
        0u8
    };
    let has_multi_donor = if events.len() >= 2
        && events
            .iter()
            .map(|e| e.donor_clade)
            .collect::<std::collections::HashSet<_>>()
            .len()
            >= 2
    {
        1u8
    } else {
        0u8
    };

    TrialResult {
        trial_id,
        is_recombinant_true: if is_recombinant && n_true_events > 0 {
            1
        } else {
            0
        },
        length,
        num_taxa: actual_taxa,
        d_parent,
        tract_len: total_tract_len,
        k_true,
        u1_true: u1_primary,
        u2_true: u2_primary,
        mut_in_tract: total_mut_in_tract,
        mut_total,
        heterotachy_alpha,
        informative_sites: scan_res.informative_sites,
        detected,
        unique_events_inferred,
        recall,
        precision,
        bg_fp_count,
        bg_fpr,
        jaccard: if is_recombinant && n_true_events > 0 {
            best_jaccard
        } else {
            if detected == 0 {
                1.0
            } else {
                0.0
            }
        },
        bp1_error: if is_recombinant && detected > 0 {
            best_bp1_err
        } else {
            0
        },
        bp2_error: if is_recombinant && detected > 0 {
            best_bp2_err
        } else {
            0
        },
        run_time_ms,
        num_events_true: n_true_events,
        cardinality_error,
        event_recall,
        event_precision,
        event_f1,
        mean_jaccard,
        mean_bp_error,
        has_crossover,
        has_cassette,
        has_multi_donor,
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "grand_sweep",
    author = "Sergei L. Kosakovsky Pond & Antigravity AI",
    version = "2.0",
    about = "Grand-Scale Parameter Sweep Harness for RhizAeon (100k-1M Multi-Event In-Memory Simulations)"
)]
struct Args {
    #[arg(short, long, default_value_t = 1000000)]
    trials: usize,

    #[arg(long, default_value_t = 0)]
    start_trial: usize,

    #[arg(short, long, default_value = "grand_sweep_results.csv")]
    output: String,

    #[arg(short = 'j', long, default_value_t = 0)]
    threads: usize,

    #[arg(short, long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap_or(());
    }

    let num_threads = rayon::current_num_threads();

    println!("================================================================================");
    println!("     RhizAeon Grand-Scale Parameter Sweep Harness (1,000,000 Genomes)           ");
    println!("================================================================================");
    println!("Total Trials:         {}", args.trials);
    println!("Start Trial ID:       {}", args.start_trial);
    println!(
        "Trial ID Range:       {} .. {}",
        args.start_trial,
        args.start_trial + args.trials
    );
    println!("Worker Threads:       {}", num_threads);
    println!("Output File:          {}", args.output);
    println!("Random Seed:          {}", args.seed);
    println!("--------------------------------------------------------------------------------");

    let engine = RhizAeonEngine::new();
    let file = File::create(&args.output)?;
    let writer = Mutex::new(BufWriter::with_capacity(4 * 1024 * 1024, file));

    // Write CSV header
    {
        let mut w = writer.lock().unwrap();
        writeln!(
            w,
            "trial_id,is_recombinant_true,length,num_taxa,d_parent,tract_len,k_true,u1_true,u2_true,mut_in_tract,mut_total,heterotachy_alpha,informative_sites,detected,unique_events_inferred,recall,precision,bg_fp_count,bg_fpr,jaccard,bp1_error,bp2_error,run_time_ms,num_events_true,cardinality_error,event_recall,event_precision,event_f1,mean_jaccard,mean_bp_error,has_crossover,has_cassette,has_multi_donor"
        )?;
        w.flush()?;
    }

    let progress_counter = AtomicUsize::new(0);
    let total_recombinant_detected = AtomicUsize::new(0);
    let total_null_clean = AtomicUsize::new(0);
    let t_start = Instant::now();

    let trials: Vec<usize> = (args.start_trial..(args.start_trial + args.trials)).collect();

    // Process in parallel chunks to stream output efficiently across all cores
    trials.par_chunks(16).for_each(|chunk| {
        let mut local_csv = String::with_capacity(chunk.len() * 180);
        let mut rec_det_chunk = 0;
        let mut null_clean_chunk = 0;

        for &trial_id in chunk {
            let res = run_single_simulation(trial_id, args.seed, &engine);
            local_csv.push_str(&res.to_csv_row());
            local_csv.push('\n');

            if res.is_recombinant_true == 1 && res.detected == 1 {
                rec_det_chunk += 1;
            } else if res.is_recombinant_true == 0 && res.bg_fp_count == 0 {
                null_clean_chunk += 1;
            }
        }

        // Flush chunk to buffered writer
        {
            let mut w = writer.lock().unwrap();
            w.write_all(local_csv.as_bytes()).unwrap();
            let _ = w.flush();
        }

        total_recombinant_detected.fetch_add(rec_det_chunk, Ordering::Relaxed);
        total_null_clean.fetch_add(null_clean_chunk, Ordering::Relaxed);

        let completed = progress_counter.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
        if completed % 1000 == 0 || completed >= args.trials {
            let elapsed = t_start.elapsed().as_secs_f64();
            let rate = completed as f64 / elapsed;
            let remaining = (args.trials.saturating_sub(completed)) as f64 / rate;
            println!(
                "  Progress: {:6}/{} ({:5.1}%) | Rate: {:6.1} trials/s | Elapsed: {:5.1}s | ETA: {:5.1}s",
                completed,
                args.trials,
                (completed as f64 / args.trials as f64) * 100.0,
                rate,
                elapsed,
                remaining
            );
            use std::io::stdout;
            let _ = stdout().flush();
        }
    });

    let total_elapsed = t_start.elapsed().as_secs_f64();
    let rec_det = total_recombinant_detected.load(Ordering::Relaxed);
    let null_clean = total_null_clean.load(Ordering::Relaxed);

    println!("================================================================================");
    println!("                         GRAND SWEEP COMPLETE                                   ");
    println!("================================================================================");
    println!(
        "Total Execution Time:      {:.2} seconds ({:.1} minutes)",
        total_elapsed,
        total_elapsed / 60.0
    );
    println!(
        "Average Throughput:        {:.1} trials/second",
        args.trials as f64 / total_elapsed
    );
    println!("Recombinant Trials Detected: {}", rec_det);
    println!("Non-Recombinant Nulls Clean: {}", null_clean);
    println!("Results saved to:          {}", args.output);
    println!("================================================================================");

    Ok(())
}
