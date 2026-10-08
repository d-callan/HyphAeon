use crate::attribution::TrajectoryParentalPair;
use crate::types::{Alignment, LandmarkSet, PhysicalTract, RecombinationEvent};

pub struct SequenceDueDiligence {
    pub poisson_floor: usize,
    pub alpha: f64,
    log_factorials: Vec<f64>,
}

impl Default for SequenceDueDiligence {
    fn default() -> Self {
        Self::new(3, 0.05, 100_000)
    }
}

impl SequenceDueDiligence {
    pub fn new(poisson_floor: usize, alpha: f64, max_fact: usize) -> Self {
        let mut log_facts = Vec::with_capacity(max_fact + 1);
        log_facts.push(0.0); // 0!
        let mut sum_ln = 0.0;
        for i in 1..=max_fact {
            sum_ln += (i as f64).ln();
            log_facts.push(sum_ln);
        }
        Self {
            poisson_floor,
            alpha,
            log_factorials: log_facts,
        }
    }

    #[inline(always)]
    fn log_factorial(&self, n: usize) -> f64 {
        if n < self.log_factorials.len() {
            self.log_factorials[n]
        } else {
            // Stirling's approximation
            let nf = n as f64;
            0.5 * (2.0 * std::f64::consts::PI * nf).ln() + nf * (nf / std::f64::consts::E).ln()
        }
    }

    /// Exact Fisher 2x2 one-sided test for table:
    /// [a, b]
    /// [c, d]
    /// Testing if a/(a+b) > c/(c+d)
    pub fn fisher_exact_2x2(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        let n = a + b + c + d;
        if n == 0 {
            return 1.0;
        }

        let r1 = a + b;
        let r2 = c + d;
        let c1 = a + c;
        let c2 = b + d;

        let log_denom = self.log_factorial(n);
        let log_margins = self.log_factorial(r1) + self.log_factorial(r2)
            + self.log_factorial(c1) + self.log_factorial(c2);

        let max_k = r1.min(c1);
        let mut p_sum = 0.0;

        // Sum hypergeometric probabilities for k >= a
        for k in a..=max_k {
            let cell_b = r1 - k;
            let cell_c = c1 - k;
            let cell_d = r2 - cell_c;

            let log_cells = self.log_factorial(k) + self.log_factorial(cell_b)
                + self.log_factorial(cell_c) + self.log_factorial(cell_d);

            let log_prob = log_margins - log_denom - log_cells;
            p_sum += log_prob.exp();
        }

        p_sum.min(1.0).max(0.0)
    }

    /// Verifies physical tract via orthogonal sequence due diligence
    pub fn verify_tract(
        &self,
        tract: &PhysicalTract,
        parental: &TrajectoryParentalPair,
        aln: &Alignment,
        _landmarks: &LandmarkSet,
    ) -> Option<RecombinationEvent> {
        let cand_idx = parental.candidate_idx;
        let cand_name = aln.taxa[cand_idx].clone();
        let l = aln.length;

        let u1 = tract.u1_discrete;
        let u2 = tract.u2_discrete;
        let tract_len = if u2 >= u1 { u2 - u1 + 1 } else { 0 };

        if tract_len < 20 || !tract.is_physically_significant {
            return None;
        }

        // Bounded cassette check for ghost donors: reject full genome spans
        if parental.is_ghost_donor && (u1 <= 5 && u2 >= l - 5) {
            return None;
        }

        let home_global = parental.home_global_idx.unwrap_or(0);
        let cand_row = aln.row(cand_idx);
        let home_row = aln.row(home_global);

        let (donor_row_opt, donor_name) = if parental.is_ghost_donor {
            (None, "Ghost".to_string())
        } else if let Some(dg) = parental.donor_global_idx {
            (Some(aln.row(dg)), parental.donor_name.clone())
        } else {
            (None, "Ghost".to_string())
        };

        // Site counters
        let mut s_donor = 0usize;
        let mut s_home_flank = 0usize;
        let mut mismatch_tract_donor = 0usize;
        let mut mismatch_tract_home = 0usize;
        let mut valid_tract = 0usize;

        let mut mismatch_flank_donor = 0usize;
        let mut mismatch_flank_home = 0usize;
        let mut valid_flank = 0usize;

        let s_idx = u1 - 1;
        let e_idx = u2;

        for u in 0..l {
            let rc = cand_row[u];
            let rh = home_row[u];
            let in_tract = u >= s_idx && u < e_idx;

            if let Some(donor_row) = donor_row_opt {
                let rd = donor_row[u];
                if rc > 0 && rh > 0 && rd > 0 {
                    if in_tract {
                        valid_tract += 1;
                        if rc != rd { mismatch_tract_donor += 1; }
                        if rc != rh { mismatch_tract_home += 1; }
                        // Informative site: candidate matches donor while home differs
                        if rc == rd && rh != rd {
                            s_donor += 1;
                        }
                    } else {
                        valid_flank += 1;
                        if rc != rd { mismatch_flank_donor += 1; }
                        if rc != rh { mismatch_flank_home += 1; }
                        // Flank informative site: candidate matches home while donor differs
                        if rc == rh && rd != rh {
                            s_home_flank += 1;
                        }
                    }
                }
            } else {
                // Ghost donor: test against home divergence
                if rc > 0 && rh > 0 {
                    if in_tract {
                        valid_tract += 1;
                        if rc != rh {
                            mismatch_tract_home += 1;
                            s_donor += 1;
                        }
                    } else {
                        valid_flank += 1;
                        if rc != rh {
                            mismatch_flank_home += 1;
                        } else {
                            s_home_flank += 1;
                        }
                    }
                }
            }
        }

        // Informative site floor check
        if s_donor < self.poisson_floor {
            return None;
        }

        // Private Autapomorphy Ratio Sieve (DIR-RHIZ-PAFF-012.8 Requirement 1.3)
        // Rejects lineage-specific substitution rate acceleration (Darren's heterotachy)
        if aln.num_taxa >= 3 {
            let mut n_priv = 0usize;
            let mut n_total_mut = 0usize;

            for u in s_idx..e_idx {
                let rc = cand_row[u];
                let rh = home_row[u];
                if rc > 0 && rh > 0 && rc != rh {
                    n_total_mut += 1;
                    let mut is_private = true;
                    for t_idx in 0..aln.num_taxa {
                        if t_idx != cand_idx && aln.row(t_idx)[u] == rc {
                            is_private = false;
                            break;
                        }
                    }
                    if is_private {
                        n_priv += 1;
                    }
                }
            }

            if n_total_mut >= 4 {
                let rho_priv = (n_priv as f64) / (n_total_mut as f64);
                if rho_priv >= 0.50 && n_priv >= 3 {
                    // Heterotachy / autapomorphic rate burst:
                    // If >= 50% of mutations in the tract are private to the candidate alone
                    // (not shared with the candidate donor or any other taxon), it represents
                    // terminal branch rate variation, not recombination.
                    return None;
                }
            }
        }

        // Fisher exact test contingency table
        let a = s_donor;
        let b = mismatch_tract_donor.saturating_sub(s_donor);
        let c = mismatch_flank_home.saturating_sub(s_home_flank);
        let d = s_home_flank;

        let p_fisher = self.fisher_exact_2x2(a, b, c, d);

        // Bonferroni critical threshold over all candidate-donor pairs in alignment
        let num_pairs = ((aln.num_taxa * (aln.num_taxa - 1) / 2).max(1)) as f64;
        let p_crit = self.alpha / num_pairs;

        let d_tract_donor = if valid_tract > 0 { (mismatch_tract_donor as f64) / (valid_tract as f64) } else { 0.0 };
        let d_tract_home = if valid_tract > 0 { (mismatch_tract_home as f64) / (valid_tract as f64) } else { 0.0 };
        let d_flank_donor = if valid_flank > 0 { (mismatch_flank_donor as f64) / (valid_flank as f64) } else { 0.0 };
        let d_flank_home = if valid_flank > 0 { (mismatch_flank_home as f64) / (valid_flank as f64) } else { 0.0 };

        // Relative recoil check (when donor is sampled)
        if !parental.is_ghost_donor {
            let delta_donor_recoil = d_flank_donor - d_tract_donor;
            let delta_home_recoil = d_tract_home - d_flank_home;
            let delta_net_recoil = delta_donor_recoil + delta_home_recoil;

            if delta_donor_recoil <= 0.0 {
                return None;
            }

            let n_tr = valid_tract.max(1) as f64;
            let n_fl = valid_flank.max(1) as f64;
            let se_recoil = ((d_tract_donor * (1.0 - d_tract_donor) / n_tr)
                + (d_flank_donor * (1.0 - d_flank_donor) / n_fl))
                .sqrt();
            let z_crit_recoil = crate::screener::normal_ppf(1.0 - self.alpha);
            if delta_net_recoil < (z_crit_recoil * se_recoil).max(0.0) {
                return None;
            }

            let ratio_donor = (d_tract_donor + 1e-4) / (d_flank_donor + 1e-4);
            let ratio_home = (d_tract_home + 1e-4) / (d_flank_home + 1e-4);
            if ratio_donor >= ratio_home {
                return None;
            }

            // Buneman 4-Point Metric and Topological Invariant (DIR-RHIZ-PAFF-013.4 / 013.7)
            if aln.num_taxa >= 4 {
                let donor_global = parental.donor_global_idx.unwrap();
                let mut best_outgroup = None;
                let mut max_out_div = -1.0;

                for o_idx in 0..aln.num_taxa {
                    if o_idx != cand_idx && o_idx != home_global && o_idx != donor_global {
                        let d_oh = crate::landmarks::compute_pairwise_distance(aln, o_idx, home_global);
                        let d_od = crate::landmarks::compute_pairwise_distance(aln, o_idx, donor_global);
                        let sum_d = d_oh + d_od;
                        if sum_d > max_out_div {
                            max_out_div = sum_d;
                            best_outgroup = Some(o_idx);
                        }
                    }
                }

                if let Some(o_idx) = best_outgroup {
                    let o_row = aln.row(o_idx);
                    let mut d_tr_oh = 0usize;
                    let mut d_tr_od = 0usize;
                    let mut d_tr_ro = 0usize;
                    let mut d_tr_hd = 0usize;
                    let mut v_tr_o = 0usize;

                    let mut d_fl_oh = 0usize;
                    let mut d_fl_od = 0usize;
                    let mut d_fl_ro = 0usize;
                    let mut d_fl_hd = 0usize;
                    let mut v_fl_o = 0usize;

                    for u in 0..l {
                        let ro = o_row[u];
                        let rh = home_row[u];
                        let rd = donor_row_opt.unwrap()[u];
                        let rc = cand_row[u];

                        if ro > 0 && rh > 0 && rd > 0 && rc > 0 {
                            if u >= s_idx && u < e_idx {
                                v_tr_o += 1;
                                if ro != rh { d_tr_oh += 1; }
                                if ro != rd { d_tr_od += 1; }
                                if rc != ro { d_tr_ro += 1; }
                                if rh != rd { d_tr_hd += 1; }
                            } else {
                                v_fl_o += 1;
                                if ro != rh { d_fl_oh += 1; }
                                if ro != rd { d_fl_od += 1; }
                                if rc != ro { d_fl_ro += 1; }
                                if rh != rd { d_fl_hd += 1; }
                            }
                        }
                    }

                    if v_tr_o > 0 && v_fl_o > 0 {
                        let tr_oh = (d_tr_oh as f64) / (v_tr_o as f64);
                        let tr_od = (d_tr_od as f64) / (v_tr_o as f64);
                        let tr_ro = (d_tr_ro as f64) / (v_tr_o as f64);
                        let tr_hd = (d_tr_hd as f64) / (v_tr_o as f64);

                        let fl_oh = (d_fl_oh as f64) / (v_fl_o as f64);
                        let fl_od = (d_fl_od as f64) / (v_fl_o as f64);
                        let fl_ro = (d_fl_ro as f64) / (v_fl_o as f64);
                        let fl_hd = (d_fl_hd as f64) / (v_fl_o as f64);

                        // Unrooted quartet topology check (DIR-RHIZ-PAFF-012.7):
                        // S1 = D(R, H) + D(D, O)  [Split 0: (RH)|(DO)]
                        // S2 = D(R, D) + D(H, O)  [Split 1: (RD)|(HO)]
                        // S3 = D(R, O) + D(H, D)  [Split 2: (RO)|(HD)]
                        let s1_tr = d_tract_home + tr_od;
                        let s2_tr = d_tract_donor + tr_oh;
                        let s3_tr = tr_ro + tr_hd;

                        let s1_fl = d_flank_home + fl_od;
                        let s2_fl = d_flank_donor + fl_oh;
                        let s3_fl = fl_ro + fl_hd;

                        let topo_tr = if s1_tr <= s2_tr && s1_tr <= s3_tr { 0 } else if s2_tr <= s3_tr { 1 } else { 2 };
                        let topo_fl = if s1_fl <= s2_fl && s1_fl <= s3_fl { 0 } else if s2_fl <= s3_fl { 1 } else { 2 };

                        if topo_tr == topo_fl {
                            return None;
                        }
                        if topo_tr == 0 {
                            return None;
                        }

                        // Differential Buneman Polarization:
                        let delta_q_flank = (d_flank_donor + fl_oh) - (d_flank_home + fl_od);
                        let delta_q_tract = (d_tract_donor + tr_oh) - (d_tract_home + tr_od);
                        let delta_q_diff = delta_q_tract - delta_q_flank;

                        // Under terminal heterotachy / rate burst, delta_q_diff >= 0.0
                        if delta_q_diff >= 0.0 {
                            return None;
                        }

                        // Topological Double Difference:
                        let delta_flank = (d_flank_home - d_flank_donor) - (fl_oh - fl_od);
                        let delta_tract = (d_tract_home - d_tract_donor) - (tr_oh - tr_od);
                        let delta_topo = (delta_tract - delta_flank).abs();

                        let se_topo = ((d_tract_home * (1.0 - d_tract_home) / (v_tr_o as f64))
                            + (d_flank_home * (1.0 - d_flank_home) / (v_fl_o as f64)))
                            .sqrt();
                        let resolution_floor = 1.0 / ((v_tr_o.min(v_fl_o)) as f64);
                        let z_crit_topo = crate::screener::normal_ppf(1.0 - self.alpha);
                        let thresh = (z_crit_topo * se_topo).max(resolution_floor);

                        if delta_topo < thresh {
                            return None;
                        }
                    }
                }
            }
        } else {
            // For ghost donor, tract divergence to home must strictly exceed flank divergence to home
            if d_tract_home <= d_flank_home {
                return None;
            }
        }


        let is_verified = p_fisher <= p_crit && s_donor >= self.poisson_floor;

        Some(RecombinationEvent {
            candidate_idx: cand_idx,
            candidate_name: cand_name,
            home_idx: parental.home_global_idx,
            home_name: parental.home_name.clone(),
            donor_idx: parental.donor_global_idx,
            donor_name,
            is_ghost_donor: parental.is_ghost_donor,
            is_crossover: tract.is_crossover,
            u1,
            u2,
            u1_continuous: tract.u1_continuous,
            u2_continuous: tract.u2_continuous,
            tract_length: tract_len,
            s_informative: s_donor,
            p_fisher,
            z_phys: tract.z_phys,
            d_flank_home,
            d_flank_donor,
            d_tract_home,
            d_tract_donor,
            is_verified,
        })
    }

    /// Adjudicates reticulation graph to eliminate reflected mosaic illusions (DIR-RHIZ-PAFF-013.1)
    pub fn adjudicate_reticulation_graph(
        &self,
        events: Vec<RecombinationEvent>,
        _aln: &Alignment,
        _screening: &crate::types::ScreeningResult,
    ) -> Vec<RecombinationEvent> {
        if events.len() <= 1 {
            return events;
        }

        let n_events = events.len();
        let mut pruned = vec![false; n_events];

        // Pass 1: Mutual 2-Cycle Adjudication (C_1 <-> C_2)
        // Resolves cases where candidate C1 cites C2 as parent, and candidate C2 cites C1 as parent.
        for i in 0..n_events {
            if pruned[i] {
                continue;
            }
            let e1 = &events[i];
            let c1 = e1.candidate_idx;

            for j in (i + 1)..n_events {
                if pruned[j] {
                    continue;
                }
                let e2 = &events[j];
                let c2 = e2.candidate_idx;

                if c1 == c2 {
                    continue;
                }

                // Check if e1 cites c2 (as Home or Donor)
                let e1_cites_c2 = (e1.home_idx == Some(c2)) || (e1.donor_idx == Some(c2));
                // Check if e2 cites c1 (as Home or Donor)
                let e2_cites_c1 = (e2.home_idx == Some(c1)) || (e2.donor_idx == Some(c1));

                if e1_cites_c2 && e2_cites_c1 {
                    // Mutual 2-cycle detected!
                    // Check external donor authenticity:
                    let e1_has_ext_donor = e1.donor_idx.is_some() && e1.donor_idx != Some(c2);
                    let e2_has_ext_donor = e2.donor_idx.is_some() && e2.donor_idx != Some(c1);

                    if e1_has_ext_donor && !e2_has_ext_donor {
                        // e1 has authentic external donor, while e2's donor is its cyclic partner
                        pruned[j] = true;
                    } else if e2_has_ext_donor && !e1_has_ext_donor {
                        // e2 has authentic external donor, while e1's donor is its cyclic partner
                        pruned[i] = true;
                        break;
                    } else {
                        // Compute physical transition contrast and home backbone disruption for each candidate:
                        // A true recombinant child undergoes a large home disruption (d_tract_home >> d_flank_home),
                        // moving away from its home clade into the donor clade during the tract.
                        // A stationary clonal parent never leaves its home clade (d_tract_home ~ d_flank_home).
                    let delta_h1 = (e1.d_tract_home - e1.d_flank_home).max(0.0);
                    let delta_h2 = (e2.d_tract_home - e2.d_flank_home).max(0.0);
                    let delta_d1 = (e1.d_flank_donor - e1.d_tract_donor).max(0.0);
                    let delta_d2 = (e2.d_flank_donor - e2.d_tract_donor).max(0.0);
                    let trans1 = delta_h1 + delta_d1;
                    let trans2 = delta_h2 + delta_d2;

                    if (trans1 - trans2).abs() > 0.02 {
                        if trans1 > trans2 {
                            pruned[j] = true;
                        } else {
                            pruned[i] = true;
                            break;
                        }
                    } else if (delta_h1 - delta_h2).abs() > 0.02 {
                        if delta_h1 > delta_h2 {
                            pruned[j] = true;
                        } else {
                            pruned[i] = true;
                            break;
                        }
                    } else {
                        // Tie-break by mosaic reconstruction residual, z_phys, and informative sites
                        let res1 = e1.d_tract_donor + e1.d_flank_home;
                        let res2 = e2.d_tract_donor + e2.d_flank_home;
                        if (res1 - res2).abs() > 1e-4 {
                            if res1 < res2 {
                                pruned[j] = true;
                            } else {
                                pruned[i] = true;
                                break;
                            }
                        } else if e1.z_phys != e2.z_phys {
                            if e1.z_phys > e2.z_phys {
                                pruned[j] = true;
                            } else {
                                pruned[i] = true;
                                break;
                            }
                        } else if e1.s_informative >= e2.s_informative {
                            pruned[j] = true;
                        } else {
                            pruned[i] = true;
                            break;
                        }
                    }
                }
            }
        }
        }

        // Pass 2: Iterative Recombinant Donor & Home Pruning
        // Eliminate spurious events that cite confirmed recombinant children as their Donor or Home.
        let mut changed = true;
        while changed {
            changed = false;

            let mut verified_recombinants = std::collections::HashSet::new();
            for (idx, e) in events.iter().enumerate() {
                if !pruned[idx] {
                    verified_recombinants.insert(e.candidate_idx);
                }
            }

            for (idx, e) in events.iter().enumerate() {
                if pruned[idx] {
                    continue;
                }

                // Rule 2: Recombinant Donor Pruning
                // A contemporary recombinant child cannot be the donor of an ancestral/reference lineage.
                if let Some(donor) = e.donor_idx {
                    if verified_recombinants.contains(&donor) && donor != e.candidate_idx {
                        pruned[idx] = true;
                        changed = true;
                        continue;
                    }
                }

                // Rule 3: Recombinant Home Pruning
                // Any candidate event citing an active verified recombinant as Home is pruned if:
                // 1. The recombinant derived its Home from the candidate (or candidate's sister clade).
                // 2. The candidate's mosaic residual is poor (res >= 0.04 or d_tract_donor >= 0.03).
                if let Some(home) = e.home_idx {
                    if verified_recombinants.contains(&home) && home != e.candidate_idx {
                        let home_derived_from_cand = events.iter().enumerate().any(|(h_idx, h_ev)| {
                            !pruned[h_idx]
                                && h_ev.candidate_idx == home
                                && (h_ev.home_idx == Some(e.candidate_idx) || h_ev.donor_idx == Some(e.candidate_idx))
                        });

                        let res = e.d_tract_donor + e.d_flank_home;
                        if home_derived_from_cand || res >= 0.04 || e.d_tract_donor >= 0.03 {
                            pruned[idx] = true;
                            changed = true;
                            continue;
                        }
                    }
                }
            }
        }

        events
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !pruned[*idx])
            .map(|(_, e)| e)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_exact_2x2_basic() {
        let dd = SequenceDueDiligence::new(3, 0.05, 1000);
        // Table:
        // [10, 0]
        // [ 0, 10]
        // Strongly significant
        let p = dd.fisher_exact_2x2(10, 0, 0, 10);
        assert!(p < 1e-4);

        // Null table:
        // [5, 5]
        // [5, 5]
        let p_null = dd.fisher_exact_2x2(5, 5, 5, 5);
        assert!(p_null > 0.4);
    }

    #[test]
    fn test_adjudicate_reticulation_graph_cycle() {
        let dd = SequenceDueDiligence::new(3, 0.05, 1000);
        let aln = Alignment::new(
            vec!["R".into(), "H".into(), "D".into()],
            1000,
            vec![1; 3000],
        );

        let screening = crate::types::ScreeningResult {
            v_cont: vec![5.0, 3.0, 1.0],
            v_root: vec![0.0; 3],
            r_root: vec![0.0; 3],
            gamma_root: vec![0.0; 3],
            rate_burst_indices: vec![],
            ghost_candidate_indices: vec![],
            candidate_recombinant_indices: vec![0, 1],
            clonal_backbone_indices: vec![2],
            median_v_cont: 3.0,
            is_non_recombinant_h0: false,
        };

        // Event 1: R with Home H (1), Donor D (2)
        let e1 = RecombinationEvent {
            candidate_idx: 0,
            candidate_name: "R".into(),
            home_idx: Some(1),
            home_name: "H".into(),
            donor_idx: Some(2),
            donor_name: "D".into(),
            is_ghost_donor: false,
            is_crossover: true,
            u1: 1,
            u2: 500,
            u1_continuous: 1.0,
            u2_continuous: 500.0,
            tract_length: 500,
            s_informative: 50,
            p_fisher: 1e-20,
            z_phys: 10.0,
            d_flank_home: 0.01,
            d_flank_donor: 0.10,
            d_tract_home: 0.10,
            d_tract_donor: 0.01,
            is_verified: true,
        };

        // Event 2: H with Home D (2), Donor R (0) [Reflected illusion!]
        let e2 = RecombinationEvent {
            candidate_idx: 1,
            candidate_name: "H".into(),
            home_idx: Some(2),
            home_name: "D".into(),
            donor_idx: Some(0),
            donor_name: "R".into(),
            is_ghost_donor: false,
            is_crossover: true,
            u1: 501,
            u2: 1000,
            u1_continuous: 501.0,
            u2_continuous: 1000.0,
            tract_length: 500,
            s_informative: 50,
            p_fisher: 1e-20,
            z_phys: 8.0,
            d_flank_home: 0.10,
            d_flank_donor: 0.10,
            d_tract_home: 0.10,
            d_tract_donor: 0.01,
            is_verified: true,
        };

        let adjudicated = dd.adjudicate_reticulation_graph(vec![e1, e2], &aln, &screening);
        assert_eq!(adjudicated.len(), 1);
        assert_eq!(adjudicated[0].candidate_name, "R");
        assert_eq!(adjudicated[0].home_name, "H");
        assert_eq!(adjudicated[0].donor_name, "D");
    }
}

