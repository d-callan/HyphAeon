use crate::attention::PhyloAttentionEngine;
use crate::attribution::TrajectoryParentalResolver;
use crate::due_diligence::SequenceDueDiligence;
use crate::fasta::{
    compute_min_informative_sites, count_parsimony_informative_sites,
    extract_informative_snp_alignment, trim_coverage_envelope,
};
use crate::landmarks::select_landmarks;
use crate::polishing::CumulativeTrajectoryPolisher;
use crate::prior::PhyloBias;
use crate::screener::LocalFluxScreener;
use crate::soft_root::compute_consensus_soft_root;
use crate::tree_rope::TreeRoPE;
use crate::types::{Alignment, ScanResult, ScreeningResult};

pub struct RhizAeonEngine {
    pub target_landmarks: usize,
    pub dim: usize,
    pub epsilon_0: f64,
    pub decay_lambda: f64,
    pub root_sink_factor: f64,
    pub poisson_floor: usize,
    pub alpha: f64,
    /// Informative SNP compression mode:
    /// None = Auto-trigger (L > 50,000 bp OR (L > 10,000 bp AND M_inf / L < 0.10))
    /// Some(true) = Force SNP compression
    /// Some(false) = Force full physical nucleotide scan
    pub compress_snps: Option<bool>,
}

impl Default for RhizAeonEngine {
    fn default() -> Self {
        Self {
            target_landmarks: 16,
            dim: 8,
            epsilon_0: 0.25,
            decay_lambda: 4.0 / 3.0,
            root_sink_factor: 1.0,
            poisson_floor: 3,
            alpha: 0.05,
            compress_snps: None,
        }
    }
}

impl RhizAeonEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_landmarks(mut self, target_landmarks: usize) -> Self {
        self.target_landmarks = target_landmarks;
        self
    }

    pub fn with_compress_snps(mut self, compress: bool) -> Self {
        self.compress_snps = Some(compress);
        self
    }

    pub fn with_auto_compress_snps(mut self) -> Self {
        self.compress_snps = None;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_poisson_floor(mut self, floor: usize) -> Self {
        self.poisson_floor = floor;
        self
    }

    /// Scans an alignment for mosaic recombination events using the full PA-FF v2.1 pipeline
    pub fn scan(&self, raw_aln: &Alignment) -> ScanResult {
        let (aln_trimmed, u_offset) = trim_coverage_envelope(raw_aln);
        let n = aln_trimmed.num_taxa;
        let l_chromosomal = raw_aln.length;

        let m_inf = count_parsimony_informative_sites(&aln_trimmed);
        let m_min = compute_min_informative_sites(n);

        // Early exit: non-recombinant H0 if informative sites below admission floor
        if m_inf < m_min {
            return ScanResult {
                num_taxa: n,
                length: l_chromosomal,
                informative_sites: m_inf,
                is_non_recombinant_h0: true,
                screening: ScreeningResult {
                    v_cont: vec![0.0; n],
                    v_root: vec![0.0; n],
                    r_root: vec![0.0; n],
                    gamma_root: vec![0.0; n],
                    rate_burst_indices: Vec::new(),
                    ghost_candidate_indices: Vec::new(),
                    candidate_recombinant_indices: Vec::new(),
                    clonal_backbone_indices: (0..n).collect(),
                    median_v_cont: 0.0,
                    is_non_recombinant_h0: true,
                },
                events: Vec::new(),
                run_time_ms: 0.0,
            };
        }

        // Auto-Trigger Threshold Policy:
        // Automatically compress to informative SNPs when L > 50,000 bp OR (L > 10,000 bp AND M_inf / L < 0.10)
        let should_compress = self.compress_snps.unwrap_or_else(|| {
            aln_trimmed.length > 50_000
                || (aln_trimmed.length > 10_000 && (m_inf as f64) / (aln_trimmed.length as f64) < 0.10)
        });

        let (aln, snp_map_opt) = if should_compress && m_inf >= m_min {
            let (snp_aln, map) = extract_informative_snp_alignment(&aln_trimmed);
            (snp_aln, Some(map))
        } else {
            (aln_trimmed, None)
        };
        let l = aln.length;


        // 1. Landmark Selection on Delta^K with adaptive scaling for large cohorts
        let effective_k = self.target_landmarks.max(n / 2).min(32).min(n);
        let landmarks = select_landmarks(&aln, effective_k);

        // 2. Tree-RoPE & Prior
        let rope = TreeRoPE::new(self.dim);
        let m_matrix = rope.compute_query_landmark_rotary_factors(&aln, &landmarks);

        let prior = PhyloBias::new(self.epsilon_0, self.decay_lambda);
        let b_prior = prior.compute_prior_matrix(&aln, &landmarks);

        // 3. Ancestral Simplex Soft Root
        let root_seq = compute_consensus_soft_root(&aln, None);

        // 4. Attention & Zero-drift Force Field
        let attention = PhyloAttentionEngine::new(self.dim, self.root_sink_factor);
        let (a_tensor, f_tensor, c_tensor) = attention.compute_attention_and_force_tensors(
            &aln,
            &landmarks,
            &m_matrix,
            &b_prior,
            &root_seq,
        );

        // 5. Pass 1 Screener
        let screener = LocalFluxScreener::new();
        let screening = screener.screen(&f_tensor, &c_tensor, &aln, &landmarks);

        if screening.is_non_recombinant_h0 {
            return ScanResult {
                num_taxa: n,
                length: l,
                informative_sites: m_inf,
                is_non_recombinant_h0: true,
                screening,
                events: Vec::new(),
                run_time_ms: 0.0,
            };
        }

        // 6. Pass 2 Polishing & Pass 3 Due Diligence on candidates
        let resolver = TrajectoryParentalResolver::new();
        let polisher = CumulativeTrajectoryPolisher::new();
        let due_diligence = SequenceDueDiligence::new(self.poisson_floor, self.alpha, 100_000);

        let mut verified_events = Vec::new();
        let num_channels = landmarks.num_landmarks + 1;

        for &cand_idx in &screening.candidate_recombinant_indices {
            let parental_pairs = resolver.resolve_all_reticulation_channels(
                cand_idx,
                &c_tensor,
                &a_tensor,
                l,
                n,
                &landmarks,
            );

            let mut candidate_tracts = Vec::new();
            for pair in &parental_pairs {
                let tracts = polisher.deconvolve_candidate(
                    cand_idx,
                    pair,
                    &c_tensor,
                    l,
                    n,
                    num_channels,
                );
                candidate_tracts.extend(tracts);
            }

            // Deduplicate overlapping tracts across different parental channels
            let deduplicated_tracts = deduplicate_candidate_tracts(candidate_tracts);

            for tract in deduplicated_tracts {
                let local_parental = resolver.resolve_reticulation_channels_localized(
                    cand_idx,
                    tract.u1_discrete,
                    tract.u2_discrete,
                    &c_tensor,
                    l,
                    n,
                    &landmarks,
                    &aln,
                    Some(tract.home_channel),
                    Some(tract.donor_channel),
                );
                let tract_opt = due_diligence.verify_tract(&tract, &local_parental, &aln, &landmarks, &screening.rate_burst_indices);

                if let Some(mut event) = tract_opt {
                    if event.is_verified {
                        if let Some(ref snp_map) = snp_map_opt {
                            // Map from 1-indexed SNP coordinates [1..M] to original chromosomal coordinates
                            let s_snp = event.u1.saturating_sub(1).min(snp_map.len().saturating_sub(1));
                            let e_snp = event.u2.saturating_sub(1).min(snp_map.len().saturating_sub(1));
                            let orig_u1 = snp_map[s_snp] + 1 + u_offset;
                            let orig_u2 = snp_map[e_snp] + 1 + u_offset;
                            event.u1 = orig_u1;
                            event.u2 = orig_u2;
                            event.u1_continuous = orig_u1 as f64;
                            event.u2_continuous = orig_u2 as f64;
                            event.tract_length = if orig_u2 >= orig_u1 { orig_u2 - orig_u1 + 1 } else { 0 };
                        } else {
                            // Standard full-length alignment offset
                            event.u1 += u_offset;
                            event.u2 += u_offset;
                            event.u1_continuous += u_offset as f64;
                            event.u2_continuous += u_offset as f64;
                        }
                        verified_events.push(event);
                    }
                }
            }
        }

        // Sort events by candidate_idx and u1
        verified_events.sort_by(|a, b| {
            a.candidate_idx
                .cmp(&b.candidate_idx)
                .then(a.u1.cmp(&b.u1))
        });

        // Deduplicate adjacent identical calls
        verified_events.dedup_by(|a, b| {
            a.candidate_idx == b.candidate_idx && a.u1 == b.u1 && a.u2 == b.u2
        });

        // Adjudicate reticulation graph (DIR-RHIZ-PAFF-013.1)
        let verified_events = due_diligence.adjudicate_reticulation_graph(verified_events, &raw_aln, &screening);

        let is_h0 = verified_events.is_empty();

        ScanResult {
            num_taxa: n,
            length: raw_aln.length,
            informative_sites: m_inf,
            is_non_recombinant_h0: is_h0,
            screening,
            events: verified_events,
            run_time_ms: 0.0,
        }
    }
}

/// Deduplicates overlapping tracts across different parental channels for a single candidate taxon.
/// Preserves distinct non-overlapping tracts (Multi-Way Mosaicism) while suppressing redundant
/// sister-channel duplicates (IoU >= 0.50 or overlap >= 70% of either tract).
fn deduplicate_candidate_tracts(mut tracts: Vec<crate::types::PhysicalTract>) -> Vec<crate::types::PhysicalTract> {
    if tracts.len() <= 1 {
        return tracts;
    }

    // Sort by physical significance score descending so strongest tracts have priority
    tracts.sort_by(|a, b| {
        b.z_phys
            .partial_cmp(&a.z_phys)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<crate::types::PhysicalTract> = Vec::new();

    for tract in tracts {
        let mut is_redundant = false;
        let l_tract = (tract.u2_discrete.saturating_sub(tract.u1_discrete) + 1) as f64;

        for existing in &kept {
            let s_ov = tract.u1_discrete.max(existing.u1_discrete);
            let e_ov = tract.u2_discrete.min(existing.u2_discrete);

            if e_ov >= s_ov {
                let inter = (e_ov - s_ov + 1) as f64;
                let l_exist = (existing.u2_discrete.saturating_sub(existing.u1_discrete) + 1) as f64;
                let union_span = (tract.u2_discrete.max(existing.u2_discrete) - tract.u1_discrete.min(existing.u1_discrete) + 1) as f64;
                let iou = inter / union_span.max(1.0);
                let frac_tract = inter / l_tract.max(1.0);
                let frac_exist = inter / l_exist.max(1.0);

                if iou >= 0.50 || frac_tract >= 0.70 || frac_exist >= 0.70 {
                    is_redundant = true;
                    break;
                }
            }
        }

        if !is_redundant {
            kept.push(tract);
        }
    }

    // Restore coordinate order
    kept.sort_by_key(|t| t.u1_discrete);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Alignment;

    #[test]
    fn test_deduplicate_candidate_tracts() {
        let t1 = crate::types::PhysicalTract {
            candidate_idx: 0,
            home_channel: 1,
            donor_channel: 2,
            is_crossover: false,
            u1_discrete: 100,
            u2_discrete: 500,
            u1_continuous: 100.0,
            u2_continuous: 500.0,
            delta_c_max: 20.0,
            z_phys: 12.0,
            z_crit: 3.0,
            is_physically_significant: true,
        };

        // Redundant with t1 (nearly identical span)
        let t2 = crate::types::PhysicalTract {
            candidate_idx: 0,
            home_channel: 1,
            donor_channel: 3,
            is_crossover: false,
            u1_discrete: 110,
            u2_discrete: 490,
            u1_continuous: 110.0,
            u2_continuous: 490.0,
            delta_c_max: 18.0,
            z_phys: 9.0,
            z_crit: 3.0,
            is_physically_significant: true,
        };

        // Distinct non-overlapping tract (Multi-Way mosaic second tract)
        let t3 = crate::types::PhysicalTract {
            candidate_idx: 0,
            home_channel: 1,
            donor_channel: 4,
            is_crossover: false,
            u1_discrete: 800,
            u2_discrete: 1200,
            u1_continuous: 800.0,
            u2_continuous: 1200.0,
            delta_c_max: 22.0,
            z_phys: 11.0,
            z_crit: 3.0,
            is_physically_significant: true,
        };

        let dedup = deduplicate_candidate_tracts(vec![t1, t2, t3]);
        assert_eq!(dedup.len(), 2, "Expected 2 tracts after deduplicating redundant overlap");
        assert_eq!(dedup[0].u1_discrete, 100);
        assert_eq!(dedup[0].u2_discrete, 500);
        assert_eq!(dedup[0].donor_channel, 2, "Stronger tract should survive");
        assert_eq!(dedup[1].u1_discrete, 800);
        assert_eq!(dedup[1].u2_discrete, 1200);
        assert_eq!(dedup[1].donor_channel, 4);
    }

    #[test]
    fn test_multiway_mosaic_detection() {
        let l = 2000;
        let taxa = vec!["R".into(), "P1".into(), "P2".into(), "P3".into(), "O".into()];
        let n = taxa.len();

        let mut data = vec![1u8; n * l];

        // Seed distinctive patterns
        // P1 is mostly state 1
        // P2 has state 2 every 6 sites
        // P3 has state 3 every 6 sites
        // O has state 4 every 4 sites
        for u in 0..l {
            if u % 6 == 0 {
                data[2 * l + u] = 2; // P2
            }
            if u % 6 == 3 {
                data[3 * l + u] = 3; // P3
            }
            if u % 4 == 0 {
                data[4 * l + u] = 4; // O
            }
        }

        // Construct R as multiway mosaic:
        // 0..400: P1
        // 400..800: P2
        // 800..1200: P1
        // 1200..1600: P3
        // 1600..2000: P1
        for u in 0..l {
            if u >= 400 && u < 800 {
                data[0 * l + u] = data[2 * l + u];
            } else if u >= 1200 && u < 1600 {
                data[0 * l + u] = data[3 * l + u];
            } else {
                data[0 * l + u] = data[1 * l + u];
            }
        }

        let aln = Alignment::new(taxa, l, data);
        let engine = RhizAeonEngine::new();
        let res = engine.scan(&aln);

        assert!(!res.is_non_recombinant_h0, "Expected multiway mosaic to be detected");
        println!("Detected events: {:?}", res.events.len());
        for ev in &res.events {
            println!("Event: {} in [{}, {}] Home: {} Donor: {}",
                ev.candidate_name, ev.u1, ev.u2, ev.home_name, ev.donor_name);
        }

        // Verify that R is the candidate and both tracts are detected
        let r_events: Vec<_> = res.events.iter().filter(|e| e.candidate_name == "R").collect();
        assert!(r_events.len() >= 2, "Expected at least 2 events for multi-way mosaic child R");

        let donors: Vec<_> = r_events.iter().map(|e| e.donor_name.as_str()).collect();
        assert!(donors.contains(&"P2"), "Expected P2 to be identified as donor");
        assert!(donors.contains(&"P3"), "Expected P3 to be identified as donor");
    }
}



