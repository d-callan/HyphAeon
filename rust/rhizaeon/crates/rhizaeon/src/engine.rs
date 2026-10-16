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
    pub tier2_stringency: f64,
    pub tier3_stringency: f64,
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
            tier2_stringency: 1.0,
            tier3_stringency: 1.0,
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

    pub fn with_tier2_stringency(mut self, stringency: f64) -> Self {
        self.tier2_stringency = stringency;
        self
    }

    pub fn with_tier3_stringency(mut self, stringency: f64) -> Self {
        self.tier3_stringency = stringency;
        self
    }

    /// Scans an alignment for mosaic recombination events using the full PA-FF v2.1 pipeline
    pub fn scan(&self, raw_aln: &Alignment) -> ScanResult {
        let (aln_trimmed, u_offset) = trim_coverage_envelope(raw_aln);
        let n = aln_trimmed.num_taxa;
        let l_chromosomal = raw_aln.length;

        let m_inf = count_parsimony_informative_sites(&aln_trimmed);
        let base_m_min = compute_min_informative_sites(n);
        let m_min = if (self.tier3_stringency - 1.0).abs() < 1e-9 {
            base_m_min
        } else {
            ((base_m_min as f64 * self.tier3_stringency.min(1.0)).round() as usize).max(2)
        };

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
        // Automatically compress to informative SNPs when L > 30,000 bp (bacterial/chromosomal scale)
        // or when L > 10,000 bp AND N >= 48, or when N * L > 600,000 matrix cells
        let should_compress = self.compress_snps.unwrap_or_else(|| {
            aln_trimmed.length > 30_000
                || (aln_trimmed.length > 10_000 && n >= 48)
                || (n * aln_trimmed.length > 600_000)
        });

        let (aln, snp_map_opt) = if should_compress && m_inf >= m_min {
            let (snp_aln, map) = extract_informative_snp_alignment(&aln_trimmed);
            (snp_aln, Some(map))
        } else {
            (aln_trimmed, None)
        };
        let l = aln.length;

        // 1. Landmark Selection on Delta^K with adaptive scaling for large cohorts
        let effective_k = if self.target_landmarks == 16 {
            self.target_landmarks.max(n / 2).min(32).min(n)
        } else {
            self.target_landmarks.max(2).min(n).min(127)
        };
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
        let (a_tensor, f_tensor, c_tensor) = attention
            .compute_attention_and_force_tensors(&aln, &landmarks, &m_matrix, &b_prior, &root_seq);

        // 5. Pass 1 Screener
        let screener = LocalFluxScreener::new().with_tier2_stringency(self.tier2_stringency);
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

        // Pass 2: Retain full canonical landmark basis (DIR-RHIZ-PAFF-012)
        // Discarding candidate taxa from the landmark basis corrupts authentic parental lineages
        // (e.g. Subtype B in HIV-1) that exhibit legitimate contrast with recombinant children.
        let num_channels = landmarks.num_landmarks + 1;

        // 6. Pass 2 Polishing & Pass 3 Due Diligence on candidates
        let (d_sister, _) = crate::landmarks::compute_scale_invariant_distance_thresholds(&aln);
        let compression_ratio = if snp_map_opt.is_some() {
            (l as f64) / (l_chromosomal.max(1) as f64)
        } else {
            1.0
        };
        let resolver = TrajectoryParentalResolver::new().with_stringency(self.tier2_stringency);
        let polisher = CumulativeTrajectoryPolisher::new()
            .with_stringency(self.tier2_stringency)
            .with_compression_ratio(compression_ratio);
        let due_diligence = SequenceDueDiligence::new(self.poisson_floor, self.alpha, 100_000)
            .with_stringency(self.tier3_stringency)
            .with_compression_ratio(compression_ratio)
            .with_d_sister(d_sister);

        let mut verified_events = Vec::new();

        for &cand_idx in &screening.candidate_recombinant_indices {
            let parental_pairs = resolver.resolve_all_reticulation_channels(
                cand_idx,
                &c_tensor,
                &a_tensor,
                l,
                n,
                &landmarks,
                &aln,
                &screening.ghost_candidate_indices,
            );

            let mut candidate_tracts = Vec::new();
            for pair in &parental_pairs {
                let tracts =
                    polisher.deconvolve_candidate(cand_idx, pair, &c_tensor, l, n, num_channels);
                candidate_tracts.extend(tracts);
            }

            let mut candidate_verified = Vec::new();
            for tract in candidate_tracts {
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
                    &screening.candidate_recombinant_indices,
                    &screening.ghost_candidate_indices,
                );

                let mut tract_opt = due_diligence.verify_tract(
                    &tract,
                    &local_parental,
                    &aln,
                    &landmarks,
                    &screening.rate_burst_indices,
                    &screening.candidate_recombinant_indices,
                );

                // Ghost Donor Fallback for Unsampled Outgroup Cassettes:
                // When an introgressed tract comes from an unsampled outgroup (Ghost), contemporary
                // landmarks from external clades can still capture partial softmax flux (because the
                // candidate departs sharply from its Home clade), causing the polisher to segment the
                // tract on a contemporary channel while `verify_tract` rejects that contemporary donor
                // (since the candidate does not match it in the tract). In that case, test whether the
                // tract is a verified Ghost donor cassette against the resolved Home parent.
                if tract_opt.is_none() && !local_parental.is_ghost_donor {
                    let mut ghost_parental = local_parental.clone();
                    ghost_parental.donor_channel = 0;
                    ghost_parental.donor_landmark_idx = None;
                    ghost_parental.is_ghost_donor = true;
                    ghost_parental.donor_name = "Ghost".to_string();
                    ghost_parental.donor_global_idx = None;

                    tract_opt = due_diligence.verify_tract(
                        &tract,
                        &ghost_parental,
                        &aln,
                        &landmarks,
                        &screening.rate_burst_indices,
                        &screening.candidate_recombinant_indices,
                    );
                }

                if let Some(mut event) = tract_opt {
                    if event.is_verified {
                        if let Some(ref snp_map) = snp_map_opt {
                            // Map from 1-indexed SNP coordinates [1..M] to original chromosomal coordinates
                            let s_snp = event
                                .u1
                                .saturating_sub(1)
                                .min(snp_map.len().saturating_sub(1));
                            let e_snp = event
                                .u2
                                .saturating_sub(1)
                                .min(snp_map.len().saturating_sub(1));
                            let orig_u1 = snp_map[s_snp] + 1 + u_offset;
                            let orig_u2 = snp_map[e_snp] + 1 + u_offset;
                            event.u1 = orig_u1;
                            event.u2 = orig_u2;
                            event.u1_continuous = orig_u1 as f64;
                            event.u2_continuous = orig_u2 as f64;
                            event.tract_length = if orig_u2 >= orig_u1 {
                                orig_u2 - orig_u1 + 1
                            } else {
                                0
                            };
                        } else {
                            // Standard full-length alignment offset
                            event.u1 += u_offset;
                            event.u2 += u_offset;
                            event.u1_continuous += u_offset as f64;
                            event.u2_continuous += u_offset as f64;
                        }
                        candidate_verified.push(event);
                    }
                }
            }

            let deduplicated_events = deduplicate_candidate_events(candidate_verified, &aln);
            verified_events.extend(deduplicated_events);
        }

        // Sort events by candidate_idx and u1
        verified_events.sort_by(|a, b| a.candidate_idx.cmp(&b.candidate_idx).then(a.u1.cmp(&b.u1)));

        // Deduplicate adjacent identical calls
        verified_events
            .dedup_by(|a, b| a.isolates[0] == b.isolates[0] && a.u1 == b.u1 && a.u2 == b.u2);

        // Adjudicate reticulation graph (DIR-RHIZ-PAFF-013.1)
        let mut verified_events =
            due_diligence.adjudicate_reticulation_graph(verified_events, &raw_aln, &screening);

        // Resolve transitive donor & home chains for co-recombinant isolates
        resolve_donor_chains(&mut verified_events, &raw_aln);

        // Group ancestral events
        verified_events = group_ancestral_events(verified_events, &raw_aln);

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

/// Resolves transitive donor chains and co-recombinant lineages across overlapping tracts.
/// When sequence A is detected with donor B, but B is itself a co-recombinant with donor C
/// sharing the same Home lineage over an overlapping tract, A inherits donor C from B.
/// Similarly, if sequence A cites B as Home, but B is a co-recombinant sharing the same donor
/// over an overlapping tract with Home C, A inherits Home C from B.
fn resolve_donor_chains(
    events: &mut [crate::types::RecombinationEvent],
    aln: &crate::types::Alignment,
) {
    if events.len() <= 1 {
        return;
    }

    let max_passes = events.len().min(10);
    for _ in 0..max_passes {
        let mut changed = false;

        for i in 0..events.len() {
            let e1_home = events[i].home_name.clone();
            let e1_donor = events[i].donor_name.clone();
            let e1_cand = events[i].candidate_name.clone();
            let e1_u1 = events[i].u1;
            let e1_u2 = events[i].u2;

            // 1. Resolve Donor Chain: e1 -> e2 (e1 cited e2 as Donor)
            if !e1_donor.is_empty() && e1_donor != "Ghost" {
                let mut resolved_donor_idx = None;
                let mut resolved_donor_name = None;
                let mut resolved_is_ghost = false;

                for (j, e2) in events.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    let e2_is_donor_isolate =
                        e2.candidate_name == e1_donor || e2.isolates.contains(&e1_donor);
                    if !e2_is_donor_isolate {
                        continue;
                    }

                    // Check tract overlap: >= 50% of the minimum length
                    let ov_start = e1_u1.max(e2.u1);
                    let ov_end = e1_u2.min(e2.u2);
                    if ov_end < ov_start {
                        continue;
                    }
                    let ov_len = ov_end - ov_start + 1;
                    let min_len =
                        (e1_u2.saturating_sub(e1_u1) + 1).min(e2.u2.saturating_sub(e2.u1) + 1);
                    if (ov_len as f64) < 0.5 * (min_len as f64) {
                        continue;
                    }

                    // Compatible Home lineage
                    if e1_home != e2.home_name {
                        continue;
                    }

                    // e2's donor must not be e1 (prevent cyclic attribution)
                    if e2.donor_name.is_empty()
                        || e2.donor_name == e1_cand
                        || events[i].isolates.contains(&e2.donor_name)
                    {
                        continue;
                    }

                    resolved_donor_name = Some(e2.donor_name.clone());
                    resolved_donor_idx = e2.donor_idx;
                    resolved_is_ghost = e2.is_ghost_donor;
                    break;
                }

                if let Some(new_donor) = resolved_donor_name {
                    if new_donor != events[i].donor_name {
                        events[i].donor_name = new_donor;
                        events[i].donor_idx = resolved_donor_idx;
                        events[i].is_ghost_donor = resolved_is_ghost;

                        if let Some(dg) = resolved_donor_idx {
                            if dg < aln.num_taxa && events[i].candidate_idx < aln.num_taxa {
                                let cand_row = aln.row(events[i].candidate_idx);
                                let donor_row = aln.row(dg);
                                let s_idx = events[i].u1.saturating_sub(1);
                                let e_idx = events[i].u2.min(aln.length);

                                let mut d_tr = 0usize;
                                let mut v_tr = 0usize;
                                let mut d_fl = 0usize;
                                let mut v_fl = 0usize;

                                for u in 0..aln.length {
                                    let rc = cand_row[u];
                                    let rd = donor_row[u];
                                    if rc > 0 && rd > 0 {
                                        if u >= s_idx && u < e_idx {
                                            v_tr += 1;
                                            if rc != rd {
                                                d_tr += 1;
                                            }
                                        } else {
                                            v_fl += 1;
                                            if rc != rd {
                                                d_fl += 1;
                                            }
                                        }
                                    }
                                }
                                if v_tr > 0 {
                                    events[i].d_tract_donor = (d_tr as f64) / (v_tr as f64);
                                }
                                if v_fl > 0 {
                                    events[i].d_flank_donor = (d_fl as f64) / (v_fl as f64);
                                }
                            }
                        }
                        changed = true;
                    }
                }
            }

            // 2. Resolve Home Chain: e1 -> e2 (e1 cited e2 as Home)
            if !e1_home.is_empty() {
                let mut resolved_home_idx = None;
                let mut resolved_home_name = None;

                for (j, e2) in events.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    let e2_is_home_isolate =
                        e2.candidate_name == e1_home || e2.isolates.contains(&e1_home);
                    if !e2_is_home_isolate {
                        continue;
                    }

                    // Check tract overlap
                    let ov_start = e1_u1.max(e2.u1);
                    let ov_end = e1_u2.min(e2.u2);
                    if ov_end < ov_start {
                        continue;
                    }
                    let ov_len = ov_end - ov_start + 1;
                    let min_len =
                        (e1_u2.saturating_sub(e1_u1) + 1).min(e2.u2.saturating_sub(e2.u1) + 1);
                    if (ov_len as f64) < 0.5 * (min_len as f64) {
                        continue;
                    }

                    // Compatible Donor lineage
                    if events[i].donor_name != e2.donor_name {
                        continue;
                    }

                    // e2's home must not be e1
                    if e2.home_name.is_empty()
                        || e2.home_name == e1_cand
                        || events[i].isolates.contains(&e2.home_name)
                    {
                        continue;
                    }

                    resolved_home_name = Some(e2.home_name.clone());
                    resolved_home_idx = e2.home_idx;
                    break;
                }

                if let Some(new_home) = resolved_home_name {
                    if new_home != events[i].home_name {
                        events[i].home_name = new_home;
                        events[i].home_idx = resolved_home_idx;

                        if let Some(hg) = resolved_home_idx {
                            if hg < aln.num_taxa && events[i].candidate_idx < aln.num_taxa {
                                let cand_row = aln.row(events[i].candidate_idx);
                                let home_row = aln.row(hg);
                                let s_idx = events[i].u1.saturating_sub(1);
                                let e_idx = events[i].u2.min(aln.length);

                                let mut d_tr = 0usize;
                                let mut v_tr = 0usize;
                                let mut d_fl = 0usize;
                                let mut v_fl = 0usize;

                                for u in 0..aln.length {
                                    let rc = cand_row[u];
                                    let rh = home_row[u];
                                    if rc > 0 && rh > 0 {
                                        if u >= s_idx && u < e_idx {
                                            v_tr += 1;
                                            if rc != rh {
                                                d_tr += 1;
                                            }
                                        } else {
                                            v_fl += 1;
                                            if rc != rh {
                                                d_fl += 1;
                                            }
                                        }
                                    }
                                }
                                if v_tr > 0 {
                                    events[i].d_tract_home = (d_tr as f64) / (v_tr as f64);
                                }
                                if v_fl > 0 {
                                    events[i].d_flank_home = (d_fl as f64) / (v_fl as f64);
                                }
                            }
                        }
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn group_ancestral_events(
    mut events: Vec<crate::types::RecombinationEvent>,
    aln: &crate::types::Alignment,
) -> Vec<crate::types::RecombinationEvent> {
    let (_d_sister, d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
    let mut grouped: Vec<crate::types::RecombinationEvent> = Vec::new();
    events.sort_by_key(|e| (e.home_idx, e.donor_idx, e.u1));
    for e in events {
        let mut found = false;
        for g in &mut grouped {
            // Parental compatibility via midpoint bisection against inter-parental divergence or clade threshold:
            let d_inter = match (g.home_idx, g.donor_idx) {
                (Some(gh), Some(gd)) => crate::landmarks::compute_pairwise_distance(aln, gh, gd),
                _ => d_clade,
            };
            let clade_rad = d_clade.max(0.5 * d_inter);

            // 1. Home compatibility: exact name match OR home leaves cluster together within clade radius
            let home_matches = g.home_name == e.home_name
                || match (g.home_idx, e.home_idx) {
                    (Some(gh), Some(eh)) => {
                        crate::landmarks::compute_pairwise_distance(aln, gh, eh) <= clade_rad
                    }
                    _ => false,
                };

            // 2. Donor compatibility: exact name match OR both Ghost OR donor leaves cluster together
            let donor_matches = g.donor_name == e.donor_name
                || match (g.donor_idx, e.donor_idx) {
                    (Some(gd), Some(ed)) => {
                        crate::landmarks::compute_pairwise_distance(aln, gd, ed) <= clade_rad
                    }
                    _ => false,
                };

            if home_matches && donor_matches {
                let overlap_start = g.u1.max(e.u1);
                let overlap_end = g.u2.min(e.u2);
                if overlap_end >= overlap_start {
                    let overlap_len = overlap_end - overlap_start + 1;
                    let min_len =
                        (g.u2.saturating_sub(g.u1) + 1).min(e.u2.saturating_sub(e.u1) + 1);
                    if overlap_len as f64 >= 0.5 * min_len as f64 {
                        g.isolates.extend(e.isolates.clone());
                        g.isolates.sort();
                        g.isolates.dedup();
                        g.u1 = g.u1.min(e.u1);
                        g.u2 = g.u2.max(e.u2);
                        g.u1_continuous = g.u1_continuous.min(e.u1_continuous);
                        g.u2_continuous = g.u2_continuous.max(e.u2_continuous);
                        g.tract_length = if g.u2 >= g.u1 { g.u2 - g.u1 + 1 } else { 0 };
                        if e.p_fisher < g.p_fisher {
                            g.p_fisher = e.p_fisher;
                        }
                        if e.z_phys > g.z_phys {
                            g.z_phys = e.z_phys;
                        }
                        g.s_informative = g.s_informative.max(e.s_informative);
                        found = true;
                        break;
                    }
                }
            }
        }
        if !found {
            grouped.push(e);
        }
    }

    // Cohort Roster Recruitment (DIR-RHIZ-PAFF-013.9)
    // Once an ancestral event is verified between Home and Donor over [u1, u2],
    // enroll all cohort isolates that carry this verified mosaic haplotype into the event roster.
    // Governed strictly by the 4 canonical physical polarity invariants (Zero Magic Numbers):
    //   1. d_flank(S, H) < d_flank(S, D)
    //   2. d_tract(S, D) < d_tract(S, H)
    //   3. d_flank(S, D) - d_tract(S, D) > 0 (positive recoil towards Donor)
    //   4. d_tract(S, H) - d_flank(S, H) > 0 (positive departure from Home)
    for g in &mut grouped {
        if let Some(hg) = g.home_idx {
            let s_idx = g.u1.saturating_sub(1);
            let e_idx = g.u2.min(aln.length);
            if e_idx <= s_idx {
                continue;
            }
            let home_row = aln.row(hg);
            let donor_row_opt = g.donor_idx.map(|dg| aln.row(dg));

            for t_idx in 0..aln.num_taxa {
                let t_name = &aln.taxa[t_idx];
                if g.isolates.contains(t_name) {
                    continue;
                }
                let t_row = aln.row(t_idx);

                let mut d_tr_h = 0usize;
                let mut v_tr_h = 0usize;
                let mut d_fl_h = 0usize;
                let mut v_fl_h = 0usize;
                for u in 0..aln.length {
                    let rt = t_row[u];
                    let rh = home_row[u];
                    if rt > 0 && rh > 0 {
                        if u >= s_idx && u < e_idx {
                            v_tr_h += 1;
                            if rt != rh {
                                d_tr_h += 1;
                            }
                        } else {
                            v_fl_h += 1;
                            if rt != rh {
                                d_fl_h += 1;
                            }
                        }
                    }
                }
                let dist_tr_h = if v_tr_h > 0 {
                    (d_tr_h as f64) / (v_tr_h as f64)
                } else {
                    1.0
                };
                let dist_fl_h = if v_fl_h > 0 {
                    (d_fl_h as f64) / (v_fl_h as f64)
                } else {
                    1.0
                };

                if let Some(donor_row) = donor_row_opt {
                    let mut d_tr_d = 0usize;
                    let mut v_tr_d = 0usize;
                    let mut d_fl_d = 0usize;
                    let mut v_fl_d = 0usize;
                    let mut d_tr_hd = 0usize;
                    let mut v_tr_hd = 0usize;
                    let mut d_fl_hd = 0usize;
                    let mut v_fl_hd = 0usize;
                    for u in 0..aln.length {
                        let rt = t_row[u];
                        let rd = donor_row[u];
                        let rh = home_row[u];
                        if rt > 0 && rd > 0 {
                            if u >= s_idx && u < e_idx {
                                v_tr_d += 1;
                                if rt != rd {
                                    d_tr_d += 1;
                                }
                            } else {
                                v_fl_d += 1;
                                if rt != rd {
                                    d_fl_d += 1;
                                }
                            }
                        }
                        if rh > 0 && rd > 0 {
                            if u >= s_idx && u < e_idx {
                                v_tr_hd += 1;
                                if rh != rd {
                                    d_tr_hd += 1;
                                }
                            } else {
                                v_fl_hd += 1;
                                if rh != rd {
                                    d_fl_hd += 1;
                                }
                            }
                        }
                    }
                    let dist_tr_d = if v_tr_d > 0 {
                        (d_tr_d as f64) / (v_tr_d as f64)
                    } else {
                        1.0
                    };
                    let dist_fl_d = if v_fl_d > 0 {
                        (d_fl_d as f64) / (v_fl_d as f64)
                    } else {
                        1.0
                    };
                    let dist_tr_hd = if v_tr_hd > 0 {
                        (d_tr_hd as f64) / (v_tr_hd as f64)
                    } else {
                        1.0
                    };
                    let dist_fl_hd = if v_fl_hd > 0 {
                        (d_fl_hd as f64) / (v_fl_hd as f64)
                    } else {
                        1.0
                    };

                    let matches_mosaic = dist_fl_h < dist_fl_d
                        && dist_tr_d < dist_tr_h
                        && dist_fl_d > dist_tr_d
                        && dist_tr_h > dist_fl_h
                        && dist_fl_h < 0.5 * dist_fl_hd
                        && dist_tr_d < 0.5 * dist_tr_hd;

                    if matches_mosaic {
                        g.isolates.push(t_name.clone());
                    }
                } else if g.is_ghost_donor {
                    let cand_row = aln.row(g.candidate_idx);
                    let mut d_tr_c = 0usize;
                    let mut v_tr_c = 0usize;
                    for u in s_idx..e_idx {
                        let rt = t_row[u];
                        let rc = cand_row[u];
                        if rt > 0 && rc > 0 {
                            v_tr_c += 1;
                            if rt != rc {
                                d_tr_c += 1;
                            }
                        }
                    }
                    let dist_tr_c = if v_tr_c > 0 {
                        (d_tr_c as f64) / (v_tr_c as f64)
                    } else {
                        1.0
                    };
                    let (_d_sister, d_clade) =
                        crate::landmarks::compute_scale_invariant_distance_thresholds(aln);
                    if dist_tr_c < dist_tr_h && dist_tr_h > dist_fl_h && dist_fl_h < d_clade {
                        g.isolates.push(t_name.clone());
                    }
                }
            }
            g.isolates.sort();
            g.isolates.dedup();
        }
    }

    // Sort grouped events by statistical significance (lowest p-value first, highest z-phys tiebreaker)
    grouped.sort_by(|a, b| {
        a.p_fisher
            .partial_cmp(&b.p_fisher)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.z_phys
                    .partial_cmp(&a.z_phys)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Post-recruitment consolidation: merge redundant events that recruited the same recipient cohort
    // over the same genomic interval via sister donors/homes from the same lineage cluster.
    let mut consolidated: Vec<crate::types::RecombinationEvent> = Vec::new();
    for ev in grouped {
        let mut merged_into_existing = false;
        for existing in &mut consolidated {
            let ov_start = ev.u1.max(existing.u1);
            let ov_end = ev.u2.min(existing.u2);
            if ov_end < ov_start {
                continue;
            }
            let ov_len = (ov_end - ov_start + 1) as f64;
            let min_len = ((ev.u2.saturating_sub(ev.u1) + 1)
                .min(existing.u2.saturating_sub(existing.u1) + 1)) as f64;
            let union_len = (ev.u2.max(existing.u2) - ev.u1.min(existing.u1) + 1) as f64;
            let iou_span = ov_len / union_len.max(1.0);
            let frac_span = ov_len / min_len.max(1.0);
            if iou_span < 0.50 && frac_span < 0.70 {
                continue;
            }

            // Recipient roster overlap
            let shared_isolates = ev
                .isolates
                .iter()
                .filter(|iso| existing.isolates.contains(iso))
                .count();
            let min_iso = ev.isolates.len().min(existing.isolates.len()).max(1);
            let union_iso = (ev.isolates.len() + existing.isolates.len() - shared_isolates).max(1);
            let jaccard_iso = (shared_isolates as f64) / (union_iso as f64);
            let subset_iso = (shared_isolates as f64) / (min_iso as f64);

            if jaccard_iso >= 0.50 || subset_iso >= 0.75 {
                let same_home_clade = existing.home_name == ev.home_name
                    || match (existing.home_idx, ev.home_idx) {
                        (Some(h1), Some(h2)) => {
                            crate::landmarks::compute_pairwise_distance(aln, h1, h2) <= d_clade
                        }
                        _ => false,
                    };
                let same_donor_clade = existing.donor_name == ev.donor_name
                    || (existing.is_ghost_donor && ev.is_ghost_donor)
                    || match (existing.donor_idx, ev.donor_idx) {
                        (Some(d1), Some(d2)) => {
                            crate::landmarks::compute_pairwise_distance(aln, d1, d2) <= d_clade
                        }
                        _ => false,
                    };

                if same_home_clade && same_donor_clade {
                    existing.isolates.extend(ev.isolates.clone());
                    existing.isolates.sort();
                    existing.isolates.dedup();
                    if ev.p_fisher < existing.p_fisher {
                        existing.p_fisher = ev.p_fisher;
                    }
                    if ev.z_phys > existing.z_phys {
                        existing.z_phys = ev.z_phys;
                    }
                    existing.s_informative = existing.s_informative.max(ev.s_informative);
                    merged_into_existing = true;
                    break;
                }
            }
        }
        if !merged_into_existing {
            consolidated.push(ev);
        }
    }

    // Assign 1-based sequential event ID and sort by it
    for (i, g) in consolidated.iter_mut().enumerate() {
        g.event_id = i + 1;
    }
    consolidated
}

/// Deduplicates overlapping verified events across different parental channels for a single candidate taxon.
/// Preserves distinct non-overlapping tracts and nested multi-way mosaics (e.g. Russian dolls from distinct donors)
/// while suppressing redundant sister-channel duplicates (IoU >= 0.50 or overlap >= 70% for same-donor clade).
fn deduplicate_candidate_events(
    mut events: Vec<crate::types::RecombinationEvent>,
    aln: &crate::types::Alignment,
) -> Vec<crate::types::RecombinationEvent> {
    if events.len() <= 1 {
        return events;
    }

    let (_d_sister, d_clade) = crate::landmarks::compute_scale_invariant_distance_thresholds(aln);

    // Sort by physical significance score descending so strongest events have priority
    events.sort_by(|a, b| {
        b.z_phys
            .partial_cmp(&a.z_phys)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<crate::types::RecombinationEvent> = Vec::new();

    for ev in events {
        let mut is_redundant = false;
        let l_ev = (ev.u2.saturating_sub(ev.u1) + 1) as f64;

        for existing in &kept {
            let s_ov = ev.u1.max(existing.u1);
            let e_ov = ev.u2.min(existing.u2);

            if e_ov >= s_ov {
                let inter = (e_ov - s_ov + 1) as f64;
                let l_exist = (existing.u2.saturating_sub(existing.u1) + 1) as f64;
                let union_span = (ev.u2.max(existing.u2) - ev.u1.min(existing.u1) + 1) as f64;
                let iou = inter / union_span.max(1.0);
                let frac_ev = inter / l_ev.max(1.0);
                let frac_exist = inter / l_exist.max(1.0);

                let is_same_donor = match (ev.donor_idx, existing.donor_idx) {
                    (Some(d1), Some(d2)) => {
                        d1 == d2
                            || crate::landmarks::compute_pairwise_distance(aln, d1, d2) < d_clade
                    }
                    (None, None) => true,
                    _ => false,
                };

                if is_same_donor {
                    // For the same donor clade, any substantial overlap is redundant
                    if iou >= 0.50 || frac_ev >= 0.70 || frac_exist >= 0.70 {
                        is_redundant = true;
                        break;
                    }
                } else {
                    // For distinct donor clades, only compete if they claim the exact same span (IoU >= 0.50)
                    // Nested Russian dolls (IoU < 0.50) from distinct donors are preserved.
                    if iou >= 0.50 {
                        is_redundant = true;
                        break;
                    }
                }
            }
        }

        if !is_redundant {
            kept.push(ev);
        }
    }

    // Restore coordinate order
    kept.sort_by_key(|e| e.u1);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Alignment;

    #[test]
    fn test_deduplicate_candidate_events() {
        let make_event =
            |u1: usize, u2: usize, donor: &str, z: f64| crate::types::RecombinationEvent {
                event_id: 1,
                isolates: vec!["R".to_string()],
                candidate_idx: 0,
                candidate_name: "R".to_string(),
                home_idx: Some(1),
                home_name: "P1".to_string(),
                donor_idx: Some(2),
                donor_name: donor.to_string(),
                is_ghost_donor: false,
                is_crossover: false,
                u1,
                u2,
                u1_continuous: u1 as f64,
                u2_continuous: u2 as f64,
                tract_length: u2 - u1 + 1,
                s_informative: 20,
                p_fisher: 1e-10,
                z_phys: z,
                d_flank_home: 0.005,
                d_flank_donor: 0.05,
                d_tract_home: 0.05,
                d_tract_donor: 0.005,
                is_verified: true,
            };

        let t1 = make_event(100, 500, "P2", 12.0);
        let t2 = make_event(110, 490, "P3", 9.0);
        let t3 = make_event(800, 1200, "P4", 11.0);

        let dummy_aln = Alignment {
            taxa: vec![
                "R".into(),
                "P1".into(),
                "P2".into(),
                "P3".into(),
                "P4".into(),
            ],
            num_taxa: 5,
            length: 1500,
            matrix: vec![1u8; 5 * 1500],
        };
        let dedup = deduplicate_candidate_events(vec![t1, t2, t3], &dummy_aln);
        assert_eq!(
            dedup.len(),
            2,
            "Expected 2 events after deduplicating redundant overlap"
        );
        assert_eq!(dedup[0].u1, 100);
        assert_eq!(dedup[0].u2, 500);
        assert_eq!(dedup[0].donor_name, "P2", "Stronger event should survive");
        assert_eq!(dedup[1].u1, 800);
        assert_eq!(dedup[1].u2, 1200);
        assert_eq!(dedup[1].donor_name, "P4");
    }

    #[test]
    fn test_multiway_mosaic_detection() {
        let l = 2000;
        let taxa = vec![
            "R".into(),
            "P1".into(),
            "P2".into(),
            "P3".into(),
            "O".into(),
        ];
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

        assert!(
            !res.is_non_recombinant_h0,
            "Expected multiway mosaic to be detected"
        );
        println!("Detected events: {:?}", res.events.len());
        for ev in &res.events {
            println!(
                "Event: {} in [{}, {}] Home: {} Donor: {}",
                ev.candidate_name, ev.u1, ev.u2, ev.home_name, ev.donor_name
            );
        }

        // Verify that R is the candidate and both tracts are detected
        let r_events: Vec<_> = res
            .events
            .iter()
            .filter(|e| e.candidate_name == "R")
            .collect();
        assert!(
            r_events.len() >= 2,
            "Expected at least 2 events for multi-way mosaic child R"
        );

        let donors: Vec<_> = r_events.iter().map(|e| e.donor_name.as_str()).collect();
        assert!(
            donors.contains(&"P2"),
            "Expected P2 to be identified as donor"
        );
        assert!(
            donors.contains(&"P3"),
            "Expected P3 to be identified as donor"
        );
    }

    #[test]
    fn test_resolve_donor_chains_and_grouping() {
        let taxa = vec!["H".into(), "L".into(), "X".into(), "I".into()];
        let l = 1000;
        let data = vec![1u8; 4 * l];
        let aln = Alignment::new(taxa, l, data);

        let e1 = crate::types::RecombinationEvent {
            event_id: 1,
            isolates: vec!["H".into()],
            candidate_idx: 0,
            candidate_name: "H".into(),
            home_idx: Some(3),
            home_name: "I".into(),
            donor_idx: Some(1),
            donor_name: "L".into(),
            is_ghost_donor: false,
            is_crossover: false,
            u1: 1,
            u2: 500,
            u1_continuous: 1.0,
            u2_continuous: 500.0,
            tract_length: 500,
            s_informative: 50,
            p_fisher: 1e-10,
            z_phys: 8.0,
            d_flank_home: 0.05,
            d_flank_donor: 0.30,
            d_tract_home: 0.30,
            d_tract_donor: 0.02,
            is_verified: true,
        };

        let e2 = crate::types::RecombinationEvent {
            event_id: 2,
            isolates: vec!["L".into()],
            candidate_idx: 1,
            candidate_name: "L".into(),
            home_idx: Some(3),
            home_name: "I".into(),
            donor_idx: Some(2),
            donor_name: "X".into(),
            is_ghost_donor: false,
            is_crossover: false,
            u1: 1,
            u2: 500,
            u1_continuous: 1.0,
            u2_continuous: 500.0,
            tract_length: 500,
            s_informative: 60,
            p_fisher: 1e-15,
            z_phys: 9.0,
            d_flank_home: 0.04,
            d_flank_donor: 0.35,
            d_tract_home: 0.35,
            d_tract_donor: 0.01,
            is_verified: true,
        };

        let mut events = vec![e1, e2];
        resolve_donor_chains(&mut events, &aln);

        // Both should now cite donor X
        assert_eq!(events[0].donor_name, "X");
        assert_eq!(events[1].donor_name, "X");

        // Grouping should merge H and L into a single ancestral event
        let grouped = group_ancestral_events(events, &aln);
        assert_eq!(
            grouped.len(),
            1,
            "Expected H and L to merge into a single co-recombinant event"
        );
        assert_eq!(grouped[0].isolates, vec!["H", "L"]);
        assert_eq!(grouped[0].home_name, "I");
        assert_eq!(grouped[0].donor_name, "X");
        assert_eq!(grouped[0].u1, 1);
        assert_eq!(grouped[0].u2, 500);
        assert_eq!(
            grouped[0].p_fisher, 1e-15,
            "Strongest p-value should be preserved"
        );
        assert_eq!(
            grouped[0].z_phys, 9.0,
            "Highest z-score should be preserved"
        );
    }

    #[test]
    fn test_tier2_and_tier3_stringency_controls() {
        let l = 2000;
        let taxa = vec![
            "R".into(),
            "P1".into(),
            "P2".into(),
            "P3".into(),
            "O".into(),
        ];
        let n = taxa.len();
        let mut data = vec![1u8; n * l];

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
        // R has a primary 400 bp tract from P2 [400..800] and a moderate 90 bp tract from P3 [1200..1290]
        for u in 0..l {
            if u >= 400 && u < 800 {
                data[0 * l + u] = data[2 * l + u];
            } else if u >= 1200 && u < 1290 {
                data[0 * l + u] = data[3 * l + u];
            } else {
                data[0 * l + u] = data[1 * l + u];
            }
        }
        let aln = Alignment::new(taxa, l, data);

        let res_relaxed = RhizAeonEngine::new()
            .with_tier2_stringency(0.40)
            .with_tier3_stringency(0.40)
            .scan(&aln);

        let res_default = RhizAeonEngine::new().scan(&aln);

        let res_strict = RhizAeonEngine::new()
            .with_tier2_stringency(2.50)
            .with_tier3_stringency(2.50)
            .scan(&aln);

        assert!(
            res_relaxed.events.len() >= res_default.events.len()
                && res_default.events.len() >= res_strict.events.len(),
            "Event count must be non-increasing with stringency: relaxed={}, default={}, strict={}",
            res_relaxed.events.len(),
            res_default.events.len(),
            res_strict.events.len()
        );
        assert!(
            res_relaxed.events.len() > res_strict.events.len(),
            "Relaxed stringency must retain more tracts than high stringency (relaxed={}, strict={})",
            res_relaxed.events.len(),
            res_strict.events.len()
        );
    }
}
