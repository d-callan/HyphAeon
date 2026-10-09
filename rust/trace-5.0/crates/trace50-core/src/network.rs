//! Network topology, connected components, and cluster analysis for TRACE-5.0.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Disjoint Set Union (Union-Find) data structure for connected component analysis.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    pub fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while root != self.parent[root] {
            root = self.parent[root];
        }
        let mut curr = i;
        while curr != root {
            let next = self.parent[curr];
            self.parent[curr] = root;
            curr = next;
        }
        root
    }

    pub fn union(&mut self, i: usize, j: usize) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i == root_j {
            return false;
        }

        if self.rank[root_i] < self.rank[root_j] {
            self.parent[root_i] = root_j;
            self.size[root_j] += self.size[root_i];
        } else if self.rank[root_i] > self.rank[root_j] {
            self.parent[root_j] = root_i;
            self.size[root_i] += self.size[root_j];
        } else {
            self.parent[root_j] = root_i;
            self.size[root_i] += self.size[root_j];
            self.rank[root_i] += 1;
        }
        true
    }

    pub fn component_size(&mut self, i: usize) -> usize {
        let root = self.find(i);
        self.size[root]
    }
}

/// A transmission cluster extracted from connected components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub cluster_id: usize,
    pub size: usize,
    pub members: Vec<usize>,
    pub mean_dist: f64,
    pub time_span_years: f64,
}

/// Analyzes connected components from an edge list.
pub fn extract_clusters(
    n_nodes: usize,
    edges: &[(usize, usize, f64)],
    dates: &[f64],
) -> (Vec<Cluster>, Vec<usize>, usize) {
    let mut uf = UnionFind::new(n_nodes);
    for &(u, v, _) in edges {
        uf.union(u, v);
    }

    // Group members by component root
    let mut comp_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n_nodes {
        let root = uf.find(i);
        comp_map.entry(root).or_default().push(i);
    }

    let mut clusters = Vec::new();
    let mut cluster_assignments = vec![0usize; n_nodes];
    let mut max_cluster_size = 0usize;

    let mut cluster_id = 1usize;
    // Only return clusters of size >= 2
    let mut sorted_comps: Vec<(usize, Vec<usize>)> = comp_map.into_iter().collect();
    sorted_comps.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (_, members) in sorted_comps {
        let size = members.len();
        if size > max_cluster_size {
            max_cluster_size = size;
        }

        if size >= 2 {
            for &m in &members {
                cluster_assignments[m] = cluster_id;
            }

            // Calculate mean internal distance and time span
            let min_date = members.iter().map(|&m| dates[m]).fold(f64::INFINITY, f64::min);
            let max_date = members.iter().map(|&m| dates[m]).fold(f64::NEG_INFINITY, f64::max);
            let time_span = if max_date >= min_date { max_date - min_date } else { 0.0 };

            clusters.push(Cluster {
                cluster_id,
                size,
                members,
                mean_dist: 0.0,
                time_span_years: time_span,
            });
            cluster_id += 1;
        }
    }

    (clusters, cluster_assignments, max_cluster_size)
}
