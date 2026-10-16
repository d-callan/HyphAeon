#!/usr/bin/env python3
"""Regenerate the Rust-vs-Python parity golden files (tests/test_parity_golden.rs).

The fixture lives in crates/trace50-core/tests/fixtures/parity/. Every golden
value is the output of an external tool run by this script; nothing is typed in.

  1. Cohort (only with --select, or when subset.fasta is missing): a fixed
     subset of the Washington, DC alignment (Perez-Losada 2017,
     validation/results/Perez-Losada-2017-28961263/msa.fasta). Rule:
       a. connected components of the 151 Python-supported DC edges at L = 1023
          (parity_2026-10-12/ingest/py_L1023_edges.csv); keep every member of
          every component with >= MIN_CLUSTER (3) members;
       b. background: the remaining DC sequences ordered by sha256(header),
          first N_BACKGROUND (115) of them;
       c. written in DC file order, rows unmodified (alignment width kept).
     meta.csv holds each sequence's decimal date from run_danno.parse_date.
  2. tn93 -q -t 1.0 -a RMYSKB (default -l 100) on subset.fasta
     -> tn93_RMYSKB.csv.gz (the tn93 output with its data rows sorted, since
        tn93's row order varies between runs; gzip mtime 0) and
        golden_distances.csv (same rows as i,j,distance with i<j FASTA row
        indices, sorted; the distance field is tn93's text, unchanged).
  3. validation/run_danno.py, unmodified, on subset.fasta + the tn93 CSV
     (Python path (b): L = median non-gap). It is run in-process so that the
     supported dyads returned by DannoEstimator.book_em_danno (BF, PEP, q) and
     the edge lists passed to run_danno.topology are captured as they are
     computed; run_danno.json and run_danno.log are its own output.
  4. hivnetworkcsv -f plain -t 0.015 -c on the tn93 CSV: its cluster list must
     equal the static partition from run_danno's static arm (checked here).
  5. trace50.chin.ChinEstimator(1.5) on (n, E) and on the DANNO cluster sizes.
  6. golden.json, then MANIFEST.sha256 over every fixture file.

Usage (from anywhere):
  python3 scripts/regen_parity_golden.py            # regenerate goldens for the committed subset
  python3 scripts/regen_parity_golden.py --select   # also re-derive subset.fasta / meta.csv from DC
Needs tn93 (v1.1.0), hivnetworkcsv (hivclustering 1.9.2), numpy, scipy.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
RUST_ROOT = os.path.dirname(HERE)                      # repo root (Cargo workspace)
CODE_ROOT = RUST_ROOT                                   # code/
PROJECT_ROOT = os.path.dirname(CODE_ROOT)              # trace5/
FIXTURE = os.path.join(RUST_ROOT, 'crates', 'trace50-core', 'tests', 'fixtures', 'parity')
def trace50_oracle_path(code_root):
    """Python trace50 package (removed from this repo in 98f44ba):
    $TRACE50_PY_ORACLE, else code_root while it still ships trace50/, else the
    frozen copy at <trace5>/validation/py_oracle (see its README.md)."""
    env = os.environ.get('TRACE50_PY_ORACLE')
    if env:
        return env
    if os.path.exists(os.path.join(code_root, 'trace50', '__init__.py')):
        return code_root
    return os.path.join(os.path.dirname(code_root), 'validation', 'py_oracle')


RUN_DANNO = os.path.join(PROJECT_ROOT, 'validation', 'run_danno.py')
DC_DIR = os.path.join(PROJECT_ROOT, 'validation', 'results', 'Perez-Losada-2017-28961263')
DC_MSA = os.path.join(DC_DIR, 'msa.fasta')
DC_EDGES = os.path.join(DC_DIR, 'parity_2026-10-12', 'ingest', 'py_L1023_edges.csv')

MIN_CLUSTER = 3
N_BACKGROUND = 115
STATIC_T = 0.015


def read_fasta(path):
    recs, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if name is not None:
                    recs.append((name, ''.join(buf)))
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        recs.append((name, ''.join(buf)))
    return recs


def components(nodes, edges):
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = defaultdict(set)
    for x in nodes:
        groups[find(x)].add(x)
    return [g for g in groups.values() if len(g) >= 2]


def canon_partition(comps):
    return sorted(sorted(c) for c in comps)


def load_run_danno():
    spec = importlib.util.spec_from_file_location('run_danno', RUN_DANNO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_subset(args, rd):
    recs = read_fasta(args.dc_msa)
    with open(args.dc_edges) as fh:
        edges = [(r['id1'], r['id2']) for r in csv.DictReader(fh)]
    nodes = sorted({x for e in edges for x in e})
    keep = set()
    comps = components(nodes, edges)
    for c in comps:
        if len(c) >= MIN_CLUSTER:
            keep |= c
    n_clusters = sum(1 for c in comps if len(c) >= MIN_CLUSTER)
    rest = sorted((hashlib.sha256(h.encode()).hexdigest(), h) for h, _ in recs if h not in keep)
    background = {h for _, h in rest[:N_BACKGROUND]}
    chosen = [(h, s) for h, s in recs if h in keep or h in background]
    with open(os.path.join(args.out, 'subset.fasta'), 'w') as fh:
        for h, s in chosen:
            fh.write(f'>{h}\n{s}\n')
    with open(os.path.join(args.out, 'meta.csv'), 'w') as fh:
        fh.write('id,date\n')
        for h, _ in chosen:
            d, _p = rd.parse_date(h)
            if d is None:
                sys.exit(f'undated sequence selected: {h}')
            fh.write(f'{h},{d!r}\n')
    print(f'[select] {len(keep)} members of {n_clusters} oracle clusters (size >= {MIN_CLUSTER}) '
          f'+ {len(background)} sha256-ordered background = {len(chosen)} sequences')


def tool_version(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        return (out.stdout + out.stderr).strip().splitlines()[0]
    except Exception as e:  # recorded, not fatal
        return f'unavailable ({e})'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--select', action='store_true', help='re-derive subset.fasta and meta.csv from DC')
    ap.add_argument('--dc-msa', default=DC_MSA)
    ap.add_argument('--dc-edges', default=DC_EDGES)
    ap.add_argument('--out', default=FIXTURE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = lambda name: os.path.join(args.out, name)  # noqa: E731

    rd = load_run_danno()
    if args.select or not os.path.exists(out('subset.fasta')):
        select_subset(args, rd)

    recs = read_fasta(out('subset.fasta'))
    ids = [h for h, _ in recs]
    row = {h: i for i, h in enumerate(ids)}

    # ---- 2. tn93
    raw_csv = os.path.join(args.out, '_tn93_RMYSKB.csv')
    tn93_cmd = ['tn93', '-q', '-t', '1.0', '-a', 'RMYSKB', '-o', raw_csv, out('subset.fasta')]
    subprocess.run(tn93_cmd, check=True, stdout=subprocess.DEVNULL)
    # tn93 is multithreaded and writes pairs in a run-dependent order; sort the
    # data rows (header first) so the committed files are reproducible.
    with open(raw_csv) as fh:
        lines = fh.read().splitlines()
    raw = ('\n'.join([lines[0]] + sorted(lines[1:])) + '\n').encode()
    with open(out('tn93_RMYSKB.csv.gz'), 'wb') as fh:
        with gzip.GzipFile(filename='tn93_RMYSKB.csv', mode='wb', fileobj=fh, mtime=0) as gz:
            gz.write(raw)
    n_pairs = 0
    with open(out('golden_distances.csv'), 'w') as fh:
        fh.write('i,j,distance\n')
        rows = []
        for a, b, d in list(csv.reader(io.StringIO(raw.decode())))[1:]:
            i, j = row[a], row[b]
            rows.append((min(i, j), max(i, j), d))
        for i, j, d in sorted(rows):
            fh.write(f'{i},{j},{d}\n')
            n_pairs += 1
    print(f'[tn93] {n_pairs} pairs reported (tn93 -t 1.0 -a RMYSKB -l 100)')

    # ---- 3. run_danno.py (unmodified), with book_em_danno / topology observed
    captured = {}
    orig_book = rd.DannoEstimator.book_em_danno
    orig_topology = rd.topology

    def book(self, *a, **kw):
        sig = orig_book(self, *a, **kw)
        captured['est'] = self
        captured['sig'] = sig
        return sig

    def topology(edges):
        captured.setdefault('topology_edges', []).append(list(edges))
        return orig_topology(edges)

    rd.DannoEstimator.book_em_danno = book
    rd.topology = topology
    danno_json = out('run_danno.json')
    argv = ['run_danno.py', '--msa', out('subset.fasta'), '--tn93', raw_csv, '--out', danno_json]
    log = io.StringIO()
    old_argv = sys.argv
    sys.argv = argv
    try:
        with redirect_stdout(log):
            rd.main()
    finally:
        sys.argv = old_argv
        rd.DannoEstimator.book_em_danno = orig_book
        rd.topology = orig_topology
    log_text = log.getvalue().replace(args.out + os.sep, '')
    with open(out('run_danno.log'), 'w') as fh:
        fh.write(log_text)
    print(log_text, end='')
    res = json.load(open(danno_json))
    # keep the committed JSON free of absolute scratch paths (it has none) and stable
    danno_named, static_named = captured['topology_edges']
    sig = [e for e in captured['sig'] if e['bayes_factor'] > 0.0]
    assert len(sig) == len(danno_named) == res['danno']['edges']
    dated = sorted(ids)  # run_danno indexes the matrix by sorted dated IDs; all are dated here
    assert res['n_dated'] == len(ids)
    supported = []
    for e, (na, nb) in zip(sig, danno_named):
        a, b = dated[e['idx1']], dated[e['idx2']]
        assert (a, b) == (na, nb)
        a, b = sorted((a, b))
        supported.append({'id1': a, 'id2': b, 'distance': e['dist'], 'delta_t': e['delta_t'],
                          'bayes_factor': e['bayes_factor'], 'pep': e['pep'], 'q_value': e['q_value']})
    supported.sort(key=lambda r: (r['id1'], r['id2']))
    danno_part = canon_partition(components(sorted({x for e in danno_named for x in e}), danno_named))
    static_part = canon_partition(components(sorted({x for e in static_named for x in e}), static_named))
    assert [len(c) for c in sorted(danno_part, key=len, reverse=True)] == res['danno']['cluster_sizes']

    # ---- 4. hivnetworkcsv static partition on the same distances
    hiv_clusters = out('_hivnetworkcsv_clusters.csv')
    hiv_cmd = ['hivnetworkcsv', '-f', 'plain', '-i', raw_csv, '-t', str(STATIC_T), '-c', hiv_clusters,
               '-J', '-O', out('_hivnetworkcsv.json'), '--no-degree-fit']
    subprocess.run(hiv_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hc = defaultdict(set)
    with open(hiv_clusters) as fh:
        for r in csv.DictReader(fh):
            hc[r['ClusterID']].add(r['SequenceID'])
    hiv_part = canon_partition(hc.values())
    if hiv_part != static_part:
        sys.exit('hivnetworkcsv partition differs from the run_danno static arm')
    print(f'[hivnetworkcsv] {len(hiv_part)} clusters at t={STATIC_T}, identical to run_danno static arm')

    # ---- 5. CHIN (Python ChinEstimator)
    sys.path.insert(0, trace50_oracle_path(CODE_ROOT))
    from trace50.chin import ChinEstimator
    ch = ChinEstimator(R0_default=1.5)
    n, e_obs = len(ids), len(supported)
    chin = {'n': n, 'e': e_obs, 'r0': 1.5}
    if e_obs > 0:
        ap_ = ch.estimate_active_population(n, e_obs, R0=1.5)
        chin['active_population'] = {k: float(ap_[k]) for k in ('N_act', 'ci_lower', 'ci_upper', 'relative_error')}
    sizes = sorted((len(c) for c in danno_part), reverse=True)
    b = ch.test_borel_branching_decay(sizes, 12)
    chin['borel'] = {'slope': float(b['slope']), 'r_squared': float(b['r_squared']),
                     'p_value': float(b['p_value']), 'n_clusters': int(b['n_clusters']),
                     'n_megaclusters': int(b['n_megaclusters']),
                     'rejects_uniform_sampling': bool(b['rejects_uniform_sampling'])}

    import numpy, scipy
    golden = {
        'generated_by': 'scripts/regen_parity_golden.py',
        'tools': {
            'tn93': tool_version(['tn93', '--version']),
            'hivclustering': tool_version([sys.executable, '-c',
                                           'import importlib.metadata as m; print(m.version("hivclustering"))']),
            'python': sys.version.split()[0], 'numpy': numpy.__version__, 'scipy': scipy.__version__,
        },
        'commands': {
            'tn93': 'tn93 -q -t 1.0 -a RMYSKB -o tn93_RMYSKB.csv subset.fasta',
            'run_danno': 'validation/run_danno.py --msa subset.fasta --tn93 tn93_RMYSKB.csv --out run_danno.json',
            'hivnetworkcsv': f'hivnetworkcsv -f plain -i tn93_RMYSKB.csv -t {STATIC_T} -c clusters.csv -J -O net.json --no-degree-fit',
        },
        'n_sequences': n,
        'alignment_width': res['msa_width'],
        'seq_len': res['seq_len_used'],
        'tn93_pairs': n_pairs,
        'params': res['params'],
        'calibrated': res['calibrated'],
        'supported_edges': supported,
        'danno_partition': danno_part,
        'static_threshold': STATIC_T,
        'static_partition': static_part,
        'chin': chin,
    }
    with open(out('golden.json'), 'w') as fh:
        json.dump(golden, fh, indent=1)
        fh.write('\n')
    for tmp in (raw_csv, hiv_clusters, out('_hivnetworkcsv.json')):
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f'[golden] L={golden["seq_len"]} width={golden["alignment_width"]} '
          f'supported={len(supported)} danno_clusters={len(danno_part)} max={max(sizes) if sizes else 0} '
          f'static_clusters={len(static_part)} p_bg={res["calibrated"]["p_bg"]!r} r_bg={res["calibrated"]["r_bg"]!r}')

    # ---- 6. manifest
    files = sorted(f for f in os.listdir(args.out) if f not in ('MANIFEST.sha256', 'README.md') and not f.startswith('_'))
    with open(out('MANIFEST.sha256'), 'w') as fh:
        for f in files:
            with open(out(f), 'rb') as g:
                fh.write(f'{hashlib.sha256(g.read()).hexdigest()}  {f}\n')


if __name__ == '__main__':
    main()
