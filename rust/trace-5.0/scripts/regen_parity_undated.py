#!/usr/bin/env python3
"""Regenerate the undated-sequence parity golden file (tests/test_parity_undated.rs).

The fixture lives in crates/trace50-core/tests/fixtures/parity_undated/. It is
the parity cohort (fixtures/parity/subset.fasta) with one sequence made
undated: the first (sorted) member of the largest static 1.5% cluster in
fixtures/parity/golden.json is renamed UNDATED_SEQ_X, a header that
run_danno.parse_date cannot date. Sequence rows and order are unchanged, so
the tn93 pair set and distances are the parity fixture's (checked here by
re-running tn93 on the renamed file).

Every golden value is the output of an external tool run by this script:

  1. tn93 -q -t 1.0 -a RMYSKB on subset_undated.fasta; its distances must equal
     fixtures/parity/golden_distances.csv row for row (else exit).
  2. validation/run_danno.py, unmodified, on subset_undated.fasta + that CSV
     (Python path (b)). It drops the undated sequence before DANNO and before
     its static arm ('static_<t>_on_dated_subset'). It runs in-process so the
     supported dyads (book_em_danno) and the edge lists given to
     run_danno.topology are captured.
  3. trace50.chin.ChinEstimator(1.5) on (n_dated, E) and on the DANNO cluster
     sizes. run_danno has no CHIN arm; n_dated is used because only dated
     sequences can carry a DANNO edge.
  4. golden.json, then MANIFEST.sha256.

Usage: python3 scripts/regen_parity_undated.py
Needs tn93 (v1.1.0), numpy, scipy.
"""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regen_parity_golden as base  # noqa: E402  (shared helpers, same directory)

PARITY = base.FIXTURE
OUT = os.path.join(os.path.dirname(PARITY), 'parity_undated')
UNDATED_ID = 'UNDATED_SEQ_X'


def main():
    os.makedirs(OUT, exist_ok=True)
    out = lambda name: os.path.join(OUT, name)  # noqa: E731
    rd = base.load_run_danno()

    golden = json.load(open(os.path.join(PARITY, 'golden.json')))
    giant = max(golden['static_partition'], key=lambda c: (len(c), c))
    renamed = sorted(giant)[0]
    assert rd.parse_date(UNDATED_ID) == (None, None)

    recs = base.read_fasta(os.path.join(PARITY, 'subset.fasta'))
    assert renamed in {h for h, _ in recs}
    recs = [(UNDATED_ID if h == renamed else h, s) for h, s in recs]
    with open(out('subset_undated.fasta'), 'w') as fh:
        for h, s in recs:
            fh.write(f'>{h}\n{s}\n')
    ids = [h for h, _ in recs]
    row = {h: i for i, h in enumerate(ids)}

    # ---- 1. tn93: same pairs and distances as the parity fixture
    raw_csv = out('_tn93_RMYSKB.csv')
    subprocess.run(['tn93', '-q', '-t', '1.0', '-a', 'RMYSKB', '-o', raw_csv, out('subset_undated.fasta')],
                   check=True, stdout=subprocess.DEVNULL)
    with open(raw_csv) as fh:
        rows = list(csv.reader(fh))[1:]
    got = sorted((min(row[a], row[b]), max(row[a], row[b]), d) for a, b, d in rows)
    with open(os.path.join(PARITY, 'golden_distances.csv')) as fh:
        want = sorted((int(i), int(j), d) for i, j, d in list(csv.reader(fh))[1:])
    if got != want:
        sys.exit('tn93 distances on the renamed file differ from fixtures/parity/golden_distances.csv')
    print(f'[tn93] {len(got)} pairs, identical to fixtures/parity/golden_distances.csv')

    # ---- 2. run_danno.py (unmodified), with book_em_danno / topology observed
    captured = {}
    orig_book = rd.DannoEstimator.book_em_danno
    orig_topology = rd.topology

    def book(self, *a, **kw):
        sig = orig_book(self, *a, **kw)
        captured['sig'] = sig
        return sig

    def topology(edges):
        captured.setdefault('topology_edges', []).append(list(edges))
        return orig_topology(edges)

    rd.DannoEstimator.book_em_danno = book
    rd.topology = topology
    danno_json = out('_run_danno.json')
    log = io.StringIO()
    old_argv = sys.argv
    sys.argv = ['run_danno.py', '--msa', out('subset_undated.fasta'), '--tn93', raw_csv, '--out', danno_json]
    try:
        with redirect_stdout(log):
            rd.main()
    finally:
        sys.argv = old_argv
        rd.DannoEstimator.book_em_danno = orig_book
        rd.topology = orig_topology
    log_text = log.getvalue().replace(OUT + os.sep, '')
    with open(out('run_danno.log'), 'w') as fh:
        fh.write(log_text)
    print(log_text, end='')
    res = json.load(open(danno_json))
    static_key = f'static_{base.STATIC_T}_on_dated_subset'
    assert static_key in res, sorted(res)

    danno_named, static_named = captured['topology_edges']
    sig = [e for e in captured['sig'] if e['bayes_factor'] > 0.0]
    assert len(sig) == len(danno_named) == res['danno']['edges']
    dated = sorted(h for h in ids if h != UNDATED_ID)  # run_danno's matrix order
    assert res['n_dated'] == len(dated) == len(ids) - 1
    supported = []
    for e, (na, nb) in zip(sig, danno_named):
        a, b = dated[e['idx1']], dated[e['idx2']]
        assert (a, b) == (na, nb)
        a, b = sorted((a, b))
        supported.append({'id1': a, 'id2': b, 'distance': e['dist'], 'delta_t': e['delta_t'],
                          'bayes_factor': e['bayes_factor'], 'pep': e['pep'], 'q_value': e['q_value']})
    supported.sort(key=lambda r: (r['id1'], r['id2']))
    danno_part = base.canon_partition(base.components(sorted({x for e in danno_named for x in e}), danno_named))
    static_part = base.canon_partition(base.components(sorted({x for e in static_named for x in e}), static_named))
    assert [len(c) for c in sorted(static_part, key=len, reverse=True)] == res[static_key]['cluster_sizes']
    assert [len(c) for c in sorted(danno_part, key=len, reverse=True)] == res['danno']['cluster_sizes']

    # ---- 3. CHIN on the dated sequences
    sys.path.insert(0, base.trace50_oracle_path(base.CODE_ROOT))
    from trace50.chin import ChinEstimator
    ch = ChinEstimator(R0_default=1.5)
    n_dated, e_obs = len(dated), len(supported)
    ap_ = ch.estimate_active_population(n_dated, e_obs, R0=1.5)
    chin = {'n': n_dated, 'e': e_obs, 'r0': 1.5,
            'active_population': {k: float(ap_[k]) for k in ('N_act', 'ci_lower', 'ci_upper', 'relative_error')},
            'theta': float(ch.estimate_composite_parameter(n_dated, e_obs)),
            'rho': float(ch.estimate_sampling_fraction(n_dated, e_obs, R0=1.5))}

    import numpy, scipy
    out_json = {
        'generated_by': 'scripts/regen_parity_undated.py',
        'tools': {'tn93': base.tool_version(['tn93', '--version']), 'python': sys.version.split()[0],
                  'numpy': numpy.__version__, 'scipy': scipy.__version__},
        'renamed': {'from': renamed, 'to': UNDATED_ID},
        'n_sequences': len(ids),
        'n_dated': n_dated,
        'seq_len': res['seq_len_used'],
        'calibrated': res['calibrated'],
        'supported_edges': supported,
        'danno_partition': danno_part,
        'static_threshold': base.STATIC_T,
        'static_partition_on_dated_subset': static_part,
        'chin': chin,
    }
    with open(out('golden.json'), 'w') as fh:
        json.dump(out_json, fh, indent=1)
        fh.write('\n')
    for tmp in (raw_csv, danno_json):
        if os.path.exists(tmp):
            os.remove(tmp)
    sizes = sorted((len(c) for c in static_part), reverse=True)
    print(f'[golden] renamed {renamed} -> {UNDATED_ID}; n_dated={n_dated} supported={len(supported)} '
          f'danno_clusters={len(danno_part)} static_on_dated: {len(static_part)} clusters, '
          f'{sum(sizes)} nodes, max {sizes[0]}; N_act(n_dated)={chin["active_population"]["N_act"]!r} '
          f'rho={chin["rho"]!r}')

    files = sorted(f for f in os.listdir(OUT) if f not in ('MANIFEST.sha256',) and not f.startswith('_'))
    with open(out('MANIFEST.sha256'), 'w') as fh:
        for f in files:
            with open(out(f), 'rb') as g:
                fh.write(f'{hashlib.sha256(g.read()).hexdigest()}  {f}\n')


if __name__ == '__main__':
    main()
