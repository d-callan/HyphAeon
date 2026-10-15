"""Oracle for tests/test_chin_oracle.rs (C6: CHIN Borel statistics).

Runs the unmodified Python reference trace50.chin.ChinEstimator and
scipy.stats.linregress / scipy.stats.norm on cases.json (copied from
parity_2026-10-12/chin/cases.json: DC 151- and 163-edge DANNO cluster sizes,
the DC static sizes, and edge cases) and writes oracle.json.

    python make_fixture.py > oracle.json
"""
import json, math, os, sys
CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))


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


sys.path.insert(0, trace50_oracle_path(CODE_ROOT))
import numpy as np
from scipy import stats
from trace50.chin import ChinEstimator

here = os.path.dirname(os.path.abspath(__file__))
cases = json.load(open(os.path.join(here, "cases.json")))

out = {"chin": [], "linregress": [], "norm_ppf": []}
for c in cases:
    ch = ChinEstimator(R0_default=1.5)
    rec = {"name": c["name"], "n": c["n"], "e": c["e"], "r0": c["r0"], "sizes": c["sizes"]}
    if c["n"] > 0 and c["e"] > 0:  # Python returns inf/0 otherwise; Rust refuses (D4, tested elsewhere)
        ap = ch.estimate_active_population(c["n"], c["e"], R0=c["r0"])
        rec["active_population"] = {k: float(ap[k]) for k in ("N_act", "ci_lower", "ci_upper", "relative_error")}
    b = ch.test_borel_branching_decay(c["sizes"], 12)
    rec["borel"] = {
        "slope": float(b["slope"]), "r_squared": float(b["r_squared"]), "p_value": float(b["p_value"]),
        "n_clusters": int(b["n_clusters"]), "n_megaclusters": int(b["n_megaclusters"]),
        "max_cluster_size": int(b.get("max_cluster_size", 0)),
        "rejects_uniform_sampling": bool(b["rejects_uniform_sampling"]),
    }
    out["chin"].append(rec)

# Direct scipy.stats.linregress cases for the p-value path (deterministic inputs).
x30 = [float(i) for i in range(30)]
reg_cases = [
    ("n2_distinct", [1.0, 2.0], [0.5, 3.0]),
    ("n2_equal_y", [1.0, 2.0], [3.0, 3.0]),
    ("perfect_fit", [1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0]),
    ("flat_y", [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]),
    ("df28_strong", x30, [0.1 * v + math.sin(3.0 * v) for v in x30]),
    ("df28_weak", x30, [0.01 * v + math.sin(3.0 * v) for v in x30]),
]
for name, x, y in reg_cases:
    r = stats.linregress(x, y)
    out["linregress"].append({"name": name, "x": x, "y": y, "slope": float(r.slope),
                              "intercept": float(r.intercept), "r_value": float(r.rvalue),
                              "p_value": float(r.pvalue)})

for p in (1e-10, 0.001, 0.02, 0.025, 0.3, 0.5, 0.8, 0.975, 0.995, 1 - 1e-10):
    out["norm_ppf"].append({"p": p, "x": float(stats.norm.ppf(p))})

print(json.dumps(out, indent=1))
