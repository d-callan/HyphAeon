"""
Report: composition-only baseline vs model AUC.

This is NOT a pass/fail test. It generates a JSON artifact comparing the
model's ability to distinguish variable from invariable sites against a
trivial composition-only baseline (number of distinct amino acids per column).

The model's high all-sites AUC is largely driven by correctly identifying
invariable sites (which is easy — they have 1 distinct AA). The interesting
question is how much the model adds over the baseline on variable sites only.

The artifact is written to model_eval/_artifacts/composition_baseline.json
and uploaded by CI for the model team to review.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def test_composition_baseline_report(model, smc6_base, artifacts_dir):
    base = smc6_base
    a = base["a"][:, :, 0].numpy()  # [L, N] aa tokens
    tested = base["tested"]
    lrt = base["lrt"]
    inv = base["inv"]

    # Composition features
    n_distinct_aa = np.array([len(np.unique(r[r < 20])) for r in a])
    n_minor_aa = np.array([
        len(r[r < 20]) - np.bincount(r[r < 20], minlength=21).max()
        if (r < 20).any() else 0
        for r in a
    ])

    # All-sites AUC: variable (1) vs invariable (0)
    y_all = (~inv).astype(int)
    auc_model_all = roc_auc_score(y_all, lrt)
    auc_baseline_all = roc_auc_score(y_all, n_distinct_aa)

    # Variable-sites-only: does LRT correlate with MEME's positive selection
    # calls? We don't have MEME labels here, so we report the composition-LRT
    # correlation as a proxy for "how much does the model add over composition."
    if tested.sum() >= 3:
        rho_distinct, _ = spearmanr(n_distinct_aa[tested], lrt[tested])
        rho_minor, _ = spearmanr(n_minor_aa[tested], lrt[tested])
    else:
        rho_distinct = rho_minor = float("nan")

    report = {
        "dataset": "Smc6",
        "n_sites": int(len(lrt)),
        "n_variable": int(tested.sum()),
        "n_invariable": int(inv.sum()),
        "all_sites_auc": {
            "model": float(auc_model_all),
            "baseline_distinct_aa": float(auc_baseline_all),
        },
        "variable_sites_composition_correlation": {
            "spearman_lrt_vs_n_distinct_aa": float(rho_distinct),
            "spearman_lrt_vs_n_minor_aa": float(rho_minor),
        },
        "interpretation": (
            "all_sites_auc.model is dominated by the easy task of "
            "identifying invariable sites. The baseline_distinct_aa AUC "
            "shows how much of that is trivial. The variable_sites "
            "correlations show how much the model's LRT on variable sites "
            "is explained by composition alone (high rho = model adds "
            "little over counting amino acids)."
        ),
    }

    out = os.path.join(artifacts_dir, "composition_baseline.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    # No assertion — this is a report. Print for visibility in CI logs.
    print(f"\n[report] {out}")
    print(json.dumps(report, indent=2))
