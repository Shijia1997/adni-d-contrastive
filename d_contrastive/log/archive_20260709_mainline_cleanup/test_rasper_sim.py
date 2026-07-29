"""Self-check for the RASPER implementation (no torch, no data needed).

Run: python test_rasper_sim.py
Verifies (1) the concordance gradients match finite differences, and (2) the
paper's qualitative claim -- when the internal and external models share a
ranking but differ greatly in score scale, borrowing the external RANKING beats
ridge (Henderson 2026, Settings 3/5/7).
"""
import numpy as np
from sklearn.linear_model import Ridge

from rasper import RASPER, kendall_concordance, spearman_concordance


def test_gradients():
    rng = np.random.default_rng(0)
    n, p = 30, 4
    X = rng.standard_normal((n, p))
    r = rng.permutation(n).astype(float) + 1
    b = rng.standard_normal(p)
    for fn in (kendall_concordance, spearman_concordance):
        _, g = fn(b, X, r, nu=0.5)
        gnum = np.zeros(p)
        for k in range(p):
            e = np.zeros(p); e[k] = 1e-6
            gnum[k] = (fn(b + e, X, r, 0.5)[0] - fn(b - e, X, r, 0.5)[0]) / 2e-6
        err = np.abs(g - gnum).max()
        assert err < 1e-5, f"{fn.__name__} gradient mismatch {err}"
        print(f"  {fn.__name__:22s} grad err = {err:.2e}  OK")


def test_paper_scenario():
    def trial(seed):
        rg = np.random.default_rng(seed)
        nI, nte, p = 60, 500, 5
        beta_I = np.array([1., .8, .6, .4, .2])
        beta_E = 8.0 * beta_I + rg.standard_normal(p) * 0.3   # same order, big scale gap
        Xtr, Xte = rg.standard_normal((nI, p)), rg.standard_normal((nte, p))
        ytr = Xtr @ beta_I + rg.standard_normal(nI)
        mu_te = Xte @ beta_I
        r_ext = np.argsort(np.argsort(Xtr @ beta_E)).astype(float) + 1
        rid = Ridge(alpha=5.0).fit(Xtr, ytr)
        mse_rid = np.mean((Xte @ rid.coef_ + rid.intercept_ - mu_te) ** 2)
        est = RASPER(family="gaussian", penalty="kendall", lam=5.0, alpha=5.0).fit(Xtr, ytr, r_ext)
        mse_ras = np.mean((est.b0_ + Xte @ est.beta_ - mu_te) ** 2)
        return mse_rid, mse_ras
    res = np.array([trial(s) for s in range(20)])
    wins = int((res[:, 1] < res[:, 0]).sum())
    print(f"  MSE ridge={res[:,0].mean():.3f}  RASPER={res[:,1].mean():.3f}  "
          f"RASPER wins {wins}/20")
    assert wins >= 16, "RASPER should win the large-score-distance scenario"


if __name__ == "__main__":
    print("gradient check:")
    test_gradients()
    print("paper scenario:")
    test_paper_scenario()
    print("ALL RASPER TESTS PASSED")
