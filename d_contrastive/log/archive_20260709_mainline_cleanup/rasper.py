"""RASPER: Rank-ASsociated PEnalized Regression (Henderson 2026).

Pure numpy/scipy implementation -- no torch, runs on CPU. Borrows an external
risk *ranking* (not scores or coefficients) to regularize a small internal risk
model.

Objective (paper eq. 14):
    minimize_{b0, b}   L_I(b0, b; alpha)  -  lambda * log D(b; r_ext)
where
    L_I = GLM negative log-likelihood + (alpha/2) ||b||^2     (local objective)
    D   = smooth rank concordance between the internal score ranking implied by
          x_i^T b and the external ranks r_ext (paper Section 2.3).

We minimize the objective directly with L-BFGS-B (analytic gradient). For the
small internal cohorts here this is equivalent to, and simpler than, the paper's
MM/IRLS update, and reaches the same minimizer of eq. (14).
"""
import numpy as np
from scipy.optimize import minimize


def _sigmoid(x):
    # numerically stable logistic via tanh identity
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _bce_nll(eta, y):
    # sum_i [ log(1+exp(eta_i)) - y_i eta_i ], stable
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def _pair_diffs(X, nu):
    """A[i,j,:] = (x_i - x_j) / nu, shape (n, n, p)."""
    return (X[:, None, :] - X[None, :, :]) / nu


def kendall_concordance(beta, X, r_ext, nu):
    """Smooth Kendall concordance D_Ke (paper eq. 11) and d D / d beta.

    D = sum_{i!=j} w_ij g_nu((x_i - x_j)^T beta),
        w_ij = (2 I(r_i > r_j) - 1) / (n (n-1)),  g_nu = sigmoid(./nu).
    """
    n = X.shape[0]
    A = _pair_diffs(X, nu)            # (n,n,p)
    g = _sigmoid(A @ beta)           # (n,n)
    w = (2.0 * (r_ext[:, None] > r_ext[None, :]) - 1.0) / (n * (n - 1))
    np.fill_diagonal(w, 0.0)
    D = float((w * g).sum())
    grad = np.einsum("ij,ijp->p", w * (g * (1.0 - g)), A)
    return D, grad


def spearman_concordance(beta, X, r_ext, nu):
    """Smooth Spearman concordance D_Sp (paper eq. 9), strictly positive.

    D = (1 / 4n^2) sum_{i!=j} r_i g_nu((x_i - x_j)^T beta).
    """
    n = X.shape[0]
    A = _pair_diffs(X, nu)
    g = _sigmoid(A @ beta)
    w = np.tile(r_ext[:, None] / (4.0 * n * n), (1, n))
    np.fill_diagonal(w, 0.0)
    D = float((w * g).sum())
    grad = np.einsum("ij,ijp->p", w * (g * (1.0 - g)), A)
    return D, grad


_CONCORDANCE = {"kendall": kendall_concordance, "spearman": spearman_concordance}


class RASPER:
    """Rank-associated penalized regression.

    Parameters
    ----------
    family : 'binomial' (logistic, for conversion) or 'gaussian' (least squares).
    penalty : 'kendall' (default, matches the pure-rank W0 loss) or 'spearman'.
    lam : ranking-penalty strength (lambda in the paper). lam=0 -> ridge/logistic.
    alpha : L2 shrinkage on beta (not on the intercept).
    nu : smoothing scale of g_nu. None -> principled default 0.1 * ||beta_MLE||.
    """

    def __init__(self, family="binomial", penalty="kendall", lam=1.0, alpha=1.0,
                 nu=None, max_iter=500):
        self.family = family
        self.penalty = penalty
        self.lam = lam
        self.alpha = alpha
        self.nu = nu
        self.max_iter = max_iter

    # -- local objective (negative log-likelihood + L2) and its gradient --------
    def _local(self, params, X, y):
        b0, b = params[0], params[1:]
        eta = b0 + X @ b
        if self.family == "gaussian":
            r = y - eta
            L = 0.5 * float(r @ r)
            gb0, gb = -float(r.sum()), -(X.T @ r)
        elif self.family == "binomial":
            p = _sigmoid(eta)
            L = _bce_nll(eta, y)
            gb0, gb = float((p - y).sum()), X.T @ (p - y)
        else:
            raise ValueError(f"Unknown family: {self.family}")
        # L2 on beta only
        L += 0.5 * self.alpha * float(b @ b)
        gb = gb + self.alpha * b
        return L, gb0, gb

    def _objective(self, params, X, y, r_ext):
        b0, b = params[0], params[1:]
        L, gb0, gb = self._local(params, X, y)
        D, gD = _CONCORDANCE[self.penalty](b, X, r_ext, self.nu_)
        D = max(D, 1e-8)
        L = L - self.lam * np.log(D)
        gb = gb - self.lam * (gD / D)
        return L, np.concatenate([[gb0], gb])

    def _fit_mle(self, X, y):
        """Unpenalized (lam=0, alpha=0) fit, to set nu = 0.1 ||beta_MLE||."""
        p = X.shape[1]
        save_lam, save_alpha = self.lam, self.alpha
        self.lam, self.alpha = 0.0, 0.0
        self.nu_ = 1.0  # unused when lam=0
        res = minimize(lambda par: self._objective(par, X, y, np.zeros(len(y))),
                       np.zeros(p + 1), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.lam, self.alpha = save_lam, save_alpha
        return res.x[1:]

    def fit(self, X, y, r_ext):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        r_ext = np.asarray(r_ext, float)
        p = X.shape[1]
        beta_mle = self._fit_mle(X, y)
        self.nu_ = self.nu if self.nu is not None else max(0.1 * np.linalg.norm(beta_mle), 1e-3)
        x0 = np.concatenate([[0.0], beta_mle])
        res = minimize(self._objective, x0, args=(X, y, r_ext), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.b0_, self.beta_ = float(res.x[0]), res.x[1:]
        self.success_ = bool(res.success)
        return self

    def decision_function(self, X):
        """Risk score (ranking). Intercept is irrelevant for ranking/AUC."""
        return np.asarray(X, float) @ self.beta_

    def predict_proba(self, X):
        return _sigmoid(self.b0_ + self.decision_function(X))


def _cv_folds(y, n_splits, seed, stratify):
    """Index folds; stratified by y for binomial so each fold has converters."""
    n = len(y)
    rng = np.random.default_rng(seed)
    if n_splits is None or n_splits >= n:          # leave-one-out
        return [np.array([i]) for i in range(n)]
    if stratify:
        folds = [[] for _ in range(n_splits)]
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            for k, i in enumerate(idx):
                folds[k % n_splits].append(i)
        return [np.array(sorted(f)) for f in folds if f]
    idx = rng.permutation(n)
    return [np.array(sorted(f)) for f in np.array_split(idx, n_splits)]


def select_lambda_alpha_cv(X, y, r_ext, family="binomial", penalty="kendall",
                           lambdas=None, alphas=None, nu=None, n_splits=5, seed=0):
    """Pick (lambda, alpha) by CV (paper Section 3.3).

    n_splits=None -> leave-one-out; otherwise stratified k-fold (faster and more
    stable than LOO for tiny converter counts). The held-out score is the local
    objective (no L2 term), as in the paper. lambda=0 reduces RASPER to
    ridge/logistic, so it is always in the grid.
    Returns (best_lambda, best_alpha, fitted_RASPER_on_full_data).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    r_ext = np.asarray(r_ext, float)
    if lambdas is None:
        lambdas = np.concatenate([[0.0], np.geomspace(1e-1, 1e2, 5)])
    if alphas is None:
        alphas = np.concatenate([[0.0], np.geomspace(1e-1, 1e1, 4)])
    folds = _cv_folds(y, n_splits, seed, stratify=(family == "binomial"))

    best = (np.inf, 0.0, 0.0)
    for lam in lambdas:
        for al in alphas:
            score, ok = 0.0, True
            for te in folds:
                tr = np.setdiff1d(np.arange(len(y)), te)
                try:
                    est = RASPER(family=family, penalty=penalty, lam=lam, alpha=al,
                                 nu=nu).fit(X[tr], y[tr], r_ext[tr])
                    eta = est.b0_ + X[te] @ est.beta_
                    if family == "gaussian":
                        score += 0.5 * float(((y[te] - eta) ** 2).sum())
                    else:
                        score += float(np.sum(np.logaddexp(0.0, eta) - y[te] * eta))
                except Exception:
                    ok = False
                    break
            if ok and score < best[0]:
                best = (score, lam, al)
    _, best_lam, best_al = best
    final = RASPER(family=family, penalty=penalty, lam=best_lam, alpha=best_al,
                   nu=nu).fit(X, y, r_ext)
    return best_lam, best_al, final
