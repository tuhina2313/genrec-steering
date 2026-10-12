"""CPU logistic probes, raw-coordinate conversion, mean-difference directions, projection diagnostics.

Inputs are arrays already joined by pair_id; this module never reads dataset files or labels
from validation/test splits.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def binary_diagnostics(y: np.ndarray, s: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    out = {"n": int(len(y)), "positives": pos, "negatives": neg,
           "prevalence": pos / len(y) if len(y) else None}
    if pos == 0 or neg == 0:
        out.update(auroc=None, auprc=None, not_estimable_reason="one class absent")
    else:
        out.update(auroc=float(roc_auc_score(y, s)), auprc=float(average_precision_score(y, s)))
    return out


@dataclass
class RawProbe:
    """Logistic probe expressed in raw residual coordinates: decision(h) = w_raw . h + b_raw."""

    w_raw: np.ndarray
    b_raw: float
    coef_std: np.ndarray
    intercept_std: float
    mean: np.ndarray
    scale: np.ndarray
    C: float
    classes: list
    converged: bool

    def decision(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float64) @ self.w_raw.astype(np.float64) + self.b_raw


def to_raw_coordinates(coef_std: np.ndarray, intercept_std: float, mean: np.ndarray, scale: np.ndarray):
    """For z = (h - mu) / scale and f = w.z + b:  w_raw = w / scale,  b_raw = b - sum(w * mu / scale)."""
    w_raw = coef_std / scale
    b_raw = float(intercept_std - np.sum(coef_std * mean / scale))
    return w_raw, b_raw


def fit_probe(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              C_grid: list[float], seed: int, max_iter: int = 5000) -> tuple[Optional[RawProbe], list[dict]]:
    """L2 logistic regression; scaler on the fit split only; C by probe-val AUROC (ties -> smaller C)."""
    X_fit = np.asarray(X_fit, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    scaler = StandardScaler().fit(X_fit)          # sklearn sets scale=1 for constant dimensions
    Zf, Zv = scaler.transform(X_fit), scaler.transform(X_val)
    grid = []
    for C in sorted(C_grid):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=seed).fit(Zf, y_fit)
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        diag = binary_diagnostics(y_val, clf.decision_function(Zv))
        grid.append({"C": C, "converged": converged, **{f"val_{k}": v for k, v in diag.items()}, "_clf": clf})
    scored = [g for g in grid if g["val_auroc"] is not None]
    if not scored:
        best = grid[0]   # strongest regularization when validation AUROC is not estimable
    else:
        top = max(g["val_auroc"] for g in scored)
        best = [g for g in scored if g["val_auroc"] >= top - 1e-12][0]   # grid sorted ascending: smaller C wins
    clf = best["_clf"]
    if list(clf.classes_) != [0, 1]:
        raise AssertionError(f"unexpected class order {clf.classes_}")
    coef, b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
    w_raw, b_raw = to_raw_coordinates(coef, b, scaler.mean_, scaler.scale_)
    probe = RawProbe(w_raw=w_raw, b_raw=b_raw, coef_std=coef, intercept_std=b, mean=scaler.mean_.copy(),
                     scale=scaler.scale_.copy(), C=best["C"], classes=[int(c) for c in clf.classes_],
                     converged=best["converged"])
    # raw vs standardized decision parity
    std_dec = clf.decision_function(Zf)
    raw_dec = probe.decision(X_fit)
    if not np.allclose(std_dec, raw_dec, rtol=1e-6, atol=1e-6 * max(1.0, np.abs(std_dec).max())):
        raise AssertionError("raw-coordinate probe decision does not match standardized decision")
    for g in grid:
        g.pop("_clf")
        g["selected"] = g["C"] == best["C"]
    return probe, grid


def mean_difference(X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """mean(h | y=1) - mean(h | y=0) on unstandardized fit activations (not normalized)."""
    X = np.asarray(X_fit, dtype=np.float64)
    y = np.asarray(y_fit).astype(int)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("mean difference needs both classes")
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError("zero or non-finite direction (failed fit)")
    return (v / n).astype(np.float32)


def activation_radius(X_fit: np.ndarray) -> float:
    """Median L2 norm of raw inner-fit activations at this layer (frozen from fitting data)."""
    return float(np.median(np.linalg.norm(np.asarray(X_fit, dtype=np.float64), axis=1)))


def projection_diagnostics(direction: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    proj = np.asarray(X, dtype=np.float64) @ np.asarray(direction, dtype=np.float64)
    return binary_diagnostics(y, proj)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fit_directions(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                   C_grid: list[float], seed: int, max_iter: int = 5000) -> dict:
    """Mean-difference and logistic-probe directions on exactly the same fit pairs.

    Returns unit vectors, the probe, radius, C grid diagnostics, and projection diagnostics.
    Raises ValueError for a failed (zero/non-finite) fit, which the caller records explicitly.
    """
    y_fit = np.asarray(y_fit).astype(int)
    md = unit(mean_difference(X_fit, y_fit))
    probe, grid = fit_probe(X_fit, y_fit, X_val, y_val, C_grid, seed, max_iter)
    pv = unit(probe.w_raw)
    return {
        "mean_diff": md, "probe": pv, "probe_model": probe, "C_grid": grid, "radius": activation_radius(X_fit),
        "support": {"fit_pos": int((y_fit == 1).sum()), "fit_neg": int((y_fit == 0).sum())},
        "diagnostics": {
            "mean_diff_projection_probe_val": projection_diagnostics(md, X_val, y_val),
            "probe_decision_probe_val": binary_diagnostics(y_val, probe.decision(X_val)),
            "probe_decision_fit": binary_diagnostics(y_fit, probe.decision(X_fit)),
            "cosine_mean_diff_probe": cosine(md, pv),
        },
    }


def shuffled_label_direction(X_fit: np.ndarray, y_fit: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return unit(mean_difference(X_fit, rng.permutation(np.asarray(y_fit).astype(int))))
