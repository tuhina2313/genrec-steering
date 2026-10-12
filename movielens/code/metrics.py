"""Query ranking metrics, paired comparisons, user bootstrap, confidence intervals, selection.

Conventions (documented in docs/run_guide.md):
- Rank by descending score; ties broken by ascending item_id (stable, deterministic).
- Gains are used as supplied (linear): DCG@k = sum_{r<=k} gain_r / log2(r + 1). No exponentiation.
- NDCG/recall/MRR are 0 for zero-positive (zero-ideal-gain) queries in the all-query macro average and
  are additionally reported conditionally over queries with a positive denominator.
- precision@k and mean@k use the top min(k, n_candidates) items as denominator.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from utils import assert_unique, safe_merge


def rank_scores(scored: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    assert_unique(scored, ["query_id", "item_id"], "rank_scores input")
    df = scored.sort_values(["query_id", score_col, "item_id"], ascending=[True, False, True], kind="mergesort")
    df = df.copy()
    df["rank"] = df.groupby("query_id").cumcount() + 1
    return df


def _dcg(g: np.ndarray) -> float:
    return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))


def _query_metrics(gains: dict[str, np.ndarray], cols: dict[str, np.ndarray], specs: list[dict]) -> dict:
    out = {}
    n = len(next(iter(cols.values())) if cols else next(iter(gains.values())))
    for m in specs:
        name, typ = m["name"], m["type"]
        if typ == "ndcg":
            g = gains[m["gain"]]
            k = min(m["k"], n)
            ideal = _dcg(np.sort(g)[::-1][:k])
            if ideal > 0:
                out[name], out[f"{name}__defined"] = _dcg(g[:k]) / ideal, True
            else:
                out[name], out[f"{name}__defined"] = 0.0, False
        elif typ == "precision":
            g = gains[m["gain"]] > 0
            k = min(m["k"], n)
            out[name], out[f"{name}__defined"] = float(g[:k].sum() / k), True
        elif typ == "recall":
            g = gains[m["gain"]] > 0
            tot = int(g.sum())
            out[name] = float(g[:m["k"]].sum() / tot) if tot else 0.0
            out[f"{name}__defined"] = tot > 0
        elif typ == "mrr":
            g = gains[m["gain"]] > 0
            hit = np.flatnonzero(g)
            out[name] = float(1.0 / (hit[0] + 1)) if len(hit) else 0.0
            out[f"{name}__defined"] = len(hit) > 0
        elif typ == "mean_at_k":
            c = cols[m["column"]].astype(float)
            k = min(m["k"], n)
            out[name], out[f"{name}__defined"] = float(c[:k].mean()), True
        else:
            raise ValueError(f"unknown metric type {typ}")
    return out


def query_metrics(scored: pd.DataFrame, labels: pd.DataFrame, specs: list[dict], score_col: str = "score") -> pd.DataFrame:
    """Per-query metrics for one configuration over the exact stored candidate sets."""
    need = sorted({m.get("gain") for m in specs if m.get("gain")} | {m.get("column") for m in specs if m.get("column")})
    lab = labels[["query_id", "item_id", "user_id"] + need]
    df = safe_merge(scored[["query_id", "item_id", score_col]], lab, on=["query_id", "item_id"], how="left",
                    what="scores x labels")
    if df[need].isna().any().any():
        raise AssertionError("scored pair without a label row")
    df = rank_scores(df, score_col)
    rows = []
    for qid, g in df.groupby("query_id", sort=True):
        arrs = {c: g[c].to_numpy() for c in need}
        r = {"query_id": qid, "user_id": g["user_id"].iat[0], "n_candidates": len(g)}
        r.update(_query_metrics(arrs, arrs, specs))
        rows.append(r)
    return pd.DataFrame(rows)


def aggregate(pq: pd.DataFrame, specs: list[dict]) -> dict:
    out = {"n_queries": int(len(pq)), "n_users": int(pq["user_id"].nunique()) if len(pq) else 0,
           "n_candidates": int(pq["n_candidates"].sum()) if len(pq) else 0}
    for m in specs:
        name = m["name"]
        out[name] = float(pq[name].mean()) if len(pq) else None
        d = pq[f"{name}__defined"]
        out[f"{name}__conditional"] = float(pq.loc[d, name].mean()) if d.any() else None
        out[f"{name}__n_defined"] = int(d.sum())
        if m["type"] in ("ndcg", "recall", "mrr"):
            out[f"{name}__zero_denominator_rate"] = float(1 - d.mean()) if len(pq) else None
    return out


def assert_same_queries(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    ka = a[["query_id", "n_candidates"]].sort_values("query_id").reset_index(drop=True)
    kb = b[["query_id", "n_candidates"]].sort_values("query_id").reset_index(drop=True)
    if not ka.equals(kb):
        raise AssertionError(f"{what}: query sets or candidate counts differ across methods")


def paired_bootstrap(base_pq: pd.DataFrame, meth_pq: pd.DataFrame, metric: str, draws: int, seed: int,
                     level: float = 0.95, conditional: bool = False) -> dict:
    """Paired user-level bootstrap for mean(method) - mean(baseline).

    All queries of a sampled user are kept together; candidate pairs are never resampled units.
    """
    assert_same_queries(base_pq, meth_pq, f"bootstrap {metric}")
    m = safe_merge(base_pq[["query_id", "user_id", metric, f"{metric}__defined"]],
                   meth_pq[["query_id", metric]].rename(columns={metric: "_m"}), on=["query_id"], what="paired")
    if conditional:
        m = m[m[f"{metric}__defined"]]
    if len(m) == 0:
        return {"baseline": None, "method": None, "abs_delta": None, "rel_delta": None, "ci_lower": None,
                "ci_upper": None, "n_queries": 0, "n_users": 0, "undefined_reason": "no queries with a defined metric"}
    m["_d"] = m["_m"] - m[metric]
    per_user = m.groupby("user_id").agg(d=("_d", "sum"), b=(metric, "sum"), n=("_d", "size"))
    base_mean, meth_mean = float(m[metric].mean()), float(m["_m"].mean())
    delta = meth_mean - base_mean
    rng = np.random.default_rng(seed)
    U = len(per_user)
    d, n = per_user["d"].to_numpy(), per_user["n"].to_numpy()
    idx = rng.integers(0, U, size=(draws, U))
    boot = d[idx].sum(1) / n[idx].sum(1)
    a = (1 - level) / 2
    lo, hi = (float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))) if U > 1 else (None, None)
    rel = delta / base_mean if base_mean != 0 else None
    return {"baseline": base_mean, "method": meth_mean, "abs_delta": delta, "rel_delta": rel,
            "rel_delta_undefined_reason": None if rel is not None else "baseline value is zero",
            "ci_lower": lo, "ci_upper": hi, "ci_undefined_reason": None if U > 1 else "fewer than two users",
            "n_queries": int(len(m)), "n_users": int(U), "draws": draws, "seed": seed, "level": level}


# ------------------------------------------------------------------ operating-point selection

def guardrail_feasible(guard: float, base_guard: float, budget: float) -> Optional[bool]:
    """Feasible iff guard >= (1 - budget) * base_guard (relative loss <= budget). None if base is 0."""
    if base_guard is None or base_guard == 0 or not math.isfinite(base_guard):
        return None
    return guard >= (1.0 - budget) * base_guard - 1e-12


def select_operating_point(candidates: list[dict], baseline: dict, budget: float) -> dict:
    """Among feasible settings (baseline included) maximize target; ties: better guardrail,
    smaller |strength|, then deterministic layer order, then config_id."""
    bg = baseline["guardrail"]
    if bg is None or bg == 0:
        return {"status": "not_estimable", "reason": "baseline guardrail is zero; relative budget undefined",
                "budget": budget}
    pool = [dict(baseline, config_id="baseline", strength=0.0, layer=-1, is_baseline=True)]
    pool += [dict(c, is_baseline=False) for c in candidates]
    feas = [c for c in pool if guardrail_feasible(c["guardrail"], bg, budget)]
    feas.sort(key=lambda c: (-c["target"], -c["guardrail"], abs(c.get("strength") or 0.0),
                             c.get("layer") if c.get("layer") is not None else -1, str(c["config_id"])))
    best = feas[0]
    return {"status": "selected", "budget": budget, "config_id": best["config_id"], "is_baseline": best["is_baseline"],
            "validation_target": best["target"], "validation_guardrail": best["guardrail"],
            "baseline_target": baseline["target"], "baseline_guardrail": bg,
            "n_considered": len(pool), "n_feasible": len(feas)}
