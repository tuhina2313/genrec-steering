"""Final JSON, CSV, Markdown, and figures from saved results. Makes no LLM calls."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import metrics as met
from contracts import SCHEMA_VERSION
from utils import now_iso, read_json, sha256_file, to_jsonable, write_json, write_text

# Reference categorical palette (fixed order, never cycled); gray for controls and references.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
METHOD_COLOR = {"mean_diff_steer": SERIES[0], "probe_steer": SERIES[1], "linear_reranker": SERIES[2]}


def _opt(path: Path):
    return path if path.exists() else None


def _num(x) -> Optional[float]:
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def metric_rows(ctx, pqs: pd.DataFrame, selected: dict) -> list[dict]:
    cfg = ctx.cfg
    draws = int(ctx.profile["bootstrap_draws"])
    seed = int(cfg["seeds"]["bootstrap_seed"])
    level = float(cfg["evaluation"]["ci_level"])
    specs = {m["name"]: m for m in cfg["metrics"]}
    sel_by_id = {s["selection_id"]: s for s in selected.get("selections", [])}
    rows = []
    for obj, oc in cfg["objectives"].items():
        base_all = pqs[(pqs["selection_id"] == "unsteered") & (pqs["objective"] == obj)]
        names = [oc["target"], oc["guardrail"]] + [m for m in specs if m not in (oc["target"], oc["guardrail"])]
        for sid in sorted(set(pqs.loc[(pqs["objective"] == obj) & (pqs["selection_id"] != "unsteered"), "selection_id"])):
            meth_all = pqs[(pqs["selection_id"] == sid) & (pqs["objective"] == obj)]
            root = sel_by_id.get(sid.split("|reversed")[0].split("|item_shift")[0], {})
            for subgroup in ("all", "unseen_by_direction"):
                b = base_all if subgroup == "all" else base_all[base_all["user_group"] == subgroup]
                m = meth_all if subgroup == "all" else meth_all[meth_all["user_group"] == subgroup]
                for name in names:
                    variants = [(name, False)]
                    if specs[name]["type"] in ("ndcg", "recall", "mrr"):
                        variants.append((name + "__conditional", True))
                    for label, cond in variants:
                        if len(b) == 0:
                            res = {"undefined_reason": "no queries in subgroup", "n_queries": 0, "n_users": 0}
                        else:
                            res = met.paired_bootstrap(b, m, name, draws, seed, level, conditional=cond)
                        rows.append({
                            "objective": obj, "method": meth_all["method"].iat[0], "selection_id": sid,
                            "split": "test", "subgroup": subgroup, "metric": label,
                            "role": "target" if name == oc["target"] else "guardrail" if name == oc["guardrail"] else "secondary",
                            "vector_seed": root.get("seed"), "layer": _num(root.get("layer")),
                            "strength_or_lambda": _num(root.get("strength")) if root.get("strength") is not None
                            else _num(root.get("lambda")),
                            "guardrail_budget": root.get("budget"),
                            "baseline_value": res.get("baseline"), "method_value": res.get("method"),
                            "abs_delta": res.get("abs_delta"), "rel_delta": res.get("rel_delta"),
                            "ci_lower": res.get("ci_lower"), "ci_upper": res.get("ci_upper"),
                            "n_queries": res.get("n_queries", 0), "n_users": res.get("n_users", 0),
                            "n_candidates": int(m["n_candidates"].sum()) if len(m) else 0,
                            "undefined_reason": res.get("undefined_reason") or res.get("rel_delta_undefined_reason")
                            or res.get("ci_undefined_reason"),
                        })
    return rows


def guardrail_check(rows: list[dict], cfg: dict) -> list[dict]:
    out = []
    for r in rows:
        if r["role"] != "guardrail" or r["subgroup"] != "all" or r["metric"].endswith("__conditional"):
            continue
        b, v, budget = r["baseline_value"], r["method_value"], r["guardrail_budget"]
        ok = met.guardrail_feasible(v, b, budget) if None not in (b, v, budget) else None
        out.append({"selection_id": r["selection_id"], "objective": r["objective"], "budget": budget,
                    "test_guardrail": v, "test_baseline_guardrail": b, "satisfied_on_test": ok})
    return out


def seed_variability(rows: list[dict]) -> list[dict]:
    """Variability over direction-estimation seeds (distinct from user-bootstrap uncertainty)."""
    df = pd.DataFrame([r for r in rows if r["vector_seed"] is not None and r["subgroup"] == "all"
                       and r["role"] in ("target", "guardrail") and not r["metric"].endswith("__conditional")])
    if df.empty:
        return []
    out = []
    for (obj, meth, budget, metric), g in df.groupby(["objective", "method", "guardrail_budget", "metric"]):
        d = g["abs_delta"].dropna()
        out.append({"objective": obj, "method": meth, "guardrail_budget": budget, "metric": metric,
                    "n_seeds": int(g["vector_seed"].nunique()), "mean_abs_delta": float(d.mean()) if len(d) else None,
                    "sd_abs_delta_across_seeds": float(d.std(ddof=1)) if len(d) > 1 else None,
                    "note": "spread over vector-estimation seeds on one frozen model; not trained-model seeds"})
    return out


# ------------------------------------------------------------------ figures

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(ctx, fig_dir: Path, probes: Optional[pd.DataFrame], sweep: Optional[pd.DataFrame],
                 rows: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    objs = list(ctx.cfg["objectives"])

    if probes is not None and len(probes):
        fig, axes = plt.subplots(1, len(objs), figsize=(4.2 * len(objs), 3.2), squeeze=False, facecolor=SURFACE)
        for ax, obj in zip(axes[0], objs):
            _style(ax)
            for i, est in enumerate(["mean_diff", "probe"]):
                d = probes[(probes["objective"] == obj) & (probes["estimator"] == est)]
                g = d.groupby("layer")["auroc"].mean().sort_index()
                ax.plot(g.index, g.values, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                        label="mean-difference projection" if est == "mean_diff" else "logistic probe")
            ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
            ax.set_title(f"{obj}: probe-validation AUROC", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("layer index", fontsize=8, color=INK2)
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "probe_by_layer.png", dpi=150)
        plt.close(fig)
        made.append("figures/probe_by_layer.png")

    if sweep is not None and len(sweep):
        fig, axes = plt.subplots(1, len(objs), figsize=(4.4 * len(objs), 3.6), squeeze=False, facecolor=SURFACE)
        for ax, obj in zip(axes[0], objs):
            _style(ax)
            d = sweep[sweep["objective"] == obj]
            ctrl = d[d["method"].isin(["random_direction", "shuffled_label_direction", "item_prior", "attribute_boost"])]
            ax.scatter(ctrl["guardrail"], ctrl["target"], s=14, color=MUTED, alpha=0.7, label="controls / other comparators")
            for meth, col in METHOD_COLOR.items():
                dm = d[d["method"] == meth]
                if len(dm):
                    ax.scatter(dm["guardrail"], dm["target"], s=22, color=col, edgecolor=SURFACE, linewidth=1.5,
                               label=meth.replace("_", " "))
            b = d[d["method"] == "unsteered"]
            if len(b):
                ax.scatter(b["guardrail"], b["target"], s=70, marker="*", color=INK, label="unsteered", zorder=5)
            oc = ctx.cfg["objectives"][obj]
            ax.set_xlabel(f"guardrail: {oc['guardrail']}", fontsize=8, color=INK2)
            ax.set_ylabel(f"target: {oc['target']}", fontsize=8, color=INK2)
            ax.set_title(f"{obj}: validation sweep", fontsize=9, color=INK, loc="left")
        axes[0][0].legend(fontsize=6.5, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "validation_tradeoff.png", dpi=150)
        plt.close(fig)
        made.append("figures/validation_tradeoff.png")

    tr = [r for r in rows if r["role"] == "target" and r["subgroup"] == "all" and not r["metric"].endswith("__conditional")
          and r["abs_delta"] is not None]
    if tr:
        fig, axes = plt.subplots(1, len(objs), figsize=(4.8 * len(objs), 0.35 * max(4, len(tr)) + 1.2), squeeze=False,
                                 facecolor=SURFACE)
        for ax, obj in zip(axes[0], objs):
            _style(ax)
            d = [r for r in tr if r["objective"] == obj]
            y = np.arange(len(d))
            for i, r in enumerate(d):
                lo = r["ci_lower"] if r["ci_lower"] is not None else r["abs_delta"]
                hi = r["ci_upper"] if r["ci_upper"] is not None else r["abs_delta"]
                col = METHOD_COLOR.get(r["method"], MUTED)
                ax.plot([lo, hi], [i, i], color=col, linewidth=2)
                ax.plot([r["abs_delta"]], [i], "o", color=col, markersize=6)
            ax.axvline(0, color=INK2, linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels([r["selection_id"].split("|", 1)[1] for r in d], fontsize=6.5)
            ax.set_title(f"{obj}: test Δ {d[0]['metric'] if d else ''} (95% user CI)",
                         fontsize=8.5, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(fig_dir / "test_comparison.png", dpi=150)
        plt.close(fig)
        made.append("figures/test_comparison.png")

    t = {k: v for k, v in ctx.timing.items() if not k.startswith("stage:") and v.get("pairs")}
    if t:
        fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=SURFACE)
        _style(ax)
        ks = list(t)
        ax.barh(ks, [t[k]["pairs"] for k in ks], color=SERIES[0], height=0.5)
        for i, k in enumerate(ks):
            pps = t[k].get("pairs_per_second")
            ax.text(t[k]["pairs"], i, f"  {t[k]['pairs']} pairs, {pps:.1f}/s" if pps else "", va="center",
                    fontsize=7, color=INK2)
        ax.set_title("forward pairs by stage (this run, cached work excluded)", fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(fig_dir / "cost.png", dpi=150)
        plt.close(fig)
        made.append("figures/cost.png")
    return made


# ------------------------------------------------------------------ markdown

def _fmt(x, nd=4):
    return "null" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def results_markdown(final: dict, rows: list[dict], gchk: list[dict]) -> str:
    L = [f"# Activation steering on {final['dataset']} — {final['profile']} profile",
         "", f"Run `{final['run_id']}`, protocol `{final['protocol_id']}`, generated {final['generated_at']}.",
         f"Model revision `{final['model_revision']}`; backbone trained: **{str(final['backbone_trained']).lower()}**.",
         f"Completed: **{final['completed']}**.", ""]
    L += ["## Limitations (read first)", ""]
    L += [f"- {x}" for x in final["limitations"]]
    L += ["", "## Stage status", "", "| stage | status | note |", "|---|---|---|"]
    for s, v in final["stage_status"].items():
        if s == "report":
            continue
        L.append(f"| {s} | {v.get('status')} | {(v.get('error') or '').replace('|', '/')[:160]} |")
    L += ["", "## Test results (selected on validation; paired user bootstrap vs unsteered)", ""]
    sel_rows = [r for r in rows if r["subgroup"] in ("all", "unseen_by_direction") and r["role"] in ("target", "guardrail")
                and not r["metric"].endswith("__conditional")]
    if not sel_rows:
        L.append("No test results were produced. See stage status above; nothing is inferred from missing evidence.")
    else:
        L += ["| objective | selection | subgroup | metric | baseline | method | Δ | 95% CI | queries/users |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in sel_rows:
            ci = f"[{_fmt(r['ci_lower'])}, {_fmt(r['ci_upper'])}]" if r["ci_lower"] is not None else "null"
            L.append(f"| {r['objective']} | {r['selection_id'].split('|', 1)[1]} | {r['subgroup']} | {r['metric']} | "
                     f"{_fmt(r['baseline_value'])} | {_fmt(r['method_value'])} | {_fmt(r['abs_delta'])} | {ci} | "
                     f"{r['n_queries']}/{r['n_users']} |")
    if gchk:
        L += ["", "## Did the validation-selected guardrail budget hold on test?", "",
              "| selection | budget | test guardrail | baseline | satisfied |", "|---|---|---|---|---|"]
        for g in gchk:
            L.append(f"| {g['selection_id']} | {g['budget']} | {_fmt(g['test_guardrail'])} | "
                     f"{_fmt(g['test_baseline_guardrail'])} | {g['satisfied_on_test']} |")
        L += ["", "A validation-selected budget is not a guarantee of test performance."]
    if final["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in final["warnings"]]
    L += ["", "## How to reproduce", "", "```bash", "cd movielens",
          f"python code/run.py run --dataset {final['dataset']} --profile {final['profile']} --through report --resume",
          "```", "", "See `docs/run_guide.md` for stage-by-stage commands and artifact descriptions.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ entry point

def build_report(ctx) -> dict:
    out = ctx.out_dir
    status = ctx.status.data["stages"]
    audit = read_json(out / "data_audit.json") if (out / "data_audit.json").exists() else {}
    resolved = yaml.safe_load((out / "config.resolved.yaml").read_text()) if (out / "config.resolved.yaml").exists() else {}
    selected = read_json(out / "selected.json") if (out / "selected.json").exists() else {}
    probes = pd.read_csv(out / "probes.csv") if (out / "probes.csv").exists() else None
    sweep = pd.read_csv(out / "validation_sweep.csv") if (out / "validation_sweep.csv").exists() else None
    pqs = pd.read_parquet(out / "per_query_metrics.parquet") if (out / "per_query_metrics.parquet").exists() else None
    warnings = []

    pipeline = [s for s in ("prepare", "baseline", "fit-directions", "sweep", "select", "evaluate")]
    completed = all(status.get(s, {}).get("status") == "completed" for s in pipeline)
    for s in pipeline:
        st = status.get(s, {}).get("status", "not run")
        if st != "completed":
            warnings.append(f"stage {s} is {st}" + (f": {status[s].get('error')}" if status.get(s, {}).get("error") else ""))
    if pqs is not None and not completed:
        warnings.append("test outputs exist but upstream stages are not all completed for this configuration; "
                        "treat results as partial")

    rows = metric_rows(ctx, pqs, selected) if pqs is not None and completed else []
    gchk = guardrail_check(rows, ctx.cfg)
    for g in gchk:
        if g["satisfied_on_test"] is False:
            warnings.append(f"{g['selection_id']}: selected configuration violates the guardrail budget on test")
    bs = read_json(out / "baseline_summary.json") if (out / "baseline_summary.json").exists() else None
    if bs:
        for obj, c in bs["competitiveness"].items():
            if c["unsteered"] is not None and c["random_ranking"] is not None and c["unsteered"] <= c["random_ranking"]:
                warnings.append(f"{obj}: unsteered frozen LLM is not better than random ranking on validation "
                                f"({c['unsteered']:.4f} vs {c['random_ranking']:.4f})")
            if c["unsteered"] is not None and c["item_prior_only"] is not None and c["unsteered"] < c["item_prior_only"]:
                warnings.append(f"{obj}: unsteered frozen LLM is below the item-rating prior on validation")
    if probes is not None:
        ne = probes[probes["auroc"].isna()]
        if len(ne):
            warnings.append(f"{len(ne)} probe diagnostics not estimable (one class absent)")
    idx = out / "directions" / "index.json"
    if idx.exists() and read_json(idx).get("failed_fits"):
        warnings.append(f"failed direction fits: {read_json(idx)['failed_fits']}")
    ev = read_json(out / "evaluate_notes.json") if (out / "evaluate_notes.json").exists() else {}
    warnings += ev.get("notes", [])

    figs = make_figures(ctx, out / "figures", probes, sweep, rows)
    model_rev = (resolved.get("resolved") or {}).get("model_revision")
    final = {
        "schema_version": SCHEMA_VERSION, "run_id": ctx.run_id, "dataset": ctx.dataset,
        "protocol_id": ctx.cfg["dataset"].get("protocol_id"), "profile": ctx.profile_name,
        "model": ctx.cfg["model"]["name"], "model_revision": model_rev, "backbone_trained": False,
        "completed": completed, "generated_at": now_iso(),
        "realized_counts": audit.get("realized"), "candidate_source": audit.get("candidate_source"),
        "split_cutoffs": audit.get("split_cutoffs"),
        "selection_hash": sha256_file(out / "selected.json") if (out / "selected.json").exists() else None,
        "metrics": rows,
        "confidence_intervals": {"method": "paired user-level percentile bootstrap", "unit": "user",
                                 "draws": int(ctx.profile["bootstrap_draws"]), "seed": ctx.cfg["seeds"]["bootstrap_seed"],
                                 "level": ctx.cfg["evaluation"]["ci_level"]},
        "vector_seed_variability": seed_variability(rows),
        "guardrail_on_test": gchk,
        "item_shift_coverage": ev.get("item_shift_coverage", []),
        "timing": ctx.timing, "warnings": warnings,
        "limitations": list(dict.fromkeys(audit.get("limitations", []) + [
            "Candidates are movies the user actually rated in the outcome period (retrospective benchmark "
            "information), not a deployable retrieval system; metrics measure preference ranking within "
            "that pool, not next-watch prediction or exhaustive retrieval.",
            "Frozen general-purpose LLM reranking of supplied candidate sets; not the GenRecs catalog-head "
            "architecture and not full-catalog retrieval.",
            "The satisfaction target and graded-rating guardrail are correlated views of the same ratings.",
            "Higher mean Yes score or probe score is not evidence of better ranking; only within-query changes count.",
        ])),
        "stage_status": {k: {kk: vv for kk, vv in v.items() if kk != "traceback"} for k, v in status.items()},
        "artifact_paths": {k: str(p.relative_to(out)) for k, p in {
            "config": out / "config.resolved.yaml", "environment": out / "environment.json",
            "data_audit": out / "data_audit.json", "baseline_summary": out / "baseline_summary.json",
            "probes": out / "probes.csv", "directions": out / "directions" / "index.json",
            "validation_sweep": out / "validation_sweep.csv", "selected": out / "selected.json",
            "scores_test": out / "scores_test.parquet", "per_query_metrics": out / "per_query_metrics.parquet",
            "budget": out / "budget.json", "stage_status": out / "stage_status.json"}.items() if p.exists()}
        | {f"figure:{Path(f).stem}": f for f in figs},
    }
    write_json(out / "final_results.json", final)
    pd.DataFrame(rows).to_csv(out / "final_results.csv", index=False)
    write_text(out / "results.md", results_markdown(final, rows, gchk))
    return {"artifacts": ["final_results.json", "final_results.csv", "results.md"] + figs, "completed": completed,
            "metric_rows": len(rows)}
