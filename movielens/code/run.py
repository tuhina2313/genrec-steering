"""CLI, stage dependencies, run orchestration, resume and failure status.

Usage (from the project root, i.e. the `movielens/` folder):
  python code/run.py prepare        --dataset movielens_1m --profile pilot
  python code/run.py baseline       --dataset movielens_1m --profile pilot
  python code/run.py fit-directions --dataset movielens_1m --profile pilot
  python code/run.py sweep          --dataset movielens_1m --profile pilot
  python code/run.py select         --dataset movielens_1m --profile pilot
  python code/run.py evaluate       --dataset movielens_1m --profile pilot
  python code/run.py report         --dataset movielens_1m --profile pilot
  python code/run.py run            --dataset movielens_1m --profile pilot --through report --resume

`run` executes the same stage functions and checks as the individual commands.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import data as data_mod  # noqa: E402
import directions as dir_mod  # noqa: E402
import metrics as met  # noqa: E402
import prompts  # noqa: E402
from steering import random_unit_vector  # noqa: E402
from contracts import NO_INTERVENTION, SCHEMA_VERSION, Intervention, split_pair_id  # noqa: E402
from utils import (  # noqa: E402
    PROJECT_ROOT, NotEstimable, PrerequisiteMissing, WorkCapExceeded, array_hash, assert_unique, capture_environment,
    deep_merge, hash_uint, now_iso, read_json, set_dotted, set_seeds, sha256_file, short_hash, stable_hash,
    write_json, write_npz, write_parquet, write_text, write_yaml,
)

STAGES = ["prepare", "baseline", "fit-directions", "sweep", "select", "evaluate", "report"]
PREREQS = {"prepare": [], "baseline": ["prepare"], "fit-directions": ["baseline"], "sweep": ["fit-directions"],
           "select": ["sweep"], "evaluate": ["select"], "report": []}
STEER_METHODS = ("mean_diff_steer", "probe_steer")
CONTROL_METHODS = ("random_direction", "shuffled_label_direction")


# =================================================================== configuration

def load_config(dataset: str, profile: str, overrides: list[str], root: Path) -> dict:
    with open(root / "configs" / "common.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(root / "configs" / f"{dataset}.yaml") as f:
        cfg = deep_merge(cfg, yaml.safe_load(f))
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        set_dotted(cfg, k.strip(), yaml.safe_load(v))
    if profile not in cfg["profiles"]:
        raise SystemExit(f"unknown profile {profile!r}; known: {sorted(cfg['profiles'])}")
    p = cfg["profiles"][profile]
    p.setdefault("min_eval_queries", cfg["splits"]["min_eval_queries"])
    p.setdefault("bootstrap_draws", cfg["evaluation"]["bootstrap_draws"])
    cfg["profile_name"] = profile
    return cfg


class StageStatus:
    def __init__(self, path: Path):
        self.path = path
        self.data = read_json(path) if path.exists() else {"schema_version": SCHEMA_VERSION, "stages": {}}

    def get(self, stage: str) -> dict:
        return self.data["stages"].get(stage, {})

    def set(self, stage: str, **kw):
        cur = self.data["stages"].get(stage, {})
        cur.update(kw)
        self.data["stages"][stage] = cur
        write_json(self.path, self.data)


class Ctx:
    def __init__(self, dataset: str, profile: str, run_id: Optional[str], overrides: list[str], root: Path,
                 work_cap: Optional[int] = None):
        self.root = Path(root)
        self.cfg = load_config(dataset, profile, overrides, self.root)
        self.dataset, self.profile_name = dataset, profile
        self.profile = self.cfg["profiles"][profile]
        self.run_id = run_id or profile
        protocol = self.cfg["dataset"].get("protocol_id") or "unspecified"
        self.raw_dir = self.root / "data" / "raw" / dataset
        self.proc_dir = self.root / "data" / "processed" / dataset / protocol / profile
        self.cache_dir = self.root / "cache" / dataset / self.run_id
        self.out_dir = self.root / "outputs" / dataset / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.status = StageStatus(self.out_dir / "stage_status.json")
        cap = work_cap if work_cap is not None else self.cfg["work_cap"]["max_forward_pairs"]
        self.work_cap = int(cap) if cap else None
        self.forward_pairs = 0
        self._scorer = None
        self._data = None
        self._enc_cache: dict[str, pd.DataFrame] = {}
        self.timing = read_json(self.out_dir / "timing.json") if (self.out_dir / "timing.json").exists() else {}

    # ------------------------------------------------------------ shared resources
    @property
    def data(self) -> dict:
        if self._data is None:
            if not (self.proc_dir / "manifest.json").exists():
                raise PrerequisiteMissing(str(self.proc_dir / "manifest.json"), "run the prepare stage")
            self._data = data_mod.load_processed(self.proc_dir)
            self._data["queries_by_id"] = {q["query_id"]: q for q in self._data["queries"]}
            self._data["items_by_id"] = {r["item_id"]: r for r in self._data["items"].to_dict("records")}
        return self._data

    @property
    def scorer(self):
        if self._scorer is None:
            from model import FrozenScorer

            set_seeds(self.cfg["seeds"]["data_seed"])
            self._scorer = FrozenScorer.load(self.cfg)
        return self._scorer

    def layers(self) -> list[int]:
        from steering import resolve_layers

        return resolve_layers(self.profile["layer_fractions"], self.scorer.num_layers)

    def fingerprint(self) -> str:
        c = {k: v for k, v in self.cfg.items() if k != "work_cap"}
        return stable_hash(c)

    def add_timing(self, key: str, seconds: float, pairs: int):
        t = self.timing.setdefault(key, {"seconds": 0.0, "pairs": 0})
        t["seconds"] += seconds
        t["pairs"] += pairs
        t["pairs_per_second"] = t["pairs"] / t["seconds"] if t["seconds"] > 0 else None
        write_json(self.out_dir / "timing.json", self.timing)

    # ------------------------------------------------------------ prompts
    def encodings(self, pair_ids: list[str], variant: str = "base", objective: Optional[str] = None) -> pd.DataFrame:
        """Tokenized prompts for pairs, cached by template, tokenizer revision and query data hash."""
        s = self.scorer
        tkey = short_hash({"template": prompts.prompt_template_hash(self.cfg, variant, objective),
                           "tokenizer": s.tokenizer_revision, "verbalizer": s.verbalizer,
                           "queries": self.data["manifest"]["queries_sha256"]})
        path = self.cache_dir / "prompts" / f"{variant}-{objective or 'none'}-{tkey}.parquet"
        if tkey not in self._enc_cache:
            self._enc_cache[tkey] = pd.read_parquet(path) if path.exists() else pd.DataFrame(
                columns=["pair_id", "input_ids", "token_len", "n_history_total", "n_history_used",
                         "n_history_truncated", "prompt_hash"])
        have = self._enc_cache[tkey]
        missing = sorted(set(pair_ids) - set(have["pair_id"]))
        if missing:
            rows = []
            for pid in missing:
                qid, item = split_pair_id(pid)
                q = self.data["queries_by_id"][qid]
                cand = prompts.make_candidate(self.data["items_by_id"][item])
                r = prompts.render_pair(s.tokenizer, q, cand, self.cfg, s.verbalizer, variant, objective)
                r["pair_id"] = pid
                rows.append(r)
            new = pd.DataFrame(rows)
            have = pd.concat([have, new], ignore_index=True) if len(have) else new
            assert_unique(have, ["pair_id"], "prompt encodings")
            write_parquet(path, have)
            self._enc_cache[tkey] = have
        idx = have.set_index("pair_id")
        return idx.loc[pair_ids].reset_index()

    # ------------------------------------------------------------ resumable scoring jobs
    def job_key(self, variant: str, objective: Optional[str], intervention: Intervention) -> tuple[str, dict]:
        ident = {"scorer": self.scorer.identity(),
                 "template": prompts.prompt_template_hash(self.cfg, variant, objective),
                 "queries": self.data["manifest"]["queries_sha256"],
                 "intervention": intervention.identity()}
        return short_hash(ident, 24), ident

    def run_job(self, pair_ids: list[str], variant: str = "base", objective: Optional[str] = None,
                intervention: Intervention = NO_INTERVENTION, delta: Optional[np.ndarray] = None,
                capture_layers: tuple = (), require_cached: bool = False, timing_key: str = "scoring"):
        """Score pairs for one configuration, resuming from completed parts keyed by pair ID."""
        assert len(set(pair_ids)) == len(pair_ids), "duplicate pair ids in job request"
        key, ident = self.job_key(variant, objective, intervention)
        jdir = self.cache_dir / "jobs" / key
        if not (jdir / "job.json").exists():
            write_json(jdir / "job.json", ident)
        enc = self.encodings(pair_ids, variant, objective)
        cur_hash = dict(zip(enc["pair_id"], enc["prompt_hash"]))
        want = set(pair_ids)
        done: dict[str, pd.DataFrame] = {}
        states: dict[int, dict[str, np.ndarray]] = {li: {} for li in capture_layers}
        for part in sorted(jdir.glob("part-*.parquet")):
            df = pd.read_parquet(part)
            df = df[df["pair_id"].isin(want)]
            if not len(df):
                continue
            df = df[df["prompt_hash"] == df["pair_id"].map(cur_hash)]   # stale prompts are not reused
            if capture_layers:
                npz_path = part.with_suffix(".npz")
                if not npz_path.exists():
                    continue
                z = np.load(npz_path)
                if not all(f"L{li}" in z for li in capture_layers):
                    continue
                pos = {p: i for i, p in enumerate(z["pair_ids"])}
                for li in capture_layers:
                    arr = z[f"L{li}"]
                    for p in df["pair_id"]:
                        states[li][p] = arr[pos[p]]
            for r in df.itertuples(index=False):
                done[r.pair_id] = r
        todo = [p for p in pair_ids if p not in done]
        if todo and require_cached:
            raise PrerequisiteMissing(f"{jdir} ({len(todo)} uncached pairs)", "expected cached scores from an earlier stage")
        part_size = int(self.cfg["scoring"]["part_size"])
        enc_idx = enc.set_index("pair_id")
        delta_arr = None if delta is None else np.asarray(delta, dtype=np.float32)
        for i in range(0, len(todo), part_size):
            chunk = todo[i:i + part_size]
            if self.work_cap is not None and self.forward_pairs + len(chunk) > self.work_cap:
                raise WorkCapExceeded(f"work cap {self.work_cap} forward pairs reached; "
                                      f"{len(todo) - i} pairs of job {key} remain")
            seqs = [list(enc_idx.at[p, "input_ids"]) for p in chunk]
            res = self.scorer.score(chunk, seqs, capture_layers=tuple(capture_layers), intervention=intervention,
                                    delta=delta_arr)
            self.forward_pairs += len(chunk)
            self.add_timing(timing_key, res.seconds, len(chunk))
            df = pd.DataFrame({"pair_id": chunk, "score": res.scores, "logit_pos": res.logit_pos,
                               "logit_neg": res.logit_neg, "prompt_hash": [cur_hash[p] for p in chunk]})
            name = "part-" + short_hash(sorted(chunk) + [sorted(capture_layers)], 20)
            if capture_layers:
                write_npz(jdir / f"{name}.npz", pair_ids=np.array(chunk),
                          **{f"L{li}": res.states[li] for li in capture_layers})
            write_parquet(jdir / f"{name}.parquet", df)      # parquet presence marks the part complete
            for r in df.itertuples(index=False):
                done[r.pair_id] = r
            for li in capture_layers:
                for j, p in enumerate(chunk):
                    states[li][p] = res.states[li][j]
        out = pd.DataFrame([done[p] for p in pair_ids])
        q_i = [split_pair_id(p) for p in out["pair_id"]]
        out["query_id"] = [a for a, _ in q_i]
        out["item_id"] = [b for _, b in q_i]
        st = {li: np.stack([states[li][p] for p in pair_ids]) if pair_ids else np.zeros((0, 0), np.float32)
              for li in capture_layers}
        return out, st, key


# =================================================================== helpers

def split_pairs(ctx: Ctx, split: str) -> list[str]:
    lab = ctx.data["labels"]
    return sorted(lab.loc[lab["split"] == split, "pair_id"])


def capture_pool(ctx: Ctx) -> list[str]:
    d = ctx.data
    return sorted(set(d["direction_pairs"]["pair_id"]) | set(d["probe_val_pairs"]["pair_id"]))


def labels_for(ctx: Ctx, split: str) -> pd.DataFrame:
    lab = ctx.data["labels"]
    return lab[lab["split"] == split].reset_index(drop=True)


def metric_specs(ctx: Ctx) -> list[dict]:
    return ctx.cfg["metrics"]


def per_query(ctx: Ctx, scored: pd.DataFrame, split: str, score_col: str = "score") -> pd.DataFrame:
    return met.query_metrics(scored, labels_for(ctx, split), metric_specs(ctx), score_col)


def load_directions(ctx: Ctx) -> dict:
    idx_path = ctx.out_dir / "directions" / "index.json"
    if not idx_path.exists():
        raise PrerequisiteMissing(str(idx_path), "run fit-directions")
    index = read_json(idx_path)
    out = {}
    for did, meta in index["directions"].items():
        z = np.load(ctx.out_dir / "directions" / meta["file"])
        out[did] = dict(meta, vector=z["vector"].astype(np.float32))
    return out


def steer_intervention(d: dict, strength: float, kind: str = "add_block") -> tuple[Intervention, np.ndarray]:
    iv = Intervention(kind=kind, layer_index=d["layer_index"], direction_id=d["direction_id"],
                      vector_sha=array_hash(d["vector"]), strength=float(strength), radius=float(d["radius"]))
    delta = (iv.strength * iv.radius * d["vector"]).astype(np.float32)
    return iv, delta


def standardized_adjust(base: np.ndarray, adj: np.ndarray, stats: dict, lam: float) -> np.ndarray:
    """baseline + lambda * sd(baseline_fit) * (adj - mean_fit) / sd_fit; all statistics from fitting data."""
    sd = stats["adj_sd"] if stats["adj_sd"] and stats["adj_sd"] > 0 else 1.0
    return base + lam * stats["base_sd"] * (adj - stats["adj_mean"]) / sd


def random_ranking_scores(ctx: Ctx, pair_ids: list[str]) -> np.ndarray:
    seed = ctx.cfg["seeds"]["data_seed"]
    return np.array([hash_uint(seed, "random_rank", p) / 2**64 for p in pair_ids])


def item_prior_values(ctx: Ctx, items: list[str]) -> np.ndarray:
    pr = ctx.data["item_prior"].set_index("item_id")["prior"]
    g = ctx.data["audit"]["item_prior"]["global_mean"]
    return np.array([pr.get(i, g) for i in items], dtype=np.float64)


def attribute_values(ctx: Ctx, items: list[str]) -> np.ndarray:
    return np.array([ctx.data["items_by_id"][i]["attribute_flag"] for i in items], dtype=np.float64)


# =================================================================== stages

def stage_prepare(ctx: Ctx) -> dict:
    man = data_mod.prepare_dataset(ctx.cfg, ctx.profile_name, ctx.raw_dir, ctx.proc_dir)
    audit = read_json(ctx.proc_dir / "data_audit.json")
    write_json(ctx.out_dir / "data_audit.json", audit)
    write_yaml(ctx.out_dir / "config.resolved.yaml", dict(ctx.cfg, resolved={
        "processed_dir": str(ctx.proc_dir.relative_to(ctx.root)), "data_input_key": man["input_key"],
        "queries_sha256": man["queries_sha256"], "labels_sha256": man["labels_sha256"]}))
    write_json(ctx.out_dir / "environment.json", capture_environment({"stage": "prepare"}))
    write_json(ctx.out_dir / "plan.json", plan_work(ctx, n_layers=len(ctx.profile["layer_fractions"])))
    if man["status"] == "not_estimable":
        raise NotEstimable("; ".join(audit["not_estimable_reasons"]))
    return {"artifacts": ["data_audit.json", "config.resolved.yaml", "plan.json"], "reused": man.get("reused", False)}


def plan_work(ctx: Ctx, n_layers: int) -> dict:
    a = ctx.data["audit"] if (ctx.proc_dir / "manifest.json").exists() else None
    if a is None:
        return {}
    p = ctx.profile
    n_val, n_test = a["realized"]["validation"]["pairs"], a["realized"]["test"]["pairs"]
    n_obj, n_seeds = len(ctx.cfg["objectives"]), len(p["direction_seeds"])
    n_str = len(p["strengths"])
    steer = n_seeds * n_layers * n_obj * 2 * (n_str + 1)               # + reversal diagnostic
    controls = n_seeds * n_layers * n_str * (1 + n_obj) if ctx.cfg["steering"]["controls_on_full_grid"] else 0
    sweep_cfgs = steer + controls + 1 + n_obj                          # + post-norm null, objective prompts
    n_capture = len(capture_pool(ctx))
    n_sel_llm = n_obj * n_seeds * 2 * len(p["guardrail_budgets"])
    fit_pairs = int(ctx.data["direction_pairs"]["pair_id"].nunique())
    return {
        "profile": ctx.profile_name, "optional_analyses": p.get("optional_analyses", []),
        "pairs": {"validation": n_val, "test": n_test, "capture_pool": n_capture},
        "baseline_forward_pairs": n_val + n_capture,
        "sweep_configs": sweep_cfgs, "sweep_forward_pairs_upper_bound": sweep_cfgs * n_val
        + n_layers * 2 * ctx.cfg["steering"]["identity_check_pairs"],
        "evaluate_forward_pairs_upper_bound": n_test * (1 + 2 * n_sel_llm + n_obj) + n_sel_llm * fit_pairs,
        "note": "upper bounds; distinct selected configurations are deduplicated at evaluation time",
    }


def stage_baseline(ctx: Ctx) -> dict:
    s = ctx.scorer
    layers = ctx.layers()
    env = capture_environment({"stage": "baseline", **s.identity(), "device": s.device,
                               "layers": layers, "num_layers": s.num_layers, "hidden_size": s.hidden_size})
    write_json(ctx.out_dir / "environment.json", env)
    resolved = yaml.safe_load((ctx.out_dir / "config.resolved.yaml").read_text())
    resolved.setdefault("resolved", {}).update({"layers": layers, "num_layers": s.num_layers, **s.identity()})
    write_yaml(ctx.out_dir / "config.resolved.yaml", resolved)

    val = split_pairs(ctx, "validation")
    pool = capture_pool(ctx)
    enc = ctx.encodings(val[: ctx.cfg["scoring"]["lm_head_parity_pairs"]])
    parity = s.lm_head_parity([list(x) for x in enc["input_ids"]])
    if not parity["passed"]:
        raise RuntimeError(f"two-row LM-head adapter does not match full forward logits: {parity}")

    t0 = time.time()
    n0 = ctx.forward_pairs
    vs, _, vkey = ctx.run_job(val, capture_layers=tuple(layers), timing_key="baseline")
    ps, _, _ = ctx.run_job(pool, capture_layers=tuple(layers), timing_key="baseline")
    elapsed, done_pairs = time.time() - t0, ctx.forward_pairs - n0

    ident = s.identity()
    base_tab = pd.concat([vs.assign(split="validation"), ps.assign(split="fit")], ignore_index=True)
    base_tab = base_tab.rename(columns={"score": "baseline_score"})
    base_tab["verbalizer_ids"] = [list(ident["verbalizer_ids"])] * len(base_tab)
    base_tab["model_revision"] = ident["model_revision"]
    base_tab["scorer_version"] = ident["scorer_version"]
    write_parquet(ctx.out_dir / "baseline_scores.parquet", base_tab)

    enc_all = ctx.encodings(val + pool)
    specs = metric_specs(ctx)
    pq_base = per_query(ctx, vs, "validation")
    rr = vs[["query_id", "item_id"]].assign(score=random_ranking_scores(ctx, list(vs["pair_id"])))
    ip = vs[["query_id", "item_id"]].assign(score=item_prior_values(ctx, list(vs["item_id"])))
    summary = {
        "schema_version": SCHEMA_VERSION, "split": "validation", "job_key": vkey,
        "metrics": {"unsteered": met.aggregate(pq_base, specs),
                    "random_ranking": met.aggregate(per_query(ctx, rr, "validation"), specs),
                    "item_prior_only": met.aggregate(per_query(ctx, ip, "validation"), specs)},
        "lm_head_parity": parity, "verbalizer": ident["verbalizer"], "verbalizer_ids": ident["verbalizer_ids"],
        "prompt": {"version": prompts.PROMPT_VERSION, "template_hash": prompts.prompt_template_hash(ctx.cfg),
                   "token_len_max": int(enc_all["token_len"].max()), "token_len_median": float(enc_all["token_len"].median()),
                   "pairs_with_history_truncation": int((enc_all["n_history_truncated"] > 0).sum()),
                   "history_items_truncated": int(enc_all["n_history_truncated"].sum())},
        "throughput": {"forward_pairs_this_invocation": done_pairs, "seconds": elapsed,
                       "pairs_per_second": ctx.timing.get("baseline", {}).get("pairs_per_second"),
                       "effective_microbatch": s.microbatch, "peak_memory_bytes": s.peak_memory_bytes or None},
    }
    objs = ctx.cfg["objectives"]
    summary["competitiveness"] = {
        o: {"target": oc["target"], "unsteered": summary["metrics"]["unsteered"][oc["target"]],
            "random_ranking": summary["metrics"]["random_ranking"][oc["target"]],
            "item_prior_only": summary["metrics"]["item_prior_only"][oc["target"]]} for o, oc in objs.items()}
    write_json(ctx.out_dir / "baseline_summary.json", summary)
    write_budget(ctx, layers)
    return {"artifacts": ["baseline_summary.json", "baseline_scores.parquet", "environment.json", "budget.json"]}


def write_budget(ctx: Ctx, layers: list[int]):
    plan = plan_work(ctx, n_layers=len(layers))
    pps = ctx.timing.get("baseline", {}).get("pairs_per_second")
    remaining = plan.get("sweep_forward_pairs_upper_bound", 0) + plan.get("evaluate_forward_pairs_upper_bound", 0)
    write_json(ctx.out_dir / "budget.json", {
        "measured": {k: v for k, v in ctx.timing.items()},
        "peak_memory_bytes": ctx._scorer.peak_memory_bytes if ctx._scorer else None,
        "effective_microbatch": ctx._scorer.microbatch if ctx._scorer else None,
        "plan": plan, "estimated_remaining_forward_pairs": remaining,
        "estimated_remaining_seconds": remaining / pps if pps else None,
        "work_cap_forward_pairs": ctx.work_cap,
        "note": "estimates use measured baseline throughput; cached pair-configurations are not re-run",
    })


def stage_fit_directions(ctx: Ctx) -> dict:
    d = ctx.data
    layers = ctx.layers()
    cfg = ctx.cfg
    pool = capture_pool(ctx)
    ps, states, _ = ctx.run_job(pool, capture_layers=tuple(layers), require_cached=True)
    row_of = {p: i for i, p in enumerate(pool)}
    dp, pv = d["direction_pairs"], d["probe_val_pairs"]
    # leakage guards: only fitting-period pairs from the correct user groups reach this stage
    grp = d["groups"].set_index("user_id")["user_group"]
    for name, df, g in (("direction", dp, "inner_fit"), ("probe-val", pv, "probe_val")):
        splits = {d["queries_by_id"][q]["split"] for q in df["query_id"]}
        if splits - {"fit"} or set(grp.loc[df["user_id"].unique()]) - {g}:
            raise AssertionError(f"{name} pairs include non-fitting data")

    ddir = ctx.out_dir / "directions"
    index, probe_rows, fail_rows = {}, [], []
    seeds = ctx.profile["direction_seeds"]

    def save_dir(did: str, vec: np.ndarray, meta: dict):
        vec = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if not np.all(np.isfinite(vec)) or abs(n - 1) > 1e-4:
            raise ValueError(f"{did}: invalid unit vector (norm {n})")
        write_npz(ddir / f"{did}.npz", vector=vec)
        meta = dict(meta, direction_id=did, file=f"{did}.npz", vector_sha=array_hash(vec), norm=n,
                    hook_location=cfg["scoring"]["hook_location"])
        write_json(ddir / f"{did}.json", meta)
        index[did] = meta

    stats = {"seeds": {}}
    for seed in seeds:
        fit_all = dp[dp["direction_seed"] == seed]
        union_ids = sorted(set(fit_all["pair_id"]))
        rows_u = [row_of[p] for p in union_ids]
        base_fit = ps.set_index("pair_id").loc[union_ids, "score"].to_numpy()
        items_fit = [split_pair_id(p)[1] for p in union_ids]
        sstats = {"base_sd": float(np.std(base_fit)), "n_fit_pairs": len(union_ids), "layers": {}}
        for adj_name, vals in (("item_prior", item_prior_values(ctx, items_fit)),
                               ("attribute_boost", attribute_values(ctx, items_fit))):
            sstats[adj_name] = {"adj_mean": float(vals.mean()), "adj_sd": float(vals.std()), "base_sd": sstats["base_sd"]}
        for li in layers:
            X_union = states[li][rows_u]
            radius = dir_mod.activation_radius(X_union)
            sstats["layers"][str(li)] = {"radius": radius}
            save_dir(f"random-L{li}-s{seed}", random_unit_vector(ctx.scorer.hidden_size, seed, li),
                     {"estimator": "random", "objective": None, "layer_index": li, "seed": seed, "radius": radius})
            for obj, ocfg in cfg["objectives"].items():
                f = fit_all[fit_all["objective"] == obj]
                v = pv[pv["objective"] == obj]
                X_fit, y_fit = states[li][[row_of[p] for p in f["pair_id"]]], f["label"].to_numpy()
                X_val, y_val = states[li][[row_of[p] for p in v["pair_id"]]], v["label"].to_numpy()
                common = {"objective": obj, "layer_index": li, "seed": seed, "radius": radius,
                          "fit_pair_ids_sha": stable_hash(sorted(f["pair_id"])), "label_rule": ocfg["fit_class_column"],
                          "support": {"fit_pos": int((y_fit == 1).sum()), "fit_neg": int((y_fit == 0).sum()),
                                      "fit_users": int(f["user_id"].nunique()), "probe_val": int(len(v))}}
                try:
                    res = dir_mod.fit_directions(X_fit, y_fit, X_val, y_val, cfg["directions"]["C_grid"], seed,
                                                 cfg["directions"]["max_iter"])
                except ValueError as e:
                    fail_rows.append({"objective": obj, "layer": li, "seed": seed, "error": str(e)})
                    continue
                pm = res["probe_model"]
                write_npz(ddir / f"{obj}-probe-L{li}-s{seed}.model.npz", w_raw=pm.w_raw, b_raw=np.array([pm.b_raw]),
                          coef_std=pm.coef_std, intercept_std=np.array([pm.intercept_std]), mean=pm.mean, scale=pm.scale)
                save_dir(f"{obj}-mean_diff-L{li}-s{seed}", res["mean_diff"], dict(common, estimator="mean_diff"))
                save_dir(f"{obj}-probe-L{li}-s{seed}", res["probe"], dict(common, estimator="probe", C=pm.C,
                         classes=pm.classes, converged=pm.converged, C_grid=res["C_grid"],
                         probe_model_file=f"{obj}-probe-L{li}-s{seed}.model.npz"))
                save_dir(f"{obj}-shuffled-L{li}-s{seed}", dir_mod.shuffled_label_direction(X_fit, y_fit, seed),
                         dict(common, estimator="shuffled_mean_diff"))
                dec_fit = pm.decision(states[li][rows_u])
                sstats["layers"][str(li)][f"probe_decision__{obj}"] = {
                    "adj_mean": float(dec_fit.mean()), "adj_sd": float(dec_fit.std()), "base_sd": sstats["base_sd"]}
                diag = res["diagnostics"]
                for est, dg in (("probe", diag["probe_decision_probe_val"]),
                                ("mean_diff", diag["mean_diff_projection_probe_val"])):
                    probe_rows.append({"objective": obj, "estimator": est, "layer": li, "seed": seed,
                                       "C": pm.C if est == "probe" else None,
                                       "converged": pm.converged if est == "probe" else None,
                                       "fit_pos": common["support"]["fit_pos"], "fit_neg": common["support"]["fit_neg"],
                                       "fit_users": common["support"]["fit_users"], "probe_val_n": dg["n"],
                                       "probe_val_positives": dg["positives"], "probe_val_negatives": dg["negatives"],
                                       "probe_val_users": int(v["user_id"].nunique()),
                                       "auroc": dg["auroc"], "auprc": dg["auprc"], "prevalence": dg["prevalence"],
                                       "not_estimable_reason": dg.get("not_estimable_reason"),
                                       "cosine_mean_diff_probe": diag["cosine_mean_diff_probe"]})
        stats["seeds"][str(seed)] = sstats
    write_json(ddir / "index.json", {"schema_version": SCHEMA_VERSION, "directions": index, "failed_fits": fail_rows})
    write_json(ctx.out_dir / "comparator_stats.json", stats)
    pd.DataFrame(probe_rows).to_csv(ctx.out_dir / "probes.csv", index=False)
    return {"artifacts": ["directions/index.json", "probes.csv", "comparator_stats.json"],
            "n_directions": len(index), "failed_fits": len(fail_rows)}


def sweep_configs(ctx: Ctx, dirs: dict) -> list[dict]:
    """Every validation LLM configuration (deduplicated later by intervention identity)."""
    p, cfg = ctx.profile, ctx.cfg
    layers = ctx.layers()
    strengths = [float(x) for x in p["strengths"]]
    rev = float(cfg["steering"]["reversal_strength"])
    out = []
    for seed in p["direction_seeds"]:
        for li in layers:
            for obj in cfg["objectives"]:
                for est, method in (("mean_diff", "mean_diff_steer"), ("probe", "probe_steer")):
                    did = f"{obj}-{est}-L{li}-s{seed}"
                    if did not in dirs:
                        continue
                    for s in strengths + [rev]:
                        out.append({"method": method, "objective": obj, "seed": seed, "layer": li, "strength": s,
                                    "direction_id": did, "selectable": s > 0, "diagnostic": "reversal" if s < 0 else None})
                if cfg["steering"]["controls_on_full_grid"]:
                    did = f"{obj}-shuffled-L{li}-s{seed}"
                    if did in dirs:
                        for s in strengths:
                            out.append({"method": "shuffled_label_direction", "objective": obj, "seed": seed, "layer": li,
                                        "strength": s, "direction_id": did, "selectable": True})
            if cfg["steering"]["controls_on_full_grid"]:
                for s in strengths:
                    for obj in cfg["objectives"]:      # one scoring job, evaluated under each objective
                        out.append({"method": "random_direction", "objective": obj, "seed": seed, "layer": li,
                                    "strength": s, "direction_id": f"random-L{li}-s{seed}", "selectable": True})
    for c in out:
        c["config_id"] = f"{c['method']}|{c['objective']}|s{c['seed']}|L{c['layer']}|{c['strength']:+g}"
    return out


def sweep_row(ctx: Ctx, cfg_row: dict, pq: pd.DataFrame, objective: str) -> dict:
    agg = met.aggregate(pq, metric_specs(ctx))
    oc = ctx.cfg["objectives"][objective]
    r = {k: cfg_row.get(k) for k in ("config_id", "method", "seed", "layer", "strength", "lambda", "selectable",
                                     "diagnostic", "job_key", "direction_id", "prompt_hash")}
    r["objective"] = objective
    r["target_metric"], r["guardrail_metric"] = oc["target"], oc["guardrail"]
    r["target"], r["guardrail"] = agg[oc["target"]], agg[oc["guardrail"]]
    r.update({f"m:{k}": v for k, v in agg.items()})
    return r


def stage_sweep(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    dirs = load_directions(ctx)
    layers = ctx.layers()
    val = split_pairs(ctx, "validation")
    base, vstates, base_key = ctx.run_job(val, capture_layers=tuple(layers), require_cached=True)
    pq_base = per_query(ctx, base, "validation")
    rows, checks = [], {}

    # ---- identity check: strength 0 reproduces unsteered scores on the same batches
    sub = val[: cfg["steering"]["identity_check_pairs"]]
    enc = ctx.encodings(sub)
    seqs = [list(x) for x in enc["input_ids"]]
    s = ctx.scorer
    ref = s.score(sub, seqs)
    checks["identity"] = []
    for li in layers:
        zero = Intervention(kind="add_block", layer_index=li, direction_id="zero", vector_sha="zero", strength=0.0, radius=1.0)
        r0 = s.score(sub, seqs, intervention=zero, delta=np.zeros(s.hidden_size, np.float32))
        diff = float(np.max(np.abs(r0.scores - ref.scores))) if len(sub) else 0.0
        cached = base.set_index("pair_id").loc[sub, "score"].to_numpy()
        checks["identity"].append({"layer": li, "max_abs_diff_same_batches": diff,
                                   "max_abs_diff_vs_cached_baseline": float(np.max(np.abs(ref.scores - cached))) if len(sub) else 0.0,
                                   "passed": diff <= 1e-6})
        if diff > 1e-6:
            raise RuntimeError(f"strength-0 identity check failed at layer {li}: {diff}")
    after = s.score(sub, seqs)
    checks["hooks_removed_unsteered_rerun_max_abs_diff"] = float(np.max(np.abs(after.scores - ref.scores))) if len(sub) else 0.0

    # ---- post-normalization constant-vector null control
    last = max(layers)
    seed0 = ctx.profile["direction_seeds"][0]
    nvec = random_unit_vector(s.hidden_size, seed0, -1, "postnorm_null")
    nd = {"layer_index": None, "direction_id": "postnorm_null", "vector": nvec,
          "radius": dirs[f"random-L{last}-s{seed0}"]["radius"]}
    iv, delta = steer_intervention(nd, cfg["steering"]["postnorm_null_strength"], kind="add_postnorm")
    pn, _, _ = ctx.run_job(val, intervention=iv, delta=delta, timing_key="sweep")
    m = base[["pair_id", "query_id", "score"]].merge(pn[["pair_id", "score"]], on="pair_id", suffixes=("", "_pn"))
    m["shift"] = m["score_pn"] - m["score"]
    within = m.groupby("query_id")["shift"].agg(lambda x: float(x.max() - x.min()))
    rb = met.rank_scores(base[["query_id", "item_id", "score"]])
    rp = met.rank_scores(pn[["query_id", "item_id", "score"]])
    same = rb.merge(rp, on=["query_id", "item_id"], suffixes=("_b", "_p"))
    changed_q = same.groupby("query_id").apply(lambda g: bool((g["rank_b"] != g["rank_p"]).any()), include_groups=False)
    checks["postnorm_null"] = {"queries": int(len(within)), "max_within_query_shift_range": float(within.max()),
                               "queries_with_changed_ranking": int(changed_q.sum()),
                               "mean_shift": float(m["shift"].mean())}
    if changed_q.mean() > 0.01 and within.max() > 1e-3:
        raise RuntimeError(f"post-normalization constant-vector control changed rankings materially: {checks['postnorm_null']}")
    write_json(ctx.out_dir / "sweep_checks.json", checks)

    # ---- unsteered and cached-score comparators
    stats = read_json(ctx.out_dir / "comparator_stats.json")
    objectives = list(cfg["objectives"])
    for obj in objectives:
        rows.append(sweep_row(ctx, {"config_id": "unsteered", "method": "unsteered", "selectable": False,
                                    "job_key": base_key}, pq_base, obj))
        rr = base[["query_id", "item_id"]].assign(score=random_ranking_scores(ctx, val))
        rows.append(sweep_row(ctx, {"config_id": "random_ranking", "method": "random_ranking", "selectable": False},
                              per_query(ctx, rr, "validation"), obj))
    rows += comparator_rows(ctx, base, vstates, stats, "validation", layers)

    # ---- LLM steering grid and controls
    cfgs = sweep_configs(ctx, dirs)
    by_job: dict[str, pd.DataFrame] = {}
    for c in cfgs:
        iv, delta = steer_intervention(dirs[c["direction_id"]], c["strength"])
        key, _ = ctx.job_key("base", None, iv)
        if key not in by_job:
            sc, _, _ = ctx.run_job(val, intervention=iv, delta=delta, timing_key="sweep")
            by_job[key] = per_query(ctx, sc, "validation")
        c["job_key"] = key
        met.assert_same_queries(pq_base, by_job[key], c["config_id"])
        rows.append(sweep_row(ctx, c, by_job[key], c["objective"]))

    # ---- objective-prompt baselines (separately versioned prompts)
    for obj in objectives:
        sc, _, key = ctx.run_job(val, variant="objective", objective=obj, timing_key="sweep")
        pq = per_query(ctx, sc, "validation")
        met.assert_same_queries(pq_base, pq, "objective_prompt")
        rows.append(sweep_row(ctx, {"config_id": f"objective_prompt|{obj}", "method": "objective_prompt",
                                    "selectable": False, "job_key": key,
                                    "prompt_hash": prompts.prompt_template_hash(cfg, "objective", obj)}, pq, obj))
    df = pd.DataFrame(rows)
    df.to_csv(ctx.out_dir / "validation_sweep.csv", index=False)
    write_budget(ctx, layers)
    return {"artifacts": ["validation_sweep.csv", "sweep_checks.json"], "rows": len(df)}


def comparator_rows(ctx: Ctx, base: pd.DataFrame, states: dict, stats: dict, split: str, layers: list[int],
                    only: Optional[dict] = None) -> list[dict]:
    """Cached-score comparators: attribute boost, item-rating prior, learned linear reranker."""
    rows = []
    lam_grid = [float(x) for x in ctx.cfg["comparators"]["lambda_grid"]]
    items = list(base["item_id"])
    b = base["score"].to_numpy()
    seeds = ctx.profile["direction_seeds"]
    s0 = stats["seeds"][str(seeds[0])]
    for obj in ctx.cfg["objectives"]:
        fams = [("item_prior", None, None, item_prior_values(ctx, items), s0["item_prior"])]
        if "attribute_column" in ctx.cfg["objectives"][obj]:
            fams.append(("attribute_boost", None, None, attribute_values(ctx, items), s0["attribute_boost"]))
        for seed in seeds:
            ss = stats["seeds"][str(seed)]
            for li in layers:
                st = ss["layers"][str(li)].get(f"probe_decision__{obj}")
                if st is None:
                    continue
                pm = np.load(ctx.out_dir / "directions" / f"{obj}-probe-L{li}-s{seed}.model.npz")
                dec = states[li].astype(np.float64) @ pm["w_raw"] + float(pm["b_raw"][0])
                fams.append(("linear_reranker", seed, li, dec, st))
        for method, seed, li, adj, st in fams:
            for lam in lam_grid:
                cid = f"{method}|{obj}|s{seed}|L{li}|lam{lam:g}"
                if only is not None and cid not in only:
                    continue
                sc = base[["query_id", "item_id"]].assign(score=standardized_adjust(b, adj, st, lam))
                row = sweep_row(ctx, {"config_id": cid, "method": method, "seed": seed, "layer": li,
                                      "lambda": lam, "selectable": True}, per_query(ctx, sc, split), obj)
                if only is not None:
                    row["_scores"] = sc
                rows.append(row)
    return rows


def stage_select(ctx: Ctx) -> dict:
    sweep_path = ctx.out_dir / "validation_sweep.csv"
    if not sweep_path.exists():
        raise PrerequisiteMissing(str(sweep_path), "run sweep")
    sweep_sha = sha256_file(sweep_path)
    sel_path = ctx.out_dir / "selected.json"
    if sel_path.exists():
        cur = read_json(sel_path)
        if cur["validation_sweep_sha256"] != sweep_sha:
            raise RuntimeError("selected.json is locked but validation_sweep.csv changed; selections are immutable. "
                               "Start a new --run-id instead of reselecting.")
        return {"artifacts": ["selected.json"], "reused": True, "selection_hash": sha256_file(sel_path)}
    df = pd.read_csv(sweep_path)
    df = df.astype(object).where(df.notna(), None)
    budgets = [float(x) for x in ctx.profile["guardrail_budgets"]]
    selections = []
    for obj in ctx.cfg["objectives"]:
        d = df[df["objective"] == obj]
        b = d[d["method"] == "unsteered"].iloc[0]
        baseline = {"target": float(b["target"]), "guardrail": float(b["guardrail"])}
        fam_keys = d[d["selectable"] == True][["method", "seed"]].drop_duplicates()  # noqa: E712
        for budget in budgets:
            for fk in fam_keys.itertuples(index=False):
                fam = d[(d["method"] == fk.method) & (d["seed"].astype(str) == str(fk.seed)) & (d["selectable"] == True)]  # noqa: E712
                cands = [{"config_id": r["config_id"], "target": float(r["target"]), "guardrail": float(r["guardrail"]),
                          "strength": abs(float(r["strength"] if r["strength"] is not None else (r["lambda"] or 0))),
                          "layer": int(r["layer"]) if r["layer"] is not None else -1} for _, r in fam.iterrows()]
                res = met.select_operating_point(cands, baseline, budget)
                chosen = fam[fam["config_id"] == res.get("config_id")]
                ch = chosen.iloc[0].to_dict() if len(chosen) else {}
                seed_v = None if fk.seed is None else int(float(fk.seed))
                sid = f"{obj}|{fk.method}|s{seed_v}|b{budget:g}"
                selections.append({"selection_id": sid, "objective": obj, "method": fk.method,
                                   "seed": seed_v, "budget": budget, **res,
                                   "layer": ch.get("layer"), "strength": ch.get("strength"), "lambda": ch.get("lambda"),
                                   "direction_id": ch.get("direction_id"),
                                   "evaluate_on_test": fk.method not in CONTROL_METHODS})
            ob = d[d["method"] == "objective_prompt"]
            if len(ob):
                r = ob.iloc[0]
                feas = met.guardrail_feasible(float(r["guardrail"]), baseline["guardrail"], budget)
                selections.append({"selection_id": f"{obj}|objective_prompt|sNone|b{budget:g}", "objective": obj,
                                   "method": "objective_prompt", "seed": None, "budget": budget, "status": "fixed",
                                   "config_id": r["config_id"], "is_baseline": False,
                                   "validation_target": float(r["target"]), "validation_guardrail": float(r["guardrail"]),
                                   "baseline_target": baseline["target"], "baseline_guardrail": baseline["guardrail"],
                                   "validation_feasible": feas, "evaluate_on_test": True})
    sel = {"schema_version": SCHEMA_VERSION, "created_at": now_iso(), "run_id": ctx.run_id,
           "validation_sweep_sha256": sweep_sha, "rule": "maximize target among feasible (guard >= (1-budget)*baseline "
           "guard, unsteered included); ties: better guardrail, smaller |strength|, layer order",
           "selections": selections}
    write_json(sel_path, sel)
    h = sha256_file(sel_path)
    write_text(ctx.out_dir / "selected.json.sha256", h + "\n")
    return {"artifacts": ["selected.json"], "selection_hash": h}


def verify_selected(ctx: Ctx) -> dict:
    p = ctx.out_dir / "selected.json"
    hp = ctx.out_dir / "selected.json.sha256"
    if not p.exists() or not hp.exists():
        raise PrerequisiteMissing(str(p), "run select")
    h = sha256_file(p)
    if h != hp.read_text().strip():
        raise RuntimeError("selected.json was modified after selection (hash mismatch)")
    return read_json(p) | {"_hash": h}


def stage_evaluate(ctx: Ctx) -> dict:
    sel = verify_selected(ctx)
    dirs = load_directions(ctx)
    layers = ctx.layers()
    stats = read_json(ctx.out_dir / "comparator_stats.json")
    test = split_pairs(ctx, "test")
    base, tstates, base_key = ctx.run_job(test, capture_layers=tuple(layers), timing_key="evaluate")
    pq_base = per_query(ctx, base, "test")
    score_frames = [base[["pair_id", "query_id", "item_id", "score"]].assign(
        selection_id="unsteered", method="unsteered", config_id="unsteered", objective=None)]
    pq_frames = []
    for obj in ctx.cfg["objectives"]:
        pq_frames.append(pq_base.assign(selection_id="unsteered", method="unsteered", objective=obj, config_id="unsteered"))
    notes, item_shift_cov = [], []
    fit_base = pd.read_parquet(ctx.out_dir / "baseline_scores.parquet")
    fit_base = fit_base[fit_base["split"] == "fit"].set_index("pair_id")["baseline_score"]

    def add(sc, pq, s, cfg_id, extra=None):
        met.assert_same_queries(pq_base, pq, s["selection_id"])
        score_frames.append(sc[["pair_id", "query_id", "item_id", "score"]].assign(
            selection_id=s["selection_id"], method=s["method"], config_id=cfg_id, objective=s["objective"], **(extra or {})))
        pq_frames.append(pq.assign(selection_id=s["selection_id"], method=s["method"], objective=s["objective"],
                                   config_id=cfg_id, **(extra or {})))

    for s in sel["selections"]:
        if not s.get("evaluate_on_test") or s.get("status") not in ("selected", "fixed"):
            continue
        if s.get("is_baseline"):
            notes.append(f"{s['selection_id']}: validation selected the unsteered point; test equals baseline")
            add(base, pq_base, s, "unsteered")
            continue
        m = s["method"]
        if m in STEER_METHODS:
            d = dirs[s["direction_id"]]
            iv, delta = steer_intervention(d, float(s["strength"]))
            sc, _, _ = ctx.run_job(test, intervention=iv, delta=delta, timing_key="evaluate")
            add(sc, per_query(ctx, sc, "test"), s, s["config_id"])
            ivr, deltar = steer_intervention(d, -float(s["strength"]))
            scr, _, _ = ctx.run_job(test, intervention=ivr, delta=deltar, timing_key="evaluate")
            rs = dict(s, selection_id=s["selection_id"] + "|reversed", method=m + "_reversed")
            add(scr, per_query(ctx, scr, "test"), rs, s["config_id"] + "|reversed")
            # mean item-score shift control from fitting pairs under the selected intervention
            dp = ctx.data["direction_pairs"]
            fp = sorted(dp[(dp["objective"] == s["objective"]) & (dp["direction_seed"] == s["seed"])]["pair_id"])
            fs, _, _ = ctx.run_job(fp, intervention=iv, delta=delta, timing_key="evaluate")
            fs["shift"] = fs["score"].to_numpy() - fit_base.loc[fs["pair_id"]].to_numpy()
            by_item = fs.groupby("item_id")["shift"].agg(["mean", "size"])
            min_sup = ctx.cfg["comparators"]["item_shift_min_support"]
            gmean = float(fs["shift"].mean())
            supported = by_item[by_item["size"] >= min_sup]["mean"]
            off = base["item_id"].map(supported)
            sup_mask = off.notna()
            ish = base[["pair_id", "query_id", "item_id"]].assign(score=base["score"] + off.fillna(gmean))
            ms = dict(s, selection_id=s["selection_id"] + "|item_shift", method="mean_item_score_shift")
            add(ish, per_query(ctx, ish, "test"), ms, s["config_id"] + "|item_shift")
            item_shift_cov.append({"selection_id": ms["selection_id"], "global_mean_offset": gmean,
                                   "fit_pairs": len(fs), "supported_items": int(len(supported)),
                                   "test_items": int(base["item_id"].nunique()),
                                   "test_items_supported": int(base.loc[sup_mask, "item_id"].nunique()),
                                   "test_interactions_supported_fraction": float(sup_mask.mean())})
        elif m == "objective_prompt":
            sc, _, _ = ctx.run_job(test, variant="objective", objective=s["objective"], timing_key="evaluate")
            add(sc, per_query(ctx, sc, "test"), s, s["config_id"])
        elif m in ("item_prior", "attribute_boost", "linear_reranker"):
            rows = comparator_rows(ctx, base, tstates, stats, "test", layers, only={s["config_id"]: True})
            r = [x for x in rows if x["objective"] == s["objective"]]
            if not r:
                raise RuntimeError(f"could not rebuild comparator {s['config_id']}")
            sc = r[0]["_scores"].assign(pair_id=base["pair_id"].to_numpy())
            add(sc, per_query(ctx, sc, "test"), s, s["config_id"])
        else:
            notes.append(f"{s['selection_id']}: method {m} has no test evaluation")
    scores = pd.concat(score_frames, ignore_index=True)
    assert_unique(scores, ["selection_id", "pair_id"], "scores_test")
    pqs = pd.concat(pq_frames, ignore_index=True)
    grp = ctx.data["groups"].set_index("user_id")["user_group"]
    pqs["user_group"] = pqs["user_id"].map(grp)
    write_parquet(ctx.out_dir / "scores_test.parquet", scores)
    write_parquet(ctx.out_dir / "per_query_metrics.parquet", pqs)
    write_json(ctx.out_dir / "evaluate_notes.json", {"selection_hash": sel["_hash"], "notes": notes,
                                                     "item_shift_coverage": item_shift_cov, "baseline_job": base_key})
    write_budget(ctx, layers)
    return {"artifacts": ["scores_test.parquet", "per_query_metrics.parquet", "evaluate_notes.json"]}


def stage_report(ctx: Ctx) -> dict:
    import report

    return report.build_report(ctx)


STAGE_FNS: dict[str, Callable[[Ctx], dict]] = {
    "prepare": stage_prepare, "baseline": stage_baseline, "fit-directions": stage_fit_directions,
    "sweep": stage_sweep, "select": stage_select, "evaluate": stage_evaluate, "report": stage_report,
}


# =================================================================== execution

def run_stage(ctx: Ctx, stage: str, resume: bool = False) -> str:
    st = ctx.status
    fp = ctx.fingerprint()
    cur = st.get(stage)
    if resume and cur.get("status") == "completed" and cur.get("fingerprint") == fp and stage != "report":
        print(f"[{stage}] completed earlier with matching configuration; skipping (--resume)")
        return "completed"
    for pre in PREREQS[stage]:
        ps = st.get(pre)
        if ps.get("status") != "completed" or ps.get("fingerprint") != fp:
            msg = f"prerequisite stage '{pre}' is {ps.get('status', 'not run')}" + \
                  (" (configuration changed)" if ps.get("status") == "completed" else "")
            st.set(stage, status="blocked", missing=f"stage:{pre}", error=msg, updated_at=now_iso())
            print(f"[{stage}] blocked: {msg}")
            return "blocked"
    st.set(stage, status="running", started_at=now_iso(), fingerprint=fp, error=None, missing=None)
    t0 = time.time()
    try:
        out = STAGE_FNS[stage](ctx) or {}
        status = out.pop("status", "completed")
        st.set(stage, status=status, finished_at=now_iso(), seconds=time.time() - t0, result=out)
    except PrerequisiteMissing as e:
        st.set(stage, status="blocked", missing=e.missing, error=str(e), finished_at=now_iso())
        status = "blocked"
    except NotEstimable as e:
        st.set(stage, status="not_estimable", error=str(e), finished_at=now_iso())
        status = "not_estimable"
    except WorkCapExceeded as e:
        st.set(stage, status="partial", error=str(e), finished_at=now_iso())
        status = "partial"
    except Exception as e:  # noqa: BLE001
        st.set(stage, status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(),
               finished_at=now_iso())
        status = "failed"
    ctx.add_timing(f"stage:{stage}", time.time() - t0, 0)
    print(f"[{stage}] {status}" + (f": {st.get(stage).get('error')}" if status != "completed" else ""))
    return status


def run_through(ctx: Ctx, through: str, resume: bool) -> str:
    last = "completed"
    for stage in STAGES[: STAGES.index(through) + 1]:
        last = run_stage(ctx, stage, resume=resume)
        if last != "completed":
            if through == "report" and stage != "report":
                run_stage(ctx, "report")          # always leave a machine-readable report of what exists
            break
    return last


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=STAGES + ["run"])
    ap.add_argument("--dataset", default="movielens_1m")
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--through", default="report", choices=STAGES)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-forward-pairs", type=int, default=None, help="work cap; exceeding it marks the stage partial")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, e.g. model.device=cpu")
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    a = ap.parse_args(argv)
    ctx = Ctx(a.dataset, a.profile, a.run_id, a.set, Path(a.root), a.max_forward_pairs)
    status = run_through(ctx, a.through, a.resume) if a.command == "run" else run_stage(ctx, a.command, a.resume)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
