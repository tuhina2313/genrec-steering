"""Dataset downloads, parsing, preprocessing, splits, histories, candidates, labels, audits.

All dataset-specific branching lives here (adapters dispatched by dataset key) and in configs.
Everything downstream consumes the dataset-agnostic tables written by `prepare_dataset`:

  items.parquet            item_id, display fields, metadata_missing flags, attribute flag
  queries.jsonl            one record per query (contracts.Query); history strictly before cutoff
  labels.parquet           keyed by (query_id, item_id); outcomes never enter prompts
  user_groups.parquet      user_id -> inner_fit | probe_val | unseen_by_direction
  direction_pairs.parquet  balanced inner-fit pairs per (objective, direction_seed)
  probe_val_pairs.parquet  natural-distribution probe-validation pairs per objective
  item_prior.parquet       smoothed item rating prior from permitted pre-validation records
  data_audit.json          splits, cutoffs, exclusions, histograms, coverage, realized counts
"""
from __future__ import annotations

import math
import re
import shutil
import ssl
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts import SCHEMA_VERSION, make_pair_id
from utils import (
    NotEstimable, PrerequisiteMissing, assert_unique, hash_uint, read_json, sha256_file, stable_hash,
    utc_iso, write_json, write_jsonl, write_parquet,
)

DATA_CODE_VERSION = "data_v1"
SPLITS = ("fit", "validation", "test")
GROUPS = ("inner_fit", "probe_val", "unseen_by_direction")


# =============================================================== adapters

class DatasetAdapter:
    key: str = ""

    def ensure_raw(self, cfg: dict, raw_dir: Path) -> dict:
        raise NotImplementedError

    def load(self, cfg: dict, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Return events [user_id, item_id, timestamp, + feedback cols], items, parse audit."""
        raise NotImplementedError

    def history_feedback(self, row) -> dict:
        raise NotImplementedError

    def label_columns(self, outcome: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        """Add label columns to outcome rows (one per query x candidate)."""
        raise NotImplementedError

    def item_prior_value_column(self) -> str:
        raise NotImplementedError


class MovieLensAdapter(DatasetAdapter):
    key = "movielens_1m"
    YEAR_RE = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")

    # ------------------------------------------------ raw files
    def ensure_raw(self, cfg: dict, raw_dir: Path) -> dict:
        ds = cfg["dataset"]
        raw_dir.mkdir(parents=True, exist_ok=True)
        archive = raw_dir / ds["archive_name"]
        inner = raw_dir / ds["inner_dir"]
        needed = [inner / ds["ratings_file"], inner / ds["movies_file"]]
        if not archive.exists() and not all(p.exists() for p in needed):
            try:
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(ds["archive_url"], timeout=120, context=ctx) as r:
                    data = r.read()
                tmp = archive.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(archive)
            except Exception as e:  # network policy, offline, etc.
                raise PrerequisiteMissing(
                    str(archive), f"download from {ds['archive_url']} failed ({type(e).__name__}: {e}); "
                    "place the official archive at this path and rerun")
        if archive.exists() and not all(p.exists() for p in needed):
            with zipfile.ZipFile(archive) as z:
                for name in (ds["ratings_file"], ds["movies_file"], "README"):
                    member = f"{ds['inner_dir']}/{name}"
                    if member in z.namelist():
                        inner.mkdir(parents=True, exist_ok=True)
                        with z.open(member) as src, open(inner / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        for p in needed:
            if not p.exists():
                raise PrerequisiteMissing(str(p), "archive did not contain the expected file")
        checksums = {"files": {str(p.relative_to(raw_dir)): sha256_file(p) for p in needed}}
        if archive.exists():
            checksums["archive_sha256"] = sha256_file(archive)
        write_json(raw_dir / "checksums.json", checksums)
        return checksums

    # ------------------------------------------------ parsing
    def load(self, cfg: dict, raw_dir: Path):
        ds = cfg["dataset"]
        inner = raw_dir / ds["inner_dir"]
        enc, delim = ds["encoding"], ds["delimiter"]
        audit: dict[str, Any] = {"encoding": enc, "delimiter": delim}

        rows, malformed = [], []
        with open(inner / ds["ratings_file"], "r", encoding=enc, newline="") as f:
            for ln, line in enumerate(f, 1):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split(delim)
                try:
                    if len(parts) != 4:
                        raise ValueError("field count")
                    u, i, r, t = (int(p) for p in parts)
                    if not (1 <= r <= 5) or t < 0:
                        raise ValueError("range")
                    rows.append((u, i, r, t))
                except ValueError as e:
                    malformed.append({"line": ln, "reason": str(e)})
        ev = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
        audit["ratings_rows_read"] = len(ev) + len(malformed)
        audit["ratings_malformed"] = len(malformed)
        audit["ratings_malformed_examples"] = malformed[:20]

        n_before = len(ev)
        ev = ev.drop_duplicates()
        audit["ratings_exact_duplicates_removed"] = n_before - len(ev)

        mrows, mmal = [], []
        with open(inner / ds["movies_file"], "r", encoding=enc, newline="") as f:
            for ln, line in enumerate(f, 1):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split(delim)
                if len(parts) != 3 or not parts[0].strip().isdigit():
                    mmal.append({"line": ln, "reason": "field count or id"})
                    continue
                mid, title_raw, genres_raw = int(parts[0]), parts[1].strip(), parts[2].strip()
                m = self.YEAR_RE.match(title_raw)
                title, year = (m.group(1), int(m.group(2))) if m else (title_raw, None)
                genres = [g for g in genres_raw.split("|") if g and g != "(no genres listed)"]
                mrows.append({"item_id": mid, "title": title, "year": year, "genres": genres})
        items = pd.DataFrame(mrows)
        audit["movies_malformed"] = len(mmal)
        audit["movies_duplicate_ids"] = int(items["item_id"].duplicated().sum())
        items = items.drop_duplicates("item_id", keep="first")
        items["metadata_missing_year"] = items["year"].isna()
        items["metadata_missing_genres"] = items["genres"].map(len) == 0
        genre = ds["attribute_genre"]
        items["attribute_flag"] = items["genres"].map(lambda g: int(genre in g))

        known = set(items["item_id"])
        unmatched = ev.loc[~ev["item_id"].isin(known), "item_id"]
        audit["ratings_unmatched_item_ids"] = int(unmatched.nunique())
        audit["ratings_unmatched_rows"] = int(len(unmatched))
        audit["ratings_unmatched_examples"] = sorted(unmatched.unique().tolist())[:20]
        ev = ev[ev["item_id"].isin(known)]

        same_ui = ev.duplicated(["user_id", "item_id"], keep=False)
        audit["repeated_user_item_rows"] = int(same_ui.sum())
        audit["metadata_coverage"] = {
            "items": int(len(items)),
            "missing_year": int(items["metadata_missing_year"].sum()),
            "missing_genres": int(items["metadata_missing_genres"].sum()),
            f"{genre}_items": int(items["attribute_flag"].sum()),
        }
        for c in ("user_id", "item_id"):
            ev[c] = ev[c].astype(str)
        items["item_id"] = items["item_id"].astype(str)
        audit["events_after_cleaning"] = int(len(ev))
        audit["users"] = int(ev["user_id"].nunique())
        audit["timestamp_unit"] = "seconds since epoch (unchanged from source)"
        return ev.reset_index(drop=True), items.reset_index(drop=True), audit

    def history_feedback(self, row) -> dict:
        return {"rating": int(row.rating)}

    def label_columns(self, outcome: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        flags = items.set_index("item_id")["attribute_flag"]
        out = outcome.copy()
        r = out["rating"].astype(int)
        out["satisfaction_label"] = (r >= 4).astype(int)
        out["graded_gain"] = (r - 1).astype(float)
        out["low_rating"] = (r <= 2).astype(int)
        out["target_attribute"] = out["item_id"].map(flags).astype(int)
        sat = pd.Series(np.nan, index=out.index)
        sat[r >= 4] = 1.0
        sat[r <= 2] = 0.0
        out["fit_class__satisfaction"] = sat
        out["fit_class__comedy"] = out["target_attribute"].astype(float)
        out["label_observed"] = True
        return out

    def item_prior_value_column(self) -> str:
        return "rating"


class KuaiRandAdapter(DatasetAdapter):
    key = "kuairand_pure"

    def ensure_raw(self, cfg, raw_dir):
        raise PrerequisiteMissing(
            "docs/kuairand_memo.md", "KuaiRand-Pure adapter awaits the companion implementation memo")

    def load(self, cfg, raw_dir):
        raise PrerequisiteMissing("docs/kuairand_memo.md", "KuaiRand-Pure adapter not implemented")


ADAPTERS: dict[str, DatasetAdapter] = {a.key: a for a in (MovieLensAdapter(), KuaiRandAdapter())}


def get_adapter(key: str) -> DatasetAdapter:
    if key not in ADAPTERS:
        raise KeyError(f"unknown dataset key {key!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[key]


# =============================================================== splits

def compute_cutoffs(timestamps: np.ndarray, percentiles: list[float]) -> list[dict]:
    """Global event-percentile cutoffs that never split equal timestamps.

    A cutoff is a timestamp value v; events with t < v are before it. For each requested
    percentile q we pick the smallest distinct timestamp v with count(t < v) >= ceil(q * n),
    i.e. we move forward to the next distinct timestamp when q falls inside a tie group.
    """
    s = np.sort(np.asarray(timestamps, dtype=np.int64))
    n = len(s)
    uniq = np.unique(s)
    before = np.searchsorted(s, uniq, side="left")
    out, prev = [], None
    for q in percentiles:
        need = int(math.ceil(q * n))
        idx = int(np.searchsorted(before, need, side="left"))
        v = int(uniq[idx]) if idx < len(uniq) else int(uniq[-1]) + 1
        nb = int(np.searchsorted(s, v, side="left"))
        if prev is not None and v <= prev:
            raise NotEstimable(f"cutoff for percentile {q} collapsed onto previous cutoff")
        out.append({"percentile": q, "timestamp": v, "utc": utc_iso(v), "events_before": nb,
                    "realized_fraction": nb / n if n else None, "moved_to_next_distinct": nb != need})
        prev = v
    return out


def assign_user_groups(users: list[str], cfg: dict) -> pd.DataFrame:
    seed = cfg["seeds"]["data_seed"]
    rf, inner = cfg["splits"]["reserved_fraction"], cfg["splits"]["inner_fit_fraction"]
    recs = []
    for u in sorted(users):
        if hash_uint(seed, "reserve", u) / 2**64 < rf:
            g = "unseen_by_direction"
        elif hash_uint(seed, "inner", u) / 2**64 < inner:
            g = "inner_fit"
        else:
            g = "probe_val"
        recs.append((u, g))
    return pd.DataFrame(recs, columns=["user_id", "user_group"])


# =============================================================== queries

def _item_record(items_idx: dict, item_id: str) -> tuple[dict, dict]:
    it = items_idx[item_id]
    meta = {"title": it["title"], "year": None if pd.isna(it["year"]) else int(it["year"]),
            "genres": list(it["genres"])}
    missing = {"year": bool(it["metadata_missing_year"]), "genres": bool(it["metadata_missing_genres"])}
    return meta, missing


def build_queries(adapter: DatasetAdapter, events: pd.DataFrame, items: pd.DataFrame, cutoffs: list[int],
                  groups: pd.DataFrame, cfg: dict, profile: dict):
    """Return (query records, outcome rows, exclusion audit)."""
    ds = cfg["dataset"]
    seed = cfg["seeds"]["data_seed"]
    hmax, hmin = cfg["history"]["max_items"], cfg["history"]["min_items"]
    cmin, cmax = cfg["candidates"]["min_count"], profile["max_candidates"]
    group_of = dict(zip(groups["user_id"], groups["user_group"]))
    items_idx = {r["item_id"]: r for r in items.to_dict("records")}
    bounds = {"fit": (cutoffs[0], cutoffs[1]), "validation": (cutoffs[1], cutoffs[2]),
              "test": (cutoffs[2], np.iinfo(np.int64).max)}

    ev = events.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    feedback_cols = [c for c in ev.columns if c not in ("user_id", "item_id", "timestamp")]
    queries, outcomes = [], []
    excl = {s: Counter() for s in SPLITS}
    hist_cand = {s: {"eligible_before_cap": Counter(), "after_cap": Counter()} for s in SPLITS}
    repeats = Counter()
    for uid, uev in ev.groupby("user_id", sort=True):
        ts = uev["timestamp"].to_numpy()
        g = group_of[uid]
        for split in SPLITS:
            start, end = bounds[split]
            qid = f"{ds['key']}:{ds['protocol_id']}:{split}:{uid}"
            outc = uev[(ts >= start) & (ts < end)]
            if len(outc) == 0:
                excl[split]["no_outcome_events"] += 1
                continue
            hist_all = uev[ts < start]
            if len(hist_all) < hmin:
                excl[split]["insufficient_history"] += 1
                continue
            if split == "fit" and g == "unseen_by_direction":
                excl[split]["reserved_user_not_used_for_fitting"] += 1
                continue
            if split == "validation" and g == "unseen_by_direction":
                excl[split]["reserved_user_excluded_from_validation"] += 1
                continue
            # repeated ratings of the same item inside the outcome window: keep earliest
            first = outc.drop_duplicates("item_id", keep="first")
            repeats[split] += len(outc) - len(first)
            seen = set(hist_all["item_id"])
            first = first[~first["item_id"].isin(seen)]
            n_elig = len(first)
            hist_cand[split]["eligible_before_cap"][n_elig] += 1
            if split == "fit":
                if n_elig < 1:
                    excl[split]["no_unseen_candidates"] += 1
                    continue
                chosen = first
            else:
                if n_elig < cmin:
                    excl[split]["too_few_candidates"] += 1
                    continue
                order = sorted(first["item_id"], key=lambda it: hash_uint(seed, "cand_sample", qid, it))
                keep = set(order[:cmax])
                chosen = first[first["item_id"].isin(keep)]
            # final presentation order: fixed item-independent hash shuffle, unrelated to labels/time
            cand_ids = sorted(chosen["item_id"], key=lambda it: hash_uint(seed, "cand_order", qid, it))
            hist_cand[split]["after_cap"][len(cand_ids)] += 1
            hist = hist_all.iloc[-hmax:]
            history = []
            for row in hist.itertuples(index=False):
                meta, missing = _item_record(items_idx, row.item_id)
                history.append({"item_id": row.item_id, "timestamp": int(row.timestamp),
                                "feedback": adapter.history_feedback(row), "metadata": meta,
                                "metadata_missing": missing})
            assert all(h["timestamp"] < start for h in history)
            queries.append({
                "schema_version": SCHEMA_VERSION, "query_id": qid, "user_id": uid, "dataset": ds["key"],
                "split": split, "cutoff_timestamp": int(start), "history": history,
                "candidate_ids": cand_ids, "candidate_source": ds["candidate_source"],
                "protocol_id": ds["protocol_id"], "user_group": g,
                "n_history_available": int(len(hist_all)), "n_candidates_eligible": int(n_elig),
            })
            o = chosen[["item_id", "timestamp"] + feedback_cols].copy()
            o = o.rename(columns={"timestamp": "observed_timestamp"})
            o.insert(0, "query_id", qid)
            o.insert(1, "user_id", uid)
            o["split"] = split
            outcomes.append(o)
    outcome_df = pd.concat(outcomes, ignore_index=True) if outcomes else pd.DataFrame()
    audit = {
        "exclusions": {s: dict(excl[s]) for s in SPLITS},
        "candidate_count_histograms": {
            s: {k: {str(n): c for n, c in sorted(v.items())} for k, v in hist_cand[s].items()} for s in SPLITS},
        "repeated_ratings_in_outcome_windows_dropped": dict(repeats),
        "repeated_rating_rule": ds.get("repeated_rating_rule"),
    }
    return queries, outcome_df, audit


def select_eval_users(queries: list[dict], split: str, max_users: int, seed: int) -> tuple[list[dict], dict]:
    """Deterministic hash selection of evaluation users, blind to labels and scores."""
    elig = [q for q in queries if q["split"] == split]
    elig.sort(key=lambda q: hash_uint(seed, "eval_select", split, q["user_id"]))
    chosen = elig[:max_users]
    return chosen, {"eligible": len(elig), "selected": len(chosen), "not_selected_cap": len(elig) - len(chosen)}


# =============================================================== direction pairs

def select_direction_pairs(fit_labels: pd.DataFrame, objective: str, class_col: str, cap_per_class: int,
                           per_user_cap: int, direction_seed: int) -> tuple[pd.DataFrame, dict]:
    """Balanced inner-fit pairs. Expects rows of split=='fit' and user_group=='inner_fit' only."""
    if not (fit_labels["split"] == "fit").all() or not (fit_labels["user_group"] == "inner_fit").all():
        raise AssertionError("direction pairs must come only from inner-fit users' fitting-period labels")
    pool = fit_labels[fit_labels[class_col].notna()].copy()
    pool["label"] = pool[class_col].astype(int)
    pool["_h"] = [hash_uint(direction_seed, "dir_sample", objective, p) for p in pool["pair_id"]]
    pool = pool.sort_values(["_h", "pair_id"])
    per_class = {}
    for cls in (1, 0):
        taken, per_user = [], Counter()
        for row in pool[pool["label"] == cls].itertuples(index=False):
            if per_user[row.user_id] >= per_user_cap:
                continue
            per_user[row.user_id] += 1
            taken.append(row)
            if len(taken) >= cap_per_class:
                break
        per_class[cls] = taken
    n = min(len(per_class[1]), len(per_class[0]))
    rows = per_class[1][:n] + per_class[0][:n]
    sel = pd.DataFrame(rows)[["pair_id", "query_id", "item_id", "user_id", "label"]] if rows else \
        pd.DataFrame(columns=["pair_id", "query_id", "item_id", "user_id", "label"])
    sel.insert(0, "direction_seed", direction_seed)
    sel.insert(0, "objective", objective)
    info = {"objective": objective, "direction_seed": direction_seed,
            "available_pos": int((pool["label"] == 1).sum()), "available_neg": int((pool["label"] == 0).sum()),
            "requested_per_class": cap_per_class, "realized_per_class": n,
            "users": int(sel["user_id"].nunique()) if len(sel) else 0, "per_user_class_cap": per_user_cap}
    return sel.reset_index(drop=True), info


def select_probe_val_pairs(pv_labels: pd.DataFrame, objective: str, class_col: str, cap: int,
                           seed: int) -> tuple[pd.DataFrame, dict]:
    """Natural class distribution, deterministic outcome-independent hash order, capped."""
    if not (pv_labels["split"] == "fit").all() or not (pv_labels["user_group"] == "probe_val").all():
        raise AssertionError("probe-validation pairs must come only from probe_val users' fitting period")
    pool = pv_labels[pv_labels[class_col].notna()].copy()
    pool["_h"] = [hash_uint(seed, "probe_val", p) for p in pool["pair_id"]]
    sel = pool.sort_values(["_h", "pair_id"]).head(cap)
    sel = sel.assign(label=sel[class_col].astype(int), objective=objective)
    sel = sel[["objective", "pair_id", "query_id", "item_id", "user_id", "label"]].reset_index(drop=True)
    info = {"objective": objective, "eligible": int(len(pool)), "selected": int(len(sel)),
            "positives": int(sel["label"].sum()), "negatives": int((sel["label"] == 0).sum()),
            "prevalence": float(sel["label"].mean()) if len(sel) else None,
            "users": int(sel["user_id"].nunique())}
    return sel, info


# =============================================================== item prior

def item_prior(events: pd.DataFrame, groups: pd.DataFrame, before_ts: int, value_col: str,
               smoothing: float) -> tuple[pd.DataFrame, dict]:
    """Smoothed item mean from records before validation, excluding reserved users' labels."""
    reserved = set(groups.loc[groups["user_group"] == "unseen_by_direction", "user_id"])
    ok = events[(events["timestamp"] < before_ts) & (~events["user_id"].isin(reserved))]
    g = float(ok[value_col].mean())
    agg = ok.groupby("item_id")[value_col].agg(["sum", "count", "mean"]).reset_index()
    agg["prior"] = (agg["sum"] + smoothing * g) / (agg["count"] + smoothing)
    agg = agg.rename(columns={"count": "n_records", "mean": "raw_mean"})[["item_id", "n_records", "raw_mean", "prior"]]
    return agg, {"global_mean": g, "n_records": int(len(ok)), "n_items": int(len(agg)),
                 "smoothing": smoothing, "records_before_timestamp": int(before_ts),
                 "excluded_user_group": "unseen_by_direction"}


# =============================================================== orchestration

def prepare_dataset(cfg: dict, profile_name: str, raw_dir: Path, out_dir: Path) -> dict:
    """Build every processed table for one dataset/profile. Returns the processed manifest."""
    profile = cfg["profiles"][profile_name]
    adapter = get_adapter(cfg["dataset"]["key"])
    checksums = adapter.ensure_raw(cfg, raw_dir)
    input_key = stable_hash({"raw": checksums, "dataset": cfg["dataset"], "objectives": cfg["objectives"],
                             "history": cfg["history"], "candidates": cfg["candidates"], "splits": cfg["splits"],
                             "seeds": cfg["seeds"]["data_seed"], "profile": profile,
                             "directions": cfg["directions"], "comparators": cfg["comparators"],
                             "code": DATA_CODE_VERSION, "schema": SCHEMA_VERSION})
    man_path = out_dir / "manifest.json"
    if man_path.exists():
        man = read_json(man_path)
        if man.get("input_key") == input_key and all((out_dir / f).exists() for f in man["files"]):
            man["reused"] = True
            return man

    events, items, parse_audit = adapter.load(cfg, raw_dir)
    pcts = cfg["splits"]["percentiles"]
    cut = compute_cutoffs(events["timestamp"].to_numpy(), pcts)
    cutoffs = [c["timestamp"] for c in cut]
    period = np.searchsorted(np.array(cutoffs), events["timestamp"].to_numpy(), side="right")
    names = np.array(["history", "fit", "validation", "test"])
    period_counts = Counter(names[period].tolist())

    groups = assign_user_groups(events["user_id"].unique().tolist(), cfg)
    queries, outcomes, q_audit = build_queries(adapter, events, items, cutoffs, groups, cfg, profile)

    seed = cfg["seeds"]["data_seed"]
    val_q, val_sel = select_eval_users(queries, "validation", profile["max_validation_users"], seed)
    test_q, test_sel = select_eval_users(queries, "test", profile["max_test_users"], seed)
    fit_q = [q for q in queries if q["split"] == "fit"]
    kept = fit_q + val_q + test_q
    kept_ids = {q["query_id"] for q in kept}
    outcomes = outcomes[outcomes["query_id"].isin(kept_ids)].reset_index(drop=True)
    gmap = groups.set_index("user_id")["user_group"]
    outcomes["user_group"] = outcomes["user_id"].map(gmap)

    labels = adapter.label_columns(outcomes, items)
    labels.insert(0, "pair_id", [make_pair_id(q, i) for q, i in zip(labels["query_id"], labels["item_id"])])
    assert_unique(labels, ["query_id", "item_id"], "labels")
    assert_unique(labels, ["pair_id"], "labels")
    n_q_pairs = sum(len(q["candidate_ids"]) for q in kept)
    if n_q_pairs != len(labels):
        raise AssertionError(f"label rows {len(labels)} != query candidate pairs {n_q_pairs}")

    # ---- direction fitting pairs and probe-validation pairs
    fit_lab = labels[labels["split"] == "fit"]
    dir_frames, dir_info, pv_frames, pv_info = [], [], [], []
    dcfg = cfg["directions"]
    for obj, ocfg in cfg["objectives"].items():
        col = ocfg["fit_class_column"]
        for dseed in profile["direction_seeds"]:
            sel, info = select_direction_pairs(fit_lab[fit_lab["user_group"] == "inner_fit"], obj, col,
                                               profile["direction_examples_per_class"],
                                               dcfg["per_user_class_cap"], dseed)
            dir_frames.append(sel)
            dir_info.append(info)
        sel, info = select_probe_val_pairs(fit_lab[fit_lab["user_group"] == "probe_val"], obj, col,
                                           profile["probe_val_cap"], seed)
        pv_frames.append(sel)
        pv_info.append(info)
    dir_pairs = pd.concat(dir_frames, ignore_index=True)
    pv_pairs = pd.concat(pv_frames, ignore_index=True)
    assert_unique(dir_pairs, ["objective", "direction_seed", "pair_id"], "direction pairs")
    assert_unique(pv_pairs, ["objective", "pair_id"], "probe-val pairs")
    assert not set(dir_pairs["user_id"]) & set(pv_pairs["user_id"]), "inner-fit and probe-val users overlap"

    prior, prior_info = item_prior(events, groups, cutoffs[1], adapter.item_prior_value_column(),
                                   cfg["comparators"]["item_prior_smoothing"])

    # ---- only fit queries that contribute capture pairs are needed downstream
    capture_ids = set(dir_pairs["pair_id"]) | set(pv_pairs["pair_id"])
    capture_qids = {p.rpartition("#")[0] for p in capture_ids}
    kept = [q for q in kept if q["split"] != "fit" or q["query_id"] in capture_qids]
    kept.sort(key=lambda q: q["query_id"])
    labels = labels[labels["query_id"].isin({q["query_id"] for q in kept})].reset_index(drop=True)

    # ---- realized counts and audit
    def split_summary(qs):
        if not qs:
            return {"queries": 0, "users": 0, "pairs": 0}
        n = [len(q["candidate_ids"]) for q in qs]
        return {"queries": len(qs), "users": len({q["user_id"] for q in qs}), "pairs": int(sum(n)),
                "candidates_min": int(min(n)), "candidates_median": float(np.median(n)), "candidates_max": int(max(n)),
                "by_group": dict(Counter(q["user_group"] for q in qs))}

    by_split = {s: [q for q in kept if q["split"] == s] for s in SPLITS}
    lab_eval = labels[labels["split"].isin(["validation", "test"])]
    zero_pos = lab_eval.groupby("query_id")["satisfaction_label"].sum().eq(0)
    audit = {
        "schema_version": SCHEMA_VERSION, "dataset": cfg["dataset"]["key"], "protocol_id": cfg["dataset"]["protocol_id"],
        "profile": profile_name, "parse": parse_audit,
        "split_cutoffs": cut, "period_event_counts": dict(period_counts),
        "split_rule": "history: t < c1; fit: c1 <= t < c2; validation: c2 <= t < c3; test: t >= c3. "
                      "Each query's cutoff is its period start; history uses only t < cutoff.",
        "user_groups": dict(Counter(groups["user_group"])),
        "query_construction": q_audit,
        "evaluation_user_selection": {"validation": val_sel, "test": test_sel,
                                      "rule": "deterministic hash of (data_seed, split, user_id); label/score blind"},
        "realized": {s: split_summary(by_split[s]) for s in SPLITS},
        "label_prevalence": {
            s: {"satisfaction_label": float(labels.loc[labels["split"] == s, "satisfaction_label"].mean())
                if (labels["split"] == s).any() else None,
                "target_attribute": float(labels.loc[labels["split"] == s, "target_attribute"].mean())
                if (labels["split"] == s).any() else None}
            for s in SPLITS},
        "zero_positive_query_rate": float(zero_pos.mean()) if len(zero_pos) else None,
        "direction_pairs": dir_info, "probe_val_pairs": pv_info, "item_prior": prior_info,
        "candidate_source": cfg["dataset"]["candidate_source"],
        "limitations": [
            "Candidates are movies the user actually rated in the outcome period (retrospective benchmark "
            "information), not a deployable retrieval system; metrics measure preference ranking within "
            "that pool, not next-watch prediction or exhaustive retrieval.",
            "MovieLens timestamps are rating times, not watch times.",
            "Item metadata and movie knowledge may be represented in the pretrained LLM.",
        ],
    }
    min_q = profile.get("min_eval_queries", cfg["splits"]["min_eval_queries"])
    status = "ok"
    reasons = []
    for s in ("validation", "test"):
        if audit["realized"][s]["queries"] < min_q:
            status = "not_estimable"
            reasons.append(f"{s}: {audit['realized'][s]['queries']} eligible queries < {min_q}")
    for info in dir_info:
        if info["realized_per_class"] == 0:
            status = "not_estimable"
            reasons.append(f"no balanced direction pairs for {info['objective']} seed {info['direction_seed']}")
    audit["status"] = status
    audit["not_estimable_reasons"] = reasons

    out_dir.mkdir(parents=True, exist_ok=True)
    items_out = items.copy()
    write_parquet(out_dir / "items.parquet", items_out)
    write_jsonl(out_dir / "queries.jsonl", kept)
    write_parquet(out_dir / "labels.parquet", labels)
    write_parquet(out_dir / "user_groups.parquet", groups)
    write_parquet(out_dir / "direction_pairs.parquet", dir_pairs)
    write_parquet(out_dir / "probe_val_pairs.parquet", pv_pairs)
    write_parquet(out_dir / "item_prior.parquet", prior)
    write_json(out_dir / "data_audit.json", audit)
    files = ["items.parquet", "queries.jsonl", "labels.parquet", "user_groups.parquet", "direction_pairs.parquet",
             "probe_val_pairs.parquet", "item_prior.parquet", "data_audit.json"]
    man = {"schema_version": SCHEMA_VERSION, "input_key": input_key, "files": files,
           "file_sha256": {f: sha256_file(out_dir / f) for f in files},
           "queries_sha256": sha256_file(out_dir / "queries.jsonl"),
           "labels_sha256": sha256_file(out_dir / "labels.parquet"),
           "status": status, "split_cutoffs": cut, "reused": False}
    write_json(man_path, man)
    return man


def load_processed(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    from utils import read_jsonl

    return {
        "items": pd.read_parquet(out_dir / "items.parquet"),
        "queries": read_jsonl(out_dir / "queries.jsonl"),
        "labels": pd.read_parquet(out_dir / "labels.parquet"),
        "groups": pd.read_parquet(out_dir / "user_groups.parquet"),
        "direction_pairs": pd.read_parquet(out_dir / "direction_pairs.parquet"),
        "probe_val_pairs": pd.read_parquet(out_dir / "probe_val_pairs.parquet"),
        "item_prior": pd.read_parquet(out_dir / "item_prior.parquet"),
        "audit": read_json(out_dir / "data_audit.json"),
        "manifest": read_json(out_dir / "manifest.json"),
    }
