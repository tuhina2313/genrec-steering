"""Temporal split integrity, candidate construction, leakage guards and prompt-label invariance."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data as data_mod
import prompts
from conftest import PROJECT, make_root
from run import load_config


def test_cutoffs_never_split_ties():
    ts = np.array([1, 1, 2, 2, 2, 3, 4, 5, 5, 5])
    cut = data_mod.compute_cutoffs(ts, [0.3, 0.6, 0.9])
    for c in cut:
        v = c["timestamp"]
        assert (ts < v).sum() == c["events_before"]        # every event with t == v falls after the cutoff
        assert v in ts or v > ts.max()
    assert cut[0]["timestamp"] == 3 and cut[0]["moved_to_next_distinct"]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = make_root(tmp_path_factory.mktemp("data"))
    cfg = load_config("movielens_1m", "smoke", [], root)
    out = root / "data" / "processed" / "x"
    man = data_mod.prepare_dataset(cfg, "smoke", root / "data" / "raw" / "movielens_1m", out)
    return cfg, root, out, man, data_mod.load_processed(out)


def test_history_strictly_before_cutoff_and_candidates_unseen(prepared):
    cfg, _, _, man, d = prepared
    cut = [c["timestamp"] for c in man["split_cutoffs"]]
    start = {"fit": cut[0], "validation": cut[1], "test": cut[2]}
    for q in d["queries"]:
        assert q["cutoff_timestamp"] == start[q["split"]]
        assert all(h["timestamp"] < q["cutoff_timestamp"] for h in q["history"])
        assert cfg["history"]["min_items"] <= len(q["history"]) <= cfg["history"]["max_items"]
        if q["split"] != "fit":
            assert len(q["candidate_ids"]) >= cfg["candidates"]["min_count"]
            assert len(q["candidate_ids"]) <= cfg["profiles"]["smoke"]["max_candidates"]
    lab = d["labels"]
    qmap = {q["query_id"]: q for q in d["queries"]}
    for r in lab.itertuples():
        q = qmap[r.query_id]
        assert r.observed_timestamp >= q["cutoff_timestamp"]
        assert r.item_id in q["candidate_ids"]


def test_user_groups_and_direction_pairs(prepared):
    cfg, _, _, _, d = prepared
    g = d["groups"].set_index("user_id")["user_group"]
    for q in d["queries"]:
        if q["split"] in ("fit", "validation"):
            assert q["user_group"] != "unseen_by_direction"
    dp, pv = d["direction_pairs"], d["probe_val_pairs"]
    assert set(g.loc[dp["user_id"].unique()]) == {"inner_fit"}
    assert set(g.loc[pv["user_id"].unique()]) == {"probe_val"}
    assert all(p.split(":")[2] == "fit" for p in dp["query_id"])
    for (obj, s), f in dp.groupby(["objective", "direction_seed"]):
        assert (f["label"] == 1).sum() == (f["label"] == 0).sum()
        assert f.groupby(["user_id", "label"]).size().max() <= cfg["directions"]["per_user_class_cap"]
    sat = d["labels"].set_index("pair_id").loc[dp[dp["objective"] == "satisfaction"]["pair_id"], "rating"]
    assert not (sat == 3).any()


def test_changing_candidate_label_does_not_change_prompt(tmp_path, tiny_model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    root = make_root(tmp_path)
    cfg = load_config("movielens_1m", "smoke", ["model.max_prompt_tokens=4096"], root)
    raw = root / "data" / "raw" / "movielens_1m"

    def render(out):
        data_mod.prepare_dataset(cfg, "smoke", raw, out)
        d = data_mod.load_processed(out)
        items = {r["item_id"]: r for r in d["items"].to_dict("records")}
        q = [q for q in d["queries"] if q["split"] == "validation"][0]
        it = q["candidate_ids"][0]
        r = prompts.render_pair(tok, q, prompts.make_candidate(items[it]), cfg, ("Yes", "No"))
        row = d["labels"].set_index("pair_id").loc[f"{q['query_id']}#{it}"]
        return q["user_id"], it, r["input_ids"], int(row["rating"]), int(row["observed_timestamp"])

    uid, it, ids1, rating, ts = render(root / "p1")
    rp = raw / "ml-1m" / "ratings.dat"
    lines = rp.read_text(encoding="latin-1").splitlines()
    new_rating = 1 if rating != 1 else 5
    changed = 0
    for i, ln in enumerate(lines):
        parts = ln.split("::")
        if len(parts) == 4 and parts[0] == uid and parts[1] == it and int(parts[3]) == ts:
            lines[i] = f"{uid}::{it}::{new_rating}::{ts}"
            changed += 1
    assert changed == 1
    rp.write_text("\n".join(lines) + "\n", encoding="latin-1")
    (raw / "ml-1m.zip").unlink()
    uid2, it2, ids2, rating2, _ = render(root / "p2")
    assert (uid2, it2) == (uid, it) and rating2 == new_rating
    assert ids1 == ids2


def test_candidate_cannot_carry_outcome():
    from contracts import Candidate
    with pytest.raises(ValueError):
        Candidate(item_id="1", display_text="x", metadata={"rating": 5})
