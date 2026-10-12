"""Metrics agree with hand-computed toy queries, including zero-positive and short-candidate cases."""
import math

import numpy as np
import pandas as pd
import pytest

import metrics as met

SPECS = [
    {"name": "ndcg_like@5", "type": "ndcg", "k": 5, "gain": "satisfaction_label"},
    {"name": "ndcg_rating@5", "type": "ndcg", "k": 5, "gain": "graded_gain"},
    {"name": "precision_like@1", "type": "precision", "k": 1, "gain": "satisfaction_label"},
    {"name": "precision_like@3", "type": "precision", "k": 3, "gain": "satisfaction_label"},
    {"name": "precision_like@10", "type": "precision", "k": 10, "gain": "satisfaction_label"},
    {"name": "recall_like@5", "type": "recall", "k": 5, "gain": "satisfaction_label"},
    {"name": "mrr_like", "type": "mrr", "gain": "satisfaction_label"},
    {"name": "mean_rating@5", "type": "mean_at_k", "k": 5, "column": "rating"},
]


def toy():
    items = list("abcde")
    lab = pd.DataFrame({"query_id": "q1", "item_id": items, "user_id": "u1",
                        "satisfaction_label": [0, 1, 0, 1, 0], "graded_gain": [1.0, 3.0, 0.0, 4.0, 2.0],
                        "rating": [2, 4, 1, 5, 3]})
    zero = pd.DataFrame({"query_id": "q2", "item_id": list("xyz"), "user_id": "u2",
                         "satisfaction_label": [0, 0, 0], "graded_gain": [0.0, 0.0, 0.0], "rating": [1, 1, 1]})
    labels = pd.concat([lab, zero], ignore_index=True)
    scores = pd.DataFrame({"query_id": ["q1"] * 5 + ["q2"] * 3, "item_id": items + list("xyz"),
                           "score": [5, 4, 3, 2, 1, 0.5, 0.5, 0.5]})
    return scores, labels


def test_hand_computed_query():
    scores, labels = toy()
    pq = met.query_metrics(scores, labels, SPECS).set_index("query_id")
    q = pq.loc["q1"]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert q["ndcg_like@5"] == pytest.approx(dcg / (1 + 1 / math.log2(3)))
    g = [1, 3, 0, 4, 2]
    dcg_r = sum(x / math.log2(i + 2) for i, x in enumerate(g))
    idcg_r = sum(x / math.log2(i + 2) for i, x in enumerate(sorted(g, reverse=True)))
    assert q["ndcg_rating@5"] == pytest.approx(dcg_r / idcg_r)       # linear gains, not exponentiated
    assert q["precision_like@1"] == 0 and q["precision_like@3"] == pytest.approx(1 / 3)
    assert q["precision_like@10"] == pytest.approx(2 / 5)            # short list: denominator min(k, n)
    assert q["recall_like@5"] == 1.0 and q["mrr_like"] == 0.5
    assert q["mean_rating@5"] == pytest.approx(3.0)


def test_zero_positive_and_conditional_aggregation():
    scores, labels = toy()
    pq = met.query_metrics(scores, labels, SPECS)
    z = pq.set_index("query_id").loc["q2"]
    assert z["ndcg_like@5"] == 0 and not z["ndcg_like@5__defined"]
    assert z["recall_like@5"] == 0 and not z["recall_like@5__defined"]
    agg = met.aggregate(pq, SPECS)
    assert agg["ndcg_like@5"] == pytest.approx(pq["ndcg_like@5"].mean())
    assert agg["ndcg_like@5__conditional"] == pytest.approx(pq.set_index("query_id").loc["q1", "ndcg_like@5"])
    assert agg["ndcg_like@5__zero_denominator_rate"] == 0.5


def test_tie_break_by_item_id():
    df = pd.DataFrame({"query_id": "q", "item_id": ["c", "a", "b"], "score": [1.0, 1.0, 1.0]})
    r = met.rank_scores(df).sort_values("rank")
    assert list(r["item_id"]) == ["a", "b", "c"]


def test_bootstrap_identical_methods_zero_delta():
    scores, labels = toy()
    pq = met.query_metrics(scores, labels, SPECS)
    res = met.paired_bootstrap(pq, pq.copy(), "ndcg_like@5", 200, 0)
    assert res["abs_delta"] == 0 and res["ci_lower"] == 0 and res["ci_upper"] == 0
    with pytest.raises(AssertionError):
        met.paired_bootstrap(pq, pq.iloc[:1], "ndcg_like@5", 10, 0)   # denominators must match


def test_selection_rule():
    base = {"target": 0.5, "guardrail": 0.8}
    cands = [{"config_id": "a", "target": 0.6, "guardrail": 0.70, "strength": 0.1, "layer": 1},   # infeasible
             {"config_id": "b", "target": 0.55, "guardrail": 0.795, "strength": 0.03, "layer": 1},
             {"config_id": "c", "target": 0.55, "guardrail": 0.795, "strength": 0.01, "layer": 2}]
    r = met.select_operating_point(cands, base, 0.01)
    assert r["config_id"] == "c"                       # tie -> smaller |strength|
    r0 = met.select_operating_point(cands, base, 0.0)
    assert r0["config_id"] == "baseline" and r0["is_baseline"]
    assert met.select_operating_point(cands, {"target": 0.5, "guardrail": 0.0}, 0.01)["status"] == "not_estimable"
