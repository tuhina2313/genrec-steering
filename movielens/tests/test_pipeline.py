"""End-to-end smoke on synthetic ML-1M-format data with a tiny local model: stage order, identical
candidate sets across methods, leakage guards, resume equivalence and cache invalidation."""
import json

import numpy as np
import pandas as pd
import pytest

import run
from conftest import make_root, overrides


def ctx_for(root, model_dir, run_id="smoke", extra=(), cap=None):
    return run.Ctx("movielens_1m", "smoke", run_id, overrides(model_dir) + list(extra), root, cap)


FORCE = ["profiles.smoke.guardrail_budgets=[1.0]", "profiles.smoke.strengths=[0.5]",
         "profiles.smoke.layer_fractions=[0.5, 1.0]"]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory, tiny_model_dir):
    root = make_root(tmp_path_factory.mktemp("e2e"))
    ctx = ctx_for(root, tiny_model_dir, extra=FORCE)
    assert run.run_through(ctx, "report", resume=False) == "completed"
    return root, ctx


def test_stage_order_enforced(tmp_path, tiny_model_dir):
    root = make_root(tmp_path)
    ctx = ctx_for(root, tiny_model_dir)
    assert run.run_stage(ctx, "sweep") == "blocked"
    st = json.loads((ctx.out_dir / "stage_status.json").read_text())["stages"]["sweep"]
    assert st["missing"] == "stage:fit-directions"


def test_outputs_and_identical_candidate_sets(full_run):
    root, ctx = full_run
    out = ctx.out_dir
    final = json.loads((out / "final_results.json").read_text())      # strict JSON: no NaN
    for k in ("schema_version", "run_id", "dataset", "protocol_id", "profile", "model_revision", "backbone_trained",
              "completed", "realized_counts", "candidate_source", "split_cutoffs", "selection_hash", "metrics",
              "confidence_intervals", "timing", "warnings", "artifact_paths"):
        assert k in final
    assert final["backbone_trained"] is False and final["completed"] is True
    pq = pd.read_parquet(out / "per_query_metrics.parquet")
    ref = pq[pq["selection_id"] == "unsteered"].drop_duplicates("query_id").set_index("query_id")["n_candidates"]
    for sid, g in pq.groupby(["selection_id", "objective"]):
        assert g.set_index("query_id")["n_candidates"].sort_index().equals(ref.sort_index()), sid
    sc = pd.read_parquet(out / "scores_test.parquet")
    assert not sc.duplicated(["selection_id", "pair_id"]).any()
    methods = set(pq["method"])
    assert methods & {"mean_diff_steer", "probe_steer"}, "forced budget should select a steering configuration"
    assert {"mean_item_score_shift"} <= methods
    for f in ("config.resolved.yaml", "environment.json", "data_audit.json", "baseline_summary.json", "probes.csv",
              "directions/index.json", "validation_sweep.csv", "selected.json", "budget.json", "results.md",
              "final_results.csv", "figures/validation_tradeoff.png"):
        assert (out / f).exists(), f
    checks = json.loads((out / "sweep_checks.json").read_text())
    assert all(c["passed"] for c in checks["identity"])
    assert checks["postnorm_null"]["queries_with_changed_ranking"] == 0


def test_fitting_ignores_validation_and_test_labels(full_run):
    root, ctx = full_run
    before = {p.name: np.load(p)["vector"] for p in (ctx.out_dir / "directions").glob("*.npz") if ".model" not in p.name}
    lab_path = ctx.proc_dir / "labels.parquet"
    orig = lab_path.read_bytes()
    lab = pd.read_parquet(lab_path)
    m = lab["split"].isin(["validation", "test"])
    lab.loc[m, "rating"] = 6 - lab.loc[m, "rating"]
    lab.loc[m, "satisfaction_label"] = 1 - lab.loc[m, "satisfaction_label"]
    lab.loc[m, "fit_class__satisfaction"] = 1 - lab.loc[m, "fit_class__satisfaction"]
    lab.to_parquet(lab_path, index=False)
    try:
        c2 = ctx_for(root, ctx.cfg["model"]["name"], extra=FORCE)
        run.stage_fit_directions(c2)
        after = {p.name: np.load(p)["vector"] for p in (c2.out_dir / "directions").glob("*.npz") if ".model" not in p.name}
        assert before.keys() == after.keys()
        for k in before:
            np.testing.assert_array_equal(before[k], after[k])
    finally:
        lab_path.write_bytes(orig)


def test_resume_reproduces_uninterrupted_run(full_run, tiny_model_dir):
    root, ctx = full_run
    c = ctx_for(root, tiny_model_dir, run_id="interrupted", extra=FORCE, cap=100)
    assert run.run_through(c, "report", resume=True) == "partial"
    assert json.loads((c.out_dir / "stage_status.json").read_text())["stages"]["baseline"]["status"] == "partial"
    c2 = ctx_for(root, tiny_model_dir, run_id="interrupted", extra=FORCE)
    assert run.run_through(c2, "report", resume=True) == "completed"
    for f in ("scores_test.parquet",):
        a = pd.read_parquet(ctx.out_dir / f).sort_values(["selection_id", "pair_id"]).reset_index(drop=True)
        b = pd.read_parquet(c2.out_dir / f).sort_values(["selection_id", "pair_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, atol=1e-5)
    va = pd.read_csv(ctx.out_dir / "validation_sweep.csv")
    vb = pd.read_csv(c2.out_dir / "validation_sweep.csv")
    pd.testing.assert_frame_equal(va, vb, atol=1e-6)


def test_prompt_change_invalidates_cache_and_selection_is_locked(full_run, tiny_model_dir):
    root, ctx = full_run
    changed = FORCE + ["dataset.feedback_convention=Ratings use a 1-5 scale."]
    c = ctx_for(root, tiny_model_dir, extra=changed)
    k_old, _ = ctx.job_key("base", None, run.NO_INTERVENTION)
    k_new, _ = c.job_key("base", None, run.NO_INTERVENTION)
    assert k_old != k_new
    assert run.run_stage(c, "sweep", resume=True) == "blocked"          # upstream fingerprint changed
    for st in ("prepare", "baseline"):
        assert run.run_stage(c, st, resume=True) == "completed"
    assert c.forward_pairs > 0                                           # baseline was recomputed, not reused
    for st in ("fit-directions", "sweep"):
        assert run.run_stage(c, st, resume=True) == "completed"
    assert run.run_stage(c, "select", resume=True) == "failed"           # selections are immutable
    err = json.loads((c.out_dir / "stage_status.json").read_text())["stages"]["select"]["error"]
    assert "locked" in err
