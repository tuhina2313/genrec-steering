"""Raw-coordinate and standardized probe scores agree; failed fits are explicit."""
import numpy as np
import pytest

import directions as D


def data(seed=0, n=200, d=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * rng.uniform(0.1, 20, size=d) + rng.normal(size=d) * 5
    X[:, 3] = 7.0                                  # constant dimension: safe scale handling
    w = rng.normal(size=d)
    y = (X @ w + rng.normal(size=n) * 3 > np.median(X @ w)).astype(int)
    return X, y


def test_raw_vs_standardized_parity():
    X, y = data()
    probe, grid = D.fit_probe(X[:150], y[:150], X[150:], y[150:], [0.01, 0.1, 1, 10], seed=0)
    from sklearn.preprocessing import StandardScaler
    z = (X - probe.mean) / probe.scale
    np.testing.assert_allclose(z @ probe.coef_std + probe.intercept_std, probe.decision(X), rtol=1e-7, atol=1e-7)
    assert sum(g["selected"] for g in grid) == 1 and probe.classes == [0, 1]


def test_directions_unit_and_same_pairs():
    X, y = data(1)
    res = D.fit_directions(X[:150], y[:150], X[150:], y[150:], [0.1, 1], seed=0)
    for k in ("mean_diff", "probe"):
        assert abs(np.linalg.norm(res["mean_diff"]) - 1) < 1e-5 and res[k].dtype == np.float32
    assert res["support"] == {"fit_pos": int(y[:150].sum()), "fit_neg": int((y[:150] == 0).sum())}
    assert res["radius"] == pytest.approx(np.median(np.linalg.norm(X[:150], axis=1)))


def test_failed_fit_and_not_estimable():
    X, y = data(2)
    with pytest.raises(ValueError):
        D.unit(np.zeros(4))
    with pytest.raises(ValueError):
        D.mean_difference(X, np.ones(len(X)))
    diag = D.binary_diagnostics(np.ones(10), np.arange(10))
    assert diag["auroc"] is None and diag["not_estimable_reason"]
