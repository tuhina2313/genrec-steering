"""Scorer correctness on a tiny Qwen2-architecture model: LM-head parity, hooks, controls, invariance."""
import numpy as np
import pytest
import torch

import prompts
from contracts import NO_INTERVENTION, Intervention
from steering import count_hooks, resolve_layers


def seqs(scorer, n=6):
    out = []
    for i in range(n):
        msgs = [{"role": "system", "content": prompts.SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"Night River {i} (19{50 + i}) | Genres: Comedy | Rating: {i % 5 + 1}/5\n"
                                            + "Blue Star " * (i * 3) + "Will this user like the candidate movie? Answer Yes or No."}]
        out.append(prompts.encode(scorer.tokenizer, prompts.render_text(scorer.tokenizer, msgs)))
    return out


def pid(n):
    return [f"q{i % 2}#item{i}" for i in range(n)]


def test_layer_resolution():
    assert resolve_layers([0.25, 0.5, 0.75, 1.0], 36) == [8, 17, 26, 35]
    assert resolve_layers([0.5, 1.0], 4) == [1, 3]
    assert resolve_layers([0.5, 0.5, 1.0], 4) == [1, 3]


def test_verbalizer_single_tokens(tiny_scorer):
    assert tiny_scorer.verbalizer == ("Yes", "No")
    a, b = tiny_scorer.verbalizer_ids
    assert a != b


def test_two_row_lm_head_matches_full_forward(tiny_scorer):
    r = tiny_scorer.lm_head_parity(seqs(tiny_scorer, 4))
    assert r["passed"], r


def test_strength_zero_identity_and_hook_removal(tiny_scorer):
    s, S = tiny_scorer, seqs(tiny_scorer)
    before = count_hooks(s.model)
    base = s.score(pid(len(S)), S)
    zero = Intervention(kind="add_block", layer_index=1, direction_id="z", vector_sha="z", strength=0.0, radius=1.0)
    z = s.score(pid(len(S)), S, intervention=zero, delta=np.zeros(s.hidden_size, np.float32))
    np.testing.assert_array_equal(z.scores, base.scores)
    big = Intervention(kind="add_block", layer_index=1, direction_id="r", vector_sha="r", strength=1.0, radius=5.0)
    rng = np.random.default_rng(0)
    d = rng.normal(size=s.hidden_size).astype(np.float32) * 5
    st = s.score(pid(len(S)), S, intervention=big, delta=d)
    assert np.max(np.abs(st.scores - base.scores)) > 1e-4                 # a real edit changes scores
    assert count_hooks(s.model) == before
    after = s.score(pid(len(S)), S)
    np.testing.assert_array_equal(after.scores, base.scores)              # later unsteered pass unchanged


def test_hooks_removed_on_exception(tiny_scorer):
    s = tiny_scorer
    before = count_hooks(s.model)
    bad = Intervention(kind="add_block", layer_index=0, direction_id="x", vector_sha="x", strength=1.0, radius=1.0)
    with pytest.raises(Exception):
        s.score(["a#1"], [[10**9]], intervention=bad, delta=np.zeros(s.hidden_size, np.float32))
    assert count_hooks(s.model) == before


def test_postnorm_constant_vector_preserves_rankings(tiny_scorer):
    s, S = tiny_scorer, seqs(tiny_scorer)
    base = s.score(pid(len(S)), S)
    rng = np.random.default_rng(1)
    d = rng.normal(size=s.hidden_size).astype(np.float32)
    iv = Intervention(kind="add_postnorm", direction_id="pn", vector_sha="pn", strength=1.0, radius=1.0)
    pn = s.score(pid(len(S)), S, intervention=iv, delta=d)
    shift = pn.scores - base.scores
    assert np.ptp(shift) < 1e-4 and abs(shift.mean()) > 1e-3             # same offset for every candidate
    assert list(np.argsort(-pn.scores)) == list(np.argsort(-base.scores))


def test_batch_size_and_order_invariance(tiny_scorer):
    s, S = tiny_scorer, seqs(tiny_scorer)
    ids = pid(len(S))
    s.microbatch = 4
    a = dict(zip(ids, s.score(ids, S, capture_layers=(1,)).scores))
    s.microbatch = 1
    rev = list(reversed(range(len(S))))
    b_res = s.score([ids[i] for i in rev], [S[i] for i in rev], capture_layers=(1,))
    s.microbatch = 4
    b = dict(zip(b_res.pair_ids, b_res.scores))
    for k in ids:
        assert a[k] == pytest.approx(b[k], abs=1e-4)
    assert b_res.states[1].shape == (len(S), s.hidden_size)


def test_capture_uses_last_valid_position_not_padding(tiny_scorer):
    s, S = tiny_scorer, seqs(tiny_scorer, 3)
    s.microbatch = 3
    together = s.score(pid(3), S, capture_layers=(0,)).states[0]
    alone = np.stack([s.score([pid(3)[i]], [S[i]], capture_layers=(0,)).states[0][0] for i in range(3)])
    np.testing.assert_allclose(together, alone, atol=1e-4)
