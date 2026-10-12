"""Frozen model loading, verbalizer validation, batched pair scoring, selected activation capture.

score(pair) = logit(positive verbalizer) - logit(negative verbalizer) at the response position,
computed with a memory-efficient two-row LM-head adapter on the final normalized hidden state.
The adapter's parity with the standard full forward is validated on real pairs (`lm_head_parity`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from contracts import NO_INTERVENTION, Intervention
from steering import PositionRef, capture_hooks, count_hooks, intervention_hooks
from utils import PrerequisiteMissing, sha256_file

SCORER_VERSION = "yesno_logit_diff_v1"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32}[name]


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_revision(name: str, revision: Optional[str], local_files_only: bool) -> tuple[str, Optional[str]]:
    """Return (revision string to record, revision to pass to from_pretrained)."""
    p = Path(name)
    if p.exists():
        files = sorted(x for x in p.iterdir() if x.is_file())
        from utils import stable_hash

        return "local-sha256:" + stable_hash({f.name: sha256_file(f) for f in files})[:40], None
    if revision and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision, revision
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(name, revision=revision or "main").sha
        return sha, sha
    except Exception as e:
        if local_files_only:
            try:
                from huggingface_hub import snapshot_download

                path = snapshot_download(name, revision=revision, local_files_only=True)
                sha = Path(path).name
                return sha, sha
            except Exception:
                pass
        raise PrerequisiteMissing(f"model {name}@{revision or 'main'}",
                                  f"could not resolve an immutable revision ({type(e).__name__}: {e})")


@dataclass
class ScoringResult:
    pair_ids: list[str]
    scores: np.ndarray
    logit_pos: np.ndarray
    logit_neg: np.ndarray
    states: dict[int, np.ndarray] = field(default_factory=dict)   # layer -> (n, d) aligned with pair_ids
    seconds: float = 0.0
    effective_microbatch: int = 0


class FrozenScorer:
    def __init__(self, model, tokenizer, cfg: dict, model_revision: str, tokenizer_revision: str,
                 device: str, dtype: torch.dtype):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.device = device
        self.dtype = dtype
        self.microbatch = int(cfg["model"]["microbatch"])
        self.min_microbatch = int(cfg["model"]["min_microbatch"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.backbone, self.blocks, self.final_norm, self.lm_head = self._resolve_architecture(model)
        self.num_layers = len(self.blocks)
        self.hidden_size = int(model.config.hidden_size)
        self.verbalizer, self.verbalizer_ids = self.validate_verbalizer()
        W = self.lm_head.weight[list(self.verbalizer_ids)].detach().float()
        b = self.lm_head.bias[list(self.verbalizer_ids)].detach().float() if self.lm_head.bias is not None else None
        self.W2, self.b2 = W, b
        self.softcap = getattr(model.config, "final_logit_softcapping", None)
        self.peak_memory_bytes = 0

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, cfg: dict) -> "FrozenScorer":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        m = cfg["model"]
        device = resolve_device(m["device"])
        dtype = resolve_dtype(m["dtype"], device)
        lfo = bool(m.get("local_files_only", False))
        rev_rec, rev = resolve_revision(m["name"], m.get("revision"), lfo)
        if m.get("tokenizer_revision"):
            tok_rec, tok_rev = resolve_revision(m["name"], m["tokenizer_revision"], lfo)
        else:
            tok_rec, tok_rev = rev_rec, rev
        try:
            tok = AutoTokenizer.from_pretrained(m["name"], revision=tok_rev, local_files_only=lfo)
            model = AutoModelForCausalLM.from_pretrained(
                m["name"], revision=rev, dtype=dtype, attn_implementation=m.get("attn_implementation") or None,
                local_files_only=lfo)
        except OSError as e:
            raise PrerequisiteMissing(f"model {m['name']}@{rev_rec}", str(e))
        model.to(device)
        return cls(model, tok, cfg, rev_rec, tok_rec, device, dtype)

    @staticmethod
    def _resolve_architecture(model):
        """Decoder blocks, final norm and LM head for Llama/Qwen2/Mistral-style CausalLMs."""
        backbone = getattr(model, "model", None)
        blocks = getattr(backbone, "layers", None)
        norm = getattr(backbone, "norm", None)
        head = getattr(model, "lm_head", None)
        if blocks is None or norm is None or head is None:
            raise NotImplementedError(f"unsupported architecture {type(model).__name__}: add an adapter with parity checks")
        return backbone, blocks, norm, head

    # ------------------------------------------------------------ verbalizer
    def validate_verbalizer(self) -> tuple[tuple[str, str], tuple[int, int]]:
        """Yes/No must each be exactly one token at the actual response position; else fall back to A/B."""
        from prompts import render_text

        v = self.cfg["scoring"]["verbalizer"]
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        prefix = self.tokenizer(render_text(self.tokenizer, msgs), add_special_tokens=False)["input_ids"]
        for pair in ((v["positive"], v["negative"]), (v["fallback_positive"], v["fallback_negative"])):
            ids = []
            for word in pair:
                full = self.tokenizer(render_text(self.tokenizer, msgs) + word, add_special_tokens=False)["input_ids"]
                if list(full[:len(prefix)]) != list(prefix) or len(full) - len(prefix) != 1:
                    break
                ids.append(int(full[-1]))
            if len(ids) == 2 and ids[0] != ids[1]:
                return pair, (ids[0], ids[1])
        raise RuntimeError("neither Yes/No nor A/B are single tokens at the response position")

    # ------------------------------------------------------------ scoring
    def _batches(self, order: list[int], lengths: list[int], mb: int):
        for i in range(0, len(order), mb):
            yield order[i:i + mb]

    @torch.inference_mode()
    def _forward_batch(self, seqs: list[list[int]], capture_layers: Sequence[int], intervention: Intervention,
                       delta: Optional[np.ndarray]):
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[r, :len(s)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        # right padding: the last valid prompt token is at (number of valid tokens - 1)
        pos = mask.sum(dim=1) - 1
        assert bool((mask.gather(1, pos[:, None]) == 1).all()), "prediction position selected a padding token"
        ref = PositionRef()
        ref.positions = pos
        store = {li: [] for li in capture_layers}
        if intervention.kind == "add_block":
            target = self.blocks[intervention.layer_index]
        elif intervention.kind == "add_postnorm":
            target = self.final_norm
        else:
            target = None
        with capture_hooks({li: self.blocks[li] for li in capture_layers}, ref, store), \
                intervention_hooks(target, delta, ref):
            out = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False)
        hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = hs[torch.arange(hs.shape[0], device=hs.device), pos].float()
        logits = h @ self.W2.to(h.device).T
        if self.b2 is not None:
            logits = logits + self.b2.to(h.device)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        logits = logits.cpu().numpy()
        states = {li: torch.cat(v).numpy() for li, v in store.items()}
        return logits, states

    def score(self, pair_ids: list[str], seqs: list[list[int]], capture_layers: Sequence[int] = (),
              intervention: Intervention = NO_INTERVENTION, delta: Optional[np.ndarray] = None,
              on_progress: Optional[Callable[[int], None]] = None) -> ScoringResult:
        """Score pairs; identity is carried by pair_ids regardless of the internal length-sorted order."""
        n = len(pair_ids)
        assert n == len(seqs) and len(set(pair_ids)) == n, "pair ids must be unique and aligned with sequences"
        if intervention.kind != "none":
            assert delta is not None and delta.shape == (self.hidden_size,)
        hooks_before = count_hooks(self.model)
        lengths = [len(s) for s in seqs]
        order = sorted(range(n), key=lambda i: (lengths[i], pair_ids[i]))
        logits = np.zeros((n, 2), dtype=np.float64)
        states = {li: np.zeros((n, self.hidden_size), dtype=np.float32) for li in capture_layers}
        t0 = time.time()
        i = 0
        mb = self.microbatch
        while i < n:
            idx = order[i:i + mb]
            try:
                lg, st = self._forward_batch([seqs[j] for j in idx], capture_layers, intervention, delta)
            except torch.OutOfMemoryError:
                if mb <= self.min_microbatch:
                    raise
                mb = max(self.min_microbatch, mb // 2)
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            logits[idx] = lg
            for li in capture_layers:
                states[li][idx] = st[li]
            i += len(idx)
            if on_progress:
                on_progress(len(idx))
        self.microbatch = mb
        if self.device.startswith("cuda"):
            self.peak_memory_bytes = max(self.peak_memory_bytes, torch.cuda.max_memory_allocated())
        assert count_hooks(self.model) == hooks_before, "intervention or capture hooks leaked"
        return ScoringResult(pair_ids=list(pair_ids), scores=logits[:, 0] - logits[:, 1], logit_pos=logits[:, 0],
                             logit_neg=logits[:, 1], states=states, seconds=time.time() - t0, effective_microbatch=mb)

    # ------------------------------------------------------------ parity
    @torch.inference_mode()
    def lm_head_parity(self, seqs: list[list[int]]) -> dict:
        """Two-row adapter vs the standard full-model forward logits on the same prompts (one at a time)."""
        diffs = []
        for s in seqs:
            ids = torch.tensor([s], dtype=torch.long, device=self.device)
            full = self.model(input_ids=ids, use_cache=False).logits[0, -1].float()
            ref = full[list(self.verbalizer_ids)].cpu().numpy()
            lg, _ = self._forward_batch([s], (), NO_INTERVENTION, None)
            diffs.append(float(np.max(np.abs(lg[0] - ref))))
        tol = self.cfg["scoring"]["lm_head_parity_atol"][str(self.dtype).replace("torch.", "")]
        return {"max_abs_diff": max(diffs) if diffs else None, "atol": tol, "n": len(diffs),
                "passed": bool(diffs) and max(diffs) <= tol}

    def identity(self) -> dict:
        return {"model_revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision,
                "dtype": str(self.dtype), "attn_implementation": self.cfg["model"].get("attn_implementation"),
                "verbalizer": list(self.verbalizer), "verbalizer_ids": list(self.verbalizer_ids),
                "scorer_version": SCORER_VERSION, "hook_location": self.cfg["scoring"]["hook_location"],
                "prediction_position": self.cfg["scoring"]["prediction_position"], "model_name": self.cfg["model"]["name"]}
