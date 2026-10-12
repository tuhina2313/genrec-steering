"""Versioned shared prompt renderer, dataset field formatting, deterministic truncation.

The renderer receives a query (history with already-observed feedback) and a Candidate
(metadata only). It never receives label rows, so current-candidate outcomes cannot leak.
Opaque user/item IDs are never rendered.
"""
from __future__ import annotations

from typing import Any, Optional

from contracts import Candidate
from utils import sha256_bytes, stable_hash

PROMPT_VERSION = "rec_yesno_v1"

SYSTEM_INSTRUCTION = (
    "You are a careful recommendation assistant. Using only the user's past feedback shown, judge whether "
    "the user will like the candidate item. Answer with exactly one word."
)

QUESTION = {
    ("Yes", "No"): "Will this user like the candidate {noun}? Answer Yes or No.",
    ("A", "B"): "Will this user like the candidate {noun}? Answer A for yes or B for no.",
}

VARIANTS = ("base", "objective")   # objective prompts are separately versioned baselines


def template_identity(cfg: dict, variant: str = "base", objective: Optional[str] = None) -> dict:
    """Everything that defines the prompt text; hashed into cache keys."""
    v = cfg["scoring"]["verbalizer"]
    ident = {"prompt_version": PROMPT_VERSION, "system": SYSTEM_INSTRUCTION, "questions": {"|".join(k): q for k, q in QUESTION.items()},
             "feedback_convention": cfg["dataset"].get("feedback_convention"), "noun": cfg["dataset"].get("item_noun"),
             "variant": variant, "verbalizer": v, "max_prompt_tokens": cfg["model"]["max_prompt_tokens"]}
    if variant == "objective":
        ident["objective"] = objective
        ident["objective_instruction"] = cfg["objectives"][objective]["prompt_instruction"]
    return ident


def prompt_template_hash(cfg: dict, variant: str = "base", objective: Optional[str] = None) -> str:
    return stable_hash(template_identity(cfg, variant, objective))


# ------------------------------------------------------------------ item formatting

def format_item(meta: dict, missing: dict) -> str:
    """Dataset field formatting driven by available metadata (title, year, genres)."""
    parts = [str(meta.get("title", "")).strip()]
    if meta.get("year") is not None and not missing.get("year", False):
        parts[0] = f"{parts[0]} ({meta['year']})"
    genres = meta.get("genres") or []
    parts.append("Genres: " + (", ".join(genres) if genres and not missing.get("genres", False) else "unknown"))
    return " | ".join(parts)


def format_feedback(feedback: dict) -> str:
    if "rating" in feedback:
        return f"Rating: {int(feedback['rating'])}/5"
    return "; ".join(f"{k}: {v}" for k, v in sorted(feedback.items()))


def make_candidate(item_row: dict) -> Candidate:
    meta = {"title": item_row["title"], "year": None if item_row.get("year") is None or item_row["year"] != item_row["year"]
            else int(item_row["year"]), "genres": list(item_row["genres"])}
    missing = {"year": bool(item_row.get("metadata_missing_year", False)),
               "genres": bool(item_row.get("metadata_missing_genres", False))}
    return Candidate(item_id=str(item_row["item_id"]), display_text=format_item(meta, missing), metadata=meta,
                     metadata_missing=missing)


# ------------------------------------------------------------------ messages

def build_messages(history: list[dict], candidate: Candidate, cfg: dict, verbalizer: tuple[str, str],
                   variant: str = "base", objective: Optional[str] = None) -> list[dict]:
    ds = cfg["dataset"]
    lines = [ds["feedback_convention"].strip(), "", "User history (oldest to newest):"]
    if history:
        for n, h in enumerate(history, 1):
            lines.append(f"{n}. {format_item(h['metadata'], h.get('metadata_missing', {}))} | "
                         f"{format_feedback(h['feedback'])}")
    else:
        lines.append("(no history shown)")
    lines += ["", f"Candidate {ds['item_noun']}:", candidate.display_text, ""]
    if variant == "objective":
        lines.append(cfg["objectives"][objective]["prompt_instruction"].strip())
    elif variant != "base":
        raise ValueError(variant)
    lines.append(QUESTION[tuple(verbalizer)].format(noun=ds["item_noun"]))
    return [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": "\n".join(lines)}]


def render_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def render_pair(tokenizer, query: dict, candidate: Candidate, cfg: dict, verbalizer: tuple[str, str],
                variant: str = "base", objective: Optional[str] = None) -> dict[str, Any]:
    """Render one (history, candidate) pair; drop oldest history items until it fits."""
    max_tokens = cfg["model"]["max_prompt_tokens"]
    history = list(query["history"])
    n_total = len(history)
    while True:
        text = render_text(tokenizer, build_messages(history, candidate, cfg, verbalizer, variant, objective))
        ids = encode(tokenizer, text)
        if len(ids) <= max_tokens or not history:
            break
        history = history[1:]   # truncate oldest first; candidate and response cue are preserved
    if len(ids) > max_tokens:
        raise ValueError(f"prompt for {query['query_id']}#{candidate.item_id} exceeds {max_tokens} tokens "
                         "even with empty history")
    return {"input_ids": ids, "token_len": len(ids), "n_history_total": n_total, "n_history_used": len(history),
            "n_history_truncated": n_total - len(history),
            "prompt_hash": sha256_bytes((PROMPT_VERSION + "|" + ",".join(map(str, ids))).encode())}
