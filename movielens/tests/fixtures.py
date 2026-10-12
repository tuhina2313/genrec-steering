"""Small offline fixtures: an ML-1M-format synthetic dataset and a tiny Qwen2-architecture model.

These exercise the real code paths (parsing, splits, chat template, hooks, LM head) without network
access. They are correctness fixtures only; nothing measured on them is an experimental result.
"""
from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import numpy as np

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
WORDS = ["Night", "River", "Last", "Summer", "City", "Ghost", "Love", "Star", "Road", "King", "Blue", "Secret",
         "House", "Dream", "Wild", "Lost", "Heart", "Storm", "Island", "Shadow"]

QWEN_LIKE_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def make_movielens(raw_root: Path, n_users: int = 80, n_items: int = 120, per_user: int = 70, seed: int = 0,
                   as_zip: bool = True) -> Path:
    """Write ml-1m/ratings.dat and movies.dat (latin-1, '::') with a malformed line, an exact
    duplicate, an unmatched movie id, and timestamp ties."""
    rng = np.random.default_rng(seed)
    movies = []
    for mid in range(1, n_items + 1):
        title = f"{WORDS[mid % len(WORDS)]} {WORDS[(mid * 7) % len(WORDS)]} {mid}"
        if mid == 3:
            title = "Café Society"
        g = sorted(set(rng.choice(GENRES, size=rng.integers(1, 4), replace=False).tolist()))
        year = "" if mid == 5 else f" ({1950 + mid % 50})"
        movies.append(f"{mid}::{title}{year}::{'|'.join(g)}")
    item_bias = rng.normal(0, 0.8, n_items + 1)
    rows = []
    t0 = 956_703_932
    for u in range(1, n_users + 1):
        items = rng.choice(np.arange(1, n_items + 1), size=per_user, replace=False)
        times = np.sort(rng.integers(0, 600, size=per_user)) * 3600 + t0     # coarse grid -> ties
        ub = rng.normal(0, 0.6)
        for i, t in zip(items, times):
            r = int(np.clip(np.round(3.4 + ub + item_bias[i] + rng.normal(0, 0.9)), 1, 5))
            rows.append(f"{u}::{i}::{r}::{t}")
    rows.append(rows[10])                       # exact duplicate
    rows.append("7::99999::4::956703999")       # unmatched movie id
    rows.append("this is not a rating line")    # malformed
    d = raw_root / "ml-1m"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ratings.dat").write_bytes(("\n".join(rows) + "\n").encode("latin-1"))
    (d / "movies.dat").write_bytes(("\n".join(movies) + "\n").encode("latin-1"))
    if as_zip:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.write(d / "ratings.dat", "ml-1m/ratings.dat")
            z.write(d / "movies.dat", "ml-1m/movies.dat")
        (raw_root / "ml-1m.zip").write_bytes(buf.getvalue())
        shutil.rmtree(d)
    return raw_root


def make_tiny_model(out_dir: Path, seed: int = 0, num_layers: int = 4, hidden: int = 64) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    corpus = [f"{a} {b} ({1950 + i % 50}) | Genres: {', '.join(GENRES[i % 18:(i % 18) + 2])} | Rating: {i % 5 + 1}/5"
              for i, (a, b) in enumerate(zip(WORDS * 10, WORDS[::-1] * 10))]
    corpus += ["Each history line is a movie the user rated earlier, with its title, release year when known, genres, "
               "and the user's rating on a 1-5 star scale (5 = loved it, 1 = disliked it).",
               "User history (oldest to newest): Candidate movie: Will this user like the candidate movie?",
               "You are a careful recommendation assistant. Using only the user's past feedback shown, judge whether "
               "the user will like the candidate item. Answer with exactly one word.", "system user assistant"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=700, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.add_tokens(["Yes", "No", "A", "B"])
    fast.chat_template = QWEN_LIKE_TEMPLATE
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, tie_word_embeddings=False, initializer_range=0.2)
    model = Qwen2ForCausalLM(cfg).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    fast.save_pretrained(out_dir)
    return out_dir
