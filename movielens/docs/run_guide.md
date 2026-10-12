# Run guide (agent-written)

This guide covers setting up the environment, running each stage, the artifacts produced, and the conventions
this implementation chose where the supplied memo (`movielens_memo.md`) left text out. No experiment results
are included here: nothing in this repository was measured on MovieLens-1M or Qwen2.5-3B-Instruct.

## 1. Environment

```bash
cd movielens
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # versions validated by the CPU test suite (see below)
python -m pytest -q tests                # ~2 min on CPU; uses a synthetic dataset and a tiny local model
```

`requirements.txt` pins the versions the test suite was validated with: CPU only, Python 3.13, torch 2.14.1 and
transformers 5.19.0. **Nobody has validated a GPU run with these pins.** Run the test suite on the GPU host first.
`environment.json` then records the CUDA version, GPU model, dtype, attention implementation and package versions
for each run.

You need network access to `files.grouplens.org` (the dataset) and `huggingface.co` (the model). If those hosts
are blocked:

- **Data:** put the official `ml-1m.zip` at `data/raw/movielens_1m/ml-1m.zip`. Extracting it to
  `data/raw/movielens_1m/ml-1m/` also works.
- **Model:** download the model ahead of time, then pass `--set model.revision=<40-char sha>` and
  `--set model.local_files_only=true`. You can also point to a local snapshot with `--set model.name=/path/to/snapshot`;
  its revision is then recorded as a hash of the snapshot's files.

If a required input is missing, the affected stage records `blocked` in `stage_status.json` along with the path or
resource it needs. The report stage still writes `final_results.json` with `completed: false`.

## 2. Running

All commands run from `movielens/`. Each stage refuses to start until its prerequisite stage has completed
under the same resolved configuration.

```bash
python code/run.py prepare        --profile smoke     # data, splits, queries, labels, audit, planned work
python code/run.py baseline       --profile smoke     # model load, parity checks, unsteered scores + capture
python code/run.py fit-directions --profile smoke     # CPU only: probes, mean-difference, controls
python code/run.py sweep          --profile smoke     # validation only: steering grid, controls, comparators
python code/run.py select         --profile smoke     # immutable selected.json (+ .sha256)
python code/run.py evaluate       --profile smoke     # test: baseline + each distinct selected config
python code/run.py report         --profile smoke     # final JSON/CSV/Markdown/figures; no LLM calls

# Same stage functions in sequence. --resume skips stages completed under an identical configuration.
python code/run.py run --profile pilot --through baseline --resume
python code/run.py run --profile pilot --through report --resume
```

Useful flags:

- `--run-id NAME` keeps separate cache and output trees. Use a new run id instead of re-selecting:
  `selected.json` cannot be changed once written, and `select` fails if the sweep has changed underneath it.
- `--max-forward-pairs N` sets a work cap. When the cap is hit, the stage is marked `partial`, not completed,
  and the report shows `completed: false`. Rerunning with `--resume` and no cap continues from the cached pairs.
- `--set key=value` overrides any configuration value, for example `--set model.microbatch=16` or
  `--set model.device=cpu`.

Recommended order, as the memo specifies: smoke, then pilot, read the pilot report, then paper. Before
`sweep`, check `budget.json`. It holds the measured pairs per second, the peak memory, and an upper bound on the
remaining forward passes.

## 3. Layout

```
code/        run.py contracts.py data.py prompts.py model.py directions.py steering.py metrics.py report.py utils.py
configs/     common.yaml (model, profiles, scoring, grids) · movielens_1m.yaml · kuairand_pure.yaml (stub)
data/raw/<dataset>/                              downloads + checksums.json
data/processed/<dataset>/<protocol_id>/<profile>/ queries.jsonl, labels.parquet, user_groups.parquet,
                                                 direction_pairs.parquet, probe_val_pairs.parquet,
                                                 item_prior.parquet, items.parquet, data_audit.json, manifest.json
cache/<dataset>/<run_id>/prompts/                tokenized prompts keyed by template, tokenizer and query hashes
cache/<dataset>/<run_id>/jobs/<config_key>/      resumable score parts (+ captured states) per configuration
outputs/<dataset>/<run_id>/                      all artifacts listed in the memo + stage_status.json
```

`data/`, `cache/` and `outputs/` are not tracked by git.

A scoring job's cache key covers: the scorer identity (model and tokenizer revisions, dtype, attention
implementation, verbalizer strings and IDs, scorer version, hook location, prediction position), the prompt
template hash, the hash of the query file, and the intervention identity (kind, layer, direction ID, vector SHA,
strength, radius). Inside a job, completed parts are keyed by pair ID, and each pair's stored `prompt_hash` must
match the current tokenization. A stale prompt is therefore never reused, and a resume never relies on input order.

## 4. Conventions resolved where the memo text was missing

| Memo gap | Resolution in code |
|---|---|
| System instruction | `prompts.SYSTEM_INSTRUCTION`: "You are a careful recommendation assistant. Using only the user's past feedback shown, judge whether the user will like the candidate item. Answer with exactly one word." |
| Final user-message line | "Will this user like the candidate movie? Answer Yes or No." If Yes/No are not single tokens, the A/B fallback asks to "Answer A for yes or B for no." |
| Score | `score = logit(Yes) − logit(No)` at the response position, after the chat template's assistant prefix. |
| Raw-coordinate probe | `w_raw = w / scale`, `b_raw = b − Σ w·μ / scale`. The steering vector is `w_raw / ‖w_raw‖`. Parity between raw and standardized decisions is asserted. |
| Mean-difference direction | `normalize(mean(h | y=1) − mean(h | y=0))` on the same unstandardized inner-fit activations. |
| Intervention | `h ← h + α · radius_l · v̂` at the last valid prompt position of block `l`'s output. `radius_l` is the median ‖h‖ over the union of all inner-fit direction pairs for that seed at that layer, so every method at a layer shares one radius (matched norm). |
| Post-norm null control | The same addition applied to the final-norm output. `score` shifts by `(W_yes − W_no)·δ` for every candidate, so rankings cannot change. The sweep fails if more than 1% of queries change ranking beyond numerical noise. |
| Metrics | `DCG@k = Σ_{r≤k} gain_r / log2(r+1)` with linear gains. Recall@k = hits@k / positives. MRR = 1 / rank of the first positive. Precision@k and mean@k use the top `min(k, n)` items. Zero-denominator queries count as 0 in the all-query mean, and conditional means are reported as well. |
| Feasibility | `guardrail ≥ (1 − budget) · baseline_guardrail`. The unsteered point is always in the pool. If the baseline guardrail is 0, the result is `not_estimable`. |
| Path placeholders | `data/raw/<dataset>`, `data/processed/<dataset>/<protocol_id>/<profile>`, `cache/<dataset>/<run_id>`, `outputs/<dataset>/<run_id>`. |

Other decisions:

- **Validation users:** validation queries exclude the reserved `unseen_by_direction` users. Validation labels
  pick operating points and the learned-reranker λ, which counts as a learned score adjustment. Test covers all
  eligible users, and the reserved group is also reported on its own.
- **Fitting-period queries:** these are built only for inner-fit and probe-validation users. Their candidate pool
  is not capped; the per-class sampling cap and the per-user cap limit what is actually used. Only queries that
  contribute a selected pair are stored.
- **Candidate order:** a fixed hash of (data_seed, query, item), unrelated to labels, ratings or time.
- **Item-rating prior:** a Bayesian-smoothed item mean (`k = 10`) over ratings before the validation cutoff,
  excluding reserved users. `data_audit.json` → `item_prior.n_records` records how many labelled records it used.
  That number is usually far above the vector-fitting budget, so a sample-efficiency claim would need the
  matched-label sensitivity check from the memo, which is not implemented yet.
- **Cached-score comparators:** `base + λ · sd(base_fit) · (adj − mean_fit) / sd_fit`, with every statistic
  computed on inner-fit pairs before validation.
- **Controls:** the random-direction and shuffled-label controls run over the same layer × strength grid and go
  through the same selection rule at the same budget. They are scored on validation only. The reversed direction
  and the mean item-score shift are evaluated on test for each selected steering configuration. Item-shift coverage
  is in `evaluate_notes.json` and in `final_results.json` → `item_shift_coverage`.
- **Identity check:** strength 0 is checked on a 64-pair validation subset with identical batching, and must match
  bit for bit. Each run also records the difference from the cached baseline, which uses different batching.
- **Exploratory per-user chronological split:** not implemented. If the global split is `not_estimable`, the
  pipeline reports that and stops; it never falls back to another split.

## 5. Planning compute (estimates, not measurements)

Each pair is one prefill forward pass of about 700–1,000 tokens with 20 history items. No text is generated.
Read `plan.json` after `prepare` and `budget.json` after `baseline` for counts based on the realized data.

| Profile | Forward pairs (upper bound) | GPU with ~30–60 pairs/s | ~3–8 pairs/s |
|---|---|---|---|
| smoke | ~1–2k | minutes | ~5–10 min |
| pilot | ~55–70k | ~0.5–1 h | ~3–6 h |
| paper | ~3–4M | ~1–2 days | impractical |

## 6. KuaiRand

`configs/kuairand_pure.yaml` and `data.KuaiRandAdapter` are placeholders. Any stage run on that dataset records
`blocked`, with `docs/kuairand_memo.md` as the missing prerequisite. Once the companion memo arrives, the
KuaiRand work is limited to an adapter in `data.py` (raw files, parsing, history feedback fields, label columns,
item prior column) and its config (objectives, metrics, prompt feedback convention). The model, prompt renderer,
directions, steering, metrics and report code are shared, and the metric definitions come from config.
