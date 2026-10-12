# MovieLens-1M activation-steering implementation memo (as supplied)

> Transcribed from the implementation handoff supplied with this task. Passages that were missing from the
> supplied text (formulas, the system instruction, the final user-message line, callable signatures, path
> placeholders) are marked **[not supplied]**; the implementation's resolution of each is recorded in
> `run_guide.md` → "Conventions resolved where the memo text was missing".

## Goal

Create a reproducible activation-steering experiment on MovieLens-1M and a shared Python implementation that
the companion KuaiRand memo can reuse. Implement the pipeline and run it in stages: prepare data, establish the
unsteered baseline, fit directions, tune on validation, evaluate locked configurations, and write final reports.

This is a coding-agent handoff, not a report of executed experiments. The command names below are an interface
to implement. No measured results are supplied.

Use a frozen general-purpose LLM as a candidate-conditioned recommendation scorer. No recommendation-trained
public checkpoint was supplied. The default is Qwen/Qwen2.5-3B-Instruct, a multilingual instruction model.
Resolve and record an immutable model and tokenizer revision when implementing.

Do not initialize a random catalog head, add new item embeddings, perform LoRA/SFT/RL, or optimize backbone
weights. The only fitted models are small CPU logistic regressions on cached activations and simple statistical
baselines. Mean-difference vectors require no classifier fitting.

This choice implements frozen-LLM candidate reranking. It is not a reproduction of the internal GenRecs
catalog-head architecture and does not establish full-catalog retrieval quality. A trained recommender can later
be added as a distinct model adapter and protocol, with separate results. Do not mix its numbers with the
frozen-LLM protocol.

We score a finite supplied candidate set. The benchmark candidate pool below comes from observed future ratings
so its outcome labels are known. Candidate membership is retrospective benchmark information, not a deployable
candidate-retrieval system. Report this limitation prominently. An exhaustive universe of every possible title
is not required.

The research question is whether intermediate activation edits improve ranking of positive-feedback items, or
steer a known genre, while preserving the declared quality guardrail. Do not promise improvements before running
the experiments.

## Project layout

Create one project root with the following paths. Both datasets use the same modules. Dataset-specific branching
belongs in data.py and configuration, not in copied model, probe, steering, or metric implementations.

| Path | Responsibility |
|---|---|
| code/run.py | CLI, stage dependencies, run orchestration, resume and failure status |
| code/contracts.py | Typed query, candidate, pair, direction, intervention, and result schemas |
| code/data.py | Dataset downloads, parsing, preprocessing, splits, history construction, candidate construction, labels, and coverage audits; dispatch by dataset key |
| code/prompts.py | Versioned shared prompt renderer, dataset field formatting, deterministic truncation |
| code/model.py | Frozen model loading, verbalizer validation, batched pair scoring, selected activation capture |
| code/directions.py | CPU logistic probes, raw-coordinate conversion, mean-difference directions, projection diagnostics |
| code/steering.py | Temporary activation hooks, strength scaling, layer selection helpers, random and null controls |
| code/metrics.py | Query ranking metrics, paired comparisons, user bootstrap, confidence intervals |
| code/report.py | Final JSON, CSV, Markdown, and figures from saved results |
| code/utils.py | Hashes, seeds, atomic writes, manifests, cache validation, environment capture |
| configs/common.yaml | Model, profiles, scoring protocol, strength grids, shared evaluation settings |
| configs/movielens_1m.yaml | MovieLens paths, labels, objectives, and split settings |
| configs/kuairand_pure.yaml | KuaiRand overrides specified by the companion memo |
| docs/ | These two implementation memos and an agent-written run guide |
| data/raw/**[not supplied]**/ | Original downloads and checksums |
| data/processed/**[not supplied]**/ | Immutable queries, labels, split assignments, coverage report |
| cache/**[not supplied]**/ | Tokenized prompts, baseline scores, activations, and reusable fit artifacts |
| outputs/**[not supplied]**/ | Stage manifests, selected configurations, per-query results, final reports |
| tests/ | Small meaningful correctness checks listed below |

Do not name the importable package code; Python has a standard-library module with that name. Running
python code/run.py can import sibling modules, or the agent may put a uniquely named package beneath code while
preserving these responsibilities.

Use ordinary PyTorch, Transformers, NumPy, pandas or PyArrow, scikit-learn, and plotting tools. Pin the versions
actually validated by the implementation. Record CUDA, GPU model, dtype, attention implementation, package
versions, model revision, tokenizer revision, and source commit in environment.json. Do not guess compatibility
from an untested version list.

Suggested callable contracts: **[not supplied]**

The data supplied to fit_directions must be sufficient for either dataset. It must not read MovieLens files
internally.

## Data contracts

Store processed data in Parquet with schema versions. JSONL is acceptable for nested query records if the schema
and ordering are stable.

A query contains query_id, user_id, dataset, split, cutoff_timestamp, history, candidate_ids, candidate_source,
and protocol_id. The history contains item metadata, interaction time, and feedback that was available before
cutoff_timestamp.

A candidate contains item_id, display_text, available metadata, and metadata_missing flags. Current-query
outcomes are not part of the candidate object passed to the prompt renderer.

Keep a separate label table keyed by query_id and item_id with rating, satisfaction_label, graded_gain,
target_attribute, observed_timestamp, and label_observed. For MovieLens, every candidate in the main protocol
has an observed rating. The corresponding KuaiRand fields are in the second memo.

A scoring pair is one query and one candidate. Its identity must survive shuffling, batching, caching, and
joining. Assert uniqueness before every join. Never join activations to labels by row position alone.

Required cached score fields include query_id, item_id, prompt_hash, baseline_score, verbalizer_ids,
model_revision, and scorer_version. Required captured-state keys additionally include layer_index,
hook_location, and prediction_position.

Current candidate feedback must never appear in the rendered prompt. Earlier feedback in history is allowed.
User IDs and opaque movie IDs remain join keys; do not present their arbitrary numeric values as semantic
preference features.

## MovieLens-1M source and preprocessing

Use the official MovieLens-1M release:

- Archive: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Schema and usage information: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt

Read ratings.dat as UserID, MovieID, Rating, Timestamp and movies.dat as MovieID, Title, Genres using the
documented double-colon delimiter. Handle the source encoding explicitly. Keep rating values and timestamp units
unchanged. Join movie metadata by MovieID and report unmatched IDs. Demographics are unnecessary.

Use title, release year if present in the title, and genres as item text. Do not ask an LLM to invent plots or
descriptions. Genre labels come from the supplied metadata.

Reject malformed records, report exact duplicate counts, and deduplicate exact repeated rows. For repeated
ratings of the same movie by the same user within an evaluation window, keep the earliest observed rating and
record the rule. Historical repeated interactions may remain, subject to the history-length cap.

Use the last 20 eligible historical items by default, requiring at least 5. Cut histories by timestamp strictly
less than the query cutoff. Same-timestamp history records may be ordered deterministically by item ID for
serialization; do not interpret that tie-breaking order as an observed behavioral sequence.

## Protocol and splits

Default protocol_id is observed_feedback_rerank_v1.

Compute global timestamp cutoffs at the 60th, 80th, and 90th event percentiles. Move a cutoff to the next
distinct timestamp where necessary so equal timestamps never straddle a boundary. Save the exact numeric cutoffs
and resolved UTC representations.

1. History period: events before the first cutoff. No backbone training occurs here.
2. Direction-fitting period: first through second cutoff.
3. Validation period: second through third cutoff.
4. Test period: third cutoff through the end.

For each user and each fitting, validation, or test period, build one snapshot query. Its cutoff is that
period's start. Its history may use all available earlier-period events, including feedback already observed
then. It may never use feedback from inside the query's outcome period.

Candidates are the distinct movies that the user rated during that outcome period, excluding movies already
present in that user's pre-cutoff history. Choose up to the profile's maximum candidate count by a deterministic
hash sample independent of rating, genre, or model score. Do not insert additional liked movies to make a query
easier. Sort or shuffle the final candidate list by a fixed item-independent convention; its order must not
expose future interaction order or label rank.

Require at least 5 distinct candidates for main reported query metrics. Profiles cap the maximum candidate count
rather than fabricating extra candidates. Persist eligible and rejected user counts, candidate-count histograms,
and every exclusion reason. Report results stratified by candidate count when the distribution is broad.

This protocol ranks a user's future observed items from a fixed earlier history. It measures preference ranking
within that retrospective pool. It does not measure next-watch prediction, exhaustive retrieval, or the chance
that a new recommendation will cause a rating.

If the global windows yield too few eligible queries, finish the data audit and produce a not_estimable result
for the planned main protocol. A separately named per-user chronological split may be provided as exploratory
output, but never silently replace the main split or claim global temporal transfer for that alternative.

Public benchmark scale varies after these filters. Profile query counts are maxima, not promises. Select
evaluation users by deterministic hash without looking at labels or scores. If fewer are eligible, use all and
report realized counts.

Reserve 20 percent of users by stable hash as unseen_by_direction. Their labels must never fit probes, mean
directions, scalers, or learned score adjustments. Their historical records can supply their own inference
context. Report test results on all eligible users and separately on this reserved group.

Within the remaining fitting users, make an 80/20 user-disjoint inner fit/probe-validation split. Fit directions
only on inner-fit users. Tune probe regularization using inner probe-validation users. Main validation is
reserved for intervention and operating-point selection. Test labels have no role in either selection.

## Objectives

1. Satisfaction: direction positives have rating 4 or 5; negatives have rating 1 or 2. Rating 3 is excluded from
   direction fitting and probe diagnostics, but retained in recommendation evaluation. Evaluation
   satisfaction_label is 1 for rating at least 4, else 0.
2. Comedy exposure: positives have Comedy among their genres; negatives do not. This is an enumerable attribute
   control, not an inferred user-satisfaction label.

For each objective, sample equal positive and negative counts from the inner-fit pool, up to the profile cap.
Limit each user's contribution to at most five examples per class before increasing any cap. If fewer examples
are available, use the realized balanced count and report it. Do not oversample duplicated examples and then call
them additional independent observations.

Inner probe-validation examples retain their natural eligible class distribution. Select them by deterministic
outcome-independent hash with caps of 128 pairs for smoke, 1,024 for pilot, and 4,096 for paper. Do not forward
the entire remaining dataset merely to validate a small probe. Report AUROC, AUPRC, prevalence, class counts, and
user counts. If one class is absent, record the diagnostic as not estimable, not 0.5 by fiat.

Save raw selected pair IDs, user groups, sample seeds, label rules, and all exclusion counts. Mean-difference and
logistic-probe directions must use exactly the same fit pairs for a given objective and seed.

## Prompting and scoring

Render each history-candidate pair independently with a fixed prompt. Use the checkpoint's chat template and its
assistant-generation prefix. The common English system instruction can be: **[not supplied]**

The user message should describe the historical feedback convention, list the available history, provide the
candidate metadata, and end with: **[not supplied]**

For MovieLens, history feedback is the already-observed numeric rating. The candidate includes title, release
year when available, and genres, but no candidate rating or future-event timestamp. Use consistent delimiters.
Keep this baseline prompt fixed after preflight. Any objective instruction is an explicitly named prompt baseline
with its own prompt hash.

For the main protocol, validate that Yes and No are single tokens in the actual response position. If necessary,
use single-token A and B with an explicitly updated prompt defining their meanings. Record the exact strings and
token IDs. Do not silently score just the first token of a multi-token verbalizer. An optional
complete-continuation scorer can be a different protocol, but is not required here.

Compute: **[not supplied]**

Sort candidates by descending score with a stable item-ID tie-break. No sampling, free-form generation, beam
search, or temperature is involved. These are preference-ranking scores, not calibrated satisfaction
probabilities.

Because every pair includes the candidate at inference, its metadata may also be present when capturing
direction-fitting activations. This is intentionally different from a history-only catalog-head probe. The
forbidden information is the candidate's outcome, not the candidate identity or available content.

A memory-efficient adapter may take the final normalized hidden state at the prediction position and multiply by
the two relevant LM-head rows. On a small fixture, prove this matches the corresponding logits from the standard
full model forward. Include any bias and model-specific output transformation. Do not assume this shortcut works
for every architecture without parity validation.

Use model.eval(), frozen parameters, and torch.inference_mode(). Capture selected layers using hooks instead of
requesting every layer's full hidden-state tensor. Store only the last valid prompt-position vector for each
selected layer. Determine the position from the attention mask; do not select a padding token.

Start with bf16 where supported and fp16 otherwise. Do not use quantization for the primary run unless needed; a
quantized run is a separately labeled configuration with its own parity checks. Truncate oldest history items
before truncating the candidate or final instruction. Save truncation counts and final token lengths.

## Directions

For every selected layer, fit an L2-regularized logistic regression on inner-fit activations with StandardScaler
fitted only on that split. Use a small C grid such as 0.01, 0.1, 1, and 10. Choose C by inner probe-validation
AUROC, breaking ties toward stronger regularization. Record nonconvergence and class order.

If standardized hidden states are (h - mu) / scale, save: **[not supplied]**

Use the scaler's safe scale handling for constant dimensions. Validate raw and standardized decision-function
parity. Do not inject standardized-coordinate coefficients directly into raw residual activations.

Compute the other direction on the same unstandardized fit activations: **[not supplied]**

A zero or nonfinite direction is an explicit failed fit. Save float32 vectors, preprocessing statistics,
classifier coefficients, selected C, layer index, hook position, support counts, training-pair hashes, and seed.
The vector norm must be one within numerical tolerance.

For injection scaling, define radius_l as the median L2 norm of raw inner-fit activations at that layer. The
intervention is: **[not supplied]**

radius_l is frozen from fitting data and does not depend on the candidate label, validation outcome, or test
outcome. Positive strengths are 0.01, 0.03, and 0.10. Include strength 0 as an identity check. Use -0.03 as a
direction-reversal diagnostic, not a test-selected sign.

Layer locations are fractions of transformer depth resolved from the actual configuration. Pilot uses 0.50 and
1.00; paper uses 0.25, 0.50, 0.75, and 1.00. Define index = ceil(fraction * num_layers) - 1, deduplicate, and
save exact zero-based indices. The 1.00 location is the last block output before the model's final normalization.

Apply one vector at one layer at the last valid prompt position, after the block output and before subsequent
blocks or final normalization. Do not modify all history positions. Hooks must restore the model after each
intervention context and must not accumulate across batches or experiments.

## Methods and controls

- Unsteered frozen LLM.
- Mean-difference steering.
- Logistic-probe steering.
- Direct attribute score boost for Comedy: baseline_score + lambda * comedy_flag.
- Item-rating prior: a smoothed historical item mean computed only from permitted fitting/history records before
  validation; fixed for the test run, with a global fallback for unseen items.
- Learned linear reranker: baseline_score + lambda * raw_probe_decision(h). It uses the same fitted probe and
  cached pair activations, making this a cheap personalized learned comparator.
- Objective-prompt baseline using one explicit instruction to favor high ratings or Comedy. Keep it as a
  separately versioned prompt.
- Random unit direction and shuffled-label direction controls, evaluated on validation at matched norm and budget.
- Reversed direction at a selected validation configuration.
- Mean item-score shift control when an item has sufficient fitting support.

Use lambda grids fixed in configuration; for example 0, 0.25, 0.5, 1, and 2 after standardizing each baseline's
adjustment using fitting-set statistics. All normalization parameters are fitted before validation. Never z-score
with test labels or tune a scale on test performance.

Allow the learned linear reranker to choose its own layer and lambda on validation under the same guardrail
budget. Do not force it to use a layer selected for activation steering. For item priors, report the number of
permitted labeled records used; if it exceeds the vector-fitting budget, include a matched-label sensitivity
check before making a sample-efficiency claim.

For the mean item-score shift, estimate score_steered - score_baseline for fitting pairs and average by item for
the selected direction and strength. The test baseline adds the saved item offset to unsteered scores. Items with
no support receive the global mean offset. Report item and interaction coverage and a supported-item subset; an
unsupported-item result is not evidence against item bias.

An especially important null control follows from this scorer. If one constant vector is added after final
normalization immediately before a linear LM head, then: **[not supplied]**

The offset is the same for every candidate. Candidate rankings should remain unchanged, apart from numerical
ties. This differs from a catalog head with a different output row for every item. Implement a small
post-normalization null check. If it changes rankings materially, investigate the scorer, position handling, hook
placement, or caching before reporting results.

A higher mean Yes score, a higher average probe score, or a higher predicted positive probability is not evidence
of better recommendation ranking. Report within-query ranking changes.

## Metrics

All ranking metrics refer to the exact stored candidate sets. Keep candidates, labels, and eligible query IDs
identical across methods.

For MovieLens satisfaction:
- Target metric: NDCG_like@5 using binary gain 1 for rating at least 4.
- Quality guardrail: NDCG_rating@5 with supplied gain rating - 1.
- Secondary metrics: Precision_like@1/3/5/10, Recall_like@5/10, MRR_like, mean rating of top 5, and rate of
  explicitly low ratings in top 5.

For Comedy:
- Target metric: ComedyExposure@5.
- Guardrail: NDCG_rating@5.
- Also report satisfaction metrics so genre movement is not mistaken for a satisfaction gain.

The MovieLens satisfaction target and graded-rating guardrail are correlated views of the same feedback. They are
not independent engagement and satisfaction signals. The genre experiment has a clearer objective-versus-quality
trade-off; KuaiRand supplies separate like and long-view signals.

Metric definitions: **[not supplied]**

The gain is already specified. Do not exponentiate it again. Set NDCG and recall to 0 for zero-positive or
zero-ideal-gain queries in the all-query macro average, and additionally report conditional metrics on queries
with a positive denominator. Always include both denominators and the zero-positive query rate.

Use paired user-level bootstrap confidence intervals for changes from baseline. Default 2,000 bootstrap draws for
final reports and a fixed bootstrap seed. Preserve all queries from a sampled user together if the implementation
later supports multiple queries per user. Do not use candidate pairs as independent units for ranking uncertainty.

## Selection

Tune on main validation only. For the pilot, choose one operating point per steering estimator and objective at a
1 percent relative guardrail-loss budget. For the paper profile, use budgets 0, 0.01, and 0.02. A feasible
configuration satisfies: **[not supplied]**

Among feasible settings, maximize the target metric. Break ties by better guardrail, smaller absolute strength,
then deterministic layer order. Include the unsteered point as feasible. If baseline guardrail is zero, mark
relative-budget selection not estimable rather than dividing by zero.

Save selected.json and its hash before any selected-method test run. Do not refit directions after selection or
choose a test Pareto winner. Validation sweep plots may contain all configurations; headline test tables contain
the frozen selections.

Report whether the selected test configuration actually satisfies the guardrail. A validation-selected budget is
not a guarantee of test performance.

## Stages

1. prepare: download or reuse raw files, build splits/queries/labels, audit data, and save the planned work counts.
2. baseline: score validation queries and the fitting/probe-validation capture pool; capture required states
   during those same forwards; write baseline diagnostics.
3. fit-directions: CPU-only fitting and projection diagnostics from cached states.
4. sweep: validation-only steering and comparator evaluation, reusing baseline caches.
5. select: write immutable validation-selected configurations.
6. evaluate: score test once per distinct selected configuration; include baseline and the cheap cached-score
   comparators.
7. report: aggregate saved outputs; no LLM calls.

Also implement a convenience run command with --through baseline or --through report and --resume. The run command
must execute the same stage functions and checks, not an independent code path.

A stage with missing prerequisites should name the missing artifact. A completed stage should reuse a matching
cache. Cache keys must include data/query hashes, model/tokenizer revisions, prompt/verbalizer version, layer/hook
specification, dtype, and intervention identity. Changing a prompt, split, candidate set, or model must invalidate
the relevant outputs.

Write outputs atomically. Failed or interrupted batches must not be marked complete. Resume from pair IDs and
configuration IDs, never from a count that assumes the input order has not changed.

## Profiles

| Profile | Direction examples per class | Max validation users | Max test users | Max candidates per query | Layers | Direction seeds |
|---|---|---|---|---|---|---|
| smoke | 32 | 8 | 16 | 5 | last block | 42 |
| pilot | 256 | 64 | 128 | 20 | middle and last block | 42 |
| paper | 1,000 | 256 | 1,000 | 50 | four depths | 42, 43, 44 |

For smoke, use only one positive strength, 0.03. Its purpose is pipeline correctness, not an estimate of
scientific effect. A profile cap never overrides the available eligible data or creates synthetic outcomes.

Keep data_seed=42 fixed for split groups, evaluation-user selection, candidate sampling, and ties. Vary
direction_seed only for the direction-fitting sample and random-direction controls. All three paper-profile
direction seeds must see the same validation and test queries and candidate sets. Also keep prompt text, model
revision, and bootstrap settings fixed across them.

Optional paper analyses must be enumerated in the saved plan before execution. Useful low-cost additions are
fitting-sample learning curves on validation and probe diagnostics at all four captured depths. A test learning
curve requires additional explicitly counted inference work. Do not let a default run expand into an unspecified
exhaustive ablation grid.

## Compute

Load one frozen model and evaluate serial intervention batches. Start with a small pair microbatch and reduce it on
CUDA OOM while recording the effective size. Default maximum prompt length is 1,536 tokens; preserve the candidate
and response cue. Avoid activation gradients and optimizer state.

Inference is not free merely because the backbone is frozen. Each candidate consumes a forward pass, and each
intervention configuration consumes new forwards. For one objective in the pilot, two estimators times two layers
times three positive strengths means 12 validation settings, in addition to controls. At the profile caps that is
15,360 scored pairs just for that sweep. Two objectives cost more; baseline passes and capture are reusable.

After the baseline, measure actual pairs per second and peak memory and estimate remaining work from uncached
pair-configuration counts. Write budget.json. Do not promise a runtime or GPU size without measuring. Any
configured work cap must produce a clearly labeled partial run if exceeded, never a complete-looking final table.

The default implementation must finish and report the pilot before expanding to the paper profile. A weak baseline
is a valid finding: compare against random ranking and the item prior on validation, and state if the frozen model
is not competitive. Do not secretly add fine-tuning or switch checkpoints to rescue results.

## Artifacts

| File | Contents |
|---|---|
| config.resolved.yaml | Fully resolved parameters, exact layers, hashes, and model revision |
| environment.json | Hardware and software provenance |
| data_audit.json | Splits, support, class balance, exclusions, candidate and metadata coverage |
| baseline_summary.json | Unsteered validation metrics and throughput |
| probes.csv | Estimator, layer, C, support, AUROC, AUPRC, and prevalence |
| directions/ | Vectors and fitting metadata |
| validation_sweep.csv | Every evaluated validation configuration |
| selected.json | Locked operating points and validation evidence |
| scores_test.parquet | Per-pair test scores with query and configuration identity |
| per_query_metrics.parquet | Per-query metrics used for paired comparisons |
| final_results.json | Authoritative machine-readable report |
| final_results.csv | Flattened comparison table |
| results.md | Human-readable findings, limitations, failures, and run instructions |
| figures/ | Probe-by-layer, validation trade-off, selected test comparison, and cost plots |
| budget.json | Measured and estimated compute |
| stage_status.json | Completed, failed, partial, skipped, and not-estimable stages |

final_results.json must include schema_version, run_id, dataset, protocol_id, profile, model_revision,
backbone_trained=false, completed, realized_counts, candidate_source, split_cutoffs, selection_hash, metrics,
confidence_intervals, timing, warnings, and artifact_paths.

Each metric row needs objective, method, split, subgroup, vector seed, layer, strength or lambda, guardrail budget,
baseline value, method value, absolute delta, relative delta when defined, lower and upper confidence bounds,
query/user counts, and candidate counts. Undefined values should be null with a reason, not NaN in JSON or an
invented zero.

Separate uncertainty over users from variability over vector-estimation seeds. Three direction seeds on one frozen
model are not three trained-model seeds. Test output must preserve this distinction.

## Tests

Use small fixtures and a tiny real-data smoke run. Check the risks that could invalidate the experiment:

- Changing the current candidate's outcome label does not change its rendered prompt or token IDs.
- No history event reaches or exceeds its query cutoff; split timestamp ties remain intact.
- Candidate sets and query denominators are identical across methods.
- CPU direction fitting does not read validation/test labels.
- Raw-coordinate and standardized probe scores agree.
- The two-row LM-head shortcut matches standard forward logits.
- Strength zero reproduces baseline scores within dtype-appropriate tolerance.
- Hooks are removed after use; a later unsteered pass remains unchanged.
- The post-normalization constant-vector control leaves rankings unchanged.
- Reordering candidate pairs or changing batch size preserves identities and rankings within numerical tolerance.
- Metrics agree with hand-computed toy queries, including zero-positive and short-candidate cases.
- Resume reproduces an uninterrupted run and does not reuse an incompatible cache.

Do not generate fake experiment results to demonstrate the report. If the data, metadata, or model is unavailable,
still write a machine-readable blocked status naming the missing prerequisite. If results are null or negative,
report them as such.

## Scope of claims

The public experiment can support claims about frozen-LLM reranking of supplied candidate sets and intervention
effects on observed feedback. It cannot establish exhaustive recommendation quality, online satisfaction lift, or
a trained GenRecs model's behavior.

MovieLens timestamps are rating times. Candidate selection from future observed items is retrospective. Metadata
and movie knowledge may already be represented in the language model; this is not a claim of
pretraining-unseen item generalization.

Relevant source links:
- MovieLens official README: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
- Model card: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Closely related recommendation steering: https://arxiv.org/abs/2604.01036
- PopSteer: https://arxiv.org/abs/2601.15122

Implementation completion means the same shared engine runs both dataset configurations, the baseline precedes
steering, the complete pilot is resumable, and the final report faithfully distinguishes measured results,
unsupported comparisons, and missing evidence.
