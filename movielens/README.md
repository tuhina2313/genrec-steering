# MovieLens-1M activation steering (frozen-LLM reranking)

This project tests whether editing a frozen LLM's activations at one layer helps it rank a user's future MovieLens
movies, for two objectives: satisfaction (ratings 4–5) and Comedy exposure. A graded-rating NDCG guardrail must
stay within a set budget. The model is Qwen/Qwen2.5-3B-Instruct with frozen weights, scoring each
(history, candidate) pair as logit(Yes) − logit(No). The only fitted models are small CPU logistic probes and
mean-difference vectors.

- Specification: [`docs/movielens_memo.md`](docs/movielens_memo.md)
- How to run, artifacts, and conventions: [`docs/run_guide.md`](docs/run_guide.md)

```bash
pip install -r requirements.txt
python -m pytest -q tests
python code/run.py run --profile smoke --through report --resume
python code/run.py run --profile pilot --through report --resume
```

Candidates are movies each user actually rated later. That is retrospective benchmark information, not a
retrieval system, so results describe preference ranking within that pool only. This repository contains no
measured results yet.
