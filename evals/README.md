# TravelMind Evaluation Sets

`v1/` is the frozen public holdout used by `scripts/evaluate_agentic_rag.py`.

- `route_cases.jsonl`: 60 six-class routing cases.
- `workflow_cases.jsonl`: 40 answerable/refusal workflow cases.
- `paid_local_cases.jsonl`: 6 explicitly authorized Official Local cases.
- `manual_annotations.jsonl`: 30 human-review tasks; pending rows do not
  produce faithfulness metrics.
- `results/offline_v1.json`: machine-readable offline result.
- `results/offline_v1.md`: compact result summary.

Do not remove failed cases or tune against this holdout. Develop fixes with
separate cases, then publish a new benchmark version and compare both versions.
