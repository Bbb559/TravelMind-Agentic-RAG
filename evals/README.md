# TravelMind Evaluation Sets

`v1/` v1/ 目录为冻结的公开保留测试集，专供 `scripts/evaluate_agentic_rag.py`评估脚本使用.

- `route_cases.jsonl`:  60 six-class routing cases.
- `workflow_cases.jsonl`: 40 answerable/refusal workflow cases.
- `paid_local_cases.jsonl`: 6 explicitly authorized Official Local cases.
- `manual_annotations.jsonl`: 30 human-review tasks; pending rows do not
  produce faithfulness metrics.
- `results/offline_v1.json`: machine-readable offline result.
- `results/offline_v1.md`: compact result summary.

请勿删除失败用例，也不要针对此保留测试集进行调优。应使用独立的测试用例开发修复方案，随后发布新的基准版本并对比两个版本的结果.
