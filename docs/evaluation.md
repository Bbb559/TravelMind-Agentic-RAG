# 评测与能力边界

## 自动化基线

当前公开版本的无付费验收包括：

| 验收项 | 结果 |
| --- | --- |
| Python unittest | 274 tests PASS |
| Python compileall | PASS |
| Strict workflow smoke | 24/24 PASS |
| HTTP API smoke | 6/6 PASS |
| Frontend contract | PASS |
| Frontend production build | PASS |

普通回归会显式清空远程 GraphRAG 凭据并将 Global 服务开关固定为 `false`，因此不产生
Global Search 付费调用。

## 覆盖场景

- 输入门禁：空白、纯标点、问候和随机乱码不进入 Agent。
- Naive：单目的地玩法、交通、住宿、美食和景点问题。
- GraphRAG：跨目的地比较、路线与归纳问题。
- Multimodal：港澳台离线 Markdown 检索。
- Hybrid：大陆与港澳台混合问题，并支持单分支降级。
- Evidence Gate：目的地、实体或意图不匹配时拒绝生成。
- 输出清洗：最终答案不暴露 GraphRAG 内部表名和调试引用。
- 成本门禁：普通 smoke 不允许 Global Search。

## 显式远程验证

Official Local 与 Global 提供独立 smoke 脚本，但不会在普通回归中运行：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_graphrag_true_local_search.py --allow-paid-local-search
.\.venv\Scripts\python.exe scripts\smoke_graphrag_true_global_search.py --allow-paid-global-search
```

Global 还要求服务环境变量已明确开启。缺少显式参数时脚本退出码为 2，并且不得初始化
远程调用。

## 能力边界

- Demo 知识覆盖取决于随仓库提供的 CSV、Markdown 和 GraphRAG 产物。
- 未覆盖目的地会返回资料不足，而不是补写通用旅游建议。
- 默认 Naive/Multimodal 回答由模板整理，不等同于远程 LLM 润色。
- GraphRAG Official Local/Global 会调用远程 chat/embedding 服务并产生费用。
- Local evidence 只是 GraphRAG 产物证据预览，不是 Official Local 的替代品。
- Hybrid 当前只声明多源候选聚合。
- 在线问答不执行 OCR 或实时视觉理解。
- 回答适合作为旅行信息参考，不替代实时交通、票务、天气或安全公告。
