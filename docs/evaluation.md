# 评测与能力边界

## 冻结评测集

`evals/v1/` 是公开、版本化的留出集。问题没有复用 README 演示问题或原有
smoke 用例，冻结后不根据本次结果删改困难样本。

| 数据集 | 数量 | 用途 |
| --- | ---: | --- |
| Route | 60 | 六类路由的 Accuracy 与 Macro-F1 |
| Offline workflow | 40 | Evidence Hit@3、安全拒答和误生成 |
| Paid Official Local | 6 | 显式授权后的真实 GraphRAG Local Search |
| Manual annotations | 30 | 陈述级来源支持率与回答级幻觉率 |

路由集包含 Naive 15、GraphRAG 10、Multimodal 10、Hybrid 10、
invalid input 5 和 fallback 10。工作流集包含 20 条可回答问题与 20 条无证据或
不受支持问题。

## 2026-06-22 无付费结果

运行环境为 Windows 11、Python 3.12.10，单线程，每题预热后运行 3 次。
评测器会显式清空普通 LLM、embedding 和 GraphRAG 凭据，并强制关闭 Global Search。

| 指标 | 结果 | 说明 |
| --- | ---: | --- |
| Route Accuracy | 98.3% | 59/60 |
| Route Macro-F1 | 0.986 | 六类宏平均 |
| Evidence Hit@3 | 95.0% | 19/20 个可回答问题 |
| Safe Refusal Rate | 90.0% | 18/20 个无证据问题 |
| Unsafe Generation Rate | 10.0% | 2/20 个无证据问题 |
| Workflow latency P50 | 约 20 ms | 本机离线运行，仅供相对参考 |
| Workflow latency P95 | 约 799 ms | 包含 GraphRAG parquet evidence 检查 |
| 远程付费调用 | 0 | Official Local/Global 均未执行 |

工作流分 Agent 延迟：

| Agent | P50 | P95 |
| --- | ---: | ---: |
| NaiveTravelAgent | 21.6 ms | 38.8 ms |
| MultimodalTravelAgent | 10.1 ms | 11.6 ms |
| GraphRAGAgent | 769.4 ms | 865.7 ms |
| HybridAggregator | 727.2 ms | 2036.4 ms |

GraphRAG 和 Hybrid 的无付费延迟主要来自本地 parquet evidence 检查；Hybrid 的 P95
还包含并行分支调度与慢分支等待。因此这些数字不能直接外推到远程 Official Local/Global。

机器可读结果见 [`evals/results/offline_v1.json`](../evals/results/offline_v1.json)，
简版摘要见 [`evals/results/offline_v1.md`](../evals/results/offline_v1.md)。

### 已保留的失败案例

- “西安和南京的人文旅行侧重点有何不同？”没有命中当前 Router 的显式比较词，
  被分到 Naive，形成 1 条路由错误。
- “大理双廊住在哪里看洱海更方便？”检索到了双廊住宿资料，但意图分类器把
  “住在哪里”归为 general，而不是 accommodation，因此不满足严格 Hit@3 定义。
- 两条港澳台内部比较/滑雪主题问题被单区域 Markdown 证据误判为可回答，
  导致 Safe Refusal 为 90%、Unsafe Generation 为 10%。

这些问题保留在 `v1` 中作为后续版本的固定回归目标，不会从分母中移除。

## 指标定义

- `Route Accuracy`：正确路由数 / 路由问题总数。
- `Macro-F1`：六个路由类别 F1 的宏平均。
- `Evidence Hit@3`：Top 3 中至少存在一条 `evidence_valid=true`，且其
  `matched_entities`、`matched_intents` 覆盖该问题预期实体和意图。
- `Safe Refusal Rate`：无证据问题中，返回稳定 fallback 且未输出具体建议的比例。
- `Unsafe Generation Rate`：无证据问题中，仍采用有效证据状态或生成具体回答的比例。
- `Claim Support Rate`：人工标注为 supported 的陈述数 / 全部可验证陈述数。
- `Answer Hallucination Rate`：至少包含一条 unsupported 陈述的回答数 /
  全部人工评审回答数。

Evidence Hit@3 不是完整语料召回率。Evidence Gate 通过率也不能直接称为幻觉率。

## 人工忠实度

`manual_annotations.jsonl` 预留了 30 条人工评审任务，标签为：

- `supported`
- `unsupported`
- `not_verifiable`

当前文件状态为 `pending`。评测器在任一记录未完成时拒绝计算忠实度，避免用自动门禁
冒充人工幻觉评测。完成标注后运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_agentic_rag.py --suite manual
```

## 复现命令

无付费离线评测：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\evaluate_agentic_rag.py `
  --suite offline `
  --repeats 3 `
  --output-json evals\results\offline_v1.json `
  --output-markdown evals\results\offline_v1.md
```

真实 Official Local 专项评测必须显式授权：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_agentic_rag.py `
  --suite paid-local `
  --allow-paid-local-search
```

缺少参数时退出码为 2，并且不会初始化远程适配器。该专项始终将 Global Search
固定为关闭。当前 `v1` 六题尚未作为本次无付费收口的一部分重新执行，因此不发布新的
6 题成功率。

## 自动化回归

效果评测之外仍保留 unittest、compileall、strict smoke、HTTP smoke、
frontend contract 和 frontend production build。普通回归不会产生付费调用。

## 能力边界

- Demo 知识覆盖取决于随仓库提供的 CSV、Markdown 和 GraphRAG 产物。
- 默认 Naive/Multimodal 使用确定性模板，不等同于远程 LLM 润色。
- GraphRAG Official Local/Global 会调用远程 chat/embedding 服务并产生费用。
- Local evidence 只是低成本证据预览，不是 Official Local 的替代品。
- Hybrid 当前只声明多源候选聚合。
- 在线问答不执行 OCR 或实时视觉理解。
- 结果不替代实时交通、票务、天气或安全公告。
