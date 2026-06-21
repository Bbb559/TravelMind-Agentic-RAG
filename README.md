# TravelMind Agentic RAG

TravelMind is a lightweight multi-agent travel assistant that routes questions
across structured travel data, offline multimodal documents, and official
GraphRAG search. It emphasizes evidence-gated answers, explicit cost controls,
and honest fallbacks.

TravelMind 是一个面向旅游问答的 Agentic RAG MVP。系统通过
SystemAgent 将问题分发到 Naive、GraphRAG、Multimodal 或 Hybrid
链路，并在回答前执行输入检查、证据有效性判断和输出清洗。

## 核心能力

- **NaiveTravelAgent**：面向中国大陆单目的地玩法、交通、住宿、美食等问题，使用 CSV/FAISS。
- **GraphRAGAgent**：面向跨目的地比较、路线归纳和实体关系问题。
- **MultimodalTravelAgent**：检索港澳台旅游 PDF 经离线模型处理后形成的 Markdown 文档及其向量索引。
- **HybridAggregator**：并行聚合 GraphRAG 与 Multimodal 的有效候选，允许单分支降级，不声明深度融合。
- **Evidence Gate**：没有目的地或主题相关证据时，不生成具体旅游建议。
- **成本保护**：Official Local Search 是 GraphRAG 默认正式链路；Global Search 必须服务端与请求端双授权。

完整设计见 [架构说明](docs/architecture.md)。

## 运行环境

- Python 3.12
- Node.js 20+，建议 20.19+ 或 22.12+
- Git LFS 3.x

本仓库包含约 201 MB 由 Git LFS 管理的预构建 Demo 资产。首次克隆后必须执行：

```bash
git lfs install
git lfs pull
```

实际下载量和 LFS 配额取决于代码托管平台。

## 快速启动

以下命令以 Windows PowerShell 为例。

### 1. 安装后端依赖

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

默认配置不启用普通 LLM，也不允许 GraphRAG Global Search。需要远程能力时，请根据
[配置说明](docs/configuration.md) 填写自己的环境变量。

### 2. 启动 FastAPI

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m uvicorn travelmind.api:app --host 127.0.0.1 --port 8000
```

健康检查：

```text
GET http://127.0.0.1:8000/api/health
```

主要接口：

- `POST /api/route`：只执行输入检查与路由。
- `POST /api/workflow`：执行完整 Agent 工作流。
- `GET /api/inventory`：查看本地资产清单。

工作流请求示例：

```json
{
  "query": "阳朔和张家界哪个更适合看山水风景？",
  "allow_global_search": false
}
```

### 3. 启动前端

```powershell
Set-Location frontend
npm.cmd ci
npm.cmd run dev
```

浏览器访问 `http://127.0.0.1:5173/`。本地开发默认通过 Vite 将 `/api`
代理到 `http://127.0.0.1:8000`。

### 4. CLI

```powershell
.\.venv\Scripts\python.exe src\travelmind\cli.py --query "香港迪士尼怎么玩？"
.\.venv\Scripts\python.exe src\travelmind\cli.py --workflow --query "大理到双廊怎么去？"
```

## 默认运行模式

默认 profile 采用本地检索和确定性模板：

- Router 使用规则护栏。
- Naive 与 Multimodal 在证据充分时使用模板组织答案，因此响应速度较快。
- Grade、Rewrite、Generate 仅在对应 LLM 开关开启且配置可用时调用远程模型。
- GraphRAG Official Local Search 需要独立的 chat/embedding 配置。
- GraphRAG Global Search 默认关闭，普通 smoke 不会触发付费调用。

这套默认值用于降低演示成本并保持可重复性，不代表系统缺少 LLM 链路。

## 推荐演示问题

| 问题 | 预期链路 |
| --- | --- |
| 贵州荔波小七孔怎么玩比较合适？ | NaiveTravelAgent |
| 大理到双廊怎么去？ | NaiveTravelAgent |
| 香港迪士尼怎么玩？ | MultimodalTravelAgent |
| 阳朔和张家界哪个更适合看山水风景？ | GraphRAGAgent |
| 香港和成都哪个更适合亲子游？ | HybridAggregator |

## 验证

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m compileall -q src tests scripts
.\.venv\Scripts\python.exe scripts\smoke_full_agentic_rag.py --strict
npm.cmd --prefix frontend run test:contract
npm.cmd --prefix frontend run build
```

HTTP smoke 需要先启动安全后端：

```powershell
.\.venv\Scripts\python.exe scripts\serve_http_smoke.py --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe scripts\smoke_api_agentic_rag.py --base-url http://127.0.0.1:8000
```

更多结果与限制见 [评测说明](docs/evaluation.md)。

## 文档

- [架构说明](docs/architecture.md)
- [配置说明](docs/configuration.md)
- [评测与能力边界](docs/evaluation.md)
- [Demo 资产说明](docs/assets.md)

## Roadmap

以下内容是后续工程计划，不代表当前版本已经完成：

- **提升 GraphRAG 知识图谱完整度**：补齐当前数据中覆盖较弱的目的地、实体关系和
  community reports，建立实体覆盖率、关系密度、来源可追溯率等离线评估指标。
- **细化 GraphRAG 图谱粒度**：区分省、市、景区、景点、交通节点、住宿与主题标签，
  改进别名归一、层级关系和跨目的地比较所需的社区划分。
- **重建低成本本地检索**：基于规范化数据重新构建 GraphRAG Local Search 所需的
  Parquet 与 LanceDB，并评估本地向量检索、关键词检索和图邻域扩展的组合策略。
- **增强通用检索质量**：加入查询分解、混合召回、元数据过滤、交叉编码器重排和
  统一置信度校准，避免仅凭向量分数判断证据是否可回答。
- **完善 Multimodal 索引元数据**：重新构建离线 Markdown 向量索引，为每个 chunk
  保存地区、来源 PDF、标题、页码与主题，减少区域错配并提升引用可解释性。
- **完善 HybridAggregator**：当前默认环境下，GraphRAG 远程配置不可用时，Hybrid
  往往只能采用 Multimodal 离线 Markdown 的有效召回证据。后续将补齐稳定的大陆
  GraphRAG/Naive 候选分支、分支级查询分解、统一证据协议、跨源重排与有引用的综合回答。
- **建立持续评测集**：扩充单目的地、跨目的地、港澳台、大陆与港澳台混合问题，
  分别跟踪路由准确率、实体覆盖率、证据有效率、回答忠实度、延迟与远程调用成本。

## License 与 Demo 资产

源代码采用 [MIT License](LICENSE)。

**MIT License 仅适用于源代码。Demo 资产仅用于复现本项目演示，不会因包含在本仓库中而自动按照 MIT License 重新授权。**

数据、PDF、离线 Markdown、向量索引和 GraphRAG 产物的具体说明见
[Demo 资产说明](docs/assets.md)。正式公开发布前应再次确认这些资产的来源与再分发许可。
