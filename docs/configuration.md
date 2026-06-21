# 配置说明

复制根目录 `.env.example` 为 `.env` 后按需填写。环境文件可能包含凭据，禁止提交。
前端仅使用 `frontend/.env.example` 中的公开 API 地址配置。

## 资产路径

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `TRAVELMIND_ASSETS_DIR` | `assets` | Demo 资产根目录 |
| `TRAVELMIND_TRAVEL_CSV` | `assets/travel_guide.csv` | Naive CSV 数据 |
| `TRAVELMIND_FAISS_INDEX_DIR` | `assets/faiss_index` | Naive FAISS 索引 |
| `TRAVELMIND_MULTIMODAL_MARKDOWN_DIR` | `assets/result_markdown` | 离线 Markdown 与索引 |
| `TRAVELMIND_GRAPHRAG_CONFIG_DIR` | `assets/graphrag_runtime` | GraphRAG 运行配置 |
| `TRAVELMIND_GRAPHRAG_OUTPUT_DIR` | `assets/graphrag_output` | Parquet 与 LanceDB |

相对路径按启动进程的工作目录解析。推荐从仓库根目录启动后端。

## 普通 TravelMind LLM

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `TRAVELMIND_RUNTIME_PROFILE` | 空 | 设置为 `full_agentic_demo` 时启用普通 Agentic LLM 阶段 |
| `TRAVELMIND_LLM_ENABLED` | `false` | 普通 LLM 总开关 |
| `TRAVELMIND_SYSTEM_AGENT_LLM_ROUTER_ENABLED` | `false` | LLM Router |
| `TRAVELMIND_LLM_GRADE_ENABLED` | `false` | LLM Grade |
| `TRAVELMIND_LLM_REWRITE_ENABLED` | `false` | LLM Rewrite |
| `TRAVELMIND_LLM_GENERATE_ENABLED` | `false` | 最终 LLM 回答生成 |
| `TRAVELMIND_NAIVE_AGENT_LLM_LOOP_ENABLED` | `false` | Naive 工具查询规划 |
| `TRAVELMIND_LLM_API_KEY` | 空 | OpenAI-compatible chat 凭据 |
| `TRAVELMIND_LLM_BASE_URL` | `https://api.deepseek.com` | Chat API 地址 |
| `TRAVELMIND_LLM_MODEL` | `deepseek-chat` | Chat 模型 |
| `TRAVELMIND_EMBEDDING_API_KEY` | 空 | FAISS 查询 embedding 凭据 |
| `TRAVELMIND_EMBEDDING_MODEL` | `text-embedding-v3` | Embedding 模型 |

`full_agentic_demo` 不会隐式开启 GraphRAG Global Search。

## GraphRAG

GraphRAG 使用独立配置，不与普通 TravelMind LLM 凭据混用。

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `TRAVELMIND_GRAPHRAG_LLM_API_KEY` | 空 | Official Local/Global 凭据 |
| `TRAVELMIND_GRAPHRAG_LLM_BASE_URL` | 空 | GraphRAG 模型服务地址 |
| `TRAVELMIND_GRAPHRAG_LLM_CHAT_MODEL` | 空 | GraphRAG chat 模型 |
| `TRAVELMIND_GRAPHRAG_LLM_EMBEDDING_MODEL` | 空 | GraphRAG embedding 模型 |
| `TRAVELMIND_GRAPHRAG_TIMEOUT_SECONDS` | `180` | 单次 GraphRAG 检索超时 |
| `TRAVELMIND_GRAPHRAG_MAX_CONTEXT_CHARS` | `6000` | 安全上下文长度 |
| `TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED` | `false` | Global 服务级许可 |
| `TRAVELMIND_HYBRID_BRANCH_TIMEOUT_SECONDS` | `20` | Hybrid 分支预算 |

Official Local Search 没有额外 ENABLE 开关。只要独立凭据、模型配置、Parquet 和
LanceDB 就绪，它就是 GraphRAG 默认正式链路。

Global Search 必须同时满足：

1. `TRAVELMIND_GRAPHRAG_GLOBAL_SEARCH_ENABLED=true`；
2. `/api/workflow` 请求包含 `"allow_global_search": true`；
3. 当前 route 为 `graphrag` 或 `hybrid_rag`；
4. key、模型、配置与索引均 ready。

前端复选框只改变请求级授权，不能绕过服务级开关。

## 日志

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `TRAVELMIND_RUN_LOG_ENABLED` | `false` | 将完整 API 请求结果写入本地运行目录 |

完整日志可能包含用户输入，默认关闭且不应提交。

## 前端

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `VITE_API_BASE_URL` | 空 | 空值时使用 Vite `/api` 代理 |

前端不提供服务级 Local/Global 配置，也不保存任何 API key。
