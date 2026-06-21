# Demo 资产说明

本项目随仓库提供预构建 Demo 资产，用于复现各条检索链路。它们不是本项目声明自研的
数据集。

## 资产清单

| 路径 | 用途 | 运行依赖 |
| --- | --- | --- |
| `assets/travel_guide.csv` | 大陆旅游结构化知识 | Naive CSV fallback |
| `assets/faiss_index/` | 大陆旅游向量索引 | Naive FAISS |
| `assets/gang_ao_pdf/` | 港澳台旅游原始 PDF | 资产核验与离线处理来源 |
| `assets/result_markdown/` | PDF 经离线模型处理后形成的 Markdown 文档及向量索引 | Multimodal |
| `assets/graphrag_output/` | Parquet、community reports 与 LanceDB | GraphRAG Local/Global |
| `assets/graphrag_runtime/` | GraphRAG 运行配置 | Official Search adapter |

Multimodal 在线问答只消费已经生成的 Markdown 和向量索引，不在请求过程中重新运行
OCR 或视觉模型。

## Git LFS

PDF、FAISS、PKL、Parquet 和 LanceDB 大文件由 Git LFS 管理。首次 clone 后运行：

```bash
git lfs install
git lfs pull
git lfs ls-files
```

若只获得几十或几百字节的 pointer 文件，说明 LFS 对象尚未下载完成。仓库约包含
201 MB Demo 资产，托管和下载前需要检查平台的 LFS 配额。

## 来源与许可边界

这些资产用于项目 Demo 的检索与复现。当前仓库不将它们表述为自研数据集；具体来源、
作者权利和再分发许可仍需项目所有者在正式公开发布前复核。

**MIT License applies to source code only. Demo assets are provided for
reproduction of this project demo and are not automatically relicensed under
MIT.**

中文说明：MIT License 仅适用于源代码。Demo 资产仅用于复现本项目演示，不会因包含
在本仓库中而自动按照 MIT License 重新授权。
