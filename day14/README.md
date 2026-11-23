# Day 14 RAG Evaluation

该项目复用 `day11_rag` 的食品知识库、向量检索和 LLM 调用，记录每次 RAG
运行的 Retrieved Context、Answer 与 Source，并计算：

- Retrieval Hit@3
- Evidence Recall@3（多证据覆盖率）
- MRR
- Answer Correctness
- Abstention Accuracy

运行：

```powershell
python day14/main.py
```

## LLM API 配置

RAG 回答和 Ragas Judge 共用 `day10/llm_client.py` 中的通用
OpenAI-compatible 客户端。推荐在项目根目录 `.env` 中统一配置：

```env
LLM_API_KEY=你的APIKey
LLM_MODEL_NAME=模型ID或推理接入点ID
LLM_BASE_URL=服务商提供的OpenAI兼容地址
```

当前仍兼容 `DOUBAO_API_KEY`、`DOUBAO_MODEL_ID` 和 `DOUBAO_BASE_URL`。
如果没有设置 `DOUBAO_BASE_URL`，豆包默认使用
`https://ark.cn-beijing.volces.com/api/v3`。以后切换供应商时只需要修改上述
三个 `LLM_*` 变量，不需要修改 Python 代码。可参考项目根目录的 `.env.example`。

为避免服务异常时长时间无响应，还可以在 `.env` 调整：

```env
LLM_TIMEOUT_SECONDS=60
LLM_CLIENT_MAX_RETRIES=0
RAGAS_METRIC_TIMEOUT_SECONDS=120
```

Ragas 会在每个指标开始、完成、失败或超时时实时打印进度。单项超时不会终止
整批 Benchmark，其分数会记为 `N/A`，错误原因写入 `evaluation_errors`。

`Faithfulness` 和 `Answer Correctness` 会为一条样本发起多轮 Judge 请求。若生成
模型较慢，建议为自动评测单独配置一个速度更快、费用更低的接入点：

```env
RAGAS_API_KEY=Judge使用的Key
RAGAS_MODEL_NAME=更快的模型ID或接入点ID
RAGAS_BASE_URL=Judge服务的OpenAI兼容地址
```

这三个变量都是可选的；没有设置时，Ragas 自动复用 `LLM_*` 配置。

Answer Correctness 使用 `expected_keywords` 判断答案是否覆盖关键事实；
Abstention Accuracy 判断系统的拒答行为是否与 `answerable` 标注一致。

## Benchmark 类型

当前数据覆盖基础问题、同义改写、多证据问题、模糊问题和不可回答问题。
`case_type` 只用于标记测试类型，不参与分数计算。

完整 Benchmark 共 50 条：

- 简单事实：20 条
- 同义改写：10 条
- 多证据：10 条
- 模糊 Query：5 条
- 不可回答：5 条

Hit@3 表示前三条结果是否至少命中一条相关文档；Evidence Recall@3 表示前三条结果
找回了多少比例的相关文档。例如一个问题需要两个文档，只找回一个时，Hit@3 为 1，
Evidence Recall@3 为 50%。

## Ground Truth 审核清单

1. 先只阅读知识库，不参考模型回答，确认问题能否由库内证据回答。
2. `ground_truth_answer` 只能包含相关文档明确支持的事实，不能补充常识。
3. `relevant_document_ids` 应覆盖回答所需的全部、且仅包含必要的文档。
4. `expected_keywords` 应选择不可缺少的事实词，不要使用“应该”“通常”等常见词。
5. 对同义改写题，改写问法但保持问题意图和 Ground Truth 不变。
6. 对模糊题，只有存在唯一合理解释时才标为可回答；信息不足或存在多种冲突解释时标为不可回答。
7. 由第二位审核者独立复核；有分歧时回到原文逐项记录“证据支持/不支持”。

## 完整自动评测

项目中的 `run_full_evaluation.py` 会先运行现有 RAG，再结合手工检索指标和
Ragas 自动指标生成完整评测表：

```powershell
python day14/run_full_evaluation.py
```

建议先用一条数据验证 API 和 Ragas Judge：

```powershell
python day14/run_full_evaluation.py --limit 1
```

只验证 RAG 记录与手工指标，不调用 Ragas Judge：

```powershell
python day14/run_full_evaluation.py --limit 3 --skip-ragas
```

输出保存在 `day14/results/`：

- `rag_records.json`：每条问题的 response、retrieved contexts 和 sources
- `evaluation_results.json`：完整逐 Case 评测记录
- `evaluation_results.csv`：可直接用 Excel 打开的模型评测表
- `evaluation_summary.json`：总体及各问题类型的平均指标

最终表包含 Retrieval Hit@3、MRR、Answer Correctness、Faithfulness、
Answer Relevance、Context Precision、Context Recall 和 Abstention Accuracy。
Hit@3、MRR、Context Precision、Context Recall 基于人工相关文档 ID 在本地
计算；Abstention Accuracy 是本地规则指标；Answer Correctness、Faithfulness
和 Answer Relevance 由 Ragas Judge 评测。不可回答问题没有参考答案，因此其
Ragas 指标记为 `N/A`，主要查看 Abstention Accuracy。

程序会逐条保存 `rag_records.json`。再次运行时会复用问题文本未变化的已有
RAG 结果，避免重复消耗生成额度；传入 `--force-rag` 才会重新生成全部回答。
Ragas 分数同样会逐条保存并默认复用；需要重新评分时传入 `--force-ragas`。

缓存分为三层：

- `rag_records.json`：复用已经生成的 response、contexts 和 sources
- `ragas_score_cache.json`：按 Case、按单个指标即时保存分数
- `ragas_prompt_cache/`：复用 Ragas 内部相同 Prompt 的模型响应

缓存指纹包含 Question、Ground Truth、Response、Contexts 和 Judge 模型名；
这些内容变化后不会误用旧的 Ragas 分数。遇到 429/Quota 错误时，本次运行会
立即停止后续 Judge 请求，但仍会用本地指标生成评测表。额度恢复后直接重新运行
原命令即可从缺失指标继续，不要添加 `--force-rag` 或 `--force-ragas`。

当前 API 调用结构：

- Embedding：本地 `BAAI/bge-small-zh-v1.5`，文档与 Query 向量化不消耗 API
- RAG Answer：每个尚未缓存的 Case 调用一次远程 LLM
- Context Precision / Recall：使用人工标注 Document ID 本地计算
- Ragas：只远程评测 Answer Correctness、Faithfulness、Answer Relevance

相比五项指标全部交给 Judge，Context 两项改为本地计算后，每条可回答 Case 的
典型 Judge 调用量由约 10 次降到约 6 次。重复运行时，Case 分数缓存和 Prompt
缓存会进一步避免已经成功完成的调用。
