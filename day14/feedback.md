## Ragas

使用的 metrics：Answer Correctness、Faithfulness、Answer Relevance。Context Precision 和 Context Recall 使用人工标注的 `relevant_document_ids` 在本地计算，Embedding 使用本地 `BAAI/bge-small-zh-v1.5`。

Faithfulness：97.73%。说明绝大多数答案中的事实都能由 Retrieved Context 支持。但 `simple_12` 的答案实际上与 Context 一致，Faithfulness 却为 0，可能是 Judge 的单次误判，因此自动指标仍需要结合人工抽查。

Answer Relevance：84.65%。简单事实题为 93.56%，但同义改写题为 74.06%、模糊 Query 为 66.32%，说明问题表达越间接，模型答案与原问题的表面语义匹配越容易下降。

Context Precision：40.74%。系统固定返回 Top-3；多数单证据问题只有 1 个相关文档，因此通常为 `1/3`，多证据问题平均为 `2/3`。该结果说明相关证据都能召回，但同时带入了无关 Context。

Context Recall：100%。45 条可回答问题所标注的相关文档均出现在 Top-3 中，多证据问题也完整覆盖了所需证据。

Answer Correctness：92.84%。该平均值基于 44 条成功获得分数的可回答问题；`paraphrase_02` 的 Answer Correctness 因 120 秒超时缺失。

Abstention Accuracy：98.00%。5 条不可回答问题中正确拒答 4 条；`unanswerable_05`“这个食品应该怎么保存？”没有明确指代对象，模型仍给出了条件式答案，因此该条拒答失败。

遇到的 API/版本问题：Ragas 0.4.3 与当前 `langchain-community` 存在 `vertexai` 导入兼容问题，项目使用兼容层处理；API 曾出现额度不足和请求耗时过长。本次结果中 `paraphrase_02` 的 Answer Correctness、`multi_10` 的 Faithfulness 分别超过 120 秒而缺失。项目已使用 RAG 结果缓存、Ragas 分数缓存和 Prompt 缓存，额度恢复后可直接续跑，不应使用 `--force-rag` 或 `--force-ragas`。HF Hub 未配置 Token 的提示只是下载限速警告，本地模型加载成功后不影响评测。

## Retriever Comparison

| System | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| Dense（Day14 当前系统） | 100.00% | 100.00% | 1.0000 |
| Hybrid | 未在 Day14 运行 | 未在 Day14 运行 | 未在 Day14 运行 |
| Hybrid + Reranker | 未在 Day14 运行 | 未在 Day14 运行 | 未在 Day14 运行 |

说明：以上 Dense 指标基于 45 条可回答问题；不可回答问题没有相关文档 ID，不参与 Hit 和 MRR 统计。当前 Day14 结果只记录了 Dense Retriever，不能用 Day13 的结果代替同一批 50 条 Benchmark 的对照实验，因此 Hybrid 和 Hybrid + Reranker 暂不填写虚构分数。

## RAG Evaluation

| System | Answer Correct | Faithfulness | Answer Relevance |
|---|---:|---:|---:|
| Baseline（Day14 当前 RAG） | 92.84% | 97.73% | 84.65% |
| Improved | 未运行 | 未运行 | 未运行 |

说明：当前只有一套 RAG 配置的完整运行结果，尚无使用相同 Benchmark、相同 Judge 和相同评测配置生成的 Improved 对照结果，因此不能判断改进幅度。

## 结果分析

- 检索覆盖表现很好：Hit@1、Hit@3、MRR 和 Context Recall 都达到 100%。这表示正确文档不仅进入前三名，而且在所有可回答问题中都排在第一名。
- Context Precision 只有 40.74%，主要是固定返回 Top-3 导致单证据题额外携带两个无关文档。后续可尝试相似度阈值、动态 Top-K 或 Reranker，在保持 Recall 的同时减少噪声。
- 回答质量总体较高，但 Answer Relevance 明显低于 Faithfulness。系统更突出的问题不是“脱离证据编造”，而是面对同义改写和模糊表达时，回答与问题意图的贴合度不够稳定。
- 不可回答问题应加强拒答判断，尤其是包含“这个”“那个”等缺少指代对象的 Query。可以在调用 LLM 前加入查询完整性判断，或在 System Prompt 中明确要求信息不足时直接拒答。
- `simple_12` 的 Faithfulness=0 与答案、Context 的实际内容不一致，说明单次 LLM Judge 存在随机误判风险。重要结果应人工抽查，或采用多次评分/多 Judge 交叉验证。

## Bad Cases

### Bad Case 1：Faithfulness Judge 疑似误判

- Case ID：`simple_12`
- Query：花生属于哪类食品安全风险？
- Expected：花生是常见的食物过敏原；应命中 `food_005`。
- Retrieved IDs：`food_005`、`food_008`、`food_001`。
- Actual：回答为“花生属于常见的食物过敏原相关的食品安全风险。”Answer Correctness 为 98.42%，Answer Relevance 为 100%，但 Faithfulness 为 0。
- Failure Type：自动评测误判 / 指标异常。
- Root Cause：回答中的“食物过敏原”可以在 `food_005` 中直接找到，检索结果也将该文档排在 Top-1，因此该结果更可能是 LLM Judge 的随机性、事实切分或判定格式造成的误判，而不是 RAG 回答真的缺乏依据。
- Possible Fix：人工复核异常分数；对低分样本重复评测 2～3 次并取平均值或多数票；固定 Judge 模型和 Temperature；保存 Judge 原始判定理由，方便定位误判。

### Bad Case 2：Ragas 指标调用超时

- Case ID：`paraphrase_02`
- Query：想从水果中补充维C，可以选择文档提到的哪些水果？
- Expected：可以选择橙子、柚子等柑橘类水果；应命中 `food_002`。
- Retrieved IDs：`food_002`、`food_006`、`food_001`。
- Actual：回答为“橙子、柚子等柑橘类水果。”Faithfulness 为 100%，但 Answer Correctness 在 120 秒内没有完成，最终记录为 `null`。
- Failure Type：评测基础设施失败，不是检索或回答内容失败。
- Root Cause：Ragas 的 Answer Correctness 通常需要多轮 Judge 调用，远程 API 响应过慢、服务繁忙或限流导致单项评测超过 `RAGAS_METRIC_TIMEOUT_SECONDS=120`。
- Possible Fix：额度和服务恢复后直接续跑并复用已有缓存，不添加 `--force-rag` 或 `--force-ragas`；为 Ragas 配置更快、更便宜的独立 Judge；适当提高单指标超时时间；继续保留按 Case、按 Metric 的断点缓存。

### Bad Case 3：不可回答问题未正确拒答

- Case ID：`unanswerable_05`
- Query：这个食品应该怎么保存？
- Expected：问题中的“这个食品”缺少明确指代，知识库无法确定食品类别，应回答“没有查找到与问题相关的事实证据”。
- Retrieved IDs：`food_001`、`food_008`、`food_005`，但这些只是不同食品的通用储存或安全信息，不构成该问题的确定证据。
- Actual：模型没有拒答，而是分别给出了冷冻食品和开封后干燥食品的条件式保存建议，Abstention Accuracy 为 0。
- Failure Type：错误回答 / 拒答失败。
- Root Cause：向量检索即使面对信息不足的问题也会强制返回 Top-3；“食品”“保存”等宽泛词又与多个文档相似。模型看到这些 Context 后尝试拼接一个条件式答案，没有严格执行“证据不足就拒答”的约束。
- Possible Fix：加入最低相似度或 Top-1 与 Top-2 分差阈值；识别“这个”“那个”等缺少指代对象的 Query；要求证据必须唯一确定回答对象，否则拒答；可增加一个 Answerability 判定步骤，并把该 Case 固定为回归测试。

## Debug 复盘

### 1. LLM 服务迁移

- 问题：Gemini API 额度不足，需要切换到豆包等 OpenAI-compatible 服务。
- 现象：最初直接把 `Doubao-Seed-2.1-pro` 当作模型 ID，API 返回 `404 InvalidEndpointOrModel.NotFound`。
- 原因：服务商展示的模型名称不一定等于 API 所需的模型 ID 或推理接入点 ID；当前账号还必须拥有对应接入权限。
- 修改：把 API Key、模型 ID、Base URL 全部移到 `.env`，业务代码统一读取 `LLM_API_KEY`、`LLM_MODEL_NAME`、`LLM_BASE_URL`，并保留对旧 `DOUBAO_*` 配置的兼容。
- 收获：切换模型供应商时不应修改 RAG 业务代码，只需更新环境变量；同时必须区分“模型展示名称”和“API 接入点 ID”。

### 2. API 额度和调用成本

- 问题：运行 50 条 Benchmark 时 API 额度很快耗尽，并出现 `429`、服务繁忙和长时间等待。
- 原因：每条未缓存的 RAG Case 至少调用一次生成模型；Ragas 的 Faithfulness、Answer Correctness 等指标还可能为同一 Case 发起多轮 Judge 请求。
- 修改：Embedding 固定使用本地 `BAAI/bge-small-zh-v1.5`；Context Precision 和 Context Recall 改为基于人工相关文档 ID 的本地计算；远程 Ragas 只保留 Answer Correctness、Faithfulness 和 Answer Relevance。
- 收获：Evaluation 的成本不仅来自生成答案，还来自 LLM-as-a-Judge。能用确定性规则计算的指标，应优先放在本地计算。

### 3. 缓存与断点续跑

- 问题：评测运行到一半失败后，如果重新开始会重复调用已经成功的 API。
- 修改：增加三层缓存：`rag_records.json` 保存 RAG 回答和 Context，`ragas_score_cache.json` 按 Case 和 Metric 保存分数，`ragas_prompt_cache/` 复用相同 Judge Prompt 的响应。
- 修改：缓存指纹包含 Question、Ground Truth、Response、Contexts 和 Judge 模型，输入变化时不会错误复用旧分数；遇到额度错误后立即停止后续远程指标，但继续保存本地评测结果。
- 注意：正常续跑时不要使用 `--force-rag` 或 `--force-ragas`，否则会绕过缓存并重新消耗额度。
- 收获：长批次评测必须做到逐条保存、按指标保存和幂等续跑，不能只在全部完成后一次性写结果。

### 4. 缓存读取错误

- 问题：程序出现 `TypeError: 'NoneType' object is not iterable`。
- 原因：`load_json()` 的代码结构错误，缓存文件存在时函数没有正确返回解析结果，导致 `existing` 为 `None`。
- 修改：缓存不存在、JSON 损坏、内容为 `null` 或类型不符合预期时统一返回默认值；正常文件则返回解析后的对象。
- 收获：缓存本身也是外部输入，读取时必须做存在性、JSON 格式和数据类型三层校验。

### 5. Ragas 兼容和超时

- 问题：Ragas 0.4.3 与当前 `langchain-community` 存在 `vertexai` 导入差异；部分指标又会长时间停留在运行状态。
- 修改：增加最小兼容层；为每个指标增加开始、完成、失败日志和独立的 120 秒超时；单项失败记录到 `evaluation_errors`，不终止整个 Benchmark。
- 收获：自动评测框架本身也可能失败。必须区分“RAG 系统回答失败”和“Judge/评测基础设施失败”，不能把超时产生的 `null` 当作 0 分。

## 本次项目完善后的结构

```text
Benchmark（50 条，含 Ground Truth）
        ↓
本地 Dense Retrieval（SentenceTransformer + FAISS）
        ↓
Retrieved IDs / Contexts / Sources
        ↓
远程 LLM 生成 Answer
        ↓
本地指标：Hit@3、MRR、Context Precision、Context Recall、Abstention Accuracy
        ↓
Ragas Judge：Answer Correctness、Faithfulness、Answer Relevance
        ↓
逐 Case 缓存与 JSON / CSV / Summary 报告
```

## 今天最重要的收获

- RAG Evaluation 不能只看答案是否“像是正确的”，需要把检索、生成和拒答分别评估。
- Hit@3 高只说明至少命中一个相关文档；多证据问题还需要 Context Recall 来检查证据是否完整。
- Faithfulness 高不代表 Answer Relevance 一定高：答案可以完全基于 Context，但仍可能没有直接回应用户真正的问题。
- 自动指标不是绝对真值。Judge 会出现随机误判、超时和版本兼容问题，异常样本必须回到 Question、Ground Truth、Context 和 Answer 做人工审核。
- 一个可实际使用的批量评测程序，除了指标之外，还必须处理配置隔离、额度控制、缓存、断点续跑、错误记录和结果可追溯。

## 今天最难理解

- Ragas 指标背后不是简单的本地公式，其中一些指标会拆分事实并多次调用 Judge，因此耗时和额度消耗比预期更高。
- Context Precision 只有 40.74%，但 Hit@1 和 Context Recall 都是 100%，三者并不矛盾：相关文档排在第一名且全部被召回，但固定 Top-3 同时带入了额外无关文档。
- `simple_12` 展示了 LLM-as-a-Judge 的局限：即使证据、答案和 Ground Truth 基本一致，Judge 仍可能给出异常分数。

## 后续改进计划

1. 在相同 50 条 Benchmark 上运行 Dense、Hybrid、Hybrid + Reranker，补齐严格可比的检索对照数据。
2. 尝试相似度阈值或动态 Top-K，观察能否在保持 Context Recall=100% 的同时提高 Context Precision。
3. 增加独立的 Answerability 判断和指代不明检测，重点回归 `unanswerable_05`。
4. 对 Faithfulness=0 或各指标差异过大的样本自动标记为“需要人工复核”。
5. 为 Judge 保存原始解释，并对异常结果做重复评分，降低单次 Judge 随机性的影响。

## 自评分

| 能力 | 评分 | 说明 |
|---|---:|---|
| Benchmark 设计 | 8/10 | 已覆盖简单事实、同义改写、多证据、模糊和不可回答问题，但知识库规模仍较小。 |
| Retrieval Evaluation | 8/10 | 能区分 Hit@K、MRR、Context Precision 和 Context Recall，并理解多证据覆盖。 |
| RAG Evaluation | 8/10 | 已完成检索、生成、拒答和 Ragas 指标的联合评测。 |
| Ragas | 7/10 | 能接入和处理缓存、超时、兼容问题，但仍需深入理解各指标的 Prompt 和 Judge 过程。 |
| Failure Analysis | 8/10 | 能区分检索失败、生成失败、拒答失败和评测基础设施失败。 |
| API 与成本控制 | 8/10 | 已实现环境变量配置、本地 Embedding、分层缓存和额度错误快速停止。 |
| Python 工程化 | 8/10 | 已完成模块拆分、错误处理、断点续跑和 JSON/CSV 报告输出。 |
| 实验严谨性 | 7/10 | 当前只有 Dense 和单套 RAG 的完整结果，仍需补齐同 Benchmark 下的对照实验。 |