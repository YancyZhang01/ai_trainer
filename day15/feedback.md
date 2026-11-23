# Day15

## Day14 补漏

Abstention Accuracy：严格按不可回答子集计算，即“不可回答问题中正确拒答的比例”。本次 5 条不可回答问题正确拒答 4 条，因此为 `4 / 5 = 80%`。

Answerability Accuracy：把每个问题看作“可回答/不可回答”二分类，判断系统的回答或拒答行为是否与 Benchmark 的 `answerable` 标签一致。本次 50 条中判断正确 49 条，因此为 `49 / 50 = 98%`。

为什么二者不同：二者的分母不同。Abstention Accuracy 只评估不可回答题，分母为 5；Answerability Accuracy 评估全部题目，分母为 50。当前代码中的 `calculate_abstention_accuracy()` 对每一条题都判断行为是否与 `answerable` 一致，所以汇总后的 overall `98%` 实际上更接近 Answerability Accuracy；不可回答分组中的 `80%` 才是严格口径的 Abstention Accuracy。

Pilot Benchmark：正式评测前使用的小规模代表性测试集，用来确认字段、Retriever、LLM、指标、缓存和报告链路能够运行。建议从每种题型抽取 1～2 条，先跑 5～10 条，避免一开始就消耗大量 API 额度。

Final Benchmark：冻结后用于正式对比的完整测试集。本项目为 50 条，包括 20 条简单事实、10 条同义改写、10 条多证据、5 条模糊 Query、5 条不可回答问题。正式对比不同系统时应使用同一份 Final Benchmark。

ID-based Context Precision：Top-K 检索结果中，`document_id` 命中人工标注 `relevant_document_ids` 的比例。本次 Top-3 平均为 `40.74%`。它较低的主要原因是单证据题固定返回 3 条文档，通常只有 1 条相关。

ID-based Context Recall：人工标注的全部相关文档中，有多少出现在 Top-K 检索结果里。本次 Top-3 为 `100%`，说明所有可回答问题所需的相关文档都被召回。

## Concept

API：Application Programming Interface，程序之间按照约定交换数据和调用功能的接口。

REST：一种面向资源设计 HTTP API 的风格，通常使用 URL 表示资源，并通过 GET、POST 等 HTTP 方法表达操作。

HTTP Request：客户端发给服务端的请求，通常包含请求方法、URL、Headers 和 Body。

HTTP Response：服务端返回给客户端的结果，通常包含状态码、Headers 和 Response Body。

GET：主要用于读取资源。Day15 的 `GET /health` 用于检查服务是否存活，不触发模型加载和 LLM 调用。

POST：主要用于提交数据或触发处理。Day15 的 `POST /retrieve` 执行检索，`POST /ask` 执行完整 RAG。

Pydantic：用于定义、校验和序列化 Python 数据模型。非法请求会在进入业务逻辑前被 FastAPI 拦截并返回 422。

Schema：API 数据结构契约，规定字段名、类型、是否必填和取值范围。例如 `QuestionRequest` 规定了 `question` 和 `top_k`。

Status Code：HTTP 对处理结果的标准化表达。本项目主要使用 200、422、500、502 和 503。

## Requirement V1

Goal：把已有食品知识 RAG 封装成 HTTP API，让其他程序提交问题并获得答案、来源、状态和耗时。

Endpoints：

- `GET /health`：服务存活检查。
- `POST /retrieve`：只调用 Retriever，不调用 LLM。
- `POST /ask`：调用 Retriever，判断证据，再调用 LLM 生成回答。

Input：`question: string`；`top_k: integer`，默认值为 3。

Output：

- `/ask`：`answer`、`sources`、`status`、`latency_ms`。
- `/retrieve`：`results`、`status`、`latency_ms`。
- `/health`：`status`。

Validation：`question` 去除首尾空白后不能为空，最大 1000 字符；`top_k` 必须为严格整数，默认 3，范围 1～10；不允许额外字段。

Error Cases：空问题、缺少问题或非法 `top_k` 返回 422；Retriever 普通异常返回 500；Retriever 依赖不可用返回 503；LLM 配置或调用失败返回 503；LLM 返回无效内容返回 502；不可回答问题返回 200，但业务状态为 `no_evidence`。

Constraints：使用 Python、FastAPI 和 Pydantic；API Key 从根目录 `.env` 读取；复用现有 Retriever 和公共 `LLMClient`；CPU 可运行；不使用 LangChain；返回 JSON；不重新实现 Retriever 或 LLM Client。

Acceptance：服务可以启动；`GET /health` 返回 `{"status": "ok"}`；合法的 `POST /ask` 返回答案和来源；非法输入在进入 RAG 前返回 422；依赖失败时返回明确错误而不是让进程崩溃。

## Requirement V2

加入技术约束：

1. Retriever 和 LLM 采用延迟初始化并缓存，避免每次请求重复加载模型、索引或客户端。
2. 同步的模型/API 调用通过 `run_in_threadpool` 执行，避免直接阻塞 FastAPI 的事件循环。
3. 对 Retriever、配置、LLM 调用和 LLM 响应分别处理异常，映射到明确的 HTTP 状态码，并在服务端保留原始日志。

## API Architecture

```text
Client / Swagger UI
        ↓ HTTP Request
FastAPI Router
        ↓
Pydantic Request Validation
        ↓
Existing Retriever（SentenceTransformer + FAISS）
        ↓
Relevant Evidence Check
        ↓ 有证据                         ↓ 无证据
Build RAG Prompt                      直接拒答
        ↓
shared.LLMClient → .env → Remote LLM API
        ↓
Pydantic Response Model
        ↓ HTTP JSON Response
Client
```

## Code Reading

FastAPI()：创建 ASGI 应用实例，并设置 API 标题、版本以及自动生成的 OpenAPI 文档。

`@app.get`：把 Python 函数注册为 GET 路由。例如 `@app.get("/health")`。

`@app.post`：把 Python 函数注册为 POST 路由。例如 `@app.post("/ask")`。

BaseModel：Pydantic 模型基类。继承后可以自动完成类型校验、JSON 解析和序列化。

Field：为字段添加约束和默认值。当前使用 `min_length`、`max_length`、`ge`、`le`、`strict` 和 `default`。

response_model：声明成功响应的 Schema。FastAPI 会校验、过滤并在 Swagger 中展示返回结构。

HTTPException：在业务处理中主动终止请求并返回指定 HTTP 状态码与错误详情。

uuid：用于生成全局唯一标识，适合为每次请求生成 `request_id` 并串联日志。当前 Day15 尚未使用，属于后续可选增强，不应把它误认为现有流程的一部分。

perf_counter：高精度单调计时器。代码在请求开始和结束各调用一次，差值乘以 1000 得到 `latency_ms`。

## API Contract

### POST /ask

Request：

```json
{
  "question": "酸奶是怎样制成的？",
  "top_k": 3
}
```

Response：

```json
{
  "answer": "酸奶是由乳酸菌发酵牛奶制成的乳制品。",
  "sources": [
    {
      "document_id": "food_003",
      "source": "食品知识库-加工",
      "score": 0.7838
    }
  ],
  "status": "ok",
  "latency_ms": 71430.53
}
```

可能状态码：

- `200`：成功回答，或证据不足并以 `status=no_evidence` 正常拒答。
- `422`：请求 Schema 校验失败。
- `500`：Retriever 或生成阶段出现未分类的内部异常。
- `502`：LLM 返回空内容或无效响应。
- `503`：Retriever 依赖、LLM 配置或远程 LLM 服务暂时不可用。

### GET /health

Response：

```json
{
  "status": "ok"
}
```

## Modification

我自己完成：

1. 使用 `python -m uvicorn day15.app:app --reload` 从项目根目录启动服务，并通过 Swagger UI 查看接口契约。
2. 构造正常、不可回答、空问题和非法 `top_k` 请求，通过响应状态码区分 Validation、Retriever 与 LLM 阶段。
3. 根据终端原始异常逐层排查 `torch_python.dll` 与 LLM Base URL，而不是只根据 Swagger 中经过封装的错误消息猜测根因。

## API Testing

| Case | Expected | Actual | Pass |
|---|---|---|---|
| Health | 200 | 200，`status=ok` | Yes |
| Normal | 200 | 200，`status=ok`，返回 answer 和 3 个 sources | Yes |
| Unknown | 200 abstained | 200，`status=no_evidence`，sources 为空 | Yes |
| Empty Question | 422 | 422，`string_too_short` | Yes |
| Invalid top_k | 422 | 422，`top_k <= 10` 校验失败 | Yes |
| LLM Failure | 503 | 503，`LLM 服务暂时不可用` | Yes |

## Latency

正常 Query：`酸奶是怎样制成的？`

耗时：`71430.53 ms`。该次包含测试进程首次加载本地模型、建立索引以及远程 LLM 调用，属于冷启动结果。

不可回答 Query：`法国总统是谁？`

耗时：`10.50 ms`。Retriever 判断没有相关证据后直接返回，没有调用 LLM；同时本地模型已经完成预热。

我的观察：两项耗时不能直接当作同条件性能对比，因为一个是冷启动并调用远程 LLM，另一个是热启动且提前拒答。应分别统计冷启动、热启动检索、LLM 网络调用，并对每类请求重复多次计算 P50/P95。

## Failure Analysis

### Case 1

Request：`POST /retrieve`，问题为“酸奶是怎样制成的？”，`top_k=3`。

Expected HTTP：200。

Actual HTTP：最初为 500，Response Body 只有“Retriever 执行失败”。

Failure Type：本地依赖加载失败。

Root Cause：`sentence_transformers → sklearn → scipy` 最终加载 PyTorch 的 `torch_python.dll` 时触发 `WinError 4551`。Code Integrity 中出现了与该 DLL 对应的 3077 阻止事件，说明 Windows 应用控制策略阻止了该文件；关闭相应限制后 `import torch`、索引构建和 `/retrieve` 均恢复。

Fix：先单独执行 `import torch` 定位依赖层，再检查同一时间的 Code Integrity 3077/3089，而不是依据无关的历史 3033 事件下结论；环境恢复后重启 Uvicorn，使服务进程重新加载依赖。

### Case 2

Request：`POST /ask`，问题为“酸奶是怎样制成的？”，`top_k=3`。

Expected HTTP：200。

Actual HTTP：503，`LLM 服务暂时不可用`。

Failure Type：LLM 配置错误。

Root Cause：`.env` 中 API Key 和模型名已存在，但 `LLM_BASE_URL` 仍是“OpenAI兼容API地址”占位文本，不是可请求的 URL。Retriever 因为完全本地运行可以成功，而 `/ask` 进入远程 LLM 阶段后失败。

Fix：将 `LLM_BASE_URL` 设置为火山方舟数据面地址 `https://ark.cn-beijing.volces.com/api/v3`；在 `shared/settings.py` 增加 HTTP(S) URL 格式校验；重启服务清除已缓存的旧客户端。最小 LLM 连通性测试返回 `OK`，端到端 `/ask` 随后返回 200。

## Debug

问题：`/retrieve` 已成功，但 `/ask` 返回 503。

Request：合法的 JSON，`question="酸奶是怎样制成的？"`，`top_k=3`。

Validation：通过；如果失败应该返回 422，因此不是 Pydantic 问题。

Retriever：通过；相同输入调用 `/retrieve` 已返回 Top-3 文档，因此本地 Embedding 和 FAISS 正常。

LLM：失败；服务端日志显示异常来自 `LLMClient.generate_text()`。

HTTP Status：503 Service Unavailable。

Raw Error：OpenAI-compatible Client 收到的 Base URL 不是合法远程 API 地址；修正后最小请求成功。

Expected：200，返回 answer、sources、status 和 latency_ms。

Actual：修复前 503；修复后 200，`status=ok`。

最终原因：`.env` 的 `LLM_BASE_URL` 使用了示例占位文字。错误发生在 LLM 阶段，与 Retriever、Pydantic 和 FastAPI 路由无关。

## 今天最重要的收获

- HTTP 状态码只是错误分类入口，真正定位仍需结合请求验证结果、分阶段测试和服务端原始 Traceback。
- `/retrieve` 与 `/ask` 的差异可以帮助快速划分故障边界：前者成功、后者失败时，应优先检查 Prompt、LLM Client 和远程 API 配置。
- 不可回答问题以 HTTP 200 加业务状态 `no_evidence` 返回，因为这是系统预期行为，不是服务器异常。

## 今天最难理解

- 需要区分 HTTP 成功与业务成功：HTTP 200 既可以表示正常回答，也可以表示系统正确拒答。
- 需要区分冷启动延迟和热启动延迟；首次模型加载不能代表每次请求的稳定耗时。
- 503 只表示依赖暂时不可用，具体是配置、权限、额度还是网络问题仍需查看原始异常。

## 自评分

REST/API：8/10  
FastAPI：8/10  
Pydantic：8/10  
Schema Design：8/10  
Status Code：8/10  
Error Handling：8/10  
API Testing：8/10  
Code Reading：8/10  
AI Coding：8/10  
Python：8/10  
Git：7/10
