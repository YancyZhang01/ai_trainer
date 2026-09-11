# Day 13 Hybrid Retriever

该项目同时执行 BM25 关键词检索和 SentenceTransformer + FAISS
向量检索，再使用 Reciprocal Rank Fusion（RRF）融合排名。

运行：

```powershell
python day13/main.py
```

运行三类 Retriever 的对比 Benchmark：

```powershell
python day13/benchmark.py
```

运行两阶段粗召回与 Reranker：

```powershell
python day13/rerank.py
```

首次运行会从 Hugging Face 下载约 1.1 GB 的 `BAAI/bge-reranker-base`，
后续运行会直接复用本地缓存。

推荐测试问题：

```text
IndexFlatIP执行什么检索？
怎样让模型先找资料再回答，而不是只依赖自己记住的内容？
```
