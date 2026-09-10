"""不依赖文本切分框架的长文本语义检索示例。"""

import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
TOP_K = 1
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


DOCUMENTS = [
    {
        "document_id": "doc_001",
        "title": "RAG 系统基础",
        "category": "大语言模型",
        "source": "人工智能课程讲义",
        "text": (
            "检索增强生成通常简称为RAG，它把外部知识检索与大语言模型生成结合起来。"
            "系统收到用户问题后，首先把问题转换成向量，再从知识库中找出语义相近的文档片段。"
            "这些片段会作为Context和用户问题一起发送给大语言模型。"
            "模型应当根据检索证据组织答案，而不是只依赖训练阶段记住的知识。"
            "为了便于验证，回答通常还要标注文档编号、标题或原始来源。"
            "如果检索结果与问题无关，系统应该拒绝回答或明确表示证据不足。"
            "RAG的效果同时受到文档质量、切分方式、Embedding模型和召回数量的影响。"
        ),
    },
    {
        "document_id": "doc_002",
        "title": "长文档切分方法",
        "category": "文本处理",
        "source": "语义检索实践手册",
        "text": (
            "长文档通常不能直接作为一个整体进行向量检索，因为过长的文本会混合多个主题。"
            "常见做法是先按照段落或句子边界把文档切成多个chunk，并限制每个chunk的最大长度。"
            "相邻chunk之间可以保留一定重叠内容，避免关键信息刚好落在切分边界而丢失语义。"
            "chunk过小会造成上下文不完整，chunk过大则可能引入无关内容并降低检索精度。"
            "切分完成后，需要为每个chunk记录原始文档编号、段落序号、字符起止位置和来源。"
            "这些Metadata能够帮助系统在返回答案时追踪事实出处，也方便开发者排查召回错误。"
        ),
    },
    {
        "document_id": "doc_003",
        "title": "Embedding 与 FAISS 检索",
        "category": "向量检索",
        "source": "向量数据库学习笔记",
        "text": (
            "Embedding模型可以把中文句子转换成高维数值向量，语义相近的文本通常具有更接近的向量表示。"
            "文档向量和查询向量使用同一个模型生成，才能放在相同的向量空间中比较。"
            "当向量已经归一化时，向量内积可以用于计算余弦相似度。"
            "FAISS提供了高效的向量索引和近邻搜索能力，IndexFlatIP会执行精确的内积检索。"
            "调用search时会返回相似度分数和向量位置，程序再根据位置找到对应的原始chunk。"
            "Top-K控制返回结果数量，K太小可能漏掉证据，K太大则会把更多无关片段放入Context。"
        ),
    },
    {
        "document_id": "doc_004",
        "title": "机器学习模型评估",
        "category": "机器学习",
        "source": "模型评估课程资料",
        "text": (
            "机器学习模型不能只观察训练集准确率，还需要在独立验证集或测试集上评估泛化能力。"
            "分类任务常见指标包括准确率、精确率、召回率和F1分数。"
            "准确率表示全部样本中预测正确的比例，但在类别极不平衡时可能产生误导。"
            "精确率关注被预测为正类的样本中有多少是真的正类，召回率关注全部真实正类中有多少被找到。"
            "F1分数是精确率与召回率的调和平均值，适合同时关注两者的场景。"
            "评估报告还应记录数据范围、指标定义和实验参数，确保不同实验结果能够公平比较。"
        ),
    },
]


# 每条查询标注一个应当召回的 (document_id, chunk_index)。
BENCHMARKS = [
    {
        "question_type": "明显问题",
        "query": "RAG是什么技术？",
        "relevant_chunks": [("doc_001", 1)],
    },
    {
        "question_type": "明显问题",
        "query": "为什么长文档不适合直接作为一个整体进行向量检索？",
        "relevant_chunks": [("doc_002", 1)],
    },
    {
        "question_type": "明显问题",
        "query": "FAISS中的IndexFlatIP执行什么检索？",
        "relevant_chunks": [("doc_003", 1)],
    },
    {
        "question_type": "明显问题",
        "query": "分类任务常见的评估指标有哪些？",
        "relevant_chunks": [("doc_004", 1)],
    },
    {
        "question_type": "同义改写",
        "query": "把资料切成小段时，为什么相邻小段要保留一些重复文字？",
        "relevant_chunks": [("doc_002", 1)],
    },
    {
        "question_type": "同义改写",
        "query": "哪个分类指标能够同时兼顾查准率和查全率？",
        "relevant_chunks": [("doc_004", 2)],
    },
    {
        "question_type": "跨chunk边界问题",
        "query": "RAG收到问题后如何检索和生成答案，检索结果无关时又应该怎么处理？",
        "relevant_chunks": [("doc_001", 1), ("doc_001", 2)],
    },
    {
        "question_type": "跨chunk边界问题",
        "query": "文档切分如何避免边界信息丢失，切分后的Metadata又如何帮助排查召回错误？",
        "relevant_chunks": [("doc_002", 1), ("doc_002", 3)],
    },
    {
        "question_type": "模糊问题",
        "query": "怎样才能让RAG的效果更好？",
        "relevant_chunks": [("doc_001", 2)],
    },
    {
        "question_type": "模糊问题",
        "query": "K是不是设置得越大越好？",
        "relevant_chunks": [("doc_003", 2)],
    },
    {
        "question_type": "困难问题",
        "query": "为什么归一化后的Embedding适合使用IndexFlatIP完成余弦相似度检索？",
        "relevant_chunks": [("doc_003", 1)],
    },
    {
        "question_type": "困难问题",
        "query": "类别严重不平衡时准确率为什么可能误导，应如何结合精确率、召回率和F1进行评估？",
        "relevant_chunks": [("doc_004", 1), ("doc_004", 2)],
    },
    {
        "question_type": "常规问题",
        "query": "每个chunk需要保存哪些来源Metadata？",
        "relevant_chunks": [("doc_002", 2)],
    },
    {
        "question_type": "常规问题",
        "query": "为什么文档向量和查询向量必须由同一个模型生成？",
        "relevant_chunks": [("doc_003", 1)],
    },
    {
        "question_type": "常规问题",
        "query": "RAG回答为什么要标注文档编号、标题或原始来源？",
        "relevant_chunks": [("doc_001", 2)],
    },
]


def _find_chunk_end(text, start, chunk_size):
    """在长度上限前优先寻找最近的中文句子边界。"""
    maximum_end = min(start + chunk_size, len(text))
    if maximum_end == len(text):
        return maximum_end

    search_start = start + chunk_size // 2
    candidate_text = text[search_start:maximum_end]
    matches = list(re.finditer(r"[。！？；\n]", candidate_text))
    if not matches:
        return maximum_end

    return search_start + matches[-1].end()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按字符长度切分文本，并尽量在句子边界结束。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = _find_chunk_end(text, start, chunk_size)
        raw_chunk = text[start:end]

        # 修正首尾空白，同时保持字符位置与原文一致。
        left_space = len(raw_chunk) - len(raw_chunk.lstrip())
        right_space = len(raw_chunk) - len(raw_chunk.rstrip())
        chunk_start = start + left_space
        chunk_end = end - right_space

        if chunk_start < chunk_end:
            chunks.append(
                {
                    "start_char": chunk_start,
                    "end_char": chunk_end,
                    "text": text[chunk_start:chunk_end],
                }
            )

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(documents):
    """切分全部文档，并为每个chunk补充来源Metadata。"""
    all_chunks = []

    for document in documents:
        document_chunks = chunk_text(document["text"])
        for chunk_index, chunk in enumerate(document_chunks, start=1):
            all_chunks.append(
                {
                    "document_id": document["document_id"],
                    "chunk_index": chunk_index,
                    "title": document["title"],
                    "category": document["category"],
                    "source": document["source"],
                    **chunk,
                }
            )

    return all_chunks


def build_index(chunks):
    """生成chunk向量并建立FAISS内积索引。"""
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index


def search(query, model, index, chunks, top_k=TOP_K):
    """检索与用户问题语义最相近的Top-K个chunk。"""
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    result_count = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, result_count)

    results = []
    for rank, (chunk_position, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        results.append(
            {
                "rank": rank,
                "score": float(score),
                **chunks[chunk_position],
            }
        )
    return results


def print_results(results):
    """输出检索结果及完整来源信息。"""
    print(f"\n=== Top {len(results)} Chunks ===")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['document_id']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(f"Source: {result['source']}")
        print(f"Start Char: {result['start_char']}")
        print(f"End Char: {result['end_char']}")
        print(f"Text: {result['text']}")
  
        


def calculate_recall(results, relevant_chunks, k):
    """计算前K个结果覆盖了多少人工标注的相关chunk。"""
    retrieved_chunks = {
        (result["document_id"], result["chunk_index"])
        for result in results[:k]
    }
    hit_count = sum(
        1
        for relevant_chunk in relevant_chunks
        if relevant_chunk in retrieved_chunks
    )
    return hit_count / len(relevant_chunks) if relevant_chunks else 0.0


def evaluate_retrieval(model, index, chunks):
    """运行Benchmark并打印宏平均Recall@1和Recall@3。"""
    total_recall_at_1 = 0.0
    total_recall_at_3 = 0.0

    print("\n=== Retrieval Evaluation ===")
    for benchmark_number, benchmark in enumerate(BENCHMARKS, start=1):
        results = search(
            benchmark["query"],
            model,
            index,
            chunks,
            top_k=3,
        )
        recall_at_1 = calculate_recall(
            results,
            benchmark["relevant_chunks"],
            k=1,
        )
        recall_at_3 = calculate_recall(
            results,
            benchmark["relevant_chunks"],
            k=3,
        )
        total_recall_at_1 += recall_at_1
        total_recall_at_3 += recall_at_3

        print(
            f"\nBenchmark {benchmark_number} "
            f"[{benchmark['question_type']}]: {benchmark['query']}"
        )
        print(f"Recall@1: {recall_at_1:.2%}")
        print(f"Recall@3: {recall_at_3:.2%}")

    benchmark_count = len(BENCHMARKS)
    average_recall_at_1 = total_recall_at_1 / benchmark_count
    average_recall_at_3 = total_recall_at_3 / benchmark_count

    print("\n=== Final Evaluation ===")
    print(f"Recall@1: {average_recall_at_1:.2%}")
    print(f"Recall@3: {average_recall_at_3:.2%}")


def main():
    chunks = build_chunks(DOCUMENTS)
    model, index = build_index(chunks)
    print("Document 数量:", len(DOCUMENTS))
    print("Chunk 数量:", len(chunks))
    query = input("请输入检索问题：").strip()
    if not query:
        print("输入无效：问题不能为空。")
        return

    results = search(query, model, index, chunks)
    print_results(results)
    evaluate_retrieval(model, index, chunks)


if __name__ == "__main__":
    main()
