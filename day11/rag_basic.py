import faiss

from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 160
CHUNK_OVERLAP = 40
TOP_K = 3


# =========================================================
# 1. 教学用长文本
# 后续项目替换成真实法规数据
# =========================================================

documents = [
    {
        "document_id": "food_001",
        "title": "冷冻食品知识",
        "source": "教学知识库",
        "category": "食品储存",
        "text": (
            "冷冻可以通过降低温度减缓微生物生长和食品内部的化学反应。"
            "冷冻食品在储存过程中应尽量维持稳定的低温环境。"
            "教学示例中将零下18摄氏度或更低作为典型冷冻储存条件。"
            "储存过程中如果出现明显温度波动，可能导致食品品质下降。"
            "包装完整性、储存时间以及运输过程中的温度控制也会影响最终品质。"
        )
    },
    {
        "document_id": "food_002",
        "title": "发酵食品知识",
        "source": "教学知识库",
        "category": "食品加工",
        "text": (
            "酸奶通常通过乳酸菌发酵牛奶或其他乳原料制成。"
            "发酵过程中乳酸菌利用部分乳糖并产生乳酸，使体系酸度升高。"
            "酸度变化会影响乳蛋白结构，从而形成酸奶特有的质地和风味。"
            "实际生产还需要控制原料质量、菌种、发酵时间和温度等条件。"
            "生产完成后通常还需要适当冷藏，以维持产品质量。"
        )
    },
    {
        "document_id": "food_003",
        "title": "食品添加剂知识",
        "source": "教学知识库",
        "category": "食品安全",
        "text": (
            "食品添加剂是在食品生产加工中按照特定目的使用的一类物质。"
            "实际使用时应依据适用的食品安全标准判断允许使用的品种、"
            "使用范围和使用量。不同食品类别适用的要求可能不同。"
            "因此不能只依据添加剂名称判断是否可以使用，"
            "还应同时核对食品类别、使用目的和具体标准条款。"
            "真实法规问答应引用正式标准原文，而不是只依赖教学知识文本。"
        )
    }
]


# =========================================================
# 2. Chunking
# =========================================================

def chunk_text(
    text,
    chunk_size,
    overlap
):

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size必须大于0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap不能小于0"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap必须小于chunk_size"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append({
                "text": chunk,
                "start_char": start,
                "end_char": end
            })

        if end == len(text):
            break

        start = (
            end - overlap
        )

    return chunks


# =========================================================
# 3. Documents → Chunks
# =========================================================

def build_chunks(
    documents,
    chunk_size,
    overlap
):

    all_chunks = []

    for document in documents:

        pieces = chunk_text(
            document["text"],
            chunk_size,
            overlap
        )

        for index, piece in enumerate(
            pieces
        ):

            all_chunks.append({
                "chunk_id": (
                    f"{document['document_id']}"
                    f"_chunk_{index:03d}"
                ),
                "document_id": document[
                    "document_id"
                ],
                "title": document[
                    "title"
                ],
                "source": document[
                    "source"
                ],
                "category": document[
                    "category"
                ],
                "start_char": piece[
                    "start_char"
                ],
                "end_char": piece[
                    "end_char"
                ],
                "text": piece[
                    "text"
                ]
            })

    return all_chunks


chunks = build_chunks(
    documents,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# =========================================================
# 4. Chunk Embeddings
# =========================================================

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)


chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


chunk_embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")


# =========================================================
# 5. FAISS Index
# =========================================================

embedding_dim = (
    chunk_embeddings.shape[1]
)

index = faiss.IndexFlatIP(
    embedding_dim
)

index.add(
    chunk_embeddings
)


# =========================================================
# 6. Chunk Search
# =========================================================

def search_chunks(
    query,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(
        top_k,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for rank, chunk_index in enumerate(
        indices[0],
        start=1
    ):

        if chunk_index < 0:
            continue

        chunk = chunks[
            int(chunk_index)
        ]

        results.append({
            "rank": rank,
            "score": float(
                scores[0][rank - 1]
            ),
            **chunk
        })

    return results


# =========================================================
# 7. Test
# =========================================================

if __name__ == "__main__":

    print(
        "Document Count:",
        len(documents)
    )

    print(
        "Chunk Count:",
        len(chunks)
    )

    print(
        "Chunk Embeddings:",
        chunk_embeddings.shape
    )

    print(
        "Index Total:",
        index.ntotal
    )


    query = (
        "食品添加剂使用时"
        "需要考虑哪些要求？"
    )


    results = search_chunks(
        query,
        top_k=3
    )


    print(
        "\nQuery:",
        query
    )


    for result in results:

        print(
            f"\nRank: "
            f"{result['rank']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            "Chunk ID:",
            result["chunk_id"]
        )

        print(
            "Document:",
            result["document_id"]
        )

        print(
            "Range:",
            result["start_char"],
            "-",
            result["end_char"]
        )

        print(
            "Text:",
            result["text"]
        )