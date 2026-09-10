"""为所有文档构建带 Metadata 的 Chunks。"""

from .chunk_text import chunk_text


def build_chunks(documents):
    """切分全部文档，并为每个 Chunk 补充来源 Metadata。"""
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
