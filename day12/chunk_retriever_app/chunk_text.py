"""长文本切分。"""

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .find_chunk_end import find_chunk_end


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按字符长度切分文本，并尽量在句子边界结束。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = find_chunk_end(text, start, chunk_size)
        raw_chunk = text[start:end]

        # 去掉首尾空白，同时保持位置对应原始文档。
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
