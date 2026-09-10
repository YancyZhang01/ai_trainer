"""寻找 Chunk 的结束位置。"""

import re


def find_chunk_end(text, start, chunk_size):
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
