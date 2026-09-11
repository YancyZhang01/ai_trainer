"""不依赖第三方分词包的中英文 Tokenizer。"""

import re


def tokenize(text):
    """英文按单词切分，中文使用二元字串，适合小型 BM25 示例。"""
    normalized_text = text.lower()
    tokens = re.findall(r"[a-z0-9_@.-]+", normalized_text)

    for chinese_text in re.findall(r"[\u4e00-\u9fff]+", normalized_text):
        if len(chinese_text) == 1:
            tokens.append(chinese_text)
        else:
            tokens.extend(
                chinese_text[index:index + 2]
                for index in range(len(chinese_text) - 1)
            )

    return tokens
