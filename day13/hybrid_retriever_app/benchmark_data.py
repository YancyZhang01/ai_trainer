"""覆盖三类 Query 的 Hybrid Retrieval Benchmark。"""

BENCHMARKS = [
    {
        "query_type": "精确关键词",
        "query": "IndexFlatIP执行什么检索？",
        "relevant_document_ids": ["doc_004"],
        "expected_evidence": "IndexFlatIP可以执行精确内积检索。",
    },
    {
        "query_type": "精确关键词",
        "query": "BM25中的词频和逆文档频率有什么作用？",
        "relevant_document_ids": ["doc_002"],
        "expected_evidence": "BM25根据词频、逆文档频率和文档长度计算相关性。",
    },
    {
        "query_type": "语义改写",
        "query": "怎样让机器在作答前先查阅参考材料，以免凭空编造？",
        "relevant_document_ids": ["doc_001"],
        "expected_evidence": "先检索外部资料再生成答案，可以减少只依赖内部记忆造成的错误。",
    },
    {
        "query_type": "语义改写",
        "query": "两种搜索方式的分值单位不同，怎样只参考名次合并列表？",
        "relevant_document_ids": ["doc_005"],
        "expected_evidence": "RRF根据多路结果的排名融合，不要求原始分数处于相同范围。",
    },
    {
        "query_type": "关键词与语义混合",
        "query": "Hybrid Retrieval怎样结合关键词匹配和同义表达提高稳定性？",
        "relevant_document_ids": ["doc_006"],
        "expected_evidence": "关键词路线匹配术语，向量路线补充同义表达。",
    },
    {
        "query_type": "关键词与语义混合",
        "query": "chunk overlap为什么能减少文档截断位置的信息丢失？",
        "relevant_document_ids": ["doc_007"],
        "expected_evidence": "相邻chunk保留重叠可以减少切分边界造成的语义丢失。",
    },
]
