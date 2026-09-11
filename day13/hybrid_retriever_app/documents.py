"""Hybrid Retriever 使用的中文文档库。"""

DOCUMENTS = [
    {
        "document_id": "doc_001",
        "title": "RAG 基础",
        "category": "大语言模型",
        "source": "人工智能课程讲义",
        "text": (
            "检索增强生成简称RAG。系统先从外部知识库检索与问题相关的资料，"
            "再把检索证据交给大语言模型生成答案，从而减少模型只依赖内部记忆造成的错误。"
        ),
    },
    {
        "document_id": "doc_002",
        "title": "BM25 关键词检索",
        "category": "信息检索",
        "source": "搜索技术学习笔记",
        "text": (
            "BM25是一种稀疏关键词检索算法。它根据词频、逆文档频率和文档长度计算相关性，"
            "特别适合包含产品名称、专业术语或明确关键词的查询。"
        ),
    },
    {
        "document_id": "doc_003",
        "title": "Dense Retrieval",
        "category": "向量检索",
        "source": "语义检索实践手册",
        "text": (
            "Dense Retrieval使用Embedding模型把查询和文档转换成向量。"
            "即使查询没有出现文档中的原始关键词，只要语义相近，也可能被向量检索召回。"
        ),
    },
    {
        "document_id": "doc_004",
        "title": "FAISS 向量索引",
        "category": "向量数据库",
        "source": "向量数据库学习笔记",
        "text": (
            "FAISS用于高效执行向量近邻搜索。向量归一化后，IndexFlatIP可以执行精确内积检索，"
            "此时内积可以用于比较余弦相似度。"
        ),
    },
    {
        "document_id": "doc_005",
        "title": "Reciprocal Rank Fusion",
        "category": "混合检索",
        "source": "搜索排序课程资料",
        "text": (
            "Reciprocal Rank Fusion简称RRF，它根据文档在多路检索结果中的排名计算融合分数。"
            "RRF不要求BM25分数和向量相似度处于相同数值范围。"
        ),
    },
    {
        "document_id": "doc_006",
        "title": "Hybrid Retrieval",
        "category": "混合检索",
        "source": "RAG工程实践指南",
        "text": (
            "Hybrid Retrieval同时使用关键词检索和向量语义检索。"
            "关键词路线保证精确术语匹配，向量路线补充同义表达，融合后通常比单一检索器更稳定。"
        ),
    },
    {
        "document_id": "doc_007",
        "title": "长文档切分",
        "category": "文本处理",
        "source": "语义检索实践手册",
        "text": (
            "长文档可以按照句子或段落切成多个chunk，相邻chunk保留适当重叠，"
            "能够减少重要信息落在切分边界时造成的语义丢失。"
        ),
    },
    {
        "document_id": "doc_008",
        "title": "检索效果评估",
        "category": "模型评估",
        "source": "评测课程资料",
        "text": (
            "检索系统可以使用Recall@K评估前K个结果是否覆盖人工标注的相关证据。"
            "评测数据应记录问题、来源文档、证据字符区间和期望证据文本。"
        ),
    },
]
