"""Benchmark 问题及基于原文字符区间的证据标注。"""

from .build_evidence import build_evidence


BENCHMARKS = [
    {
        "question_type": "明显问题",
        "query": "RAG是什么技术？",
        "expected_evidences": [
            build_evidence(
                "doc_001",
                "检索增强生成通常简称为RAG，它把外部知识检索与大语言模型生成结合起来。",
            ),
        ],
    },
    {
        "question_type": "明显问题",
        "query": "为什么长文档不适合直接作为一个整体进行向量检索？",
        "expected_evidences": [
            build_evidence(
                "doc_002",
                "长文档通常不能直接作为一个整体进行向量检索，因为过长的文本会混合多个主题。",
            ),
        ],
    },
    {
        "question_type": "明显问题",
        "query": "FAISS中的IndexFlatIP执行什么检索？",
        "expected_evidences": [
            build_evidence(
                "doc_003",
                "FAISS提供了高效的向量索引和近邻搜索能力，IndexFlatIP会执行精确的内积检索。",
            ),
        ],
    },
    {
        "question_type": "明显问题",
        "query": "分类任务常见的评估指标有哪些？",
        "expected_evidences": [
            build_evidence(
                "doc_004",
                "分类任务常见指标包括准确率、精确率、召回率和F1分数。",
            ),
        ],
    },
    {
        "question_type": "同义改写",
        "query": "把资料切成小段时，为什么相邻小段要保留一些重复文字？",
        "expected_evidences": [
            build_evidence(
                "doc_002",
                "相邻chunk之间可以保留一定重叠内容，避免关键信息刚好落在切分边界而丢失语义。",
            ),
        ],
    },
    {
        "question_type": "同义改写",
        "query": "哪个分类指标能够同时兼顾查准率和查全率？",
        "expected_evidences": [
            build_evidence(
                "doc_004",
                "F1分数是精确率与召回率的调和平均值，适合同时关注两者的场景。",
            ),
        ],
    },
    {
        "question_type": "跨chunk边界问题",
        "query": "RAG收到问题后如何检索和生成答案，检索结果无关时又应该怎么处理？",
        "expected_evidences": [
            build_evidence(
                "doc_001",
                "系统收到用户问题后，首先把问题转换成向量，再从知识库中找出语义相近的文档片段。",
            ),
            build_evidence(
                "doc_001",
                "这些片段会作为Context和用户问题一起发送给大语言模型。",
            ),
            build_evidence(
                "doc_001",
                "如果检索结果与问题无关，系统应该拒绝回答或明确表示证据不足。",
            ),
        ],
    },
    {
        "question_type": "跨chunk边界问题",
        "query": "文档切分如何避免边界信息丢失，切分后的Metadata又如何帮助排查召回错误？",
        "expected_evidences": [
            build_evidence(
                "doc_002",
                "相邻chunk之间可以保留一定重叠内容，避免关键信息刚好落在切分边界而丢失语义。",
            ),
            build_evidence(
                "doc_002",
                "这些Metadata能够帮助系统在返回答案时追踪事实出处，也方便开发者排查召回错误。",
            ),
        ],
    },
    {
        "question_type": "模糊问题",
        "query": "怎样才能让RAG的效果更好？",
        "expected_evidences": [
            build_evidence(
                "doc_001",
                "RAG的效果同时受到文档质量、切分方式、Embedding模型和召回数量的影响。",
            ),
        ],
    },
    {
        "question_type": "模糊问题",
        "query": "K是不是设置得越大越好？",
        "expected_evidences": [
            build_evidence(
                "doc_003",
                "Top-K控制返回结果数量，K太小可能漏掉证据，K太大则会把更多无关片段放入Context。",
            ),
        ],
    },
    {
        "question_type": "困难问题",
        "query": "为什么归一化后的Embedding适合使用IndexFlatIP完成余弦相似度检索？",
        "expected_evidences": [
            build_evidence(
                "doc_003",
                "当向量已经归一化时，向量内积可以用于计算余弦相似度。",
            ),
            build_evidence(
                "doc_003",
                "FAISS提供了高效的向量索引和近邻搜索能力，IndexFlatIP会执行精确的内积检索。",
            ),
        ],
    },
    {
        "question_type": "困难问题",
        "query": "类别严重不平衡时准确率为什么可能误导，应如何结合精确率、召回率和F1进行评估？",
        "expected_evidences": [
            build_evidence(
                "doc_004",
                "准确率表示全部样本中预测正确的比例，但在类别极不平衡时可能产生误导。",
            ),
            build_evidence(
                "doc_004",
                "精确率关注被预测为正类的样本中有多少是真的正类，召回率关注全部真实正类中有多少被找到。",
            ),
            build_evidence(
                "doc_004",
                "F1分数是精确率与召回率的调和平均值，适合同时关注两者的场景。",
            ),
        ],
    },
    {
        "question_type": "常规问题",
        "query": "每个chunk需要保存哪些来源Metadata？",
        "expected_evidences": [
            build_evidence(
                "doc_002",
                "切分完成后，需要为每个chunk记录原始文档编号、段落序号、字符起止位置和来源。",
            ),
        ],
    },
    {
        "question_type": "常规问题",
        "query": "为什么文档向量和查询向量必须由同一个模型生成？",
        "expected_evidences": [
            build_evidence(
                "doc_003",
                "文档向量和查询向量使用同一个模型生成，才能放在相同的向量空间中比较。",
            ),
        ],
    },
    {
        "question_type": "常规问题",
        "query": "RAG回答为什么要标注文档编号、标题或原始来源？",
        "expected_evidences": [
            build_evidence(
                "doc_001",
                "为了便于验证，回答通常还要标注文档编号、标题或原始来源。",
            ),
        ],
    },
]
