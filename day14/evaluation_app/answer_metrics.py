"""答案正确性与拒答指标。"""


def calculate_answer_correctness(answer, benchmark):
    """可回答题的答案包含全部期望关键词时返回 1。"""
    if not benchmark["answerable"]:
        return None
    if not answer:
        return 0.0

    return float(
        all(
            keyword in answer
            for keyword in benchmark["expected_keywords"]
        )
    )


def calculate_abstention_accuracy(abstained, answerable):
    """拒答行为与题目是否可回答一致时返回 1。"""
    expected_abstention = not answerable
    return float(abstained == expected_abstention)
