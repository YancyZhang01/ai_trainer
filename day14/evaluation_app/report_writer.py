"""将逐 Case 结果写入 JSON、CSV，并打印紧凑评测表。"""

import csv
import json


METRIC_COLUMNS = [
    "retrieval_hit_at_3",
    "mrr",
    "answer_correctness",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "abstention_accuracy",
]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path, default):
    """读取 JSON 缓存；缺失、损坏或类型错误时返回默认值。"""
    if not path.exists():
        return default

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default

    if not isinstance(data, type(default)):
        return default
    return data


def merge_by_case_id(existing_rows, updated_rows, case_order=None):
    """按 case_id 合并局部运行结果，不让 --limit 删除其他缓存。"""
    merged = {
        row["case_id"]: row
        for row in existing_rows
    }
    merged.update({row["case_id"]: row for row in updated_rows})

    if case_order is None:
        ordered_ids = list(merged)
    else:
        ordered_ids = [case_id for case_id in case_order if case_id in merged]
        ordered_ids.extend(
            case_id for case_id in merged if case_id not in ordered_ids
        )
    return [merged[case_id] for case_id in ordered_ids]


def save_csv(path, rows):
    """列表和字典字段使用 JSON 字符串保存，避免丢失上下文结构。"""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            serializable_row = {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(serializable_row)


def _average(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


def build_summary(rows):
    """同时汇总总体得分和不同 Case 类型的宏平均得分。"""
    groups = {"overall": rows}
    for case_type in sorted({row["case_type"] for row in rows}):
        groups[case_type] = [
            row for row in rows if row["case_type"] == case_type
        ]

    return {
        group_name: {
            "case_count": len(group_rows),
            **{
                metric: _average(group_rows, metric)
                for metric in METRIC_COLUMNS
            },
        }
        for group_name, group_rows in groups.items()
    }


def _format_score(value):
    return "N/A" if value is None else f"{value:.3f}"


def print_evaluation_table(rows):
    """控制台打印每条样本的八项指标，完整文本保存在输出文件中。"""
    header = (
        "ID             Type        Hit@3   MRR     Correct Faithful "
        "Relevant CtxPrec CtxRecall Abstain"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows:
        values = " ".join(
            f"{_format_score(row[column]):>8}"
            for column in METRIC_COLUMNS
        )
        print(f"{row['case_id']:<14} {row['case_type']:<10} {values}")
