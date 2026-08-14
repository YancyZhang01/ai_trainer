import json

job = {
    "position": "RAG Intern",
    "city": "杭州",
    "skills": ["Python", "RAG"]
}

with open("job.json", "w", encoding="utf-8") as f:
    json.dump(
        job,
        f,
        ensure_ascii=False,
        indent=4
    )
