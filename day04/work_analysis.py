from cosine_similarity import cosine_similarity
from job_vector_match import skills_to_vector

jobs = [
    {
        "position": "RAG实习生",
        "skills": [
            "python",
            "rag",
            "llm"
        ]
    },

    {
        "position": "Agent实习生",
        "skills": [
            "python",
            "agent",
            "llm"
        ]
    },

    {
        "position": "AI工程实习生",
        "skills": [
            "python",
            "git"
        ]
    },

    {
        "position": "算法实习生",
        "skills": [
            "python",
            "pytorch",
            "llm"
        ]
    }
]

all_skills = [
    "python",
    "git",
    "rag",
    "agent",
    "llm",
    "pytorch"
]


my_skills = [
    "python",
    "git"
]
my_vector = skills_to_vector(
    my_skills,
    all_skills
)


results = []


for job in jobs:

    job_vector = skills_to_vector(
        job["skills"],
        all_skills
    )


    similarity = cosine_similarity(
        my_vector,
        job_vector
    )


    results.append({
        "position": job["position"],
        "similarity": similarity
    })

    # sorted(...)
#
# 对列表排序。


# key=lambda x: x["similarity"]
#
# 指定按照 similarity 排序。


# reverse=True
#
# 从大到小。

results = sorted(
    results,
    key=lambda x: x["similarity"],
    reverse=True
)

print(
        "===== AI岗位向量匹配 ====="
)

for result in results:

    print(
        result["position"],
        round(result["similarity"], 3)
    )
