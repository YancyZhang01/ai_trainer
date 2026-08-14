jobs = [
    {
        "company": "A公司",
        "position": "RAG实习生",
        "city": "杭州",
        "skills": ["Python", "RAG", "LLM"]
    },
    {
        "company": "B公司",
        "position": "Agent实习生",
        "city": "上海",
        "skills": ["Python", "Agent", "LLM"]
    },
    {
        "company": "C公司",
        "position": "模型评测实习生",
        "city": "杭州",
        "skills": ["Python", "Evaluation"]
    }
]

for job in jobs:
    if job["city"] == "杭州":
         print(f"{job['company']} - {job['position']}")


for i in range(len(jobs)):
     if jobs[i]["city"] == "杭州" :
          print(f"{jobs[i]['company']} - {jobs[i]['position']}")