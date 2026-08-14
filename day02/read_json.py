import json

with open("job.json", "r", encoding="utf-8") as f:
    job = json.load(f)

print(job)
print(job["position"])