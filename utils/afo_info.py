import json


with open("data/afo/meta.json") as f:
    data = json.load(f)

CLASS_NAME_BY_ID = {c["id"]: c["title"] for c in data["classes"]}
