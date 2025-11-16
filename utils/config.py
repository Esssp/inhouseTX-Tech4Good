import json

def load_taxonomy(path="taxonomy.json"):
    with open(path, "r") as f:
        taxonomy = json.load(f)
    return taxonomy
