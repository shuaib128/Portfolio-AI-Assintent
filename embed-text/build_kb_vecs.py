import json, httpx

with open("kb.json") as f:
    kb = json.load(f)
    
kb_with_vecs = []
for item in kb:
    r = httpx.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": item["q"]
    })
    vec = r.json()["embedding"]
    kb_with_vecs.append({ **item, "vec": vec })
    
with open("kb_embedded.json", "w") as f:
    json.dump(kb_with_vecs, f)