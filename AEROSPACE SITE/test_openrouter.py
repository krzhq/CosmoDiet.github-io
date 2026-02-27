import urllib.request, json

KEY = "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"

# Get list of free models
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": "Bearer " + KEY},
    method="GET",
)
with urllib.request.urlopen(req, timeout=15) as r:
    data = json.loads(r.read())

free_models = [m["id"] for m in data["data"] if ":free" in m["id"]]
print(f"Free models available: {len(free_models)}")
for m in free_models[:20]:
    print(" -", m)

# Try first few
print("\nTesting first 5:")
for m in free_models[:5]:
    body = json.dumps({"model": m, "messages": [{"role": "user", "content": "Say: OK"}], "max_tokens": 10}).encode()
    req2 = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY, "HTTP-Referer": "http://localhost:5000"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req2, timeout=20) as r2:
            d = json.loads(r2.read())
            reply = d["choices"][0]["message"]["content"][:50]
            print(f"WORKS: {m}")
            break
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:80]
        print(f"FAIL {e.code}: {m}")
    except Exception as e:
        print(f"ERR {m}: {e}")
