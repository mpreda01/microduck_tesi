import json
import re
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"]]
script = [c for c in cells if c.startswith("%%writefile")][0].split("\n", 1)[1]
compile(script, "eval_scenario.py", "exec")
print("embedded script compiles")

suite = [c for c in cells if c.startswith("# ======================= SUITE")][0]
ns = {"POLICY": "alpha_walking", "print": lambda *a, **k: None}
exec(suite, ns)
print(len(ns["CONDITIONS"]), "conditions, suite", ns["SUITE_NAME"])
print([c["name"] for c in ns["CONDITIONS"] if c["name"].startswith("slope_")])

text = "".join(cells)
words = ["episodi", "terreno", "cadute", "passi", "pendenza", "piano", "della", "delle", "che", "quando",
         "gradi", "carico", "salita", "discesa", "rotazione"]
hits = sorted({w for w in words if re.search(r"\b" + w + r"\b", text, re.I)})
print("italian words found:", hits or "none")
