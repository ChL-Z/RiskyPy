import json

path = "results/evaluations.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)
print({prompt_set: len(entries) for prompt_set, entries in data.items()})
if data["dropped_at_vuln_check"]:
    raise SystemExit("The run reached the end: nothing was removed.")
last_set = [prompt_set for prompt_set, entries in data.items() if entries][-1]
removed = data[last_set][-90:]
data[last_set] = data[last_set][:-90]
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Removed the last {len(removed)} entries of {last_set}: {removed[0]['run_file']} index {removed[0]['index']} "
      f"to {removed[-1]['run_file']} index {removed[-1]['index']}")