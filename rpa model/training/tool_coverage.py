"""Which toolkits and actions does the training data teach, in everyday wording? Writes a Markdown report.

Usage: python tool_coverage.py CATALOG.jsonl OUT.md TRAIN_FILE [TRAIN_FILE ...]
Counts, per action, how many training rows use it (each row counts an action once). Per toolkit: actions, actions
with at least one example, with at least 5, and the rows. Toolkits with the least coverage come first: those are
the ones to write examples for next (gen_tools.py does it for every action under a minimum).
"""
import json
import sys
from collections import Counter, defaultdict


def actions_in(answer):
    out = []
    try:
        f = json.loads(answer)
    except Exception:
        return out
    def walk(nodes):
        for n in nodes or []:
            if isinstance(n, dict):
                if isinstance(n.get("action"), str):
                    out.append(n["action"])
                for k in ("children", "else", "finally"):
                    walk(n.get(k) if isinstance(n.get(k), list) else None)
    walk(f.get("steps") if isinstance(f, dict) else None)
    return out


def coverage(catalog, files):
    seen = Counter()
    for f in files:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                seen.update(set(actions_in(json.loads(l)["messages"][-1]["content"])))
    by = defaultdict(list)
    for e in catalog:
        if e["kind"] == "action":
            by[(e["toolkit_id"], e["toolkit"])].append(e)
    rows = []
    for (tid, name), es in by.items():
        n = len(es)
        rows.append({"toolkit": name, "id": tid, "actions": n, "covered": sum(seen[e["id"]] > 0 for e in es),
                     "covered5": sum(seen[e["id"]] >= 5 for e in es), "rows": sum(seen[e["id"]] for e in es),
                     "missing": [e["id"].split("/")[1] for e in es if seen[e["id"]] == 0]})
    rows.sort(key=lambda r: (r["covered"] / r["actions"], -r["actions"]))
    return rows, seen


def main():
    cat_path, out = sys.argv[1], sys.argv[2]
    catalog = [json.loads(l) for l in open(cat_path, encoding="utf-8") if l.strip()]
    rows, seen = coverage(catalog, sys.argv[3:])
    acts = [e for e in catalog if e["kind"] == "action"]
    lines = ["# Toolkit coverage of the training data", "",
             f"{len(acts)} actions in {len(rows)} toolkits | with an example: {sum(seen[e['id']] > 0 for e in acts)} | "
             f"with 5 or more: {sum(seen[e['id']] >= 5 for e in acts)}", "",
             "| Toolkit | Actions | With an example | With 5+ | Rows | Not yet taught |", "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        miss = ", ".join(r["missing"][:8]) + (f" (+{len(r['missing']) - 8})" if len(r["missing"]) > 8 else "")
        lines.append(f"| {r['toolkit']} | {r['actions']} | {r['covered']} | {r['covered5']} | {r['rows']} | {miss} |")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
