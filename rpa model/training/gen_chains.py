"""Generate valid flows for everyday CROSS-toolkit chains (read -> change -> write, browser sequences ...).

gen_flows.build keeps multi-step chains inside one toolkit, so the training data had almost no
"CSV -> sort -> CSV" or "open browser -> type -> click" flows; those are exactly the requests the model got wrong.
This builds each chain in CHAINS with the same wiring rules as gen_flows (outputs feed later inputs by type),
in several variants (different input names, optional loop / condition / error handling around it).
Chains identical to a request of the second test set (eval/plain_requests_2.jsonl) are skipped, so that set stays
an honest check of whether the model generalises.

Usage: python gen_chains.py CATALOG.jsonl PLATFORM_REPO OUT.jsonl [variants_per_chain] [seed]
Writes one flow per line ({"flow": {...}, "chain": [...]}); requests are written separately (everyday wording).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_flows as G

O = "ooa."
CHAINS = [
    # CSV and tables
    ["csv/read", "excel/write_table"], ["csv/read", "datatable/sort", "csv/write"],
    ["csv/read", "datatable/filter", "csv/write"], ["csv/read", "datatable/row_count"],
    ["csv/read", "datatable/filter", "excel/write_table"], ["csv/read", "datatable/sort", "excel/write_table"],
    ["csv/read", "datatable/filter", "datatable/row_count"], ["csv/read", "datatable/sort", "datatable/write_to_file"],
    ["csv/read", "datatable/remove_duplicates", "excel/write_table"], ["csv/read", "datatable/sum_column", "file/write_text"],
    ["csv/read", "datatable/filter", "csv/write", "email/send"],
    # Excel
    ["excel/open", "excel/get_all_cells", "csv/write"], ["excel/open", "excel/get_all_cells", "datatable/row_count"],
    ["excel/open", "excel/get_all_cells", "datatable/sort", "excel/write_table", "excel/save"],
    ["excel/open", "excel/read_column", "excel/close"], ["excel/open", "excel/set_cell", "excel/save", "excel/close"],
    ["excel/open", "excel/get_sheet_names", "excel/close"],
    # PDF and text files
    ["pdf/extract_text", "file/write_text"], ["pdf/extract_tables", "excel/write_table"],
    ["pdf/extract_tables", "csv/write"], ["pdf/merge", "email/send"], ["pdf/extract_text", "word/create_document"],
    ["file/read_text", "file/write_text"], ["file/copy", "file/delete"], ["file/move", "file/delete"],
    ["file/list", "file/zip"], ["file/unzip", "file/list"], ["file/read_text", "email/send"],
    # Word
    ["word/open", "word/replace_text", "word/save"], ["word/open", "word/read_text", "file/write_text"],
    ["word/open", "word/add_paragraph", "word/save", "word/close"], ["word/open", "word/replace_text", "word/save_as"],
    ["word/create_document", "word/add_table", "word/save"],
    # browser
    ["browser/open", "browser/get_text", "browser/close"], ["browser/open", "browser/extract_table", "csv/write"],
    ["browser/open", "browser/extract_table", "excel/write_table"], ["browser/open", "browser/type_text", "browser/type_text", "browser/click"],
    ["browser/open", "browser/download"], ["browser/open", "browser/click", "browser/download", "browser/close"],
    ["browser/open", "browser/type_text", "browser/click", "browser/get_text", "browser/close"],
    ["browser/open", "browser/navigate", "browser/get_text"], ["browser/open", "browser/extract_table", "browser/close"],
    ["browser/open", "browser/select_option", "browser/click"],
    # email
    ["email/save_attachments", "file/list"], ["email/list", "email/move"], ["email/get", "email/forward"],
    ["email/save_attachments", "pdf/merge"],
]


def strip(chain):
    return [a for a in chain if not a.startswith("core/")]


def build_chain(seq, byid, rng):
    """One flow for the action sequence `seq`, wired like gen_flows.build."""
    variables, steps, used_names, avail, descs, in_names = [], [], set(), [], [], []

    def new_name(base):
        base = G.snake(base)[:40]
        n, k = base, 2
        while n in used_names:
            n, k = f"{base}_{k}", k + 1
        used_names.add(n)
        return n

    for si, aid in enumerate(seq):
        e = byid[aid]
        args = {}
        for i in [i for i in e["inputs"] if i["required"]]:
            t = G.vtype(i["type"])
            src = None if t == "Any" else next((a for a in avail if a[1] in G.COMPAT[t] and a[1] != "Any"), None)
            if src:
                args[i["name"]] = "$" + src[0]
                avail.remove(src)
            else:
                vn = new_name(i["name"] if rng.random() < 0.7 else rng.choice([i["name"], "my_" + i["name"], i["name"] + "_path"]))
                var = {"name": vn, "type": t, "direction": "input", "required": True}
                if i.get("description"):
                    var["description"] = i["description"][:120]
                variables.append(var)
                in_names.append(vn)
                args[i["name"]] = "$" + vn
        out = {}
        for o in e["outputs"][:2]:
            vn = new_name(o["name"])
            out[o["name"]] = vn
            variables.append({"name": vn, "type": G.vtype(o["type"]) or "Any", "direction": "local"})
            avail.append((vn, G.vtype(o["type"]) or "Any"))
        step = {"id": f"s{si + 1}", "action": aid, "description": G.clean_desc(e["description"])[:140], "args": args}
        if out:
            step["out"] = out
        steps.append(step)
        descs.append(G.clean_desc(e["description"]))
    wrap = rng.choices(["none", "loop", "if", "try"], weights=[70, 10, 10, 10])[0]
    body = steps
    if wrap == "loop":
        items, item = new_name("items"), new_name("item")
        variables += [{"name": items, "type": "List", "direction": "input", "required": True,
                       "description": "The items to process, one per loop pass."},
                      {"name": item, "type": "Any", "direction": "local"}]
        body = [{"id": "w1", "action": "core/loop", "description": "Repeat for each item",
                 "args": {"iterator": "list", "over": "$" + items, "as": item}, "children": steps}]
    elif wrap == "if":
        flag = new_name(rng.choice(["proceed", "enabled", "is_active"]))
        variables.append({"name": flag, "type": "Boolean", "direction": "input", "required": True,
                          "description": "Run the steps only when true."})
        body = [{"id": "w1", "action": "core/if", "description": "Only when the flag is true",
                 "args": {"condition": "$" + flag}, "children": steps}]
    elif wrap == "try":
        body = [{"id": "w1", "action": "core/try", "description": "Handle errors", "children": steps,
                 "catch": [{"codes": [], "children": [{"id": "w2", "action": "core/log",
                                                       "args": {"message": "Step failed", "level": "error"}}]}]}]
    counter = [0]

    def renumber(nodes):
        for n in nodes:
            counter[0] += 1
            n["id"] = f"s{counter[0]}"
            for k in ("children", "else", "finally"):
                if isinstance(n.get(k), list):
                    renumber(n[k])
            for c in n.get("catch") or []:
                renumber(c.get("children", []))
    renumber(body)
    if "out" in steps[-1]:
        for vn in steps[-1]["out"].values():
            for v in variables:
                if v["name"] == vn:
                    v["direction"] = "output"
    first = descs[0]
    return {"kind": "TaskBot", "id": G.snake(first)[:40] or "flow", "name": " ".join(first.split()[:6]).title(),
            "description": "; then ".join(descs), "variables": variables, "steps": body}


def main():
    cat_path, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
    variants = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 11)
    catalog = [json.loads(l) for l in open(cat_path, encoding="utf-8") if l.strip()]
    byid = {e["id"]: e for e in catalog}
    validate = G.make_validator(repo, catalog)
    test2 = Path(__file__).resolve().parent.parent / "eval" / "plain_requests_2.jsonl"
    held = {tuple(strip(json.loads(l)["must"])) for l in open(test2, encoding="utf-8")} if test2.exists() else set()
    seen, rows, skipped = set(), [], []
    for ch in CHAINS:
        seq = [O + a for a in ch]
        if tuple(seq) in held:
            skipped.append(ch); continue
        missing = [a for a in seq if a not in byid]
        if missing:
            skipped.append(ch); continue
        made = 0
        for _ in range(variants * 20):
            if made >= variants:
                break
            f = build_chain(seq, byid, rng)
            try:
                validate(f)
            except Exception:
                continue
            m = json.dumps(f, separators=(",", ":"), ensure_ascii=False)
            if m in seen:
                continue
            seen.add(m); made += 1
            rows.append({"flow": f, "chain": seq})
        if made == 0:
            skipped.append(ch)
    with open(out, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} flows from {len(CHAINS) - len(skipped)} chains | skipped (test set 2, missing action, or never valid): {skipped}")


if __name__ == "__main__":
    main()
