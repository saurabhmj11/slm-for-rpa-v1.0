"""Build seed SFT data for nanochat from the OOA platform repo. No LLM, no network beyond the clone.

Usage: python build_sft_data.py PLATFORM_REPO OUT_DIR [MAX_CHARS]
Writes OUT_DIR/{catalog.jsonl,train.jsonl,val.jsonl,eval.jsonl}
  train/val : {"messages":[{"role":"user"|"assistant","content":...}]}  (nanochat format)
  eval      : {"family","prompt","gold_flow"}  held-out workflow families, never in train/val
Dialogue types: (1) request -> minified Flow JSON from the gold flows (3 prompt wordings each),
                (2) catalog lookups: which action does X / what inputs does an action take.
Flows longer than MAX_CHARS (default 12000) are skipped so they are not truncated at SFT length.
"""
import glob, hashlib, json, os, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def family(path):
    return re.sub(r"-capture$", "", os.path.basename(path)[:-len(".flow.json")])


def is_heldout(fam):
    return int(hashlib.sha1(fam.encode()).hexdigest(), 16) % 4 == 0


WORDINGS = [
    "{d}",
    "Create an OOA flow for this: {d}",
    "Build me an automation. {d}",
]


def lower_first(s):
    return s[:1].lower() + s[1:] if s else s


def main():
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    max_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    out.mkdir(parents=True, exist_ok=True)

    # catalog
    import importlib.util
    here = Path(__file__).resolve().parent.parent / "data/catalog/build_catalog.py"
    if not here.exists():
        here = Path(__file__).resolve().parent / "build_catalog.py"  # notebook layout
    spec = importlib.util.spec_from_file_location("bc", here)
    bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
    cat = []
    for p in sorted((repo / "docs/reference/toolkits").glob("*.md")):
        if p.name != "README.md":
            cat += bc.parse_file(p)
    with open(out / "catalog.jsonl", "w", encoding="utf-8") as f:
        for e in cat:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    train, val, evals = [], [], []
    rng = random.Random(7)
    skipped = []
    for path in sorted(glob.glob(str(repo / "examples/**/*.flow.json"), recursive=True)):
        fam = family(path)
        d = json.load(open(path, encoding="utf-8"))
        desc = (d.get("description") or "").strip()
        if not desc or fam == "make-office-files":
            continue
        flow = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
        if is_heldout(fam):
            if not path.endswith("-capture.flow.json"):
                evals.append({"family": fam, "prompt": desc, "gold_flow": os.path.relpath(path, repo)})
            continue
        if len(flow) > max_chars:
            skipped.append((fam, len(flow)))
            continue
        for w in WORDINGS:
            train.append({"messages": [{"role": "user", "content": w.format(d=desc)},
                                       {"role": "assistant", "content": flow}]})

    # catalog dialogues (actions only)
    cat_rows = []
    for e in cat:
        if e["kind"] != "action":
            continue
        req = [i["name"] for i in e["inputs"] if i["required"]]
        opt = [i["name"] for i in e["inputs"] if not i["required"]]
        desc = e["description"].rstrip(".")
        cat_rows.append({"messages": [
            {"role": "user", "content": f"Which OOA action would I use to {lower_first(desc)}?"},
            {"role": "assistant", "content": f"Use `{e['id']}` ({e['toolkit']}: {e['name']})."}]})
        cat_rows.append({"messages": [
            {"role": "user", "content": f"What inputs does {e['id']} take?"},
            {"role": "assistant", "content":
                f"`{e['id']}` requires: {', '.join(req) if req else 'nothing'}. "
                f"Optional: {', '.join(opt[:12]) if opt else 'none'}."}]})
    rng.shuffle(cat_rows)
    cut = max(1, len(cat_rows) // 50)
    val = cat_rows[:cut] + [r for r in train[::17]]
    train = train + cat_rows[cut:]
    rng.shuffle(train)

    # separate files so the trainer can weight flows against the (much more numerous) catalog rows
    flow_rows = [r for r in train if len(r["messages"][1]["content"]) > 800 and r["messages"][1]["content"].startswith("{")]
    cat_only = [r for r in train if r not in flow_rows]
    for name, rows in (("train", train), ("train_flows", flow_rows), ("train_catalog", cat_only),
                       ("val", val), ("eval", evals)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"catalog {len(cat)} | train {len(train)} val {len(val)} eval {len(evals)} "
          f"| flows skipped for length: {len(skipped)} {skipped[:6]}")


if __name__ == "__main__":
    main()
