"""Business processes written by the design batches -> NEUTRAL flow records for render_dataset.py.

Each batch file is a JSON list of items {"title", "request", "paraphrases": [2 str], "flow"} for one business area
(<area>_<batch>.json). An independent checker read every item and kept or rejected each of its three requests; its
verdicts come as a JSON list [{"file", "returned_items", "verdicts": [[k, [keep_request, keep_p1, keep_p2]], ...]}]
where k is the item's index in the list the batch returned. When a file's list no longer has returned_items items it
was edited after the check: its verdicts are matched by title + request text of the returned items instead, given
with --returned DIR (the returned lists), else the file is used without verdicts and that is reported.

Per item: repair.repair, then ooa_reward.problems must be empty ("fails rules"); fewer than 2 non-core actions is
"too short"; each request is kept when the checker kept it, it is plain words (no action ids, $ variables or JSON;
a template's {{placeholder}} is fine),
8-1200 characters long, and not close to a test request (gen_targeted.LeakGuard: difflib ratio over 0.8 with any
eval/plain_requests*.jsonl prompt). Test set 3 stays as it is; nothing here writes a test set.

Writes {"kind": "flow", "request", "flow", "source": "business2:<area>"}, one per kept request.

Usage: python assemble_business.py WORK_DIR CATALOG.jsonl SCHEMA.json OUT.jsonl [--verdicts V.json] [--glob '*_ab.json']
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ooa_reward as R                                          # noqa: E402
from gen_targeted import Checker, LeakGuard, write_jsonl, flow_record  # noqa: E402

LEAK = re.compile(r"\$\{?[A-Za-z_]|\booa\.[a-z_]+/|\bcore/[a-z_]+|[{}]")
PLACEHOLDER = re.compile(r"\{\{\s*\w+\s*\}\}")     # a document template's {{name}} is how people say it


def area_of(name):
    return re.sub(r"_(a|b|ab)$", "", Path(name).stem)


def verdict_map(items, entry):
    """index in items -> [keep x3]. entry: the checker's verdicts for this file (or None)."""
    if not entry:
        return None, "no verdicts"
    by_k = {int(k): list(keep) for k, keep in entry.get("verdicts") or []}
    if len(items) == entry.get("returned_items", len(items)):
        return by_k, "by index"
    returned = entry.get("_returned")
    if not returned:
        return None, f"list has {len(items)} items, checker saw {entry.get('returned_items')}: no verdicts used"
    key = {(it.get("title"), " ".join(str(it.get("request")).split())): k for k, it in enumerate(returned)}
    out = {}
    for i, it in enumerate(items):
        k = key.get((it.get("title"), " ".join(str(it.get("request")).split())))
        if k is not None and k in by_k:
            out[i] = by_k[k]
    return out, "by title and request"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("work")
    ap.add_argument("catalog")
    ap.add_argument("schema")
    ap.add_argument("out")
    ap.add_argument("--verdicts", help="the checker's verdicts (JSON list, see above)")
    ap.add_argument("--returned", help="folder with the lists as the batches returned them (only for edited files)")
    ap.add_argument("--glob", default="*_ab.json")
    a = ap.parse_args(argv)

    checker, guard = Checker(a.catalog, a.schema), LeakGuard()
    verdicts = {v["file"]: v for v in json.load(open(a.verdicts, encoding="utf-8"))} if a.verdicts else {}
    rows, why, per_area, how = [], Counter(), Counter(), {}
    seen = set()
    for path in sorted(Path(a.work).glob(a.glob)):
        items = json.load(open(path, encoding="utf-8"))
        entry = verdicts.get(path.name)
        if entry and a.returned and (Path(a.returned) / path.name).exists():
            entry = {**entry, "_returned": json.load(open(Path(a.returned) / path.name, encoding="utf-8"))}
        vmap, how[path.name] = verdict_map(items, entry)
        area = area_of(path.name)
        for i, it in enumerate(items):
            fixed = checker.check(it.get("flow"))
            if fixed is None:
                why["fails rules"] += 1
                continue
            if len([x for x in R.actions_of(fixed) if not str(x).startswith("core/")]) < 2:
                why["too short"] += 1
                continue
            keep = vmap.get(i, [False] * 3) if vmap is not None else [True] * 3
            reqs = [it.get("request")] + list(it.get("paraphrases") or [])
            n0 = len(rows)
            for q, ok in zip(reqs, keep):
                q = " ".join(str(q or "").split())
                if not ok:
                    why["checker rejected request"] += 1
                elif LEAK.search(PLACEHOLDER.sub("", q)) or not 8 <= len(q) <= 1200:
                    why["request format"] += 1
                elif q.lower() in seen:
                    why["same request"] += 1
                elif guard.leaks(q):
                    why["close to a test request"] += 1
                else:
                    seen.add(q.lower())
                    rows.append(flow_record(q, fixed, f"business2:{area}"))
            if len(rows) == n0:
                why["no usable request"] += 1
            else:
                per_area[area] += 1
    write_jsonl(a.out, rows)
    print(f"{len(rows)} records from {sum(per_area.values())} processes -> {a.out}")
    print("dropped:", dict(why))
    print("processes per area:", dict(sorted(per_area.items())))
    print("verdicts matched:", how)


if __name__ == "__main__":
    main()
