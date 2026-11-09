"""Programmatic OOA flow generator: composes valid flows from the action catalog, no LLM.

Usage: python gen_flows.py CATALOG_JSONL PLATFORM_REPO OUT_JSONL [N=3000] [SEED=1]
Each example: a natural-language request -> minified Flow JSON (nanochat {"messages": [...]} format).
Flows are 1-4 steps. A step's required inputs come from earlier outputs of a matching type when possible,
otherwise they become declared flow inputs. Every flow is checked before it is kept:
  JSON schema (flow.v1.json), action exists in catalog, input and output names exist, every $variable is declared.
Limits (v1): chains stay inside one toolkit (so they stay plausible), optionally wrapped in one loop / if / try container, required inputs only, no credential actions.
"""
import json, random, re, sys
from pathlib import Path

TYPE_MAP = {"Text": "String", "File path": "String", "Number": "Number", "Yes/No": "Boolean", "List": "List",
            "Table": "Table", "Dictionary": "Dictionary", "DateTime": "DateTime", "Any": "Any"}
COMPAT = {"String": {"String", "Any"}, "Number": {"Number", "Any"}, "Boolean": {"Boolean", "Any"},
          "List": {"List", "Any"}, "Table": {"Table", "Any"}, "Dictionary": {"Dictionary", "Any"},
          "DateTime": {"DateTime", "Any"}, "Any": {"Any", "String", "Number", "Boolean", "List", "Table", "Dictionary", "DateTime"}}
WORDINGS = ["{c}", "Create an OOA flow: {c}", "Build an automation that does this. {c}"]


def vtype(t):
    return TYPE_MAP.get(t)


def snake(s):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s if s and not s[0].isdigit() else "v_" + s


def fillable(e):
    if e["toolkit_id"] == "core":   # control-flow built-ins (continue, break, ...) only make sense inside containers
        return False
    req = [i for i in e["inputs"] if i["required"]]
    return (e["kind"] == "action" and len(req) <= 6 and all(vtype(i["type"]) for i in req)
            and all(vtype(o["type"]) or True for o in e["outputs"]))


def clean_desc(d):
    return d.strip().rstrip(".").strip()


def build(cat_actions, rng, cfg=None):
    cfg = cfg or {}
    n_steps = rng.choices([1, 2, 3, 4], weights=cfg.get("step_weights", [25, 35, 25, 15]))[0]
    pat = cfg.get("patterns")          # learned from real flows: which action follows which (see real_patterns)
    if pat:
        n_steps = rng.choice(pat["lengths"])
    variables, steps, used_names = [], [], set()
    avail = []  # (var_name, vtype) outputs not yet consumed
    descs, in_names = [], []

    def new_name(base):
        base = snake(base)[:40]
        n, k = base, 2
        while n in used_names:
            n, k = f"{base}_{k}", k + 1
        used_names.add(n)
        return n

    prev = None
    descs_steps = []
    for si in range(n_steps):
        e = None
        if pat:
            pool = pat["starters"] if prev is None else pat["next"].get(prev["id"], [])
            pool = [pat["by_id"][a] for a in pool if a in pat["by_id"]]
            if pool:
                e = rng.choice(pool)        # repeats in the list = how often real flows make that move
        cands = cat_actions
        if prev is not None and avail:
            want = {t for _, t in avail}
            c2 = [e for e in cat_actions if any(vtype(i["type"]) in want for i in e["inputs"] if i["required"])]
            cands = c2 or cat_actions
        if prev is not None:   # chains stay inside one toolkit so they stay plausible
            cands = [c for c in cat_actions if c["toolkit_id"] == prev["toolkit_id"]
                     and (not avail or any(vtype(i["type"]) in {t for _, t in avail}
                                           for i in c["inputs"] if i["required"]))]
            if not cands and e is None:
                break
        if e is None:
            e = rng.choice(cands)
        args = {}
        for i in [i for i in e["inputs"] if i["required"]]:
            t = vtype(i["type"])
            src = None if t == "Any" else next((a for a in avail if a[1] in COMPAT[t] and a[1] != "Any"), None)
            if src:
                args[i["name"]] = "$" + src[0]
                avail.remove(src)
            else:
                vn = new_name(i["name"])
                var = {"name": vn, "type": t, "direction": "input", "required": True}
                if i["description"]:
                    var["description"] = i["description"][:120]
                variables.append(var)
                in_names.append(vn)
                args[i["name"]] = "$" + vn
        out = {}
        for o in e["outputs"][:2]:
            vn = new_name(o["name"])
            out[o["name"]] = vn
            variables.append({"name": vn, "type": vtype(o["type"]) or "Any", "direction": "local"})
            avail.append((vn, vtype(o["type"]) or "Any"))
        step = {"id": f"s{si + 1}", "action": e["id"], "description": clean_desc(e["description"])[:140], "args": args}
        if out:
            step["out"] = out
        steps.append(step)
        descs_steps.append(step)
        descs.append(clean_desc(e["description"]))
        prev = e
    # optionally wrap the chain in a control-flow container (loop / if / try)
    wrap_kind = rng.choices(["none", "loop", "if", "try"], weights=cfg.get("wrap_weights", [55, 17, 14, 14]))[0]
    prefix, suffix = "", ""
    body = steps
    if wrap_kind == "loop":
        items = new_name("items"); item = new_name("item")
        variables.append({"name": items, "type": "List", "direction": "input", "required": True,
                          "description": "The items to process, one per loop pass."})
        variables.append({"name": item, "type": "Any", "direction": "local"})
        in_names.append(items)
        body = [{"id": "w1", "action": "core/loop", "description": "Repeat for each item",
                 "args": {"iterator": "list", "over": "$" + items, "as": item}, "children": steps}]
        prefix = f"For each entry in {items}: "
    elif wrap_kind == "if":
        flag = new_name(rng.choice(["proceed", "enabled", "send_it", "is_active"]))
        variables.append({"name": flag, "type": "Boolean", "direction": "input", "required": True,
                          "description": "Run the steps only when true."})
        in_names.append(flag)
        body = [{"id": "w1", "action": "core/if", "description": "Only when the flag is true",
                 "args": {"condition": "$" + flag}, "children": steps}]
        prefix = f"Only when {flag} is true: "
    elif wrap_kind == "try":
        body = [{"id": "w1", "action": "core/try", "description": "Handle errors", "children": steps,
                 "catch": [{"codes": [], "children": [{"id": "w2", "action": "core/log", "args": {
                     "message": rng.choice(["Step failed", "Something went wrong", "Error in this flow"]),
                     "level": "error"}}]}]}]
        suffix = " If anything fails, write an error to the log."
    # sequential, unique ids over the whole tree
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
    steps = body
    # last step's outputs become flow outputs
    out_names = []
    last_chain = descs_steps[-1] if descs_steps else None
    if last_chain and "out" in last_chain:
        for vn in last_chain["out"].values():
            for v in variables:
                if v["name"] == vn:
                    v["direction"] = "output"
                    out_names.append(vn)
    chain = ", then ".join(d[:1].lower() + d[1:] if k else d for k, d in enumerate(descs)) + "."
    if prefix:
        chain = prefix + chain[:1].lower() + chain[1:]
    chain += suffix
    extra = ""
    if in_names:
        extra += f" The flow takes {', '.join(in_names)} as input."
    if out_names:
        extra += f" It returns {', '.join(out_names)}."
    first = descs[0]
    name = " ".join(first.split()[:6]).title()
    flow = {"kind": "TaskBot", "id": snake(first)[:40] or "flow", "name": name,
            "description": chain, "variables": variables, "steps": steps}
    return flow, chain + extra


def real_patterns(repo, catalog, max_len=10):
    """Action-to-action moves seen in the REAL example flows (excluding held-out test families), as a generator config.
    Real flows are long and cross toolkits (browser -> excel -> email); this lets generated flows do the same."""
    import glob, hashlib, os
    by_id = {e["id"]: e for e in catalog if fillable(e)}
    nxt, starters, lengths = {}, [], []
    for path in sorted(glob.glob(str(Path(repo) / "examples" / "**" / "*.flow.json"), recursive=True)):
        fam = re.sub(r"-capture$", "", os.path.basename(path)[:-len(".flow.json")])
        if int(hashlib.sha1(fam.encode()).hexdigest(), 16) % 4 == 0:      # same rule as build_sft_data: held out
            continue
        flow = json.load(open(path, encoding="utf-8"))
        seq = []
        def walk(ns):
            for n in ns or []:
                if isinstance(n, dict):
                    if not str(n.get("action", "")).startswith("core/"):
                        seq.append(n.get("action"))
                    for k in ("children", "else", "finally"):
                        if isinstance(n.get(k), list):
                            walk(n[k])
                    for c in n.get("catch") or []:
                        walk(c.get("children"))
        walk(flow.get("steps"))
        if len(seq) < 2:
            continue
        starters.append(seq[0])
        lengths.append(max(3, min(max_len, len(seq))))
        for a, b in zip(seq, seq[1:]):
            nxt.setdefault(a, []).append(b)
    return {"next": nxt, "starters": starters, "lengths": lengths, "by_id": by_id}


def make_validator(repo, catalog):
    import jsonschema
    schema = json.load(open(Path(repo) / "contracts/flow-schema/flow.v1.json"))
    cat = {e["id"]: e for e in catalog}
    jsonschema.Draft202012Validator.check_schema(schema)
    schema_validator = jsonschema.validators.validator_for(schema)(schema)   # built once; validate() rebuilt it for every flow

    def validate(flow, check_schema=True):
        if check_schema:
            schema_validator.validate(flow)
        declared = {v["name"] for v in flow["variables"]}
        def walk(nodes):
            for n in nodes:
                yield n
                for k in ("children", "else", "finally"):
                    if isinstance(n.get(k), list):
                        yield from walk(n[k])
                for c in n.get("catch") or []:
                    yield from walk(c.get("children", []))
        ids = []
        for s in walk(flow["steps"]):
            ids.append(s["id"])
            if s["action"].startswith("core/"):
                for v in s.get("args", {}).values():
                    if isinstance(v, str) and v.startswith("$"):
                        assert v[1:] in declared, f"undeclared {v}"
                continue
            e = cat[s["action"]]
            ins = {i["name"] for i in e["inputs"]}
            outs = {o["name"] for o in e["outputs"]}
            assert set(s.get("args", {})) <= ins, f"unknown input in {s['action']}"
            assert set(s.get("out", {})) <= outs, f"unknown output in {s['action']}"
            req = {i["name"] for i in e["inputs"] if i["required"]}
            assert req <= set(s.get("args", {})), "missing required input"
            for v in s.get("args", {}).values():
                if isinstance(v, str) and v.startswith("$"):
                    assert v[1:] in declared, f"undeclared {v}"
            for v in s.get("out", {}).values():
                assert v in declared, f"undeclared out {v}"
        assert len(ids) == len(set(ids)), "duplicate step ids"
    return validate


def generate(catalog, repo, n, seed, exclude=(), schema_every=1, cfg=None, max_chars=None):
    """Return n unique valid (request, flow_dict, minified_flow) triples; skips minified flows in `exclude`.
    Every flow is fully checked (schema, actions, inputs, outputs, variables, ids). schema_every=N checks the schema on
    every N-th candidate only. (The slow part used to be jsonschema.validate rebuilding its validator on every call.)"""
    rng = random.Random(seed)
    actions = [e for e in catalog if fillable(e)]
    validate = make_validator(repo, catalog)
    seen, out, tries = set(exclude), [], 0
    while len(out) < n and tries < n * 30:
        tries += 1
        flow, req = build(actions, rng, cfg)
        try:
            validate(flow, check_schema=(tries % schema_every == 0))
        except Exception:
            continue
        mini = json.dumps(flow, separators=(",", ":"), ensure_ascii=False)
        if mini in seen:
            continue
        q = rng.choice(WORDINGS).format(c=req)
        if max_chars and len(q) + len(mini) > max_chars:   # rows longer than the training window never fit nanochat's packing
            continue
        seen.add(mini)
        out.append((q, flow, mini))
    return out


def main():
    if len(sys.argv) > 6 and sys.argv[6] == "patterns":     # real-flow patterns: long, multi-toolkit flows
        cat_path, repo, out, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        catalog = [json.loads(l) for l in open(cat_path, encoding="utf-8")]
        rows = generate(catalog, repo, n, seed, cfg={"patterns": real_patterns(repo, catalog)}, max_chars=5500)
        with open(out, "w", encoding="utf-8") as f:
            for q, _, mini in rows:
                f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": mini}]}, ensure_ascii=False) + "\n")
        lens = sorted(len(m) for *_, m in rows)
        print(f"{len(rows)} real-pattern flows | chars median {lens[len(lens)//2]} max {lens[-1]}")
        return
    cat_path, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    catalog = [json.loads(l) for l in open(cat_path, encoding="utf-8")]
    actions = [e for e in catalog if fillable(e)]
    validate = make_validator(repo, catalog)
    seen, rows, rejected, lens = set(), [], 0, []
    tries = 0
    while len(rows) < n and tries < n * 20:
        tries += 1
        flow, req = build(actions, rng)
        try:
            validate(flow)
        except Exception:
            rejected += 1
            continue
        mini = json.dumps(flow, separators=(",", ":"), ensure_ascii=False)
        if mini in seen:
            continue
        seen.add(mini)
        lens.append(len(mini))
        rows.append({"messages": [{"role": "user", "content": rng.choice(WORDINGS).format(c=req)},
                                  {"role": "assistant", "content": mini}]})
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens.sort()
    print(f"{len(rows)} flows from {len(actions)} fillable actions | rejected {rejected} | chars median {lens[len(lens)//2]} max {lens[-1]}")


if __name__ == "__main__":
    main()
