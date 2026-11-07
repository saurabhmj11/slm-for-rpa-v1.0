"""Verifiable reward for OOA flows: the platform's own rules score the model's answer. No human, no LLM judge.

score(text, gold_actions=None) -> {"total": 0..1, "solved": bool, parts...}
Parts (weights): starts_obj .05 | json .20 | schema .15 | actions_exist .15 | args_ok .10 | vars_ok .10 | match .25
 - match compares the action sequence with the gold flow (partial credit by sequence similarity, so there is
   a gradient to climb even when the answer is not exact). Without gold, total is capped at .75.
 - A flow with no steps scores 0 on everything after json: empty flows cannot win.
 - solved = every part passes and (if gold given) the action sequence matches exactly.
Needs env OOA_CATALOG (catalog.jsonl) and OOA_SCHEMA (flow.v1.json), or call configure().
"""
import difflib, json, os

_CAT = None
_SCHEMA = None
_SV = None   # schema validator, built once (jsonschema.validate() rebuilds it on every call: about 70 ms each)


def configure(catalog_path=None, schema_path=None):
    global _CAT, _SCHEMA, _SV
    catalog_path = catalog_path or os.environ["OOA_CATALOG"]
    schema_path = schema_path or os.environ["OOA_SCHEMA"]
    _CAT = {}
    for l in open(catalog_path, encoding="utf-8"):
        e = json.loads(l)
        _CAT[e["id"]] = e
    _SCHEMA = json.load(open(schema_path, encoding="utf-8"))
    import jsonschema
    _SV = jsonschema.validators.validator_for(_SCHEMA)(_SCHEMA)


def walk(nodes):
    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        yield n
        for k in ("children", "else", "finally"):
            if isinstance(n.get(k), list):
                yield from walk(n[k])
        for ei in n.get("elseIf") or []:
            if isinstance(ei, dict):
                yield from walk(ei.get("children"))
        for c in n.get("catch") or []:
            if isinstance(c, dict):
                yield from walk(c.get("children"))


def actions_of(flow):
    return [s.get("action") for s in walk(flow.get("steps"))]


def gold_actions(flow_text):
    return actions_of(json.loads(flow_text))


def clean(text):
    return text.split("<|assistant_end|>")[0].strip()


W = {"starts_obj": .05, "json": .20, "schema": .15, "actions_exist": .15, "args_ok": .10, "vars_ok": .10, "match": .25}


def score(text, gold=None):
    if _CAT is None:
        configure()
    import jsonschema
    text = clean(text)
    p = {k: 0.0 for k in W}
    p["starts_obj"] = 1.0 if text.startswith("{") else 0.0
    try:
        flow = json.loads(text)
        assert isinstance(flow, dict)
        p["json"] = 1.0
    except Exception:
        return _total(p, gold)
    steps = list(walk(flow.get("steps")))
    if not steps:
        return _total(p, gold)
    try:
        _SV.validate(flow)
        p["schema"] = 1.0
    except Exception:
        pass
    acts = [s.get("action") for s in steps]
    p["actions_exist"] = 1.0 if all(a in _CAT for a in acts) else 0.0
    ok = p["actions_exist"] == 1.0
    declared = {v.get("name") for v in flow.get("variables", []) if isinstance(v, dict)}
    ids = [s.get("id") for s in steps]
    vars_ok = len(ids) == len(set(ids))
    for s in steps:
        args, out = s.get("args") or {}, s.get("out") or {}
        if not isinstance(args, dict) or not isinstance(out, dict):
            ok = vars_ok = False
            continue
        e = _CAT.get(s.get("action"))
        if e and not s["action"].startswith("core/"):
            ins = {i["name"] for i in e["inputs"]}
            outs = {o["name"] for o in e["outputs"]}
            req = {i["name"] for i in e["inputs"] if i["required"]}
            if not (set(args) <= ins and set(out) <= outs and req <= set(args)):
                ok = False
        for v in list(args.values()):
            if isinstance(v, str) and v.startswith("$") and v[1:] not in declared:
                vars_ok = False
        for v in out.values():
            if v not in declared:
                vars_ok = False
    p["args_ok"] = 1.0 if ok else 0.0
    p["vars_ok"] = 1.0 if vars_ok else 0.0
    if gold is not None:
        p["match"] = difflib.SequenceMatcher(a=acts, b=gold).ratio()
    return _total(p, gold)


def _total(p, gold):
    total = sum(W[k] * p[k] for k in W)
    valid = all(p[k] == 1.0 for k in ("json", "schema", "actions_exist", "args_ok", "vars_ok"))
    solved = valid and (p["match"] == 1.0 if gold is not None else True)
    return {"total": round(total, 4), "solved": solved, "valid": valid, **p}


def problems(text, limit=8):
    """Plain-language list of what is wrong with an answer (empty list = valid). Used to tell a model how to fix a retry."""
    if _CAT is None:
        configure()
    text = clean(text)
    try:
        flow = json.loads(text)
        assert isinstance(flow, dict)
    except Exception:
        return ["The answer is not a single valid JSON object."]
    out = []
    for e in list(_SV.iter_errors(flow))[:3]:
        out.append("Schema: " + e.message[:140] + (" at " + "/".join(map(str, e.path)) if e.path else ""))
    steps = list(walk(flow.get("steps")))
    if not steps:
        out.append("The flow has no steps.")
    declared = {v.get("name") for v in flow.get("variables", []) if isinstance(v, dict)}
    ids = [s.get("id") for s in steps]
    if len(ids) != len(set(ids)):
        out.append("Step ids are not unique.")
    for s in steps:
        a = s.get("action")
        e = _CAT.get(a)
        if e is None:
            out.append(f"Unknown action {a!r}.")
            continue
        args, o = s.get("args") or {}, s.get("out") or {}
        if not a.startswith("core/"):
            ins = {i["name"] for i in e["inputs"]}
            outs = {x["name"] for x in e["outputs"]}
            req = {i["name"] for i in e["inputs"] if i["required"]}
            for k in args:
                if k not in ins:
                    out.append(f"{a} has no input {k!r}. Inputs: {', '.join(sorted(ins)) or '-'}.")
            for k in o:
                if k not in outs:
                    out.append(f"{a} has no output {k!r}. Outputs: {', '.join(sorted(outs)) or '-'}.")
            for k in sorted(req - set(args)):
                out.append(f"{a} is missing required input {k!r}.")
        for v in args.values():
            if isinstance(v, str) and v.startswith("$") and v[1:] not in declared:
                out.append(f"Variable {v!r} is used but not declared in variables.")
        for v in o.values():
            if v not in declared:
                out.append(f"Output variable {v!r} is not declared in variables.")
    return list(dict.fromkeys(out))[:limit]
