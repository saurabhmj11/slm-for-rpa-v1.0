"""Deterministic repair of near-miss flows before verification: snap misspelled action ids and input/output names to
the closest real one (only when very close), so small typos do not cost an escalation. Never invents anything new and
never adds or removes steps. Returns (flow, notes)."""
import copy, difflib


def repair(flow, cat):
    """cat: dict action_id -> catalog entry."""
    flow = copy.deepcopy(flow)
    notes, ids = [], list(cat)
    def walk(nodes):
        for n in nodes or []:
            if isinstance(n, dict):
                yield n
                for k in ("children", "else", "finally"):
                    if isinstance(n.get(k), list):
                        yield from walk(n[k])
                for c in n.get("catch") or []:
                    if isinstance(c, dict):
                        yield from walk(c.get("children"))
    for s in walk(flow.get("steps")):
        a = s.get("action")
        if isinstance(a, str) and a not in cat:
            m = difflib.get_close_matches(a, ids, n=1, cutoff=0.88)
            if m:
                notes.append(f"action {a} -> {m[0]}")
                s["action"] = a = m[0]
        e = cat.get(a)
        if not e or a.startswith("core/"):
            continue
        for key, valid in (("args", [i["name"] for i in e["inputs"]]), ("out", [o["name"] for o in e["outputs"]])):
            d = s.get(key)
            if not isinstance(d, dict):
                continue
            for k in list(d):
                if k not in valid:
                    m = difflib.get_close_matches(k, valid, n=1, cutoff=0.8)
                    if m and m[0] not in d:
                        notes.append(f"{a}.{key} {k} -> {m[0]}")
                        d[m[0]] = d.pop(k)
    return flow, notes
