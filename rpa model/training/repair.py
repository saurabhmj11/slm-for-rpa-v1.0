"""Deterministic repair of near-miss flows before verification: snap misspelled action ids and input/output names to
the closest real one (only when very close), so small typos do not cost an escalation. Never invents anything new and
never removes steps. The only steps it may add are session steps the flow cannot run without (session_fix): the missing
open step before Excel / Word / browser steps that work on an open session, and the missing save step after Excel / Word
changes (nothing is saved automatically). Returns (flow, notes)."""
import copy, difflib, re


def bodies(n):
    """The step lists inside one container step (flow.v1.json): children, else, finally, elseIf[].children,
    catch[].children, core/switch cases[].children and default, core/state_machine states[].entry and
    states[].transitions[].steps."""
    for k in ("children", "else", "finally", "elseIf", "catch", "cases", "default"):
        v = n.get(k)
        if k in ("elseIf", "catch", "cases"):        # lists of blocks, each with its own children
            for b in v if isinstance(v, list) else []:
                if isinstance(b, dict) and isinstance(b.get("children"), list):
                    yield b["children"]
        elif isinstance(v, list):
            yield v
    for st in n.get("states") if isinstance(n.get("states"), list) else []:
        if not isinstance(st, dict):
            continue
        if isinstance(st.get("entry"), list):
            yield st["entry"]
        for t in st.get("transitions") if isinstance(st.get("transitions"), list) else []:
            if isinstance(t, dict) and isinstance(t.get("steps"), list):
                yield t["steps"]


def walk(nodes):
    """Every step, depth first, including the steps inside containers (see bodies). Same rules as ooa_reward.walk."""
    for n in nodes if isinstance(nodes, list) else []:
        if isinstance(n, dict):
            yield n
            for b in bodies(n):
                yield from walk(b)


def repair(flow, cat):
    """cat: dict action_id -> catalog entry."""
    flow = copy.deepcopy(flow)
    notes, ids = [], list(cat)
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
                    if not m:
                        m = _contained_match(k, valid, d, e, key)
                    if m and m[0] not in d:
                        notes.append(f"{a}.{key} {k} -> {m[0]}")
                        d[m[0]] = d.pop(k)
                    elif key == "args" and (d[k] is None or isinstance(d[k], (bool, int, float))):
                        # a made-up setting with a fixed value (e.g. "header": true): no data flows through it, so
                        # dropping it leaves the action's default; keys that carry data are left for the checker
                        notes.append(f"{a}: dropped unknown setting {k!r}")
                        d.pop(k)
    _fix_ids(flow, notes)
    _fix_variable_names(flow, notes)
    session_fix(flow, cat, notes)
    _declare_used_inputs(flow, notes, walk)
    return flow, notes


# toolkit -> (open action, its required input, variable name to declare for it)
OPENERS = {"ooa.excel": ("ooa.excel/open", "path", "workbook_path"),
           "ooa.word": ("ooa.word/open", "path", "document_path"),
           "ooa.browser": ("ooa.browser/open", "url", "url")}
OPENS = {"ooa.excel/open", "ooa.excel/create_workbook", "ooa.word/open", "ooa.word/create_document", "ooa.browser/open"}


CHANGES = re.compile(r"^(write|set|add|replace|insert|append|delete|clear|format|rename|merge|sort|create_(table|chart|pivot)|"
                     r"copy_sheet|remove|fill|hide|unhide|protect|unprotect|paste|move_sheet|auto_?fit)")
SAVES = {"ooa.excel": {"ooa.excel/save", "ooa.excel/save_as"}, "ooa.word": {"ooa.word/save", "ooa.word/save_as"}}


def session_fix(flow, cat, notes=None):
    """Make Excel / Word / browser session use runnable: add a missing open step, and a missing save step after changes.
    Also used by the data generators so training flows are runnable too. Returns notes."""
    notes = [] if notes is None else notes
    _add_missing_open(flow, notes, cat)
    _add_missing_save(flow, notes, cat)
    return notes


def _add_missing_save(flow, notes, cat):
    """An Excel / Word step changes the open file but nothing saves it (save, save as, or close with save on):
    add the save step at the end, before that toolkit's closing step when the flow ends with one."""
    steps = flow.get("steps")
    if not isinstance(steps, list):
        return
    flat = list(walk(steps))
    for t, saves in SAVES.items():
        acts = [(n.get("action") or "") for n in flat]
        changed = [k for k, a in enumerate(acts) if isinstance(a, str) and a.startswith(t + "/") and a not in OPENS
                   and CHANGES.match(a.split("/", 1)[1])]
        if not changed:
            continue
        last = changed[-1]
        saved = any(acts[k] in saves or (acts[k] == t + "/close" and str((flat[k].get("args") or {}).get("save", "")).lower()
                                         in ("true", "yes", "1")) for k in range(last + 1, len(flat)))
        if saved or (t + "/save") not in cat:
            continue
        step = {"id": "save_" + t.split(".")[-1], "action": t + "/save",
                "description": cat[t + "/save"]["description"][:140].rstrip("."), "args": {}}
        pos = len(steps)
        while pos > 0 and isinstance(steps[pos - 1], dict) and steps[pos - 1].get("action") == t + "/close":
            pos -= 1
        steps.insert(pos, step)
        notes.append(f"added the missing {t}/save step (the changes would not be saved)")


def _add_missing_open(flow, notes, cat):
    """Excel/Word/browser steps that act on an open session (they take 'session' and no file path of their own)
    used before any open/create step of that toolkit: insert the open step right before the first one."""
    opened = set()

    def needs_open(a):
        e = cat.get(a)
        if not e or a in OPENS:
            return False
        names = {i["name"] for i in e["inputs"]}
        req = {i["name"] for i in e["inputs"] if i["required"]}
        return "session" in names and "path" not in req

    def visit(nodes):
        if not isinstance(nodes, list):
            return
        k = 0
        while k < len(nodes):
            n = nodes[k]
            if not isinstance(n, dict):
                k += 1
                continue
            a = n.get("action") if isinstance(n.get("action"), str) else ""
            t = a.split("/")[0]
            if a in OPENS:
                opened.add(t)
            elif t in OPENERS and t not in opened and needs_open(a) and OPENERS[t][0] in cat:
                op, inp, var = OPENERS[t]
                variables = flow.setdefault("variables", [])
                if not any(isinstance(v, dict) and v.get("name") == var for v in variables):
                    variables.append({"name": var, "type": "String", "direction": "input", "required": True,
                                      "description": "Added by repair: the file or page to open first."})
                nodes.insert(k, {"id": "open_" + t.split(".")[-1], "action": op,
                                 "description": cat[op]["description"][:140].rstrip("."), "args": {inp: "$" + var}})
                notes.append(f"added the missing {op} step before {a}")
                opened.add(t)
                k += 1
            for b in bodies(n):
                visit(b)
            k += 1
    visit(flow.get("steps"))


def _contained_match(k, valid, d, e, key):
    """A wrong key that clearly means one real name: for inputs, the only missing REQUIRED input whose name is contained
    in the key or contains it ("target_folder" -> "target"); for outputs, the action's only output."""
    if key == "out":
        return valid if len(valid) == 1 else None
    missing = [i["name"] for i in e["inputs"] if i["required"] and i["name"] not in d]
    hits = [n for n in missing if n in k or k in n]
    return hits if len(hits) == 1 else None


_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _fix_ids(flow, notes):
    """Flow id must match ^[A-Za-z0-9_.-]{1,64}$ (schema); step ids must be unique."""
    fid = flow.get("id")
    if isinstance(fid, str):
        good = _ID.sub("_", fid.strip()).strip("_")[:64] or "flow"
        if good != fid:
            notes.append(f"flow id {fid!r} -> {good!r}")
            flow["id"] = good


_VAR_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _fix_variable_names(flow, notes):
    """Variable names must match ^[A-Za-z_][A-Za-z0-9_]{0,63}$. Fix stray spaces / bad characters (" new_name" ->
    "new_name") and rename every use in the steps. A name that becomes a duplicate of another is dropped."""
    variables = flow.get("variables")
    if not isinstance(variables, list):
        return
    taken = {v.get("name") for v in variables if isinstance(v, dict) and isinstance(v.get("name"), str)
             and _VAR_OK.match(v["name"])}
    renames, keep = {}, []
    for v in variables:
        name = v.get("name") if isinstance(v, dict) else None
        if not isinstance(name, str) or _VAR_OK.match(name):
            keep.append(v)
            continue
        good = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_")[:64]
        if good and good[0].isdigit():
            good = "v_" + good[:62]
        if not good:
            keep.append(v)                                  # nothing sensible to rename to: leave it for the checker
            continue
        renames[name] = good
        if good in taken:
            notes.append(f"variable {name!r} is the same as {good!r}: merged")
            continue
        v["name"] = good
        taken.add(good)
        keep.append(v)
        notes.append(f"variable {name!r} -> {good!r}")
    if not renames:
        return
    flow["variables"] = keep

    def fix(x):
        if isinstance(x, str):
            if x in renames:
                return renames[x]
            for old, new in renames.items():
                x = x.replace("${" + old + "}", "${" + new + "}").replace("$" + old, "$" + new)
            return x
        if isinstance(x, list):
            return [fix(i) for i in x]
        if isinstance(x, dict):
            return {k: fix(val) for k, val in x.items()}
        return x
    flow["steps"] = fix(flow.get("steps"))


def _declare_used_inputs(flow, notes, walk):
    """A step reads ${name} / $name but the flow never declares it: declare it as a String input the user fills in.
    Only for plain names that are not built in (sys, global) and not written by any step's out."""
    import ooa_reward as R
    variables = flow.get("variables")
    if not isinstance(variables, list):
        return
    declared = {v.get("name") for v in variables if isinstance(v, dict)}
    steps = list(walk(flow.get("steps")))
    written = {v for s in steps if isinstance(s.get("out"), dict) for v in s["out"].values() if isinstance(v, str)}
    loop_vars = {s["args"].get("as") for s in steps if isinstance(s.get("args"), dict) and isinstance(s["args"].get("as"), str)}
    ids = [s.get("id") for s in steps]
    if len(ids) != len(set(ids)):
        for k, s in enumerate(steps, 1):
            s["id"] = f"s{k}"
        notes.append("step ids renumbered (they were not unique)")
    for name in sorted((written | loop_vars) - declared):      # a step stores into it, or a loop names its item
        variables.append({"name": name, "type": "Any", "direction": "local"})
        declared.add(name)
        notes.append(f"declared missing local variable {name!r}")
    for name in sorted({r for s in steps if isinstance(s.get("args"), dict) for r in R.refs(s["args"])}):
        if name in declared:
            continue
        variables.append({"name": name, "type": "String", "direction": "input", "required": True,
                          "description": "Added by repair: the flow reads this value but did not declare it."})
        declared.add(name)
        notes.append(f"declared missing input variable {name!r}")
