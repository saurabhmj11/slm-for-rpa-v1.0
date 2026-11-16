"""The model's input / output format: one place that writes and reads it, for training data and for serving.

User turn (build_user):
    <request text>
    ### Context
    Variables: name (Type), ...                        (omitted when none)
    Actions:
    - <id>: <name> | in: a*, b | out: x, y            (* = required)
    Similar: <example request> => <id> > <id> > [loop: <id> > <id>]      (0-2 lines)
    ### Previous flow                                  (self-correction only)
    <compact JSON>
    ### Problems
    - <problem>

Assistant turn, one of:
    flow:     "Plan:\\n1. <step in plain words> (<action id>)\\n2. ...\\nFlow:\\n<compact flow JSON>"
              (plan lines of steps inside a loop / if are indented two spaces per level)
    clarify:  "Question: <one short question>"
    reply:    plain text (not a task)
parse_answer() also reads old answers that are only the flow JSON, so models trained before this format keep working.

The rendered user + assistant text must fit the model's 2048-token window (row_fits: at most 6,500 characters and
an estimated MAX_ROW_TOKENS estimated tokens, see approx_tokens): the context
is trimmed to MAX_CONTEXT_CHARS in a fixed order (see build_user) and a previous flow longer than MAX_PREVIOUS_CHARS
is shown as its outline. Pure standard library + retrieval.py.
"""
import hashlib
import json
import math
import re
from collections import Counter

import retrieval

FORMAT_VERSION = "plan-v1"
CONTEXT_HEADER = "### Context"
PREVIOUS_HEADER = "### Previous flow"
PROBLEMS_HEADER = "### Problems"
PLAN_HEAD, FLOW_HEAD, QUESTION_HEAD = "Plan:", "Flow:", "Question:"

DEFAULT_K = 12                # BM25 candidates per request
CORE_ALWAYS = ("core/loop", "core/if")   # "for each" / "if" words are stop words for BM25: always offer these two
MAX_EXAMPLES = 2
MAX_CONTEXT_CHARS = 2600      # variables + actions + similar lines
EXAMPLE_ACTIONS = 6           # actions of the similar examples added to the candidates (measured: recall of the
                              # needed actions on the plain-request sets 0.77 -> 0.86 and 0.65 -> 0.82)
MAX_PREVIOUS_CHARS = 1600     # a longer previous flow is shown as its outline (1600: measured on gen_fix rows, a
                              # quarter fewer self-correction rows lose the window than at 2400)
MAX_PROBLEMS = 6
MAX_PROBLEM_CHARS = 200
MAX_ROW_CHARS = 6500          # user + assistant: about 2,100 tokens at the measured 2.8-3.1 characters per token
MAX_ROW_TOKENS = 2020         # approx_tokens(user + assistant): 99% of such rows fit the 2048-token window
MAX_OUTLINE_CHARS = 160
MAX_EXAMPLE_REQUEST_CHARS = 140
MAX_VARIABLES = 12
MAX_PLAN_TEXT = 90            # longer step descriptions are replaced by the catalog name
MIN_ACTIONS = 4               # trimming never goes below this many action lines (required ones always stay)
# per-clause retrieval for long requests (context_actions): a whole business process names many apps and steps, and one
# BM25 query over all of it finds the few words that dominate. So a request of CLAUSE_MIN clauses or more also gets, for
# each clause (request_clauses), its BM25 top CLAUSE_K and the CLAUSE_VOTES actions most used by the training requests
# most like it (ExampleIndex.vote), plus WHOLE_VOTES voted for the whole request. Measured recall of the needed actions in
# the context (eval/plain_requests*.jsonl, sets 1 / 2 / 3, 4,800-character budget): before 0.88 / 0.89 / 0.56 (18
# actions offered on set 3); top-3 per clause + each clause's most similar example 0.93 / 0.92 / 0.69 (34 actions);
# this setting 0.98 / 0.92 / 0.75 (30 actions); with the 2,600-character budget 0.98 / 0.92 / 0.70. A one-clause
# request gets exactly the context it got before.
CLAUSE_MIN = 2                # requests with at least this many clauses get per-clause candidates
CLAUSE_K = 2                  # BM25 hits per clause
CLAUSE_VOTES = 3              # voted actions per clause
WHOLE_VOTES = 4               # voted actions for the whole request
VOTE_EXAMPLES = 20            # similar training requests that vote
MAX_CONTEXT_ACTIONS = 40      # never more action lines than this (the context budget trims further)
FOUND_FIRST = 6               # a long request's whole-request BM25 hits ranked before the per-clause candidates; the
                              # rest come after them, so a 2,600-character budget keeps the per-clause ones (set-3
                              # recall at 2,600: 0.63 with all 12 first, 0.70 with 6)
LONG_WINDOW = 4096            # a model with this sequence length or more gets the larger context budget:
MAX_CONTEXT_CHARS_LONG = 4800 # characters of context for such a model (about 1,700 tokens, leaving 2,300 for the answer)

# branches of a container step and the words a plan line starts with in that branch
_BRANCH_WORDS = {"elseIf": "Else if: ", "else": "Otherwise: ", "catch": "On error: ", "finally": "Finally: ",
                 "cases": "Case: ", "default": "Default: ", "entry": "State: ", "transition": "Then: "}


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------------------------- flow walking
def branches(step):
    """(branch name, step list) for every body of a container step, in document order. Same bodies as
    ooa_reward.bodies; the name says which branch it is ("children", "elseIf", "else", ...)."""
    for k in ("children", "elseIf", "else", "catch", "finally", "cases", "default"):
        v = step.get(k)
        if k in ("elseIf", "catch", "cases"):
            for b in v if isinstance(v, list) else []:
                if isinstance(b, dict) and isinstance(b.get("children"), list):
                    yield k, b["children"]
        elif isinstance(v, list):
            yield k, v
    for st in step.get("states") if isinstance(step.get("states"), list) else []:
        if not isinstance(st, dict):
            continue
        if isinstance(st.get("entry"), list):
            yield "entry", st["entry"]
        for t in st.get("transitions") if isinstance(st.get("transitions"), list) else []:
            if isinstance(t, dict) and isinstance(t.get("steps"), list):
                yield "transition", t["steps"]


def walk(steps, depth=0):
    """(step, depth) for every step, depth first."""
    for s in steps if isinstance(steps, list) else []:
        if isinstance(s, dict):
            yield s, depth
            for _, body in branches(s):
                yield from walk(body, depth + 1)


def actions_of(flow):
    return [s.get("action") for s, _ in walk(flow.get("steps") if isinstance(flow, dict) else flow)]


# ---------------------------------------------------------------------------------------------- variable notation
# Since 14 Nov 2026 the platform writes variables $name$ (as AA does): Studio saves whole-field references and
# calculations as $name$, and the runner replaces only $name$ (or ${name}) inside text, so "Report for $month" is left
# as typed. to_dollar_name_dollar() rewrites a flow the same way (Studio's upgradeRefs in platform/ui/src/studio/
# variables.ts) and also mends text: a known variable written $name or ${name} inside text becomes $name$.
_PATH = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+|\[[^\]$]+\])*"
_WHOLE_REF = re.compile(r"^\$(?:\{\s*)?(" + _PATH + r")(?:\s*\}|\$)?$")
_EXPR_REF = re.compile(r"\$\{?\s*(" + _PATH + r")(\s*\})?(?![A-Za-z0-9_.\[$])")
_TEXT_REF = re.compile(r"(?P<done>\$" + _PATH + r"\$)|\$\{\s*(?P<brace>" + _PATH + r")\s*\}|(?<!\[)\$(?P<bare>" + _PATH + r")(?![$\[\w]|\.\w)")
# inputs whose text is code or a pattern with its own $ words: only whole-field references are rewritten there
_CODE_INPUTS = {"script", "code", "command", "pattern", "name_pattern", "regex", "expression", "formula", "js",
                "javascript", "python", "powershell", "vbscript", "macro"}
BUILTIN_ROOTS = ("sys", "global")


def var_ref(name):
    """How a variable is written in a field: $name$."""
    return "$" + name + "$"


def _root(path):
    return re.match(r"[A-Za-z_][A-Za-z0-9_]*", path).group(0)


def _fix_text(s, known):
    def sub(m):
        if m.group("done"):
            return m.group(0)
        path = m.group("brace") or m.group("bare")
        return var_ref(path) if _root(path) in known else m.group(0)
    return _TEXT_REF.sub(sub, s)


def _fix_value(v, known, text, key=None):
    if isinstance(v, str):
        t = v.strip()
        m = _WHOLE_REF.match(t)
        if m:
            return v if t.endswith("$") else var_ref(m.group(1))
        return _fix_text(v, known) if text and key not in _CODE_INPUTS and "$" in v else v
    if isinstance(v, list):
        return [_fix_value(x, known, text, key) for x in v]
    if isinstance(v, dict):
        if len(v) == 1 and "literal" in v:
            return v
        if len(v) == 1 and isinstance(v.get("expr"), str):
            return {"expr": _EXPR_REF.sub(lambda m: m.group(0) if m.group(0).startswith("${") and not m.group(2)
                                          else var_ref(m.group(1)), v["expr"])}
        if isinstance(v.get("action"), str):
            return _fix_step(v, known)
        return {k: _fix_value(x, known, text, k if text else None) for k, x in v.items()}
    return v


def _fix_step(step, known):
    text = step.get("action") != "core/comment"
    out = {}
    for k, v in step.items():
        if k == "note":                       # the builder's own words
            out[k] = v
        elif k == "args":
            out[k] = _fix_value(v, known, text)
        else:
            out[k] = _fix_value(v, known, False)
    return out


def variable_names(flow):
    """Names a flow declares or makes: its variables, out targets, loop / set targets, plus sys and global."""
    names = set(BUILTIN_ROOTS)
    if not isinstance(flow, dict):
        return names
    names |= {v["name"] for v in flow.get("variables") or [] if isinstance(v, dict) and isinstance(v.get("name"), str)}
    bodies = [flow.get("steps")] + [sf.get("steps") for sf in flow.get("subflows") or [] if isinstance(sf, dict)]
    for body in bodies:
        for s, _ in walk(body):
            names |= {x for x in (s.get("out") or {}).values() if isinstance(x, str)} if isinstance(s.get("out"), dict) else set()
            args = s.get("args") if isinstance(s.get("args"), dict) else {}
            names |= {args[k] for k in ("variable", "as", "index") if isinstance(args.get(k), str)}
    return names


def to_dollar_name_dollar(flow, variable_names_=None):
    """The flow with its variable references written $name$, as the platform saves them: a whole field ($name,
    ${name}, with .field / [index] parts) and every reference in a calculation ({"expr": ...}) becomes $name$; inside
    other text of a step's args a known variable written $name or ${name} becomes $name$ too (the runner would leave
    it as typed). Left alone: notes, an index that is itself a variable ($rates[$reason]), {"literal": ...}, a $ with no name after it, spreadsheet references such as $F$1,
    unknown names inside text, comment steps and code / pattern inputs (script, code, pattern ...). Known names are
    variable_names_ plus what the flow declares or makes (variable_names). Returns a new flow; running it again
    changes nothing. A list of steps is accepted too."""
    if isinstance(flow, list):
        known = set(variable_names_ or ()) | variable_names({"steps": flow})
        return [_fix_value(s, known, False) for s in flow]
    if not isinstance(flow, dict):
        return flow
    known = set(variable_names_ or ()) | variable_names(flow)
    out = dict(flow)
    if isinstance(flow.get("steps"), list):
        out["steps"] = [_fix_value(s, known, False) for s in flow["steps"]]
    if isinstance(flow.get("subflows"), list):
        out["subflows"] = [{**sf, "steps": [_fix_value(s, known, False) for s in sf["steps"]]}
                           if isinstance(sf, dict) and isinstance(sf.get("steps"), list) else sf
                           for sf in flow["subflows"]]
    return out


def flow_signature(flow):
    """Short hash of the flow's compact JSON: two rows with the same signature teach the same answer."""
    return hashlib.sha1(compact(flow).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- outline
def _kind(action):
    a = str(action or "?")
    return a.split("/", 1)[1] if a.startswith("core/") else a


def _outline_steps(steps):
    return " > ".join(_outline_step(s) for s in steps if isinstance(s, dict))


def _outline_step(s):
    a = s.get("action") or "?"
    parts, first = [], True
    for name, body in branches(s):
        inner = _outline_steps(body)
        if first and name == "children":
            parts.append(f"{_kind(a)}: {inner}")
        else:
            if first:
                parts.append(_kind(a))
            parts.append(f"{name}: {inner}")
        first = False
    return "[" + " | ".join(parts) + "]" if parts else a


def outline(flow):
    """Compact action outline: 'ooa.file/list > [loop: ooa.file/read_text > [if: ooa.email/send | else: ooa.file/move]]'."""
    steps = flow.get("steps") if isinstance(flow, dict) else flow
    return _outline_steps(steps if isinstance(steps, list) else [])


def truncate_outline(text, limit=MAX_OUTLINE_CHARS):
    """Cut an outline at a step boundary, mark the cut with '...', and close any open brackets."""
    if len(text) <= limit:
        return text
    cut = text[:max(0, limit - 6)]
    i = cut.rfind(" > ")
    cut = cut[:i] if i > 0 else cut
    cut = cut.rstrip(" >|:")
    return cut + " > ..." + "]" * max(0, cut.count("[") - cut.count("]"))


# ---------------------------------------------------------------------------------------------- plan
def _plain(text):
    if not isinstance(text, str):
        return None
    t = " ".join(text.split()).rstrip(".")
    if not (3 <= len(t) <= MAX_PLAN_TEXT) or any(c in t for c in "${}\"`") or "/" in t.split(" ")[0]:
        return None
    return t


def _var(v):
    """The name a whole-field reference points to ($name$, $name or ${name}), else None."""
    m = _WHOLE_REF.match(v.strip()) if isinstance(v, str) else None
    return m.group(1) if m else None


def step_text(step, cat=None):
    """Plain words for one step: its description when that is short plain text, else words built from a loop's
    arguments, else the catalog name, else the action id."""
    t = _plain(step.get("description"))
    if t:
        return t
    a, args = step.get("action") or "?", step.get("args") if isinstance(step.get("args"), dict) else {}
    if a == "core/loop":
        over, as_ = _var(args.get("over")), args.get("as")
        if over and isinstance(as_, str) and as_:
            return f"For each {as_} in {over}"
        if args.get("count") is not None:
            return f"Repeat {args['count']} times" if isinstance(args["count"], int) else "Repeat a number of times"
        return "Repeat"
    if a == "core/if":
        c = args.get("condition")
        if isinstance(c, dict) and {"op", "left"} <= set(c) and isinstance(c.get("left"), str):
            right = c.get("right")
            right = right if isinstance(right, (str, int, float)) and len(str(right)) <= 30 else "a value"
            right = _var(right) or right
            return f"If {_var(c['left']) or c['left']} {c['op']} {right}"
        return "If the condition holds"
    if a == "core/set" and isinstance(args.get("variable"), str) and args["variable"]:
        return f"Set {args['variable']}"
    e = (cat or {}).get(a)
    if e and e.get("name"):
        return e["name"]
    return a


def plan_from_flow(flow, catalog=None):
    """Plan lines for a flow: '<n>. <plain words> (<action id>)', one per step in document order, children indented
    two spaces per level; the first step of an else / else-if / catch / ... branch starts with that branch's word."""
    cat = _as_dict(catalog)
    out = []

    def go(steps, depth, prefix=""):
        first = True
        for s in steps if isinstance(steps, list) else []:
            if not isinstance(s, dict):
                continue
            a = s.get("action") or "?"
            text = (prefix if first else "") + step_text(s, cat)
            first = False
            out.append("  " * depth + f"{len(out) + 1}. {text} ({a})")
            for name, body in branches(s):
                go(body, depth + 1, "" if name == "children" else _BRANCH_WORDS.get(name, ""))

    go(flow.get("steps") if isinstance(flow, dict) else flow, 0)
    return out


def render_flow_answer(flow, catalog=None):
    return PLAN_HEAD + "\n" + "\n".join(plan_from_flow(flow, catalog)) + "\n" + FLOW_HEAD + "\n" + compact(flow)


_PLAN_LINE = re.compile(r"^(?P<indent>\s*)(?:\d+[.)]\s*)?(?P<text>.*?)\s*(?:\((?P<action>[\w.\-]+/[\w.\-]+)\))?\s*$")
_BRANCH_PREFIX = re.compile(r"^(?:" + "|".join(re.escape(w.strip()) for w in _BRANCH_WORDS.values()) + r")\s*")


def plan_items(plan):
    """Plan lines -> [{"depth", "text", "action", "line"}]. text has no number, branch word or action id."""
    items = []
    for line in plan or []:
        if not line.strip():
            continue
        m = _PLAN_LINE.match(line.rstrip())
        indent = len(m.group("indent").replace("\t", "  "))
        items.append({"depth": indent // 2, "text": _BRANCH_PREFIX.sub("", m.group("text")).strip(),
                      "action": m.group("action"), "line": line.rstrip()})
    return items


def plan_actions(plan):
    return [i["action"] for i in plan_items(plan) if i["action"]]


def plan_coverage(plan, flow):
    """Action ids the plan names that no step of the flow uses (empty: the flow covers the plan)."""
    have = set(actions_of(flow))
    return [a for a in dict.fromkeys(plan_actions(plan)) if a not in have]


def plan_groups(plan):
    """Top-level plan items, each with the nested items that follow it: [[top, child, child], [top], ...]."""
    groups = []
    for it in plan_items(plan):
        if it["depth"] == 0 or not groups:
            groups.append([it])
        else:
            groups[-1].append(it)
    return groups


def chunk_plan(plan, min_size=2, max_size=4, max_lines=8):
    """Split a plan into parts for decomposition: 2-4 consecutive top-level items (with their nested lines) each,
    at most max_lines lines per part unless a single item is bigger. A last part of one item joins the part
    before it when that part still has room; a plan of one part is returned as one part."""
    chunks = []
    for g in plan_groups(plan):
        cur = chunks[-1] if chunks else None
        if cur is not None and len(cur) < max_size and sum(len(x) for x in cur) + len(g) <= max_lines:
            cur.append(g)
        else:
            chunks.append([g])
    if len(chunks) >= 2 and len(chunks[-1]) < min_size:
        prev = chunks[-2]
        if len(prev) + len(chunks[-1]) <= max_size + 1:
            prev.extend(chunks.pop())
    return chunks


def _subgroups(group):
    """The direct children of a group's first item, each with its own nested lines: [[child, grandchild ...], ...]."""
    d, out = group[0]["depth"], []
    for it in group[1:]:
        if it["depth"] == d + 1 or not out:
            out.append([it])
        else:
            out[-1].append(it)
    return out


def plan_parts(plan, max_lines=5, max_items=4):
    """Parts to build a long plan in, smaller than chunk_plan's, that may also cut a loop's body: a list of
    {"chunk": [groups], "inside": bool}. chunk is what chunk_request / chunk_actions read (groups as plan_groups);
    inside=True means the part's steps go inside the loop that ends the parts before it (the loop's own line and its
    first children are in the earlier part). Top-level groups are packed up to max_items items / max_lines lines; a
    group longer than max_lines whose first item is a loop (core/loop) is cut about evenly: the loop line with its first
    children, then the rest of its children (each child with its nested lines, never cut) packed the same way. Any other long
    group (an if, a try) stays one part. A plan of one part comes back as one part."""
    units = []                                        # (groups, inside)
    for g in plan_groups(plan):
        if len(g) <= max_lines or g[0]["action"] != "core/loop" or len(_subgroups(g)) < 2:
            units.append(([g], False))
            continue
        subs = _subgroups(g)
        target = -(-len(g) // -(-len(g) // max_lines))        # lines per piece when the loop is cut evenly
        head, rest = [g[0]] + subs[0], subs[1:]
        while len(rest) > 1 and len(head) + len(rest[0]) <= target:
            head += rest.pop(0)
        units.append(([head], False))
        cur = []
        for sg in rest:
            if cur and (len(cur) >= max_items or sum(len(x) for x in cur) + len(sg) > max_lines):
                units.append((cur, True))
                cur = []
            cur.append(sg)
        if cur:
            units.append((cur, True))
    parts = []
    for i, (groups, inside) in enumerate(units):
        last = parts[-1] if parts else None
        if (not inside and last is not None and not last["inside"] and not last["open"]
                and len(last["chunk"]) + len(groups) <= max_items
                and sum(len(x) for x in last["chunk"]) + sum(len(x) for x in groups) <= max_lines):
            last["chunk"].extend(groups)
        else:
            parts.append({"chunk": list(groups), "inside": inside, "open": False})
        # a cut loop's head ends its part: the inside part after it goes into that loop
        parts[-1]["open"] = i + 1 < len(units) and units[i + 1][1]
    return [{"chunk": p["chunk"], "inside": p["inside"]} for p in parts]


def insert_into_last_loop(steps, part_steps):
    """Add part_steps to the body of the last top-level step of steps (a loop cut by plan_parts). A part that wrapped
    its steps in the same loop again (one core/loop over the same item) is unwrapped first. Returns False when the
    last step is not a loop."""
    if not steps or not isinstance(steps[-1], dict) or steps[-1].get("action") != "core/loop":
        return False
    loop = steps[-1]
    body = part_steps
    if (len(body) == 1 and isinstance(body[0], dict) and body[0].get("action") == "core/loop"
            and isinstance(body[0].get("children"), list)
            and (body[0].get("args") or {}).get("as") == (loop.get("args") or {}).get("as")):
        body = body[0]["children"]
    loop.setdefault("children", [])
    loop["children"] = list(loop["children"]) + list(body)
    return True


def chunk_request(chunk):
    """Plain request text for one part: 'Read the CSV file, then for each row: send an email, then log it.'"""
    def item_text(group, i, depth):
        it = group[i]
        kids, j = [], i + 1
        while j < len(group) and group[j]["depth"] > depth:
            if group[j]["depth"] == depth + 1:
                kids.append(j)
            j += 1
        t = it["text"] or it["action"] or ""
        if kids:
            t += ": " + "".join((", " if n else "") + _joined(group, k, depth + 1, n) for n, k in enumerate(kids))
        return t

    def _joined(group, k, depth, n):
        m = _BRANCH_PREFIX.match(_PLAN_LINE.match(group[k]["line"]).group("text") or "")
        word = m.group(0).strip().lower() + " " if m else ("then " if n else "")
        return word + item_text(group, k, depth)

    parts = [item_text(g, 0, g[0]["depth"]) for g in chunk if g]
    text = ", then ".join(p for p in parts if p)
    return (text[:1].upper() + text[1:]).rstrip(".") + "."


def chunk_plan_lines(chunk):
    """The plan lines of one part, as plan_from_flow writes a part flow's plan: numbered from 1, indented from the
    part's own top level, branch words and action ids kept ("Otherwise: Move it (ooa.file/move)")."""
    items = [it for g in chunk for it in g]
    if not items:
        return []
    base = min(it["depth"] for it in items)
    out = []
    for n, it in enumerate(items, 1):
        m = _BRANCH_PREFIX.match(_PLAN_LINE.match(it["line"]).group("text") or "")
        text = (m.group(0) if m else "") + (it["text"] or it["action"] or "")
        out.append("  " * (it["depth"] - base) + f"{n}. {text}" + (f" ({it['action']})" if it["action"] else ""))
    return out


def answer_prefix(plan_lines):
    """The start of a plan-first answer given its plan: "Plan:\n<lines>\nFlow:\n" (the model writes the flow)."""
    return PLAN_HEAD + "\n" + "\n".join(plan_lines) + "\n" + FLOW_HEAD + "\n"


def chunk_actions(chunk):
    return list(dict.fromkeys(it["action"] for g in chunk for it in g if it["action"]))


def merge_flows(parts, description=None, inside=None):
    """Join part flows in order: steps appended, step ids made unique (a repeated id gets the next free sN), variables
    united (the first declaration of a name wins). inside: one bool per part (plan_parts); a True part's steps go into
    the loop that ends the steps so far (insert_into_last_loop), else after them."""
    if not parts:
        return None
    base = {k: v for k, v in parts[0].items() if k not in ("variables", "steps")}
    if description:
        base["description"] = description
    variables, names, steps, used = [], set(), [], set()
    for k, p in enumerate(parts):
        for v in p.get("variables") or []:
            if isinstance(v, dict) and v.get("name") not in names:
                names.add(v.get("name"))
                variables.append(json.loads(compact(v)))
        new = json.loads(compact(p.get("steps") or []))
        for s, _ in walk(new):
            sid = s.get("id")
            if not sid or sid in used:
                n = len(used) + 1
                while f"s{n}" in used:
                    n += 1
                s["id"] = f"s{n}"
            used.add(s["id"])
        if not (inside and k < len(inside) and inside[k] and insert_into_last_loop(steps, new)):
            steps.extend(new)
    return {**base, "variables": variables, "steps": steps}


_REF = re.compile(r"\$\{?\s*([A-Za-z_]\w*)")


def _names_used(steps):
    """Variable names a list of steps reads or writes (references in args, out targets, core/set targets)."""
    names = set(_REF.findall(compact(steps)))
    for s, _ in walk(steps):
        for v in (s.get("out") or {}).values() if isinstance(s.get("out"), dict) else []:
            if isinstance(v, str):
                names.add(v)
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        for key in ("variable", "as", "index"):
            if isinstance(args.get(key), str):
                names.add(args[key])
    return names


def split_flow(flow, catalog=None, **chunk_kw):
    """A long flow -> its parts as decomposition sees them: [(part request, part flow, carried variables)].
    The parts follow chunk_plan over plan_from_flow (one top-level plan item per top-level step), each part request
    is chunk_request of its plan lines, each part flow has that part's top-level steps and the variables they use,
    and carried variables are those made by an earlier part or given as flow inputs (name, type), the way the
    server passes them when it builds a flow in parts. Empty list when the flow is one part."""
    steps = flow.get("steps") if isinstance(flow, dict) else None
    if not isinstance(steps, list):
        return []
    plan = plan_from_flow(flow, catalog)
    chunks = chunk_plan(plan, **chunk_kw)
    if len(chunks) < 2:
        return []
    top = [s for s in steps if isinstance(s, dict)]
    variables = [v for v in flow.get("variables") or [] if isinstance(v, dict) and v.get("name")]
    out, i, made = [], 0, set()
    for n, chunk in enumerate(chunks, 1):
        part_steps = json.loads(compact(top[i:i + len(chunk)]))
        i += len(chunk)
        used = _names_used(part_steps)
        part_vars = [v for v in variables if v["name"] in used]
        carried = [(v["name"], v.get("type") or "Any") for v in part_vars
                   if v["name"] in made or v.get("direction") == "input"]
        made |= used
        part = {k: v for k, v in flow.items() if k not in ("variables", "steps")}
        part.update(id=f"{flow.get('id') or 'flow'}_part{n}", description=chunk_request(chunk),
                    variables=json.loads(compact(part_vars)), steps=part_steps)
        out.append((chunk_request(chunk), part, carried))
    return out


# ---------------------------------------------------------------------------------------------- user turn
def _as_dict(catalog):
    if catalog is None:
        return {}
    if isinstance(catalog, dict):
        return catalog
    return {e["id"]: e for e in catalog}


def action_line(e):
    ins = e.get("inputs") or []
    req = [i["name"] + "*" for i in ins if i.get("required")]
    opt = [i["name"] for i in ins if not i.get("required")][:4]
    outs = [o["name"] for o in e.get("outputs") or []][:4]
    s = f"- {e['id']}: {e.get('name') or e['id']} | in: {', '.join(req + opt) or '-'}"
    if outs:
        s += f" | out: {', '.join(outs)}"
    return s


def _variables(variables):
    out = []
    for v in variables or []:
        if isinstance(v, dict):
            name, typ = v.get("name"), v.get("type")
        elif isinstance(v, (list, tuple)) and v:
            name, typ = v[0], (v[1] if len(v) > 1 else None)
        else:
            continue
        if isinstance(name, str) and name and name not in [n for n, _ in out]:
            out.append((name, typ if isinstance(typ, str) and typ else "Any"))
    return out


def _example_line(ex):
    if isinstance(ex, dict):
        req, ol = ex.get("request", ""), ex.get("outline", "")
    else:
        req, ol = ex[0], ex[1]
    req = " ".join(str(req).split())
    if len(req) > MAX_EXAMPLE_REQUEST_CHARS:
        req = req[:MAX_EXAMPLE_REQUEST_CHARS - 3].rsplit(" ", 1)[0] + "..."
    return f"Similar: {req} => {truncate_outline(str(ol))}"


def _context(actions, examples, variables):
    lines = [CONTEXT_HEADER]
    if variables:
        lines.append("Variables: " + ", ".join(f"{n} ({t})" for n, t in variables))
    lines.append("Actions:")
    lines += [action_line(e) for e in actions]
    lines += [_example_line(x) for x in examples]
    return "\n".join(lines)


def trim_context(actions, examples, variables, required=(), limit=MAX_CONTEXT_CHARS):
    """Drop context until it fits `limit` characters, always in this order: the second example, action lines
    beyond 8 (from the end; required ids stay), the last example, more action lines down to MIN_ACTIONS,
    variables beyond 4. Returns (actions, examples, variables)."""
    actions, examples, variables = list(actions), list(examples)[:MAX_EXAMPLES], list(variables)[:MAX_VARIABLES]
    req = set(required)

    def size():
        return len(_context(actions, examples, variables))

    def drop_action(floor):
        for i in range(len(actions) - 1, -1, -1):
            if actions[i]["id"] not in req and len(actions) > floor:
                actions.pop(i)
                return True
        return False

    while size() > limit and len(examples) > 1:
        examples.pop()
    while size() > limit and drop_action(8):
        pass
    while size() > limit and examples:
        examples.pop()
    while size() > limit and drop_action(MIN_ACTIONS):
        pass
    while size() > limit and len(variables) > 4:
        variables.pop()
    return actions, examples, variables


def build_user(request, actions=(), examples=(), variables=(), previous=None, problems=None, required=(),
               context_limit=MAX_CONTEXT_CHARS):
    """The user turn. actions: catalog entries; examples: (request, outline) pairs or {"request", "outline"} dicts;
    variables: (name, type) pairs or variable dicts; previous: the flow to correct (dict or JSON text);
    problems: what is wrong with it; required: action ids that trimming must keep; context_limit: characters for
    the context block (MAX_CONTEXT_CHARS; render_chat uses less for rows that would not fit the window)."""
    request = " ".join(str(request).split("\n")).strip() if "\n" in str(request) else str(request).strip()
    acts, exs, vs = trim_context([e for e in actions if isinstance(e, dict) and e.get("id")], examples,
                                 _variables(variables), required, context_limit)
    text = request + "\n" + _context(acts, exs, vs)
    if previous is not None:
        prev = previous if isinstance(previous, str) else compact(previous)
        if len(prev) > MAX_PREVIOUS_CHARS:
            try:
                prev = "Outline: " + truncate_outline(outline(json.loads(prev)), MAX_PREVIOUS_CHARS)
            except Exception:
                prev = prev[:MAX_PREVIOUS_CHARS - 4] + " ..."
        text += "\n" + PREVIOUS_HEADER + "\n" + prev
    if problems:
        probs = [" ".join(str(p).split())[:MAX_PROBLEM_CHARS] for p in problems][:MAX_PROBLEMS]
        text += "\n" + PROBLEMS_HEADER + "\n" + "\n".join("- " + p for p in probs)
    return text


def request_of(user_text):
    """The request part of a user turn (the text before '### Context'); a plain request comes back unchanged."""
    t = str(user_text)
    i = t.find("\n" + CONTEXT_HEADER)
    return (t[:i] if i >= 0 else t).strip()


def split_user(content):
    """(request, variables) from a user turn in any format seen in data and at serving: plain text, Studio's
    'Existing variables: [...]\\nRequest: ...', retrieval.make_prompt's 'Actions: ...\\n\\nRequest: ...', or this
    format (request + context)."""
    t = str(content)
    if t.startswith("Existing variables:") and "\nRequest:" in t:
        try:
            vs = json.loads(t[len("Existing variables:"):t.index("\nRequest:")])
        except Exception:
            vs = []
        return t.split("\nRequest:", 1)[1].strip(), _variables(vs if isinstance(vs, list) else [])
    if t.startswith("Actions:") and "\n\nRequest: " in t:
        return t.split("\n\nRequest: ", 1)[1].strip(), []
    return request_of(t), []


# ---------------------------------------------------------------------------------------------- assistant turn
def parse_answer(text):
    """{"kind": "flow"|"question"|"reply", "plan": [lines], "flow_text": str|None, "question": str|None, "text": str}.
    An answer that is only flow JSON (the format before plans) is kind "flow" with an empty plan."""
    t = str(text or "").split("<|assistant_end|>")[0].strip()
    res = {"kind": "reply", "plan": [], "flow_text": None, "question": None, "text": t}
    if t.startswith(QUESTION_HEAD):
        q = t[len(QUESTION_HEAD):].strip()
        res.update(kind="question", question=q.split("\n\n")[0].strip() or q)
    elif t.startswith(PLAN_HEAD) or t.startswith(FLOW_HEAD):
        body = t[len(PLAN_HEAD):] if t.startswith(PLAN_HEAD) else "\n" + t
        m = re.search(r"(?m)^[ \t]*" + re.escape(FLOW_HEAD) + r"[ \t]*", body)
        plan_part = body[:m.start()] if m else body
        res.update(kind="flow", plan=[l.rstrip() for l in plan_part.split("\n") if l.strip()],
                   flow_text=body[m.end():].strip() if m else None)
    elif t.startswith("{") or t.startswith("```"):
        res.update(kind="flow", flow_text=t)
    return res


# ---------------------------------------------------------------------------------------------- retrieval context
_CLAUSE_SPLIT = re.compile(r"[;:]|,(?!\d)|\.(?!\w)|\b(?:then|and|if|for each|every|otherwise|else|after that)\b", re.I)


def request_clauses(request, min_terms=2):
    """The request cut into clauses (on , ; : . and the words then / and / if / for each / every / otherwise / else /
    after that), each kept when it has at least min_terms search terms (retrieval.tokens): 'For each new invoice PDF
    in the inbox folder, read the total; if it's over 10,000 email it' -> ['new invoice PDF in the inbox folder',
    'read the total', "it's over 10,000 email it"] (10,000 stays whole)."""
    out = []
    for part in _CLAUSE_SPLIT.split(str(request or "")):
        part = " ".join(part.split()).strip(" .,'\"")
        if part and len(retrieval.tokens(part)) >= min_terms and part not in out:
            out.append(part)
    return out


def window_of(max_row_tokens):
    """The window (tokens) a training row limit is meant for: 4096 when the limit is over the 2048-token one."""
    return LONG_WINDOW if max_row_tokens and max_row_tokens > 2048 else 2048


def context_for(seq_len):
    """Characters of context for a model with this sequence length (None: unknown -> the 2048-token budget)."""
    return MAX_CONTEXT_CHARS_LONG if isinstance(seq_len, int) and seq_len >= LONG_WINDOW else MAX_CONTEXT_CHARS


def _clause_ids(request, cat, index, example_index, clause_k, clause_votes, exclude_request, exclude_sig):
    """Extra candidates of a request of CLAUSE_MIN clauses or more, best first: the actions voted for the whole request
    (WHOLE_VOTES), then per clause in rounds (the clause's voted actions and its BM25 hits over the catalog, rank 1 of
    every clause, then rank 2 ...). Empty for a shorter request."""
    cl = request_clauses(request)
    if index is None or len(cl) < CLAUSE_MIN:
        return []
    ex = dict(exclude_request=exclude_request, exclude_sig=exclude_sig)
    ids = example_index.vote(request, WHOLE_VOTES, **ex) if example_index is not None else []
    hits = [[e["id"] for e in index.search(c, clause_k)] if clause_k > 0 else [] for c in cl]
    votes = [example_index.vote(c, clause_votes, **ex) if example_index is not None and clause_votes > 0 else []
             for c in cl]
    for r in range(max(clause_k, clause_votes)):
        for h, v in zip(hits, votes):
            ids += v[r:r + 1] + h[r:r + 1]
    return [a for a in dict.fromkeys(ids) if a in cat]


def context_actions(request, gold=(), catalog=None, index=None, k=DEFAULT_K, rng=None, examples=(), example_index=None,
                    clause_k=CLAUSE_K, clause_votes=CLAUSE_VOTES, exclude_request=None, exclude_sig=None,
                    max_actions=MAX_CONTEXT_ACTIONS):
    """Action entries for the context and the ids trimming must keep. Training and serving both call this, so the
    model is trained on the context it gets.
    examples: the similar examples shown (dicts with "actions"); up to EXAMPLE_ACTIONS of their actions are added.
    Serving (no gold), best first: the examples' actions, BM25 top-k for the request, then CORE_ALWAYS. A request of
    CLAUSE_MIN clauses or more: the examples' actions, the first FOUND_FIRST BM25 hits, the actions voted for it and
    per clause clause_k BM25 hits and clause_votes voted actions (votes need example_index; see _clause_ids), the
    other BM25 hits, then CORE_ALWAYS; at most max_actions. Trimming (build_user)
    drops from the end, so a smaller window loses the per-clause extras first. Training: the gold flow's actions are
    always in (they are required), plus CORE_ALWAYS and the examples' actions, filled up with the serving candidates to
    the same size, shuffled with rng so the position of an action says nothing. exclude_request / exclude_sig keep the
    row itself out of the votes (training)."""
    cat = _as_dict(catalog) or (index.by_id if index is not None else {})
    gold = [a for a in dict.fromkeys(gold) if a in cat]
    found = [e["id"] for e in index.search(request, k)] if index is not None else []
    core = [c for c in CORE_ALWAYS if c in cat]
    shown = []
    for x in examples or []:
        for a in x.get("actions") or [] if isinstance(x, dict) else []:
            if a in cat and a not in shown and a not in gold and a not in core and len(shown) < EXAMPLE_ACTIONS:
                shown.append(a)
    more = _clause_ids(request, cat, index, example_index, clause_k, clause_votes, exclude_request, exclude_sig)
    if gold:
        ids = list(dict.fromkeys(gold + core + shown))
        pool = [a for a in dict.fromkeys(found[:FOUND_FIRST] + more + found) if a not in ids]
        extra = len([a for a in dict.fromkeys(more) if a not in found and a not in shown and a not in core])
        cap = max(min(k + 2 + len(shown) + extra, max_actions), len(ids))
        for a in pool:
            if len(ids) >= cap:
                break
            ids.append(a)
    else:
        ids = list(dict.fromkeys(shown + found + core))
        if more:
            ids = [a for a in dict.fromkeys(shown + found[:FOUND_FIRST] + more + found) if a not in core]
            ids = ids[:max(0, max_actions - len(core))] + core
    if rng is not None:
        rng.shuffle(ids)
    required = list(dict.fromkeys(gold + core))
    return [cat[a] for a in ids], required


class ExampleIndex:
    """BM25 over the requests of training flows; search() returns similar (request, outline) examples.
    Terms present in more than `df_cap` requests are skipped (they carry little and cost most); when every query
    term is that common, the two rarest are used. Fast enough for ~60k rows on a CPU."""

    def __init__(self, rows=(), k1=1.2, b=0.75, df_cap=None):
        self.rows, self.k1, self.b = [], k1, b
        seen = set()
        for r in rows:
            req = " ".join(str(r.get("request", "")).split())
            if not req or req.lower() in seen:
                continue
            seen.add(req.lower())
            self.rows.append({"request": req, "outline": r.get("outline", ""), "actions": r.get("actions", []),
                              "sig": r.get("sig", "")})
        self.docs = [Counter(retrieval.tokens(r["request"])) for r in self.rows]
        self.len = [sum(d.values()) or 1 for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.df = df
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.inv = {}
        for i, d in enumerate(self.docs):
            for w, f in d.items():
                self.inv.setdefault(w, []).append((i, f))
        self.df_cap = df_cap or max(200, n // 100)

    @classmethod
    def from_flows(cls, pairs, **kw):
        """pairs: (request, flow) -> index with the flows' outlines."""
        rows = [{"request": r, "outline": outline(f), "actions": list(dict.fromkeys(actions_of(f))),
                 "sig": flow_signature(f)} for r, f in pairs]
        return cls(rows, **kw)

    @classmethod
    def load(cls, path, **kw):
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(l) for l in f if l.strip()], **kw)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self.rows)

    def vote(self, query, n, m=VOTE_EXAMPLES, exclude_request=None, exclude_sig=None):
        """The n actions (not core/...) most used by the m requests most like query, each request's vote weighted
        1 / (1 + 0.3 * rank): what flows for such a request are usually built from."""
        if n <= 0:
            return []
        votes = Counter()
        for r, x in enumerate(self.search(query, m, exclude_request, exclude_sig)):
            for a in dict.fromkeys(x.get("actions") or []):
                if isinstance(a, str) and not a.startswith("core/"):
                    votes[a] += 1.0 / (1 + 0.3 * r)
        return [a for a, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

    def search(self, query, n=MAX_EXAMPLES, exclude_request=None, exclude_sig=None):
        """Up to n {"request", "outline", "actions"} rows, best first, never the excluded request or flow signature, and never
        two with the same outline."""
        if n <= 0 or not self.rows:
            return []
        q = [w for w in dict.fromkeys(retrieval.tokens(str(query))) if w in self.inv]
        terms = [w for w in q if self.df[w] <= self.df_cap]
        if len(terms) < 2:
            terms = list(dict.fromkeys(terms + sorted(q, key=lambda w: self.df[w])[:2]))
        scores = {}
        for w in terms:
            idf, post = self.idf[w], self.inv[w]
            for i, f in post[:5000]:
                s = idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                scores[i] = scores.get(i, 0.0) + s
        ex_req = " ".join(str(exclude_request).split()).lower() if exclude_request else None
        out, outlines = [], set()
        for i in sorted(scores, key=lambda i: (-scores[i], i)):
            r = self.rows[i]
            if (ex_req and r["request"].lower() == ex_req) or (exclude_sig and r["sig"] == exclude_sig):
                continue
            if r["outline"] in outlines:
                continue
            outlines.add(r["outline"])
            out.append({"request": r["request"], "outline": r["outline"], "actions": r["actions"]})
            if len(out) >= n:
                break
        return out


# ---------------------------------------------------------------------------------------------- training rows
def input_variables(flow):
    return [(v["name"], v.get("type") or "Any") for v in flow.get("variables") or []
            if isinstance(v, dict) and v.get("name") and v.get("direction") == "input"]


def render_chat(record, catalog, action_index, example_index=None, rng=None, k=DEFAULT_K, variables_rate=0.3,
                max_chars=MAX_ROW_CHARS, max_tokens=MAX_ROW_TOKENS, context_chars=None):
    """A neutral record -> {"messages": [user, assistant]} in this format (None when the record is unusable).
    record: {"kind": "flow", "request", "flow"} | {"kind": "fix", "request", "previous", "problems", "flow"} |
            {"kind": "clarify", "request", "question"}.
    The draws from rng are made in a fixed order (number of examples, variables, shuffle), so a seed reproduces a
    dataset.
    A row that does not fit (row_fits) is rendered again with a smaller context (down to the required actions only); the
    caller still checks the final length. context_chars: the context budget (default: context_for the window that
    max_tokens implies, the same budget the server gives a model with that window)."""
    import random
    rng = rng or random.Random(0)
    cat = _as_dict(catalog)
    if context_chars is None:
        context_chars = context_for(window_of(max_tokens))
    kind, request = record.get("kind"), str(record.get("request") or "").strip()
    if not request or kind not in ("flow", "fix", "clarify"):
        return None
    u = rng.random()
    n_ex = 0 if u < 0.1 else (1 if u < 0.55 else 2)
    use_vars = rng.random() < variables_rate
    if kind == "clarify":
        q = str(record.get("question") or "").strip()
        if not q:
            return None
        examples = example_index.search(request, n_ex, exclude_request=request) if example_index else []
        actions, required = context_actions(request, (), cat, action_index, k, rng, examples, example_index,
                                            exclude_request=request)
        user = build_user(request, actions, examples, (), required=required, context_limit=context_chars)
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": QUESTION_HEAD + " " + q}]}
    flow = record.get("flow")
    if not isinstance(flow, dict) or not isinstance(flow.get("steps"), list) or not flow["steps"]:
        return None
    flow = to_dollar_name_dollar(flow)              # the platform's notation: the model learns to write $name$
    examples = (example_index.search(request, n_ex, exclude_request=request, exclude_sig=flow_signature(flow))
                if example_index else [])
    actions, required = context_actions(request, actions_of(flow), cat, action_index, k, rng, examples, example_index,
                                        exclude_request=request, exclude_sig=flow_signature(flow))
    # variables the request came with (Studio's open agent) always; else the flow's inputs for some rows
    variables = record.get("variables") or (input_variables(flow) if use_vars else ())
    previous = problems = None
    if kind == "fix":
        previous, problems = record.get("previous"), [str(p) for p in record.get("problems") or []]
        if previous is None or not problems:
            return None
        previous = to_dollar_name_dollar(previous)  # as the server shows it (Router._previous)
    answer = render_flow_answer(flow, cat)
    limits = (context_chars, context_chars // 2, 0)
    if len(answer) > max_chars or (len(answer) * 10 > max_tokens * 16 and approx_tokens(answer) > max_tokens):
        limits = (0,)                    # cannot fit whatever the context: one render, the caller splits it
    for limit in limits:
        user = build_user(request, actions, examples, variables, previous, problems, required, limit)
        row = {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}
        if row_fits(row, max_chars, max_tokens):
            break
    return row


def row_chars(row):
    return sum(len(m.get("content") or "") for m in row.get("messages", []) if isinstance(m, dict))


_PIECES = re.compile(r"[A-Za-z]+|[0-9]{1,3}|\n\s*|[ \t]+|[\x21-\x7e]|[^\x00-\x7f]")


def approx_tokens(text):
    """Token count estimate for the model's tokenizer (16k BPE) without loading it: one per punctuation mark, digit
    group, line break and word, plus one per 3 letters beyond 4, plus the bytes of non-ASCII characters. Fitted on
    rendered rows over 1,500 tokens: true / estimate averages 0.91, is under 1.0 for 99% of rows and under 1.04 for
    all, so a row estimated at MAX_ROW_TOKENS fits 2048 tokens (the rare rest loses a few tokens at the end)."""
    n = 0
    for p in _PIECES.findall(str(text)):
        c = p[0]
        if c in " \t":
            continue
        if ord(c) > 127:
            n += len(p.encode("utf-8"))
        elif p.isalpha():
            n += 1 + max(0, len(p) - 4) // 3
        else:
            n += 1
    return n


def row_tokens(row):
    """approx_tokens over every message, plus the turn markers."""
    msgs = [m for m in row.get("messages", []) if isinstance(m, dict)]
    return sum(approx_tokens(m.get("content") or "") for m in msgs) + 2 * len(msgs) + 1


def row_fits(row, max_chars=MAX_ROW_CHARS, max_tokens=MAX_ROW_TOKENS):
    n = row_chars(row)
    if n > max_chars:
        return False
    # rows run 2.5-3.1 characters per token: a short row needs no estimate (it is the slow part of rendering)
    return n * 10 <= max_tokens * 16 or row_tokens(row) <= max_tokens
