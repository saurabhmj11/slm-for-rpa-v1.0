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
              (plan lines of steps inside a loop / if are indented two spaces per level). The flow is either the
              full flow (plan-v1) or, for models trained on FORMAT_COMPACT, only what the model must decide
              (compact_flow: steps without ids / descriptions, no variables list); expand_flow rebuilds the full
              flow, and parse_answer(expand=True) / the server do that only for a compact flow.
    rewrite:  "Clear request: <one clear sentence naming the steps>\nPlan:\n...\nFlow:\n..." (a beginner's request,
              restated in tool words before the plan; rendered for "rewrite" records, gen_beginner.py)
    clarify:  "Question: <one short question>"
    think:    "Understand: <sentence>\\nFacts: k = v; ...\\nMissing: none\\nSteps:\\n1. <words> -> <id>\\n...\\nCheck: <n> steps, ...\\n
              Flow:\\n<flow>" or "...Missing: <slot>\\nQuestion: ..." (models trained with render_dataset.py --think; see the
              "thinking answers" section)
    reply:    plain text (not a task)
parse_answer() also reads old answers that are only the flow JSON, so models trained before this format keep working.

The rendered user + assistant text must fit the model's 2048-token window (row_fits: at most 6,500 characters and
an estimated MAX_ROW_TOKENS estimated tokens, see approx_tokens): the context
is trimmed to MAX_CONTEXT_CHARS in a fixed order (see build_user) and a previous flow longer than MAX_PREVIOUS_CHARS
is shown as its outline. Pure standard library + retrieval.py.
"""
import hashlib
import heapq
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
CLEAR_HEAD = "Clear request:"  # a restatement before the plan (rewrite records): the model says what it understood
ANSWER_STARTS = ("{", PLAN_HEAD, QUESTION_HEAD, FLOW_HEAD, CLEAR_HEAD, "Understand:")   # how a flow / question answer can open
MAX_CLEAR_CHARS = 320         # a longer restatement keeps only the top-level steps (clear_request)

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


def step_text(step, cat=None, describe=True):
    """Plain words for one step: its description when that is short plain text (describe=False: never), else words
    built from a loop's arguments, else the catalog name, else the action id."""
    t = _plain(step.get("description")) if describe else None
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


def render_flow_answer(flow, catalog=None, answer_format="full", clear=None):
    """The plan-first answer for a flow. answer_format "compact": the flow after "Flow:" is compact_flow(flow) (the
    plan, made from the full flow, still carries the step descriptions). clear: a restatement of the request, written
    first as "Clear request: ..." (rewrite records)."""
    body = compact_flow(flow) if answer_format == ANSWER_COMPACT else flow
    head = CLEAR_HEAD + " " + " ".join(str(clear).split()) + "\n" if clear and str(clear).strip() else ""
    return head + PLAN_HEAD + "\n" + "\n".join(plan_from_flow(flow, catalog)) + "\n" + FLOW_HEAD + "\n" + compact(body)


_HOUSEKEEPING = re.compile(r"/(close|disconnect|logout|revoke_authentication|close_all)$")


def _lower_first(t):
    return t[:1].lower() + t[1:] if t[:2] != t[:2].upper() else t


# apps a step description may name; a restatement never names one the request does not (clear_step_ok)
_APPS = re.compile(r"\b(onedrive|sharepoint|outlook|gmail|slack|microsoft teams|jira|servicenow|salesforce|zendesk|asana|"
                   r"hubspot|freshdesk|dropbox|google drive|google sheets|workday|quickbooks|trello|whatsapp|"
                   r"azure devops|confluence|notion|docusign|okta|sap)\b", re.I)


def _catalog_copy(text, entry):
    """True when a step description is (the start of) its action's catalog description: generic tool text."""
    d = " ".join(str((entry or {}).get("description") or "").split()).lower().rstrip(".")
    t = " ".join(str(text or "").split()).lower().rstrip(".")
    return bool(d and t) and (t == d or t[:40] == d[:40] or (len(t) >= 25 and d.startswith(t)))


def clear_step_ok(step, cat=None, request=None):
    """True when a step's description can stand in a restatement: short plain text (step_text), not a copy of the
    catalog's description, and naming no app the request does not name (request given)."""
    t = _plain(step.get("description"))
    if not t or _catalog_copy(t, _as_dict(cat).get(step.get("action"))):
        return False
    if request is not None:
        said = {m.lower() for m in _APPS.findall(str(request))}
        if any(m.lower() not in said for m in _APPS.findall(t)):
            return False
    return True


def clear_request(flow, catalog=None, limit=MAX_CLEAR_CHARS, request=None):
    """One clear sentence naming a flow's steps, for the "Clear request:" line: the steps' plain words in order
    (step_text; a description that copies the catalog's or names an app the request does not name, clear_step_ok,
    gives way to the catalog name), closing / disconnecting steps left out, a loop's or condition's steps after it
    ("for each row in rows: create the ticket and log it"; "if amount > 1000: email it, otherwise: log it"). Longer
    than limit: the bodies of loops and conditions are cut to their first step (below the first level, then all:
    "for each row: create the ticket"), never to nothing; then the sentence is cut at a step boundary."""
    cat = _as_dict(catalog)

    def text_of(s):
        return _lower_first(step_text(s, cat, describe=clear_step_ok(s, cat, request)))

    def real(steps):
        return [s for s in steps if isinstance(s, dict) and not _HOUSEKEEPING.search(str(s.get("action") or ""))] \
            if isinstance(steps, list) else []

    def gist(s, levels=2):              # a container named with the first step of its first non-empty body
        for name, body in branches(s):
            first = real(body)[:1]
            if first:
                word = "" if name == "children" else _BRANCH_WORDS.get(name, "").strip().lower() + " "
                inner = gist(first[0], levels - 1) if levels > 1 else ""
                return ": " + word + text_of(first[0]) + inner
        return ""

    def phrases(steps, deep):           # deep: how many levels of loop / condition bodies are named in full
        out = []
        for s in real(steps):
            t = text_of(s)
            if deep > 0:
                inner = []
                for name, body in branches(s):
                    sub = phrases(body, deep - 1)
                    if sub:
                        word = "" if name == "children" else _BRANCH_WORDS.get(name, "").strip().lower() + " "
                        inner.append(word + " and ".join(sub))
                out.append(t + (": " + ", ".join(inner) if inner else ""))
            else:
                out.append(t + gist(s))
        return out

    steps = flow.get("steps") if isinstance(flow, dict) else flow
    for deep in (99, 1, 0):
        ps = phrases(steps, deep)
        if not ps:
            return ""
        text = ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + ", then " + ps[-1]
        if len(text) + 1 <= limit or not deep:
            break
    if len(text) + 1 > limit:
        cut, n = [], 0
        for p in ps:
            if n + len(p) + 2 > limit - 12:
                break
            cut.append(p)
            n += len(p) + 2
        text = ", ".join(cut or [ps[0][:limit - 12]]) + " and more"
    text = text[:1].upper() + text[1:]
    return text if text.endswith(".") else text + "."


# ---------------------------------------------------------------------------------------------- thinking answers
# An optional block the assistant writes before the flow ("think first"), for models trained on it (render_dataset.py
# --think: per file, so a data mix can hold both answer styles):
#     Understand: <one sentence: what the user wants>
#     Facts: <key = value; ...: the literal details of the request the flow uses> | none
#     Missing: none | <slot, slot: details only the user knows -> the answer goes on with "Question: ...">
#     Steps:
#     1. <step in plain words> -> <action id>          (a loop's / if's steps indented two spaces per level,
#       2. <body step> -> <action id>                   numbered in document order, as the plan lines)
#     Check: <n> steps[, <k> loop(s)][, <k> if(s)][, all facts used]
#     Flow:
#     <compact flow JSON>
# Steps replaces the old "Plan:" (same lines, " -> id" instead of " (id)"); parse_answer gives them back as plan lines,
# so the step-by-step builder, plan_coverage and the critic read a thinking answer like a plan-first one.
UNDERSTAND_HEAD, FACTS_HEAD, MISSING_HEAD, STEPS_HEAD, CHECK_HEAD = (
    "Understand:", "Facts:", "Missing:", "Steps:", "Check:")
THINK_HEADS = (UNDERSTAND_HEAD, FACTS_HEAD, MISSING_HEAD, STEPS_HEAD, CHECK_HEAD)
FORMAT_THINK = "think-v1"
NONE_WORD = "none"
MAX_UNDERSTAND_CHARS = 200    # a flow description longer than this is not used as the "Understand" sentence
MAX_DERIVED_UNDERSTAND = 140  # a sentence made from the steps (clear_request) is cut to this: Steps says the rest
MAX_FACTS = 6                 # facts per answer (the first ones in flow order)
MAX_FACT_CHARS = 60           # a longer literal (a mail body, a script) is not a fact
_NONE_WORDS = {"", "none", "nothing", "no", "-", "n/a"}
_STEP_ARROW = re.compile(r"\s+->\s*(?P<action>[\w.\-]+/[\w.\-]+)\s*$")
_STEP_PAREN = re.compile(r"\s*\((?P<action>[\w.\-]+/[\w.\-]+)\)\s*$")
MISSING_WORDS = {"folder": "which folder", "file": "which file", "files": "which files", "recipient": "who gets it",
                 "app": "which app", "website": "which website", "condition": "which ones (the rule)",
                 "sheet": "which sheet", "report": "which report", "data": "where the data comes from",
                 "list": "which list", "place": "where it goes", "what": "what to do, with which files or app",
                 "records": "which records", "details": "the exact details"}
_QUESTION_SLOTS = [(re.compile(p, re.I), s) for p, s in (
    (r"\bwhich (?:records|accounts|leads|cases|tickets|items|rows|ones|users|orders|invoices|issues|deals)\b|"
     r"\bkaun se\b|\bkaunse\b", "records"),
    (r"\bwhich (?:documents|docs|files|attachments|pdfs)\b|\bfiles\b", "files"),
    (r"\bwhich (?:file|document|workbook|csv|pdf)\b|\bthe file\b|\bfile name\b|\bfile's\b", "file"),
    (r"\bfolder\b", "folder"), (r"\bsheet\b", "sheet"), (r"\breport\b", "report"),
    (r"\bwho\b|\brecipients?\b|email address", "recipient"),
    (r"\bwebsite\b|\burl\b|\blink\b", "website"), (r"\bapp\b|\bsystem\b|\btool\b", "app"),
    (r"\brule\b|\bwhich ones\b|\bthreshold\b|\babove\b|\bbelow\b", "condition"), (r"\blist\b", "list"),
    (r"\bwhere (?:should|do|to|is the right place)\b|\bsave it\b|\bput it\b", "place"), (r"\bdata\b", "data"))]


def plan_to_steps(plan_lines):
    """Plan lines '<n>. <words> (<id>)' -> Steps lines '<n>. <words> -> <id>' (indent and numbers kept)."""
    return [_STEP_PAREN.sub(lambda m: " -> " + m.group("action"), l.rstrip()) for l in plan_lines or []]


def steps_to_plan(step_lines):
    """Steps lines -> plan lines (the inverse of plan_to_steps); a line that already ends '(<id>)' is kept."""
    return [_STEP_ARROW.sub(lambda m: " (" + m.group("action") + ")", l.rstrip()) for l in step_lines or []
            if l.strip()]


_FACT_TOKEN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://[^\s\"']+|[\w-][\w.-]*\.(?:csv|xlsx|xlsm|xls|docx|doc|pdf|txt|"
                         r"json|xml|zip|png|jpe?g|pptx?|msg|eml|html?|log)\b", re.I)
_ANY_REF = re.compile(r"\$\{[^}]*\}|\$\w+\$?")


def _fact_value(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return str(v)
    if not isinstance(v, str):
        return None
    s = " ".join(v.split())
    if not (1 <= len(s) <= MAX_FACT_CHARS) or "$" in s or "{" in s or "\n" in v:
        return None
    return s


def _fact_values(v):
    """The fact candidates of one argument value: the whole literal (_fact_value), or else the file names, addresses
    and links written inside a text that also holds variables ('${folder}\\report.csv' -> 'report.csv')."""
    one = _fact_value(v)
    if one or not isinstance(v, str) or "$" not in v:
        return [one] if one else []
    return [m.group(0) for m in _FACT_TOKEN.finditer(_ANY_REF.sub(" ", v)) if len(m.group(0)) <= MAX_FACT_CHARS]


def _in_request(value, request_lower):
    v = value.lower()
    if len(v) < 3 and not v.isdigit():
        return False
    if re.fullmatch(r"[\w.]+", v):                     # a word / number: whole-word match ("1" is not "Sheet1")
        return re.search(r"(?<![\w.])" + re.escape(v) + r"(?![\w])", request_lower) is not None
    return v in request_lower


_SCHEDULE = re.compile(
    r"\b(?:daily|weekly|monthly|hourly|every\s+(?:day|morning|evening|night|hour|week|month|weekday|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|\d+\s+(?:minutes|hours|days))|"
    r"(?:har|roz)\s+(?:din|subah|shaam|hafte|mahine|somvar))"
    r"(?:\s+(?:at|by|before)\s+\d{1,2}(?::\d\d)?\s*(?:am|pm)?)?", re.I)


def schedule_of(request):
    """The schedule a request names ('daily at 9', 'every Monday'), or None. A flow has no schedule step (the platform
    schedules the bot), so this fact is never looked for in the flow (fact_coverage)."""
    m = _SCHEDULE.search(str(request or ""))
    return " ".join(m.group(0).split()) if m else None


def flow_facts(flow, request, limit=MAX_FACTS):
    """The request's literal details that the flow uses: [(key, value)] in flow order. A step argument whose literal
    value (a path, file name, address, folder, number, word) appears in the request is a fact named after the
    argument ("to = ap@acme.com"); a condition comparing with a literal of the request is "filter = amount > 1000".
    Values longer than MAX_FACT_CHARS or holding variables are left out; each value once. A schedule the request
    names comes first ("schedule = daily at 9", schedule_of)."""
    req = " ".join(str(request or "").split()).lower()
    out, seen = [], set()
    when = schedule_of(request)
    if when:
        out.append(("schedule", when))

    def add(key, value):
        if value and value.lower() not in seen and len(out) < limit and _in_request(value, req):
            seen.add(value.lower())
            out.append((key, value))

    def cond(c):
        if not isinstance(c, dict) or not {"op", "left"} <= set(c):
            return False
        right = _fact_value(c.get("right"))
        left = _var(c["left"]) if isinstance(c.get("left"), str) else None
        if right and left and _in_request(right, req) and right.lower() not in seen and len(out) < limit:
            seen.add(right.lower())
            out.append(("filter", f"{left} {c['op']} {right}"))
        return True

    def arg(key, v):
        if isinstance(v, dict):
            if not cond(v):
                for k2, v2 in v.items():
                    arg(key if k2 in ("args", "value", "right", "left") else k2, v2)
        elif isinstance(v, list):
            for x in v:
                arg(key, x)
        else:
            for x in _fact_values(v):
                add(key, x)

    for s, _ in walk(flow.get("steps") if isinstance(flow, dict) else flow):
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        for k, v in args.items():
            arg(k, v)
    return out


def facts_text(facts):
    """[(key, value)] or {key: value} or a string -> 'key = value; key = value', or 'none'."""
    if isinstance(facts, str):
        return " ".join(facts.split()) or NONE_WORD
    pairs = list(facts.items()) if isinstance(facts, dict) else [tuple(p) for p in facts or [] if len(p) == 2]
    return "; ".join(f"{k} = {v}" for k, v in pairs) or NONE_WORD


def parse_facts(text):
    """'key = value; ...' -> [(key, value)] ([] for none)."""
    t = " ".join(str(text or "").split())
    if t.lower().rstrip(".") in _NONE_WORDS:
        return []
    out = []
    for part in t.split(";"):
        k, eq, v = part.partition("=")
        if eq and k.strip() and v.strip():
            out.append((k.strip(), v.strip()))
    return out


def parse_missing(text):
    """'none' -> []; 'folder, recipient' -> ['folder', 'recipient'] (free words kept as they are)."""
    t = " ".join(str(text or "").split()).rstrip(".")
    if t.lower() in _NONE_WORDS:
        return []
    return [p.strip() for p in re.split(r",|;|\band\b", t) if p.strip()]


def missing_slots(record):
    """The Missing slots of a clarify record: its own "missing", else what the ask-first gate finds in the request,
    else what its question asks about (_QUESTION_SLOTS); at most two, 'details' when nothing is known."""
    m = record.get("missing")
    if isinstance(m, str):
        m = parse_missing(m)
    if isinstance(m, list) and m:
        return [str(x) for x in m][:2]
    gate = ask_first(record.get("request"))
    if gate and gate.get("missing"):
        return gate["missing"][:2]
    q = re.sub(r"\([^)]*\)", " ", str(record.get("question") or ""))
    q = q.split("?")[0]                                   # the first question only, without its examples
    found = [s for rx, s in _QUESTION_SLOTS if rx.search(q)]
    return list(dict.fromkeys(found))[:2] or ["details"]


def missing_question(missing):
    """The question for a thinking answer's Missing slots when the model wrote none: ask_question for known slots,
    else 'Happy to help. Could you tell me <the missing words>?'."""
    slots = [m for m in missing or [] if m in ASK_SLOTS]
    if slots and len(slots) == len(missing):
        return ask_question(slots)
    return "Happy to help. Could you tell me " + " and ".join(MISSING_WORDS.get(m, m) for m in (missing or ["details"])[:2]) + "?"


def _sentence(text, limit=None):
    t = " ".join(str(text or "").split())
    if limit and len(t) > limit:
        return ""
    t = t[:1].upper() + t[1:]
    return t if not t or t.endswith((".", "?", "!")) else t + "."


def understand_text(record, flow=None, catalog=None, request=None):
    """The "Understand" sentence of a record: its own "understand", else its "clear" restatement, else the flow's
    description (plain text naming no app the request does not, at most MAX_UNDERSTAND_CHARS), else
    clear_request(flow) cut to MAX_DERIVED_UNDERSTAND; a clarify record without one says what it cannot tell."""
    request = request if request is not None else record.get("request")
    for key in ("understand", "clear"):
        t = _sentence(record.get(key))
        if t:
            return t
    if isinstance(flow, dict):
        d = flow.get("description")
        if isinstance(d, str) and 15 <= len(d.strip()) <= MAX_UNDERSTAND_CHARS and not any(c in d for c in "${}`\""):
            said = {m.lower() for m in _APPS.findall(str(request or ""))}
            if all(m.lower() in said for m in _APPS.findall(d)):
                return _sentence(d)
        return _sentence(steps_sentence(flow, catalog, request))
    slots = missing_slots(record)
    said = " ".join(str(request or "").split()).rstrip(".?!")
    said = said if len(said) <= 120 else said[:117].rsplit(" ", 1)[0] + "..."
    return _sentence(said) + " Not said: " + "; ".join(MISSING_WORDS.get(s, s) for s in slots) + "."


def steps_sentence(flow, catalog=None, request=None, limit=MAX_DERIVED_UNDERSTAND):
    """A derived "Understand" sentence: clear_request(flow) when it fits limit, else its top-level steps in plain words
    from both ends ("Find the overdue issues, add up the column, ..., then email Rahul the summary"), so the start and
    the goal are both said."""
    full = clear_request(flow, catalog, limit=10 ** 6, request=request)
    if len(full) <= limit:
        return full
    cat = _as_dict(catalog)

    def words(s):                       # a loop / if is named with the first step of its body
        t = _lower_first(step_text(s, cat, describe=clear_step_ok(s, cat, request)))
        for _, body in branches(s):
            first = [b for b in body if isinstance(b, dict) and not _HOUSEKEEPING.search(str(b.get("action") or ""))]
            if first:
                return t + ": " + words(first[0])
        return t
    tops = [words(s) for s in (flow.get("steps") if isinstance(flow, dict) else flow) or []
            if isinstance(s, dict) and not _HOUSEKEEPING.search(str(s.get("action") or ""))]
    if len(tops) < 2:
        return clear_request(flow, catalog, limit=limit, request=request)
    head, last = [tops[0]], tops[-1]
    for t in tops[1:-1]:
        if len(", ".join(head + [t])) + len(last) + 12 > limit:
            break
        head.append(t)
    gap = ", ..." if len(head) < len(tops) - 1 else ""
    text = ", ".join(head) + gap + ", then " + last
    if len(text) > limit + 40:
        return clear_request(flow, catalog, limit=limit, request=request)
    return text[:1].upper() + text[1:]


def check_text(step_lines, facts=()):
    """The Check line's words for Steps lines: '<n> steps, <k> loop(s), <k> if(s)[, all facts used]'."""
    acts = [i["action"] for i in plan_items(steps_to_plan(step_lines))]
    n, loops, ifs = len(plan_items(step_lines)), acts.count("core/loop"), acts.count("core/if")
    out = [f"{n} step" + ("s" if n != 1 else "")]
    if loops:
        out.append(f"{loops} loop" + ("s" if loops > 1 else ""))
    if ifs:
        out.append(f"{ifs} if" + ("s" if ifs > 1 else ""))
    if facts:
        out.append("all facts used")
    return ", ".join(out)


def think_steps(flow, catalog=None, record=None):
    """Steps lines for a flow: the record's own "steps" when they name the flow's actions in its order, else
    plan_to_steps(plan_from_flow(flow))."""
    derived = plan_to_steps(plan_from_flow(flow, catalog))
    own = (record or {}).get("steps")
    if isinstance(own, list) and own and all(isinstance(x, str) for x in own):
        if plan_actions(steps_to_plan(own)) == plan_actions(steps_to_plan(derived)):
            return [l.rstrip() for l in own if l.strip()]
    return derived


def think_block(understand, facts, missing, step_lines=None):
    """The thinking block before "Flow:" / "Question:" (no trailing newline)."""
    lines = [UNDERSTAND_HEAD + " " + (_sentence(understand) or "-"), FACTS_HEAD + " " + facts_text(facts),
             MISSING_HEAD + " " + (", ".join(missing) if missing else NONE_WORD)]
    if step_lines is not None:
        lines += [STEPS_HEAD] + list(step_lines) + [CHECK_HEAD + " " + check_text(step_lines, facts)]
    return "\n".join(lines)


def render_think_answer(flow, catalog=None, answer_format="full", record=None, request=None):
    """The thinking answer for a flow: think_block (Understand / Facts / Missing: none / Steps / Check) + "Flow:" +
    the flow (compact_flow for answer_format "compact"). record: the neutral record (its own understand / facts /
    steps win over the derived ones; facts given as {key: value}, [[key, value]] or text)."""
    record = record or {}
    request = request if request is not None else record.get("request")
    cat = _as_dict(catalog)
    facts = record.get("facts") if record.get("facts") is not None else flow_facts(flow, request)
    body = compact_flow(flow) if answer_format == ANSWER_COMPACT else flow
    block = think_block(understand_text(record, flow, cat, request), facts, [], think_steps(flow, cat, record))
    return block + "\n" + FLOW_HEAD + "\n" + compact(body)


def render_think_question(record):
    """The thinking answer of a clarify record: Understand / Facts / Missing: <slots> + "Question: <question>"."""
    slots = missing_slots(record)
    facts = record.get("facts") if record.get("facts") is not None else []
    return think_block(understand_text(record), facts, slots) + "\n" + QUESTION_HEAD + " " + \
        " ".join(str(record.get("question") or "").split())


def think_prefix(request, plan_lines, facts=()):
    """The start of a thinking answer whose steps are known (a part of a plan built in parts): the block with
    Understand = the part's request, the given steps and their Check, then "Flow:" (the model writes the flow)."""
    steps = plan_to_steps(plan_lines)
    return think_block(request, facts, [], steps) + "\n" + FLOW_HEAD + "\n"


def parse_think(text):
    """A thinking block at the start of text -> (fields, rest): fields {"understand", "facts" [(k, v)], "facts_text",
    "missing" [slots], "steps" [lines], "check"}; rest is the text from the first "Flow:" / "Question:" / "Plan:" line
    on ('' when the answer stops before it)."""
    f = {"understand": None, "facts": [], "facts_text": None, "missing": [], "steps": [], "check": None}
    lines = str(text).split("\n")
    section = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith((FLOW_HEAD, QUESTION_HEAD, PLAN_HEAD)):
            return f, "\n".join(lines[i:]).strip()
        head = next((h for h in THINK_HEADS if s.startswith(h)), None)
        if head:
            section, val = head, s[len(head):].strip()
            if head == UNDERSTAND_HEAD:
                f["understand"] = val or None
            elif head == FACTS_HEAD:
                f["facts_text"], f["facts"] = val, parse_facts(val)
            elif head == MISSING_HEAD:
                f["missing"] = parse_missing(val)
            elif head == CHECK_HEAD:
                f["check"] = val or None
            elif val:                                     # "Steps: 1. ..." on the same line
                f["steps"].append(val)
            continue
        if not s:
            continue
        if section == STEPS_HEAD:
            f["steps"].append(line.rstrip())
        elif section == UNDERSTAND_HEAD:                  # a sentence that ran onto a second line
            f["understand"] = ((f["understand"] or "") + " " + s).strip()
    return f, ""


def is_think(text):
    t = str(text or "").lstrip()
    return t.startswith(THINK_HEADS)


def _haystack(flow):
    out = []

    def add(v):
        if isinstance(v, dict):
            for x in v.values():
                add(x)
        elif isinstance(v, list):
            for x in v:
                add(x)
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
            out.append(" ".join(str(v).split()).lower())
    for s, _ in walk(flow.get("steps") if isinstance(flow, dict) else flow):
        add(s.get("args"))
    for v in (flow.get("variables") if isinstance(flow, dict) else None) or []:
        if isinstance(v, dict):
            add(v.get("default"))
            add(v.get("value"))
    return out


def fact_coverage(facts, flow):
    """Facts [(key, value)] of a thinking answer that no step argument (or variable default) of the flow holds: the
    flow left a detail of the request out. A filter fact ('amount > 1000') is checked by its last word."""
    hay = _haystack(flow)
    missed = []
    for k, v in facts or []:
        want = " ".join(str(v).split()).lower()
        if k == "schedule":
            continue
        if k == "filter" and " " in want:
            want = want.rsplit(" ", 1)[1]
        if want and want != NONE_WORD and not any(want in h for h in hay):
            missed.append((k, v))
    return missed


# ---------------------------------------------------------------------------------------------- compact answers
# A compact answer's flow holds only what the model must decide: the steps (action, args, out, containers and their
# branches; disabled), plus "outputs" (names of the flow's output variables) and any rare flow-level setting
# (errorHandling, onError, subflows, requires, a kind other than TaskBot). Left out, because code rebuilds them
# (expand_flow): kind / id / name / description of the flow, the variables list, step ids, step descriptions and notes.
# On the long and business training flows the variables list is 35-37% of the flow JSON and descriptions much of the
# rest. Serving reads both: a flow with a variables list and step ids is used as written (models trained on full
# flows), any other flow is expanded first (is_compact).
ANSWER_FULL, ANSWER_COMPACT = "full", "compact"
FORMAT_COMPACT = "plan-v2-compact"
COMPACT_OUTPUTS = "outputs"
_COMPACT_DROP_TOP = ("kind", "id", "name", "description", "variables", "$schema", "tags", "version", "ui")
_COMPACT_DROP_STEP = ("id", "description", "note")
_CRED_NAME = re.compile(r"pass(word|wd|phrase)?(_|$)|(^|_)pwd(_|$)|secret|token|api_?key|apikey|credential|private_key",
                        re.I)
# catalog type words -> flow.v1 variable types
_CAT_TYPES = {"Text": "String", "File path": "String", "Yes/No": "Boolean", "Credential (vault)": "Credential",
              "Number": "Number", "List": "List", "Dictionary": "Dictionary", "Table": "Table", "DateTime": "DateTime",
              "Any": "Any"}
VARIABLE_TYPES = ("String", "Number", "Boolean", "DateTime", "List", "Dictionary", "Table", "Record", "File", "Window",
                  "UIElement", "Credential", "Any", "Time", "TimeSpan", "Image", "Binary")
_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_STEP_ID_OK = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
# the platform checker's reference rules (ooa_reward.refs): $name / $name$ / ${name}, never inside {"literal": ...}
_READ_PLAIN = re.compile(r"\$(?!\{)([A-Za-z_]\w*)")
_READ_TEMPLATE = re.compile(r"\$\{\s*([A-Za-z_]\w*)(?![\w(])")


def _compact_step(s):
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k in _COMPACT_DROP_STEP:
            continue
        if k in ("children", "else", "finally", "default", "entry", "steps") and isinstance(v, list):
            out[k] = [_compact_step(x) for x in v]
        elif k in ("elseIf", "catch", "cases", "transitions") and isinstance(v, list):
            out[k] = [{kk: ([_compact_step(x) for x in vv] if kk in ("children", "steps") and isinstance(vv, list)
                            else vv) for kk, vv in b.items()} if isinstance(b, dict) else b for b in v]
        elif k == "states" and isinstance(v, list):
            out[k] = [_compact_step(st) if isinstance(st, dict) else st for st in v]
        else:
            out[k] = v
    return out


def compact_flow(flow):
    """The compact form of a full flow (see above): {"steps": [...]} with step ids, descriptions and notes left out,
    plus "outputs": [names] when the flow has output variables, plus the flow's rare settings (errorHandling, onError,
    subflows, requires; kind only when not TaskBot). expand_flow rebuilds a full flow from it."""
    if not isinstance(flow, dict):
        return flow
    out = {}
    if flow.get("kind") not in (None, "TaskBot"):
        out["kind"] = flow["kind"]
    for k, v in flow.items():
        if k in _COMPACT_DROP_TOP or k == "steps":
            continue
        if k == "subflows" and isinstance(v, list):
            v = [{kk: ([_compact_step(x) for x in vv] if kk == "steps" and isinstance(vv, list) else vv)
                  for kk, vv in sf.items() if kk != "description"} if isinstance(sf, dict) else sf for sf in v]
        elif k == "onError" and isinstance(v, dict) and isinstance(v.get("steps"), list):
            v = {**v, "steps": [_compact_step(x) for x in v["steps"]]}
        out[k] = v
    out["steps"] = [_compact_step(s) for s in flow.get("steps") or []]
    outs = [v["name"] for v in flow.get("variables") or [] if isinstance(v, dict)
            and v.get("direction") in ("output", "in_out") and isinstance(v.get("name"), str)]
    if outs:
        out[COMPACT_OUTPUTS] = outs
    return out


def is_compact(obj):
    """True when a flow (dict) needs expand_flow: it has steps but no variables list, or a step without an id."""
    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        return False
    vs = obj.get("variables")
    if not isinstance(vs, list) or COMPACT_OUTPUTS in obj or any(not isinstance(v, dict) for v in vs):
        return True
    return any(isinstance(s, dict) and not s.get("id") for s, _ in walk(obj["steps"]))


def _all_bodies(flow):
    """Every top-level step list of a flow: steps, subflows[].steps, onError.steps."""
    lists = [flow.get("steps")]
    lists += [sf.get("steps") for sf in flow.get("subflows") or [] if isinstance(sf, dict)]
    if isinstance(flow.get("onError"), dict):
        lists.append(flow["onError"].get("steps"))
    return [x for x in lists if isinstance(x, list)]


def reads_of(value):
    """Root variable names an arg value reads, by the platform checker's rules (ooa_reward.refs), sys / global aside,
    in order of appearance."""
    out = []

    def go(v):
        if isinstance(v, str):
            hits = sorted([(m.start(), m.group(1)) for m in _READ_PLAIN.finditer(v)]
                          + [(m.start(), m.group(1)) for m in _READ_TEMPLATE.finditer(v)])
            out.extend(n for _, n in hits)
        elif isinstance(v, dict):
            if set(v) == {"literal"}:
                return
            for x in v.values():
                go(x)
        elif isinstance(v, list):
            for x in v:
                go(x)
    go(value)
    return [n for n in dict.fromkeys(out) if n not in BUILTIN_ROOTS]


def _bare(name):
    """A variable target as written by a model: 'x', '$x$', '$x', '${x}' -> 'x' (None when not a name)."""
    if not isinstance(name, str):
        return None
    m = _WHOLE_REF.match(name.strip())
    n = (m.group(1) if m else name.strip())
    return n if _VAR_NAME.match(n) else None


def _slug(text, limit=48):
    words = re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
    stop = {"the", "a", "an", "and", "of", "to", "in", "for", "on", "each", "every", "it", "its", "with", "from", "into",
            "then", "if", "is", "are", "all", "my", "our", "that", "this", "as", "by", "at", "or"}
    keep = [w for w in words if w not in stop][:6] or words[:6]
    return ("_".join(keep)[:limit].strip("_")) or "flow"


def _flow_name(request):
    t = " ".join(str(request or "").split())
    first = re.split(r"(?<=[.;:!?])\s|,\s(?:then|and)\s", t)[0].rstrip(".;:!? ")
    if len(first) > 60:
        first = first[:60].rsplit(" ", 1)[0].rstrip(",;: ")
    return (first[:1].upper() + first[1:]) if first else "Flow"


def _infer_set_type(value):
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, (int, float)):
        return "Number"
    if isinstance(value, list):
        return "List"
    if isinstance(value, str) and not _WHOLE_REF.match(value.strip()):
        return "String"
    return "Any"


def expand_flow(obj, catalog=None, request=None, plan=None, known=()):
    """A compact flow (compact_flow, or what a model wrote in that form) -> a full flow the platform accepts:
      - kind TaskBot (unless given), id / name / description from the request (or kept when given);
      - step ids: every step without a valid, unique id gets the next free sN (document order);
      - step descriptions from the answer's plan lines (plan_from_flow writes one line per step in the same order;
        lines and steps are matched by action id), only where a step has none;
      - variables: every name a step makes (an out mapping, a loop's item / index, core/set's variable, a catch's
        message_to, onError's as / message_to / line_to) is declared local (output when listed in "outputs"), typed
        from the catalog's output type when known; every name only read ($name$ / ${name} / $name in args, by the
        checker's own rules) is declared an input, typed Credential for secret-like names (password, token, api_key,
        secret ...) or from the catalog input it fills, else String. Names in `known` (variables the caller already
        has, e.g. Studio's open agent) and variables the flow already declares are not declared again.
    Out targets written "$x$" become "x". The input is not changed; a new flow is returned."""
    if not isinstance(obj, dict):
        return obj
    cat = _as_dict(catalog)
    flow = json.loads(compact(obj))
    outputs = [n for n in (_bare(x) for x in flow.pop(COMPACT_OUTPUTS, None) or []) if n]
    req = " ".join(str(request or "").split())
    base = {"kind": flow.pop("kind", None) or "TaskBot"}
    fid = flow.pop("id", None)
    base["id"] = fid if isinstance(fid, str) and _STEP_ID_OK.match(fid) else _slug(req or flow.get("name"))
    name = flow.pop("name", None)
    base["name"] = name if isinstance(name, str) and name.strip() else _flow_name(req or base["id"])
    desc = flow.pop("description", None)
    if isinstance(desc, str) and desc.strip():
        base["description"] = desc
    elif req:
        base["description"] = req
    # declarations the answer gave (a full variable dict) are kept; bare names are only hints, inferred like the rest
    variables = [v for v in flow.pop("variables", None) or [] if isinstance(v, dict) and isinstance(v.get("name"), str)
                 and isinstance(v.get("type"), str)]
    full = {**base, **{k: v for k, v in flow.items() if k != "steps"}, "steps": flow.get("steps") or []}
    bodies = _all_bodies(full)
    steps = [s for b in bodies for s, _ in walk(b)]

    # step ids
    used, n = set(), 0
    for s in steps:
        sid = s.get("id")
        if isinstance(sid, str) and _STEP_ID_OK.match(sid) and sid not in used:
            used.add(sid)
    seen = set()
    for s in steps:
        sid = s.get("id")
        if isinstance(sid, str) and _STEP_ID_OK.match(sid) and sid not in seen:
            seen.add(sid)
            continue
        n += 1
        while f"s{n}" in used:
            n += 1
        s["id"] = f"s{n}"
        used.add(s["id"])
        seen.add(s["id"])
    # rebuild key order: id first, as written by the platform
    def order(s):
        if not isinstance(s, dict):
            return s
        lead = {k: s[k] for k in ("id", "action", "description") if k in s}
        return {**lead, **{k: v for k, v in s.items() if k not in lead}}

    # descriptions from the plan
    main = [s for s, _ in walk(full["steps"])]
    items = plan_items(plan) if plan else []
    if items and main:
        import difflib
        a = [s.get("action") for s in main]
        b = [it["action"] for it in items]
        for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                s, it = main[blk.a + k], items[blk.b + k]
                t = _plain(it["text"])
                if t and not s.get("description"):
                    s["description"] = t

    # variables
    declared = {v["name"] for v in variables}
    skip = set(declared) | {_bare(x) if isinstance(x, str) else (x.get("name") if isinstance(x, dict) else None)
                            for x in known or ()} | set(BUILTIN_ROOTS)
    made, read = {}, {}                     # name -> type (in order of first appearance)

    def make(nm, typ):
        nm = _bare(nm)
        if nm and nm not in made:
            made[nm] = typ or "Any"
        elif nm and made.get(nm) == "Any" and typ and typ != "Any":
            made[nm] = typ

    for s in steps:
        a, args = s.get("action"), s.get("args") if isinstance(s.get("args"), dict) else {}
        e = cat.get(a) or {}
        if isinstance(s.get("out"), dict):
            outs_t = {o.get("name"): _CAT_TYPES.get(o.get("type"), "Any") for o in e.get("outputs") or []}
            for k, v in list(s["out"].items()):
                b = _bare(v)
                if b and b != v:
                    s["out"][k] = b
                make(b, outs_t.get(k))
        ins_t = {i.get("name"): _CAT_TYPES.get(i.get("type"), "String") for i in e.get("inputs") or []}
        for k, v in args.items():
            for nm in reads_of(v):
                if nm not in read:
                    whole = _var(v) == nm                       # the whole field is this variable
                    t = ins_t.get(k) if whole and not str(a).startswith("core/") else None
                    read[nm] = t if t in VARIABLE_TYPES and t != "Any" else "String"
        if a == "core/loop":
            make(args.get("as"), "Any")
            make(args.get("index"), "Number")
        elif a == "core/set":
            make(args.get("variable"), _infer_set_type(args.get("value")))
        for c in s.get("catch") or []:
            if isinstance(c, dict):
                make(c.get("message_to"), "String")
    oe = full.get("onError")
    if isinstance(oe, dict):
        make(oe.get("as"), "Dictionary")
        make(oe.get("message_to"), "String")
        make(oe.get("line_to"), "Number")
    new = []
    for nm, t in read.items():
        if nm in made or nm in skip or not _VAR_NAME.match(nm):
            continue
        typ = "Credential" if _CRED_NAME.search(nm) else t
        new.append({"name": nm, "type": typ, "direction": "input", "required": True})
    for nm, t in made.items():
        if nm in skip:
            continue
        new.append({"name": nm, "type": t if t in VARIABLE_TYPES else "Any",
                    "direction": "output" if nm in outputs else "local"})
    for nm in outputs:
        if nm not in made and nm not in skip and nm not in read:
            new.append({"name": nm, "type": "Any", "direction": "output"})
    full["variables"] = variables + new

    def reorder(steps_):
        out = []
        for s in steps_:
            if not isinstance(s, dict):
                out.append(s)
                continue
            s = order(s)
            for k in ("children", "else", "finally", "default"):
                if isinstance(s.get(k), list):
                    s[k] = reorder(s[k])
            for k in ("elseIf", "catch", "cases"):
                if isinstance(s.get(k), list):
                    s[k] = [{**b, "children": reorder(b["children"])} if isinstance(b, dict)
                            and isinstance(b.get("children"), list) else b for b in s[k]]
            out.append(s)
        return out
    full["steps"] = reorder(full["steps"])
    head = {k: full[k] for k in ("kind", "id", "name", "description", "variables") if k in full}
    return {**head, **{k: v for k, v in full.items() if k not in head}}


def flow_object(text):
    """The JSON object in a flow text (code fences / text around it allowed), or None."""
    t = str(text or "").split("<|assistant_end|>")[0].strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def answer_flow_text(ans, catalog=None, request=None, known=()):
    """The full flow JSON text of a parsed answer (parse_answer): a compact flow is expanded (expand_flow, with the
    answer's plan for step descriptions), a full one is returned as the model wrote it; None when there is no flow."""
    if not isinstance(ans, dict) or ans.get("kind") != "flow" or ans.get("flow_text") is None:
        return None
    obj = flow_object(ans["flow_text"])
    if obj is None or not is_compact(obj):
        return ans["flow_text"]
    return compact(expand_flow(obj, catalog, request, ans.get("plan"), known))


_PLAN_LINE = re.compile(r"^(?P<indent>\s*)(?:\d+[.)]\s*)?(?P<text>.*?)\s*(?:\((?P<action>[\w.\-]+/[\w.\-]+)\)|"
                        r"->\s*(?P<arrow>[\w.\-]+/[\w.\-]+))?\s*$")     # "(id)": a plan line; "-> id": a Steps line
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
                      "action": m.group("action") or m.group("arrow"), "line": line.rstrip()})
    return items


def plan_actions(plan):
    return [i["action"] for i in plan_items(plan) if i["action"]]


def plan_coverage(plan, flow, catalog=None):
    """Action ids the plan names that no step of the flow uses (empty: the flow covers the plan). With a catalog,
    ids that are not real actions are ignored: a garbled name in the plan text (e.g. "ooa.folder/create_symbolic")
    says nothing about whether the flow, whose action names the catalog lock keeps real, does the job."""
    have = set(actions_of(flow))
    real = None if catalog is None else (catalog if isinstance(catalog, (dict, set)) else set(catalog))
    return [a for a in dict.fromkeys(plan_actions(plan)) if a not in have and (real is None or a in real)]


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


_CUT_CONTAINERS = ("core/loop", "core/try", "core/if")


def _cut_units(g, level, max_lines, max_items):
    """Units ([groups], level) for one group (plan_groups / _subgroups shape) at nesting level `level`. A group longer
    than max_lines whose first item is a loop, a try or an if with 2+ body children is cut: the container line with its
    first body children and all its branch lines ("On error: ...", "Otherwise: ..." and what follows them, which stay
    in the part that writes the container), then the rest of its body children at level + 1, each cut the same way
    when it is itself a long container. A container whose body is one long container (a loop around a try) is cut
    inside that child: the container line goes with the child's first part."""
    subs = _subgroups(g) if len(g) > max_lines and g[0]["action"] in _CUT_CONTAINERS else []
    body = []
    for sg in subs:                                    # the body: the children before the first branch line
        if _BRANCH_PREFIX.match(_PLAN_LINE.match(sg[0]["line"]).group("text") or ""):
            break
        body.append(sg)
    branch_lines = [it for sg in subs[len(body):] for it in sg]
    if len(body) == 1:                                 # one long child (a loop around a try): cut that child
        deeper = _cut_units(body[0], level + 1, max_lines, max_items) if body else []
        if len(deeper) < 2:
            return [([g], level)]
        return [([[g[0]] + deeper[0][0][0] + branch_lines], level)] + deeper[1:]
    if len(body) < 2:
        return [([g], level)]
    target = -(-len(g) // -(-len(g) // max_lines))     # lines per piece when the container is cut evenly
    head, rest = [g[0]] + body[0], body[1:]
    while len(rest) > 1 and len(head) + len(branch_lines) + len(rest[0]) <= target:
        head += rest.pop(0)
    units, cur = [([head + branch_lines], level)], []
    for sg in rest:
        deeper = _cut_units(sg, level + 1, max_lines, max_items)
        if len(deeper) > 1:                            # a long container child: cut on its own
            if cur:
                units.append((cur, level + 1))
                cur = []
            units.extend(deeper)
            continue
        if cur and (len(cur) >= max_items or sum(len(x) for x in cur) + len(sg) > max_lines):
            units.append((cur, level + 1))
            cur = []
        cur.append(sg)
    if cur:
        units.append((cur, level + 1))
    return units


def plan_parts(plan, max_lines=5, max_items=4):
    """Parts to build a long plan in, smaller than chunk_plan's, that may also cut a container's body: a list of
    {"chunk": [groups], "inside": level}. chunk is what chunk_request / chunk_actions read (groups as plan_groups);
    inside=0 (False) means the part's steps follow the steps so far, inside=1 (True) that they go into the body of
    the container (loop, try, if) that ends the steps so far, inside=2 into the body of the container that ends that
    body, and so on (merge_flows / insert_inside). Top-level groups are packed up to max_items items / max_lines lines;
    a group longer than max_lines whose first item is a loop, a try or an if is cut about evenly (_cut_units): the
    container line with its first children and its branch lines (catch / else), then the rest of its children (each
    with its nested lines; a long container child is cut the same way) packed the same way. A plan of one part comes
    back as one part."""
    units = []                                        # (groups, level)
    for g in plan_groups(plan):
        units.extend(_cut_units(g, 0, max_lines, max_items))
    parts = []
    for i, (groups, level) in enumerate(units):
        last = parts[-1] if parts else None
        if (last is not None and last["inside"] == level and not last["open"]
                and len(last["chunk"]) + len(groups) <= max_items
                and sum(len(x) for x in last["chunk"]) + sum(len(x) for x in groups) <= max_lines):
            last["chunk"].extend(groups)
        else:
            parts.append({"chunk": list(groups), "inside": level, "open": False})
        # a cut container's head ends its part: the deeper part after it goes into that container
        parts[-1]["open"] = i + 1 < len(units) and units[i + 1][1] > level
    return [{"chunk": p["chunk"], "inside": p["inside"] if p["inside"] > 1 else bool(p["inside"])} for p in parts]


def container_path(steps, level):
    """The containers an inside part of this level goes into: [top-level last step, the last step of its body, ...]
    (level of them), or None when the steps so far do not end that way. A body is a container's "children"."""
    path, cur = [], steps
    for _ in range(int(level)):
        last = cur[-1] if isinstance(cur, list) and cur else None
        if not isinstance(last, dict) or last.get("action") not in _CUT_CONTAINERS:
            return None
        path.append(last)
        cur = last.get("children") if isinstance(last.get("children"), list) else []
    return path


def insert_inside(steps, part_steps, level=1):
    """Add part_steps to the body ("children") of the container at the end of steps, level containers deep
    (plan_parts' inside). A part that wrapped its steps in the same container again (one step with that action, and
    for a loop the same item) is unwrapped first. Returns False when steps do not end in such containers."""
    path = container_path(steps, level)
    if not path:
        return False
    box, body = path[-1], part_steps
    if (len(body) == 1 and isinstance(body[0], dict) and body[0].get("action") == box.get("action")
            and isinstance(body[0].get("children"), list)
            and (box.get("action") != "core/loop"
                 or (body[0].get("args") or {}).get("as") == (box.get("args") or {}).get("as"))):
        body = body[0]["children"]
    box.setdefault("children", [])
    box["children"] = list(box["children"]) + list(body)
    return True


def insert_into_last_loop(steps, part_steps):
    """insert_inside one level deep, for a loop only (kept for callers of the loop-only cut)."""
    if not steps or not isinstance(steps[-1], dict) or steps[-1].get("action") != "core/loop":
        return False
    return insert_inside(steps, part_steps, 1)


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
    united (the first declaration of a name wins). inside: one level per part (plan_parts; False/0, True/1, 2 ...); an
    inside part's steps go into the body of the container that ends the steps so far (insert_inside), else after them."""
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
        if not (inside and k < len(inside) and inside[k] and insert_inside(steps, new, inside[k])):
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
def parse_answer(text, expand=False, catalog=None, request=None, known=()):
    """{"kind": "flow"|"question"|"reply", "plan": [lines], "flow_text": str|None, "question": str|None, "text": str}.
    An answer that is only flow JSON (the format before plans) is kind "flow" with an empty plan.
    expand: a compact flow (is_compact) is rebuilt into a full one (expand_flow with catalog, request, known and the
    answer's plan) and flow_text is that full flow; "compact_text" then keeps what the model wrote. A full flow (a
    model trained before compact answers) is left exactly as written.
    "clear": the restatement of an answer that opens with "Clear request: ..." (else None); the rest is read as usual.
    A thinking answer ("Understand: ... Facts: ... Missing: ... Steps: ... Check: ...", then "Flow:" or "Question:")
    gives "think": True, its fields (parse_think: "understand", "facts", "missing", "steps", "check"), "clear" = the
    Understand sentence and "plan" = its Steps as plan lines (steps_to_plan). Missing details and no flow: a question
    (the model's own, else ask_question of the missing slots)."""
    t = str(text or "").split("<|assistant_end|>")[0].strip()
    think = None
    if is_think(t):
        think, t = parse_think(t)
        if think["steps"] and (not t or t.startswith(FLOW_HEAD)):
            t = PLAN_HEAD + "\n" + "\n".join(steps_to_plan(think["steps"])) + ("\n" + t if t else "")
        elif not t and think["missing"]:
            t = QUESTION_HEAD + " " + missing_question(think["missing"])
    clear = think["understand"] if think else None
    if t.startswith(CLEAR_HEAD):          # "Clear request: ..." then the usual answer
        m = re.search(r"(?m)^[ \t]*(?=" + "|".join(re.escape(h) for h in (PLAN_HEAD, FLOW_HEAD, QUESTION_HEAD)) + ")", t)
        head = t[len(CLEAR_HEAD):m.start()] if m else t[len(CLEAR_HEAD):]
        clear = " ".join(head.split()) or None
        t = t[m.start():].strip() if m else ""
    res = {"kind": "reply", "plan": [], "flow_text": None, "question": None, "text": t, "clear": clear}
    if clear and not t:
        res["kind"] = "flow"              # cut off after the restatement: a flow answer without plan or flow yet
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
    if think is not None:
        res.update(think=True, **think)
        if not t and (think["understand"] or think["steps"]):
            res["kind"] = "flow"          # cut off inside the block: a flow answer without a flow yet
    if expand and res["kind"] == "flow" and res["flow_text"] is not None:
        full = answer_flow_text(res, catalog, request, known)
        if full is not None and full != res["flow_text"]:
            res.update(compact_text=res["flow_text"], flow_text=full, expanded=True)
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


# ---------------------------------------------------------------------------------------------- app-aware retrieval
# Toolkits of named apps and services: their actions are offered when the request names the app (APP_NAMES) and are
# demoted to the end of the list (the context budget trims them first) when it does not. Without such a name the
# local / core toolkits (file, folder, pdf, excel, word, csv, datatable, email, browser, ocr, printer ...) come first:
# "make a new folder" is a folder on disk, not a Box folder; "delete the files in temp" is ooa.file, not SFTP.
_I = re.I
APP_NAMES = {
    "ooa.slack": [r"\bslack\b"],
    "ooa.jira": [r"\bjira\b"],
    "ooa.file_transfer": [r"\bs?ftps?\b", r"\bfile transfer\b"],
    "ooa.aws_ec2": [r"\bec2\b", r"\baws\b(?! (?:textract|comprehend|rekognition|ai\b))"],
    "ooa.aws_ai": [r"\btextract\b", r"\bcomprehend\b", r"\brekognition\b", r"\baws ai\b"],
    "ooa.azure_devops": [r"\bazure devops\b", r"\bado\b(?= (?:board|work item|pipeline|test))", r"\bwork items?\b"],
    "ooa.azure_vm": [r"\bazure (?:vms?|virtual machines?)\b"],
    "ooa.azure_ai": [r"\bazure (?:ai|cognitive|form recognizer|document intelligence|openai)\b"],
    "ooa.servicenow": [r"\bservice ?now\b", r"\bsnow (?:incident|ticket|table)"],
    "ooa.zendesk": [r"\bzendesk\b"],
    "ooa.salesforce": [r"\bsalesforce\b", r"\bsfdc\b"],
    "ooa.docusign": [r"\bdocu ?sign\b"],
    "ooa.google": [r"\bgoogle (?:sheets?|drive|docs?|calendar|workspace|mail)\b", r"\bg ?mail\b", r"\bgsheets?\b", r"\bgdrive\b"],
    "ooa.google_ai": [r"\bgoogle (?:cloud )?(?:ai|vision|document ai|natural language|translate api)\b", r"\bvertex\b"],
    "ooa.microsoft365": [r"\bmicrosoft 365\b", r"\b(?:office|o|m) ?365\b", r"\bone ?drive\b", r"\bshare ?point\b",
                         r"\boutlook\b", r"\b(?:ms |microsoft )?teams (?:channel|chat|message|meeting)s?\b",
                         r"\b(?:on|in|via|to|through|over) (?:ms |microsoft )?teams\b", r"\bmicrosoft teams\b",
                         r"\bexchange online\b", r"\bmicrosoft graph\b"],
    "ooa.outlook_desktop": [r"\boutlook\b"],
    "ooa.exchange": [r"\bexchange (?:server|on-?prem\w*|mailbox|ews)\b", r"\bews\b"],
    "ooa.box": [r"\bbox\.com\b", r"(?<!text )(?<!check )(?<!search )(?<!input )(?<!message )(?<!dialog )(?<!list )"
                r"(?<!combo )(?<!tick )(?<!drop )(?<!in)\bbox (?:folder|drive|account|storage)\b",
                r"\b(?:on|in|to|from|into) (?:our |my |the )?box\b(?! (?:of|with)\b)"],
    "ooa.snowflake": [r"\bsnowflake\b"],
    "ooa.databricks": [r"\bdatabricks\b"],
    "ooa.kafka": [r"\bkafka\b"],
    "ooa.github": [r"\bgit ?hub\b"],
    "ooa.okta": [r"\bokta\b"],
    "ooa.workday": [r"\bworkday\b"],
    "ooa.asana": [r"\basana\b"],
    "ooa.active_directory": [r"\bactive directory\b", r"(?-i:\bAD\b)", r"\bldap\b", r"\bad (?:user|group|account)s?\b",
                             r"\bdomain (?:user|account|group)s?\b"],
    "ooa.database": [r"\bdata ?base\b", r"\bsql\b", r"\bdb\b", r"\bmysql\b", r"\bpostgre(?:s|sql)\b", r"\boracle\b",
                     r"\bsql server\b", r"\bsqlite\b"],
    "ooa.sap": [r"\bsap\b"],
    "ooa.sap_bapi": [r"\bbapi\b", r"\bsap\b"],
}
_APP_RX = {t: [re.compile(p, _I) for p in ps] for t, ps in APP_NAMES.items()}
APP_TOP = 3            # best actions of each named app, by the request's words (always kept by trimming)
MAX_APP_IDS = 8        # at most this many such kept actions in all
_CONNECT = ("connect", "authenticate", "login", "sign_in")
DEMOTED_FILL = 3       # at most this many actions of apps the request does not name, only to fill a short list


def named_apps(request):
    """Toolkit ids of the apps / services the request names (APP_NAMES), in toolkit order:
    'post it in the #sales slack channel' -> ['ooa.slack']; 'move the files to the E drive' -> []."""
    text = str(request or "")
    return [t for t, rxs in _APP_RX.items() if any(r.search(text) for r in rxs)]


def _has(rx, text):
    return re.search(rx, text, _I) is not None


_PDF_TARGET = r"(?:\b(?:to|into|as|in|ko|me|mein)\s+(?:a\s+|an\s+)?pdf\b|->\s*pdf\b|\bpdf\s+(?:format|version|copy|file)\b|\blike a pdf\b)"
_EXCEL = r"\.xlsx?\b|\bexcel\b|\bworkbook\b|\bspreadsheet\b"
_WORD = r"\.docx?\b|\bword (?:doc|file|document)|\bdocument\b"
_READ = (r"\b(?:total|sum|add up|filter|read|go through|goes through|each|every|har|list is in|details|padh\w*|khol\w*|"
         r"count|how many|kitne|look|check|find|in it|from it|with every\w*|has each)\b")
VERB_HINTS = [
    # (pattern, action ids, pattern that must NOT match too or None) ; first matching rules first
    (r"\bprint(?:s|ed|ing)?\b(?! ?screen)|\bprintout\b", ["ooa.printer/print_file", "ooa.file/print_files"], None),
    (r"\bsummar(?:y|ies|i[sz]e[sd]?|i[sz]ing)\b|\bsum it up\b|\bgist\b|\btl;? ?dr\b", ["ooa.genai/summarize"], None),
    (r"\btranslat\w*|\b(?:in|into|to) (?:hindi|english|french|german|spanish|tamil|telugu|marathi|bengali|gujarati|"
     r"kannada|arabic|chinese|japanese)\b", ["ooa.genai/translate"], None),
    (r"\bforward\w*\b|\bgoes on to\b|\bpass (?:it|them) on\b", ["ooa.email/forward"], None),
    (r"[\w.+-]+@[\w-]+\.[\w.]+|\b(?:e-?mail|mail)\s+(?:it|them|this|that|everyone|every\w*|each|me|him|her)\b|"
     r"\bon mail\b|\bmail (?:karna|kar do|bhej)", ["ooa.email/send"], None),
    (r"\bunread\b|\binbox\b(?! folder)|\bmailbox\b|\bmails?\b.{0,40}\bsubject\b", ["ooa.email/list"], None),
    (r"\bnew (?:sub)?folder\b|\b(?:make|create|add)(?: \w+){0,2} (?:new )?(?:sub)?folder\b", ["ooa.folder/create"], None),
    (r"\b(?:delete|remove|get rid of|clear out|clean out|empty|purge|wipe|hata\w*)\b(?! (?:the )?dupli)",
     ["ooa.file/delete"], r"\bdupli|\btwice\b|do baar"),
    (r"\b(?:delete|remove|get rid of)\s+(?:that|the|this|my)?\s*folder\b", ["ooa.folder/delete"], None),
    (r"\bunzip\w*|\bun-zip|\.zip\b.{0,30}\b(?:open|extract|khol)|\b(?:open|extract|khol)\w*\b.{0,30}\.zip\b",
     ["ooa.file/unzip"], None),
    (r"\bzip (?:up|it|the|them|a|my)\b|\bcompress\w*|\bmake it smaller\b", ["ooa.file/zip"], None),
    (r"dupli\w*|\bdupe\w*|\btwice\b|\bdo baar\b", ["ooa.datatable/remove_duplicates"], None),
    (r"\bhow many rows\b|\bcount (?:the |its )?rows\b|\brow count\b|\bnumber of rows\b|\bkitne\b|\bginti\b",
     ["ooa.datatable/row_count"], None),
    (r"\bscreen ?shot\b|\bphoto (?:of|khinch)\w*|\bkhinch\w*\b|\bsnapshot of\b", ["ooa.browser/screenshot"], None),
    (r"https?://\S+\.(?:pdf|csv|xlsx?|zip|docx?|txt|json|xml)\b", ["ooa.browser/download_url"], None),
    (r"\bre(?:name|nmae)\w*|\brenam\w*", ["ooa.file/move"], None),
    (r"\b(?:move[sd]?|moving|shift\w*)\b", ["ooa.file/move"], r"\b(?:move|shift)\w* (?:the |all |each |every )?(?:e-?mails?|mails?|messages?)\b"),
    (r"\bpdfs?\b(?! attachments?\b).{0,60}\b(?:read|text|total|amount|number|extract|mentions?|padh\w*|nikal\w*)\b|"
     r"\b(?:read|text|extract|padh\w*)\b.{0,30}\bpdfs?\b", ["ooa.pdf/extract_text"], r"\bpdf\s+(?:tables?|forms?)\b"),
    (r"\b(?:each|every|all|har|jitni|sab\w*)\b[^.;]{0,40}\b(?:files?|pdfs?|documents?|images?|photos?|scans?|"
     r"stuff|everything)\b[^.;]{0,40}\bfold(?:e)?r\b|\b(?:list|everything) (?:of |in )?(?:my |the )?\w+ folder\b",
     ["ooa.file/list"], None),
    (r"\b(?:writ\w*|save\w*|log\w*|likh\w*|put)\b.{0,40}\b(?:text file|txt)\b", ["ooa.file/write_text"], None),
]
_VERB_RX = [(re.compile(p, _I), ids, re.compile(n, _I) if n else None) for p, ids, n in VERB_HINTS]


def verb_hints(request):
    """Actions the request's everyday verbs point at, where word search goes wrong (VERB_HINTS): 'print invoice.pdf'
    -> the printer actions; 'turn report.xlsx into a pdf' -> Excel's Export to PDF; 'summary of minutes.pdf' ->
    pdf text + summarize; 'letter.txt but in hindi' -> read the text + translate. Ids may be missing from a catalog."""
    text = str(request or "")
    out = []
    if _has(_PDF_TARGET, text) and not _has(r"\bpdf\s+(?:to|into|se)\s+(?:text|excel|word|csv)", text):
        if _has(_EXCEL, text):
            out += ["ooa.excel_desktop/export_pdf"]
        if _has(_WORD, text):
            out += ["ooa.word/open", "ooa.word/save_as"]
    for rx, ids, neg in _VERB_RX:
        if rx.search(text) and not (neg and neg.search(text)):
            out += ids
    if any(a in out for a in ("ooa.genai/summarize", "ooa.genai/translate")):
        if _has(r"\.pdf\b|\bpdf\b", text):
            out.append("ooa.pdf/extract_text")
        if _has(r"\.txt\b|\btext file\b", text):
            out.append("ooa.file/read_text")
        if _has(r"\.docx?\b|\bword\b", text):
            out.append("ooa.word/extract_text")
    if _has(_EXCEL, text) and _has(_READ, text) and "ooa.excel_desktop/export_pdf" not in out:
        out += ["ooa.excel/open", "ooa.excel/get_all_cells"]
    if _has(r"\.csv\b", text) or _has(r"\bcsv\b", text) and _has(_READ, text):
        out.append("ooa.csv/read")
    return list(dict.fromkeys(out))


def app_actions(request, cat, index, named=None):
    """(kept, extra) for the apps the request names: kept = the APP_TOP best actions of each named app for the
    request's words (at most MAX_APP_IDS; trimming keeps them), extra = each such app's connect / sign-in action."""
    named = named_apps(request) if named is None else named
    kept, extra = [], []
    if index is None or not named:
        return kept, extra
    for t in named:
        try:
            hits = index.search(request, APP_TOP, only={t})
        except TypeError:            # an older index without toolkit filtering
            hits = [e for e in index.search(request, 200) if str(e.get("id", "")).startswith(t + "/")][:APP_TOP]
        kept += [e["id"] for e in hits if e["id"] in cat]
        extra += [f"{t}/{c}" for c in _CONNECT if f"{t}/{c}" in cat][:1]
    kept = list(dict.fromkeys(kept))[:MAX_APP_IDS]
    return kept, [a for a in dict.fromkeys(extra) if a not in kept]


def demoted(action_id, named):
    """True for an action of an app toolkit (APP_NAMES) the request does not name."""
    t = str(action_id).split("/")[0]
    return t in APP_NAMES and t not in named


def context_actions(request, gold=(), catalog=None, index=None, k=DEFAULT_K, rng=None, examples=(), example_index=None,
                    clause_k=CLAUSE_K, clause_votes=CLAUSE_VOTES, exclude_request=None, exclude_sig=None,
                    max_actions=MAX_CONTEXT_ACTIONS, withheld=(), app_aware=True):
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
    row itself out of the votes (training). withheld: ids never listed (training's gold dropout, see drop_gold): the
    row then looks like a serving request whose retrieval missed them.
    app_aware (default on, at training and at serving): the actions of the apps the request names (named_apps) and
    the actions its everyday verbs point at (verb_hints) come first, the named apps' best ones are kept by trimming,
    and the actions of apps it does not name go last (demoted); False: the context as before."""
    cat = _as_dict(catalog) or (index.by_id if index is not None else {})
    gone = set(withheld or ())
    if gone:
        cat = {a: e for a, e in cat.items() if a not in gone}
    gold = [a for a in dict.fromkeys(gold) if a in cat]
    named = named_apps(request) if app_aware else []
    low = (lambda a: demoted(a, named)) if app_aware else (lambda a: False)
    # app-aware: twice as many hits, those of apps the request does not name after the others (then cut to k)
    found = ([e["id"] for e in index.search(request, (2 * k if app_aware else k) + len(gone)) if e["id"] in cat]
             if index is not None else [])
    found = ([a for a in found if not low(a)] + [a for a in found if low(a)])[:k]
    core = [c for c in CORE_ALWAYS if c in cat]
    shown = []
    for x in examples or []:
        for a in x.get("actions") or [] if isinstance(x, dict) else []:
            if a in cat and a not in shown and a not in gold and a not in core and len(shown) < EXAMPLE_ACTIONS:
                shown.append(a)
    more = _clause_ids(request, cat, index, example_index, clause_k, clause_votes, exclude_request, exclude_sig)
    # app-aware: the named apps' best actions first (kept by trimming), then the verb hints; apps the request does
    # not name go to the end, so the context budget drops them first
    app_kept, app_extra = app_actions(request, cat, index, named) if app_aware else ([], [])
    hints = [a for a in verb_hints(request) if a in cat] if app_aware else []
    front = [a for a in dict.fromkeys(app_kept + app_extra + hints) if a not in core]
    late = [a for a in shown if low(a)]
    shown = [a for a in shown if not low(a)]

    def demote(seq):
        """Unnamed apps' actions after the others; only as many of them as fill the list up to k."""
        seq = list(dict.fromkeys(seq))
        good = [a for a in seq if not low(a)]
        return good + [a for a in seq if low(a)][:max(0, min(DEMOTED_FILL, k - len(good)))]

    if gold:
        ids = list(dict.fromkeys(gold + core + shown))
        pool = [a for a in demote(front + found[:FOUND_FIRST] + more + found + late) if a not in ids]
        extra = len([a for a in dict.fromkeys(more + front) if a not in found and a not in shown and a not in core])
        cap = max(min(k + 2 + len(shown) + extra, max_actions), len(ids))
        for a in pool:
            if len(ids) >= cap:
                break
            ids.append(a)
    else:
        ids = demote(front + shown + found + late)
        ids = [a for a in ids if a not in core] + core
        if more:
            ids = [a for a in demote(front + shown + found[:FOUND_FIRST] + more + found + late) if a not in core]
            ids = ids[:max(0, max_actions - len(core))] + core
    if rng is not None:
        rng.shuffle(ids)
    required = list(dict.fromkeys(gold + core + [a for a in app_kept if a in ids]))
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
            for i, s in self._weights(w):
                scores[i] = scores.get(i, 0.0) + s
        ex_req = " ".join(str(exclude_request).split()).lower() if exclude_request else None
        # best first by (-score, index), the same order as sorting every candidate, but only the top few are ranked:
        # a small heap first, the full sort only when skipped rows (excluded / same outline) use the top up
        top = max(4 * n, 32)
        while True:
            ranked = (heapq.nsmallest(top, scores, key=lambda i: (-scores[i], i)) if top < len(scores)
                      else sorted(scores, key=lambda i: (-scores[i], i)))
            out, outlines = [], set()
            for i in ranked:
                r = self.rows[i]
                if (ex_req and r["request"].lower() == ex_req) or (exclude_sig and r["sig"] == exclude_sig):
                    continue
                if r["outline"] in outlines:
                    continue
                outlines.add(r["outline"])
                out.append({"request": r["request"], "outline": r["outline"], "actions": r["actions"]})
                if len(out) >= n:
                    return out
            if top >= len(scores):
                return out
            top *= 4

    def _weights(self, w):
        """BM25 weight of term w in each of its first 5000 postings, computed once per term (same value and order as
        computing it on every search)."""
        cache = self.__dict__.setdefault("_wcache", {})
        got = cache.get(w)
        if got is None:
            idf = self.idf[w]
            got = [(i, idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)))
                   for i, f in self.inv[w][:5000]]
            cache[w] = got
        return got


# ---------------------------------------------------------------------------------------------- training rows
def input_variables(flow):
    return [(v["name"], v.get("type") or "Any") for v in flow.get("variables") or []
            if isinstance(v, dict) and v.get("name") and v.get("direction") == "input"]


GOLD_DROPOUT = 0.5               # share of flow rows whose context misses some of the needed actions (render default)
GOLD_DROPOUT_FRAC = (0.2, 0.45)  # share of the needed actions such a row misses (at least one)
NEVER_DROPPED = CORE_ALWAYS      # core/loop and core/if are always offered at serving too


def hide_in_example(example, withheld):
    """The example with every withheld action id replaced by '...' in its outline and left out of its actions, so
    gold dropout really hides those actions from the whole prompt."""
    gone = set(withheld)
    if isinstance(example, dict):
        ol = str(example.get("outline", ""))
        acts = [a for a in example.get("actions") or [] if a not in gone]
        req = example.get("request", "")
    else:
        req, ol, acts = example[0], str(example[1]), []
    for a in sorted(gone, key=len, reverse=True):
        ol = re.sub(r"(?<![\w./])" + re.escape(a) + r"(?![\w/])", "...", ol)
    return {**example, "outline": ol, "actions": acts} if isinstance(example, dict) else {"request": req, "outline": ol, "actions": acts}


def drop_gold(gold, rng, rate=GOLD_DROPOUT, frac=GOLD_DROPOUT_FRAC):
    """Gold dropout for one training row: the needed actions the context will NOT list. At serving, retrieval offers
    only part of the needed actions (56% on whole business processes), but a model trained with every needed action
    always listed trusts the list and picks a wrong app when the right one is missing. So for `rate` of the rows,
    a share drawn from `frac` of the droppable needed actions (never NEVER_DROPPED; at least one, and one is kept
    when there are two or more) is left out; the plan and flow still use them. Draws from rng only when rate > 0,
    so a render without dropout is reproduced exactly."""
    if not rate or rate <= 0:
        return []
    u, f = rng.random(), rng.uniform(*frac)
    cand = [a for a in dict.fromkeys(gold) if a not in NEVER_DROPPED]
    if u >= rate or not cand:
        return []
    n = max(1, round(f * len(cand)))
    if len(cand) >= 2:
        n = min(n, len(cand) - 1)
    return rng.sample(cand, n)


def render_chat(record, catalog, action_index, example_index=None, rng=None, k=DEFAULT_K, variables_rate=0.3,
                max_chars=MAX_ROW_CHARS, max_tokens=MAX_ROW_TOKENS, context_chars=None, answer_format=ANSWER_FULL,
                gold_dropout=0.0, dropout_frac=GOLD_DROPOUT_FRAC, think=False):
    """A neutral record -> {"messages": [user, assistant]} in this format (None when the record is unusable).
    record: {"kind": "flow", "request", "flow"} | {"kind": "fix", "request", "previous", "problems", "flow"} |
            {"kind": "clarify", "request", "question"} |
            {"kind": "rewrite", "request", "flow", "clear"}: a flow row whose answer opens with "Clear request: <clear>"
            (clear_request(flow) when the record has none).
    The draws from rng are made in a fixed order (number of examples, variables, shuffle), so a seed reproduces a
    dataset.
    A row that does not fit (row_fits) is rendered again with a smaller context (down to the required actions only); the
    caller still checks the final length. context_chars: the context budget (default: context_for the window that
    max_tokens implies, the same budget the server gives a model with that window).
    answer_format: ANSWER_FULL (the flow as stored) or ANSWER_COMPACT (compact_flow: code fills in the rest at
    serving, expand_flow). gold_dropout: share of flow / fix rows whose Actions list misses some needed actions
    (drop_gold); the answer is unchanged.
    think: the answer is a thinking answer (render_think_answer / render_think_question: Understand, Facts, Missing,
    Steps, Check before the flow or question); a record's own boolean "think" overrides it. A rewrite record's "clear"
    then becomes the Understand sentence (no "Clear request:" line)."""
    import random
    rng = rng or random.Random(0)
    cat = _as_dict(catalog)
    if context_chars is None:
        context_chars = context_for(window_of(max_tokens))
    kind, request = record.get("kind"), str(record.get("request") or "").strip()
    if not request or kind not in ("flow", "fix", "clarify", "rewrite"):
        return None
    if isinstance(record.get("think"), bool):
        think = record["think"]
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
        answer = render_think_question(record) if think else QUESTION_HEAD + " " + q
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}
    flow = record.get("flow")
    if not isinstance(flow, dict) or not isinstance(flow.get("steps"), list) or not flow["steps"]:
        return None
    flow = to_dollar_name_dollar(flow)              # the platform's notation: the model learns to write $name$
    examples = (example_index.search(request, n_ex, exclude_request=request, exclude_sig=flow_signature(flow))
                if example_index else [])
    # fix rows show the previous flow in full, so hiding its actions from the list would hide nothing
    withheld = drop_gold(actions_of(flow), rng, gold_dropout, dropout_frac) if kind != "fix" else []
    actions, required = context_actions(request, actions_of(flow), cat, action_index, k, rng, examples, example_index,
                                        exclude_request=request, exclude_sig=flow_signature(flow), withheld=withheld)
    if withheld:                         # a withheld action must not be visible in the similar examples either
        examples = [hide_in_example(x, withheld) for x in examples]
    # variables the request came with (Studio's open agent) always; else the flow's inputs for some rows
    variables = record.get("variables") or (input_variables(flow) if use_vars else ())
    previous = problems = None
    if kind == "fix":
        previous, problems = record.get("previous"), [str(p) for p in record.get("problems") or []]
        if previous is None or not problems:
            return None
        previous = to_dollar_name_dollar(previous)  # as the server shows it (Router._previous)
    clear = None
    if think:
        answer = render_think_answer(flow, cat, answer_format, record, request)
    else:
        if kind == "rewrite":
            clear = str(record.get("clear") or "").strip() or clear_request(flow, cat)
        answer = render_flow_answer(flow, cat, answer_format, clear)
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


# ---------------------------------------------------------------------------------------------- agent team (serving)
# Helpers for serving/ooa_server.Router's agent team: which part of a plan waits for which (parallel builders), the
# assembler's shared variables, the critic's coverage check and the auto mode's request features. They read the
# formats above; they write nothing the model is trained on.

def part_dependencies(parts):
    """For plan_parts' parts: the part each one must wait for, or None. A part that goes inside a container (inside
    >= 1) needs the part that wrote that container: the nearest part before it at a lower level. Top-level parts need
    nothing and can all be built at once."""
    deps = []
    for i, p in enumerate(parts):
        level, dep = int(p.get("inside") or 0), None
        if level:
            for j in range(i - 1, -1, -1):
                if int(parts[j].get("inside") or 0) < level:
                    dep = j
                    break
        deps.append(dep)
    return deps


def _rename_text(s, old, new):
    s = re.sub(r"\$" + re.escape(old) + r"\$", "$" + new + "$", s)
    s = re.sub(r"\$\{(\s*)" + re.escape(old) + r"(?=[\s}.\[])", "${\\g<1>" + new, s)
    return re.sub(r"\$" + re.escape(old) + r"(?![\w$])", "$" + new, s)


def rename_variable(flow, old, new):
    """A copy of flow (or a step list) with variable old called new: declarations, $refs in args ($x$, ${x}, $x, also
    x.field), out targets, loop items and core/set targets."""
    def go(v, key=None, parent_action=None):
        if isinstance(v, str):
            if key in ("variable", "as", "index", "name") and v == old:
                return new
            return _rename_text(v, old, new)
        if isinstance(v, list):
            return [go(x, key, parent_action) for x in v]
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if k == "out" and isinstance(x, dict):
                    out[k] = {ok: (new if ov == old else go(ov)) for ok, ov in x.items()}
                elif k in ("action", "id", "kind", "description", "type", "direction"):
                    out[k] = x
                else:
                    out[k] = go(x, k)
            return out
        return v
    return go(json.loads(compact(flow)))


_SCALAR_TYPES = {"String", "Number", "Boolean", "DateTime", "Time", "TimeSpan", "Credential"}


def _name_words(name):
    return {w for w in re.split(r"[_\W\d]+", str(name).lower()) if len(w) > 2}


def link_part_variables(parts, known=()):
    """The assembler's shared variables. Parts built at the same time cannot see each other's variables, so a later
    part may read a value under its own name ('$rows' for what an earlier part made as 'table'). For each part, a
    variable it takes as input that no earlier part declares and the open agent does not have is linked to a variable
    an earlier part made (local or output): one whose name shares a word with it (the latest such), else, for a
    non-text type (Table, List, Dictionary, File ...), the latest one of the same type. Returns (parts, links) with
    links as (part number from 1, old name, new name)."""
    have = {v.get("name") for v in known if isinstance(v, dict)} | {v for v in known if isinstance(v, str)}
    made, declared, out, links = [], set(have), [], []
    for n, p in enumerate(parts, 1):
        p = json.loads(compact(p))
        for v in list(p.get("variables") or []):
            name = v.get("name") if isinstance(v, dict) else None
            if not name or v.get("direction") != "input" or name in declared or n == 1:
                continue
            words = _name_words(name)
            pick = next((m for m in reversed(made) if words & _name_words(m["name"])), None)
            if pick is None and v.get("type") not in _SCALAR_TYPES:
                pick = next((m for m in reversed(made) if m.get("type") == v.get("type") and v.get("type")), None)
            if pick is None or pick["name"] == name:
                continue
            p = rename_variable(p, name, pick["name"])
            p["variables"] = [x for x in p.get("variables") or [] if not (isinstance(x, dict) and x.get("name") ==
                                                                           pick["name"] and x.get("direction") == "input")]
            links.append((n, name, pick["name"]))
        for v in p.get("variables") or []:
            if isinstance(v, dict) and v.get("name"):
                declared.add(v["name"])
                if v.get("direction") != "input":
                    made.append(v)
        out.append(p)
    return out, links


# words that make a clause a thing to do (the critic checks such clauses only); English, and common Hinglish verbs
# ("list" and "log" are left out: "each file in a list", "log in to the site" are not steps of their own)
ACTION_WORDS = set("""read write save send email mail copy move delete remove rename download upload open close
extract filter sort merge split zip unzip compress convert print create make add update insert append fill
click type search find check compare calculate sum count export import fetch get post call run start stop attach
reply forward archive backup notify alert schedule scan parse translate summarize resize crop rotate encrypt
decrypt sign share sync publish submit approve reject padh likh bhej bhejo karo kar nikal daal hata""".split())
# verbs that name the same kind of step ("save it as a CSV" and ooa.csv/write)
VERB_GROUPS = [{"save", "write", "store", "export"}, {"make", "create", "new"}, {"get", "read", "extract", "fetch", "pull"},
               {"add", "sum", "insert", "append", "total"}, {"send", "email", "mail", "forward"},
               {"delete", "remove"}, {"find", "search", "look"}, {"zip", "compress"}, {"padh", "read"},
               {"likh", "write"}, {"bhej", "bhejo", "send"}, {"hata", "delete"}]


def clause_has_action(clause):
    """True when a request clause names something to do (a word of ACTION_WORDS, plural / -ing / -ed folded)."""
    for w in re.findall(r"[a-z]+", str(clause).lower()):
        for form in (w, w[:-1] if w.endswith("s") else w, w[:-3] if w.endswith("ing") else w,
                     w[:-2] if w.endswith("ed") else w, w[:-1] if w.endswith("e") else w):
            if form in ACTION_WORDS:
                return True
    return False


def _verbs(text):
    """The ACTION_WORDS of a text, plural / -ing / -ed folded, with the verbs of the same group (VERB_GROUPS)."""
    out = set()
    for w in re.findall(r"[a-z]+", str(text).lower()):
        for form in (w, w[:-1], w[:-3], w[:-2], w + "e"):
            if form in ACTION_WORDS:
                out.add(form)
    for g in VERB_GROUPS:
        if out & g:
            out |= g
    return out


def uncovered_clauses(request, flow, index, k=6, plan=None):
    """The critic: request clauses (request_clauses) that name something to do (clause_has_action) and that no step
    of the flow covers. A clause is covered when one of the flow's actions is among its k best catalog matches
    (index.search); or a step's action names the clause's verb (id or catalog name: "delete the original" and
    ooa.file/delete); or a step's action shares 2+ words with the clause (id, name and description: "count its rows"
    and ooa.datatable/row_count); or a plan line repeats half its words (the model planned it). Kept strict on purpose:
    what it reports is built as a new part, so a false alarm would add steps nobody asked for."""
    have = list(dict.fromkeys(actions_of(flow))) if isinstance(flow, dict) else []
    if not have:
        return []
    entries = {e["id"]: e for e in getattr(index, "entries", [])}
    names, words_of = {}, {}
    for a in have:
        e = entries.get(a, {})
        names[a] = set(re.findall(r"[a-z]+", (a + " " + str(e.get("name", ""))).lower()))    # verbs are stop words there
        words_of[a] = names[a] | set(retrieval.tokens(str(e.get("description", ""))))
    plan_words = [set(retrieval.tokens(i["text"])) for i in plan_items(plan or [])]
    out = []
    for c in request_clauses(request):
        if not clause_has_action(c):
            continue
        if set(have) & {e["id"] for e in index.search(c, k)}:
            continue
        words, verbs = set(retrieval.tokens(c)), _verbs(c)
        if any(verbs & names[a] for a in have) or any(len(words & words_of[a]) >= 2 for a in have):
            continue
        if words and any(len(words & pw) >= max(2, (len(words) + 1) // 2) for pw in plan_words):
            continue
        out.append(c)
    return out


_LOOP_WORDS = re.compile(r"\b(each|every|loop|repeat|for every|one by one|har|while|until)\b",
                         re.I)
_COND_WORDS = re.compile(r"\b(if|when|whenever|otherwise|else|unless|only|in case|agar|jab|warna|nahi to)\b", re.I)
_THING_WORDS = re.compile(r"(file|folder|excel|xlsx|xls|sheet|csv|pdf|docx?|word|txt|text|json|xml|zip|image|photo|"
                          r"e-?mail|mail|outlook|gmail|inbox|attachment|browser|website|web ?page|url|link|portal|"
                          r"database|sql|table|api|ftp|sftp|sap|erp|crm|invoice|report|form|calendar|teams|slack|"
                          r"whatsapp|sms|printer|drive|sharepoint|onedrive|directory|path|cell|row|column|[\\/]\w)", re.I)


def request_features(request):
    """What the auto mode looks at: clauses (request_clauses), words, loop words, condition words, and whether the
    request names any file, app or data thing at all (has_thing; none suggests a vague beginner request)."""
    text = str(request or "")
    return {"clauses": len(request_clauses(text)), "words": len(text.split()),
            "loop": bool(_LOOP_WORDS.search(text)), "condition": bool(_COND_WORDS.search(text)),
            "has_thing": bool(_THING_WORDS.search(text))}


# ---------------------------------------------------------------------------------------------- ask-first gate
# A request that points at something only the user knows ("the usual folder", "rename them properly", "put it in the
# tracker") is answered with one short question before anything is built (serving/ooa_server.Router, ask-first layer).
# Two kinds of vague words:
#   sure    always ask, whatever else is named: "the usual folder", "the right person", "the same file as last time",
#           "our system", "the important ones" (the beginner clarify data's own phrases, gen_beginner.SLOTS)
#   loose   ask only when nothing concrete is named: a pronoun with nothing it can stand for ("mail them when its
#           ready"), "the tracker", "the system", "the app", "the website", "the report", "that document", "stuff"
# Concrete: a file name (.csv, .xlsx ...), a path, a web address, an email address, a named app (named_apps), a named
# folder ("the Reports folder", "my photos folder"), or a kind of thing given generally ("a CSV file", "all PDFs",
# "every email in the inbox", "an Excel workbook"): the model then builds with inputs for the details.
ASK_SLOTS = {
    "folder": ("Which folder do you mean?", "Please tell me its name or path (for example C:\\Invoices)."),
    "file": ("Which file do you mean?", "Please tell me its name and which folder it is in."),
    "files": ("Which files do you mean?", "Please tell me the folder they are in, or their names."),
    "recipient": ("Who should get it?", "Please give me their name or email address."),
    "app": ("Which app or system do you mean?", "Please tell me its name (for example Outlook, Jira or Salesforce)."),
    "website": ("Which website do you mean?", "Please give me its address (for example https://example.com)."),
    "condition": ("How do I tell which ones you mean?", "Tell me the rule, for example amount above 10,000."),
    "sheet": ("Which sheet or file do you mean?", "Please tell me the file's name and where it is saved."),
    "report": ("Which report do you mean?", "Please tell me the file's name and where it is saved."),
    "data": ("Where does the data come from?", "Please tell me the file, sheet or app it is in."),
    "list": ("Which list do you mean?", "Please tell me the file or sheet it is in."),
    "place": ("Where should it go?", "Please tell me the folder's name or path."),
    "what": ("What exactly should the bot do, and with which files or app?", "Please name the file, folder or app."),
}
_ASK_SURE = [
    (r"\b(?:the|our|my) usual (?:folder|place|location|directory)\b|\bthe folder we always use\b|\bour normal folder\b|"
     r"\bthe right folder\b|\bthe other folder\b|\bthe right place\b|\bsomewhere\b", "folder"),
    (r"\bthe usual (?:file|sheet|workbook|excel|document)\b|\bthe (?:file|sheet|workbook) (?:we|i) al?ways use\b|"
     r"\bthe same file as last time\b", "file"),
    (r"\bthe usual (?:people|guys|person|team|recipients?)\b|\b(?:to )?(?:the )?right person\b|"
     r"\bthe people who need it\b|\busual guys\b", "recipient"),
    (r"\b(?:our|the) (?:usual )?(?:system|tool|app)\b(?!'s| (?:tray|settings|clock|date|time|log|folder|servers?|project|data|store))|"
     r"\bthe app we use\b|\bthe usual tool\b|\bour usual app\b", "app"),
    (r"\bthe (?:important|small|big|right|relevant|good|bad) ones\b|\bthat (?:are big|matter|look right|are the "
     r"important ones)\b", "condition"),
    (r"\b(?:files?|stuff|things?) i (?:do ?n[o']?t|dont) need\b|\bmy old stuff\b", "files"),
    (r"\bthe usual (?:report|summary|numbers|data|list|one)\b", "report"),
    (r"\bthe excel thing\b|\bthe (?:sheet|spreadsheet) thing\b", "sheet"),
]
_ASK_LOOSE = [
    (r"\bthe (?:website|site|portal|web ?page)\b", "website"),
    (r"\bthat (?:document|file|sheet|workbook|doc)\b|\bthe (?:sheet|files|documents)\b", "file"),
    (r"\bstuff\b|\bthings\b", "files"),
    (r"\bthe (?:report|summary)\b", "report"),
    (r"\bthe list\b", "list"),
    (r"\bthe (?:sales |)(?:data|numbers|figures|entries)\b|\btodays? entries\b|\buska data\b", "data"),
    (r"\b(?:e-?mail|mail|send|share|forward)\w* (?:it|them|this|that)\b(?! to\b)", "recipient"),
    (r"\bthe (?:system|app|application|tool)\b(?!'s| (?:servers?|info|log|tray|settings|project|data|store))", "app"),
    (r"\bthe tracker\b", "sheet"),
    (r"\bpeople\b|\beveryone\b", "recipient"),
    (r"^(?:\S+\s+){0,2}(?:them|those|these|it|they|usko|unko|uska|unka|isko)\b|^jo bhi\b", "files"),
]
_ASK_SURE_RX = [(re.compile(p, re.I), s) for p, s in _ASK_SURE]
_ASK_LOOSE_RX = [(re.compile(p, re.I), s) for p, s in _ASK_LOOSE]
_EXT = r"(?:csv|xlsx|xlsm|xls|docx|doc|pdf|txt|json|xml|zip|png|jpe?g|pptx?|msg|eml|html?|log|tiff?|bmp|gif|rtf|odt)"
_CONCRETE = re.compile(
    r"[\w-]\.(?:" + _EXT + r")\b|\b(?:dot|\.) ?(?:" + _EXT + r")\b|"           # a file name (also "orders dot csv")
    r"\b[A-Za-z]:\\|\\\\\w|(?:^|\s)~?/\w+/|"                                     # a path
    r"https?://|\bwww\.|\b[\w-]+\.(?:com|in|org|net|io|local|gov)\b|"            # a web address / host
    r"[\w.+-]+@[\w-]+\.\w|"                                                      # an email address
    r"#[a-z][\w-]+|\bcell [A-Z]+\d+\b|\b[A-Z]{1,2}\d{1,5}\b|"                   # a channel, a cell
    r"\b(?:a|an|any|all|each|every|some|new|two|several|multiple|the following|har|saari|sab|jitni)\s+(?:of the\s+)?"
    r"(?:\w+\s+){0,2}?(?:csv|pdf|excel|xlsx|word|docx|text|json|xml|zip|image|photo|png|jpe?g|scan|invoice|"
    r"e-?mail|mail|message|attachment|files?|folders?|workbooks?|documents?|spreadsheets?|sheets?|web ?(?:site|page)s?|"
    r"websites?|urls?|links?|tables?|rows?|records?|tickets?|issues?|orders?|customers?|vendors?|pdfs|csvs)s?\b|"
    r"\b(?:inbox|mailbox|outlook|gmail|desktop|downloads|documents folder|clipboard|printer|"
    r"excel (?:file|workbook|sheet)|csv file|pdf files?|word (?:file|document|doc))\b",
    re.I)
_TYPE_WORDS = re.compile(
    r"\b(?:pdfs?|csvs?|excel|xlsx|workbooks?|spreadsheets?|word (?:files?|documents?|docs?)|docx|jpe?g|png|images?|"
    r"photos?|scans?|scanned|zip|e-?mails|mails|inbox|mailbox|attachments?|browser|text files?|json|xml|"
    r"(?:[\w'-]+ )(?:files?|workbook|statement|csv|pdf|document|log|table|invoices?))\b", re.I)
_NOT_A_TYPE = re.compile(r"\b(?:that|the usual|the same|the right|the other|this|those|these) (?:files?|document|"
                         r"workbook|table|log)\b", re.I)
_NAMED_FOLDER = re.compile(r"\b(?:the |my |our |in |from |to |into )?([\w&'-]+) (?:sub)?fold(?:e)?r\b", re.I)
_NAMED_PLACE = re.compile(r"\b(?:to|into|in|from|under|inside) (?:the )?[A-Z][\w&'-]+")   # "copy it into Vendor Docs"
_VAGUE_FOLDER_WORDS = {"the", "a", "an", "that", "this", "other", "right", "usual", "same", "some", "any", "new",
                       "another", "correct", "proper", "normal", "which", "my", "our", "your", "one", "each", "every",
                       "certain", "given", "specific", "target", "destination", "source", "relevant"}


def concrete_things(request):
    """What the request names concretely: file names, paths, addresses, apps, named folders, kinds of things given
    generally ('a CSV file', 'every email in the inbox'). Empty: it names nothing the bot can act on."""
    text = str(request or "")
    out = [m.group(0).strip() for m in _CONCRETE.finditer(text)]
    out += [m.group(0) for m in _TYPE_WORDS.finditer(_NOT_A_TYPE.sub(" ", text))]
    out += named_apps(text)
    out += [m.group(0) for m in _NAMED_FOLDER.finditer(text) if m.group(1).lower() not in _VAGUE_FOLDER_WORDS]
    out += [m.group(0) for m in _NAMED_PLACE.finditer(text)]
    return out


def ask_first(request, has_variables=False):
    """The ask-first gate: None when the request can be built, else {"missing": [slots], "question": text} for a
    request that points at something only the user knows (sure vague words always; loose ones only when nothing
    concrete is named, see concrete_things). has_variables: the open agent already has variables the request may
    refer to (Studio): then only the sure words ask."""
    text = " ".join(str(request or "").split())
    if not text:
        return None
    missing = [s for rx, s in _ASK_SURE_RX if rx.search(text)]
    if not missing and not has_variables and not concrete_things(text):
        missing = [s for rx, s in _ASK_LOOSE_RX if rx.search(text)]
    missing = list(dict.fromkeys(missing))
    if not missing:
        return None
    return {"missing": missing, "question": ask_question(missing)}


def ask_question(missing):
    """One short, friendly question about the missing details (at most two of them), in the wording style of the
    beginner clarify data: 'Which folder do you mean? Please tell me its name or path (for example C:\\Invoices).'"""
    slots = [s for s in missing if s in ASK_SLOTS][:2] or ["what"]
    if len(slots) == 1:
        q, hint = ASK_SLOTS[slots[0]]
        return f"Happy to help. {q} {hint}"
    (q1, h1), (q2, _) = ASK_SLOTS[slots[0]], ASK_SLOTS[slots[1]]
    return f"Happy to help, I just need two details. {q1} {h1} And {q2[0].lower() + q2[1:]}"
