"""Catalog knowledge for the OOA assistant: action cards, search, lessons, plain-English flow explanations and quizzes.

No model and no torch: everything here comes from the catalog (catalog.jsonl), which is the source of truth.

    k = Knowledge("catalog.jsonl")
    k.search("read a csv file")          -> short cards
    k.explain_flow(flow)                 -> step-by-step plain English
    q = k.quiz(); k.check_quiz(q["id"], 2)

Quiz ids are stateless: the quiz state (kind, action, choices, answer) is encrypted and signed with a random per-process
key, so the server can check an answer without storing anything and the browser can neither read nor forge the answer.
Ids stop being valid when the process restarts.
"""
import base64, collections, hashlib, hmac, json, os, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from retrieval import ActionIndex  # noqa: E402

BLOCK_WORDS = {
    "if": "inside the if branch", "else": "inside the else branch", "elseif": "inside an else-if branch",
    "loop": "inside the loop", "try": "inside the try block", "catch": "inside the error handler (catch)",
    "finally": "inside the finally block", None: "inside the group",
}
OPS = {
    "eq": "equals", "ne": "does not equal", "neq": "does not equal", "gt": "is greater than", "gte": "is at least",
    "ge": "is at least", "lt": "is less than", "lte": "is at most", "le": "is at most", "contains": "contains",
    "not_contains": "does not contain", "starts_with": "starts with", "ends_with": "ends with", "matches": "matches",
    "in": "is one of", "not_in": "is not one of",
}
UNARY = {"empty": "is empty", "not_empty": "is not empty", "true": "is true", "false": "is false",
         "exists": "exists", "not_exists": "does not exist", "is_empty": "is empty", "is_not_empty": "is not empty"}
QUIZ_KINDS = ("which_action", "required_inputs", "which_toolkit")


def _list(x):
    """x if it is a list, else [] (malformed flows must not crash the explainer)."""
    return x if isinstance(x, list) else []


def _short(text, n=80):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _val(v, n=80):
    """Display form of an argument value."""
    if isinstance(v, str):
        return _short(v, n)
    return _short(json.dumps(v, ensure_ascii=False), n)


class Knowledge:
    def __init__(self, catalog_path: str):
        with open(catalog_path, encoding="utf-8") as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        self.by_id = {}
        for e in self.entries:      # a few ids exist as both condition and action: keep the action
            if e["id"] not in self.by_id or e["kind"] == "action":
                self.by_id[e["id"]] = e
        self.index = ActionIndex(self.entries)
        self._tk = collections.OrderedDict()
        for e in self.entries:
            self._tk.setdefault(e["toolkit_id"], {"id": e["toolkit_id"], "name": e["toolkit"], "entries": []})
            self._tk[e["toolkit_id"]]["entries"].append(e)
        self.actions = [e for e in self.entries if e["kind"] == "action"]
        self._key = os.urandom(32)

    # ---------------------------------------------------------------- catalog
    def toolkits(self):
        out = [{"id": t["id"], "name": t["name"], "count": len(t["entries"])} for t in self._tk.values()]
        return sorted(out, key=lambda t: (t["name"].lower(), t["id"]))

    @staticmethod
    def _label(e):
        return f"{e['toolkit']} -> {e['name']}"

    @staticmethod
    def _short_card(e):
        return {"id": e["id"], "name": e["name"], "toolkit": e["toolkit"], "description": e["description"],
                "kind": e["kind"]}

    def action_card(self, action_id):
        e = self.by_id.get(action_id)
        if e is None:
            return None
        io = lambda i: {"name": i["name"], "type": i.get("type", ""), "description": i.get("description", "")}
        return {"id": e["id"], "name": e["name"], "toolkit": e["toolkit"], "toolkit_id": e["toolkit_id"],
                "kind": e["kind"], "description": e["description"], "notes": e.get("notes", ""),
                "required": [io(i) for i in e["inputs"] if i.get("required")],
                "optional": [io(i) for i in e["inputs"] if not i.get("required")],
                "outputs": [io(o) for o in e["outputs"]]}

    def search(self, query, k=10):
        if not query or not str(query).strip():
            return []
        q = str(query).strip()
        hits = self.index.search(q, max(1, int(k)))
        exact = self.by_id.get(q)       # an exact id typed in the box goes first
        if exact and exact["kind"] == "action":
            hits = [exact] + [h for h in hits if h["id"] != q][: max(0, int(k) - 1)]
        return [self._short_card(e) for e in hits]

    def lesson(self, toolkit_id):
        """None when the toolkit id is unknown."""
        t = self._tk.get(toolkit_id)
        if t is None:
            return None
        ents = sorted(t["entries"], key=lambda e: (e["kind"] != "action", e["name"].lower()))
        acts = [e for e in ents if e["kind"] == "action"]
        verbs = collections.Counter(e["name"].split()[0].lower() for e in acts if e["name"].split())
        top = [v for v, _ in verbs.most_common(5)]
        n_other = len(ents) - len(acts)
        intro = f"The {t['name']} toolkit has {len(acts)} action{'s' if len(acts) != 1 else ''}"
        intro += (f" (plus {n_other} condition or loop type{'s' if n_other != 1 else ''})." if n_other else ".")
        if top:
            examples = ", ".join(e["name"] for e in acts[:3])
            intro += f" Its actions mostly {', '.join(top[:-1]) + ' and ' + top[-1] if len(top) > 1 else top[0]}" \
                     f" things, for example: {examples}."
        return {"toolkit": t["name"], "toolkit_id": t["id"], "intro": intro,
                "actions": [self._short_card(e) for e in ents]}

    # ---------------------------------------------------------------- flow explanation
    def _cond_text(self, c):
        if isinstance(c, str):
            return _short(c, 120)
        if isinstance(c, bool) or c is None or isinstance(c, (int, float)):
            return json.dumps(c)
        if isinstance(c, list):
            return " and ".join(self._cond_text(x) for x in c) or "(no condition)"
        if not isinstance(c, dict):
            return _val(c)
        for key, word in (("all", " and "), ("and", " and "), ("any", " or "), ("or", " or ")):
            if isinstance(c.get(key), list):
                return "(" + word.join(self._cond_text(x) for x in c[key]) + ")"
        if "not" in c:
            return "not (" + self._cond_text(c["not"]) + ")"
        if isinstance(c.get("condition"), str) and "/" in c["condition"] and " " not in c["condition"]:
            e = self.by_id.get(c["condition"])
            name = f"'{e['name']}' ({e['toolkit']})" if e else f"'{c['condition']}' (not in the catalog)"
            args = c.get("args") if isinstance(c.get("args"), dict) else {}
            extra = ", ".join(f"{k}={_val(v, 40)}" for k, v in list(args.items())[:3] if k != "object")
            return f"the check {name}" + (f" with {extra}" if extra else "") + " is true"
        if "condition" in c:
            return self._cond_text(c["condition"])
        if "expr" in c:
            return _short(c["expr"], 120)
        op = str(c.get("op", "")).lower()
        left = _val(c.get("left", ""), 40)
        if op in UNARY:
            return f"{left} {UNARY[op]}"
        if op:
            return f"{left} {OPS.get(op, op)} {_val(c.get('right', ''), 40)}"
        return _val(c)

    def _step_what(self, s, e):
        a = s.get("action")
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        if a == "core/if":
            t = f"Check whether {self._cond_text(args.get('condition'))}; if so, run the steps inside it"
            for b in _list(s.get("elseIf")):
                if isinstance(b, dict):
                    t += f"; otherwise, if {self._cond_text(b.get('condition'))}, run that branch"
            if s.get("else"):
                t += "; otherwise run the else branch"
            return t + "."
        if a == "core/loop":
            it = args.get("iterator")
            if it == "list" or "over" in args:
                t = f"Repeat the steps inside for each item of {_val(args.get('over', '?'), 40)}"
                if args.get("as"):
                    t += f", calling the current item {args['as']}"
            elif it == "times" or "count" in args:
                t = f"Repeat the steps inside {_val(args.get('count', '?'), 30)} times"
            elif it == "while" or "condition" in args:
                t = f"Repeat the steps inside while {self._cond_text(args.get('condition'))}"
            elif isinstance(it, str) and it in self.by_id:
                t = f"Repeat the steps inside for each {self.by_id[it]['name'].lower()} ({self.by_id[it]['description']})"
            else:
                t = f"Repeat the steps inside ({_val(it, 30) if it else 'loop'})"
            if args.get("max_iterations"):
                t += f", at most {_val(args['max_iterations'], 20)} times"
            return t + "."
        if a == "core/try":
            t = "Run the steps inside; if one fails, run the error handler"
            if s.get("finally"):
                t += ", and always run the finally block at the end"
            return t + "."
        if a == "core/set" and "variable" in args:
            return f"Set the variable {args['variable']} to {_val(args.get('value', ''), 60)}."
        if a == "core/log":
            return f"Write a log message: {_val(args.get('message', ''), 80)}."
        if a == "core/comment":
            return f"Note (does nothing): {_val(args.get('text', ''), 80)}."
        if e is None:
            return f"The action '{a}' is not in the catalog, so this step cannot be checked or run."
        return e["description"]

    def _args_list(self, s, e):
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        ins = {i["name"]: i for i in e["inputs"]} if e else {}
        out = []
        for k, v in args.items():
            i = ins.get(k)
            if i:
                meaning = i.get("description", "")
            elif e:
                meaning = "Not a known input of this action."
            else:
                meaning = ""
            out.append({"name": k, "value": _val(v), "meaning": meaning})
        return out

    def explain_flow(self, flow):
        if not isinstance(flow, dict):
            return {"summary": "This is not a flow (expected a JSON object).", "inputs": [], "outputs": [], "steps": []}
        variables = [v for v in _list(flow.get("variables")) if isinstance(v, dict)]
        vio = lambda v: {"name": str(v.get("name", "")), "type": str(v.get("type", "")),
                         "description": str(v.get("description", ""))}
        inputs = [vio(v) for v in variables if v.get("direction") in ("input", "inout", "in_out")]
        outputs = [vio(v) for v in variables if v.get("direction") in ("output", "inout", "in_out")]
        steps = []

        def add(nodes, depth, block, where):
            for s in nodes if isinstance(nodes, list) else []:
                if not isinstance(s, dict):
                    continue
                a = s.get("action")
                a = a if isinstance(a, str) else str(a)
                e = self.by_id.get(a)
                n = len(steps) + 1
                title = self._label(e) if e else a
                what = self._step_what(s, e)
                args = self._args_list(s, e)
                outs_d = s.get("out") if isinstance(s.get("out"), dict) else {}
                outs = [{"name": str(k), "variable": _val(v, 40)} for k, v in outs_d.items()]
                text = f"Step {n}" + (f" ({where})" if where else "") + f": {title}: {what}"
                if s.get("description"):
                    text += f" Note: {_short(s['description'], 100)}."
                reads = [f"{x['name']}={x['value']}" for x in args if "$" in x["value"]]
                if reads and a not in ("core/if", "core/loop", "core/set"):
                    text += " Reads from: " + ", ".join(reads) + "."
                if outs:
                    descs = {o["name"]: o.get("description", "") for o in (e["outputs"] if e else [])}
                    parts = []
                    for o in outs:
                        d = (descs.get(o["name"]) or "").strip().rstrip(".")
                        what_o = (d[0].lower() + d[1:]) if d else f"the {o['name']}"
                        parts.append(f"{what_o} in: {o['variable']}")
                    text += " Saves " + "; ".join(parts) + "."
                steps.append({"id": str(s.get("id", f"step{n}")), "depth": depth, "action": a, "title": title,
                              "what": text, "args": args, "outs": outs, "known": e is not None, "block": block})
                if a == "core/if":
                    add(s.get("children"), depth + 1, "if", "inside the if branch")
                    for b in _list(s.get("elseIf")):
                        if isinstance(b, dict):
                            add(b.get("children"), depth + 1, "elseif",
                                f"inside 'otherwise, if {_short(self._cond_text(b.get('condition')), 50)}'")
                    add(s.get("else"), depth + 1, "else", "inside the else branch")
                else:
                    block_c = {"core/loop": "loop", "core/try": "try"}.get(a)
                    add(s.get("children"), depth + 1, block_c, BLOCK_WORDS[block_c])
                    for b in _list(s.get("elseIf")):      # unusual, but never drop steps
                        if isinstance(b, dict):
                            add(b.get("children"), depth + 1, "elseif", BLOCK_WORDS["elseif"])
                    add(s.get("else"), depth + 1, "else", BLOCK_WORDS["else"])
                for c in _list(s.get("catch")):
                    if isinstance(c, dict):
                        codes = [str(x) for x in _list(c.get("codes"))]
                        where_c = "inside the error handler" + (f" for {', '.join(codes)}" if codes else "")
                        add(c.get("children"), depth + 1, "catch", where_c)
                add(s.get("finally"), depth + 1, "finally", BLOCK_WORDS["finally"])

        add(flow.get("steps"), 0, None, "")
        unknown = sorted({s["action"] for s in steps if not s["known"]})
        tks = []
        for s in steps:
            e = self.by_id.get(s["action"])
            if e and e["toolkit_id"] != "core" and e["toolkit"] not in tks:
                tks.append(e["toolkit"])
        name = flow.get("name") or flow.get("id") or "This flow"
        summary = f"{name}: " if flow.get("name") or flow.get("id") else ""
        if flow.get("description"):
            summary += str(flow["description"]).rstrip(".") + ". "
        summary += f"It has {len(steps)} step{'s' if len(steps) != 1 else ''}"
        summary += f" using {', '.join(tks)}." if tks else "."
        if inputs:
            summary += " You give it: " + ", ".join(i["name"] for i in inputs) + "."
        if outputs:
            summary += " It gives back: " + ", ".join(o["name"] for o in outputs) + "."
        if unknown:
            summary += " Not in the catalog: " + ", ".join(unknown) + "."
        return {"summary": summary, "inputs": inputs, "outputs": outputs, "steps": steps}

    # ---------------------------------------------------------------- quizzes
    def _seal(self, state):
        data = json.dumps(state, separators=(",", ":")).encode()
        nonce = os.urandom(12)
        ct = bytes(b ^ k for b, k in zip(data, self._stream(nonce, len(data))))
        tag = hmac.new(self._key, b"mac" + nonce + ct, hashlib.sha256).digest()[:16]
        return base64.urlsafe_b64encode(nonce + tag + ct).decode().rstrip("=")

    def _open(self, token):
        try:
            raw = base64.urlsafe_b64decode(str(token) + "=" * (-len(str(token)) % 4))
        except Exception:
            raise ValueError("invalid quiz id")
        if len(raw) < 29:
            raise ValueError("invalid quiz id")
        nonce, tag, ct = raw[:12], raw[12:28], raw[28:]
        good = hmac.new(self._key, b"mac" + nonce + ct, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, good):
            raise ValueError("invalid or expired quiz id")
        return json.loads(bytes(b ^ k for b, k in zip(ct, self._stream(nonce, len(ct)))))

    def _stream(self, nonce, n):
        out, i = b"", 0
        while len(out) < n:
            out += hmac.new(self._key, b"enc" + nonce + i.to_bytes(4, "big"), hashlib.sha256).digest()
            i += 1
        return out[:n]

    @staticmethod
    def _req_text(e):
        req = sorted(i["name"] for i in e["inputs"] if i.get("required"))
        return ", ".join(req) if req else "(no required inputs)"

    def _pick_distinct(self, rng, correct, same_pool, other_pool, n=3):
        """n distinct wrong choices, from same_pool first then other_pool."""
        wrong = []
        for pool in (same_pool, other_pool):
            pool = list(dict.fromkeys(x for x in pool if x != correct and x not in wrong))
            rng.shuffle(pool)
            wrong += pool[: n - len(wrong)]
            if len(wrong) >= n:
                break
        return wrong

    def quiz(self, rng=None, kind=None):
        rng = rng or random.Random()
        if kind is None:
            kind = rng.choice(QUIZ_KINDS)
        if kind not in QUIZ_KINDS:
            raise ValueError(f"unknown quiz kind {kind!r}; use one of {', '.join(QUIZ_KINDS)}")
        for _ in range(200):
            e = rng.choice(self.actions)
            same = [x for x in self.actions if x["toolkit_id"] == e["toolkit_id"] and x["id"] != e["id"]]
            if kind == "which_action":
                correct = self._label(e)
                question = f"Which action does this: \"{e['description']}\""
                wrong = self._pick_distinct(rng, correct,
                                            [self._label(x) for x in same if x["description"] != e["description"]],
                                            [self._label(x) for x in rng.sample(self.actions, 40)
                                             if x["description"] != e["description"]])
            elif kind == "required_inputs":
                if not any(i.get("required") for i in e["inputs"]):
                    continue
                correct = self._req_text(e)
                question = f"Which inputs does {self._label(e)} require?"
                mutations = []
                opt = [i["name"] for i in e["inputs"] if not i.get("required")]
                req = sorted(i["name"] for i in e["inputs"] if i.get("required"))
                if opt:
                    mutations.append(", ".join(sorted(req + [rng.choice(opt)])))
                if len(req) > 1:
                    drop = rng.choice(req)
                    mutations.append(", ".join(r for r in req if r != drop))
                wrong = self._pick_distinct(rng, correct, [self._req_text(x) for x in same] + mutations,
                                            [self._req_text(x) for x in rng.sample(self.actions, 40)])
            else:
                correct = e["toolkit"]
                question = f"Which toolkit has the action \"{e['name']}\" ({e['description'].rstrip('.')})?"
                # only toolkits with no action of the same name and description, so there is one right answer
                names = [t["name"] for t in self._tk.values()
                         if not any(x["kind"] == "action" and x["name"] == e["name"]
                                    and x["description"] == e["description"] for x in t["entries"])]
                wrong = self._pick_distinct(rng, correct, [], names)
            if len(wrong) < 3:
                continue
            choices = wrong[:3] + [correct]
            rng.shuffle(choices)
            state = {"k": kind, "a": e["id"], "c": choices, "x": choices.index(correct)}
            return {"id": self._seal(state), "kind": kind, "question": question, "choices": choices}
        raise RuntimeError("could not build a quiz from this catalog")

    def check_quiz(self, quiz_id, choice_index):
        """Raises ValueError for a forged/expired id or a choice outside 0..3."""
        st = self._open(quiz_id)
        try:
            choice = int(choice_index)
        except (TypeError, ValueError):
            raise ValueError("choice must be a number 0-3")
        if not 0 <= choice < len(st["c"]):
            raise ValueError("choice must be a number 0-3")
        e = self.by_id.get(st["a"])
        ans = st["x"]
        expl = f"The answer is \"{st['c'][ans]}\"."
        if e:
            expl += f" {self._label(e)} ({e['id']}): {e['description']}"
            expl += f" Required inputs: {self._req_text(e)}."
        return {"correct": choice == ans, "answer_index": ans, "explanation": expl}
