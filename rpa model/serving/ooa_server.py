"""OOA assistant server: one OpenAI-compatible endpoint, a router, a verifier, and client data-policy switches.

   request -> [mask PII for cloud] -> router
        simple?  -> LOCAL model (our own): context (actions, similar examples, variables) -> plan + flow
                    -> question? return it -> repair -> verify + plan coverage -> ok? return
                    -> self-correction (previous flow + problems) -> long plan: build it in parts and merge
        else / failed -> escalation model if the client policy allows (Anthropic API, or any OpenAI-compatible
                    endpoint: escalate_url) -> repair -> verify (one retry with the errors)
        verified -> safety warnings for destructive steps (notes, "warnings", the ooa field)
        nothing verified -> honest error (never returns an unchecked flow as if it were fine)
   POST /v1/chat/completions   OpenAI-style; the last user message is the request; the answer is the flow JSON
   POST /v1/feedback           {"request","flow","verdict": inserted|discarded|edited,"edited_flow"} (or the older
                               {"prompt","flow","accepted"}) -> feedback.jsonl (only if the policy allows)
   GET  /health

Run:  python ooa_server.py --catalog catalog.jsonl --platform PLATFORM_REPO --policy policy.json [--local nanochat]
                           [--examples examples.jsonl] [--prompt-format auto|plain|context] [--port 8080]
Local backend: any object with generate(prompt, attempt) -> str (see backends.py). Escalation: Anthropic Messages API
(env ANTHROPIC_API_KEY) or an OpenAI-compatible endpoint (policy escalate_url / escalate_model / escalate_key_env).
Default policy is LOCAL-ONLY and stores nothing.
"""
import argparse, inspect, json, os, re, sys, time, threading
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))
import ooa_reward as R
import ooa_format as F
import retrieval
import repair as RP
import safety


@dataclass
class Policy:
    allow_escalation: bool = False       # may a request leave for the cloud model at all
    mask_pii: bool = True                # mask emails, phones, card-like numbers before anything leaves
    store_logs: bool = False             # keep a log of requests and results (masked)
    allow_training_on_data: bool = False  # may logged data / feedback be used to train models
    local_retries: int = 2
    max_clauses_local: int = 6           # longer / more complex requests go to the cloud straight away ...
    local_first: bool = False            # ... unless this is on: our own model tries every request first
    cloud_model: str = "claude-sonnet-5-5"
    cloud_max_tokens: int = 6000
    log_dir: str = "ooa_logs"
    # escalation to any OpenAI-compatible endpoint instead of the Anthropic API (off unless allow_escalation is on)
    escalate_url: str = ""               # e.g. https://gateway.example/v1 ; empty: the Anthropic API (cloud_model)
    escalate_model: str = ""
    escalate_key_env: str = "OOA_ESCALATE_API_KEY"   # environment variable that holds its API key
    escalate_plan_lines: int = 12        # a failed local flow whose plan is longer goes to that model straight away
    # the local model's layers (see Router)
    prompt_format: str = "auto"          # "plain" | "context" | "auto" (context when the model pack has examples.jsonl)
    context_k: int = 12                  # BM25 candidate actions in the context
    context_chars: int = 0               # context budget in characters; 0: what the model was trained with
    decompose_lines: int = 6             # a plan longer than this is built in parts (at once with parts_first)
    part_attempts: int = 2               # attempts per part (the second sees the first one's problems)
    parts_first: bool = True             # a long plan: stop the answer after its plan and build the flow in parts at
                                         # once (needs a backend whose generate takes on_plan; else after one failure)
    long_clauses: int = 5                # a request of this many clauses (ooa_format.request_clauses) or more ...
    long_plan_lines: int = 4             # ... is built in parts when its plan is longer than this
    max_tokens: int = 2500               # tokens for a whole answer (the backend also keeps it inside its window)
    part_max_tokens: int = 1500          # tokens for the answer of one part
    part_plan_given: bool = True         # a part's answer starts with its plan lines (backends that take prefix)
    part_lines: int = 5                  # plan lines per part (ooa_format.plan_parts; a long loop / try / if is cut too)

    @classmethod
    def load(cls, path=None):
        d = json.load(open(path)) if path else {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
       (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD_OR_ID_NUMBER]"),
       (re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d"), "[PHONE]"),
       (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "[IP]")]


def mask(text):
    for rx, rep in PII:
        text = rx.sub(rep, text)
    return text


def clauses(text):
    parts = re.split(r",|;|\bthen\b|\bafter that\b|\band\b", text, flags=re.I)
    return len([p for p in parts if len(p.split()) >= 2])


def extract_json(text):
    text = R.clean(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if i != -1 and j > i else text


def parse_messages(messages):
    """OpenAI-style messages -> (request, system, assistant_mode, existing_variables, full_user_text).
    The platform's Studio assistant sends: system = long instructions + its action catalog, user = 'Existing variables: [...]\nRequest: ...'
    and reads back {"variables": [...], "steps": [...], "notes": "..."}. Anything else is treated as a plain request."""
    system = "\n".join(m["content"] for m in messages if m.get("role") == "system" and isinstance(m.get("content"), str))
    users = [m for m in messages if m.get("role") == "user"]
    text = users[-1].get("content", "") if users else ""
    if isinstance(text, list):
        text = " ".join(x.get("text", "") for x in text if isinstance(x, dict))
    assistant = text.startswith("Existing variables:") and "\nRequest:" in text
    existing = []
    if assistant:
        try:
            existing = json.loads(text[len("Existing variables:"):text.index("\nRequest:")])
            existing = existing if isinstance(existing, list) else []
        except Exception:
            existing = []
        request = text.split("\nRequest:", 1)[1].strip()
    else:
        request = text
    return request, system, assistant, existing, text


def call_generate(model, prompt, attempt=0, **options):
    """model.generate(prompt, attempt, **options) with only the options its generate takes (all of them when it takes
    **kwargs): max_tokens, on_plan and prefix are newer than some backends (serving/backends.py, test doubles)."""
    fn = model.generate
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        options = {k: v for k, v in options.items() if k in params}
    return fn(prompt, attempt, **{k: v for k, v in options.items() if v is not None})


NOT_A_TASK = ("The model did not read this as a task to automate, so it answered with text instead of a flow. "
              "Describe what the bot should do, step by step, for example: 'Read a CSV file and write it to an "
              "Excel sheet'. Questions about actions go in the Learn tab.")


OUT_OF_TIME = "The time for this request ran out, so no further attempt was made."
LOOPED = "The answer started repeating itself and was stopped."


STUDIO_NOTE = "Written by the OOA assistant and checked against the platform's rules."
ASKED = "the model asked a question instead of guessing: the request does not say enough to automate"


class Router:
    """Layers, in order: retrieval context (actions, similar examples, the open agent's variables) -> plan-first answer
    -> clarify (a question comes back as the answer) -> repair + verify + plan coverage -> self-correction (the next
    attempt sees its previous flow and the problems) -> decomposition (a long plan is built in parts and merged) ->
    escalation (a bigger model through the Anthropic API or any OpenAI-compatible endpoint, when the policy allows)
    -> safety warnings on the verified flow. A model trained before context and plans (prompt_format "plain") gets
    the request only and answers with flow JSON; that path works as before."""

    def __init__(self, catalog, platform, policy, local=None, cloud=None, examples=None, prompt_format=None):
        os.environ["OOA_CATALOG"] = catalog
        os.environ["OOA_SCHEMA"] = str(Path(platform) / "contracts/flow-schema/flow.v1.json")
        R.configure()
        self.cat = {}
        entries = [json.loads(l) for l in open(catalog, encoding="utf-8")]
        for e in entries:
            self.cat[e["id"]] = e
        self.index = retrieval.ActionIndex(entries)
        rules = Path(platform) / "docs/llms.txt"
        self.rules = rules.read_text(encoding="utf-8") if rules.exists() else "Write an OOA Flow v1 JSON document."
        self.policy, self.local, self.cloud = policy, local, cloud
        self.examples = F.ExampleIndex.load(examples) if examples and Path(examples).is_file() else None
        self.trained_context = self._trained_context(examples) if self.examples is not None else None
        self.prompt_format = prompt_format or policy.prompt_format
        self.lock = threading.Lock()

    # -- prompt format ------------------------------------------------------------------------------
    def format(self):
        """"context" (training/ooa_format.build_user) or "plain" (the request only). auto: what the local backend
        says in info["prompt_format"], else context when the model pack came with its example index."""
        if self.prompt_format in ("plain", "context"):
            return self.prompt_format
        info = getattr(self.local, "info", None)
        said = info.get("prompt_format") if isinstance(info, dict) else None
        if said in ("plain", "context"):
            return said
        return "context" if self.examples is not None else "plain"

    def prompt(self, request, existing=None, previous=None, problems=None, required=()):
        """The user turn for the local model: the request + retrieval context (+ previous flow and problems)."""
        examples = self.examples.search(request, F.MAX_EXAMPLES) if self.examples is not None else []
        actions, keep = F.context_actions(request, (), self.cat, self.index, self.policy.context_k, examples=examples,
                                          example_index=self.examples)
        have = {e["id"] for e in actions}
        actions = [self.cat[a] for a in required if a in self.cat and a not in have] + actions
        variables = [v for v in existing or [] if isinstance(v, dict)]
        return F.build_user(request, actions, examples, variables, previous, problems,
                            required=list(dict.fromkeys(list(required) + keep)), context_limit=self.context_limit())

    def sequence_len(self):
        """The local model's window in tokens when its backend says (info["sequence_len"]), else None."""
        info = getattr(self.local, "info", None)
        n = info.get("sequence_len") if isinstance(info, dict) else None
        return n if isinstance(n, int) and n > 0 else None

    def context_limit(self):
        """Characters of context the model gets: the policy's context_chars when set; else the budget its training data
        was rendered with (render_report.json next to examples.jsonl, settings.max_context_chars), never more than
        its window allows (ooa_format.context_for(info["sequence_len"])); else MAX_CONTEXT_CHARS. A bigger context
        than the model was trained with hurts: the first 4096-token model (trained with 2,600 characters) wrote plans
        naming 39% of the needed actions on plain_requests_3 with 4,800 characters, 49% with 2,600."""
        if self.policy.context_chars > 0:
            return self.policy.context_chars
        trained = self.trained_context
        if trained:
            seq = self.sequence_len()
            return min(trained, F.context_for(seq)) if seq else trained
        return F.MAX_CONTEXT_CHARS

    @staticmethod
    def _trained_context(examples):
        """settings.max_context_chars of the render_report.json beside the example index, or None."""
        if not examples:
            return None
        f = Path(examples).resolve().parent / "render_report.json"
        try:
            n = json.loads(f.read_text(encoding="utf-8"))["settings"]["max_context_chars"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return n if isinstance(n, int) and n > 0 else None

    # -- one answer: parse -> repair -> verify -------------------------------------------------------
    def verify(self, text, existing=None, plan=None):
        """Flow text -> (flow dict, repair notes) when it passes, else (None, problems). plan: the answer's plan
        lines; an action the plan names that no step uses is a problem too."""
        if text is None:
            return None, ["The answer has a plan but no flow after 'Flow:'."]
        text = extract_json(text)
        try:
            obj = json.loads(text)
        except Exception:
            return None, R.problems(text)
        if not isinstance(obj, dict):
            return None, ["The answer is not a JSON object."]
        # Studio's assistant format has no kind/id/name: wrap it so the same verifier applies
        flow = obj if "kind" in obj else {"kind": "TaskBot", "id": "assistant_check", "name": "assistant check",
                                          "variables": list(obj.get("variables", [])), "steps": obj.get("steps", [])}
        names = {v.get("name") for v in flow.get("variables", []) if isinstance(v, dict)}
        extra = [v for v in (existing or []) if isinstance(v, dict) and v.get("name") not in names]
        if extra:
            flow = {**flow, "variables": list(flow.get("variables", [])) + extra}   # variables already in the user's open agent
        flow, notes = RP.repair(flow, self.cat)
        problems = R.problems(json.dumps(flow, separators=(",", ":"), ensure_ascii=False))
        problems += [f"The plan names {a} but no step of the flow uses it." for a in F.plan_coverage(plan, flow)]
        if problems:
            return None, problems
        # the platform writes variables $name$ (Studio saves them so, and only $name$ is replaced inside text)
        return F.to_dollar_name_dollar(flow, names | {v.get("name") for v in extra}), notes

    def answer_text(self, flow, existing=None, assistant=False, warnings=()):
        """The verified flow as the client reads it: Studio's {variables, steps, notes, warnings}, or the flow JSON."""
        if assistant:
            have = {v.get("name") for v in (existing or []) if isinstance(v, dict)}
            notes = " ".join([STUDIO_NOTE] + list(warnings))
            out = {"variables": [v for v in flow.get("variables", []) if v.get("name") not in have], "steps": flow["steps"],
                   "notes": notes, "warnings": list(warnings)}
            return json.dumps(out, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(flow, separators=(",", ":"), ensure_ascii=False)

    def check(self, raw, existing=None, assistant=False):
        """One raw answer -> (answer text, repair notes) or (None, problems). Reads plan-first and JSON-only answers."""
        ans = F.parse_answer(raw)
        flow, info = self.verify(ans["flow_text"] if ans["kind"] == "flow" else raw, existing, ans["plan"])
        if flow is None:
            return None, info
        return self.answer_text(flow, existing, assistant, safety.warnings(flow, self.cat)), info

    @staticmethod
    def _previous(text):
        """The previous answer's flow for a self-correction prompt (dict), or None when it was not JSON."""
        if not text:
            return None
        try:
            obj = json.loads(extract_json(text))
        except Exception:
            return None
        return F.to_dollar_name_dollar(obj) if isinstance(obj, dict) else None

    def cloud_prompt(self, request, errors):
        cands = self.index.search(request, 40)
        # whole toolkits of the best matches: descriptions rarely name low-level actions (click, type, open ...)
        top_tk = []
        for e in self.index.search(request, 200):
            if e["toolkit_id"] not in top_tk:
                top_tk.append(e["toolkit_id"])
            if len(top_tk) == 4:
                break
        lines = {e["id"]: retrieval.candidate_line(e) for e in cands}
        for e in self.index.entries:
            if e["toolkit_id"] in top_tk and len(lines) < 160:
                lines.setdefault(e["id"], retrieval.candidate_line(e))
        for cid in ("core/if", "core/loop", "core/try", "core/set", "core/log"):
            if cid in self.cat:
                lines.setdefault(cid, retrieval.candidate_line(self.cat[cid]))
        system = (self.rules + "\n\nReturn ONLY one minified OOA Flow JSON object, no prose, no code fences. "
                  "Use only action ids from the list below, with exactly the input names shown (in: required, opt: optional).\n"
                  "Format of each line: action id | in: required inputs | opt: optional inputs | out: outputs\n"
                  + "\n".join(lines.values()))
        user = request if not errors else request + "\n\nYour previous answer had these problems, fix them:\n- " + "\n- ".join(errors)
        return system, user

    # -- the local model's layers ------------------------------------------------------------------------
    def generate(self, prompt, attempt, **options):
        """The local model's answer; options (max_tokens, on_plan, prefix) only when its generate takes them. Returns
        (text, stop) where stop is the backend's last_stats["stop"] ("end", "max_tokens", "loop: ...", "plan: ...")
        or None when it does not say."""
        raw = call_generate(self.local, prompt, attempt, **options)
        st = getattr(self.local, "last_stats", None)
        stop = st.get("stop") if isinstance(st, dict) else None
        return raw, stop if isinstance(stop, str) else None

    def parts_rule(self, request, can_escalate=False):
        """on_plan for the first answer: True (stop after the plan) when the plan is long enough to build in parts
        (more than decompose_lines lines, or than long_plan_lines for a request of long_clauses clauses or more) and
        splits into 2+ parts, or when it goes to the escalation model anyway (longer than escalate_plan_lines)."""
        p = self.policy
        limit = p.long_plan_lines if len(F.request_clauses(request)) >= p.long_clauses else p.decompose_lines

        def rule(plan):
            if can_escalate and len(plan) > p.escalate_plan_lines:
                return True
            return len(plan) > limit and len(self.parts_of(plan)) >= 2
        return rule

    def _local(self, request, existing, tried, late, can_escalate):
        """Attempts of the local model. Returns {"flow", "notes", "plan", "parts"} on success, {"question"} when it
        asks, else {"errors", "plan"}. A long plan is built in parts at once (parts_first: the first answer stops
        after its plan); an answer that loops (the backend's guards) counts as a failed attempt."""
        p, fmt = self.policy, self.format()
        prev = probs = None
        plan, errors, decomposed = [], [], False
        rule = self.parts_rule(request, can_escalate) if fmt == "context" and p.parts_first else None
        for attempt in range(1 + p.local_retries):
            if late():
                return {"errors": list(errors) + [OUT_OF_TIME], "plan": plan}
            prompt = self.prompt(request, existing, prev, probs) if fmt == "context" else request
            try:
                raw, stop = self.generate(prompt, attempt, max_tokens=p.max_tokens,
                                          on_plan=rule if not decomposed else None)
            except Exception as ex:
                return {"errors": [f"local model error: {ex}"], "plan": plan}
            ans = F.parse_answer(raw)
            if ans["kind"] == "question":
                tried.append({"source": "local", "ok": True, "kind": "question"})
                return {"question": ans["question"], "plan": []}
            if stop and stop.startswith("plan") and ans["plan"]:
                plan = ans["plan"]
                tried.append({"source": "local", "ok": True, "plan_lines": len(plan), "plan_only": True})
                if can_escalate and len(plan) > p.escalate_plan_lines:
                    return {"errors": ["a long process: the plan has %d lines" % len(plan)], "plan": plan}
                decomposed = True
                flow, info = self._decompose(request, plan, existing, tried, late)
                if flow is not None:
                    return {"flow": flow, "notes": info, "plan": plan, "parts": True}
                errors = info
                continue                                     # whole answers next
            flow, info = self.verify(ans["flow_text"] if ans["kind"] == "flow" else raw, existing, ans["plan"])
            tried.append({"source": "local", "ok": flow is not None})
            looped = bool(stop and stop.startswith("loop"))
            if stop:
                tried[-1]["stop"] = stop
            if ans["plan"] and not (looped and ans["flow_text"] is None):     # a plan that loops is not a plan
                plan = ans["plan"]
                tried[-1]["plan_lines"] = len(plan)
            if flow is not None:
                return {"flow": flow, "notes": info, "plan": ans["plan"]}
            tried[-1]["preview"] = (raw or "")[:300]      # what the model wrote, so a failure can be read
            errors = info
            if looped:
                errors = [LOOPED] + list(info)
            if ans["kind"] == "reply":
                # Text instead of a flow: the request was not read as a task (a question, an error message pasted
                # in, chat). A retry only loops on the same text, and forcing JSON makes up an unrelated flow, so
                # stop here and say so.
                return {"errors": [NOT_A_TASK], "plan": plan}
            if fmt == "context":
                # self-correction: the next attempt sees the flow it wrote and what the checker found (a looping
                # answer has no flow to show: the next attempt starts afresh at a higher temperature)
                prev = None if looped else self._previous(ans["flow_text"])
                probs = None if looped else info
            if plan and can_escalate and len(plan) > p.escalate_plan_lines:
                return {"errors": errors, "plan": plan}      # a long process: the bigger model takes it
            if not decomposed and len(plan) > p.decompose_lines and len(self.parts_of(plan)) >= 2:
                break                                        # a long plan: build it in parts instead of retrying
        if plan and not decomposed and len(self.parts_of(plan)) >= 2 and not late():
            flow, info = self._decompose(request, plan, existing, tried, late)
            if flow is not None:
                return {"flow": flow, "notes": info, "plan": plan, "parts": True}
            errors = list(errors) + info
        elif plan and late():
            errors = list(errors) + [OUT_OF_TIME]
        return {"errors": errors, "plan": plan}

    def parts_of(self, plan):
        """How a plan is built in parts (ooa_format.plan_parts with part_lines): [{"chunk", "inside"}]."""
        return F.plan_parts(plan, max_lines=self.policy.part_lines)

    def _decompose(self, request, plan, existing, tried, late):
        """Build a flow from its plan in parts (ooa_format.plan_parts: up to 4 plan items / part_lines lines each; a long
        loop, try or if is cut, its later children built as parts that go inside it), each a short answer
        (part_max_tokens). Each part is its own request with the variables made so far (and the item of each loop a
        part is inside);
        the parts are merged (unique step ids, variables united) and the merged flow must pass the checker and cover
        the plan. Returns (flow, notes) or (None, problems)."""
        chunks = self.parts_of(plan)
        if len(chunks) < 2:
            return None, []
        fmt, have, parts, inside = self.format(), [dict(v) for v in existing or [] if isinstance(v, dict)], [], []
        info = []
        for n, ch in enumerate(chunks, 1):
            chunk = ch["chunk"]
            creq, need = F.chunk_request(chunk), F.chunk_actions(chunk)
            known = list(have)
            if ch["inside"]:
                so_far = F.merge_flows(parts, inside=inside) or {}
                path = F.container_path(so_far.get("steps") or [], ch["inside"])
                if not path:
                    return None, [f"Part {n} of {len(chunks)} goes inside a loop, try or if, but the parts before it "
                                  "did not end with one."]
                # the items of the loops it is inside are known variables there
                items = [b["args"]["as"] for b in path if b.get("action") == "core/loop"
                         and isinstance((b.get("args") or {}).get("as"), str)]
                for item in items:
                    if item not in {v.get("name") for v in known}:
                        known.append({"name": item, "type": "Any", "direction": "local"})
                if items:
                    creq = f"With the current {items[-1]}: {creq[:1].lower() + creq[1:]}"
            part, prev, probs = None, None, None
            prefix = F.answer_prefix(F.chunk_plan_lines(chunk)) if self.policy.part_plan_given else None
            for attempt in range(max(1, self.policy.part_attempts)):
                if late():
                    return None, [OUT_OF_TIME]
                prompt = self.prompt(creq, known, prev, probs, required=need) if fmt == "context" else creq
                try:
                    # the part's plan is known: give it as the answer's start, so the model writes only the flow
                    raw, stop = self.generate(prompt, attempt, max_tokens=self.policy.part_max_tokens,
                                              prefix=prefix if fmt == "context" else None)
                except Exception as ex:
                    return None, [f"local model error: {ex}"]
                ans = F.parse_answer(raw)
                text = ans["flow_text"] if ans["kind"] == "flow" else raw
                part, info = self.verify(text, known, ans["plan"])
                tried.append({"source": "local", "part": n, "of": len(chunks), "ok": part is not None})
                if ch["inside"]:
                    tried[-1]["inside"] = int(ch["inside"])
                if stop and stop != "end":
                    tried[-1]["stop"] = stop
                if part is not None:
                    break
                looped = bool(stop and stop.startswith("loop"))
                prev, probs = (None, None) if looped else (self._previous(text), info)
            if part is None:
                return None, [f"Part {n} of {len(chunks)} ('{creq[:80]}') did not pass the checker: "
                              + "; ".join(info[:2])]
            parts.append(part)
            inside.append(ch["inside"])
            names = {v.get("name") for v in have}
            have += [v for v in part.get("variables", []) if isinstance(v, dict) and v.get("name") not in names]
        merged = F.merge_flows(parts, request, inside)
        flow, info = self.verify(F.compact(merged), existing, plan)
        tried.append({"source": "local", "merged_parts": len(parts), "ok": flow is not None})
        if flow is None:
            return None, ["The flow built in parts did not pass the checker: " + "; ".join(info[:2])]
        return flow, info + [f"built in {len(parts)} parts from the plan"]

    def run(self, request, system=None, assistant=False, existing=None, full_user=None, deadline=None):
        """deadline: time.time() value after which no further attempt (retry or cloud call) starts."""
        t0, tried, errors, plan = time.time(), [], [], []
        late = lambda: deadline is not None and tried and time.time() >= deadline
        p = self.policy
        can_escalate = self.cloud is not None and p.allow_escalation
        simple = p.local_first or (clauses(request) <= p.max_clauses_local and len(request) <= 500)
        if not can_escalate:
            simple = True      # no cloud fallback: always let the local model try instead of failing unseen
        if self.local is not None and simple:
            out = self._local(request, existing, tried, late, can_escalate)
            if out.get("flow") is not None:
                return self.done(request, out["flow"], "local", tried, out["notes"], t0, assistant, existing,
                                 plan=out["plan"], parts=out.get("parts", False))
            if out.get("question"):
                return self.done(request, None, "local", tried, [], t0, assistant, existing, reason=ASKED,
                                 question=out["question"])
            errors, plan = out["errors"], out["plan"]
        elif self.local is not None:
            errors = ["request looks complex; skipped the local model"]
        if can_escalate:
            req = mask(request) if p.mask_pii else request
            err = [mask(e) if p.mask_pii else e for e in errors if not e.startswith("request looks")]
            for attempt in range(2):
                if late():
                    err = list(err) + [OUT_OF_TIME]
                    break
                if assistant and system:      # the platform's own prompt carries its live action catalog: use it
                    sys_p = system
                    user = (mask(full_user) if p.mask_pii else full_user) + ("\n\nYour previous answer had these problems, fix them:\n- " + "\n- ".join(err) if err else "")
                else:
                    sys_p, user = self.cloud_prompt(req, err)
                try:
                    raw = self.cloud.generate(sys_p, user)
                except Exception as ex:
                    err = [f"cloud call failed: {ex}"]
                    tried.append({"source": "cloud", "ok": False, "error": str(ex)[:120]})
                    break
                flow, info = self.verify(raw, existing)
                tried.append({"source": "cloud", "ok": flow is not None})
                if flow is not None:
                    return self.done(request, flow, "cloud", tried, info, t0, assistant, existing, plan=plan)
                err = info
            errors = err
        reason = ("no model produced a flow that passed verification" if can_escalate
                  else "the local model could not produce a verified flow, and cloud escalation is disabled by this client's policy")
        return self.done(request, None, "none", tried, errors or [reason], t0, assistant, existing, reason=reason,
                         plan=plan)

    def done(self, request, flow, source, tried, info, t0, assistant=False, existing=None, reason=None, plan=None,
             question=None, parts=False):
        """flow: the verified flow dict or None. The result's "flow" is the text the client gets (answer_text)."""
        warnings = safety.warnings(flow, self.cat) if flow is not None else []
        text = self.answer_text(flow, existing, assistant, warnings) if flow is not None else None
        res = {"flow": text, "source": source, "verified": flow is not None, "attempts": tried,
               "repairs" if flow is not None else "problems": info, "seconds": round(time.time() - t0, 3),
               "plan": list(plan or []), "warnings": warnings}
        if parts:
            res["built_in_parts"] = True
        if question:
            res["question"] = question
        if reason:
            res["reason"] = reason
        if self.policy.store_logs:
            self.log("requests.jsonl", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "request": mask(request),
                                        "source": source, "verified": res["verified"], "flow": text,
                                        "question": question, "training_ok": self.policy.allow_training_on_data})
        return res

    def log(self, name, row):
        d = Path(self.policy.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        with self.lock, open(d / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def feedback(self, body):
        """POST /v1/feedback. Stored only when the policy allows logs; the request text is masked."""
        if not self.policy.store_logs:
            return {"stored": False, "reason": "policy: store_logs is off"}
        try:
            row = feedback_row(body)
        except ValueError as ex:
            return {"stored": False, "error": str(ex)}
        row = {**row, "prompt": mask(row["prompt"]), "training_ok": self.policy.allow_training_on_data}
        self.log("feedback.jsonl", row)
        return {"stored": True, "usable_for_training": self.policy.allow_training_on_data}


# Studio's verdicts on a suggested flow (inserted / discarded / edited) and the assistant page's (accept / reject /
# edit) -> the stored verdict. training/ingest_feedback.py turns the stored rows into training records.
VERDICTS = {"inserted": "accept", "discarded": "reject", "edited": "edit",
            "accept": "accept", "reject": "reject", "edit": "edit"}


def _flow_arg(v, key):
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError as e:
            raise ValueError(f"{key} is not valid JSON: {e}")
    if not isinstance(v, dict):
        raise ValueError(f"{key} must be a flow JSON object")
    return v


def feedback_row(body):
    """A feedback body -> the row stored in feedback.jsonl, or ValueError with a plain message.
    Accepts {request, flow, verdict: inserted|discarded|edited, edited_flow?} (Studio) and the older
    {prompt, flow, accepted, edited_flow?}."""
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    request = body.get("request", body.get("prompt"))
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request is required (text)")
    if len(request) > 5000:
        raise ValueError("request is too long (max 5000 characters)")
    verdict = body.get("verdict")
    if verdict is None and "accepted" in body:
        verdict = "accept" if body.get("accepted") is True else ("edit" if body.get("edited_flow") else "reject")
    if verdict not in VERDICTS:
        raise ValueError("verdict must be one of inserted, discarded, edited")
    v = VERDICTS[verdict]
    flow, edited = _flow_arg(body.get("flow"), "flow"), _flow_arg(body.get("edited_flow"), "edited_flow")
    if v == "accept" and flow is None:
        raise ValueError("an inserted flow needs the flow")
    if v == "edit" and edited is None:
        raise ValueError("an edited flow needs edited_flow")
    return {"prompt": request.strip(), "flow": flow, "accepted": v == "accept", "edited_flow": edited if v == "edit" else None,
            "verdict": v, "client_verdict": verdict, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


class CloudBackend:
    """Claude through the Anthropic Messages API (needs ANTHROPIC_API_KEY). In production route this through the
    platform's AI Gateway instead (it already adds PII masking, budgets and the governance log)."""
    def __init__(self, model, max_tokens=6000, api_key=None, url="https://api.anthropic.com/v1/messages"):
        self.model, self.max_tokens, self.url = model, max_tokens, url
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def generate(self, system, user):
        import urllib.request
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "system": system,
                           "messages": [{"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.url, body, {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                                      "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAICompatBackend:
    """Any OpenAI-compatible chat completions endpoint (a hosted model, a gateway, a model server on the network).
    url: the base URL ("https://host/v1") or the full ".../chat/completions" URL. The API key is read from the
    environment variable key_env when it is set; endpoints without auth need none."""
    def __init__(self, url, model, key_env="OOA_ESCALATE_API_KEY", max_tokens=6000, timeout=120, api_key=None):
        u = url.rstrip("/")
        self.url = u if u.endswith("/chat/completions") else u + "/chat/completions"
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.key = api_key or (os.environ.get(key_env) if key_env else None)

    def generate(self, system, user):
        import urllib.request
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        headers = {"content-type": "application/json"}
        if self.key:
            headers["authorization"] = "Bearer " + self.key
        with urllib.request.urlopen(urllib.request.Request(self.url, body, headers), timeout=self.timeout) as r:
            data = json.loads(r.read())
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        if isinstance(content, list):            # some servers return content parts
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        if not isinstance(content, str):
            raise RuntimeError("the escalation endpoint returned no message content")
        return content


def make_cloud(policy):
    """The escalation backend the policy names, or None when escalation is off: an OpenAI-compatible endpoint when
    escalate_url is set, else the Anthropic API."""
    if not policy.allow_escalation:
        return None
    if policy.escalate_url:
        return OpenAICompatBackend(policy.escalate_url, policy.escalate_model or "default", policy.escalate_key_env,
                                   policy.cloud_max_tokens)
    return CloudBackend(policy.cloud_model, policy.cloud_max_tokens)


MODEL_ID = "ooa-assistant"
# Model connection checks send one of these and expect a one-word answer (the platform's connection wizard Test
# button and Command Center's model probe). They are answered at once, without running the model.
PROBES = {"reply with the single word ok", "answer with the single word ok"}
NO_TOOLS = "ooa-assistant writes flows only; it does not support tools"


def is_probe(text):
    return " ".join(str(text).lower().split()).rstrip(".!") in PROBES


def openai_error(message, type_="invalid_request_error", code=None):
    err = {"message": message, "type": type_}
    if code:
        err["code"] = code
    return {"error": err}


def _approx_tokens(text):
    return (len(text) + 3) // 4


def quick_reply(body, count_tokens=None, version=None):
    """The answers to /v1/chat/completions that need no model: (http_status, response), or None when the model must
    run. A malformed request or one with tools gets 400; a connection probe gets "OK"."""
    msgs = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(msgs, list) or not [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]:
        return 400, openai_error("send at least one user message")
    if body.get("tools"):
        return 400, openai_error(NO_TOOLS)
    if is_probe(parse_messages([m for m in msgs if isinstance(m, dict)])[4]):
        return 200, _completion(msgs, "OK", {"probe": True}, count_tokens, version)
    return None


def chat_completion(router, body, count_tokens=None, version=None, deadline=None):
    """OpenAI-style /v1/chat/completions on the router: (http_status, response). Used by this server and by the
    PC assistant (assistant/app.py), so One Agent Studio's Assistant can use the model as an OpenAI-compatible
    One Model Connection.
    count_tokens(text) -> int counts usage with the model's own tokenizer (default: characters / 4).
    version: dict added to the response's extra "ooa" field (pack / model version).
    deadline: time.time() after which no further retry attempt starts (see Router.run)."""
    quick = quick_reply(body, count_tokens, version)
    if quick is not None:
        return quick
    msgs = [m for m in body["messages"] if isinstance(m, dict)]
    request, system, assistant, existing, full_user = parse_messages(msgs)
    res = router.run(request, system, assistant, existing, full_user, deadline=deadline)
    if res["verified"]:
        content = res["flow"]
    elif res.get("question"):     # the model asks before it guesses: Studio shows notes
        content = (json.dumps({"variables": [], "steps": [], "notes": res["question"], "warnings": []})
                   if assistant else res["question"])
    elif assistant:   # Studio reads steps/notes: an empty answer with the honest reason is shown to the user
        content = json.dumps({"variables": [], "steps": [], "notes": "No flow was returned: " + res["reason"] + ". " + " ".join(res.get("problems", [])[:2])})
    else:
        content = json.dumps({"error": res["reason"], "problems": res.get("problems", [])})
    return 200, _completion(msgs, content, {k: v for k, v in res.items() if k != "flow"}, count_tokens, version)


def _completion(msgs, content, ooa, count_tokens, version):
    texts = [m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content", ""))
             for m in msgs if isinstance(m, dict)]
    try:
        count = count_tokens or _approx_tokens
        pt, ct = sum(count(t) for t in texts), count(content)
    except Exception:       # a tokenizer problem must not fail the request
        pt, ct = sum(_approx_tokens(t) for t in texts), _approx_tokens(content)
    if version:
        ooa = {**ooa, "version": version}
    # finish_reason is "stop" whether or not a flow was produced: the answer is complete either way (a missing flow
    # is explained in the content), and "length" would tell the client to retry with more tokens, which does not help.
    return {"id": "ooa-" + str(int(time.time() * 1000)), "object": "chat.completion",
            "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
            "ooa": ooa}


def make_handler(router):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass

        def do_GET(self):
            self._send(200, {"ok": True, "policy": asdict(router.policy)}) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            except Exception:
                return self._send(400, {"error": "invalid JSON body"})
            if self.path == "/v1/chat/completions":
                return self._send(*chat_completion(router, body))
            if self.path == "/v1/feedback":
                out = router.feedback(body)
                return self._send(400 if out.get("error") else 200, out)
            self._send(404, {"error": "not found"})
    return H


def serve(router, port=8080, host="127.0.0.1"):
    srv = ThreadingHTTPServer((host, port), make_handler(router))
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--local", default=None, help="'nanochat' to load the trained model (needs GPU/CPU + nanochat)")
    ap.add_argument("--source", default="sft")
    ap.add_argument("--examples", default=None, help="example index from render_dataset.py (default: examples.jsonl "
                                                       "next to the catalog when it exists)")
    ap.add_argument("--prompt-format", default=None, choices=["auto", "plain", "context"],
                    help="what the local model is given (default: the policy's prompt_format)")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    pol = Policy.load(a.policy)
    local = None
    if a.local == "nanochat":
        from backends import NanochatBackend
        local = NanochatBackend(a.source)
    cloud = make_cloud(pol)
    examples = a.examples or str(Path(a.catalog).resolve().parent / "examples.jsonl")
    r = Router(a.catalog, a.platform, pol, local, cloud, examples, a.prompt_format)
    print(f"OOA assistant on http://127.0.0.1:{a.port}  | local model: {bool(local)} ({r.format()} prompts) | "
          f"escalation: {pol.allow_escalation}")
    serve(r, a.port).serve_forever()
