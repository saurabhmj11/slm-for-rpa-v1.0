"""OOA assistant server: one OpenAI-compatible endpoint, a router, a verifier, and client data-policy switches.

   request -> [mask PII for cloud] -> router
        ask-first gate: points only at something the user knows ("rename them properly")? -> a question back
        simple?  -> LOCAL model (our own): context (actions, similar examples, variables) -> plan + flow
                    -> question? return it -> repair -> verify + plan coverage -> ok? return
                    -> self-correction (previous flow + problems) -> long plan: build it in parts and merge
        else / failed -> escalation model if the client policy allows (Anthropic API, or any OpenAI-compatible
                    endpoint: escalate_url) -> repair -> verify (one retry with the errors)
        verified -> safety warnings for destructive steps (notes, "warnings", the ooa field)
        nothing verified -> honest error (never returns an unchecked flow as if it were fine)
   POST /v1/chat/completions   OpenAI-style; the last user message is the request; the answer is the flow JSON.
                               Mode (MODES: fast / smart / ultra, or auto): model name ooa-flow-assistant[-fast|
                               -smart|-ultra], a "mode" field or the X-OOA-Mode header; else the server's --mode
   GET  /v1/models             ooa-assistant (the server's default mode) and one name per mode
   POST /v1/feedback           {"request","flow","verdict": inserted|discarded|edited,"edited_flow"} (or the older
                               {"prompt","flow","accepted"}) -> feedback.jsonl (only if the policy allows)
   GET  /health

Run:  python ooa_server.py --catalog catalog.jsonl --platform PLATFORM_REPO --policy policy.json [--local nanochat]
                           [--examples examples.jsonl] [--prompt-format auto|plain|context] [--port 8080]
                           [--mode auto|fast|smart|ultra|none]
Local backend: any object with generate(prompt, attempt) -> str (see backends.py). Escalation: Anthropic Messages API
(env ANTHROPIC_API_KEY) or an OpenAI-compatible endpoint (policy escalate_url / escalate_model / escalate_key_env).
Default policy is LOCAL-ONLY and stores nothing.
"""
import argparse, copy, inspect, json, os, re, sys, time, threading
from dataclasses import dataclass, asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))
import ooa_reward as R
import ooa_format as F
import retrieval
import repair as RP
import safety
import step_builder as SB


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
    # the agent team (Router._team), used when a mode is chosen (MODES); empty mode: the layers above, as before
    mode: str = ""                       # fast | smart | ultra (set by Policy.for_mode; "" = no mode)
    use_parts: bool = True               # a long plan may be built in parts at all
    parallel_parts: bool = True          # build the parts at once in one batched generation (backend generate_batch)
    fixer_rounds: int = 1                # rounds that rebuild only the failed parts, each seeing its own problems
    understander: bool = False           # ask for "Clear request: ..." first (models trained with rewrite rows only)
    critic: bool = False                 # coverage check against the request's clauses; a missing piece is built once
    best_of: int = 1                     # answers sampled at once per attempt; the first verified one is kept
    part_best_of: int = 1                # the same per part
    sample_temperature: float = 0.25     # temperature of the extra samples (best_of > 1)
    stop_after: float = 0.0              # seconds after which no new attempt starts (0: only the caller's deadline)
    time_limit: float = 0.0              # seconds after which a generation still running is stopped (0: none)
    ask_first: bool = True               # ask-first gate: a request that points at something only the user knows
                                         # ("rename them properly", "the usual folder") gets a question before any
                                         # model runs (ooa_format.ask_first); off in fast mode
    step_builder: str = "long"           # the step-by-step builder (serving/step_builder.py) in smart / ultra (fast
                                         # mode sets "off"; without a mode it never runs): "off" | "long" (a long or
                                         # multi-app request, step_builder.wants) | "always" (any request of 2+ steps);
                                         # its flow is kept when it covers as much of the request as the single
                                         # answer or more (step_builder.prefer), or the single answer did not verify
    step_best_of: int = 3                # samples per step (one batch)
    step_time: float = 60.0              # seconds the step builder may take: its generations are cut then and no
                                         # second wave starts after STEP_WAVE_SHARE of it, so the single answer
                                         # always has the rest of the mode's time (0: only the mode's own limits)

    @classmethod
    def load(cls, path=None):
        d = json.load(open(path)) if path else {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def for_mode(self, mode):
        """These settings with a mode's preset on top (MODES); the client's data settings (escalation, masking,
        logs, context) stay as they are. An unknown mode raises ValueError."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}: one of {', '.join(MODE_NAMES)}")
        return replace(self, mode=mode, **MODES[mode])


# The assistant's modes: Policy presets on top of the client's policy (the only place they are defined). Studio and
# OpenAI-style clients pick one by model name (MODEL_NAMES), a "mode" field or the X-OOA-Mode header; auto (the server
# default) picks fast, smart or ultra per request (choose_mode).
#   fast   planner + one builder + checker: one attempt, no parts, no self-correction (simple requests, 1-5 s on CPU)
#   smart  the agent team without the critic: planner, parallel builders for long plans, checker, fixer (2 rounds),
#          reporter; whole answers get 2 self-correction retries; a long or multi-app request is also built step by
#          step (serving/step_builder.py) and the better of the two flows is kept
#   ultra  the whole team: understander (restatement first, when the model was trained for it), best-of-4 answers,
#          parts from shorter plans, parallel builders (2 samples each), fixer (3 rounds), step builder (as smart),
#          critic, reporter; stops starting attempts after 120 s and cuts a generation still running at 150 s
#          (Studio waits 180 s)
MODES = {
    "fast": dict(local_retries=0, use_parts=False, parts_first=False, fixer_rounds=0, understander=False, critic=False,
                 best_of=1, part_best_of=1, stop_after=0.0, time_limit=60.0, ask_first=False, step_builder="off"),
    "smart": dict(local_retries=2, use_parts=True, parts_first=True, parallel_parts=True, fixer_rounds=2,
                  understander=False, critic=False, best_of=1, part_best_of=1, stop_after=100.0, time_limit=140.0,
                  ask_first=True),
    "ultra": dict(local_retries=3, use_parts=True, parts_first=True, parallel_parts=True, fixer_rounds=3, ask_first=True,
                  understander=True, critic=True, best_of=4, part_best_of=2, sample_temperature=0.25,
                  decompose_lines=4, long_plan_lines=3, part_lines=4, stop_after=120.0, time_limit=150.0),
}
# A caller's deadline (Router.run deadline=, e.g. the PC app's: arrival + its time budget, queue wait included) also
# caps time_limit: a running generation is cut at most HARD_GRACE seconds after it. The cut needs a backend with
# generate_batch(deadline=) (NanochatLocal); other backends (generate() only, models without plans) are only stopped
# from starting new attempts at the deadline.
HARD_GRACE = 30.0
STEP_WAVE_SHARE = 0.6                # the step builder's second wave starts only in this share of Policy.step_time
MODE_NAMES = ("auto", "fast", "smart", "ultra")
MODE_LABELS = {"auto": "Auto", "fast": "Fast", "smart": "Smart", "ultra": "Ultra"}
AUTO_LONG_CLAUSES, AUTO_LONG_WORDS = 5, 60        # a long process
AUTO_FAST_CLAUSES, AUTO_FAST_WORDS = 2, 25        # short and simple


def choose_mode(request, rewrite=False):
    """Auto's choice for one request -> (mode, reason in plain words). From ooa_format.request_features:
      1. ultra  a long process: 5+ clauses or more than 60 words
      2. vague  no file, folder, app or data named at all: ultra when the model can restate a request
                (rewrite rows in its training: the understander then runs), else smart
      3. fast   2 clauses or fewer, 25 words or fewer, no loop words (each, every, repeat ...) and no condition words
                (if, when, otherwise, only ...), and not work across apps that the step builder takes
                (step_builder.wants: 2+ steps in 2+ named apps)
      4. smart  everything else (a loop, a condition, 3-4 clauses, a longer sentence, steps in several apps)
    Smart and ultra build a long or multi-app request step by step too (Policy.step_builder)."""
    f = F.request_features(request)
    if f["clauses"] >= AUTO_LONG_CLAUSES or f["words"] > AUTO_LONG_WORDS:
        return "ultra", f"a long process of {f['clauses']} parts"
    if not f["has_thing"]:
        return ("ultra" if rewrite else "smart"), "no file, app or data is named"
    if (f["clauses"] <= AUTO_FAST_CLAUSES and f["words"] <= AUTO_FAST_WORDS and not f["loop"]
            and not f["condition"]):
        if SB.wants(request):
            return "smart", "steps in several apps"
        return "fast", "a short, simple request"
    why = "a loop" if f["loop"] else "a condition" if f["condition"] else f"{f['clauses']} parts"
    return "smart", why


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
UNDERSTOOD = "I understood: "     # + the model's restatement of a beginner's request ("Clear request: ..."), in notes
ASKED = "the model asked a question instead of guessing: the request does not say enough to automate"
ASKED_FIRST = ("the assistant asked a question before building: the request points at something only you know "
               "(which files, folder, app or person)")


class Router:
    """Layers, in order: ask-first gate (a request that only points at something the user knows gets a question,
    ooa_format.ask_first; Policy.ask_first, off in fast mode) -> retrieval context (actions, similar examples, the
    open agent's variables; app-aware, ooa_format.context_actions) -> plan-first answer
    -> clarify (a question comes back as the answer) -> repair + verify + plan coverage -> self-correction (the next
    attempt sees its previous flow and the problems) -> decomposition (a long plan is built in parts and merged) ->
    escalation (a bigger model through the Anthropic API or any OpenAI-compatible endpoint, when the policy allows)
    -> safety warnings on the verified flow. A model trained before context and plans (prompt_format "plain") gets
    the request only and answers with flow JSON; that path works as before."""

    def __init__(self, catalog, platform, policy, local=None, cloud=None, examples=None, prompt_format=None, mode=None):
        """mode: the mode a request runs in when it names none (auto | fast | smart | ultra; None: the policy's own
        settings, without the agent team, as before modes existed)."""
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
        if mode is not None and mode not in MODE_NAMES:
            raise ValueError(f"unknown mode {mode!r}: one of {', '.join(MODE_NAMES)}")
        self.default_mode = mode
        self.trained_rewrite = self._trained_rewrite(examples) if self.examples is not None else False
        self.trained_think = self._trained_think(examples) if self.examples is not None else False
        self._agents = None          # per request (a copy of the router runs each moded request): the agents that ran

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

    def prompt(self, request, existing=None, previous=None, problems=None, required=(), similar=True):
        """The user turn for the local model: the request + retrieval context (+ previous flow and problems).
        similar=False: no similar examples (the step builder's freer samples: a short step request copies a similar
        example's whole outline too readily)."""
        examples = self.examples.search(request, F.MAX_EXAMPLES) if self.examples is not None and similar else []
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

    @staticmethod
    def _trained_rewrite(examples):
        """True when the model pack's training data had restatement rows ("Clear request: ...", render_report.json
        totals.rewrite beside the example index)."""
        f = Path(examples).resolve().parent / "render_report.json"
        try:
            n = json.loads(f.read_text(encoding="utf-8"))["totals"].get("rewrite", 0)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        return isinstance(n, int) and n > 0

    @staticmethod
    def _trained_think(examples):
        """True when the model pack's training data had thinking answers (render_report.json beside the example index:
        settings.think names files, or totals.think > 0)."""
        f = Path(examples).resolve().parent / "render_report.json"
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        st, tot = r.get("settings") if isinstance(r, dict) else None, r.get("totals") if isinstance(r, dict) else None
        n = tot.get("think", 0) if isinstance(tot, dict) else 0
        return bool(isinstance(st, dict) and st.get("think")) or (isinstance(n, int) and n > 0)

    def supports_think(self):
        """Whether the local model writes thinking answers (Understand / Facts / Missing / Steps / Check before the
        flow): the backend says so (info["think"]) or its training data had them. Such a model's parts start with
        ooa_format.think_prefix instead of "Plan:", and it needs no "Clear request:" understander (it says what it
        understood by itself)."""
        info = getattr(self.local, "info", None)
        if isinstance(info, dict) and isinstance(info.get("think"), bool):
            return info["think"]
        return bool(self.trained_think)

    def part_prefix(self, request, chunk):
        """The given start of a part's answer: its plan lines (and "Flow:"), as a thinking block for a model that
        thinks (ooa_format.think_prefix), else "Plan: ..." (ooa_format.answer_prefix)."""
        lines = F.chunk_plan_lines(chunk)
        return F.think_prefix(request, lines) if self.supports_think() else F.answer_prefix(lines)

    def supports_rewrite(self):
        """Whether the local model writes "Clear request: ..." when asked: the backend says so (info["rewrite"]) or
        its training data had rewrite rows. Older models skip the understander."""
        info = getattr(self.local, "info", None)
        if isinstance(info, dict) and isinstance(info.get("rewrite"), bool):
            return info["rewrite"]
        return bool(self.trained_rewrite)

    # -- one answer: parse -> repair -> verify -------------------------------------------------------
    def verify(self, text, existing=None, plan=None, request=None, facts=None):
        """Flow text -> (flow dict, repair notes) when it passes, else (None, problems). plan: the answer's plan
        lines; an action the plan names that no step uses is a problem too. A compact flow (a model trained on
        compact answers: no variables list, steps without ids) is first rebuilt into a full one (ooa_format.expand_flow:
        variables declared from what the steps make and read, step ids, descriptions from the plan, name from the
        request); a full flow is checked as written. facts: a thinking answer's Facts [(key, value)]; a fact no step
        uses is a problem too (ooa_format.fact_coverage: the flow left a detail of the request out)."""
        if text is None:
            return None, ["The answer has a plan but no flow after 'Flow:'."]
        text = extract_json(text)
        try:
            obj = json.loads(text)
        except Exception:
            return None, R.problems(text)
        if not isinstance(obj, dict):
            return None, ["The answer is not a JSON object."]
        if F.is_compact(obj):
            obj = F.expand_flow(obj, self.cat, request, plan, existing or ())
        # Studio's assistant format has no kind/id/name: wrap it so the same verifier applies
        flow = obj if "kind" in obj else {"kind": "TaskBot", "id": "assistant_check", "name": "assistant check",
                                          "variables": list(obj.get("variables", [])), "steps": obj.get("steps", [])}
        names = {v.get("name") for v in flow.get("variables", []) if isinstance(v, dict)}
        extra = [v for v in (existing or []) if isinstance(v, dict) and v.get("name") not in names]
        if extra:
            flow = {**flow, "variables": list(flow.get("variables", [])) + extra}   # variables already in the user's open agent
        flow, notes = RP.repair(flow, self.cat)
        problems = R.problems(json.dumps(flow, separators=(",", ":"), ensure_ascii=False))
        problems += [f"The plan names {a} but no step of the flow uses it." for a in F.plan_coverage(plan, flow, self.cat)]
        problems += [f"The facts say {k} = {v} but no step of the flow uses it." for k, v in F.fact_coverage(facts, flow)]
        if problems:
            return None, problems
        # the platform writes variables $name$ (Studio saves them so, and only $name$ is replaced inside text)
        return F.to_dollar_name_dollar(flow, names | {v.get("name") for v in extra}), notes

    def answer_text(self, flow, existing=None, assistant=False, warnings=(), understood=None, report=None):
        """The verified flow as the client reads it: Studio's {variables, steps, notes, warnings}, or the flow JSON.
        understood: the model's restatement of the request; Studio shows it first in notes ("I understood: ...").
        report: the reporter's summary (modes), last in notes."""
        if assistant:
            have = {v.get("name") for v in (existing or []) if isinstance(v, dict)}
            said = [UNDERSTOOD + understood.rstrip(".") + "."] if understood else []
            notes = " ".join(said + [STUDIO_NOTE] + list(warnings) + ([report] if report else []))
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

    def _previous(self, text, request=None, plan=None, existing=None):
        """The previous answer's flow for a self-correction prompt (dict), or None when it was not JSON. A compact
        flow is shown expanded, as the full flows of the self-correction training rows are."""
        if not text:
            return None
        try:
            obj = json.loads(extract_json(text))
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        if F.is_compact(obj):
            try:
                obj = F.expand_flow(obj, self.cat, request, plan, existing or ())
            except Exception:
                pass
        return F.to_dollar_name_dollar(obj)

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
        clear = None                                   # the latest "Clear request: ..." the model wrote
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
            clear = ans.get("clear") or clear
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
                    return {"flow": flow, "notes": info, "plan": plan, "parts": True, "clear": clear}
                errors = info
                continue                                     # whole answers next
            flow, info = self.verify(ans["flow_text"] if ans["kind"] == "flow" else raw, existing, ans["plan"], request,
                                     ans.get("facts"))
            tried.append({"source": "local", "ok": flow is not None})
            if flow is None and ans.get("missing") and attempt == 0:
                # a thinking answer that says a detail is missing and whose flow does not verify: ask for it
                tried[-1]["kind"] = "question"
                return {"question": F.missing_question(ans["missing"]), "plan": [], "clear": clear}
            looped = bool(stop and stop.startswith("loop"))
            if stop:
                tried[-1]["stop"] = stop
            if ans["plan"] and not (looped and ans["flow_text"] is None):     # a plan that loops is not a plan
                plan = ans["plan"]
                tried[-1]["plan_lines"] = len(plan)
            if flow is not None:
                return {"flow": flow, "notes": info, "plan": ans["plan"], "clear": clear}
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
                prev = None if looped else self._previous(ans["flow_text"], request, ans["plan"], existing)
                probs = None if looped else info
            if plan and can_escalate and len(plan) > p.escalate_plan_lines:
                return {"errors": errors, "plan": plan}      # a long process: the bigger model takes it
            if not decomposed and len(plan) > p.decompose_lines and len(self.parts_of(plan)) >= 2:
                break                                        # a long plan: build it in parts instead of retrying
        if plan and not decomposed and len(self.parts_of(plan)) >= 2 and not late():
            flow, info = self._decompose(request, plan, existing, tried, late)
            if flow is not None:
                return {"flow": flow, "notes": info, "plan": plan, "parts": True, "clear": clear}
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
            prefix = self.part_prefix(creq, chunk) if self.policy.part_plan_given else None
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
                part, info = self.verify(text, known, ans["plan"], creq)
                tried.append({"source": "local", "part": n, "of": len(chunks), "ok": part is not None})
                if ch["inside"]:
                    tried[-1]["inside"] = int(ch["inside"])
                if stop and stop != "end":
                    tried[-1]["stop"] = stop
                if part is not None:
                    break
                looped = bool(stop and stop.startswith("loop"))
                prev, probs = (None, None) if looped else (self._previous(text, creq, ans["plan"], known), info)
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

    # -- the agent team (modes) ----------------------------------------------------------------------------
    # One model plays every role with a different prompt:
    #   understander  "Clear request: ..." first (ultra, models trained with rewrite rows); a question ends the request
    #   planner       the plan; stops at "Flow:" when the plan is long enough to build in parts (parts_rule), else it
    #                 goes on and writes the flow itself (the single builder)
    #   builders      every part of a long plan at once, in ONE batched generation (backend generate_batch); a part
    #                 inside a loop / try / if waits for the part that writes it (ooa_format.part_dependencies)
    #   assembler     links variables the parts named differently (ooa_format.link_part_variables), merges them
    #   checker       repair + verify per answer, per part and for the merged flow
    #   fixer         rebuilds only the failed parts (batched too), each with its own problems; whole answers get
    #                 self-correction retries
    #   critic        request clauses no step covers (ooa_format.uncovered_clauses) are built once as new parts
    #   reporter      a plain-English summary in the notes: mode, what the flow does, inputs to fill in, agents, time
    def _agent(self, name, t0, **info):
        if self._agents is not None:
            self._agents.append({"agent": name, "seconds": round(time.time() - t0, 2), **info})

    def can_batch(self):
        return callable(getattr(self.local, "generate_batch", None))

    def generate_many(self, items, hard=None):
        """Answers for several prompts: one batched generation when the backend has generate_batch, else one call
        after another (backends without it, e.g. an older serving/backends.py: same results, no speed-up). items as
        NanochatLocal.generate_batch takes them. hard: time.time() after which a running generation is cut.
        Returns [(text, stop)]."""
        if self.can_batch() and (len(items) > 1 or hard is not None):
            fn = self.local.generate_batch
            try:
                takes = "deadline" in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                takes = False
            texts = fn(items, deadline=hard) if takes else fn(items)
            stats = getattr(self.local, "last_batch_stats", None)
            stats = stats if isinstance(stats, list) and len(stats) == len(texts) else [{}] * len(texts)
            return [(t, st.get("stop") if isinstance(st, dict) and isinstance(st.get("stop"), str) else None)
                    for t, st in zip(texts, stats)]
        return [self.generate(it["prompt"], it.get("attempt", 0), max_tokens=it.get("max_tokens"),
                              on_plan=it.get("on_plan"), prefix=it.get("prefix")) for it in items]

    def _samples(self, base, n, attempt):
        """n copies of a generation item: the first as given, the others at the policy's sample temperature with
        their own seeds (best-of-N). A backend without generate_batch gets one."""
        n = max(1, n) if self.can_batch() else 1
        out = [dict(base)]
        for k in range(1, n):
            out.append({**base, "temperature": self.policy.sample_temperature, "seed": 7000 + 101 * attempt + k})
        return out

    def _team(self, request, existing, tried, late, can_escalate, hard):
        """The agent team's attempts (see above). Same result shape as _local. A model that gets plain prompts (older
        than context and plans) runs the old layers."""
        if self.format() != "context":
            t = time.time()
            out = self._local(request, existing, tried, late, can_escalate)
            self._agent("builder (flow only, a model without plans)", t, attempts=len(tried))
            return out
        if self.wants_steps(request):
            out = self._steps_then_single(request, existing, tried, late, can_escalate, hard)
        else:
            out = self._team_answer(request, existing, tried, late, can_escalate, hard)
        if out.get("flow") is not None and self.policy.critic and not late():
            out = self._critic(request, out, existing, tried, late, hard)
        return out

    def _team_answer(self, request, existing, tried, late, can_escalate, hard):
        p = self.policy
        rewrite = p.understander and self.supports_rewrite() and not self.supports_think()
        rule = self.parts_rule(request, can_escalate) if p.use_parts and p.parts_first else None
        prev = probs = None
        plan, errors, decomposed, clear = [], [], False, None
        for attempt in range(1 + max(0, p.local_retries)):
            if late():
                return {"errors": list(errors) + [OUT_OF_TIME], "plan": plan}
            prompt = self.prompt(request, existing, prev, probs)
            base = {"prompt": prompt, "attempt": attempt, "max_tokens": p.max_tokens,
                    "on_plan": rule if not decomposed else None}
            items = self._samples(base, p.best_of, attempt)
            if rewrite and prev is None and clear is None and len(items) > 1:
                # the understander: every sample but the last starts with "Clear request: "; the last is free, so the
                # model can still ask a question (a restatement start never turns into one)
                for it in items[:-1]:
                    it["prefix"] = F.CLEAR_HEAD      # no trailing space: in training the space belongs to the next
                    #                                  word's token (" Read"); a lone " " token garbled the restatement
            t = time.time()
            try:
                outs = self.generate_many(items, hard)
            except Exception as ex:
                return {"errors": [f"local model error: {ex}"], "plan": plan}
            first = attempt == 0
            understander = any(it.get("prefix") for it in items)
            self._agent(("understander + planner" if understander else "planner") if first else "fixer", t,
                        samples=len(items), attempt=attempt)
            plan_stop, fail, asked = None, None, None
            t = time.time()
            for (raw, stop), it in zip(outs, items):
                ans = F.parse_answer(raw)
                clear = ans.get("clear") or clear
                if ans["kind"] == "question":
                    # a question counts only from a sample without a restatement start (the free one), and only when
                    # no sample of the batch verifies and none stops after a plan (decided below): best-of-N never
                    # trades a verified flow for a question. A fixer retry never asks once a plan exists.
                    if asked is None and not it.get("prefix") and not (attempt and plan):
                        asked = ans
                    continue
                if stop and stop.startswith("plan") and ans["plan"]:
                    tried.append({"source": "local", "ok": True, "plan_lines": len(ans["plan"]), "plan_only": True})
                    plan_stop = plan_stop or ans["plan"]
                    continue
                flow, info = self.verify(ans["flow_text"] if ans["kind"] == "flow" else raw, existing, ans["plan"],
                                         request, ans.get("facts"))
                tried.append({"source": "local", "ok": flow is not None})
                if stop:
                    tried[-1]["stop"] = stop
                if flow is not None:
                    self._agent("checker", t, ok=True)
                    return {"flow": flow, "notes": info, "plan": ans["plan"], "clear": clear}
                tried[-1]["preview"] = (raw or "")[:300]
                if ans.get("missing") and asked is None and not it.get("prefix") and not (attempt and plan):
                    # a thinking answer that says a detail is missing and whose flow does not verify: ask for it
                    asked = {"question": F.missing_question(ans["missing"]), "clear": ans.get("clear")}
                if fail is None:
                    fail = (raw, stop, ans, info)
            self._agent("checker", t, ok=False)
            if asked is not None and plan_stop is None:
                tried.append({"source": "local", "ok": True, "kind": "question"})
                self._agent("understander" if understander else "planner", t, asked=True)
                return {"question": asked["question"], "plan": [], "clear": asked.get("clear")}
            if plan_stop is not None:
                plan = plan_stop
                if can_escalate and len(plan) > p.escalate_plan_lines:
                    return {"errors": ["a long process: the plan has %d lines" % len(plan)], "plan": plan}
                decomposed = True
                flow, info = self._build_parts(request, plan, existing, tried, late, hard)
                if flow is not None:
                    return {"flow": flow, "notes": info, "plan": plan, "parts": True, "clear": clear}
                errors = info
                continue
            if fail is None:
                continue
            raw, stop, ans, info = fail
            looped = bool(stop and stop.startswith("loop"))
            if ans["plan"] and not (looped and ans["flow_text"] is None):
                plan = ans["plan"]
            errors = ([LOOPED] if looped else []) + list(info)
            if ans["kind"] == "reply":
                return {"errors": [NOT_A_TASK], "plan": plan}
            prev = None if looped else self._previous(ans["flow_text"], request, ans["plan"], existing)
            probs = None if looped else info
            if plan and can_escalate and len(plan) > p.escalate_plan_lines:
                return {"errors": errors, "plan": plan}
            if p.use_parts and not decomposed and len(plan) > p.decompose_lines and len(self.parts_of(plan)) >= 2:
                break
        if p.use_parts and plan and not decomposed and len(self.parts_of(plan)) >= 2 and not late():
            flow, info = self._build_parts(request, plan, existing, tried, late, hard)
            if flow is not None:
                return {"flow": flow, "notes": info, "plan": plan, "parts": True, "clear": clear}
            errors = list(errors) + info
        elif plan and late():
            errors = list(errors) + [OUT_OF_TIME]
        return {"errors": errors, "plan": plan}

    def _part_spec(self, i, chunks, built, base, deps):
        """(request, known variables, required actions) for part i, or None when the container it goes into is not
        there. Known: the open agent's variables, those of the parts before it built so far, and the items of the
        loops it is inside."""
        ch = chunks[i]
        creq, need = F.chunk_request(ch["chunk"]), F.chunk_actions(ch["chunk"])
        known = list(base)
        for j in range(i):
            if built[j] is not None:
                names = {v.get("name") for v in known}
                known += [v for v in built[j].get("variables", []) if isinstance(v, dict) and v.get("name") not in names]
        if ch["inside"]:
            upto = deps[i] if deps[i] is not None else i - 1
            prior = [(built[j], chunks[j]["inside"]) for j in range(upto + 1) if built[j] is not None]
            so_far = F.merge_flows([b for b, _ in prior], inside=[x for _, x in prior]) or {}
            path = F.container_path(so_far.get("steps") or [], ch["inside"])
            if not path:
                return None
            items = [b["args"]["as"] for b in path if b.get("action") == "core/loop"
                     and isinstance((b.get("args") or {}).get("as"), str)]
            for item in items:
                if item not in {v.get("name") for v in known}:
                    known.append({"name": item, "type": "Any", "direction": "local"})
            if items:
                creq = f"With the current {items[-1]}: {creq[:1].lower() + creq[1:]}"
        return creq, known, need

    def _build_parts(self, request, plan, existing, tried, late, hard):
        """Builders + assembler + fixer for a long plan: every part at once in one batched generation (parts inside a
        container wait for the part that writes it), the failed ones rebuilt with their problems for fixer_rounds
        rounds, then variables linked and the parts merged; the merged flow must pass the checker and cover the plan.
        Without generate_batch (or parallel_parts off) the parts are built one after another (_decompose).
        Returns (flow, notes) or (None, problems)."""
        p = self.policy
        chunks = self.parts_of(plan)
        n = len(chunks)
        if n < 2:
            return None, []
        if not (p.parallel_parts and self.can_batch()):
            t = time.time()
            res = self._decompose(request, plan, existing, tried, late)
            self._agent("builders (one after another)", t, parts=n)
            return res
        deps = F.part_dependencies(chunks)
        base = [dict(v) for v in existing or [] if isinstance(v, dict)]
        built, last, problems, reqs = [None] * n, [(None, None)] * n, [[] for _ in range(n)], [None] * n
        for rnd in range(1 + max(0, p.fixer_rounds)):
            todo = [i for i in range(n) if built[i] is None]
            if not todo:
                break
            while todo:
                if late():
                    return None, [OUT_OF_TIME]
                ready = [i for i in todo if deps[i] is None or built[deps[i]] is not None]
                if not ready:
                    break
                items, owner = [], []
                for i in ready:
                    spec = self._part_spec(i, chunks, built, base, deps)
                    if spec is None:
                        problems[i] = ["it goes inside a loop, try or if, but the parts before it did not end with one"]
                        continue
                    creq, known, need = spec
                    reqs[i] = (creq, known)
                    prev, probs = last[i]
                    prompt = self.prompt(creq, known, prev, probs, required=need)
                    prefix = self.part_prefix(creq, chunks[i]["chunk"]) if p.part_plan_given else None
                    for it in self._samples({"prompt": prompt, "attempt": rnd, "max_tokens": p.part_max_tokens,
                                             "prefix": prefix}, p.part_best_of, rnd * 31 + i):
                        items.append(it)
                        owner.append(i)
                todo = [i for i in todo if i not in ready]
                if not items:
                    continue
                t = time.time()
                try:
                    outs = self.generate_many(items, hard)
                except Exception as ex:
                    return None, [f"local model error: {ex}"]
                self._agent("builders" if rnd == 0 else "fixer", t, parts=len(set(owner)), batch=len(items), round=rnd)
                t = time.time()
                for i, (raw, stop) in zip(owner, outs):
                    if built[i] is not None:
                        continue
                    creq, known = reqs[i]
                    ans = F.parse_answer(raw)
                    text = ans["flow_text"] if ans["kind"] == "flow" else raw
                    part, info = self.verify(text, known, ans["plan"], creq)
                    tried.append({"source": "local", "part": i + 1, "of": n, "ok": part is not None, "round": rnd})
                    if chunks[i]["inside"]:
                        tried[-1]["inside"] = int(chunks[i]["inside"])
                    if stop and stop != "end":
                        tried[-1]["stop"] = stop
                    if part is not None:
                        built[i] = part
                        continue
                    problems[i] = info
                    looped = bool(stop and stop.startswith("loop"))
                    last[i] = (None, None) if looped else (self._previous(text, creq, ans["plan"], known), info)
                self._agent("checker", t, parts=len(set(owner)))
        missing = [i for i in range(n) if built[i] is None]
        if missing:
            i = missing[0]
            creq = reqs[i][0] if reqs[i] else F.chunk_request(chunks[i]["chunk"])
            return None, [f"Part {i + 1} of {n} ('{creq[:80]}') did not pass the checker: " + "; ".join(problems[i][:2])]
        t = time.time()
        parts, links = F.link_part_variables(built, base)
        merged = F.merge_flows(parts, request, [ch["inside"] for ch in chunks])
        flow, info = self.verify(F.compact(merged), existing, plan)
        self._agent("assembler", t, parts=n, linked=len(links), ok=flow is not None)
        tried.append({"source": "local", "merged_parts": n, "ok": flow is not None})
        if flow is None:
            return None, ["The flow built in parts did not pass the checker: " + "; ".join(info[:2])]
        return flow, info + [f"built in {n} parts from the plan, at once"] + [
            f"part {k}: '{a}' is '{b}' of an earlier part" for k, a, b in links]

    def wants_steps(self, request):
        """Whether the step-by-step builder runs for this request (Policy.step_builder): "always" for any request of
        2+ steps, "long" for a long or multi-app one (step_builder.wants), never with "off" or a model without plans."""
        mode = self.policy.step_builder
        if mode not in ("long", "always") or self.format() != "context":
            return False
        return SB.wants(request) if mode == "long" else len(SB.split_steps(request)) >= 2

    def _steps_then_single(self, request, existing, tried, late, can_escalate, hard):
        """A long or multi-app request (wants_steps): the step builder first (serving/step_builder.py: the request
        cut into steps, all built at once as short checked pieces, joined and checked), then the single answer (the
        planner and its builders, as for any request) in the time left, and the better of the two flows is kept
        (step_builder.prefer: the single answer when the step path failed or the single answer covers more of the
        request). A step-built flow that leaves no clause of the request uncovered is kept without the single answer
        (it cannot be beaten on coverage; this also keeps such requests inside the time limits). A question from the
        single answer is asked only when the step path failed."""
        steps = SB.split_steps(request)
        t = time.time()
        s_late, s_hard = late, hard
        if self.policy.step_time:
            wave_end, cut = t + STEP_WAVE_SHARE * self.policy.step_time, t + self.policy.step_time
            s_late = lambda: late() or time.time() >= wave_end
            s_hard = min(hard, cut) if hard is not None else cut
        try:
            flow, notes, info = SB.build(self, request, existing, tried, s_late, s_hard, steps=steps,
                                         temperatures=SB.STEP_TEMPERATURES[:max(1, self.policy.step_best_of)])
        except Exception as ex:                       # never lose the request to the step path
            flow, notes, info = None, [f"step builder error: {ex}"], {"steps": [s["text"] for s in steps]}
        summary = {"steps": info.get("steps") or [], "built": info.get("built", 0), "failed": info.get("failed") or [],
                   "verified": flow is not None, "kept": False, "seconds": info.get("seconds"),
                   "model_calls": info.get("model_calls", 0)}
        self._agent("step builder", t, steps=len(summary["steps"]), built=summary["built"], ok=flow is not None)
        stepped = None
        if flow is not None:
            stepped = {"flow": flow, "notes": notes, "plan": F.plan_from_flow(flow, self.cat), "clear": None,
                       "step_builder": summary}
            if SB.coverage(self, request, flow) == 0:
                summary.update(kept=True, single="skipped: the steps cover the whole request")
                return stepped
        if late():
            if stepped is not None:
                summary["kept"] = True
                return stepped
            return {"errors": list(notes) + [OUT_OF_TIME], "plan": [], "step_builder": summary}
        out = self._team_answer(request, existing, tried, late, can_escalate, hard)
        if stepped is None:
            return {**out, "step_builder": summary}
        if out.get("question") or SB.prefer(self, request, flow, out.get("flow"), steps):
            summary["kept"] = True
            return {**stepped, "clear": out.get("clear")}
        return {**out, "step_builder": summary}

    def _critic(self, request, out, existing, tried, late, hard):
        """Coverage check of a verified flow against the request (ooa_format.uncovered_clauses: a clause that names
        something to do and that no step covers). Each such clause is built once as a new part (one batch) and
        added after the steps; the result must pass the checker again, else the flow stays as it was and the notes
        say what may be missing."""
        t = time.time()
        flow = out["flow"]
        missing = F.uncovered_clauses(request, flow, self.index, plan=out.get("plan"))
        self._agent("critic", t, missing=len(missing))
        if not missing:
            return out
        known = [v for v in flow.get("variables", []) if isinstance(v, dict)]
        items = [{"prompt": self.prompt(c, known), "attempt": 0, "max_tokens": self.policy.part_max_tokens}
                 for c in missing[:3]]
        t = time.time()
        try:
            outs = self.generate_many(items, hard)
        except Exception:
            outs = []
        self._agent("builders (critic)", t, parts=len(items))
        added, names = [], []
        for c, (raw, _stop) in zip(missing, outs):
            ans = F.parse_answer(raw)
            if ans["kind"] != "flow":
                continue
            part, _ = self.verify(ans["flow_text"], known, ans["plan"], c)
            tried.append({"source": "local", "critic": True, "ok": part is not None})
            if part is not None:
                added.append(part)
                names.append(c)
        notes = list(out.get("notes") or [])
        if added:
            parts, _links = F.link_part_variables([flow] + added, existing or ())
            merged, info = self.verify(F.compact(F.merge_flows(parts, flow.get("description"))), existing, out.get("plan"))
            if merged is not None:
                return {**out, "flow": merged, "notes": notes + info + [
                    "the critic added steps for: " + "; ".join(names)], "critic_added": names}
        return {**out, "notes": notes, "critic_missing": missing}

    def run(self, request, system=None, assistant=False, existing=None, full_user=None, deadline=None, mode=None):
        """deadline: time.time() value after which no further attempt (retry or cloud call) starts.
        mode: auto | fast | smart | ultra (None: the router's default mode; without one the policy's own settings run,
        as before modes). A moded request runs on a copy of the router with the mode's policy (Policy.for_mode), so
        requests in different modes never share settings; auto picks the mode (choose_mode). The result says which
        mode ran ("mode", "mode_asked", "mode_reason") and which agents ran, with their times ("agents")."""
        mode = mode or self.default_mode
        if mode:
            if mode not in MODE_NAMES:
                raise ValueError(f"unknown mode {mode!r}: one of {', '.join(MODE_NAMES)}")
            chosen, why = choose_mode(request, self.supports_rewrite()) if mode == "auto" else (mode, "chosen")
            r = copy.copy(self)
            r.policy, r.default_mode, r._agents = self.policy.for_mode(chosen), None, []
            r._mode = {"mode": chosen, "asked": mode, "why": why}
            return r._run(request, system, assistant, existing, full_user, deadline)
        return self._run(request, system, assistant, existing, full_user, deadline)

    def _run(self, request, system=None, assistant=False, existing=None, full_user=None, deadline=None):
        t0, tried, errors, plan, failed = time.time(), [], [], [], None
        p = self.policy
        asked_by = deadline                     # the caller's own deadline (set when the request arrived)
        if p.stop_after:
            deadline = min(deadline, t0 + p.stop_after) if deadline is not None else t0 + p.stop_after
        hard = t0 + p.time_limit if p.time_limit else None
        if hard is not None and asked_by is not None:
            # the caller's deadline was set when the request arrived (time spent queued for the model counts): a
            # running generation is never cut later than HARD_GRACE seconds after it (Studio waits 180 s)
            hard = min(hard, asked_by + HARD_GRACE)
        late = lambda: deadline is not None and tried and time.time() >= deadline
        if p.ask_first:
            # ask-first gate: no model guesses the folder / files / app / person a request only points at
            t = time.time()
            gate = F.ask_first(request, has_variables=bool(existing))
            if gate:
                tried.append({"source": "ask_first", "ok": True, "kind": "question", "missing": gate["missing"]})
                self._agent("ask-first gate", t, asked=True)
                return self.done(request, None, "local", tried, [], t0, assistant, existing, reason=ASKED_FIRST,
                                 question=gate["question"])
        can_escalate = self.cloud is not None and p.allow_escalation
        simple = p.local_first or (clauses(request) <= p.max_clauses_local and len(request) <= 500)
        if not can_escalate:
            simple = True      # no cloud fallback: always let the local model try instead of failing unseen
        if self.local is not None and simple:
            if p.mode:
                out = self._team(request, existing, tried, late, can_escalate, hard)
            else:
                out = self._local(request, existing, tried, late, can_escalate)
            if out.get("flow") is not None:
                return self.done(request, out["flow"], "local", tried, out["notes"], t0, assistant, existing,
                                 plan=out["plan"], parts=out.get("parts", False), understood=out.get("clear"),
                                 critic=out)
            if out.get("question"):
                return self.done(request, None, "local", tried, [], t0, assistant, existing, reason=ASKED,
                                 question=out["question"])
            errors, plan = out["errors"], out["plan"]
            failed = {k: out[k] for k in ("step_builder",) if out.get(k)}
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
                         plan=plan, critic=failed)

    def done(self, request, flow, source, tried, info, t0, assistant=False, existing=None, reason=None, plan=None,
             question=None, parts=False, understood=None, critic=None):
        """flow: the verified flow dict or None. The result's "flow" is the text the client gets (answer_text).
        understood: the model's restatement of the request (result "understood"; Studio's notes start with it).
        In a mode, the reporter's summary (report) ends Studio's notes and is the result's "report"."""
        warnings = safety.warnings(flow, self.cat) if flow is not None else []
        mode = getattr(self, "_mode", None) if self.policy.mode else None
        report = None
        if mode:
            t = time.time()
            report = self.report(flow, existing, understood, time.time() - t0, critic or {}, question is not None)
            self._agent("reporter", t)
        text = self.answer_text(flow, existing, assistant, warnings, understood, report) if flow is not None else None
        res = {"flow": text, "source": source, "verified": flow is not None, "attempts": tried,
               "repairs" if flow is not None else "problems": info, "seconds": round(time.time() - t0, 3),
               "plan": list(plan or []), "warnings": warnings}
        if mode:
            res.update(mode=mode["mode"], mode_asked=mode["asked"], mode_reason=mode["why"], report=report,
                       agents=list(self._agents or []))
            for k in ("critic_added", "critic_missing", "step_builder"):
                if (critic or {}).get(k):
                    res[k] = critic[k]
        if parts:
            res["built_in_parts"] = True
        if understood and flow is not None:
            res["understood"] = understood
        if question:
            res["question"] = question
        if reason:
            res["reason"] = reason
        if self.policy.store_logs:
            self.log("requests.jsonl", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "request": mask(request),
                                        "source": source, "verified": res["verified"], "flow": text,
                                        "question": question, "training_ok": self.policy.allow_training_on_data,
                                        "mode": res.get("mode"), "agents": res.get("agents"),
                                        "seconds": res["seconds"]})
        return res

    def report(self, flow, existing, understood, seconds, team, asked=False):
        """The reporter: a plain-English summary for the notes. Which mode ran (and why, when auto picked it) and how
        long it took; what the flow does (the model's restatement, else the steps' own words); the values the user
        fills in when the agent runs (inputs nobody gave: folders, files, recipients); what the critic added or
        thinks is missing; which agents ran."""
        m = self._mode
        label = MODE_LABELS.get(m["mode"], m["mode"])
        head = f"Mode: {label}" + (f" (picked by Auto: {m['why']})" if m["asked"] == "auto" else "") + \
            f", {seconds:.1f} s."
        out = [head]
        if flow is not None:
            if not understood:
                try:
                    what = F.clear_request(flow, self.cat)
                except Exception:
                    what = None
                if what:
                    out.append("What it does: " + what.rstrip(".") + ".")
            have = {v.get("name") for v in existing or [] if isinstance(v, dict)}
            ins = [v.get("name") for v in flow.get("variables", []) if isinstance(v, dict)
                   and v.get("direction") == "input" and v.get("name") not in have]
            if ins:
                out.append("Assumed: you give " + ", ".join(ins[:8]) + (" and more" if len(ins) > 8 else "")
                           + " when it runs (the request did not name them).")
            if team.get("critic_added"):
                out.append("Added steps the first answer missed, for: " + "; ".join(team["critic_added"]) + ".")
            elif team.get("critic_missing"):
                out.append("Check that it also does: " + "; ".join(team["critic_missing"]) + ".")
        elif asked:
            out.append("It asked before guessing.")
        names = []
        for a in self._agents or []:
            n = a["agent"]
            if n == "builders" and a.get("parts"):
                n = f"builders ({a['parts']} parts at once)"
            if n not in names:
                names.append(n)
        if names:
            out.append("Agents: " + ", ".join(names + ["reporter"]) + ".")
        return " ".join(out)

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
# The stable id platforms register as their model connection's model ("OneOffice in-house model"): it runs the
# server's default mode like MODEL_ID, is echoed back in the reply, and stays the same across model versions (the
# version is in the reply's "ooa.version"). The mode names below still pick a mode.
STABLE_MODEL_ID = "ooa-flow-1"
# OpenAI-style model names (GET /v1/models): the name picks the mode. MODEL_ID runs the server's default mode.
MODEL_BASE = "ooa-flow-assistant"
MODEL_NAMES = {MODEL_BASE: "auto", MODEL_BASE + "-fast": "fast", MODEL_BASE + "-smart": "smart",
               MODEL_BASE + "-ultra": "ultra"}
ECHOED_NAMES = {STABLE_MODEL_ID} | set(MODEL_NAMES)     # reply "model" is the name asked for; others get MODEL_ID
MODE_HEADER = "X-OOA-Mode"
RESULT_CONTRACT = "ooa-flow-result/1"   # the "ooa.result" object below


def models_list():
    """GET /v1/models: MODEL_ID (the server's default mode), one name per mode, then the stable id (last, so clients
    that take the first entry keep getting MODEL_ID)."""
    return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "local"}
                                       for n in [MODEL_ID] + list(MODEL_NAMES) + [STABLE_MODEL_ID]]}


def request_mode(body, header=None):
    """The mode a request asks for -> (mode or None, error or None). In order: the body's "mode" field, the
    X-OOA-Mode header, the model name (MODEL_NAMES; MODEL_ID and other names: None, the server's default)."""
    for given in ((body or {}).get("mode") if isinstance(body, dict) else None, header):
        if given is None or given == "":
            continue
        m = str(given).strip().lower()
        if m not in MODE_NAMES:
            return None, f"mode must be one of {', '.join(MODE_NAMES)}, not {given!r}"
        return m, None
    name = body.get("model") if isinstance(body, dict) else None
    return MODEL_NAMES.get(str(name).strip().lower()) if isinstance(name, str) else None, None
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


def chat_completion(router, body, count_tokens=None, version=None, deadline=None, mode_header=None):
    """OpenAI-style /v1/chat/completions on the router: (http_status, response). Used by this server and by the
    PC assistant (assistant/app.py), so One Agent Studio's Assistant can use the model as an OpenAI-compatible
    One Model Connection.
    count_tokens(text) -> int counts usage with the model's own tokenizer (default: characters / 4).
    version: dict added to the response's extra "ooa" field (pack / model version).
    deadline: time.time() after which no further retry attempt starts (see Router.run).
    mode_header: the X-OOA-Mode header's value; the mode comes from request_mode (body "mode", the header, the
    model name), else the router's default. The reply's "model" is the name asked for when it is one of ours."""
    quick = quick_reply(body, count_tokens, version)
    if quick is not None:
        return quick
    mode, err = request_mode(body, mode_header)
    if err:
        return 400, openai_error(err)
    msgs = [m for m in body["messages"] if isinstance(m, dict)]
    request, system, assistant, existing, full_user = parse_messages(msgs)
    res = router.run(request, system, assistant, existing, full_user, deadline=deadline, mode=mode)
    tail = (" " + res["report"]) if res.get("report") else ""
    if res["verified"]:
        content = res["flow"]
    elif res.get("question"):     # the model asks before it guesses: Studio shows notes
        content = (json.dumps({"variables": [], "steps": [], "notes": res["question"] + tail, "warnings": []})
                   if assistant else res["question"])
    elif assistant:   # Studio reads steps/notes: an empty answer with the honest reason is shown to the user
        content = json.dumps({"variables": [], "steps": [], "notes": "No flow was returned: " + res["reason"] + ". " + " ".join(res.get("problems", [])[:2]) + tail})
    else:
        content = json.dumps({"error": res["reason"], "problems": res.get("problems", [])})
    name = body.get("model") if isinstance(body.get("model"), str) and body.get("model") in ECHOED_NAMES else MODEL_ID
    ooa = {k: v for k, v in res.items() if k != "flow"}
    ooa["result"] = structured_result(res)
    return 200, _completion(msgs, content, ooa, count_tokens, version, name)


def _json_object(text):
    """The JSON object in a reply text (first '{' to last '}', as Studio reads it), or None."""
    if not isinstance(text, str) or "{" not in text:
        return None
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def structured_result(res):
    """The machine-readable answer next to the message content ("ooa.result", contract RESULT_CONTRACT), so a program
    (the platform's flow-assistant endpoint) does not have to guess from the text:
      status    "flow" (a verified flow), "question" (the model asks first) or "no_flow" (nothing passed the checker)
      flow      the flow as a JSON object (flow.v1, or Studio's {variables, steps, notes} when Studio's message shape was
                sent), None unless status is "flow"
      plan      the plan lines the planner wrote; question: the question text or None
      verdict   the checker's verdict: verified, problems / repairs, attempts, warnings, reason
    """
    verified = bool(res.get("verified"))
    status = "flow" if verified else ("question" if res.get("question") else "no_flow")
    attempts = res.get("attempts")
    return {"contract": RESULT_CONTRACT, "status": status,
            "flow": _json_object(res.get("flow")) if verified else None,
            "plan": list(res.get("plan") or []), "question": res.get("question") or None,
            "verdict": {"verified": verified, "problems": list(res.get("problems") or []),
                        "repairs": list(res.get("repairs") or []),
                        "attempts": len(attempts) if isinstance(attempts, list) else (attempts or 0),
                        "warnings": list(res.get("warnings") or []), "reason": res.get("reason")},
            "mode": res.get("mode"), "report": res.get("report"), "seconds": res.get("seconds")}


def _completion(msgs, content, ooa, count_tokens, version, model=MODEL_ID):
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
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
            "ooa": ooa}


API_KEY_ENV = "OOA_API_KEY"         # optional bearer token for /v1/* (this server and assistant/app.py)
MAX_BODY = 4 * 1024 * 1024          # bytes: a larger request body gets 413


def bearer_ok(header, token):
    """True when no token is set, or the Authorization header carries it (constant-time compare)."""
    if not token:
        return True
    import hmac
    h = header or ""
    given = h[7:].strip() if h[:7].lower() == "bearer " else ""
    return bool(given) and hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8"))


def make_handler(router, token=None):
    """token: bearer token /v1/* needs (default: the OOA_API_KEY environment variable; None or empty: no token)."""
    token = token if token is not None else (os.environ.get(API_KEY_ENV) or None)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj, headers=None):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass

        def _authorized(self):
            if self.path.startswith("/v1/") and not bearer_ok(self.headers.get("Authorization"), token):
                self._send(401, openai_error("missing or wrong API key: send Authorization: Bearer <key>",
                                             "authentication_error"), {"WWW-Authenticate": "Bearer"})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "policy": asdict(router.policy), "mode": router.default_mode})
            if not self._authorized():
                return
            if self.path == "/v1/models":
                return self._send(200, models_list())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return
            try:
                n = int(self.headers.get("content-length", 0))
            except ValueError:
                return self._send(400, openai_error("bad Content-Length"))
            if n < 0 or n > MAX_BODY:
                return self._send(413 if n > 0 else 400, openai_error("request body too large" if n > 0
                                                                       else "bad Content-Length"))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, {"error": "invalid JSON body"})
            if not isinstance(body, dict):
                return self._send(400, openai_error("the body must be a JSON object"))
            if self.path == "/v1/chat/completions":
                return self._send(*chat_completion(router, body, mode_header=self.headers.get(MODE_HEADER)))
            if self.path == "/v1/feedback":
                out = router.feedback(body)
                return self._send(400 if out.get("error") else 200, out)
            self._send(404, {"error": "not found"})
    return H


def serve(router, port=8080, host="127.0.0.1", token=None):
    srv = ThreadingHTTPServer((host, port), make_handler(router, token))
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
    ap.add_argument("--mode", default="auto", choices=list(MODE_NAMES) + ["none"],
                    help="mode of a request that names none (model ooa-assistant): auto (default) picks fast, smart or "
                         "ultra per request; none: the policy's own settings, without the agent team")
    a = ap.parse_args()
    pol = Policy.load(a.policy)
    local = None
    if a.local == "nanochat":
        from backends import NanochatBackend
        local = NanochatBackend(a.source)
    cloud = make_cloud(pol)
    examples = a.examples or str(Path(a.catalog).resolve().parent / "examples.jsonl")
    r = Router(a.catalog, a.platform, pol, local, cloud, examples, a.prompt_format,
               mode=None if a.mode == "none" else a.mode)
    print(f"OOA assistant on http://127.0.0.1:{a.port}  | local model: {bool(local)} ({r.format()} prompts) | "
          f"escalation: {pol.allow_escalation} | mode: {a.mode}")
    serve(r, a.port).serve_forever()
