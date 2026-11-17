"""OOA Assistant: a local web app that runs OUR flow-writing model on this PC (CPU is enough).

    python app.py --model-dir DIR --nanochat DIR --platform DIR [--catalog FILE] [--port 8765] [--data-dir DIR] [--fake-model]
                  [--host 127.0.0.1] [--allowed-hosts NAME ...] [--token TOKEN | env OOA_ASSISTANT_TOKEN]

Binds 127.0.0.1 by default (nothing is reachable from other machines). Another --host (a server, a container) needs a
bearer token: the model routes (/v1/*, /api/assistant/v1/*) then require "Authorization: Bearer <token>", and the
browser page and its /api/* routes are off (403): they have no login, and behind a proxy every caller would look like
this machine. The page also refuses requests that came through a proxy (Forwarded / X-Forwarded-For / X-Real-IP).
See DEPLOY.md.
Loads the model once at start; one generation at a time, with a bounded queue (429 when it is full).
Tabs in the page (static/index.html): Build, Learn, Practice, Run, Progress. See README.md and SPEC.md.

Model layers (serving/ooa_server.Router): retrieval context, plan-first answers, clarifying questions, self-correction,
building a long plan in parts, safety warnings, and optional escalation to an OpenAI-compatible endpoint
(--escalate-url, off by default; our own model always tries first). A model trained before context and plans keeps
working (--prompt-format auto picks "plain" when no examples.jsonl came with the model pack).

Self-learning: Accept / Edit / Reject on a built flow is appended to DATA_DIR/feedback.jsonl, and so is Studio's
inserted / discarded / edited verdict posted to /v1/feedback (a model route: bearer token when set). "Export training data"
runs training/ingest_feedback.py on it and writes DATA_DIR/exports/<timestamp>/feedback_train.jsonl (only flows that pass
the verifier). The model does NOT change live: it improves at the next training run that mixes this file in.
"""
from __future__ import annotations

import argparse, hashlib, hmac, ipaddress, json, mimetypes, os, re, socket, subprocess, sys, threading, time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                    # "rpa model/"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "serving"))
sys.path.insert(0, str(ROOT / "training"))

from knowledge import Knowledge              # noqa: E402
from runner_bridge import RunnerBridge       # noqa: E402
from ooa_server import (Router, Policy, chat_completion, quick_reply, openai_error, feedback_row, make_cloud,  # noqa: E402
                        MODEL_ID, call_generate, models_list, MODE_NAMES, MODE_HEADER)
import ooa_reward                            # noqa: E402

STATIC = HERE / "static"
INGEST = ROOT / "training" / "ingest_feedback.py"
MAX_BODY = 4 * 1024 * 1024
MAX_REQUEST = 500          # serving/ooa_server.Router skips the local model for longer requests
CSRF_HEADER = "X-OOA-Assistant"   # every POST must carry it: a web page on another site cannot send it without a
                                  # CORS preflight, which this server never approves
VERDICTS = ("accept", "edit", "reject")
TOKEN_ENV = "OOA_ASSISTANT_TOKEN"
MIN_TOKEN = 16
CONTRACT = "studio-assistant/v1"
MAX_WAITING = 4            # requests that may wait for the model while one runs; more get 429
TIME_BUDGET = 120.0        # seconds per request: no retry attempt starts after this
RETRY_AFTER = 10           # seconds, sent with 429
OPEN_PATHS = ("/health", "/ready")                         # no checks at all: process and readiness probes
API_PREFIXES = ("/v1/", "/api/assistant/v1/")              # model routes for programs (Studio): bearer token when set
PROGRAM_POSTS = ("/v1/chat/completions", "/v1/feedback")   # POSTs programs on this computer send without our header
PROXY_HEADERS = ("Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Real-IP")   # the page refuses proxied requests

GOOD_AT = ["Step-by-step requests worded like the catalog's action descriptions, with the inputs and outputs named, "
           "e.g. 'Read a text file, then parse JSON text into a Dictionary or List. The flow takes path as input. "
           "It returns value.'",
           "Questions about the action catalog (which action does X, what inputs does Y need)."]
WEAK_AT = ["Everyday wording ('sort out my invoices') is weak: say each step plainly.",
           "Long business flows with many steps usually fail verification.",
           "Answers to catalog questions can be wrong: the catalog shown next to them is the source of truth."]
LEARNING_NOTE = ("The model does not change while you use it. Your Accept / Edit / Reject feedback is saved on this PC; "
                 "'Export training data' turns it into verified training rows for the next training run.")
EXPORT_STEPS = ["Upload feedback_train.jsonl from the folder above to Google Drive, folder ooa-nanochat/sft/.",
                "Open the training notebook in Colab and run it: its training cell mixes feedback_train.jsonl in.",
                "Download the new checkpoint (chatsft_checkpoints/d8) into your model folder and restart the assistant.",
                "rejected_prompts.jsonl lists requests you rejected without an edit: good cases to write correct "
                "answers for."]


class ApiError(Exception):
    def __init__(self, code: int, message: str, headers=None):
        super().__init__(message)
        self.code, self.message, self.headers = code, message, headers or {}


ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "invalid_request_error", 429: "rate_limit_error",
               503: "service_unavailable"}


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)          # ::ffff:127.0.0.1 on a dual-stack socket
    return ip.is_loopback or bool(mapped and mapped.is_loopback)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_catalog(model_dir: str | None) -> str:
    if model_dir:
        c = Path(model_dir).resolve().parent / "sft" / "catalog.jsonl"
        if c.exists():
            return str(c)
    return str(ROOT / "data" / "catalog" / "catalog.jsonl")


def load_model(args, catalog=None):
    if args.fake_model:
        from model_backend import FakeModel
        return FakeModel()
    if not args.model_dir or not args.nanochat:
        raise SystemExit("--model-dir and --nanochat are required (or use --fake-model to try the app without the model)")
    from model_backend import NanochatLocal
    return NanochatLocal(args.model_dir, args.nanochat, catalog=None if args.no_catalog_lock else catalog)


class _StatsRecorder:
    """Wraps the model for the Router: records why each generation stopped (end / max_tokens / loop) and how long it took."""

    def __init__(self, model):
        self.model, self.stats = model, []

    @property
    def info(self):
        return getattr(self.model, "info", None)

    @property
    def last_stats(self):
        return getattr(self.model, "last_stats", None)

    def generate(self, request, attempt=0, **options):
        """options (max_tokens, on_plan) go to the model when its generate takes them (Router.generate checks)."""
        t0 = time.time()
        try:
            return call_generate(self.model, request, attempt, **options)
        finally:
            st = dict(getattr(self.model, "last_stats", None) or {})
            st.pop("ids", None)
            st.setdefault("seconds", round(time.time() - t0, 2))
            self.stats.append(st)

    @property
    def last_batch_stats(self):
        return getattr(self.model, "last_batch_stats", None)

    def __getattr__(self, name):
        # generate_batch (the agent team's parallel builders) only when the model has it: the Router checks
        if name == "generate_batch" and callable(getattr(self.model, "generate_batch", None)):
            return self._generate_batch
        raise AttributeError(name)

    def _generate_batch(self, items, deadline=None):
        t0 = time.time()
        try:
            return self.model.generate_batch(items, deadline=deadline)
        finally:
            for st in list(getattr(self.model, "last_batch_stats", None) or []):
                st = dict(st)
                st.pop("ids", None)
                st.setdefault("seconds", round(time.time() - t0, 2))
                self.stats.append(st)


class Assistant:
    """Everything the routes need, wired once."""

    def __init__(self, catalog: str, platform: str, data_dir: str, model, max_waiting: int = MAX_WAITING,
                 time_budget: float = TIME_BUDGET, policy: Policy | None = None, cloud=None, examples: str | None = None,
                 prompt_format: str = "auto", mode: str | None = None):
        """model may be None while it loads (main() loads it after the server is up): /ready says so and the
        model routes answer 503 until set_model() is called.
        policy / cloud: escalation (off by default: Policy(allow_escalation=False)); examples: the example index of
        the model pack (default: examples.jsonl next to the catalog); prompt_format: auto | plain | context."""
        self.catalog = str(Path(catalog).resolve())
        self.platform = str(Path(platform).resolve())
        self.schema = str(Path(self.platform) / "contracts" / "flow-schema" / "flow.v1.json")
        if not Path(self.catalog).exists():
            raise SystemExit(f"catalog not found: {self.catalog}")
        if not Path(self.schema).exists():
            raise SystemExit(f"flow schema not found: {self.schema} (is --platform the platform repo folder?)")
        self.data = Path(data_dir).resolve()
        (self.data / "sandbox").mkdir(parents=True, exist_ok=True)
        (self.data / "exports").mkdir(parents=True, exist_ok=True)
        self.feedback_path = self.data / "feedback.jsonl"
        self.model = None
        self.model_error = None
        self.knowledge = Knowledge(self.catalog)
        self.runner = RunnerBridge(self.platform, str(self.data / "sandbox"))
        self.recorder = _StatsRecorder(model)
        if examples is None:
            found = Path(self.catalog).parent / "examples.jsonl"
            examples = str(found) if found.is_file() else None
        self.router = Router(self.catalog, self.platform, policy or Policy(allow_escalation=False, local_retries=1),
                             local=self.recorder, cloud=cloud, examples=examples, prompt_format=prompt_format,
                             mode=mode)
        self.catalog_sha256, self.schema_sha256 = _sha256(self.catalog), _sha256(self.schema)
        self.model_lock = threading.Lock()      # one generation at a time
        self.wait_lock = threading.Lock()
        self.waiting, self.max_waiting, self.time_budget = 0, max_waiting, time_budget
        self.fb_lock = threading.Lock()
        if model is not None:
            self.set_model(model)

    def set_model(self, model):
        self.recorder.model = model
        self.model = model

    # ------------------------------------------------------------------ model queue / readiness / version
    @contextmanager
    def generation(self):
        """Wait for the model (one generation at a time). At most max_waiting requests wait; one more, or a wait
        longer than the time budget, gets 429. A request that finds the model free never counts as waiting, so
        max_waiting 0 means: no queue, one request at a time. Yields the deadline after which no retry attempt starts."""
        if self.model is None:
            raise ApiError(503, "the model is still loading" if not self.model_error else
                           "the model failed to load: " + self.model_error)
        deadline = time.time() + self.time_budget
        got = self.model_lock.acquire(blocking=False)       # the model is free: no waiting, whatever max_waiting is
        if not got:
            with self.wait_lock:
                if self.waiting >= self.max_waiting:
                    raise ApiError(429, f"busy: {self.waiting} requests are already waiting for the model; try "
                                        "again shortly", {"Retry-After": str(RETRY_AFTER)})
                self.waiting += 1
            try:
                got = self.model_lock.acquire(timeout=self.time_budget)
            finally:
                with self.wait_lock:
                    self.waiting -= 1
        if not got:
            raise ApiError(429, "busy: the model did not become free in time; try again shortly",
                           {"Retry-After": str(RETRY_AFTER)})
        try:
            yield deadline
        finally:
            self.model_lock.release()

    def missing(self):
        """What is not loaded yet (empty list = ready)."""
        out = []
        if self.model is None:
            out.append("model" + (" (failed to load: " + self.model_error + ")" if self.model_error else " (loading)"))
        if not self.router.cat or not self.knowledge.entries:
            out.append("catalog")
        if getattr(ooa_reward, "_SV", None) is None:
            out.append("schema")
        return out

    def version(self):
        info = dict(getattr(self.model, "info", {}) or {})
        return {"version": info.get("version") or (f"{info.get('source', 'unknown')}-step{info.get('step')}"
                                                   if self.model is not None else None),
                "contract": CONTRACT, "model_step": info.get("step"), "catalog_actions": len(self.router.cat),
                "catalog_sha256": self.catalog_sha256, "schema_sha256": self.schema_sha256}

    def chat(self, body, mode_header=None):
        """POST /v1/chat/completions -> (status, body). A probe or a refused request never waits for the model.
        mode_header: the X-OOA-Mode header (see ooa_server.request_mode)."""
        counter = getattr(self.model, "count_tokens", None)      # the real tokenizer; None: characters / 4
        v = self.version()
        version = {"version": v["version"], "model_step": v["model_step"], "catalog_sha256": v["catalog_sha256"]}
        quick = quick_reply(body, counter, version)
        if quick is not None:
            return quick
        with self.generation() as deadline:
            return chat_completion(self.router, body, counter, version, deadline, mode_header)

    # ------------------------------------------------------------------ status / build
    def status(self):
        info = dict(getattr(self.model, "info", {}) or {})
        return {"model": info, "catalog_size": len(self.knowledge.entries), "catalog": self.catalog,
                "runner": {"available": self.runner.available,
                           "supported_count": len(self.runner.supported_actions()),
                           "reason": getattr(self.runner, "unavailable_reason", None)},
                "feedback": self._counts(self._feedback_rows()),
                "good_at": GOOD_AT, "weak_at": WEAK_AT, "learning": LEARNING_NOTE}

    def build(self, body):
        request = _text(body, "request", MAX_REQUEST)
        mode = body.get("mode") or None
        if mode is not None and mode not in MODE_NAMES:
            raise ApiError(400, "mode must be one of " + ", ".join(MODE_NAMES))
        with self.generation() as deadline:
            self.recorder.stats = []
            res = self.router.run(request, deadline=deadline, mode=mode)
            stats = self.recorder.stats
        attempts = [dict(a) for a in res.get("attempts", [])]
        if not res.get("verified") and not attempts:
            # The Router skips the local model for complex requests: say so instead of "failed the checker".
            return {"verified": False, "model_tried": False, "flow": None, "repairs": [], "attempts": [],
                    "seconds": res.get("seconds"), "explanation": None, "can_run": None,
                    "problems": res.get("problems", []),
                    "reason": "The model was not tried: requests with more than 6 steps (or longer than "
                              f"{MAX_REQUEST} characters) are skipped because the local model fails on them. "
                              "Split it into smaller requests."}
        for a, st in zip([a for a in attempts if a.get("source") == "local"], stats):
            a.update({k: st[k] for k in ("tokens", "stop", "seconds") if k in st})
        flow = json.loads(res["flow"]) if res.get("verified") and res.get("flow") else None
        out = {"verified": flow is not None, "flow": flow, "problems": res.get("problems", []) if flow is None else [],
               "repairs": res.get("repairs", []) if flow is not None else [], "seconds": res.get("seconds"),
               "attempts": attempts, "model_tried": True, "reason": res.get("reason"), "explanation": None, "can_run": None,
               "plan": res.get("plan", []), "warnings": res.get("warnings", []), "question": res.get("question"),
               "source": res.get("source"), "mode": res.get("mode"), "report": res.get("report"),
               "agents": res.get("agents", [])}
        if flow is not None:
            out["explanation"] = self.knowledge.explain_flow(flow)
            out["can_run"] = self.runner.can_run(flow)
        return out

    def ask(self, body):
        q = _text(body, "question", 1000)
        with self.generation():
            answer = self.model.ask(q)
        return {"model_answer": answer, "catalog_matches": self.knowledge.search(q, 5),
                "note": "The catalog matches are the source of truth; the model's answer can be wrong."}

    # ------------------------------------------------------------------ run
    def run(self, body):
        flow = _flow(body, "flow")
        inputs = body.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise ApiError(400, "inputs must be an object")
        return self.runner.run(flow, inputs, allow_risky=body.get("allow_risky") is True)

    # ------------------------------------------------------------------ feedback / progress / export
    def feedback(self, body):
        request = _text(body, "request", 5000)
        verdict = body.get("verdict")
        if verdict not in VERDICTS:
            raise ApiError(400, "verdict must be one of accept, edit, reject")
        flow = body.get("flow")
        if flow is not None and not isinstance(flow, dict):
            raise ApiError(400, "flow must be a JSON object")
        if verdict == "accept" and flow is None:
            raise ApiError(400, "accept needs the flow")
        edited = None
        if verdict == "edit":
            edited = _flow(body, "edited_flow")
        note = body.get("note") or ""
        if not isinstance(note, str):
            raise ApiError(400, "note must be text")
        row = {"prompt": request, "flow": flow, "accepted": verdict == "accept", "edited_flow": edited,
               "verdict": verdict, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note[:2000]}
        with self.fb_lock, open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"stored": True, "counts": self._counts(self._feedback_rows()), "learning": LEARNING_NOTE}

    def program_feedback(self, body):
        """POST /v1/feedback from a program (Studio): {request, flow, verdict: inserted|discarded|edited, edited_flow?}.
        Stored in the same feedback.jsonl as the page's Accept / Edit / Reject."""
        try:
            row = feedback_row(body)
        except ValueError as e:
            raise ApiError(400, str(e))
        note = body.get("note")
        if isinstance(note, str) and note:
            row["note"] = note[:2000]
        row["source"] = "api"
        with self.fb_lock, open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"stored": True, "counts": self._counts(self._feedback_rows())}

    def _feedback_rows(self):
        if not self.feedback_path.exists():
            return []
        rows = []
        with self.fb_lock, open(self.feedback_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    @staticmethod
    def _counts(rows):
        c = {v: 0 for v in VERDICTS}
        for r in rows:
            v = r.get("verdict") or ("accept" if r.get("accepted") else "edit" if r.get("edited_flow") else "reject")
            c[v] = c.get(v, 0) + 1
        c["total"] = len(rows)
        return c

    def progress(self):
        rows = self._feedback_rows()
        recent = [{"prompt": r.get("prompt", ""), "verdict": r.get("verdict"), "time": r.get("time")}
                  for r in reversed(rows[-20:])]
        return {"counts": self._counts(rows), "recent": recent, "learning": LEARNING_NOTE}

    def export(self):
        if not self._feedback_rows():
            raise ApiError(400, "no feedback yet: accept, edit or reject some built flows first")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = self.data / "exports" / stamp
        n = 1
        while out.exists():
            n += 1
            out = self.data / "exports" / f"{stamp}-{n}"
        with self.fb_lock:   # copy a consistent snapshot so a concurrent append cannot cut a line
            snap = self.feedback_path.read_bytes()
        out.mkdir(parents=True)
        src = out / "feedback_snapshot.jsonl"
        src.write_bytes(snap)
        p = subprocess.run([sys.executable, str(INGEST), str(src), str(out), self.catalog, self.schema],
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            raise ApiError(500, "ingest_feedback failed: " + (p.stderr.strip().splitlines() or ["unknown error"])[-1])
        m = re.search(r"(\d+) usable, (\d+) dropped.*?, (\d+) rejected", p.stdout)
        if not m:
            raise ApiError(500, "unexpected ingest_feedback output: " + p.stdout.strip()[:200])
        usable, dropped, rejected = map(int, m.groups())
        return {"usable": usable, "dropped": dropped, "rejected": rejected, "path": str(out),
                "train_file": str(out / "feedback_train.jsonl"), "records_file": str(out / "feedback_records.jsonl"),
                "instructions": EXPORT_STEPS,
                "note": "Dropped rows failed the verifier and are never trained on."}


# ---------------------------------------------------------------------- body helpers
def _text(body, key, limit):
    v = body.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ApiError(400, f"{key} is required (text)")
    if len(v) > limit:
        raise ApiError(400, f"{key} is too long (max {limit} characters)")
    return v.strip()


def _flow(body, key):
    v = body.get(key)
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError as e:
            raise ApiError(400, f"{key} is not valid JSON: {e}")
    if not isinstance(v, dict):
        raise ApiError(400, f"{key} must be a flow JSON object")
    return v


# ---------------------------------------------------------------------- HTTP
def make_handler(app: Assistant, token: str | None = None, allowed_hosts=()):
    """token: when set, the model routes (API_PREFIXES) need "Authorization: Bearer <token>", and the browser page and
    its other /api/* routes (which have no login: /api/run runs a posted flow) are off.
    allowed_hosts: extra Host header values accepted ("name" or "name:port"; a bare name matches any port)."""
    extra_hosts = {h.strip().lower() for h in allowed_hosts if h and h.strip()}

    class Handler(BaseHTTPRequestHandler):
        server_version = "OOAAssistant/1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj=None, raw=None, ctype="application/json; charset=utf-8", headers=None):
            b = raw if raw is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(b)

        def _static(self, name):
            base = STATIC.resolve()
            try:
                p = (base / name).resolve()
            except (OSError, ValueError):
                raise ApiError(404, "not found")
            if not name or "\\" in name or base not in p.parents or not p.is_file():
                raise ApiError(404, "not found")
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, raw=p.read_bytes(), ctype=ctype)

        def _host_ok(self, host, port):
            host = host.lower()
            if host in {f"127.0.0.1:{port}", f"localhost:{port}"} or host in extra_hosts:
                return True
            return host.rsplit(":", 1)[0] in extra_hosts if ":" in host else False

        def _check_token(self):
            """Bearer token for the model routes (constant-time compare)."""
            auth = self.headers.get("Authorization") or ""
            given = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
            if not given or not hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8")):
                raise ApiError(401, "missing or wrong API key: send Authorization: Bearer <token>",
                               {"WWW-Authenticate": "Bearer"})

        def _check_access(self, method, path):
            if path in OPEN_PATHS:
                return
            api = path.startswith(API_PREFIXES)
            if api and token:
                self._check_token()          # the token replaces the Host / Origin checks for programs
                if method == "POST":
                    ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                    if ctype != "application/json":
                        raise ApiError(400, "POST needs Content-Type application/json")
                return
            if not api and token:
                raise ApiError(403, "forbidden: the assistant page is off while a token is set; start another copy "
                                    "without --token and --host to use it on this machine")
            if not api and any(self.headers.get(h) is not None for h in PROXY_HEADERS):
                raise ApiError(403, "forbidden: the assistant page is not served through a proxy; open it on this "
                                    "machine at http://127.0.0.1:%d/" % self.server.server_address[1])
            self._check_origin(method)

        def _check_origin(self, method):
            """Blocks other web sites (CSRF) and DNS rebinding: only this page on 127.0.0.1/localhost may call us."""
            port = self.server.server_address[1]
            if not self._host_ok(self.headers.get("Host") or "", port):
                raise ApiError(403, "forbidden: open the assistant at http://127.0.0.1:%d/" % port)
            origin = self.headers.get("Origin")
            scheme, _, rest = (origin or "").lower().partition("://")
            if origin is not None and not (scheme in ("http", "https") and self._host_ok(rest, port)):
                raise ApiError(403, "forbidden: requests from other web sites are refused")
            if method == "POST":
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                # /v1/chat/completions and /v1/feedback are for programs on this computer (One Agent Studio), which
                # do not send our header. Still safe: only JSON from this computer, and a web page cannot send JSON
                # to it without the browser asking first (we never say yes), so other sites cannot reach it.
                openai = urlparse(self.path).path in PROGRAM_POSTS and origin is None
                if ctype != "application/json" or (self.headers.get(CSRF_HEADER) != "1" and not openai):
                    raise ApiError(403, f"forbidden: POST needs Content-Type application/json and {CSRF_HEADER}: 1")

        def _dispatch(self, method):
            u = urlparse(self.path)
            path, q = u.path, {k: v[-1] for k, v in parse_qs(u.query).items()}
            self._check_access(method, path)
            if method == "GET":
                if path == "/health":             # the process is up
                    return self._send(200, {"ok": True})
                if path == "/ready":              # model, catalog and schema loaded
                    missing = app.missing()
                    return self._send(503 if missing else 200, {"ready": not missing, "missing": missing})
                if path == "/api/assistant/v1/admin/version":
                    return self._send(200, app.version())
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/v1/models":          # OpenAI-compatible model list (for model connection set-up screens)
                    return self._send(200, models_list())
                k = app.knowledge
                if path == "/api/status":
                    return self._send(200, app.status())
                if path == "/api/search":
                    query = q.get("q", "").strip()
                    if not query:
                        raise ApiError(400, "q is required")
                    return self._send(200, {"results": k.search(query, _int(q.get("k"), 10, 1, 50))})
                if path == "/api/action":
                    card = k.action_card(q.get("id", ""))
                    if card is None:
                        raise ApiError(404, f"no action with id {q.get('id', '')!r} in the catalog")
                    return self._send(200, card)
                if path == "/api/toolkits":
                    return self._send(200, {"toolkits": k.toolkits()})
                if path == "/api/lesson":
                    tk = q.get("toolkit", "")
                    if tk not in {t["id"] for t in k.toolkits()}:
                        raise ApiError(404, f"no toolkit {tk!r}")
                    return self._send(200, k.lesson(tk))
                if path == "/api/quiz":
                    kind = q.get("kind") or None
                    try:
                        return self._send(200, k.quiz(kind=kind))
                    except (ValueError, KeyError) as e:
                        raise ApiError(400, f"cannot make a quiz of kind {kind!r}: {e}")
                if path == "/api/progress":
                    return self._send(200, app.progress())
                raise ApiError(404, "not found")
            # POST
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError(400, "bad Content-Length")
            if n < 0:
                raise ApiError(400, "bad Content-Length")
            if n > MAX_BODY:
                raise ApiError(413, "request body too large")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                raise ApiError(400, "invalid JSON body")
            if not isinstance(body, dict):
                raise ApiError(400, "the body must be a JSON object")
            if path == "/v1/chat/completions":
                return self._send(*app.chat(body, self.headers.get(MODE_HEADER)))   # one generation at a time
            if path == "/v1/feedback":
                return self._send(200, app.program_feedback(body))
            routes = {
                "/api/build": app.build,
                "/api/explain": lambda b: app.knowledge.explain_flow(_flow(b, "flow")),
                "/api/ask": app.ask,
                "/api/quiz/check": self._check_quiz,
                "/api/can_run": lambda b: app.runner.can_run(_flow(b, "flow")),
                "/api/run": app.run,
                "/api/feedback": app.feedback,
                "/api/export": lambda b: app.export(),
            }
            if path not in routes:
                raise ApiError(404, "not found")
            return self._send(200, routes[path](body))

        @staticmethod
        def _check_quiz(b):
            qid, choice = b.get("id"), b.get("choice")
            if not isinstance(qid, str) or not isinstance(choice, int) or isinstance(choice, bool):
                raise ApiError(400, "id (text) and choice (number 0-3) are required")
            try:
                return app.knowledge.check_quiz(qid, choice)
            except (ValueError, KeyError, TypeError) as e:
                raise ApiError(400, f"invalid quiz id or choice: {e}")

        def _error(self, code, message, headers=None):
            if urlparse(self.path).path.startswith(API_PREFIXES):     # OpenAI-style error body for programs
                obj = openai_error(message, ERROR_TYPES.get(code, "server_error"))
            else:
                obj = {"error": message}
            self._send(code, obj, headers=headers)

        def _handle(self, method):
            try:
                self._dispatch(method)
            except ApiError as e:
                self._error(e.code, e.message, e.headers)
            except Exception as e:  # noqa: BLE001 - one bad request must not kill the server
                self._error(500, f"{type(e).__name__}: {e}")

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def _int(v, default, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


class _Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_server(app: Assistant, port: int = 8765, host: str = "127.0.0.1", token: str | None = None,
                allowed_hosts=()) -> ThreadingHTTPServer:
    """Binding beyond loopback needs a token (SystemExit otherwise). With a token the browser page is off."""
    if token is not None and len(token) < MIN_TOKEN:
        raise SystemExit(f"the token must be at least {MIN_TOKEN} characters long")
    local = is_loopback(host)
    if not local and not token:
        raise SystemExit(f"--host {host} makes the assistant reachable from other machines: set a token first "
                         f"(--token or the environment variable {TOKEN_ENV})")
    cls = _Server6 if ":" in host else ThreadingHTTPServer
    srv = cls((host, port), make_handler(app, token, allowed_hosts))
    srv.daemon_threads = True
    return srv


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="OOA Assistant: local web app for the OOA flow-writing model")
    ap.add_argument("--model-dir", help="nanochat base dir (has tokenizer/ and chatsft_checkpoints/d8/)")
    ap.add_argument("--nanochat", help="nanochat source code folder (commit 92d63d4)")
    ap.add_argument("--platform", required=True, help="platform repo folder (has contracts/flow-schema/flow.v1.json)")
    ap.add_argument("--catalog", help="catalog.jsonl (default: MODEL_DIR/../sft/catalog.jsonl, else the repo copy)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (default 127.0.0.1); any other address needs a token")
    ap.add_argument("--allowed-hosts", nargs="*", default=[], metavar="HOST",
                    help="extra Host header values accepted (name or name:port), e.g. the server's DNS name")
    ap.add_argument("--token", default=None,
                    help=f"bearer token the model routes require (default: environment variable {TOKEN_ENV}); the browser "
                         "page is then off; "
                         "prefer the environment variable, a command line is visible to other users")
    ap.add_argument("--max-waiting", type=int, default=MAX_WAITING,
                    help=f"requests that may wait for the model while one runs; more get 429 (default {MAX_WAITING}; "
                         "0: no queue, a request that finds the model busy gets 429 at once)")
    ap.add_argument("--time-budget", type=float, default=TIME_BUDGET,
                    help=f"seconds per request after which no retry attempt starts (default {TIME_BUDGET:g})")
    ap.add_argument("--data-dir", default="assistant_data", help="feedback, sandbox and exports go here")
    ap.add_argument("--fake-model", action="store_true", help="use a canned fake model (for trying the app / tests)")
    ap.add_argument("--examples", default=None,
                    help="example index (examples.jsonl from training/render_dataset.py; default: next to the catalog)")
    ap.add_argument("--prompt-format", default="auto", choices=["auto", "plain", "context"],
                    help="plain: the model gets the request only (models trained before context and plans); context: "
                         "request + actions + similar examples; auto (default): context when an example index is found")
    ap.add_argument("--escalate-url", default=None,
                    help="OpenAI-compatible endpoint (e.g. https://host/v1) that gets a request when our model's "
                         "attempts all fail or its plan is long; off by default. PII is masked before it leaves")
    ap.add_argument("--escalate-model", default="", help="model name to send to --escalate-url")
    ap.add_argument("--escalate-key-env", default="OOA_ESCALATE_API_KEY",
                    help="environment variable holding the API key for --escalate-url (default OOA_ESCALATE_API_KEY)")
    ap.add_argument("--escalate-plan-lines", type=int, default=12,
                    help="a failed flow whose plan has more lines goes to --escalate-url at once (default 12)")
    ap.add_argument("--mode", default="auto", choices=list(MODE_NAMES) + ["none"],
                    help="assistant mode when a request names none: auto (default) picks per request; fast (one "
                         "attempt, simple requests), smart (checks and fixes its work, long plans in parts), ultra "
                         "(several tries, for long processes); none: the settings before modes. Studio picks one with "
                         "the model name ooa-flow-assistant[-fast|-smart|-ultra]; the Build tab has its own menu")
    ap.add_argument("--no-catalog-lock", action="store_true",
                    help="let the model write any action / input name (default: only real catalog names, see constrained.py)")
    return ap.parse_args(argv)


def policy_from_args(args) -> Policy:
    """Local only unless --escalate-url is given; our own model always tries first."""
    esc = bool(args.escalate_url)
    return Policy(allow_escalation=esc, local_retries=1, local_first=True, escalate_url=args.escalate_url or "",
                  escalate_model=args.escalate_model, escalate_key_env=args.escalate_key_env,
                  escalate_plan_lines=args.escalate_plan_lines, prompt_format=args.prompt_format)


def main(argv=None):
    args = parse_args(argv)
    token = args.token or os.environ.get(TOKEN_ENV) or None
    if args.max_waiting < 0 or args.time_budget <= 0:
        raise SystemExit("--max-waiting must be 0 or more and --time-budget more than 0")
    if not args.fake_model and (not args.model_dir or not args.nanochat):
        raise SystemExit("--model-dir and --nanochat are required (or use --fake-model to try the app without the model)")
    catalog = args.catalog or default_catalog(args.model_dir)
    policy = policy_from_args(args)
    app = Assistant(catalog, args.platform, args.data_dir, None, args.max_waiting, args.time_budget, policy,
                    make_cloud(policy), args.examples, args.prompt_format, None if args.mode == "none" else args.mode)
    srv = make_server(app, args.port, args.host, token, args.allowed_hosts)
    failed = []

    def load():          # the server answers /health (and says "loading" on /ready) while the model loads
        try:
            app.set_model(load_model(args, catalog))
            where = ("the browser page is off while a token is set" if token else
                     f"open http://127.0.0.1:{srv.server_address[1]}/ in your browser on this machine")
            print(f"OOA Assistant is running: {where} (Ctrl+C to stop). Data folder: {app.data}", flush=True)
        except BaseException as e:  # noqa: BLE001 - report it and stop: a server without its model is no use
            app.model_error = f"{type(e).__name__}: {e}"
            print("The model failed to load: " + app.model_error, file=sys.stderr, flush=True)
            failed.append(e)
            threading.Thread(target=srv.shutdown, daemon=True).start()

    print(f"Listening on {args.host}:{srv.server_address[1]}. Loading the model (this can take a minute on CPU) ...",
          flush=True)
    threading.Thread(target=load, daemon=True).start()
    if token:
        print("Model routes (/v1/*, /api/assistant/v1/*) need the bearer token; the browser page is off.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        srv.server_close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
