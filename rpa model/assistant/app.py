"""OOA Assistant: a local web app that runs OUR flow-writing model on this PC (CPU is enough).

    python app.py --model-dir DIR --nanochat DIR --platform DIR [--catalog FILE] [--port 8765] [--data-dir DIR] [--fake-model]

Binds 127.0.0.1 only (nothing is reachable from other machines). Loads the model once at start; one generation at a time.
Tabs in the page (static/index.html): Build, Learn, Practice, Run, Progress. See README.md and SPEC.md.

Self-learning: Accept / Edit / Reject on a built flow is appended to DATA_DIR/feedback.jsonl. "Export training data"
runs training/ingest_feedback.py on it and writes DATA_DIR/exports/<timestamp>/feedback_train.jsonl (only flows that pass
the verifier). The model does NOT change live: it improves at the next training run that mixes this file in.
"""
from __future__ import annotations

import argparse, json, mimetypes, re, subprocess, sys, threading, time
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
from ooa_server import Router, Policy        # noqa: E402

STATIC = HERE / "static"
INGEST = ROOT / "training" / "ingest_feedback.py"
MAX_BODY = 4 * 1024 * 1024
MAX_REQUEST = 500          # serving/ooa_server.Router skips the local model for longer requests
CSRF_HEADER = "X-OOA-Assistant"   # every POST must carry it: a web page on another site cannot send it without a
                                  # CORS preflight, which this server never approves
VERDICTS = ("accept", "edit", "reject")

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
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def default_catalog(model_dir: str | None) -> str:
    if model_dir:
        c = Path(model_dir).resolve().parent / "sft" / "catalog.jsonl"
        if c.exists():
            return str(c)
    return str(ROOT / "data" / "catalog" / "catalog.jsonl")


def load_model(args):
    if args.fake_model:
        from model_backend import FakeModel
        return FakeModel()
    if not args.model_dir or not args.nanochat:
        raise SystemExit("--model-dir and --nanochat are required (or use --fake-model to try the app without the model)")
    from model_backend import NanochatLocal
    return NanochatLocal(args.model_dir, args.nanochat)


class _StatsRecorder:
    """Wraps the model for the Router: records why each generation stopped (end / max_tokens / loop) and how long it took."""

    def __init__(self, model):
        self.model, self.stats = model, []

    def generate(self, request, attempt=0):
        t0 = time.time()
        try:
            return self.model.generate(request, attempt)
        finally:
            st = dict(getattr(self.model, "last_stats", None) or {})
            st.pop("ids", None)
            st.setdefault("seconds", round(time.time() - t0, 2))
            self.stats.append(st)


class Assistant:
    """Everything the routes need, wired once."""

    def __init__(self, catalog: str, platform: str, data_dir: str, model):
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
        self.model = model
        self.knowledge = Knowledge(self.catalog)
        self.runner = RunnerBridge(self.platform, str(self.data / "sandbox"))
        self.recorder = _StatsRecorder(model)
        self.router = Router(self.catalog, self.platform, Policy(allow_escalation=False, local_retries=1),
                             local=self.recorder)
        self.model_lock = threading.Lock()      # one generation at a time
        self.fb_lock = threading.Lock()

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
        with self.model_lock:
            self.recorder.stats = []
            res = self.router.run(request)
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
               "attempts": attempts, "model_tried": True, "reason": res.get("reason"), "explanation": None, "can_run": None}
        if flow is not None:
            out["explanation"] = self.knowledge.explain_flow(flow)
            out["can_run"] = self.runner.can_run(flow)
        return out

    def ask(self, body):
        q = _text(body, "question", 1000)
        with self.model_lock:
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
                "train_file": str(out / "feedback_train.jsonl"), "instructions": EXPORT_STEPS,
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
def make_handler(app: Assistant):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OOAAssistant/1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj=None, raw=None, ctype="application/json; charset=utf-8"):
            b = raw if raw is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
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

        def _check_origin(self, method):
            """Blocks other web sites (CSRF) and DNS rebinding: only this page on 127.0.0.1/localhost may call us."""
            port = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if (self.headers.get("Host") or "").lower() not in allowed:
                raise ApiError(403, "forbidden: open the assistant at http://127.0.0.1:%d/" % port)
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() not in {"http://" + h for h in allowed}:
                raise ApiError(403, "forbidden: requests from other web sites are refused")
            if method == "POST":
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype != "application/json" or self.headers.get(CSRF_HEADER) != "1":
                    raise ApiError(403, f"forbidden: POST needs Content-Type application/json and {CSRF_HEADER}: 1")

        def _dispatch(self, method):
            self._check_origin(method)
            u = urlparse(self.path)
            path, q = u.path, {k: v[-1] for k, v in parse_qs(u.query).items()}
            if method == "GET":
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
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

        def _handle(self, method):
            try:
                self._dispatch(method)
            except ApiError as e:
                self._send(e.code, {"error": e.message})
            except Exception as e:  # noqa: BLE001 - one bad request must not kill the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

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


def make_server(app: Assistant, port: int = 8765) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    srv.daemon_threads = True
    return srv


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="OOA Assistant: local web app for the OOA flow-writing model")
    ap.add_argument("--model-dir", help="nanochat base dir (has tokenizer/ and chatsft_checkpoints/d8/)")
    ap.add_argument("--nanochat", help="nanochat source code folder (commit 92d63d4)")
    ap.add_argument("--platform", required=True, help="platform repo folder (has contracts/flow-schema/flow.v1.json)")
    ap.add_argument("--catalog", help="catalog.jsonl (default: MODEL_DIR/../sft/catalog.jsonl, else the repo copy)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data-dir", default="assistant_data", help="feedback, sandbox and exports go here")
    ap.add_argument("--fake-model", action="store_true", help="use a canned fake model (for trying the app / tests)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    catalog = args.catalog or default_catalog(args.model_dir)
    print("Loading the model (this can take a minute on CPU) ...", flush=True)
    model = load_model(args)
    app = Assistant(catalog, args.platform, args.data_dir, model)
    srv = make_server(app, args.port)
    print(f"OOA Assistant is running: open http://127.0.0.1:{srv.server_address[1]}/ in your browser "
          f"(Ctrl+C to stop). Data folder: {app.data}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
