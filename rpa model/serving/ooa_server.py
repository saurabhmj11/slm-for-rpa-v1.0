"""OOA assistant server: one OpenAI-compatible endpoint, a router, a verifier, and client data-policy switches.

   request -> [mask PII for cloud] -> router
        simple?  -> LOCAL model (our own) -> repair -> verify -> ok? return
        else / failed -> CLOUD model (Claude) if the client policy allows -> repair -> verify (one retry with the errors)
        nothing verified -> honest error (never returns an unchecked flow as if it were fine)
   POST /v1/chat/completions   OpenAI-style; the last user message is the request; the answer is the flow JSON
   POST /v1/feedback           {"prompt","flow","accepted","edited_flow"}  -> feedback.jsonl (only if the policy allows)
   GET  /health

Run:  python ooa_server.py --catalog catalog.jsonl --platform PLATFORM_REPO --policy policy.json [--local nanochat] [--port 8080]
Local backend: any object with generate(request, attempt) -> str (see backends.py). Cloud: Anthropic Messages API (env ANTHROPIC_API_KEY).
Default policy is LOCAL-ONLY and stores nothing.
"""
import argparse, json, os, re, sys, time, threading
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "training"))
import ooa_reward as R
import retrieval
import repair as RP


@dataclass
class Policy:
    allow_escalation: bool = False       # may a request leave for the cloud model at all
    mask_pii: bool = True                # mask emails, phones, card-like numbers before anything leaves
    store_logs: bool = False             # keep a log of requests and results (masked)
    allow_training_on_data: bool = False  # may logged data / feedback be used to train models
    local_retries: int = 2
    max_clauses_local: int = 6           # longer / more complex requests go to the cloud straight away
    cloud_model: str = "claude-sonnet-5-5"
    cloud_max_tokens: int = 6000
    log_dir: str = "ooa_logs"

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


class Router:
    def __init__(self, catalog, platform, policy, local=None, cloud=None):
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
        self.lock = threading.Lock()

    # -- one attempt: parse -> repair -> verify ---------------------------------------------------
    def check(self, raw):
        text = extract_json(raw)
        try:
            flow = json.loads(text)
        except Exception:
            return None, R.problems(text)
        flow, notes = RP.repair(flow, self.cat)
        mini = json.dumps(flow, separators=(",", ":"), ensure_ascii=False)
        return (mini, notes) if not R.problems(mini) else (None, R.problems(mini))

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

    def run(self, request):
        t0, tried, errors = time.time(), [], []
        p = self.policy
        simple = clauses(request) <= p.max_clauses_local and len(request) <= 500
        if self.local is not None and simple:
            for attempt in range(1 + p.local_retries):
                try:
                    raw = self.local.generate(request, attempt)
                except Exception as ex:
                    errors = [f"local model error: {ex}"]
                    break
                flow, info = self.check(raw)
                tried.append({"source": "local", "ok": flow is not None})
                if flow:
                    return self.done(request, flow, "local", tried, info, t0)
                errors = info
        elif self.local is not None:
            errors = ["request looks complex; skipped the local model"]
        if self.cloud is not None and p.allow_escalation:
            req = mask(request) if p.mask_pii else request
            err = [mask(e) if p.mask_pii else e for e in errors if not e.startswith("request looks")]
            for attempt in range(2):
                system, user = self.cloud_prompt(req, err)
                try:
                    raw = self.cloud.generate(system, user)
                except Exception as ex:
                    err = [f"cloud call failed: {ex}"]
                    tried.append({"source": "cloud", "ok": False, "error": str(ex)[:120]})
                    break
                flow, info = self.check(raw)
                tried.append({"source": "cloud", "ok": flow is not None})
                if flow:
                    return self.done(request, flow, "cloud", tried, info, t0)
                err = info
            errors = err
        reason = ("no model produced a flow that passed verification" if (self.cloud and p.allow_escalation)
                  else "the local model could not produce a verified flow, and cloud escalation is disabled by this client's policy")
        return self.done(request, None, "none", tried, errors or [reason], t0, reason=reason)

    def done(self, request, flow, source, tried, info, t0, reason=None):
        res = {"flow": flow, "source": source, "verified": flow is not None, "attempts": tried,
               "repairs" if flow else "problems": info, "seconds": round(time.time() - t0, 3)}
        if reason:
            res["reason"] = reason
        if self.policy.store_logs:
            self.log("requests.jsonl", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "request": mask(request),
                                        "source": source, "verified": res["verified"], "flow": flow,
                                        "training_ok": self.policy.allow_training_on_data})
        return res

    def log(self, name, row):
        d = Path(self.policy.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        with self.lock, open(d / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def feedback(self, row):
        if not self.policy.store_logs:
            return {"stored": False, "reason": "policy: store_logs is off"}
        row = {**row, "prompt": mask(row.get("prompt", "")), "training_ok": self.policy.allow_training_on_data,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.log("feedback.jsonl", row)
        return {"stored": True, "usable_for_training": self.policy.allow_training_on_data}


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
                msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
                if not msgs:
                    return self._send(400, {"error": "send at least one user message"})
                res = router.run(msgs[-1]["content"])
                content = res["flow"] if res["verified"] else json.dumps({"error": res["reason"], "problems": res.get("problems", [])})
                return self._send(200, {"id": "ooa-" + str(int(time.time() * 1000)), "object": "chat.completion",
                                        "created": int(time.time()), "model": "ooa-assistant",
                                        "choices": [{"index": 0, "finish_reason": "stop" if res["verified"] else "length",
                                                     "message": {"role": "assistant", "content": content}}],
                                        "ooa": {k: v for k, v in res.items() if k != "flow"}})
            if self.path == "/v1/feedback":
                return self._send(200, router.feedback(body))
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
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    pol = Policy.load(a.policy)
    local = None
    if a.local == "nanochat":
        from backends import NanochatBackend
        local = NanochatBackend(a.source)
    cloud = CloudBackend(pol.cloud_model, pol.cloud_max_tokens) if pol.allow_escalation else None
    r = Router(a.catalog, a.platform, pol, local, cloud)
    print(f"OOA assistant on http://127.0.0.1:{a.port}  | local model: {bool(local)} | cloud escalation: {pol.allow_escalation}")
    serve(r, a.port).serve_forever()
