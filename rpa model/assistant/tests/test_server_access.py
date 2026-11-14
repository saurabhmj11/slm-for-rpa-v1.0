"""Tests for app.py as a server for programs (One Agent Studio's model connection): --host / token / allowed hosts,
/health, /ready, the version route, the bounded model queue (429) and the per-request time budget.
Run: python -m pytest tests/test_server_access.py   (needs the platform repo, see test_app.py: OOA_PLATFORM, OOA_CATALOG)
"""
import hashlib
import http.client
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ASSISTANT = HERE.parent
sys.path.insert(0, str(ASSISTANT))
import app as A  # noqa: E402
from model_backend import FakeModel  # noqa: E402

REPO = ASSISTANT.parent.parent
PLATFORM = os.environ.get("OOA_PLATFORM") or str(
    REPO.parent / "oneofficeautomation-private-limited" / "oneofficeautomation-platform")
CATALOG = os.environ.get("OOA_CATALOG") or str(ASSISTANT.parent / "data" / "catalog" / "catalog.jsonl")
SCHEMA = Path(PLATFORM) / "contracts" / "flow-schema" / "flow.v1.json"
TOKEN = "t0ken-for-tests-0123456789"
GOOD = "List the files in a folder"

pytestmark = pytest.mark.skipif(not SCHEMA.exists(), reason="platform repo not found (set OOA_PLATFORM)")


class SlowModel(FakeModel):
    """FakeModel that blocks in generate() until `go` is set (to hold the model lock), or sleeps `delay` seconds."""

    def __init__(self, delay=0.0, answer=None):
        super().__init__()
        self.go, self.started, self.delay, self.answer = threading.Event(), threading.Event(), delay, answer
        self.go.set()

    def generate(self, request, attempt=0, max_tokens=2500):
        self.started.set()
        self.go.wait(30)
        time.sleep(self.delay)
        out = super().generate(request, attempt)
        return self.answer if self.answer is not None else out


@pytest.fixture
def serve():
    started = []

    def start(model=None, host="127.0.0.1", token=None, allowed_hosts=(), **kw):
        tmp = tempfile.mkdtemp(prefix="ooa_access_test_")
        # model None: a FakeModel; model False: no model yet (still loading)
        app = A.Assistant(CATALOG, PLATFORM, tmp, FakeModel() if model is None else (model or None), **kw)
        srv = A.make_server(app, 0, host, token, allowed_hosts)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append((srv, tmp))
        return app, srv.server_address[1]

    yield start
    for srv, tmp in started:
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


def call(port, method, path, body=None, headers=None, addr="127.0.0.1"):
    c = http.client.HTTPConnection(addr, port, timeout=60)
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"} if method == "POST" else {}
    h.update(headers or {})
    c.request(method, path, body=data, headers={k: v for k, v in h.items() if v is not None})
    r = c.getresponse()
    b = r.read()
    c.close()
    return r.status, json.loads(b), r


def bearer(token=TOKEN):
    return {"Authorization": "Bearer " + token}


CHAT = {"model": "ooa-assistant", "messages": [{"role": "user", "content": GOOD}]}


# ---------------------------------------------------------------------- default: loopback, no token (as before)
def test_default_is_loopback_without_token(serve):
    _, port = serve()
    st, r, _ = call(port, "GET", "/v1/models")
    assert st == 200 and r["data"][0]["id"] == "ooa-assistant"
    st, r, _ = call(port, "POST", "/v1/chat/completions", CHAT)
    assert st == 200 and r["choices"][0]["finish_reason"] == "stop", r
    st, r, _ = call(port, "GET", "/v1/models", headers={"Host": "evil.example:%d" % port})   # DNS rebinding
    assert st == 403 and r["error"]["type"] == "permission_error", r
    st, _, _ = call(port, "POST", "/v1/chat/completions", CHAT, headers={"Origin": "https://evil.example"})
    assert st == 403
    st, _, _ = call(port, "POST", "/api/build", {"request": GOOD})        # the page's header is still required
    assert st == 403


def test_non_loopback_needs_a_token():
    tmp = tempfile.mkdtemp(prefix="ooa_access_test_")
    try:
        app = A.Assistant(CATALOG, PLATFORM, tmp, FakeModel())
        for host in ("0.0.0.0", "192.168.1.10", "::"):
            with pytest.raises(SystemExit, match="token"):
                A.make_server(app, 0, host, None)
        with pytest.raises(SystemExit, match="16"):
            A.make_server(app, 0, "127.0.0.1", "short")
        assert A.is_loopback("127.0.0.1") and A.is_loopback("::1") and A.is_loopback("localhost")
        assert A.is_loopback("::ffff:127.0.0.1") and not A.is_loopback("0.0.0.0") and not A.is_loopback("10.0.0.1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_parse_args_defaults():
    a = A.parse_args(["--platform", "p"])
    assert (a.host, a.allowed_hosts, a.token, a.max_waiting, a.time_budget) == ("127.0.0.1", [], None, 4, 120.0)


# ---------------------------------------------------------------------- token
def test_token_required_on_model_routes(serve):
    _, port = serve(token=TOKEN)
    for method, path, body in (("GET", "/v1/models", None), ("POST", "/v1/chat/completions", CHAT),
                               ("GET", "/api/assistant/v1/admin/version", None)):
        for h in ({}, bearer("wrong-token-wrong-token"), {"Authorization": TOKEN}, {"Authorization": "Basic " + TOKEN}):
            st, r, resp = call(port, method, path, body, headers=h)
            assert st == 401, (path, h, st, r)
            assert r["error"]["type"] == "authentication_error" and resp.getheader("WWW-Authenticate") == "Bearer"
        st, r, _ = call(port, method, path, body, headers=bearer())
        assert st == 200, (path, r)
    # with the token, any Host is fine (a server is reached by its own name); the token is what protects it
    st, _, _ = call(port, "GET", "/v1/models", headers={**bearer(), "Host": "assistant.internal:8765"})
    assert st == 200
    st, r, _ = call(port, "POST", "/v1/chat/completions", CHAT, headers={**bearer(), "Content-Type": "text/plain"})
    assert st == 400
    # the browser page keeps its own protection and needs no token on this machine
    st, _, _ = call(port, "GET", "/api/status")
    assert st == 200
    st, _, _ = call(port, "GET", "/api/status", headers={"Host": "evil.example"})
    assert st == 403
    st, _, _ = call(port, "POST", "/api/build", {"request": GOOD}, headers={A.CSRF_HEADER: "1"})
    assert st == 200


def test_allowed_hosts(serve):
    _, port = serve()
    assert call(port, "GET", "/api/status", headers={"Host": "assistant.lan:%d" % port})[0] == 403
    _, port = serve(allowed_hosts=["assistant.lan", "other.lan:9999"])
    assert call(port, "GET", "/api/status", headers={"Host": "assistant.lan:%d" % port})[0] == 200
    assert call(port, "GET", "/api/status", headers={"Host": "assistant.lan"})[0] == 200
    assert call(port, "GET", "/api/status", headers={"Host": "other.lan:9999"})[0] == 200
    assert call(port, "GET", "/api/status", headers={"Host": "other.lan:1"})[0] == 403
    assert call(port, "GET", "/api/status", headers={"Host": "evil.example"})[0] == 403
    # the page opened through an allowed name may POST (its Origin is that name); other sites still may not
    page = {"Host": "assistant.lan:%d" % port, A.CSRF_HEADER: "1"}
    flow = {"kind": "TaskBot", "id": "t", "name": "t", "steps": []}
    for origin, want in (("http://assistant.lan:%d" % port, 200), ("http://assistant.lan", 200),
                         ("https://evil.example", 403), ("null", 403), ("http://other.lan:1", 403)):
        st, r, _ = call(port, "POST", "/api/explain", {"flow": flow}, headers={**page, "Origin": origin})
        assert st == want, (origin, st, r)


def _lan_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))       # no packet is sent: this only picks the outgoing interface
        ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return None if A.is_loopback(ip) else ip


def test_bound_beyond_loopback_page_only_for_this_machine(serve):
    _, port = serve(host="0.0.0.0", token=TOKEN)
    assert call(port, "GET", "/api/status")[0] == 200                       # from this machine
    assert call(port, "GET", "/health")[0] == 200
    lan = _lan_address()
    if lan is None:
        pytest.skip("no non-loopback address to call from")
    st, r, _ = call(port, "GET", "/api/status", addr=lan, headers={"Host": "127.0.0.1:%d" % port})
    assert st == 403 and "this machine" in r["error"], r
    assert call(port, "GET", "/health", addr=lan)[0] == 200
    assert call(port, "GET", "/v1/models", addr=lan)[0] == 401
    assert call(port, "GET", "/v1/models", addr=lan, headers=bearer())[0] == 200


# ---------------------------------------------------------------------- health / ready / version
def test_health_ready_version(serve):
    app, port = serve(model=False, token=TOKEN)
    st, r, _ = call(port, "GET", "/health", headers={"Host": "anything"})
    assert st == 200 and r == {"ok": True}
    st, r, _ = call(port, "GET", "/ready")
    assert st == 503 and r["ready"] is False and r["missing"] == ["model (loading)"], r
    st, r, _ = call(port, "POST", "/v1/chat/completions", CHAT, headers=bearer())
    assert st == 503 and "loading" in r["error"]["message"], r
    app.model_error = "OSError: no checkpoint"
    st, r, _ = call(port, "GET", "/ready")
    assert st == 503 and "no checkpoint" in r["missing"][0]
    app.model_error = None
    app.set_model(FakeModel())
    st, r, _ = call(port, "GET", "/ready")
    assert st == 200 and r == {"ready": True, "missing": []}
    st, v, _ = call(port, "GET", "/api/assistant/v1/admin/version", headers=bearer())
    assert st == 200
    assert set(v) == {"version", "contract", "model_step", "catalog_actions", "catalog_sha256", "schema_sha256"}
    assert v["contract"] == "studio-assistant/v1" and v["model_step"] == 0 and v["version"] == "fake-step0"
    assert v["catalog_actions"] == len({json.loads(line)["id"] for line in open(CATALOG, encoding="utf-8") if line.strip()})
    assert v["catalog_sha256"] == hashlib.sha256(Path(CATALOG).read_bytes()).hexdigest()
    assert v["schema_sha256"] == hashlib.sha256(SCHEMA.read_bytes()).hexdigest()
    st, r, _ = call(port, "POST", "/v1/chat/completions", CHAT, headers=bearer())
    assert st == 200 and r["ooa"]["version"]["version"] == "fake-step0" and r["model"] == "ooa-assistant"


# ---------------------------------------------------------------------- busy / time budget
def test_queue_full_gets_429(serve):
    m = SlowModel()
    m.go.clear()
    app, port = serve(model=m, max_waiting=1)
    res = []
    t1 = threading.Thread(target=lambda: res.append(call(port, "POST", "/v1/chat/completions", CHAT)[0]))
    t1.start()
    assert m.started.wait(10)                                   # the first request holds the model
    t2 = threading.Thread(target=lambda: res.append(call(port, "POST", "/v1/chat/completions", CHAT)[0]))
    t2.start()
    for _ in range(200):                                         # the second one waits
        if app.waiting == 1:
            break
        time.sleep(0.02)
    assert app.waiting == 1
    st, r, resp = call(port, "POST", "/v1/chat/completions", CHAT)   # the third is refused at once
    assert st == 429 and r["error"]["type"] == "rate_limit_error" and resp.getheader("Retry-After") == str(A.RETRY_AFTER)
    st, r, resp = call(port, "POST", "/api/build", {"request": GOOD}, headers={A.CSRF_HEADER: "1"})
    assert st == 429 and "busy" in r["error"] and resp.getheader("Retry-After")
    probe = {"messages": [{"role": "user", "content": "Reply with the single word OK."}]}
    st, r, _ = call(port, "POST", "/v1/chat/completions", probe)    # a probe never waits for the model
    assert st == 200 and r["choices"][0]["message"]["content"] == "OK"
    m.go.set()
    t1.join(30)
    t2.join(30)
    assert sorted(res) == [200, 200]
    assert app.waiting == 0


def test_time_budget_stops_retries(serve):
    m = SlowModel(delay=0.3, answer="{not a flow")             # starts with "{": the Router would retry
    app, port = serve(model=m, time_budget=0.2)
    st, r, _ = call(port, "POST", "/api/build", {"request": "read a csv file"}, headers={A.CSRF_HEADER: "1"})
    assert st == 200 and r["verified"] is False and len(m.calls) == 1, (m.calls, r)
    m2 = SlowModel(answer="{not a flow")
    _, port2 = serve(model=m2)
    st, r, _ = call(port2, "POST", "/api/build", {"request": "read a csv file"}, headers={A.CSRF_HEADER: "1"})
    assert st == 200 and len(m2.calls) == 2                       # without the budget running out: one retry
