"""The platform-facing contract of /v1/chat/completions (the platform's "OneOffice in-house model" connection and its
POST /api/ai/flow-assistant endpoint): the stable model id ooa-flow-1, the structured "ooa.result" next to the message
content, the optional bearer token OOA_API_KEY and the request size limit. Uses the FakeModel backend.
Run: python -m pytest tests/test_platform_endpoint.py   (needs the platform repo: OOA_PLATFORM, see test_app.py)
"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ASSISTANT = HERE.parent
ROOT = ASSISTANT.parent
for p in (ASSISTANT, ROOT / "serving", ROOT / "training"):
    sys.path.insert(0, str(p))
from model_backend import FakeModel  # noqa: E402

REPO = ROOT.parent
PLATFORM = os.environ.get("OOA_PLATFORM") or str(
    REPO.parent / "oneofficeautomation-private-limited" / "oneofficeautomation-platform")
CATALOG = os.environ.get("OOA_CATALOG") or str(ROOT / "data" / "catalog" / "catalog.jsonl")
SCHEMA = Path(PLATFORM) / "contracts" / "flow-schema" / "flow.v1.json"
KEY = "platform-key-0123456789abcdef"
GOOD = "Read a CSV file into a table, then write the table to an Excel sheet"

pytestmark = pytest.mark.skipif(not SCHEMA.exists(), reason="platform repo not found (set OOA_PLATFORM)")


@pytest.fixture(scope="module")
def S():
    import ooa_server
    return ooa_server


@pytest.fixture
def router(S):
    return S.Router(CATALOG, PLATFORM, S.Policy(allow_escalation=False, local_retries=1), local=FakeModel())


def body(text, model="ooa-flow-1", studio=False):
    content = ("Existing variables: []\nRequest: " + text) if studio else text
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def test_stable_model_id_is_listed_and_echoed(S, router):
    ids = [m["id"] for m in S.models_list()["data"]]
    assert ids[0] == "ooa-assistant" and ids[-1] == "ooa-flow-1" and "ooa-flow-assistant-fast" in ids
    code, out = S.chat_completion(router, body(GOOD))
    assert code == 200 and out["model"] == "ooa-flow-1"
    assert S.request_mode(body(GOOD)) == (None, None)            # the server's default mode
    code, out = S.chat_completion(router, body(GOOD, model="something-else"))
    assert out["model"] == "ooa-assistant"                         # unknown names: as before


def test_structured_result_for_a_flow(S, router):
    code, out = S.chat_completion(router, body(GOOD))
    r = out["ooa"]["result"]
    assert r["contract"] == "ooa-flow-result/1" and r["status"] == "flow"
    assert r["flow"] == json.loads(out["choices"][0]["message"]["content"])      # the same flow, as an object
    assert r["flow"]["kind"] and [s["action"] for s in r["flow"]["steps"]][:1] == ["ooa.csv/read"]
    assert r["verdict"]["verified"] is True and r["verdict"]["problems"] == [] and r["verdict"]["attempts"] >= 1
    assert r["question"] is None and isinstance(r["plan"], list)
    assert out["ooa"]["verified"] is True and "flow" not in out["ooa"]                # old fields unchanged


def test_structured_result_in_studio_shape(S, router):
    code, out = S.chat_completion(router, body(GOOD, studio=True))
    r = out["ooa"]["result"]
    assert r["status"] == "flow" and set(r["flow"]) >= {"variables", "steps", "notes"}


def test_structured_result_without_a_flow(S, router):
    code, out = S.chat_completion(router, body("read a csv file"))       # FakeModel cannot do this one
    r = out["ooa"]["result"]
    assert code == 200 and r["status"] == "no_flow" and r["flow"] is None
    assert r["verdict"]["verified"] is False and r["verdict"]["problems"] and r["verdict"]["reason"]


def test_structured_result_question(S):
    res = {"verified": False, "question": "Which folder?", "plan": [], "attempts": [], "problems": [],
           "flow": None, "reason": "asked"}
    r = S.structured_result(res)
    assert r["status"] == "question" and r["question"] == "Which folder?" and r["flow"] is None


def test_probe_has_no_result_and_needs_no_model(S, router):
    code, out = S.chat_completion(router, body("Answer with the single word OK."))
    assert out["choices"][0]["message"]["content"] == "OK" and "result" not in out["ooa"]


# ---------------------------------------------------------------------- serving/ooa_server.py's own HTTP server
@pytest.fixture
def bare(S, router):
    started = []

    def start(token=None):
        srv = S.serve(router, 0, token=token)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return srv.server_address[1]

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def call(port, method, path, payload=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode())
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    b = r.read()
    c.close()
    return r.status, json.loads(b)


def test_bare_server_token(bare):
    port = bare(token=KEY)
    assert call(port, "GET", "/health")[0] == 200                               # probes stay open
    code, out = call(port, "POST", "/v1/chat/completions", body(GOOD))
    assert code == 401 and out["error"]["type"] == "authentication_error"
    assert call(port, "GET", "/v1/models")[0] == 401
    assert call(port, "GET", "/v1/models", headers={"Authorization": "Bearer wrong"})[0] == 401
    code, out = call(port, "POST", "/v1/chat/completions", body(GOOD), {"Authorization": "Bearer " + KEY})
    assert code == 200 and out["ooa"]["result"]["status"] == "flow"


def test_bare_server_token_from_environment(S, router, monkeypatch, bare):
    monkeypatch.setenv("OOA_API_KEY", KEY)
    port = bare()
    assert call(port, "GET", "/v1/models")[0] == 401
    assert call(port, "GET", "/v1/models", headers={"Authorization": "Bearer " + KEY})[0] == 200


def test_bare_server_no_token_by_default(bare, monkeypatch):
    monkeypatch.delenv("OOA_API_KEY", raising=False)
    port = bare()
    assert call(port, "GET", "/v1/models")[0] == 200


def test_bare_server_body_limit(S, bare, monkeypatch):
    monkeypatch.delenv("OOA_API_KEY", raising=False)
    monkeypatch.setattr(S, "MAX_BODY", 1000)
    port = bare()
    code, out = call(port, "POST", "/v1/chat/completions", body("x" * 2000))
    assert code == 413 and "too large" in out["error"]["message"]
    assert call(port, "POST", "/v1/chat/completions", raw=b"[1]")[0] == 400


# ---------------------------------------------------------------------- assistant/app.py
def test_app_reads_the_platform_key_name():
    import app as A
    args = SimpleNamespace(token=None)
    assert A.resolve_token(args, {"OOA_API_KEY": KEY}) == KEY
    assert A.resolve_token(args, {"OOA_API_KEY": KEY, "OOA_ASSISTANT_TOKEN": "x" * 20}) == "x" * 20
    assert A.resolve_token(SimpleNamespace(token="y" * 20), {"OOA_API_KEY": KEY}) == "y" * 20
    assert A.resolve_token(args, {}) is None


def test_app_chat_carries_the_result():
    import app as A
    tmp = tempfile.mkdtemp(prefix="ooa_platform_test_")
    try:
        a = A.Assistant(CATALOG, PLATFORM, tmp, FakeModel())
        code, out = a.chat(body(GOOD))
        assert code == 200 and out["model"] == "ooa-flow-1" and out["ooa"]["result"]["status"] == "flow"
        assert out["ooa"]["version"]["catalog_sha256"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
