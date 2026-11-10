"""Tests for app.py: starts the server with --fake-model on a free port in a thread and calls every route over HTTP.
Run: python tests/test_app.py

Needs the platform repo (flow schema; reference runner optional) and a catalog:
  OOA_PLATFORM  platform repo folder (default: ../../oneofficeautomation-private-limited/oneofficeautomation-platform
                next to this repo's parent, i.e. the build environment's path)
  OOA_CATALOG   catalog.jsonl (default: rpa model/data/catalog/catalog.jsonl)
"""
import http.client, json, os, shutil, sys, tempfile, threading, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSISTANT = HERE.parent
sys.path.insert(0, str(ASSISTANT))
import app as A  # noqa: E402

REPO = ASSISTANT.parent.parent
PLATFORM = os.environ.get("OOA_PLATFORM") or str(
    REPO.parent / "oneofficeautomation-private-limited" / "oneofficeautomation-platform")
CATALOG = os.environ.get("OOA_CATALOG") or str(ASSISTANT.parent / "data" / "catalog" / "catalog.jsonl")
GOOD = "List the files in a folder"     # FakeModel answers this with a valid flow
BAD = "make my invoices nice"           # FakeModel answers garbage

STATE = {}


def setup():
    tmp = tempfile.mkdtemp(prefix="ooa_app_test_")
    args = A.parse_args(["--fake-model", "--platform", PLATFORM, "--catalog", CATALOG, "--data-dir", tmp, "--port", "0"])
    app = A.Assistant(args.catalog, args.platform, args.data_dir, A.load_model(args))
    srv = A.make_server(app, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    STATE.update(app=app, srv=srv, tmp=tmp, port=srv.server_address[1])


def teardown():
    STATE["srv"].shutdown()
    STATE["srv"].server_close()
    shutil.rmtree(STATE["tmp"], ignore_errors=True)


def call(method, path, body=None, raw=False, headers=None):
    """Like the page: POSTs carry Content-Type application/json and X-OOA-Assistant: 1. `headers` overrides them."""
    c = http.client.HTTPConnection("127.0.0.1", STATE["port"], timeout=60)
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    h = {"Content-Type": "application/json", A.CSRF_HEADER: "1"} if method == "POST" else {}
    h.update(headers or {})
    c.request(method, path, body=data, headers={k: v for k, v in h.items() if v is not None})
    r = c.getresponse()
    b = r.read()
    c.close()
    if raw:
        return r.status, b, r.getheader("Content-Type")
    return r.status, json.loads(b)


# ---------------------------------------------------------------------- tests
def test_binds_localhost_only():
    assert STATE["srv"].server_address[0] == "127.0.0.1"


def test_index_and_static():
    index = A.STATIC / "index.html"
    st, b, ct = call("GET", "/", raw=True)
    if index.exists():
        assert st == 200 and ct.startswith("text/html") and b == index.read_bytes()
        st2, b2, _ = call("GET", "/static/index.html", raw=True)
        assert st2 == 200 and b2 == b
    else:
        assert st == 404, st   # the UI file is written by another agent; the route must still answer cleanly


def test_static_path_traversal_refused():
    for p in ["/static/../app.py", "/static/../../README.md", "/static/%2e%2e/app.py", "/static/..%2fapp.py",
              "/static/", "/static/..\\app.py", "/static//etc/passwd", "/static/../tests/test_app.py"]:
        st, b, _ = call("GET", p, raw=True)
        assert st == 404, (p, st)
        assert b"import" not in b and b"root:" not in b, p


def test_status():
    st, s = call("GET", "/api/status")
    assert st == 200
    assert s["model"]["source"] == "fake"
    assert s["catalog_size"] > 0
    assert isinstance(s["runner"]["available"], bool) and isinstance(s["runner"]["supported_count"], int)
    assert set(s["feedback"]) >= {"accept", "edit", "reject", "total"}
    assert s["good_at"] and s["weak_at"] and "does not change" in s["learning"]


def test_build_verified():
    st, r = call("POST", "/api/build", {"request": GOOD})
    assert st == 200, r
    assert r["verified"] is True and isinstance(r["flow"], dict) and r["problems"] == []
    assert isinstance(r["repairs"], list) and isinstance(r["seconds"], (int, float))
    assert r["explanation"]["steps"] and r["explanation"]["steps"][0]["action"] == "ooa.file/list"
    assert set(r["can_run"]) >= {"ok", "unsupported", "risky", "problems", "inputs"}
    STATE["flow"] = r["flow"]


def test_build_unverified_is_honest():
    st, r = call("POST", "/api/build", {"request": BAD})
    assert st == 200
    assert r["verified"] is False and r["flow"] is None and r["problems"]
    assert r["explanation"] is None and r["can_run"] is None


def test_build_bad_input():
    assert call("POST", "/api/build", {})[0] == 400
    assert call("POST", "/api/build", {"request": "   "})[0] == 400
    assert call("POST", "/api/build", b"{not json")[0] == 400
    assert call("POST", "/api/build", [1, 2])[0] == 400


def test_explain():
    st, r = call("POST", "/api/explain", {"flow": STATE["flow"]})
    assert st == 200 and r["summary"] and r["steps"][0]["known"] is True
    st, r = call("POST", "/api/explain", {"flow": json.dumps(STATE["flow"])})   # a JSON string is accepted too
    assert st == 200 and r["steps"]
    assert call("POST", "/api/explain", {"flow": "{oops"})[0] == 400
    assert call("POST", "/api/explain", {})[0] == 400


def test_ask():
    st, r = call("POST", "/api/ask", {"question": "Which action reads a CSV file?"})
    assert st == 200 and "ooa.csv/read" in r["model_answer"]
    assert 0 < len(r["catalog_matches"]) <= 5 and "id" in r["catalog_matches"][0]
    assert call("POST", "/api/ask", {"question": ""})[0] == 400


def test_search_action_toolkits_lesson():
    st, r = call("GET", "/api/search?q=read+a+csv+file&k=3")
    assert st == 200 and 0 < len(r["results"]) <= 3
    assert call("GET", "/api/search?q=")[0] == 400
    st, card = call("GET", "/api/action?id=" + r["results"][0]["id"])
    assert st == 200 and card["id"] == r["results"][0]["id"] and "required" in card and "outputs" in card
    assert call("GET", "/api/action?id=no.such/action")[0] == 404
    st, t = call("GET", "/api/toolkits")
    assert st == 200 and t["toolkits"] and {"id", "name", "count"} <= set(t["toolkits"][0])
    st, les = call("GET", "/api/lesson?toolkit=" + t["toolkits"][0]["id"])
    assert st == 200 and les["intro"] and les["actions"]
    assert call("GET", "/api/lesson?toolkit=nope")[0] == 404


def test_quiz():
    for kind in ("which_action", "required_inputs", "which_toolkit", ""):
        st, q = call("GET", "/api/quiz?kind=" + kind)
        assert st == 200 and len(q["choices"]) == 4 and q["question"]
        assert set(q) == {"id", "kind", "question", "choices"}      # the answer is never sent
        st, c = call("POST", "/api/quiz/check", {"id": q["id"], "choice": 0})
        assert st == 200 and isinstance(c["correct"], bool) and 0 <= c["answer_index"] < 4 and c["explanation"]
        st, c2 = call("POST", "/api/quiz/check", {"id": q["id"], "choice": c["answer_index"]})
        assert st == 200 and c2["correct"] is True
    assert call("GET", "/api/quiz?kind=bogus")[0] == 400
    assert call("POST", "/api/quiz/check", {"id": "forged", "choice": 0})[0] == 400
    assert call("POST", "/api/quiz/check", {"id": q["id"], "choice": 9})[0] == 400
    assert call("POST", "/api/quiz/check", {"id": q["id"]})[0] == 400


def test_can_run_and_run():
    st, cr = call("POST", "/api/can_run", {"flow": STATE["flow"]})
    assert st == 200 and set(cr) >= {"ok", "unsupported", "risky", "problems", "inputs"}
    sandbox = Path(STATE["tmp"]) / "sandbox"
    (sandbox / "docs").mkdir(parents=True, exist_ok=True)
    (sandbox / "docs" / "a.txt").write_text("hi")
    st, r = call("POST", "/api/run", {"flow": STATE["flow"], "inputs": {"folder": "docs"}})
    assert st == 200 and "ok" in r
    if cr["ok"] and STATE["app"].runner.available:
        assert r["ok"] is True, r
    st, r = call("POST", "/api/run", {"flow": STATE["flow"], "inputs": {"folder": "../.."}})
    assert st == 200 and r["ok"] is False      # path guard refuses escapes
    assert call("POST", "/api/run", {"flow": STATE["flow"], "inputs": [1]})[0] == 400
    assert call("POST", "/api/can_run", {})[0] == 400


def test_feedback_progress_export():
    assert call("POST", "/api/export")[0] == 400           # nothing to export yet
    flow = STATE["flow"]
    st, r = call("POST", "/api/feedback", {"request": GOOD, "flow": flow, "verdict": "accept", "note": "looks right"})
    assert st == 200 and r["stored"] is True and r["counts"]["accept"] == 1
    edited = json.loads(json.dumps(flow))
    edited["name"] = "List files (edited)"
    assert call("POST", "/api/feedback", {"request": "List all files in a folder", "flow": flow, "verdict": "edit",
                                          "edited_flow": edited})[0] == 200
    assert call("POST", "/api/feedback", {"request": BAD, "flow": None, "verdict": "reject"})[0] == 200
    broken = {"kind": "flow", "id": "x", "name": "x", "steps": [{"id": "s1", "action": "no.such/thing"}]}
    assert call("POST", "/api/feedback", {"request": "broken one", "flow": broken, "verdict": "accept"})[0] == 200
    # bad feedback
    assert call("POST", "/api/feedback", {"request": GOOD, "flow": flow, "verdict": "maybe"})[0] == 400
    assert call("POST", "/api/feedback", {"request": GOOD, "flow": flow, "verdict": "edit"})[0] == 400
    assert call("POST", "/api/feedback", {"request": GOOD, "verdict": "accept"})[0] == 400

    rows = [json.loads(l) for l in open(Path(STATE["tmp"]) / "feedback.jsonl", encoding="utf-8")]
    assert len(rows) == 4
    assert set(rows[0]) >= {"prompt", "flow", "accepted", "edited_flow", "verdict", "time", "note"}
    assert rows[0]["accepted"] is True and rows[1]["accepted"] is False and rows[1]["edited_flow"]["name"].endswith("(edited)")

    st, p = call("GET", "/api/progress")
    assert st == 200 and p["counts"] == {"accept": 2, "edit": 1, "reject": 1, "total": 4}
    assert p["recent"][0]["prompt"] == "broken one" and set(p["recent"][0]) == {"prompt", "verdict", "time"}

    st, e = call("POST", "/api/export")
    assert st == 200, e
    assert (e["usable"], e["dropped"], e["rejected"]) == (2, 1, 1), e
    out = Path(e["path"])
    assert out.parent == Path(STATE["tmp"]).resolve() / "exports"
    train = [json.loads(l) for l in open(out / "feedback_train.jsonl", encoding="utf-8")]
    assert [t["messages"][0]["content"] for t in train] == [GOOD, "List all files in a folder"]
    assert json.loads(train[0]["messages"][1]["content"]) == flow           # the accepted flow, verbatim
    assert json.loads(train[1]["messages"][1]["content"])["name"] == "List files (edited)"   # edit wins
    assert (out / "rejected_prompts.jsonl").read_text(encoding="utf-8").count(BAD) == 1
    assert e["instructions"] and any("ooa-nanochat/sft" in s for s in e["instructions"])
    st, e2 = call("POST", "/api/export")                          # second export goes to a new folder
    assert st == 200 and e2["path"] != e["path"]


def _sandbox_file(name, text):
    sandbox = Path(STATE["tmp"]) / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / name).write_text(text)
    return sandbox / name


DELETE_FLOW = {"kind": "TaskBot", "id": "t", "name": "t", "variables": [],
               "steps": [{"id": "s1", "action": "ooa.file/delete", "args": {"path": "keep.txt"}}]}


def test_other_sites_cannot_post():
    """Fix: a web page on another site sent a no-preflight text/plain POST to /api/run and the flow ran."""
    keep = _sandbox_file("keep.txt", "x")
    body = {"flow": DELETE_FLOW, "inputs": {}, "allow_risky": True}
    for hdr in ({"Content-Type": "text/plain"},                           # CORS "simple" request
                {A.CSRF_HEADER: None},                                   # no custom header
                {"Origin": "https://evil.example"},                      # another site
                {"Host": "evil.example:%d" % STATE["port"]}):            # DNS rebinding
        st, r = call("POST", "/api/run", body, headers=hdr)
        assert st == 403, (hdr, st, r)
        assert keep.exists(), hdr
    st, r = call("POST", "/api/feedback", {"request": "x", "verdict": "reject"}, headers={"Content-Type": "text/plain"})
    assert st == 403, r
    assert call("GET", "/api/progress", headers={"Host": "evil.example"})[0] == 403
    # the page itself (same origin, localhost or 127.0.0.1) still works
    for origin in ("http://127.0.0.1:%d" % STATE["port"], "http://localhost:%d" % STATE["port"]):
        assert call("POST", "/api/explain", {"flow": DELETE_FLOW}, headers={"Origin": origin})[0] == 200
    assert call("GET", "/api/toolkits", headers={"Host": "localhost:%d" % STATE["port"]})[0] == 200


def test_allow_risky_must_be_true():
    """Fix: allow_risky was parsed with bool(), so the string "false" allowed a delete."""
    keep = _sandbox_file("keep.txt", "x")
    for v in ("false", "0", "no", 1, None):
        st, r = call("POST", "/api/run", {"flow": DELETE_FLOW, "inputs": {}, "allow_risky": v})
        assert st == 200 and r["ok"] is False, (v, r)
        assert keep.exists(), v
    if STATE["app"].runner.available:
        st, r = call("POST", "/api/run", {"flow": DELETE_FLOW, "inputs": {}, "allow_risky": True})
        assert st == 200 and r["ok"] is True and not keep.exists(), r


def test_bad_content_length():
    """Fix: Content-Length -1 skipped the size limit; a non-number gave a 500."""
    for n, want in (("-1", 400), ("abc", 400), (str(A.MAX_BODY + 1), 413)):
        c = http.client.HTTPConnection("127.0.0.1", STATE["port"], timeout=30)
        c.putrequest("POST", "/api/explain")
        c.putheader("Content-Type", "application/json")
        c.putheader(A.CSRF_HEADER, "1")
        c.putheader("Content-Length", n)
        c.endheaders()
        c.send(b'{"flow": {}}')
        r = c.getresponse()
        assert r.status == want, (n, r.status)
        c.close()


def test_build_not_tried_is_said_plainly():
    """Fix: a request the Router skips (over 6 steps) said 'failed the checker' though the model never ran."""
    calls = len(STATE["app"].model.calls)
    req = ("Read a CSV file, then sort it, then filter it, then dedupe it, then write it, then email it, "
           "then log it.")
    st, r = call("POST", "/api/build", {"request": req})
    assert st == 200, r
    assert r["verified"] is False and r["model_tried"] is False and r["attempts"] == [], r
    assert "not tried" in r["reason"] and "smaller" in r["reason"], r["reason"]
    assert len(STATE["app"].model.calls) == calls          # the model really was not called
    st, r = call("POST", "/api/build", {"request": "Read a text file. " * 40})   # over 500 characters
    assert st == 400 and "500" in r["error"], r
    st, r = call("POST", "/api/build", {"request": GOOD})
    assert r["model_tried"] is True


def test_malformed_flow_is_not_a_500():
    """Fix: non-list variables/elseIf/catch raised TypeError -> HTTP 500."""
    for f in ({"variables": 5, "steps": []},
              {"steps": [{"id": "a", "action": "core/if", "elseIf": 5}]},
              {"steps": [{"id": "a", "action": "core/try", "catch": 5}]}):
        for route in ("/api/explain", "/api/can_run"):
            st, r = call("POST", route, {"flow": f})
            assert st == 200, (route, f, r)


def test_not_found():
    assert call("GET", "/api/nope")[0] == 404
    assert call("POST", "/api/nope", {})[0] == 404


def test_parallel_builds_are_serialised():
    res = []
    ts = [threading.Thread(target=lambda: res.append(call("POST", "/api/build", {"request": GOOD})[0])) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert res == [200] * 4


if __name__ == "__main__":
    setup()
    failed = 0
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    try:
        for name, fn in tests:
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    finally:
        teardown()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
