"""UI test for static/index.html: a tiny stub API (canned JSON per route) + Playwright (Chromium).

    python tests/test_ui.py

Needs the Python `playwright` package and a Chromium it can launch (PLAYWRIGHT_BROWSERS_PATH). Without it the test prints
SKIP and exits 0, because the page is not needed to use the rest of the assistant. Drives every tab at 1280 px and 360 px,
asserts no console errors and no horizontal scroll, checks the JSON the page sends, and saves screenshots to tests/screens/.
"""
from __future__ import annotations

import json, sys, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "static"
SCREENS = HERE / "screens"

FLOW = {"kind": "TaskBot", "id": "csv_to_excel", "name": "CSV to Excel", "description": "CSV to Excel",
        "variables": [{"name": "csv_path", "type": "String", "direction": "input", "required": True},
                      {"name": "rows", "type": "Number", "direction": "input", "required": False, "default": 5},
                      {"name": "table", "type": "Table", "direction": "local"}],
        "steps": [{"id": "s1", "action": "core/loop", "args": {}, "children": [
            {"id": "s2", "action": "ooa.csv/read", "args": {"path": "$csv_path"}, "out": {"table": "table"}},
            {"id": "s3", "action": "ooa.file/delete", "args": {"path": "old.txt"}}]}]}
EXPLANATION = {
    "summary": "This flow reads a CSV file into a table inside a loop, and deletes an old file.",
    "inputs": [{"name": "csv_path", "type": "String", "description": "the CSV file to read"}],
    "outputs": [],
    "steps": [
        {"id": "s1", "depth": 0, "action": "core/loop", "title": "Core -> Loop", "what": "Step 1: Repeat the steps inside.",
         "args": [], "outs": [], "known": True, "block": None},
        {"id": "s2", "depth": 1, "action": "ooa.csv/read", "title": "CSV/TXT -> Read to table",
         "what": "Step 2 (inside the loop): Read a CSV file into a table. Reads from: $csv_path. Saves the table in: table.",
         "args": [{"name": "path", "value": "$csv_path", "meaning": "the file to read"}],
         "outs": [{"name": "table", "variable": "table"}], "known": True, "block": "loop"},
        {"id": "s3", "depth": 1, "action": "ooa.file/delete", "title": "File -> Delete a very long action title that wraps",
         "what": "Step 3 (inside the loop): Delete a file.", "args": [{"name": "path", "value": "old.txt", "meaning": ""}],
         "outs": [], "known": False, "block": "loop"}]}
CAN_RUN = {"ok": True, "unsupported": [], "risky": ["ooa.file/delete"], "problems": [],
           "inputs": [{"name": "csv_path", "type": "String", "required": True, "default": None, "description": "CSV file"},
                      {"name": "rows", "type": "Number", "required": False, "default": 5, "description": ""}]}
CARD = {"id": "ooa.csv/read", "name": "Read to table", "toolkit": "CSV/TXT", "toolkit_id": "ooa.csv", "kind": "action",
        "description": "Read a CSV file into a table.", "notes": "Time limit 60 s.",
        "required": [{"name": "path", "type": "String", "description": "File path"}],
        "optional": [{"name": "delimiter", "type": "String", "description": "Separator character"}],
        "outputs": [{"name": "table", "type": "Table", "description": "The data"}]}
SHORT = {k: CARD[k] for k in ("id", "name", "toolkit", "description", "kind")}

RECEIVED: list[tuple[str, dict]] = []   # (path, json body) of every POST
POST_HEADERS: list[tuple[str, str | None, str | None]] = []   # (path, X-OOA-Assistant, Content-Type) of every POST


def canned(method, path, q, body):
    if method == "GET":
        return {
            "/api/status": {"model": {"device": "cpu", "params_M": 42.0, "step": 2000, "source": "sft"},
                            "catalog_size": 1351, "runner": {"available": True, "supported_count": 120},
                            "feedback": {"accept": 1, "edit": 0, "reject": 1, "total": 2},
                            "good_at": ["Step-by-step requests."], "weak_at": ["Everyday wording."],
                            "learning": "The model does not change while you use it."},
            "/api/search": {"results": [SHORT, dict(SHORT, id="ooa.csv/write", name="Write table to CSV")]},
            "/api/action": CARD,
            "/api/toolkits": {"toolkits": [{"id": "ooa.csv", "name": "CSV/TXT", "count": 6},
                                           {"id": "ooa.excel", "name": "Excel", "count": 40}]},
            "/api/lesson": {"toolkit": "CSV/TXT", "toolkit_id": "ooa.csv", "intro": "The CSV/TXT toolkit has 6 actions.",
                            "actions": [SHORT]},
            "/api/quiz": {"id": "opaque-token", "kind": q.get("kind", "which_action"),
                          "question": "Which action reads a CSV file into a table?",
                          "choices": ["Read to table", "Write table", "Open workbook", "Delete file"]},
            "/api/progress": {"counts": {"accept": 1, "edit": 0, "reject": 1, "total": 2},
                              "recent": [{"prompt": "List the files in a folder.", "verdict": "accept",
                                          "time": "2026-11-10T10:00:00"},
                                         {"prompt": "sort out my invoices", "verdict": "reject",
                                          "time": "2026-11-10T09:00:00"}],
                              "learning": "The model does not change while you use it."},
        }.get(path)
    if path == "/api/build":
        if "email it" in body.get("request", "").lower():   # the app skips the model for > 6 steps
            return {"verified": False, "model_tried": False, "flow": None, "problems": [], "repairs": [],
                    "seconds": 0.01, "attempts": [], "explanation": None, "can_run": None,
                    "reason": "The model was not tried: requests with more than 6 steps are skipped. "
                              "Split it into smaller requests."}
        time.sleep(2.3)   # long enough to see the elapsed counter tick and the button disabled
        if "csv" in body.get("request", "").lower():
            return {"verified": True, "flow": FLOW, "problems": [], "repairs": ["added missing id to step s3"],
                    "seconds": 2.3, "attempts": [{"source": "local", "ok": True, "tokens": 280, "stop": "end", "seconds": 2.3}], "reason": None,
                    "explanation": EXPLANATION, "can_run": CAN_RUN}
        return {"verified": False, "flow": None, "problems": ["output is not valid JSON", "unknown action ooa.x/y"],
                "repairs": [], "seconds": 2.3, "attempts": [], "explanation": None, "can_run": None,
                "reason": "the local model could not produce a verified flow"}
    return {
        "/api/explain": EXPLANATION,
        "/api/ask": {"model_answer": "Use `ooa.csv/read`.", "catalog_matches": [SHORT],
                     "note": "The catalog matches are the source of truth; the model's answer can be wrong."},
        "/api/quiz/check": {"correct": body.get("choice") == 0, "answer_index": 0,
                            "explanation": "Read to table (ooa.csv/read) reads a CSV file into a table."},
        "/api/can_run": CAN_RUN,
        "/api/run": {"ok": True, "outputs": {"count": 3}, "seconds": 0.42, "error": None,
                     "events": [{"type": "step_started", "step": "s2", "message": "Read to table"},
                                {"type": "step_finished", "step": "s2", "message": "ok"}],
                     "files": ["data.csv", "out/report.xlsx"]},
        "/api/feedback": {"stored": True, "counts": {"accept": 2, "edit": 0, "reject": 1, "total": 3},
                          "learning": "The model does not change while you use it."},
        "/api/export": {"usable": 2, "dropped": 1, "rejected": 1, "path": "/data/exports/20261110-100000",
                        "train_file": "/data/exports/20261110-100000/feedback_train.jsonl",
                        "instructions": ["Upload feedback_train.jsonl to Drive ooa-nanochat/sft/.",
                                         "Run the training notebook."],
                        "note": "Dropped rows failed the verifier and are never trained on."},
    }.get(path)


class StubHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _api(self, method):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        body = {}
        if method == "POST":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            RECEIVED.append((u.path, body))
            POST_HEADERS.append((u.path, self.headers.get("X-OOA-Assistant"), self.headers.get("Content-Type")))
        res = canned(method, u.path, q, body)
        return self._json(200, res) if res is not None else self._json(404, {"error": "not found"})

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._api("GET")
        return super().do_GET()

    def do_POST(self):
        return self._api("POST")


def last_post(path):
    return next(b for p, b in reversed(RECEIVED) if p == path)


def check_layout(page, label, shot):
    sw = page.evaluate("document.documentElement.scrollWidth")
    cw = page.evaluate("document.documentElement.clientWidth")
    assert sw <= cw, f"{label}: horizontal scroll ({sw} > {cw})"
    page.screenshot(path=str(SCREENS / f"{shot}.png"), full_page=True)


def drive(browser, base, width):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    w = str(width)
    page.goto(base + "/")
    page.wait_for_function("document.querySelector('#status-line').textContent.includes('catalog: 1351')")

    # ---- Build: unverified answer must show the warning with problems, no flow
    page.fill("#build-request", "sort out my invoices")
    page.click("#build-btn")
    page.wait_for_timeout(1300)
    assert page.is_disabled("#build-btn"), "Build button must be disabled while building"
    assert not page.is_hidden("#build-progress")
    assert page.inner_text("#build-elapsed").strip() in ("1 s", "2 s"), page.inner_text("#build-elapsed")
    page.wait_for_selector("#build-result .box.warn")
    warn = page.inner_text("#build-result .box.warn")
    assert "Not verified" in warn and "unknown action ooa.x/y" in warn, warn
    assert page.query_selector("#build-result ol.steps") is None and page.query_selector("#try-run") is None
    assert page.query_selector("#fb-accept") is None, "an unverified answer must not offer Accept"
    assert page.is_enabled("#build-btn")
    check_layout(page, f"build-unverified@{w}", f"build_unverified_{w}")
    page.click("#fb-reject")
    page.wait_for_selector("#build-result .box.ok")
    fb = last_post("/api/feedback")
    assert fb["verdict"] == "reject" and fb["flow"] is None and fb["request"] == "sort out my invoices", fb

    # ---- Build: a request the model is not tried on says so (not "failed the checker")
    page.fill("#build-request", "Read a CSV file, then sort it, then filter it, then dedupe it, then write it, "
                                "then email it, then log it.")
    page.click("#build-btn")
    page.wait_for_selector("#build-result .box.warn >> text=not tried")
    warn = page.inner_text("#build-result .box.warn")
    assert "failed the checker" not in warn and "smaller requests" in warn, warn
    assert page.get_attribute("#build-request", "maxlength") == "500"
    chips = page.inner_text("#build-chips")
    assert "browser" not in chips.lower() and "Read a text file. The flow takes path as input." in chips, chips

    # ---- Build: verified answer via an example chip
    page.click("#build-chips button:has-text('CSV')")   # the stub verifies requests that mention CSV
    page.click("#build-btn")
    page.wait_for_selector("#build-result ol.steps", timeout=10000)
    assert "Verified" in page.inner_text("#build-result .box.ok")
    assert "Attempt 1: passed the checker; stopped at the natural end, 280 tokens" in page.text_content("#build-result")
    lis = page.query_selector_all("#build-result ol.steps > li")
    assert len(lis) == 3
    m0 = lis[0].evaluate("e => parseFloat(getComputedStyle(e).marginLeft)")
    m1 = lis[1].evaluate("e => parseFloat(getComputedStyle(e).marginLeft)")
    assert m1 > m0, "nested step must be indented"
    assert "inside the loop" in lis[1].inner_text() and "not in the catalog" in lis[2].inner_text()
    check_layout(page, f"build-verified@{w}", f"build_verified_{w}")
    page.click("#fb-edit-btn")
    page.fill("#fb-edit", "{ not json")
    page.click("#fb-save-edit")
    page.wait_for_selector("#build-result .box.bad")
    page.fill("#fb-edit", json.dumps(dict(FLOW, name="Edited")))
    page.fill("#fb-note", "renamed")
    page.click("#fb-save-edit")
    page.wait_for_selector("#build-result .box.ok >> text=Saved (edit)")
    fb = last_post("/api/feedback")
    assert fb["verdict"] == "edit" and fb["edited_flow"]["name"] == "Edited" and fb["flow"] == FLOW \
        and fb["note"] == "renamed", fb
    assert page.is_disabled("#fb-accept"), "feedback buttons are disabled once a verdict is saved"

    # ---- Run (from "Try running it")
    page.click("#try-run")
    assert page.get_attribute("#tab-run", "aria-selected") == "true"
    page.wait_for_selector("#run-form-card:not([hidden])")
    assert page.is_disabled("#run-btn"), "risky flow: Run disabled until confirmed"
    assert page.input_value("#run-in-1") == "5"
    page.fill("#run-in-0", "data.csv")
    page.check("#run-allow-risky")
    assert page.is_enabled("#run-btn")
    page.click("#run-btn")
    page.wait_for_selector("#run-result .box.ok")
    run = last_post("/api/run")
    assert run["inputs"] == {"csv_path": "data.csv", "rows": 5} and run["allow_risky"] is True and run["flow"] == FLOW, run
    assert "out/report.xlsx" in page.inner_text("#run-result")
    page.click("#run-explain")
    page.wait_for_selector("#run-explanation ol.steps")
    assert "has not been checked" in page.inner_text("#explain-unchecked")
    check_layout(page, f"run@{w}", f"run_{w}")
    page.fill("#run-flow", '{"kind": "TaskBot"}')
    assert page.is_disabled("#run-btn"), "editing the flow must require a new check"
    page.click("#run-check")
    page.wait_for_selector("#run-form-card:not([hidden])")
    assert last_post("/api/can_run")["flow"] == {"kind": "TaskBot"}

    # ---- Learn
    page.click("#tab-learn")
    page.fill("#search-q", "read csv")
    page.click("#search-btn")
    page.wait_for_selector("#search-results li")
    page.click("#search-results li button >> nth=0")
    page.wait_for_selector("#action-card h2")
    assert "Required inputs" in page.inner_text("#action-card") and "delimiter" in page.inner_text("#action-card")
    page.wait_for_function("document.querySelectorAll('#toolkit-select option').length === 3")
    page.select_option("#toolkit-select", "ooa.csv")
    page.click("#lesson-btn")
    page.wait_for_selector("#lesson li")
    page.fill("#ask-q", "Which action reads a csv?")
    page.click("#ask-btn")
    page.wait_for_selector("#ask-result .grid2")
    assert "source of truth" in page.inner_text("#ask-result")
    check_layout(page, f"learn@{w}", f"learn_{w}")

    # ---- Practice (score persists in localStorage)
    page.click("#tab-practice")
    page.click("#quiz-reset")
    page.select_option("#quiz-kind", "which_action")
    page.click("#quiz-new")
    page.wait_for_selector("#quiz-choices button")
    page.click("#quiz-choices button >> nth=1")
    page.wait_for_selector("#quiz-feedback .box.bad")
    assert last_post("/api/quiz/check") == {"id": "opaque-token", "choice": 1}
    assert "right" in page.get_attribute("#quiz-choices button >> nth=0", "class")
    page.click("#quiz-feedback button")
    page.wait_for_selector("#quiz-choices button:not([disabled])")
    page.click("#quiz-choices button >> nth=0")
    page.wait_for_selector("#quiz-feedback .box.ok")
    assert page.inner_text("#quiz-score") == "1 / 2"
    check_layout(page, f"practice@{w}", f"practice_{w}")

    # ---- Progress
    page.click("#tab-progress")
    page.wait_for_selector("#progress-body .tiles")
    page.click("#export-btn")
    page.wait_for_selector("#export-result ol li")
    assert "2 usable" in page.inner_text("#export-result")
    check_layout(page, f"progress@{w}", f"progress_{w}")

    # ---- reload: last tab and quiz score come back from localStorage
    page.reload()
    page.wait_for_function("document.querySelector('#tab-progress').getAttribute('aria-selected') === 'true'")
    assert page.inner_text("#quiz-score") == "1 / 2"

    # ---- keyboard: arrow keys move between tabs, focus is visible
    page.focus("#tab-progress")
    page.keyboard.press("ArrowRight")
    assert page.get_attribute("#tab-build", "aria-selected") == "true"
    outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
    assert outline != "none", "focused tab must show a focus outline"

    assert POST_HEADERS and all(h == "1" and (ct or "").startswith("application/json") for _, h, ct in POST_HEADERS), \
        [x for x in POST_HEADERS if x[1] != "1" or not (x[2] or "").startswith("application/json")]
    assert not errors, f"console errors at {w}px: {errors}"
    ctx.close()


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SKIP test_ui: the Python 'playwright' package is not installed")
        return 0
    SCREENS.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                for width in (1280, 360):
                    drive(browser, base, width)
                    print(f"ok  test_ui at {width}px")
            finally:
                browser.close()
    finally:
        srv.shutdown()
    print(f"test_ui passed; screenshots in {SCREENS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
