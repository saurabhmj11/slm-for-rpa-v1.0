"""Tests of the Router's layers with scripted local models and a stub OpenAI-compatible server (python http.server):
retrieval context, plan-first answers, plan coverage, self-correction prompts, clarifying questions, building a long
plan in parts, safety warnings, escalation to any OpenAI-compatible endpoint, /v1/feedback, and that a model trained
before context and plans still works.
Run: python test_layers.py PLATFORM_REPO   (or OOA_PLATFORM=... python -m pytest test_layers.py; needs jsonschema)"""
import json, os, sys, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "training")); sys.path.insert(0, str(HERE.parent / "assistant"))
import ooa_server as S, ooa_format as F
from model_backend import FakeModel

PLATFORM = next((a for a in sys.argv[1:] if os.path.isdir(a)), None) or os.environ.get("OOA_PLATFORM", "")
CATALOG = str(HERE.parent / "data/catalog/catalog.jsonl")
CAT = {e["id"]: e for e in (json.loads(l) for l in open(CATALOG, encoding="utf-8"))}
CSV_TO_EXCEL = FakeModel.FLOWS["read a csv file into a table, then write the table to an excel sheet"]
CSV_REQ = "Read a CSV file into a table, then write the table to an Excel sheet"
LIST_REQ = "List the files in a folder"
PART2 = {"kind": "TaskBot", "id": "p2", "name": "p2",
         "variables": [{"name": "folder", "type": "String", "direction": "input", "required": True},
                       {"name": "files", "type": "List", "direction": "local"},
                       {"name": "path", "type": "String", "direction": "input", "required": True},
                       {"name": "text", "type": "String", "direction": "output"}],
         "steps": [{"id": "s1", "action": "ooa.file/list", "description": "List the files", "args": {"folder": "$folder"}, "out": {"files": "files"}},
                   {"id": "s2", "action": "ooa.file/read_text", "description": "Read the text file", "args": {"path": "$path"}, "out": {"text": "text"}},
                   {"id": "s3", "action": "core/log", "description": "Log the text", "args": {"message": "${text}"}}]}


class Scripted:
    """Local model stand-in: answer(request_part, prompt, attempt) -> text; records every prompt it was given."""
    def __init__(self, answer, info=None):
        self.answer, self.prompts = answer, []
        if info is not None:
            self.info = info

    def generate(self, prompt, attempt=0):
        self.prompts.append(prompt)
        return self.answer(F.request_of(prompt), prompt, attempt)


class Cloud:
    def __init__(self, outs):
        self.outs, self.seen = list(outs), []

    def generate(self, system, user):
        self.seen.append((system, user))
        return self.outs[min(len(self.seen) - 1, len(self.outs) - 1)]


def router(local, cloud=None, examples=None, prompt_format="context", **policy):
    return S.Router(CATALOG, PLATFORM, S.Policy(**policy), local, cloud, examples, prompt_format)


def plan_answer(flow, plan=None):
    return "Plan:\n" + "\n".join(plan or F.plan_from_flow(flow, CAT)) + "\nFlow:\n" + F.compact(flow)


def studio_body(request, existing="[]"):
    return {"model": "ooa-assistant", "messages": [{"role": "system", "content": "catalog ..."},
            {"role": "user", "content": "Existing variables: " + existing + "\nRequest: " + request}]}


# ---------------------------------------------------------------------------------------------- tests
def test_context_prompt_and_plan_answer(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    ex = F.ExampleIndex.from_flows([("Read a csv file and save it as excel", CSV_TO_EXCEL)])
    ex.save(d / "examples.jsonl")
    m = Scripted(lambda req, p, a: plan_answer(CSV_TO_EXCEL))
    r = router(m, examples=str(d / "examples.jsonl"), prompt_format=None)          # auto: examples -> context
    assert r.format() == "context"
    code, out = S.chat_completion(r, studio_body(CSV_REQ, json.dumps([{"name": "folder", "type": "String", "direction": "input"}])))
    p = m.prompts[0]
    assert p.startswith(CSV_REQ + "\n" + F.CONTEXT_HEADER) and "Variables: folder (String)" in p
    assert "- ooa.csv/read:" in p and "- core/loop:" in p and "Similar: Read a csv file and save it as excel =>" in p
    ans = json.loads(out["choices"][0]["message"]["content"])
    assert out["ooa"]["verified"] and [s["action"] for s in ans["steps"]][0] == "ooa.csv/read"
    assert out["ooa"]["plan"][0].endswith("(ooa.csv/read)") and ans["warnings"] == []


def test_old_model_plain_requests_still_work():
    m = FakeModel()                                     # answers with flow JSON only, as models before plans did
    r = router(m, prompt_format=None)                   # auto without examples -> plain
    assert r.format() == "plain"
    res = r.run(LIST_REQ)
    assert res["verified"] and m.calls == [(LIST_REQ, 0)] and res["plan"] == []
    # a JSON-only answer also works when the model is given context
    res = router(FakeModel()).run(LIST_REQ)
    assert res["verified"] and res["source"] == "local"


def test_plan_coverage_triggers_self_correction():
    flow_no_close = json.loads(json.dumps(CSV_TO_EXCEL))
    plan = F.plan_from_flow(CSV_TO_EXCEL, CAT) + ["5. Email the workbook (ooa.email/send)"]

    def answer(req, prompt, attempt):
        return plan_answer(flow_no_close, plan) if attempt == 0 else plan_answer(CSV_TO_EXCEL)
    m = Scripted(answer)
    res = router(m, local_retries=1).run(CSV_REQ)
    assert res["verified"] and len(m.prompts) == 2, res
    second = m.prompts[1]
    assert F.PREVIOUS_HEADER in second and F.PROBLEMS_HEADER in second
    assert "- The plan names ooa.email/send but no step of the flow uses it." in second
    assert '"action":"ooa.csv/read"' in second.split(F.PREVIOUS_HEADER)[1]
    # the old path (plain prompts) only retries with the same request
    m2 = Scripted(lambda req, p, a: "{not json")
    router(m2, local_retries=1, prompt_format="plain").run(CSV_REQ)
    assert m2.prompts == [CSV_REQ, CSV_REQ]


def test_clarifying_question():
    m = FakeModel(style="plan")
    r = router(m)
    code, out = S.chat_completion(r, studio_body("Automate my work"))
    ans = json.loads(out["choices"][0]["message"]["content"])
    assert ans["steps"] == [] and ans["notes"] == FakeModel.QUESTIONS["automate my work"] and ans["warnings"] == []
    assert out["ooa"]["question"] and len(m.calls) == 1                 # asked once, no retries
    code, out = S.chat_completion(r, {"messages": [{"role": "user", "content": "automate my work"}]})
    assert out["choices"][0]["message"]["content"] == FakeModel.QUESTIONS["automate my work"]
    # FakeModel's plan style also builds flows
    assert router(FakeModel(style="plan", catalog=CAT)).run(CSV_REQ)["verified"]


def test_long_plan_is_built_in_parts():
    whole_plan = F.plan_from_flow(CSV_TO_EXCEL, CAT) + [l.replace(l.split(".")[0], str(int(l.split(".")[0]) + 4), 1)
                                                        for l in F.plan_from_flow(PART2, CAT)]
    chunks = F.chunk_plan(whole_plan)
    assert len(whole_plan) == 7 and [len(c) for c in chunks] == [4, 3]
    part_req = {F.chunk_request(chunks[0]): CSV_TO_EXCEL, F.chunk_request(chunks[1]): PART2}
    big = "Read the CSV into Excel, then list the folder, read the text file and log it"

    def answer(req, prompt, attempt):
        if req == big:
            return "Plan:\n" + "\n".join(whole_plan) + "\nFlow:\n{\"kind\":\"TaskBot\",\"steps\":["      # cut off
        return plan_answer(part_req[req])
    m = Scripted(answer)
    res = router(m, local_retries=2).run(big)
    assert res["verified"] and res.get("built_in_parts"), res
    flow = json.loads(res["flow"])
    assert F.actions_of(flow) == F.actions_of(CSV_TO_EXCEL) + F.actions_of(PART2)
    ids = [s["id"] for s, _ in F.walk(flow["steps"])]
    assert len(ids) == len(set(ids)) == 7
    names = [v["name"] for v in flow["variables"]]
    assert len(names) == len(set(names))
    assert len(m.prompts) == 3                       # one whole attempt (plan > 6 lines), then two parts
    assert "- ooa.file/read_text:" in m.prompts[2] and "Variables: " in m.prompts[2]     # part 2 sees part 1's variables
    assert [a.get("part") for a in res["attempts"]] == [None, 1, 2, None]
    # a part that never passes: no flow, a plain reason
    m2 = Scripted(lambda req, p, a: answer(req, p, a) if req == big else "{\"steps\":[]}")
    res = router(m2, local_retries=0).run(big)
    assert not res["verified"] and any("Part 1 of 2" in p for p in res["problems"]), res["problems"]


def test_safety_warnings_reach_the_client():
    flow = {"kind": "TaskBot", "id": "d", "name": "d",
            "variables": [{"name": "path", "type": "String", "direction": "input", "required": True}],
            "steps": [{"id": "s1", "action": "ooa.file/delete", "description": "Delete the file", "args": {"path": "$path"}}]}
    r = router(Scripted(lambda req, p, a: plan_answer(flow)))
    code, out = S.chat_completion(r, studio_body("delete the file"))
    ans = json.loads(out["choices"][0]["message"]["content"])
    assert ans["warnings"] and ans["warnings"][0].startswith("Check before running: step s1 deletes")
    assert ans["warnings"][0] in ans["notes"] and out["ooa"]["warnings"] == ans["warnings"]
    code, out = S.chat_completion(r, {"messages": [{"role": "user", "content": "delete the file"}]})
    assert json.loads(out["choices"][0]["message"]["content"])["steps"] and out["ooa"]["warnings"]


class _Stub(BaseHTTPRequestHandler):
    seen = []
    answer = ""

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Stub.seen.append((self.path, self.headers.get("authorization"), body))
        b = json.dumps({"id": "x", "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": _Stub.answer}, "finish_reason": "stop"}]}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)


def _stub():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_escalation_to_openai_compatible_endpoint():
    srv = _stub()
    try:
        _Stub.seen, _Stub.answer = [], "```json\n" + F.compact(CSV_TO_EXCEL) + "\n```"
        os.environ["OOA_TEST_ESC_KEY"] = "k-123"
        pol = S.Policy(allow_escalation=True, local_first=True, local_retries=1, prompt_format="context",
                       escalate_url=f"http://127.0.0.1:{srv.server_address[1]}/v1", escalate_model="big-model",
                       escalate_key_env="OOA_TEST_ESC_KEY")
        cloud = S.make_cloud(pol)
        assert isinstance(cloud, S.OpenAICompatBackend)
        local = Scripted(lambda req, p, a: "{not json")
        r = S.Router(CATALOG, PLATFORM, pol, local, cloud)
        res = r.run("Read a CSV file and email it to bob@example.com")
        assert res["verified"] and res["source"] == "cloud" and len(local.prompts) == 2       # ours tried first
        path, auth, body = _Stub.seen[0]
        assert path == "/v1/chat/completions" and auth == "Bearer k-123" and body["model"] == "big-model"
        assert "bob@example.com" not in json.dumps(body) and "[EMAIL]" in body["messages"][1]["content"]
        # a long plan goes to the bigger model after one failed local attempt
        long_plan = [f"{i}. Step {i} (ooa.csv/read)" for i in range(1, 15)]
        local2 = Scripted(lambda req, p, a: "Plan:\n" + "\n".join(long_plan) + "\nFlow:\n{}")
        res = S.Router(CATALOG, PLATFORM, pol, local2, cloud).run("a long process")
        assert res["source"] == "cloud" and len(local2.prompts) == 1 and res["plan"] == long_plan
        # off by default: no backend
        assert S.make_cloud(S.Policy()) is None
        assert isinstance(S.make_cloud(S.Policy(allow_escalation=True)), S.CloudBackend)   # the Anthropic path stays
    finally:
        srv.shutdown()


def test_feedback_endpoint_shapes():
    import tempfile
    d = tempfile.mkdtemp()
    r = router(FakeModel(), store_logs=True, log_dir=d)
    srv = S.serve(r, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/feedback", json.dumps(body).encode(), {"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert post({"request": "mail jo@x.com the list", "flow": CSV_TO_EXCEL, "verdict": "inserted"})[1]["stored"]
        assert post({"request": "r", "flow": CSV_TO_EXCEL, "verdict": "edited", "edited_flow": PART2})[0] == 200
        assert post({"request": "r", "verdict": "discarded"})[0] == 200
        assert post({"prompt": "old", "flow": CSV_TO_EXCEL, "accepted": True})[0] == 200
        assert post({"request": "r", "verdict": "maybe"})[0] == 400
        assert post({"request": "r", "verdict": "edited"})[0] == 400
        rows = [json.loads(l) for l in open(Path(d) / "feedback.jsonl")]
        assert [x["verdict"] for x in rows] == ["accept", "edit", "reject", "accept"]
        assert "jo@x.com" not in rows[0]["prompt"] and rows[1]["edited_flow"] == PART2
    finally:
        srv.shutdown()


def test_flows_come_back_written_dollar_name_dollar():
    """The platform writes variables $name$: what the model wrote as $name / ${name} reaches Studio as $name$, the
    self-correction prompt shows the previous flow the same way, and a $name$ answer passes the checker as is."""
    flow = json.loads(json.dumps(PART2))
    flow["steps"][2]["args"]["message"] = "Read ${text} from $path"
    bad = json.loads(json.dumps(flow))
    del bad["steps"][1]

    def answer(req, prompt, attempt):
        return plan_answer(bad, F.plan_from_flow(flow, CAT)) if attempt == 0 else plan_answer(flow)
    m = Scripted(answer)
    code, out = S.chat_completion(router(m, local_retries=1), studio_body(LIST_REQ))
    assert code == 200 and out["ooa"]["verified"], out
    steps = json.loads(out["choices"][0]["message"]["content"])["steps"]
    assert steps[0]["args"] == {"folder": "$folder$"} and steps[1]["args"] == {"path": "$path$"}
    assert steps[2]["args"]["message"] == "Read $text$ from $path$"
    assert '"folder":"$folder$"' in m.prompts[1].split(F.PREVIOUS_HEADER)[1]
    res = router(Scripted(lambda req, p, a: plan_answer(F.to_dollar_name_dollar(flow))), local_retries=0).run(LIST_REQ)
    assert res["verified"] and json.loads(res["flow"])["steps"][0]["args"] == {"folder": "$folder$"}, res


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, type(e).__name__, e)
    print("ALL LAYER TESTS PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
