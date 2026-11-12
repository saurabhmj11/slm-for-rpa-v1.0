"""End-to-end tests of the server with fake local/cloud models. Run: python test_server.py PLATFORM_REPO  (needs jsonschema)"""
import json, os, sys, tempfile, threading, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "training"))
import ooa_server as S, gen_flows as G

PLATFORM = sys.argv[1]
CATALOG = str(HERE.parent / "data/catalog/catalog.jsonl")
cat = [json.loads(l) for l in open(CATALOG, encoding="utf-8")]
good = [m for _, f, m in G.generate(cat, PLATFORM, 40, 11) if not f["steps"][0]["action"].startswith("core/")]
GOOD, GOOD2 = good[0], good[1]

class Local:
    def __init__(self, outs): self.outs, self.calls = list(outs), 0
    def generate(self, request, attempt=0):
        self.calls += 1
        return self.outs[min(self.calls - 1, len(self.outs) - 1)]
class Cloud:
    def __init__(self, outs): self.outs, self.seen = list(outs), []
    def generate(self, system, user):
        self.seen.append((system, user))
        o = self.outs[min(len(self.seen) - 1, len(self.outs) - 1)]
        if isinstance(o, Exception): raise o
        return o

def post(port, path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(r).read())

def start(policy, local, cloud):
    router = S.Router(CATALOG, PLATFORM, policy, local, cloud)
    srv = S.serve(router, 0); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return router, srv, port

def ask(port, text):
    return post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": text}]})

tmp = tempfile.mkdtemp()
# 1. local success, OpenAI shape, local-only default policy
loc = Local([GOOD]); r, srv, port = start(S.Policy(log_dir=tmp + "/a"), loc, None)
res = ask(port, "read a csv file into a table")
assert res["object"] == "chat.completion" and res["choices"][0]["message"]["role"] == "assistant"
assert res["ooa"]["source"] == "local" and res["ooa"]["verified"] and json.loads(res["choices"][0]["message"]["content"])["kind"]
assert not os.path.exists(tmp + "/a"), "default policy must store nothing"
print("1 ok: local answer, OpenAI shape, nothing stored by default")

# 2. local keeps failing, escalation disabled -> honest error, no unchecked flow, cloud never used
cl = Cloud([GOOD]); r, srv, port = start(S.Policy(allow_escalation=False), Local(["not json"]), cl)
res = ask(port, "do something")
assert not res["ooa"]["verified"] and "disabled by this client's policy" in res["ooa"]["reason"] and cl.seen == []
assert "error" in json.loads(res["choices"][0]["message"]["content"])
print("2 ok: escalation off -> clear error, cloud never called")

# 3. escalation on: cloud answers (in code fences), and the cloud saw a MASKED request
req = "Email the report to john.doe@acme.com and call +91 98765 43210 about card 4111 1111 1111 1111"
cl = Cloud(["```json\n" + GOOD2 + "\n```"]); r, srv, port = start(S.Policy(allow_escalation=True, local_retries=1), Local(["nope"]), cl)
res = ask(port, req)
assert res["ooa"]["source"] == "cloud" and res["ooa"]["verified"]
system, user = cl.seen[0]
assert "john.doe@acme.com" not in user + system and "4111" not in user and "98765" not in user and "[EMAIL]" in user
assert "Actions" in system or "action id" in system
print("3 ok: escalated to cloud, PII masked before it left, answer verified")

# 4. cloud's first answer invalid -> retried with the exact problems -> verified
cl = Cloud(['{"kind":"TaskBot","id":"x","name":"x","variables":[],"steps":[{"id":"s1","action":"ooa.fake/none"}]}', GOOD])
r, srv, port = start(S.Policy(allow_escalation=True), None, cl)
res = ask(port, "do a thing")
assert res["ooa"]["verified"] and len(cl.seen) == 2 and "Unknown action" in cl.seen[1][1]
print("4 ok: cloud retried with the verifier's problems")

# 5. complex request skips the local model entirely
loc = Local([GOOD]); cl = Cloud([GOOD])
r, srv, port = start(S.Policy(allow_escalation=True), loc, cl)
long = "Sign in to the portal, open the queue, read each invoice, check the totals, compare with the ERP, email the owner, update the sheet, then archive the file, and write a summary report"
res = ask(port, long)
assert loc.calls == 0 and res["ooa"]["source"] == "cloud"
print("5 ok: complex request went straight to the cloud model")

# 5b. same complex request with no cloud fallback (local-only setup): the local model must still try
loc = Local([GOOD])
r, srv, port = start(S.Policy(allow_escalation=False), loc, None)
res = ask(port, long)
assert loc.calls == 1 and res["ooa"]["source"] == "local" and res["ooa"]["verified"]
print("5b ok: no cloud -> complex request still answered by the local model")

# 6. repair: a near-miss typo is fixed without escalating
bad = json.loads(GOOD); a = bad["steps"][0]["action"]; bad["steps"][0]["action"] = a[:-1]
cl = Cloud([GOOD]); r, srv, port = start(S.Policy(allow_escalation=True), Local([json.dumps(bad)]), cl)
res = ask(port, "do a thing")
assert res["ooa"]["source"] == "local" and res["ooa"]["verified"] and res["ooa"]["repairs"] and cl.seen == []
print("6 ok: typo repaired locally:", res["ooa"]["repairs"])

# 7. cloud error and cloud invalid twice -> never an unchecked flow
for outs in ([RuntimeError("network down")], ["{}", "{}"]):
    r, srv, port = start(S.Policy(allow_escalation=True), Local(["x"]), Cloud(outs))
    res = ask(port, "do a thing")
    assert not res["ooa"]["verified"] and "error" in json.loads(res["choices"][0]["message"]["content"])
print("7 ok: cloud failure or invalid output -> honest error")

# 8. logging and feedback respect the policy; stored data is masked and flagged for training use
pol = S.Policy(store_logs=True, allow_training_on_data=False, log_dir=tmp + "/b")
r, srv, port = start(pol, Local([GOOD]), None)
ask(port, "send mail to jane@corp.com")
fb = post(port, "/v1/feedback", {"prompt": "send mail to jane@corp.com", "flow": json.loads(GOOD), "accepted": True})
assert fb["stored"] and fb["usable_for_training"] is False
logged = open(tmp + "/b/requests.jsonl").read() + open(tmp + "/b/feedback.jsonl").read()
assert "jane@corp.com" not in logged and "[EMAIL]" in logged and '"training_ok": false' in logged
r, srv, port = start(S.Policy(store_logs=False), Local([GOOD]), None)
assert post(port, "/v1/feedback", {"prompt": "x", "accepted": True})["stored"] is False
print("8 ok: logs and feedback follow the client policy, masked and flagged")
# 9. the platform's Studio assistant: system = its catalog prompt, user = "Existing variables: [...]\nRequest: ...", answer = {variables, steps, notes}
STUDIO_SYSTEM = "You design steps for ONEOFFICEAUTOMATION agents. Answer with one JSON object only. " + "ooa.csv/read | Read to table | inputs: path* | outputs: table\n" * 60
def studio(port, text, existing="[]", model="ooa-assistant"):
    return post(port, "/v1/chat/completions", {"model": model, "temperature": 0, "max_tokens": 4000,
                "messages": [{"role": "system", "content": STUDIO_SYSTEM}, {"role": "user", "content": "Existing variables: " + existing + "\nRequest: " + text}]})
seen = []
class Spy(Local):
    def generate(self, request, attempt=0):
        seen.append(request); return super().generate(request, attempt)
r, srv, port = start(S.Policy(), Spy([GOOD]), None)
res = studio(port, "read a csv file into a table")
ans = json.loads(res["choices"][0]["message"]["content"])
assert seen == ["read a csv file into a table"], "the local model must see only the request, not Studio's wrapper: " + str(seen)
assert set(ans) == {"variables", "steps", "notes"} and ans["steps"] and "kind" not in ans, ans.keys()
assert res["usage"]["prompt_tokens"] > 0 and res["ooa"]["verified"]
print("9 ok: Studio request -> local model sees only the request; answer has variables/steps/notes; usage reported")

# 10. Studio + escalation: the cloud model gets the platform's own system prompt and answers in Studio's shape; existing variables are respected
CLOUD_STUDIO = json.dumps({"variables": [{"name": "report_path", "type": "String", "direction": "input"}, {"name": "tbl", "type": "Table"}],
    "steps": [{"id": "a1", "action": "ooa.csv/read", "args": {"path": "$report_path"}, "out": {"table": "tbl"}}], "notes": "reads the file"})
cl = Cloud([CLOUD_STUDIO]); r, srv, port = start(S.Policy(allow_escalation=True), Local(["nope"]), cl)
res = studio(port, "read my csv", existing=json.dumps([{"name": "folder", "type": "String", "direction": "input"}]))
ans = json.loads(res["choices"][0]["message"]["content"])
assert res["ooa"]["source"] == "cloud" and ans["steps"][0]["action"] == "ooa.csv/read" and all(v["name"] != "folder" for v in ans["variables"])
assert cl.seen[0][0] == STUDIO_SYSTEM and "Existing variables:" in cl.seen[0][1]
print("10 ok: Studio + cloud: platform's own system prompt used, answer in Studio's shape, existing variables not duplicated")

# 11. Studio + nothing verifies: empty steps with the honest reason in notes (Studio shows notes)
r, srv, port = start(S.Policy(allow_escalation=False), Local(["garbage"]), None)
ans = json.loads(studio(port, "do a thing")["choices"][0]["message"]["content"])
assert ans["steps"] == [] and "disabled by this client's policy" in ans["notes"]
print("11 ok: Studio + failure -> empty steps and a clear reason in notes")
print("ALL SERVER TESTS PASSED")
