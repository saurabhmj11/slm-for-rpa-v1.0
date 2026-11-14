"""Contract test: the exact request shape One Agent Studio's Assistant sends (StudioServer.assistantBuild through
LlmClient), replayed against serving/ooa_server.chat_completion with the FakeModel backend.

Studio sends system = long instructions + one catalog line per installed action, user = 'Existing variables: <JSON>\\n
Request: <prompt>', temperature 0, max_tokens 4000; it reads choices[0].message.content and parses the text from the
first '{' to the last '}' as {variables, steps, notes}. The connection wizard's Test button and Command Center's model
probe send a one-word question and expect "OK".
Run: python -m pytest tests/test_studio_contract.py   (needs the platform repo: OOA_PLATFORM, see test_app.py)
"""
import json
import os
import sys
from pathlib import Path

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

pytestmark = pytest.mark.skipif(not SCHEMA.exists(), reason="platform repo not found (set OOA_PLATFORM)")


def studio_system():
    """Like Studio's system text: instructions, then one line per installed action."""
    lines = []
    for line in open(CATALOG, encoding="utf-8"):
        e = json.loads(line)
        ins = ", ".join(i["name"] + ("*" if i["required"] else "") for i in e["inputs"])
        outs = ", ".join(o["name"] for o in e["outputs"])
        lines.append(f"{e['id']} | {e['name']} | inputs: {ins} | outputs: {outs}")
    return ("You design steps for ONEOFFICEAUTOMATION agents. Answer with one JSON object only: "
            '{"variables": [...], "steps": [...], "notes": "..."}. Use only these actions:\n' + "\n".join(lines))


def studio_body(prompt, existing='[{"name":"x","type":"String"}]'):
    return {"model": "ooa-assistant", "temperature": 0, "max_tokens": 4000,
            "messages": [{"role": "system", "content": studio_system()},
                         {"role": "user", "content": "Existing variables: " + existing + "\nRequest: " + prompt}]}


@pytest.fixture(scope="module")
def S():
    import ooa_server
    return ooa_server


@pytest.fixture
def router(S):
    return S.Router(CATALOG, PLATFORM, S.Policy(allow_escalation=False, local_retries=1), local=FakeModel())


def studio_parse(content):
    """What Studio does with the reply text."""
    i, j = content.index("{"), content.rindex("}")
    obj = json.loads(content[i:j + 1])
    assert isinstance(obj, dict)
    return obj


def test_studio_request_gets_studio_shape(S, router):
    assert len(studio_system()) > 50_000                       # really the long catalog text
    code, out = S.chat_completion(router, studio_body("read a csv file"))
    assert code == 200, out
    assert out["object"] == "chat.completion" and out["model"] == "ooa-assistant"
    choice = out["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["message"]["role"] == "assistant"
    ans = studio_parse(choice["message"]["content"])
    assert isinstance(ans["variables"], list) and isinstance(ans["steps"], list) and isinstance(ans["notes"], str)
    assert ans["steps"] == [] and ans["notes"].startswith("No flow was returned")   # FakeModel cannot do this one
    assert router.local.calls[0][0] == "read a csv file"         # the model sees only the request
    u = out["usage"]
    assert u["prompt_tokens"] > 10_000 and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_studio_request_with_a_flow(S, router):
    code, out = S.chat_completion(router, studio_body(
        "Read a CSV file into a table, then write the table to an Excel sheet"))
    assert code == 200 and out["choices"][0]["finish_reason"] == "stop"
    ans = studio_parse(out["choices"][0]["message"]["content"])
    assert set(ans) == {"variables", "steps", "notes"}
    assert [s["action"] for s in ans["steps"]][:2] == ["ooa.csv/read", "ooa.excel/open"]
    assert all(v["name"] != "x" for v in ans["variables"])     # Studio's existing variables are not repeated
    assert out["ooa"]["verified"] is True


@pytest.mark.parametrize("text", ["Reply with the single word OK.", "Answer with the single word OK.",
                                  "  reply WITH the   single word ok  ", "answer with the single word OK"])
def test_probe_answers_ok_without_the_model(S, router, text):
    body = {"model": "ooa-assistant", "max_tokens": 5, "messages": [{"role": "user", "content": text}]}
    code, out = S.chat_completion(router, body)
    assert code == 200 and out["choices"][0]["message"]["content"] == "OK"
    assert out["choices"][0]["finish_reason"] == "stop" and router.local.calls == []


def test_tools_refused(S, router):
    body = studio_body("read a csv file")
    body["tools"] = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    code, out = S.chat_completion(router, body)
    assert code == 400
    assert out == {"error": {"type": "invalid_request_error",
                             "message": "ooa-assistant writes flows only; it does not support tools"}}
    assert router.local.calls == []
    body["tools"] = []                                           # an empty list is not a tools request
    assert S.chat_completion(router, body)[0] == 200


def test_bad_requests_are_openai_errors(S, router):
    for body in ({}, {"messages": []}, {"messages": [{"role": "system", "content": "x"}]}, {"messages": "hi"}):
        code, out = S.chat_completion(router, body)
        assert code == 400 and out["error"]["type"] == "invalid_request_error", body
