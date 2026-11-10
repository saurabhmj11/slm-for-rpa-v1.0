"""Tests for runner_bridge.py. Run: python tests/test_runner_bridge.py (no pytest needed).
Set OOA_PLATFORM_REPO to the platform repo if it is not at the default location (or install the reference package)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from runner_bridge import RunnerBridge, is_risky, path_problem  # noqa: E402

PLATFORM = os.environ.get("OOA_PLATFORM_REPO",
                          str(Path.home() / "oneofficeautomation-private-limited" / "oneofficeautomation-platform"))
SANDBOX = Path(tempfile.mkdtemp(prefix="ooa_sandbox_test_"))
BRIDGE = RunnerBridge(PLATFORM if Path(PLATFORM).is_dir() else None, str(SANDBOX))


def bot(steps, variables=None):
    return {"kind": "TaskBot", "id": "t", "name": "t", "variables": variables or [], "steps": steps}


def test_available_and_supported():
    assert BRIDGE.available, BRIDGE.unavailable_reason
    sup = BRIDGE.supported_actions()
    assert {"ooa.file/write_text", "ooa.file/read_text", "core/loop", "ooa.excel/open"} <= sup
    assert "ooa.browser/open" not in sup


def test_write_then_read_runs():
    flow = bot([
        {"id": "w", "action": "ooa.file/write_text", "args": {"path": "notes/hello.txt", "text": "$msg"}},
        {"id": "r", "action": "ooa.file/read_text", "args": {"path": "notes/hello.txt"}, "out": {"text": "result"}},
    ], [{"name": "msg", "type": "String", "direction": "input", "required": True, "description": "Text to write"},
        {"name": "result", "type": "String", "direction": "output"}])
    chk = BRIDGE.can_run(flow)
    assert chk["ok"], chk
    assert chk["inputs"] == [{"name": "msg", "type": "String", "required": True, "default": None,
                              "description": "Text to write"}]
    res = BRIDGE.run(flow, {"msg": "hi there"})
    assert res["ok"], res
    assert res["outputs"]["result"] == "hi there"
    assert "notes/hello.txt" in res["files"]
    assert (SANDBOX / "notes" / "hello.txt").read_text() == "hi there"
    assert any(e["type"] == "run.finished" for e in res["events"])
    assert all(set(e) == {"type", "step", "message"} for e in res["events"])


def test_unsupported_refused():
    flow = bot([{"id": "b", "action": "ooa.browser/open", "args": {"url": "https://example.com"}}])
    chk = BRIDGE.can_run(flow)
    assert not chk["ok"] and chk["unsupported"] == ["ooa.browser/open"], chk
    res = BRIDGE.run(flow, {}, allow_risky=True)
    assert not res["ok"] and "ooa.browser/open" in res["refused"]


def test_delete_needs_allow_risky():
    (SANDBOX / "gone.txt").write_text("x")
    flow = bot([{"id": "d", "action": "ooa.file/delete", "args": {"path": "gone.txt"}}])
    chk = BRIDGE.can_run(flow)
    assert chk["ok"] and chk["risky"] == ["ooa.file/delete"], chk
    res = BRIDGE.run(flow, {})
    assert not res["ok"] and "risky" in res["refused"]
    assert (SANDBOX / "gone.txt").exists()
    res = BRIDGE.run(flow, {}, allow_risky=True)
    assert res["ok"], res
    assert not (SANDBOX / "gone.txt").exists()


def test_risky_rules():
    assert is_risky("ooa.rest/request") and is_risky("ooa.genai/prompt") and is_risky("core/run_bot")
    assert is_risky("ooa.agents/run_agent") and is_risky("ooa.browser/open")
    assert not is_risky("ooa.file/write_text") and not is_risky("core/loop")
    assert is_risky("ooa.file/copy", {"overwrite": True}) and not is_risky("ooa.file/copy", {"overwrite": False})


def test_path_guard():
    for bad in ("/etc/passwd", "../../x"):
        flow = bot([{"id": "r", "action": "ooa.file/read_text", "args": {"path": bad}}])
        chk = BRIDGE.can_run(flow)
        assert not chk["ok"] and any("sandbox" in p for p in chk["problems"]), chk
        res = BRIDGE.run(flow, {})
        assert not res["ok"] and "sandbox" in res["refused"], res
    # through a run input
    flow = bot([{"id": "r", "action": "ooa.file/read_text", "args": {"path": "$p"}}],
               [{"name": "p", "type": "String", "direction": "input"}])
    assert BRIDGE.can_run(flow)["ok"]
    res = BRIDGE.run(flow, {"p": "/etc/passwd"})
    assert not res["ok"] and "sandbox" in res["refused"], res
    # inside the sandbox is fine, '..' that stays inside too, variables and URLs are not paths
    assert path_problem("a/b/../c.txt", SANDBOX) is None
    assert path_problem(str(SANDBOX / "x.txt"), SANDBOX) is None
    assert path_problem("$p", SANDBOX, "path") is None
    assert path_problem("https://example.com/a", SANDBOX) is None
    assert path_problem("${folder}/../../x", SANDBOX) is not None


def test_runtime_guard_blocks_paths_built_at_run_time():
    """Fix: paths from ${...}, {"expr": ...}, string ops or parsed JSON passed the static check and read/wrote
    files outside the sandbox. The child now refuses them in resolve_path."""
    outside = Path(tempfile.mkdtemp(prefix="ooa_outside_"))
    secret = outside / "secret.txt"
    secret.write_text("TOPSECRET")
    out_var = [{"name": "c", "type": "String", "direction": "output"},
               {"name": "p", "type": "String", "direction": "local"}, {"name": "d", "type": "Any", "direction": "local"}]
    read_p = {"id": "s1", "action": "ooa.file/read_text", "args": {"path": "$p"}, "out": {"text": "c"}}
    cases = {
        "interp": [{"id": "s1", "action": "ooa.file/read_text", "args": {"path": "${'%s'}" % secret.as_posix()},
                    "out": {"text": "c"}}],
        "replace": [{"id": "s0", "action": "ooa.string/replace",
                     "args": {"text": secret.as_posix().replace("/", "X"), "find": "X", "replace_with": "/"},
                     "out": {"result": "p"}}, read_p],
        "json": [{"id": "s0", "action": "ooa.json/parse", "args": {"text": '{"p": "%s"}' % secret.as_posix()},
                  "out": {"value": "d"}},
                 {"id": "s1", "action": "ooa.file/read_text", "args": {"path": "$d.p"}, "out": {"text": "c"}}],
        "write": [{"id": "s1", "action": "ooa.file/write_text",
                   "args": {"path": "${'%s'}" % (outside / "pwned.txt").as_posix(), "text": "x"}}],
    }
    for name, steps in cases.items():
        res = BRIDGE.run(bot(steps, out_var), {})
        assert not res["ok"], (name, res)
        assert "TOPSECRET" not in str(res["outputs"]), (name, res["outputs"])
        assert "refused" in res or "OUTSIDE_SANDBOX" in (res["error"] or ""), (name, res)
    assert not (outside / "pwned.txt").exists()
    # {"expr": "'/abs/...'"} is now also caught by the static check
    flow = bot([{"id": "s1", "action": "ooa.file/read_text", "args": {"path": {"expr": "'%s'" % secret.as_posix()}},
                 "out": {"text": "c"}}], out_var)
    assert not BRIDGE.can_run(flow)["ok"]
    # folder listings never return files outside the sandbox, even with '..' in the glob pattern
    (SANDBOX / "lst").mkdir(exist_ok=True)
    (SANDBOX / "lst" / "in.txt").write_text("x")
    flow = bot([{"id": "l", "action": "ooa.file/list", "args": {"folder": "lst", "pattern": "${'../../*/*'}"},
                 "out": {"files": "f"}}], [{"name": "f", "type": "List", "direction": "output"}])
    res = BRIDGE.run(flow, {})
    assert res["ok"], res
    assert all(Path(x).resolve().is_relative_to(SANDBOX.resolve()) for x in res["outputs"]["f"]), res["outputs"]
    # normal paths inside the sandbox still work at run time
    flow = bot([{"id": "l", "action": "ooa.file/list", "args": {"folder": "lst"}, "out": {"files": "f"}}],
               [{"name": "f", "type": "List", "direction": "output"}])
    res = BRIDGE.run(flow, {})
    assert res["ok"] and len(res["outputs"]["f"]) == 1, res


def test_malformed_flow_does_not_crash():
    for f in ({"variables": 5, "steps": []}, {"steps": [{"id": "a", "action": "core/if", "elseIf": 5}]},
              {"steps": [{"id": "a", "action": "core/try", "catch": 5}]}):
        chk = BRIDGE.can_run(f)
        assert chk["ok"] is False and chk["problems"], chk


def test_child_gets_user_site():
    import runner_bridge, site
    us = runner_bridge._user_site()
    if site.ENABLE_USER_SITE and Path(site.getusersitepackages()).is_dir():
        assert us == site.getusersitepackages()
    else:
        assert us is None


def test_timeout_kills_infinite_loop():
    flow = bot([{"id": "l", "action": "core/loop",
                 "args": {"iterator": "while", "condition": "true", "max_iterations": 1000000000},
                 "children": [{"id": "z", "action": "core/delay", "args": {"seconds": 0.01}}]}])
    assert BRIDGE.can_run(flow)["ok"], BRIDGE.can_run(flow)
    t0 = time.monotonic()
    res = BRIDGE.run(flow, {}, timeout_s=3)
    took = time.monotonic() - t0
    assert not res["ok"] and "longer than 3 s" in res["error"], res
    assert took < 15, took
    assert 0 < len(res["events"]) <= 200


def test_validation_problems_reported():
    flow = bot([{"id": "o", "action": "ooa.excel/open", "args": {"path": "book.xlsx", "header": True}}])
    chk = BRIDGE.can_run(flow)
    assert not chk["ok"] and not chk["unsupported"], chk
    assert any("Unknown input 'header'" in p and "step o" in p for p in chk["problems"]), chk
    res = BRIDGE.run(flow, {})
    assert not res["ok"] and "cannot run locally" in res["refused"] and "header" in res["refused"]


def test_events_capped():
    flow = bot([{"id": "l", "action": "core/loop", "args": {"iterator": "times", "count": 300},
                 "children": [{"id": "g", "action": "core/log", "args": {"message": "tick"}}]}])
    res = BRIDGE.run(flow, {})
    assert res["ok"], res
    assert len(res["events"]) == 200 and res["events"][-1]["type"] == "run.finished"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # report every test, then exit non-zero
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{'FAILED' if failed else 'OK'}: {failed} failure(s)")
    sys.exit(1 if failed else 0)
