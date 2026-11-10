"""Tests for repair.py. Run: python test_repair.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repair

CAT = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "..", "data", "catalog", "catalog.jsonl"), encoding="utf-8")}


def flow(variables, steps):
    return {"kind": "TaskBot", "id": "t", "name": "t", "description": "d", "variables": variables, "steps": steps}


def test_variable_names_with_spaces():
    f, notes = repair.repair(flow(
        [{"name": "url", "type": "String", "direction": "input"}, {"name": " page", "type": "Any", "direction": "local"},
         {"name": "page", "type": "Any", "direction": "local"}],
        [{"id": "s1", "action": "ooa.browser/open", "description": "o", "args": {"url": "$url"}, "out": {"session": " page"}},
         {"id": "s2", "action": "ooa.browser/close", "description": "c", "args": {"session": "${ page}"}}]), CAT)
    assert [v["name"] for v in f["variables"]] == ["url", "page"], f["variables"]       # duplicate merged
    assert f["steps"][0]["out"]["session"] == "page" and f["steps"][1]["args"]["session"] == "${page}", f["steps"]


def test_unknown_fixed_setting_dropped_data_key_kept():
    f, _ = repair.repair(flow([{"name": "url", "type": "String", "direction": "input"}],
        [{"id": "s1", "action": "ooa.browser/open", "description": "o", "args": {"url": "$url", "header": True, "zzz_data": "$url"}}]), CAT)
    args = f["steps"][0]["args"]
    assert "header" not in args and "zzz_data" in args, args


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ALL PASS")
