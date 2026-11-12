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




def test_missing_open_step_added():
    f, notes = repair.repair(flow([], [{"id": "s1", "action": "ooa.excel/row_count", "description": "n",
                                        "out": {"count": "count"}}]), CAT)
    assert [s["action"] for s in f["steps"]] == ["ooa.excel/open", "ooa.excel/row_count"], f["steps"]
    assert f["steps"][0]["args"] == {"path": "$workbook_path"} and any(v["name"] == "workbook_path" for v in f["variables"])
    assert len({s["id"] for s in f["steps"]}) == 2 and any("missing" in n for n in notes), notes


def test_open_step_not_added_when_already_open():
    steps = [{"id": "s1", "action": "ooa.excel/open", "description": "o", "args": {"path": "$p"}},
             {"id": "s2", "action": "ooa.excel/row_count", "description": "n"}]
    f, notes = repair.repair(flow([{"name": "p", "type": "String", "direction": "input"}], steps), CAT)
    assert [s["action"] for s in f["steps"]] == ["ooa.excel/open", "ooa.excel/row_count"] and not notes, notes


def test_missing_save_added_before_close():
    steps = [{"id": "s1", "action": "ooa.excel/open", "description": "o", "args": {"path": "$p"}},
             {"id": "s2", "action": "ooa.excel/write_table", "description": "w", "args": {"table": "$t"}},
             {"id": "s3", "action": "ooa.excel/close", "description": "c"}]
    f, notes = repair.repair(flow([{"name": "p", "type": "String", "direction": "input"},
                                   {"name": "t", "type": "Table", "direction": "input"}], steps), CAT)
    assert [s["action"].split("/")[1] for s in f["steps"]] == ["open", "write_table", "save", "close"], f["steps"]


def test_no_save_when_only_reading_or_already_saved():
    read = [{"id": "s1", "action": "ooa.excel/open", "description": "o", "args": {"path": "$p"}},
            {"id": "s2", "action": "ooa.excel/get_all_cells", "description": "r"}]
    f, notes = repair.repair(flow([{"name": "p", "type": "String", "direction": "input"}], read), CAT)
    assert len(f["steps"]) == 2 and not notes, notes
    saved = [{"id": "s1", "action": "ooa.word/open", "description": "o", "args": {"path": "$p"}},
             {"id": "s2", "action": "ooa.word/add_paragraph", "description": "a", "args": {"text": "$p"}},
             {"id": "s3", "action": "ooa.word/close", "description": "c", "args": {"save": True}}]
    f, notes = repair.repair(flow([{"name": "p", "type": "String", "direction": "input"}], saved), CAT)
    assert len(f["steps"]) == 3 and not notes, notes


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ALL PASS")
