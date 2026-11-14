"""Tests for safety.py (warnings for destructive steps) and ingest_feedback.py (feedback -> training records).
Standard library only. Run: python test_safety_feedback.py   (works under pytest too)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import safety, ingest_feedback as IF
from test_ooa_format import CAT, ENTRIES, LOOP_FLOW, linear_flow


# ---------------------------------------------------------------------------------------------- safety
def test_safety_warnings():
    w = safety.warnings(LOOP_FLOW, CAT)
    assert len(w) == 1 and w[0].startswith("Check before running: step s5 deletes") and "loop" in w[0], w
    flow = {"steps": [{"id": "a", "action": "ooa.database/execute", "args": {"sql": "DELETE FROM orders WHERE x=1"}},
                      {"id": "b", "action": "ooa.database/execute", "args": {"sql": "INSERT INTO t VALUES (1)"}},
                      {"id": "c", "action": "ooa.file/copy", "args": {"source": "a", "target": "b", "overwrite": True}},
                      {"id": "d", "action": "ooa.datatable/delete_row", "args": {}},
                      {"id": "e", "action": "ooa.system/empty_recycle_bin", "args": {}},
                      {"id": "f", "action": "ooa.okta/revoke_authentication" if "ooa.okta/revoke_authentication" in CAT else "ooa.salesforce/revoke_authentication"},
                      {"id": "g", "action": "ooa.csv/read", "args": {"path": "x"}}]}
    w = safety.warnings(flow, ENTRIES)
    steps = [x.split("step ")[1].split(" ")[0] for x in w]
    assert steps == ["a", "c", "e", "f"], w                # in-memory table edits and plain inserts are not flagged
    assert "DELETE FROM" in w[0] and "overwrites" in w[1] and "empties the trash" in w[2] and "revokes" in w[3]
    assert safety.warnings({"steps": [{"id": "s1", "action": "ooa.excel/select_dropdown"}]}, CAT) == []


# ---------------------------------------------------------------------------------------------- ingest_feedback
def test_ingest_feedback_records():
    good = linear_flow(["ooa.csv/read"], "g")
    edited = linear_flow(["ooa.csv/read", "ooa.email/send"], "e")
    rows = [{"prompt": "read it", "flow": good, "verdict": "accept"},
            {"prompt": "read and mail", "flow": good, "verdict": "edit", "edited_flow": edited},
            {"prompt": "nope", "flow": good, "verdict": "reject"},
            {"prompt": "studio", "flow": {"variables": [], "steps": good["steps"]}, "verdict": "inserted"},
            {"prompt": "old row", "flow": good, "accepted": True},
            {"prompt": "bad", "flow": {"steps": "x"}, "verdict": "accept"}]
    chat, recs, rejected, dropped = IF.ingest(rows, CAT, lambda f: True)
    assert [r["kind"] for r in recs] == ["flow", "flow", "fix", "flow", "flow"]
    assert recs[1]["flow"] == edited and recs[2]["previous"] == good and recs[2]["flow"] == edited
    assert any("ooa.email/send" in p for p in recs[2]["problems"])
    assert recs[3]["flow"]["kind"] == "TaskBot"                      # Studio's shape wrapped
    assert rejected == [{"prompt": "nope"}] and dropped == 1 and len(chat) == 4
    assert IF.ingest(rows, CAT, lambda f: False)[1] == []            # the verifier is the gate


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
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
